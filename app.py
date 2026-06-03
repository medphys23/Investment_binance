from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.binance_client import BinanceClient, BinanceClientError
from src.config import (
    DASHBOARD_TABS,
    DEFAULT_CANDLE_LIMIT,
    KEY_TIMEFRAMES,
    MATRIX_CANDLE_LIMIT,
    PAPER_POLICY_BATCH,
    SYMBOLS,
    TIMEFRAMES,
)
from src.indicators import add_indicators
from src.signals import Alert, analyze_timeframe, btc_risk_from_analysis, summarize_alert_counts
from src.storage import connect, get_state, initialize_database, latest_model_run, latest_policy_state, table


st.set_page_config(
    page_title="Crypto Paper Bot Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_pwa() -> None:
    """Make the dashboard installable to a phone home screen (Add to Home Screen).

    Uses an inline data-URI manifest so no external hosting is required; the
    deployment target (tunnel/cloud) can be decided later without code changes.
    """
    manifest = {
        "name": "Crypto Paper Bot Monitor",
        "short_name": "PaperBot",
        "display": "standalone",
        "background_color": "#0e1117",
        "theme_color": "#0e1117",
        "start_url": ".",
        "icons": [],
    }
    manifest_uri = "data:application/manifest+json," + json.dumps(manifest)
    st.markdown(
        f"""
        <link rel="manifest" href="{manifest_uri}">
        <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
        <meta name="apple-mobile-web-app-title" content="PaperBot">
        """,
        unsafe_allow_html=True,
    )


inject_pwa()


@st.cache_data(ttl=45, show_spinner=False)
def load_spot_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    client = BinanceClient()
    return add_indicators(client.spot_klines(symbol, interval, limit))


@st.cache_data(ttl=45, show_spinner=False)
def load_futures_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
    client = BinanceClient()
    try:
        return add_indicators(client.futures_klines(symbol, interval, limit))
    except BinanceClientError:
        return None


@st.cache_data(ttl=45, show_spinner=False)
def load_market_snapshot(symbol: str) -> dict[str, Any]:
    client = BinanceClient()
    spot_24h = client.spot_24h(symbol)
    futures_24h = client.futures_24h(symbol)
    book = client.book_ticker(symbol)
    open_interest = client.open_interest(symbol)
    mark = client.mark_price(symbol)
    return {
        "spot_24h": spot_24h,
        "futures_24h": futures_24h,
        "book": book,
        "open_interest": open_interest,
        "mark": mark,
    }


@st.cache_data(ttl=90, show_spinner=False)
def load_futures_context(symbol: str, timeframe: str) -> dict[str, Any]:
    snapshot = load_market_snapshot(symbol)
    spot_quote_volume = _to_float(snapshot["spot_24h"].get("quoteVolume"))
    futures_quote_volume = _to_float((snapshot["futures_24h"] or {}).get("quoteVolume"))
    ratio = futures_quote_volume / spot_quote_volume if spot_quote_volume and futures_quote_volume else None
    open_interest = _to_float((snapshot["open_interest"] or {}).get("openInterest"))
    return {
        "spot_quote_volume": spot_quote_volume,
        "futures_quote_volume": futures_quote_volume,
        "futures_spot_ratio": ratio,
        "open_interest": open_interest,
        "open_interest_change": "expanding" if open_interest and ratio and ratio >= 2 else None,
        "funding_rate": _to_float((snapshot["mark"] or {}).get("lastFundingRate")),
        "mark_price": _to_float((snapshot["mark"] or {}).get("markPrice")),
    }


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_money(value: float | None) -> str:
    if value is None:
        return "-"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.2f}K"
    return f"${value:.2f}"


def _base_asset(symbol: str) -> str:
    return symbol.removesuffix("USDC")


def _trend_side(analysis: dict[str, Any]) -> str:
    ema = analysis["ema_state"]
    rsi = analysis["rsi"]
    if ema.startswith("bullish") and rsi >= 50:
        return "bullish"
    if ema.startswith("bearish") and rsi <= 50:
        return "bearish"
    return "mixed"


def _comparison_summary(analyses: dict[str, dict[str, Any]]) -> tuple[str, str, list[str]]:
    sides = {timeframe: _trend_side(analysis) for timeframe, analysis in analyses.items()}
    bullish = sum(1 for side in sides.values() if side == "bullish")
    bearish = sum(1 for side in sides.values() if side == "bearish")
    mixed = len(sides) - bullish - bearish
    short_side = sides.get("1h", "mixed")
    mid_side = sides.get("4h", "mixed")
    regime_side = sides.get("12h", "mixed")
    macro_sides = {sides.get("1d", "mixed"), sides.get("1w", "mixed")}
    notes: list[str] = []

    if bullish >= 4:
        read = "bullish multi-timeframe alignment"
        confidence = "high"
    elif bearish >= 4:
        read = "bearish multi-timeframe alignment"
        confidence = "high"
    elif short_side != "mixed" and short_side != regime_side:
        read = "short-term conflict with 12h regime"
        confidence = "medium"
        notes.append(f"1h is {short_side}, while 12h is {regime_side}. Treat short-term signals as lower quality.")
    elif regime_side not in macro_sides and regime_side != "mixed":
        read = "12h transition against macro"
        confidence = "medium"
        notes.append("12h is moving differently from daily/weekly, which can mark either early reversal or countertrend noise.")
    elif mixed >= 3:
        read = "mixed / compression"
        confidence = "low"
    else:
        read = "partial alignment"
        confidence = "medium"

    if mid_side != "mixed" and regime_side != "mixed" and mid_side != regime_side:
        notes.append(f"4h is {mid_side}, but 12h is {regime_side}; wait for one side to resolve.")
    if sides.get("1d") != "mixed" and sides.get("1w") != "mixed" and sides["1d"] != sides["1w"]:
        notes.append(f"Daily is {sides['1d']}, while weekly is {sides['1w']}; macro trend is not clean.")
    if not notes:
        notes.append("Key timeframes are not showing a major contradiction.")
    return read, confidence, notes


def make_chart(df: pd.DataFrame, analysis: dict[str, Any], show_elliott: bool) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.62, 0.18, 0.2],
        specs=[[{"type": "candlestick"}], [{"type": "bar"}], [{"type": "scatter"}]],
    )
    fig.add_trace(
        go.Candlestick(
            x=df["open_time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )
    colors = {9: "#2ca02c", 20: "#1f77b4", 50: "#9467bd", 100: "#ff7f0e", 200: "#d62728"}
    for period, color in colors.items():
        fig.add_trace(
            go.Scatter(x=df["open_time"], y=df[f"ema_{period}"], name=f"EMA {period}", line={"width": 1.3, "color": color}),
            row=1,
            col=1,
        )
    fig.add_trace(go.Scatter(x=df["open_time"], y=df["bb_upper"], name="BB upper", line={"width": 1, "color": "#666"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["open_time"], y=df["bb_mid"], name="BB mid", line={"width": 1, "color": "#999"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["open_time"], y=df["bb_lower"], name="BB lower", line={"width": 1, "color": "#666"}), row=1, col=1)

    fib = analysis["fib"]
    for name, level in fib.levels.items():
        fig.add_hline(y=level, line_dash="dot", line_color="#8c8c8c", annotation_text=f"Fib {name}", row=1, col=1)
    for name, level in fib.extensions.items():
        fig.add_hline(y=level, line_dash="dash", line_color="#b08d57", annotation_text=f"Ext {name}", row=1, col=1)

    volume_colors = ["#2ca02c" if close >= open_ else "#d62728" for open_, close in zip(df["open"], df["close"])]
    fig.add_trace(go.Bar(x=df["open_time"], y=df["volume"], marker_color=volume_colors, name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["open_time"], y=df["rsi"], name="RSI 14", line={"color": "#1f77b4", "width": 1.5}), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#d62728", row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#777", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#2ca02c", row=3, col=1)

    if show_elliott:
        labels = ["1", "2", "3", "4", "5", "A", "B", "C"]
        for label, pivot in zip(labels[-len(analysis["pivots"]) :], analysis["pivots"]):
            fig.add_annotation(
                x=pivot["time"],
                y=pivot["price"],
                text=label,
                showarrow=True,
                arrowhead=2,
                row=1,
                col=1,
            )

    fig.update_layout(
        height=760,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0},
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    return fig


def build_watchlist(selected_timeframe: str) -> tuple[pd.DataFrame, list[Alert], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    all_alerts: list[Alert] = []
    analyses: dict[str, dict[str, Any]] = {}

    btc_df = load_spot_klines("BTCUSDC", selected_timeframe, MATRIX_CANDLE_LIMIT)
    btc_context = load_futures_context("BTCUSDC", selected_timeframe)
    btc_analysis = analyze_timeframe("BTCUSDC", selected_timeframe, btc_df, btc_context)
    btc_risk = btc_risk_from_analysis(btc_analysis)

    for symbol in SYMBOLS:
        snapshot = load_market_snapshot(symbol)
        df = load_spot_klines(symbol, selected_timeframe, MATRIX_CANDLE_LIMIT)
        context = load_futures_context(symbol, selected_timeframe)
        analysis = analyze_timeframe(symbol, selected_timeframe, df, context, btc_risk if symbol != "BTCUSDC" else None)
        counts = summarize_alert_counts(analysis["alerts"])
        spot = snapshot["spot_24h"]
        book = snapshot["book"] or {}
        bid = _to_float(book.get("bidPrice"))
        ask = _to_float(book.get("askPrice"))
        spread = ((ask - bid) / ask * 100) if bid and ask else None
        rows.append(
            {
                "Symbol": symbol,
                "Price": _to_float(spot.get("lastPrice")),
                "24h %": _to_float(spot.get("priceChangePercent")),
                "Spot volume": _format_money(_to_float(spot.get("quoteVolume"))),
                "Futures volume": _format_money(context.get("futures_quote_volume")),
                "Fut/spot": context.get("futures_spot_ratio"),
                "Spread %": spread,
                "RSI": analysis["rsi"],
                "Trend": analysis["ema_state"],
                "Scenario": analysis["scenario"],
                "Confidence": analysis["confidence"],
                "Alerts": len(analysis["alerts"]),
                "Watch": counts["Watch"],
                "Signal": counts["Signal"],
                "Risk": counts["Risk"],
            }
        )
        analyses[symbol] = analysis
        all_alerts.extend(analysis["alerts"])
    return pd.DataFrame(rows), all_alerts, analyses


def build_matrix() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    btc_cache: dict[str, str] = {}
    for timeframe in KEY_TIMEFRAMES:
        btc_df = load_spot_klines("BTCUSDC", timeframe, MATRIX_CANDLE_LIMIT)
        btc_analysis = analyze_timeframe("BTCUSDC", timeframe, btc_df, load_futures_context("BTCUSDC", timeframe))
        btc_cache[timeframe] = btc_risk_from_analysis(btc_analysis)

    for symbol in SYMBOLS:
        row: dict[str, Any] = {"Coin": _base_asset(symbol)}
        for timeframe in KEY_TIMEFRAMES:
            df = load_spot_klines(symbol, timeframe, MATRIX_CANDLE_LIMIT)
            context = load_futures_context(symbol, timeframe)
            analysis = analyze_timeframe(symbol, timeframe, df, context, btc_cache[timeframe] if symbol != "BTCUSDC" else None)
            marker = (
                f"{analysis['scenario']} | {analysis['ema_state']} | "
                f"RSI {analysis['rsi']:.0f} | {analysis['bollinger_state']} | {analysis['confidence']}"
            )
            row[timeframe] = marker
        rows.append(row)
    return pd.DataFrame(rows)


def build_key_timeframe_analyses(symbol: str) -> dict[str, dict[str, Any]]:
    analyses: dict[str, dict[str, Any]] = {}
    btc_risk_by_timeframe: dict[str, str] = {}

    for timeframe in KEY_TIMEFRAMES:
        btc_df = load_spot_klines("BTCUSDC", timeframe, MATRIX_CANDLE_LIMIT)
        btc_analysis = analyze_timeframe("BTCUSDC", timeframe, btc_df, load_futures_context("BTCUSDC", timeframe))
        btc_risk_by_timeframe[timeframe] = btc_risk_from_analysis(btc_analysis)

    for timeframe in KEY_TIMEFRAMES:
        df = load_spot_klines(symbol, timeframe, MATRIX_CANDLE_LIMIT)
        context = load_futures_context(symbol, timeframe)
        analysis = analyze_timeframe(
            symbol,
            timeframe,
            df,
            context,
            btc_risk_by_timeframe[timeframe] if symbol != "BTCUSDC" else None,
        )
        analyses[timeframe] = analysis
    return analyses


def build_timeframe_comparison(symbol: str) -> tuple[pd.DataFrame, str, str, list[str], list[Alert], dict[str, dict[str, Any]]]:
    analyses = build_key_timeframe_analyses(symbol)
    alerts: list[Alert] = []
    for analysis in analyses.values():
        alerts.extend(analysis["alerts"])

    rows = []
    for timeframe in KEY_TIMEFRAMES:
        analysis = analyses[timeframe]
        rows.append(
            {
                "Timeframe": timeframe,
                "Scenario": analysis["scenario"],
                "Confidence": analysis["confidence"],
                "Trend": analysis["ema_state"],
                "RSI": round(analysis["rsi"], 1),
                "RSI state": analysis["rsi_state"],
                "Bollinger": analysis["bollinger_state"],
                "Rel volume": round(analysis["relative_volume"], 2),
                "Structure": analysis["market_structure"],
                "Nearest Fib": analysis["nearest_fib"],
                "Fib distance %": round(analysis["fib_distance_pct"], 2),
                "Elliott": f"{analysis['elliott_state']} ({analysis['elliott_confidence']})",
                "Side": _trend_side(analysis),
            }
        )
    read, confidence, notes = _comparison_summary(analyses)
    return pd.DataFrame(rows), read, confidence, notes, alerts, analyses


def render_alerts(alerts: list[Alert]) -> None:
    severity_order = {"Risk": 0, "Signal": 1, "Watch": 2}
    for alert in sorted(alerts, key=lambda item: (severity_order.get(item.severity, 9), item.symbol, item.timeframe))[:25]:
        if alert.severity == "Risk":
            st.error(f"{alert.symbol} {alert.timeframe}: {alert.title} - {alert.detail}")
        elif alert.severity == "Signal":
            st.success(f"{alert.symbol} {alert.timeframe}: {alert.title} - {alert.detail}")
        else:
            st.info(f"{alert.symbol} {alert.timeframe}: {alert.title} - {alert.detail}")


def render_bnb_relationship(selected_timeframe: str) -> None:
    bnb_context = load_futures_context("BNBUSDC", selected_timeframe)
    bnb_df = load_spot_klines("BNBUSDC", selected_timeframe, MATRIX_CANDLE_LIMIT)
    bnb_volatility = bnb_df["range_pct"].tail(40).mean()
    rows = []
    for symbol in ["SUIUSDC", "SOLUSDC", "ADAUSDC"]:
        df = load_spot_klines(symbol, selected_timeframe, MATRIX_CANDLE_LIMIT)
        joined = pd.DataFrame(
            {
                "bnb_rel_volume": bnb_df["relative_volume"].tail(len(df)).reset_index(drop=True),
                "alt_range": df["range_pct"].tail(len(bnb_df)).reset_index(drop=True),
            }
        ).dropna()
        corr = joined["bnb_rel_volume"].corr(joined["alt_range"]) if len(joined) > 10 else None
        rows.append(
            {
                "Alt": _base_asset(symbol),
                "BNB rel-volume vs alt volatility": corr,
                "Alt recent volatility": df["range_pct"].tail(40).mean(),
            }
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("BNB futures / spot", f"{bnb_context['futures_spot_ratio']:.2f}x" if bnb_context["futures_spot_ratio"] else "-")
    col2.metric("BNB futures volume", _format_money(bnb_context["futures_quote_volume"]))
    col3.metric("BNB open interest", f"{bnb_context['open_interest']:,.0f}" if bnb_context["open_interest"] else "-")
    col4.metric("BNB avg range", f"{bnb_volatility * 100:.2f}%")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


@st.cache_data(ttl=20, show_spinner=False)
def load_bot_state() -> dict[str, Any]:
    conn = connect()
    initialize_database(conn)
    state = {
        "worker_status": get_state(conn, "worker_status", "not_started"),
        "last_heartbeat": get_state(conn, "last_heartbeat", ""),
        "last_cycle_error": get_state(conn, "last_cycle_error", ""),
        "paper_equity": get_state(conn, "paper_equity", 10000.0),
        "drawdown_pct": get_state(conn, "drawdown_pct", 0.0),
        "simulated_leverage": get_state(conn, "simulated_leverage", 1.0),
        "entry_prob_threshold": get_state(conn, "entry_prob_threshold", 0.5),
        "size_multiplier": get_state(conn, "size_multiplier", 1.0),
        "policy_version": get_state(conn, "policy_version", 0),
        "snapshot": get_state(conn, "dashboard_snapshot", {}) or {},
    }
    conn.close()
    return state


@st.cache_data(ttl=20, show_spinner=False)
def load_tables() -> dict[str, pd.DataFrame | dict | None]:
    conn = connect()
    initialize_database(conn)
    trades = table(conn, "paper_trades")
    policy_history = pd.read_sql_query("SELECT * FROM policy_state ORDER BY id ASC", conn)
    model = latest_model_run(conn)
    policy = latest_policy_state(conn)
    conn.close()
    return {"trades": trades, "policy_history": policy_history, "model": model, "policy": policy}


def _heartbeat_age(last_heartbeat: str) -> str:
    if not last_heartbeat:
        return "never"
    try:
        ts = pd.Timestamp(last_heartbeat)
        delta = pd.Timestamp.now(tz="UTC") - ts
        minutes = int(delta.total_seconds() // 60)
        if minutes < 1:
            return "just now"
        if minutes < 60:
            return f"{minutes} min ago"
        return f"{minutes // 60}h {minutes % 60}m ago"
    except Exception:
        return last_heartbeat


def _closed_with_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    closed = trades[trades["status"].eq("closed")].copy()
    if closed.empty:
        return closed
    closed["closed_at"] = pd.to_datetime(closed["closed_at"], errors="coerce", utc=True)
    closed = closed.sort_values("closed_at")
    closed["realized_pnl"] = pd.to_numeric(closed["realized_pnl"], errors="coerce").fillna(0.0)
    closed["realized_return_pct"] = pd.to_numeric(closed["realized_return_pct"], errors="coerce")
    return closed


def equity_curve_fig(closed: pd.DataFrame, starting_equity: float) -> go.Figure:
    equity = starting_equity + closed["realized_pnl"].cumsum()
    fig = go.Figure(go.Scatter(x=closed["closed_at"], y=equity, mode="lines", line={"color": "#2ca02c", "width": 2}))
    fig.update_layout(height=240, margin={"l": 10, "r": 10, "t": 10, "b": 10}, yaxis_title="Equity (USDC)")
    return fig


def drawdown_fig(closed: pd.DataFrame, starting_equity: float) -> go.Figure:
    equity = starting_equity + closed["realized_pnl"].cumsum()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak * 100
    fig = go.Figure(go.Scatter(x=closed["closed_at"], y=drawdown, fill="tozeroy", line={"color": "#d62728"}))
    fig.update_layout(height=200, margin={"l": 10, "r": 10, "t": 10, "b": 10}, yaxis_title="Drawdown %")
    return fig


def pnl_hist_fig(closed: pd.DataFrame) -> go.Figure:
    returns = closed["realized_return_pct"].dropna()
    fig = go.Figure(go.Histogram(x=returns, nbinsx=30, marker_color="#1f77b4"))
    fig.update_layout(height=220, margin={"l": 10, "r": 10, "t": 10, "b": 10}, xaxis_title="Realized return %", yaxis_title="Trades")
    return fig


def winrate_over_time_fig(closed: pd.DataFrame, window: int = 20) -> go.Figure:
    wins = (closed["realized_return_pct"] > 0).astype(float)
    rolling = wins.rolling(window, min_periods=max(3, window // 4)).mean() * 100
    fig = go.Figure(go.Scatter(x=closed["closed_at"], y=rolling, mode="lines", line={"color": "#9467bd"}))
    fig.add_hline(y=50, line_dash="dot", line_color="#777")
    fig.update_layout(height=220, margin={"l": 10, "r": 10, "t": 10, "b": 10}, yaxis_title=f"Rolling win % ({window})")
    return fig


def long_short_fig(closed: pd.DataFrame) -> go.Figure:
    grouped = closed.groupby("side")["realized_return_pct"].mean()
    fig = go.Figure(
        go.Bar(
            x=list(grouped.index),
            y=list(grouped.values),
            marker_color=["#2ca02c" if v >= 0 else "#d62728" for v in grouped.values],
        )
    )
    fig.update_layout(height=220, margin={"l": 10, "r": 10, "t": 10, "b": 10}, yaxis_title="Avg return % by side")
    return fig


def reward_per_batch_fig(policy_history: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=policy_history["policy_version"],
            y=pd.to_numeric(policy_history["batch_reward"], errors="coerce"),
            mode="lines+markers",
            line={"color": "#ff7f0e"},
        )
    )
    fig.update_layout(height=220, margin={"l": 10, "r": 10, "t": 10, "b": 10}, xaxis_title="Policy version", yaxis_title="Batch reward (PnL)")
    return fig


def calibration_fig(closed: pd.DataFrame) -> go.Figure | None:
    subset = closed.dropna(subset=["p_win"]).copy()
    subset["p_win"] = pd.to_numeric(subset["p_win"], errors="coerce")
    subset = subset.dropna(subset=["p_win"])
    if len(subset) < 10:
        return None
    subset["bucket"] = (subset["p_win"] * 5).round() / 5
    grouped = subset.groupby("bucket").agg(
        predicted=("p_win", "mean"),
        actual=("realized_return_pct", lambda s: float((s > 0).mean())),
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dot", "color": "#777"}, name="Ideal"))
    fig.add_trace(go.Scatter(x=grouped["predicted"], y=grouped["actual"], mode="markers+lines", marker={"color": "#1f77b4"}, name="Observed"))
    fig.update_layout(height=240, margin={"l": 10, "r": 10, "t": 10, "b": 10}, xaxis_title="Predicted win prob", yaxis_title="Actual win rate")
    return fig


def feature_importance_fig(model: dict | None) -> go.Figure | None:
    if not model or not model.get("top_features"):
        return None
    features = pd.DataFrame(json.loads(model["top_features"] or "[]"))
    if features.empty:
        return None
    features = features.sort_values("importance")
    fig = go.Figure(go.Bar(x=features["importance"], y=features["feature"], orientation="h", marker_color="#17becf"))
    fig.update_layout(height=300, margin={"l": 10, "r": 10, "t": 10, "b": 10}, xaxis_title="Importance")
    return fig


def pwin_gauge_fig(p_win: float, threshold: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(p_win * 100, 1),
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2ca02c" if p_win >= threshold else "#d62728"},
                "threshold": {"line": {"color": "#fff", "width": 3}, "value": threshold * 100},
            },
        )
    )
    fig.update_layout(height=180, margin={"l": 10, "r": 10, "t": 10, "b": 10})
    return fig


def timeframe_heatmap_fig(snapshot: dict[str, Any]) -> go.Figure | None:
    symbols = snapshot.get("symbols", {})
    if not symbols:
        return None
    side_value = {"bullish": 1, "mixed": 0, "bearish": -1}
    coins = [s.removesuffix("USDC") for s in symbols]
    z, text = [], []
    for sym in symbols:
        tfs = symbols[sym].get("timeframes", {})
        row_z, row_t = [], []
        for tf in KEY_TIMEFRAMES:
            brief = tfs.get(tf, {})
            side = brief.get("side", "mixed")
            row_z.append(side_value.get(side, 0))
            row_t.append(f"{side}<br>{brief.get('scenario', '')}")
        z.append(row_z)
        text.append(row_t)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=KEY_TIMEFRAMES,
            y=coins,
            text=text,
            texttemplate="%{text}",
            colorscale=[[0, "#d62728"], [0.5, "#444"], [1, "#2ca02c"]],
            zmid=0,
            showscale=False,
        )
    )
    fig.update_layout(height=260, margin={"l": 10, "r": 10, "t": 10, "b": 10})
    return fig


def render_kpis(state: dict[str, Any], trades: pd.DataFrame, policy: dict | None) -> None:
    closed = trades[trades["status"].eq("closed")] if not trades.empty else trades
    open_trades = trades[trades["status"].eq("open")] if not trades.empty else trades
    win_rate = (
        (pd.to_numeric(closed["realized_return_pct"], errors="coerce") > 0).mean() * 100 if not closed.empty else None
    )
    status = state["worker_status"]
    status_icon = "🟢" if status == "ok" else "🔴" if status == "error" else "⚪"

    row1 = st.columns(3)
    row1[0].metric("Worker", f"{status_icon} {status}", _heartbeat_age(state["last_heartbeat"]))
    row1[1].metric("Paper equity", f"{float(state['paper_equity']):,.0f}", f"DD {float(state['drawdown_pct']):.1f}%")
    row1[2].metric("Open / Closed", f"{len(open_trades)} / {len(closed)}")

    row2 = st.columns(3)
    row2[0].metric("Win rate", "-" if win_rate is None else f"{win_rate:.0f}%")
    row2[1].metric("Sim leverage", f"{float(state['simulated_leverage']):.1f}x")
    row2[2].metric("Policy v", f"{state['policy_version']}", f"thr {float(state['entry_prob_threshold']):.2f}")
    if state["last_cycle_error"]:
        st.error(f"Last worker error: {state['last_cycle_error']}")


def render_monitor() -> None:
    st.subheader("Live Monitor")
    st.caption("Phone-friendly snapshot of the always-on paper bot. Simulated long/short research only - not financial advice.")
    if st.button("Refresh", key="monitor_refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    state = load_bot_state()
    tables = load_tables()
    trades = tables["trades"]
    snapshot = state["snapshot"]

    render_kpis(state, trades, tables["policy"])

    if snapshot:
        st.caption(f"Market snapshot generated {snapshot.get('generated_at', 'n/a')}")
        heatmap = timeframe_heatmap_fig(snapshot)
        if heatmap is not None:
            st.write("**Timeframe bias (green=bullish, red=bearish)**")
            st.plotly_chart(heatmap, use_container_width=True, config={"displayModeBar": False}, key="monitor_heatmap")

        candidates = pd.DataFrame(snapshot.get("candidates", []))
        if not candidates.empty:
            st.write("**Current candidates**")
            st.dataframe(candidates[["Symbol", "Action", "Confidence", "Score"]], use_container_width=True, hide_index=True)
    else:
        st.info("No worker snapshot yet. Start the worker: `python -m src.paper_worker`.")

    closed = _closed_with_pnl(trades)
    if not closed.empty:
        st.write("**Equity curve**")
        st.plotly_chart(equity_curve_fig(closed, float(state["paper_equity"]) - closed["realized_pnl"].sum()), use_container_width=True, config={"displayModeBar": False}, key="monitor_equity_curve")


def render_paper_bot() -> None:
    st.subheader("Paper Bot & Trades")
    st.caption(
        "Local paper simulation only (simulated long and short). Uses live Binance public spot prices; "
        "no account access, API keys, real orders, futures execution, or real leverage."
    )

    if st.button("Refresh now", key="paper_bot_refresh"):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"Panel loaded at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    state = load_bot_state()
    tables = load_tables()
    trades = tables["trades"]
    model = tables["model"]
    policy = tables["policy"]
    snapshot = state["snapshot"]

    render_kpis(state, trades, policy)

    candidate_rows = pd.DataFrame(snapshot.get("candidates", [])) if snapshot else pd.DataFrame()
    st.write("**Current paper candidates**")
    if candidate_rows.empty:
        st.info("No candidates yet. Start the worker to populate this from live market data.")
    else:
        st.dataframe(candidate_rows, use_container_width=True, hide_index=True)
        threshold = float(state["entry_prob_threshold"])
        tradable = candidate_rows[candidate_rows.get("Action", "").isin(["paper_long", "paper_short"])] if "Action" in candidate_rows else pd.DataFrame()
        gauges = [r for _, r in tradable.iterrows() if r.get("p_win") is not None]
        if gauges:
            st.caption("Advisory ML win probability (white line = current entry threshold)")
            gauge_cols = st.columns(min(len(gauges), 3))
            for idx, row in enumerate(gauges[:3]):
                with gauge_cols[idx]:
                    st.caption(f"{row['Symbol']} {row['Action']}")
                    st.plotly_chart(pwin_gauge_fig(float(row["p_win"]), threshold), use_container_width=True, config={"displayModeBar": False}, key=f"pwin_gauge_{row['Symbol']}_{row['Action']}_{idx}")

    active = trades[trades["status"].eq("open")] if not trades.empty else trades
    closed = _closed_with_pnl(trades)

    st.write("**Open paper trades**")
    st.dataframe(active, use_container_width=True, hide_index=True)
    st.write("**Closed paper trades**")
    st.dataframe(closed.tail(50), use_container_width=True, hide_index=True)

    if not closed.empty:
        st.write("**Performance**")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Equity curve")
            st.plotly_chart(equity_curve_fig(closed, float(state["paper_equity"]) - closed["realized_pnl"].sum()), use_container_width=True, config={"displayModeBar": False}, key="paper_equity_curve")
            st.caption("Return distribution")
            st.plotly_chart(pnl_hist_fig(closed), use_container_width=True, config={"displayModeBar": False}, key="paper_pnl_hist")
        with c2:
            st.caption("Drawdown")
            st.plotly_chart(drawdown_fig(closed, float(state["paper_equity"]) - closed["realized_pnl"].sum()), use_container_width=True, config={"displayModeBar": False}, key="paper_drawdown")
            st.caption("Rolling win rate")
            st.plotly_chart(winrate_over_time_fig(closed), use_container_width=True, config={"displayModeBar": False}, key="paper_winrate")
        if closed["side"].nunique() > 1:
            st.caption("Average return by side (long vs short)")
            st.plotly_chart(long_short_fig(closed), use_container_width=True, config={"displayModeBar": False}, key="paper_long_short")

    st.write("**Reinforcement-style policy**")
    st.caption(
        f"The bot retrains on its experience every {PAPER_POLICY_BATCH} closed trades and adjusts an advisory "
        "entry-probability threshold and a size multiplier (capped at the risk ceiling) for the next batch."
    )
    if policy:
        pc = st.columns(4)
        pc[0].metric("Entry threshold", f"{float(policy['entry_prob_threshold']):.2f}")
        pc[1].metric("Size multiplier", f"{float(policy['size_multiplier']):.2f}x")
        pc[2].metric("Batch win rate", "-" if policy.get("batch_win_rate") is None else f"{float(policy['batch_win_rate']):.0f}%")
        pc[3].metric("Batch reward", "-" if policy.get("batch_reward") is None else f"{float(policy['batch_reward']):.1f}")
        if policy.get("note"):
            st.caption(f"Status: {policy['note']}")
    policy_history = tables["policy_history"]
    if isinstance(policy_history, pd.DataFrame) and policy_history["batch_reward"].notna().any():
        st.caption("Reward per learning batch")
        st.plotly_chart(reward_per_batch_fig(policy_history), use_container_width=True, config={"displayModeBar": False}, key="paper_reward_batch")

    calib = calibration_fig(closed) if not closed.empty else None
    if calib is not None:
        st.caption("Model calibration (predicted vs actual win rate)")
        st.plotly_chart(calib, use_container_width=True, config={"displayModeBar": False}, key="paper_calibration")

    st.write("**ML learning status**")
    if model is None:
        st.info("No model run yet. Start the paper worker to begin collecting closed paper-trade outcomes.")
    elif model.get("skipped_reason"):
        st.warning(f"Model status: {model['skipped_reason']} ({model['sample_count']} closed trades).")
    else:
        ml_cols = st.columns(4)
        ml_cols[0].metric("Samples", model["sample_count"])
        ml_cols[1].metric("Accuracy", f"{float(model['accuracy']):.2f}")
        ml_cols[2].metric("Precision", f"{float(model['precision']):.2f}")
        ml_cols[3].metric("Recall", f"{float(model['recall']):.2f}")
        importance = feature_importance_fig(model)
        if importance is not None:
            st.plotly_chart(importance, use_container_width=True, config={"displayModeBar": False}, key="paper_feature_importance")


def render_alert_dicts(alerts: list[dict[str, Any]]) -> None:
    severity_order = {"Risk": 0, "Signal": 1, "Watch": 2}
    ordered = sorted(alerts, key=lambda a: (severity_order.get(a.get("severity"), 9), a.get("symbol"), a.get("timeframe")))
    for alert in ordered[:25]:
        line = f"{alert.get('symbol')} {alert.get('timeframe')}: {alert.get('title')} - {alert.get('detail')}"
        if alert.get("severity") == "Risk":
            st.error(line)
        elif alert.get("severity") == "Signal":
            st.success(line)
        else:
            st.info(line)


def snapshot_watchlist(snapshot: dict[str, Any], timeframe: str) -> pd.DataFrame:
    rows = []
    for symbol, payload in snapshot.get("symbols", {}).items():
        brief = payload.get("timeframes", {}).get(timeframe, {})
        rows.append(
            {
                "Symbol": symbol,
                "Price": payload.get("price"),
                "24h %": payload.get("price_change_pct"),
                "Spot vol": _format_money(payload.get("spot_quote_volume")),
                "Fut/spot": payload.get("futures_spot_ratio"),
                "Funding": payload.get("funding_rate"),
                "BTC corr": payload.get("btc_corr"),
                "RSI": brief.get("rsi"),
                "Trend": brief.get("ema_state"),
                "ADX": brief.get("adx"),
                "MACD": brief.get("macd_state"),
                "Regime": brief.get("volatility_regime"),
                "Scenario": brief.get("scenario"),
                "Confidence": brief.get("confidence"),
                "Alerts": brief.get("alerts_total"),
            }
        )
    return pd.DataFrame(rows)


def snapshot_matrix(snapshot: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for symbol, payload in snapshot.get("symbols", {}).items():
        row: dict[str, Any] = {"Coin": _base_asset(symbol)}
        for timeframe in KEY_TIMEFRAMES:
            brief = payload.get("timeframes", {}).get(timeframe, {})
            rsi = brief.get("rsi")
            rsi_text = f"{rsi:.0f}" if isinstance(rsi, (int, float)) else "-"
            row[timeframe] = (
                f"{brief.get('scenario', '-')} | {brief.get('ema_state', '-')} | "
                f"RSI {rsi_text} | {brief.get('bollinger_state', '-')} | {brief.get('confidence', '-')}"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def snapshot_comparison(snapshot: dict[str, Any], symbol: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = snapshot.get("symbols", {}).get(symbol, {})
    rows = []
    for timeframe in KEY_TIMEFRAMES:
        brief = payload.get("timeframes", {}).get(timeframe, {})
        rows.append(
            {
                "Timeframe": timeframe,
                "Scenario": brief.get("scenario"),
                "Confidence": brief.get("confidence"),
                "Trend": brief.get("ema_state"),
                "RSI": brief.get("rsi"),
                "MACD": brief.get("macd_state"),
                "ADX": brief.get("adx"),
                "Bollinger": brief.get("bollinger_state"),
                "Regime": brief.get("volatility_regime"),
                "Structure": brief.get("market_structure"),
                "Elliott": f"{brief.get('elliott_state')} ({brief.get('elliott_confidence')})",
                "Side": brief.get("side"),
            }
        )
    return pd.DataFrame(rows), payload.get("comparison", {})


def snapshot_alerts(snapshot: dict[str, Any], timeframe: str | None = None) -> list[dict[str, Any]]:
    alerts = snapshot.get("alerts", [])
    if timeframe is None:
        return alerts
    return [a for a in alerts if a.get("timeframe") == timeframe]


st.title("Crypto Paper Bot Monitor")
st.caption("Read-only market analysis and simulated paper trading. No account access, no real execution, no financial advice.")

with st.sidebar:
    selected_symbol = st.selectbox("Symbol", SYMBOLS, index=0)
    selected_timeframe = st.selectbox("Chart timeframe", TIMEFRAMES, index=8)
    matrix_timeframe = st.selectbox("Watchlist timeframe", KEY_TIMEFRAMES, index=2)
    show_elliott = st.toggle("Show Elliott labels", value=True)
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

monitor_tab, market_tab, paper_tab = st.tabs(DASHBOARD_TABS)

with monitor_tab:
    try:
        render_monitor()
    except Exception as exc:
        st.exception(exc)

with market_tab:
    try:
        snapshot = load_bot_state()["snapshot"]
        st.subheader("Market Overview")
        if snapshot:
            st.caption(f"From worker snapshot generated {snapshot.get('generated_at', 'n/a')} (timeframe: {matrix_timeframe})")
            st.dataframe(snapshot_watchlist(snapshot, matrix_timeframe), use_container_width=True, hide_index=True)
        else:
            st.info("No worker snapshot yet. Showing live computation as fallback.")
            watchlist, _, _ = build_watchlist(matrix_timeframe)
            st.dataframe(watchlist, use_container_width=True, hide_index=True)

        chart_df = load_spot_klines(selected_symbol, selected_timeframe, DEFAULT_CANDLE_LIMIT)
        btc_df = load_spot_klines("BTCUSDC", selected_timeframe, MATRIX_CANDLE_LIMIT)
        btc_analysis = analyze_timeframe("BTCUSDC", selected_timeframe, btc_df, load_futures_context("BTCUSDC", selected_timeframe))
        selected_analysis = analyze_timeframe(
            selected_symbol,
            selected_timeframe,
            chart_df,
            load_futures_context(selected_symbol, selected_timeframe),
            btc_risk_from_analysis(btc_analysis) if selected_symbol != "BTCUSDC" else None,
        )

        left, right = st.columns([0.72, 0.28])
        with left:
            st.subheader(f"{selected_symbol} {selected_timeframe}")
            st.plotly_chart(make_chart(chart_df, selected_analysis, show_elliott), use_container_width=True, key="market_main_chart")
        with right:
            st.subheader("Scenario")
            st.metric("Read", selected_analysis["scenario"], selected_analysis["confidence"])
            st.write(f"**EMA:** {selected_analysis['ema_state']}")
            st.write(f"**RSI:** {selected_analysis['rsi']:.1f} ({selected_analysis['rsi_state']})")
            st.write(f"**Bollinger:** {selected_analysis['bollinger_state']}")
            st.write(f"**Relative volume:** {selected_analysis['relative_volume']:.2f}x")
            st.write(f"**Structure:** {selected_analysis['market_structure']}")
            st.write(
                f"**Nearest Fib:** {selected_analysis['nearest_fib']} at "
                f"{selected_analysis['nearest_fib_value']:.8g} "
                f"({selected_analysis['fib_distance_pct']:.2f}% away)"
            )
            st.write(f"**Elliott:** {selected_analysis['elliott_state']} ({selected_analysis['elliott_confidence']})")
            st.caption(selected_analysis["elliott_detail"])
            st.subheader("Selected Alerts")
            render_alerts(selected_analysis["alerts"])

        st.subheader("Key Timeframe Matrix")
        if snapshot:
            st.dataframe(snapshot_matrix(snapshot), use_container_width=True, hide_index=True)
        else:
            st.dataframe(build_matrix(), use_container_width=True, hide_index=True)

        st.subheader(f"{_base_asset(selected_symbol)} Timeframe Comparison")
        if snapshot and selected_symbol in snapshot.get("symbols", {}):
            comparison, comparison_meta = snapshot_comparison(snapshot, selected_symbol)
            comparison_read = comparison_meta.get("read", "-")
            comparison_confidence = comparison_meta.get("confidence", "-")
            comparison_notes = comparison_meta.get("notes", [])
        else:
            comparison, comparison_read, comparison_confidence, comparison_notes, _, _ = build_timeframe_comparison(selected_symbol)
        col_a, col_b = st.columns([0.35, 0.65])
        with col_a:
            st.metric("Cross-timeframe read", comparison_read, comparison_confidence)
            for note in comparison_notes:
                st.info(note)
        with col_b:
            st.dataframe(comparison, use_container_width=True, hide_index=True)

        st.subheader("Active Alerts")
        if snapshot:
            render_alert_dicts(snapshot_alerts(snapshot, matrix_timeframe))
        else:
            _, alerts, _ = build_watchlist(matrix_timeframe)
            render_alerts(alerts)

        st.subheader("BNB Relationship Panel")
        render_bnb_relationship(matrix_timeframe)
    except BinanceClientError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.exception(exc)

with paper_tab:
    try:
        render_paper_bot()
    except Exception as exc:
        st.exception(exc)
