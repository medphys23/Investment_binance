from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.binance_client import BinanceClient, BinanceClientError
from src.config import DASHBOARD_TABS, DEFAULT_CANDLE_LIMIT, KEY_TIMEFRAMES, MATRIX_CANDLE_LIMIT, SYMBOLS, TIMEFRAMES
from src.indicators import add_indicators
from src.signals import Alert, analyze_timeframe, btc_risk_from_analysis, summarize_alert_counts
from src.storage import connect, get_state, initialize_database, latest_model_run, table
from src.strategy import candidate_to_row, generate_paper_trade_candidate


st.set_page_config(page_title="Binance Technical Signal Dashboard", layout="wide")


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


def render_paper_bot() -> None:
    st.subheader("Paper Bot & Trades")
    st.caption("Local paper simulation only. Uses live Binance public spot prices; no account access, API keys, real orders, futures execution, or real leverage.")

    refresh_col, hint_col = st.columns([1, 4])
    if refresh_col.button("Refresh now", key="paper_bot_refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    hint_col.caption("Click Refresh now to reload worker status, trades, and candidates from SQLite and Binance.")

    refreshed_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.caption(f"Panel loaded at {refreshed_at}")

    candidates = []
    for symbol in SYMBOLS:
        analyses = build_key_timeframe_analyses(symbol)
        candidate = generate_paper_trade_candidate(symbol, analyses)
        candidates.append(candidate)

    candidate_rows = pd.DataFrame([candidate_to_row(candidate) for candidate in candidates])
    st.write("**Current paper candidates**")
    st.dataframe(candidate_rows, use_container_width=True, hide_index=True)

    conn = connect()
    initialize_database(conn)
    state = {
        "worker_status": get_state(conn, "worker_status", "not_started"),
        "last_heartbeat": get_state(conn, "last_heartbeat", ""),
        "last_cycle_error": get_state(conn, "last_cycle_error", ""),
        "paper_equity": get_state(conn, "paper_equity", 10000.0),
        "drawdown_pct": get_state(conn, "drawdown_pct", 0.0),
        "simulated_leverage": get_state(conn, "simulated_leverage", 1.0),
    }
    trades = table(conn, "paper_trades")
    model = latest_model_run(conn)
    conn.close()
    active = trades[trades["status"].eq("open")]
    closed = trades[trades["status"].eq("closed")]
    avg_realized = closed["realized_return_pct"].dropna().astype(float).mean() if not closed.empty else None
    win_rate = (closed["realized_return_pct"].dropna().astype(float) > 0).mean() * 100 if not closed.empty else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Worker", state["worker_status"])
    col2.metric("Paper equity", f"{float(state['paper_equity']):,.2f} USDC")
    col3.metric("Drawdown", f"{float(state['drawdown_pct']):.2f}%")
    col4.metric("Sim leverage tier", f"{float(state['simulated_leverage']):.1f}x")
    st.write(f"**Last worker heartbeat:** {state['last_heartbeat'] or 'not started'}")
    if state["last_cycle_error"]:
        st.error(f"Last worker error: {state['last_cycle_error']}")

    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("Open paper trades", len(active))
    metric_b.metric("Closed paper trades", len(closed))
    metric_c.metric("Win rate", "-" if win_rate is None else f"{win_rate:.2f}%")
    metric_d.metric("Average return", "-" if avg_realized is None else f"{avg_realized:.2f}%")

    st.write("**Open paper trades**")
    st.dataframe(active, use_container_width=True, hide_index=True)
    st.write("**Closed paper trades**")
    st.dataframe(closed.tail(50), use_container_width=True, hide_index=True)

    st.write("**ML learning status**")
    if model is None:
        st.info("No model run yet. Start the paper worker to begin collecting closed paper-trade outcomes.")
    else:
        if model.get("skipped_reason"):
            st.warning(f"Model skipped: {model['skipped_reason']} ({model['sample_count']} closed trades).")
        else:
            ml_cols = st.columns(4)
            ml_cols[0].metric("Samples", model["sample_count"])
            ml_cols[1].metric("Accuracy", f"{float(model['accuracy']):.2f}")
            ml_cols[2].metric("Precision", f"{float(model['precision']):.2f}")
            ml_cols[3].metric("Recall", f"{float(model['recall']):.2f}")
            top_features = pd.DataFrame(json.loads(model["top_features"] or "[]"))
            st.dataframe(top_features, use_container_width=True, hide_index=True)


st.title("Binance Technical Signal Dashboard")
st.caption("Read-only market analysis. No account access, no trade execution, and no financial advice.")

with st.sidebar:
    selected_symbol = st.selectbox("Symbol", SYMBOLS, index=0)
    selected_timeframe = st.selectbox("Chart timeframe", TIMEFRAMES, index=8)
    matrix_timeframe = st.selectbox("Watchlist timeframe", KEY_TIMEFRAMES, index=2)
    show_elliott = st.toggle("Show Elliott labels", value=True)
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

market_tab, paper_tab = st.tabs(DASHBOARD_TABS)

with market_tab:
    try:
        watchlist, alerts, watchlist_analyses = build_watchlist(matrix_timeframe)
        st.subheader("Market Overview")
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
            st.plotly_chart(make_chart(chart_df, selected_analysis, show_elliott), use_container_width=True)
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
        st.dataframe(build_matrix(), use_container_width=True, hide_index=True)

        st.subheader(f"{_base_asset(selected_symbol)} Timeframe Comparison")
        comparison, comparison_read, comparison_confidence, comparison_notes, comparison_alerts, selected_analyses = build_timeframe_comparison(selected_symbol)
        col_a, col_b = st.columns([0.35, 0.65])
        with col_a:
            st.metric("Cross-timeframe read", comparison_read, comparison_confidence)
            for note in comparison_notes:
                st.info(note)
        with col_b:
            st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.subheader("Comparison Alerts")
        render_alerts(comparison_alerts)

        st.subheader("Active Alerts")
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
