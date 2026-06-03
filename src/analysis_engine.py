"""Shared market analysis routines for Streamlit and the paper worker."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .binance_client import BinanceClient
from .config import DERIVATIVES_PERIOD, KEY_TIMEFRAMES, MATRIX_CANDLE_LIMIT, SYMBOLS
from .indicators import add_indicators
from .signals import Alert, analyze_timeframe, btc_risk_from_analysis, summarize_alert_counts


def to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def trend_side(analysis: dict[str, Any]) -> str:
    ema = analysis["ema_state"]
    rsi = analysis["rsi"]
    if ema.startswith("bullish") and rsi >= 50:
        return "bullish"
    if ema.startswith("bearish") and rsi <= 50:
        return "bearish"
    return "mixed"


def derivatives_context(client: BinanceClient, symbol: str) -> dict[str, Any]:
    """Optional futures-derived context. Never raises: futures endpoints may be
    region-restricted (451) and must not block the spot-only pipeline."""
    spot_quote_volume = to_float((client.spot_24h(symbol) or {}).get("quoteVolume"))
    futures_quote_volume = to_float((client.futures_24h(symbol) or {}).get("quoteVolume"))
    ratio = (
        futures_quote_volume / spot_quote_volume
        if spot_quote_volume and futures_quote_volume
        else None
    )
    open_interest = to_float((client.open_interest(symbol) or {}).get("openInterest"))
    funding_rate = to_float((client.mark_price(symbol) or {}).get("lastFundingRate"))

    taker = client.taker_buy_sell_volume(symbol, DERIVATIVES_PERIOD, limit=1)
    taker_buy_ratio = to_float(taker[-1].get("buySellRatio")) if taker else None
    ls = client.long_short_ratio(symbol, DERIVATIVES_PERIOD, limit=1)
    long_short_ratio = to_float(ls[-1].get("longShortRatio")) if ls else None

    return {
        "spot_quote_volume": spot_quote_volume,
        "futures_quote_volume": futures_quote_volume,
        "futures_spot_ratio": ratio,
        "open_interest": open_interest,
        "open_interest_change": "expanding" if open_interest and ratio and ratio >= 2 else None,
        "funding_rate": funding_rate,
        "taker_buy_ratio": taker_buy_ratio,
        "long_short_ratio": long_short_ratio,
    }


def correlation_with_btc(symbol_df: pd.DataFrame, btc_df: pd.DataFrame) -> float | None:
    try:
        length = min(len(symbol_df), len(btc_df))
        if length < 20:
            return None
        sym = symbol_df["close"].tail(length).pct_change().dropna().reset_index(drop=True)
        btc = btc_df["close"].tail(length).pct_change().dropna().reset_index(drop=True)
        joined = pd.concat([sym, btc], axis=1).dropna()
        if len(joined) < 20:
            return None
        value = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
        return None if pd.isna(value) else round(value, 4)
    except Exception:
        return None


def market_snapshot(client: BinanceClient, symbol: str) -> dict[str, Any]:
    spot = client.spot_24h(symbol)
    book = client.book_ticker(symbol) or {}
    bid = to_float(book.get("bidPrice"))
    ask = to_float(book.get("askPrice"))
    spread = ((ask - bid) / ask * 100) if bid and ask else None
    return {
        "symbol": symbol,
        "price": to_float(spot.get("lastPrice")),
        "price_change_pct": to_float(spot.get("priceChangePercent")),
        "volume": to_float(spot.get("volume")),
        "quote_volume": to_float(spot.get("quoteVolume")),
        "bid": bid,
        "ask": ask,
        "spread_pct": spread,
    }


def key_timeframe_analyses(symbol: str, client: BinanceClient | None = None) -> dict[str, dict[str, Any]]:
    client = client or BinanceClient()
    analyses: dict[str, dict[str, Any]] = {}
    btc_risk_by_timeframe: dict[str, str] = {}
    for timeframe in KEY_TIMEFRAMES:
        btc_df = add_indicators(client.spot_klines("BTCUSDC", timeframe, MATRIX_CANDLE_LIMIT))
        btc_analysis = analyze_timeframe("BTCUSDC", timeframe, btc_df, None)
        btc_risk_by_timeframe[timeframe] = btc_risk_from_analysis(btc_analysis)
    for timeframe in KEY_TIMEFRAMES:
        df = add_indicators(client.spot_klines(symbol, timeframe, MATRIX_CANDLE_LIMIT))
        analyses[timeframe] = analyze_timeframe(
            symbol,
            timeframe,
            df,
            None,
            btc_risk_by_timeframe[timeframe] if symbol != "BTCUSDC" else None,
        )
    return analyses


def latest_prices(client: BinanceClient | None = None) -> dict[str, float]:
    client = client or BinanceClient()
    prices: dict[str, float] = {}
    for symbol in SYMBOLS:
        snapshot = market_snapshot(client, symbol)
        price = snapshot["price"]
        if price is not None:
            prices[symbol] = price
    return prices


def comparison_summary(analyses: dict[str, dict[str, Any]]) -> tuple[str, str, list[str]]:
    sides = {timeframe: trend_side(analysis) for timeframe, analysis in analyses.items()}
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


def _clean(value: Any) -> Any:
    number = to_float(value)
    if number is None:
        return None
    if number != number:  # NaN
        return None
    return number


def gather_cycle_data(client: BinanceClient | None = None) -> dict[str, dict[str, Any]]:
    """Single-pass computation of everything the worker needs per cycle.

    Fetches BTC reference frames once, then per symbol computes key-timeframe
    analyses, the live market snapshot, optional derivatives context, and the
    BTC correlation. Heavy work runs here (in the worker), not in the UI.
    """
    client = client or BinanceClient()
    btc_dfs = {tf: add_indicators(client.spot_klines("BTCUSDC", tf, MATRIX_CANDLE_LIMIT)) for tf in KEY_TIMEFRAMES}
    btc_risk = {
        tf: btc_risk_from_analysis(analyze_timeframe("BTCUSDC", tf, btc_dfs[tf], None)) for tf in KEY_TIMEFRAMES
    }

    data: dict[str, dict[str, Any]] = {}
    for symbol in SYMBOLS:
        analyses: dict[str, dict[str, Any]] = {}
        dfs: dict[str, pd.DataFrame] = {}
        for tf in KEY_TIMEFRAMES:
            df = add_indicators(client.spot_klines(symbol, tf, MATRIX_CANDLE_LIMIT))
            dfs[tf] = df
            analyses[tf] = analyze_timeframe(
                symbol,
                tf,
                df,
                None,
                btc_risk[tf] if symbol != "BTCUSDC" else None,
            )
        reference_tf = "12h" if "12h" in dfs else KEY_TIMEFRAMES[0]
        data[symbol] = {
            "analyses": analyses,
            "snapshot": market_snapshot(client, symbol),
            "derivatives": derivatives_context(client, symbol),
            "btc_corr": correlation_with_btc(dfs[reference_tf], btc_dfs[reference_tf]),
        }
    return data


def analysis_brief(analysis: dict[str, Any], side: str) -> dict[str, Any]:
    counts = summarize_alert_counts(analysis.get("alerts", []))
    return {
        "scenario": analysis.get("scenario"),
        "ema_state": analysis.get("ema_state"),
        "rsi": _clean(analysis.get("rsi")),
        "rsi_state": analysis.get("rsi_state"),
        "bollinger_state": analysis.get("bollinger_state"),
        "confidence": analysis.get("confidence"),
        "relative_volume": _clean(analysis.get("relative_volume")),
        "market_structure": analysis.get("market_structure"),
        "nearest_fib": analysis.get("nearest_fib"),
        "fib_distance_pct": _clean(analysis.get("fib_distance_pct")),
        "elliott_state": analysis.get("elliott_state"),
        "elliott_confidence": analysis.get("elliott_confidence"),
        "macd_state": analysis.get("macd_state"),
        "adx": _clean(analysis.get("adx")),
        "adx_state": analysis.get("adx_state"),
        "vwap_state": analysis.get("vwap_state"),
        "vwap_distance_pct": _clean(analysis.get("vwap_distance_pct")),
        "volatility_regime": analysis.get("volatility_regime"),
        "side": side,
        "alerts_total": len(analysis.get("alerts", [])),
        "watch": counts["Watch"],
        "signal": counts["Signal"],
        "risk": counts["Risk"],
    }


def build_dashboard_snapshot(
    now: pd.Timestamp,
    data: dict[str, dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a JSON-safe snapshot the dashboard can render without recomputation."""
    symbols_payload: dict[str, Any] = {}
    alerts_payload: list[dict[str, Any]] = []

    for symbol, bundle in data.items():
        analyses = bundle["analyses"]
        snapshot = bundle["snapshot"]
        derivatives = bundle["derivatives"]
        sides = {tf: trend_side(analyses[tf]) for tf in analyses}
        read, confidence, notes = comparison_summary(analyses)
        timeframes = {tf: analysis_brief(analyses[tf], sides[tf]) for tf in analyses}
        for tf, analysis in analyses.items():
            for alert in analysis.get("alerts", []):
                alerts_payload.append(
                    {
                        "severity": alert.severity,
                        "symbol": alert.symbol,
                        "timeframe": alert.timeframe,
                        "title": alert.title,
                        "detail": alert.detail,
                    }
                )
        symbols_payload[symbol] = {
            "price": _clean(snapshot.get("price")),
            "price_change_pct": _clean(snapshot.get("price_change_pct")),
            "spot_quote_volume": _clean(snapshot.get("quote_volume")),
            "spread_pct": _clean(snapshot.get("spread_pct")),
            "futures_quote_volume": _clean(derivatives.get("futures_quote_volume")),
            "futures_spot_ratio": _clean(derivatives.get("futures_spot_ratio")),
            "funding_rate": _clean(derivatives.get("funding_rate")),
            "taker_buy_ratio": _clean(derivatives.get("taker_buy_ratio")),
            "long_short_ratio": _clean(derivatives.get("long_short_ratio")),
            "open_interest": _clean(derivatives.get("open_interest")),
            "btc_corr": _clean(bundle.get("btc_corr")),
            "timeframes": timeframes,
            "comparison": {"read": read, "confidence": confidence, "notes": notes},
            "side_counts": {
                "bullish": sum(1 for s in sides.values() if s == "bullish"),
                "bearish": sum(1 for s in sides.values() if s == "bearish"),
                "mixed": sum(1 for s in sides.values() if s == "mixed"),
            },
        }

    return {
        "generated_at": now.isoformat(),
        "symbols": symbols_payload,
        "alerts": alerts_payload,
        "candidates": candidate_rows,
    }
