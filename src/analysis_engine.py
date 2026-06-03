"""Shared market analysis routines for Streamlit and the paper worker."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .binance_client import BinanceClient
from .config import KEY_TIMEFRAMES, MATRIX_CANDLE_LIMIT, SYMBOLS
from .indicators import add_indicators
from .signals import analyze_timeframe, btc_risk_from_analysis


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
