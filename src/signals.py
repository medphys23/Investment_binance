"""Signal and alert logic for the Binance technical dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import EMA_PERIODS, INTERMEDIATE_TIMEFRAMES
from .indicators import FibLevels, fibonacci_levels, find_pivots, market_structure


@dataclass(frozen=True)
class Alert:
    severity: str
    symbol: str
    timeframe: str
    title: str
    detail: str


def _crossed_above(series_a: pd.Series, series_b: pd.Series) -> bool:
    return bool(len(series_a) >= 2 and series_a.iloc[-2] <= series_b.iloc[-2] and series_a.iloc[-1] > series_b.iloc[-1])


def _crossed_below(series_a: pd.Series, series_b: pd.Series) -> bool:
    return bool(len(series_a) >= 2 and series_a.iloc[-2] >= series_b.iloc[-2] and series_a.iloc[-1] < series_b.iloc[-1])


def ema_state(row: pd.Series) -> str:
    values = [row[f"ema_{period}"] for period in EMA_PERIODS]
    if any(pd.isna(values)):
        return "forming"
    if all(values[i] > values[i + 1] for i in range(len(values) - 1)):
        return "bullish stack"
    if all(values[i] < values[i + 1] for i in range(len(values) - 1)):
        return "bearish stack"
    if row["close"] > row["ema_20"] > row["ema_50"]:
        return "bullish bias"
    if row["close"] < row["ema_20"] < row["ema_50"]:
        return "bearish bias"
    return "mixed"


def rsi_state(rsi: float) -> str:
    if rsi >= 70:
        return "overbought"
    if rsi <= 30:
        return "oversold"
    if rsi >= 55:
        return "bullish"
    if rsi <= 45:
        return "bearish"
    return "neutral"


def bollinger_state(df: pd.DataFrame) -> str:
    row = df.iloc[-1]
    width = row["bb_width"]
    width_rank = df["bb_width"].tail(120).rank(pct=True).iloc[-1]
    if pd.isna(width):
        return "forming"
    if width_rank <= 0.2:
        return "squeeze"
    if row["close"] > row["bb_upper"]:
        return "upper expansion"
    if row["close"] < row["bb_lower"]:
        return "lower expansion"
    if width_rank >= 0.8:
        return "wide bands"
    return "normal"


def macd_state(df: pd.DataFrame) -> str:
    if "macd_hist" not in df or pd.isna(df["macd_hist"].iloc[-1]):
        return "forming"
    hist = float(df["macd_hist"].iloc[-1])
    prev_hist = float(df["macd_hist"].iloc[-2]) if len(df) >= 2 else hist
    if hist > 0 and hist >= prev_hist:
        return "bullish expanding"
    if hist > 0:
        return "bullish fading"
    if hist < 0 and hist <= prev_hist:
        return "bearish expanding"
    if hist < 0:
        return "bearish fading"
    return "flat"


def adx_state(df: pd.DataFrame) -> tuple[float, str]:
    if "adx" not in df or pd.isna(df["adx"].iloc[-1]):
        return float("nan"), "forming"
    adx = float(df["adx"].iloc[-1])
    if adx >= 40:
        return adx, "very strong trend"
    if adx >= 25:
        return adx, "trending"
    if adx >= 20:
        return adx, "weak trend"
    return adx, "ranging"


def vwap_state(df: pd.DataFrame) -> tuple[float, str]:
    if "vwap_distance_pct" not in df or pd.isna(df["vwap_distance_pct"].iloc[-1]):
        return float("nan"), "forming"
    distance = float(df["vwap_distance_pct"].iloc[-1])
    if distance >= 0.25:
        return distance, "above VWAP"
    if distance <= -0.25:
        return distance, "below VWAP"
    return distance, "at VWAP"


def volatility_regime(df: pd.DataFrame, bollinger: str) -> str:
    atr_pct = df["atr_percentile"].iloc[-1] if "atr_percentile" in df else np.nan
    adx = df["adx"].iloc[-1] if "adx" in df else np.nan
    if pd.isna(atr_pct) or pd.isna(adx):
        return "forming"
    if bollinger == "squeeze" or atr_pct <= 0.25:
        return "compressed"
    if adx >= 25 and atr_pct >= 0.5:
        return "trending-expansion"
    if atr_pct >= 0.8:
        return "high-volatility"
    return "normal"


def _nearest_fib(price: float, fib: FibLevels) -> tuple[str, float, float]:
    candidates = {**fib.levels, **fib.extensions}
    name, level = min(candidates.items(), key=lambda item: abs(price - item[1]))
    distance_pct = abs(price - level) / price if price else np.nan
    return name, level, distance_pct


def _rsi_divergence(df: pd.DataFrame, pivots: list[dict[str, Any]]) -> str | None:
    highs = [p for p in pivots if p["kind"] == "high"]
    lows = [p for p in pivots if p["kind"] == "low"]
    if len(highs) >= 2:
        last, prev = highs[-1], highs[-2]
        if float(last["price"]) > float(prev["price"]) and df.iloc[int(last["index"])]["rsi"] < df.iloc[int(prev["index"])]["rsi"]:
            return "bearish RSI divergence"
    if len(lows) >= 2:
        last, prev = lows[-1], lows[-2]
        if float(last["price"]) < float(prev["price"]) and df.iloc[int(last["index"])]["rsi"] > df.iloc[int(prev["index"])]["rsi"]:
            return "bullish RSI divergence"
    return None


def elliott_wave_read(df: pd.DataFrame, fib: FibLevels, pivots: list[dict[str, Any]]) -> dict[str, Any]:
    if len(pivots) < 5:
        return {"state": "insufficient pivots", "confidence": "low", "detail": "Need more pivots for a wave read."}

    recent = pivots[-6:]
    kinds = [p["kind"] for p in recent]
    prices = [float(p["price"]) for p in recent]
    rsi = float(df["rsi"].iloc[-1])
    rel_volume = float(df["relative_volume"].iloc[-1]) if not pd.isna(df["relative_volume"].iloc[-1]) else 1.0

    bullish_impulse = kinds in (
        ["low", "high", "low", "high", "low", "high"],
        ["high", "low", "high", "low", "high"],
    )
    bearish_impulse = kinds in (
        ["high", "low", "high", "low", "high", "low"],
        ["low", "high", "low", "high", "low"],
    )
    score = 0
    detail_parts: list[str] = []

    if bullish_impulse and fib.direction == "bullish":
        score += 2
        detail_parts.append("pivot sequence resembles a bullish impulse")
        if rsi >= 50:
            score += 1
            detail_parts.append("RSI supports bullish continuation")
        if rel_volume >= 1.2:
            score += 1
            detail_parts.append("relative volume supports participation")
        state = "possible bullish impulse"
    elif bearish_impulse and fib.direction == "bearish":
        score += 2
        detail_parts.append("pivot sequence resembles a bearish impulse")
        if rsi <= 50:
            score += 1
            detail_parts.append("RSI supports bearish continuation")
        if rel_volume >= 1.2:
            score += 1
            detail_parts.append("relative volume supports participation")
        state = "possible bearish impulse"
    elif len(recent) >= 4 and kinds[-4:] in (["high", "low", "high", "low"], ["low", "high", "low", "high"]):
        score += 2
        state = "possible ABC correction"
        detail_parts.append("latest pivots resemble an ABC corrective leg")
        nearest_name, _, distance = _nearest_fib(float(df["close"].iloc[-1]), fib)
        if distance <= 0.015:
            score += 1
            detail_parts.append(f"price is near Fib {nearest_name}")
    else:
        state = "no clear wave"
        detail_parts.append("pivot sequence is not clean enough for a wave label")

    confidence = "high" if score >= 4 else "medium" if score >= 3 else "low"
    return {"state": state, "confidence": confidence, "detail": "; ".join(detail_parts)}


def analyze_timeframe(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    futures_context: dict[str, Any] | None = None,
    btc_risk: str | None = None,
) -> dict[str, Any]:
    row = df.iloc[-1]
    fib = fibonacci_levels(df)
    pivots = find_pivots(df)
    structure = market_structure(pivots)
    nearest_fib_name, nearest_fib_value, nearest_fib_distance = _nearest_fib(float(row["close"]), fib)
    bb = bollinger_state(df)
    ema = ema_state(row)
    rsi = rsi_state(float(row["rsi"]))
    macd = macd_state(df)
    adx_value, adx_label = adx_state(df)
    vwap_distance, vwap_label = vwap_state(df)
    regime = volatility_regime(df, bb)
    elliott = elliott_wave_read(df, fib, pivots)
    alerts: list[Alert] = []

    if _crossed_above(df["ema_9"], df["ema_20"]):
        alerts.append(Alert("Signal", symbol, timeframe, "EMA bullish inversion", "EMA 9 crossed above EMA 20."))
    if _crossed_below(df["ema_9"], df["ema_20"]):
        alerts.append(Alert("Signal", symbol, timeframe, "EMA bearish inversion", "EMA 9 crossed below EMA 20."))
    if len(df) >= 2 and df["rsi"].iloc[-2] < 50 <= df["rsi"].iloc[-1]:
        alerts.append(Alert("Signal", symbol, timeframe, "RSI reclaimed 50", "Momentum moved back above the RSI midpoint."))
    if len(df) >= 2 and df["rsi"].iloc[-2] > 50 >= df["rsi"].iloc[-1]:
        alerts.append(Alert("Risk", symbol, timeframe, "RSI lost 50", "Momentum moved below the RSI midpoint."))
    if bb == "squeeze":
        alerts.append(Alert("Watch", symbol, timeframe, "Bollinger squeeze", "Band width is in the lowest recent quintile."))
    if "expansion" in bb and row["relative_volume"] >= 1.2:
        alerts.append(Alert("Signal", symbol, timeframe, "Bollinger expansion with volume", f"{bb} confirmed by relative volume."))
    if nearest_fib_distance <= 0.01 and row["relative_volume"] >= 1.0:
        alerts.append(
            Alert("Watch", symbol, timeframe, "Fib level in play", f"Price is near Fib {nearest_fib_name} at {nearest_fib_value:.8g}.")
        )

    divergence = _rsi_divergence(df, pivots)
    if divergence:
        severity = "Risk" if divergence.startswith("bearish") else "Signal"
        alerts.append(Alert(severity, symbol, timeframe, "RSI divergence", divergence))

    if elliott["state"] != "no clear wave" and elliott["state"] != "insufficient pivots" and elliott["confidence"] != "low":
        alerts.append(Alert("Watch", symbol, timeframe, "Elliott Wave candidate", f"{elliott['state']} ({elliott['confidence']})."))

    if timeframe in INTERMEDIATE_TIMEFRAMES and ema in ("bullish stack", "bearish stack"):
        alerts.append(Alert("Signal", symbol, timeframe, "6h/8h/12h alignment marker", f"{timeframe} shows {ema}."))

    if not symbol.startswith("BTC") and btc_risk == "bearish" and ema.startswith("bullish"):
        alerts.append(Alert("Risk", symbol, timeframe, "BTC risk filter", "BTC trend is bearish, downgrading bullish altcoin read."))

    futures_score = 0
    if futures_context:
        futures_ratio = futures_context.get("futures_spot_ratio")
        if symbol.startswith("BNB") and futures_ratio and futures_ratio >= 3:
            alerts.append(Alert("Watch", symbol, timeframe, "BNB futures/spot ratio elevated", f"Ratio is {futures_ratio:.2f}x."))
            futures_score += 1
        if futures_context.get("open_interest_change") == "expanding":
            alerts.append(Alert("Watch", symbol, timeframe, "Open interest expansion", "Open interest is present and price volatility is elevated."))
            futures_score += 1

    if adx_label in ("trending", "very strong trend"):
        if macd.startswith("bullish") and ema.startswith("bullish"):
            alerts.append(Alert("Signal", symbol, timeframe, "Trend-strength confirmation", f"ADX {adx_value:.0f} with bullish MACD."))
        elif macd.startswith("bearish") and ema.startswith("bearish"):
            alerts.append(Alert("Signal", symbol, timeframe, "Trend-strength confirmation", f"ADX {adx_value:.0f} with bearish MACD."))

    score = 0
    score += 2 if ema in ("bullish stack", "bearish stack") else 1 if "bias" in ema else 0
    score += 1 if rsi in ("bullish", "bearish") else 0
    score += 1 if bb in ("upper expansion", "lower expansion", "squeeze") else 0
    score += 1 if row["relative_volume"] >= 1.2 else 0
    score += 1 if nearest_fib_distance <= 0.015 else 0
    score += 1 if elliott["confidence"] == "medium" else 2 if elliott["confidence"] == "high" else 0
    score += 1 if adx_label in ("trending", "very strong trend") else 0
    score += 1 if macd in ("bullish expanding", "bearish expanding") else 0
    score += futures_score
    confidence = "high" if score >= 7 else "medium" if score >= 4 else "low"

    if ema.startswith("bullish") and rsi in ("bullish", "overbought") and bb != "lower expansion":
        scenario = "bullish continuation"
    elif ema.startswith("bearish") and rsi in ("bearish", "oversold") and bb != "upper expansion":
        scenario = "bearish continuation"
    elif bb == "squeeze":
        scenario = "breakout forming"
    elif "expansion" in bb and row["relative_volume"] < 1:
        scenario = "mean-reversion risk"
    else:
        scenario = "compression / mixed"

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "price": float(row["close"]),
        "ema_state": ema,
        "rsi": float(row["rsi"]),
        "rsi_state": rsi,
        "bollinger_state": bb,
        "relative_volume": float(row["relative_volume"]) if not pd.isna(row["relative_volume"]) else np.nan,
        "atr": float(row["atr"]),
        "market_structure": structure,
        "nearest_fib": nearest_fib_name,
        "nearest_fib_value": nearest_fib_value,
        "fib_distance_pct": nearest_fib_distance * 100,
        "elliott_state": elliott["state"],
        "elliott_confidence": elliott["confidence"],
        "elliott_detail": elliott["detail"],
        "macd_state": macd,
        "macd_hist": float(row["macd_hist"]) if "macd_hist" in df and not pd.isna(row["macd_hist"]) else np.nan,
        "adx": adx_value,
        "adx_state": adx_label,
        "vwap_distance_pct": vwap_distance,
        "vwap_state": vwap_label,
        "volatility_regime": regime,
        "scenario": scenario,
        "confidence": confidence,
        "alerts": alerts,
        "fib": fib,
        "pivots": pivots,
    }


def btc_risk_from_analysis(analysis: dict[str, Any]) -> str:
    if analysis["ema_state"].startswith("bearish") or analysis["rsi"] < 45:
        return "bearish"
    if analysis["ema_state"].startswith("bullish") and analysis["rsi"] > 50:
        return "bullish"
    return "neutral"


def summarize_alert_counts(alerts: list[Alert]) -> dict[str, int]:
    return {
        "Watch": sum(1 for alert in alerts if alert.severity == "Watch"),
        "Signal": sum(1 for alert in alerts if alert.severity == "Signal"),
        "Risk": sum(1 for alert in alerts if alert.severity == "Risk"),
    }
