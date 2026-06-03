"""Technical indicator calculations for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    ADX_PERIOD,
    ATR_PERIOD,
    BB_PERIOD,
    BB_STD,
    EMA_PERIODS,
    FIB_LOOKBACK,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    REL_VOLUME_PERIOD,
    RSI_PERIOD,
    VOLATILITY_RANK_WINDOW,
    VWAP_PERIOD,
)


@dataclass(frozen=True)
class FibLevels:
    direction: str
    swing_high: float
    swing_low: float
    levels: dict[str, float]
    extensions: dict[str, float]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]

    for period in EMA_PERIODS:
        out[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

    bb_mid = close.rolling(BB_PERIOD).mean()
    bb_std = close.rolling(BB_PERIOD).std(ddof=0)
    out["bb_mid"] = bb_mid
    out["bb_upper"] = bb_mid + BB_STD * bb_std
    out["bb_lower"] = bb_mid - BB_STD * bb_std
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / bb_mid.replace(0, np.nan)

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))
    out["rsi"] = out["rsi"].fillna(50)

    macd_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    macd_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    out["macd"] = macd_fast - macd_slow
    out["macd_signal"] = out["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    out["atr"] = true_range.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()
    out["range_pct"] = (high - low) / close.replace(0, np.nan)
    out["atr_percentile"] = out["atr"].rolling(VOLATILITY_RANK_WINDOW).rank(pct=True)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_for_dm = true_range.ewm(alpha=1 / ADX_PERIOD, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=out.index).ewm(alpha=1 / ADX_PERIOD, adjust=False).mean() / atr_for_dm
    minus_di = 100 * pd.Series(minus_dm, index=out.index).ewm(alpha=1 / ADX_PERIOD, adjust=False).mean() / atr_for_dm
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    out["plus_di"] = plus_di
    out["minus_di"] = minus_di
    out["adx"] = dx.ewm(alpha=1 / ADX_PERIOD, adjust=False).mean()

    typical_price = (high + low + close) / 3
    tp_volume = typical_price * out["volume"]
    rolling_volume = out["volume"].rolling(VWAP_PERIOD).sum().replace(0, np.nan)
    out["vwap"] = tp_volume.rolling(VWAP_PERIOD).sum() / rolling_volume
    out["vwap_distance_pct"] = (close - out["vwap"]) / close.replace(0, np.nan) * 100

    out["volume_sma"] = out["volume"].rolling(REL_VOLUME_PERIOD).mean()
    out["relative_volume"] = out["volume"] / out["volume_sma"].replace(0, np.nan)
    return out


def fibonacci_levels(df: pd.DataFrame, lookback: int = FIB_LOOKBACK) -> FibLevels:
    recent = df.tail(min(lookback, len(df)))
    high_idx = recent["high"].idxmax()
    low_idx = recent["low"].idxmin()
    swing_high = float(recent.loc[high_idx, "high"])
    swing_low = float(recent.loc[low_idx, "low"])
    price_range = swing_high - swing_low
    direction = "bullish" if low_idx < high_idx else "bearish"

    ratios = {
        "0.236": 0.236,
        "0.382": 0.382,
        "0.500": 0.5,
        "0.618": 0.618,
        "0.786": 0.786,
    }
    if direction == "bullish":
        levels = {name: swing_high - ratio * price_range for name, ratio in ratios.items()}
        extensions = {"1.272": swing_high + 0.272 * price_range, "1.618": swing_high + 0.618 * price_range}
    else:
        levels = {name: swing_low + ratio * price_range for name, ratio in ratios.items()}
        extensions = {"1.272": swing_low - 0.272 * price_range, "1.618": swing_low - 0.618 * price_range}
    return FibLevels(direction=direction, swing_high=swing_high, swing_low=swing_low, levels=levels, extensions=extensions)


def find_pivots(df: pd.DataFrame, window: int = 4, min_move_atr: float = 0.6) -> list[dict[str, float | str | pd.Timestamp]]:
    if len(df) < window * 2 + 1:
        return []

    atr = float(df["atr"].tail(50).median()) if "atr" in df else 0.0
    threshold = max(atr * min_move_atr, float(df["close"].iloc[-1]) * 0.001)
    pivots: list[dict[str, float | str | pd.Timestamp]] = []

    for idx in range(window, len(df) - window):
        neighborhood = df.iloc[idx - window : idx + window + 1]
        row = df.iloc[idx]
        if row["high"] == neighborhood["high"].max():
            pivots.append({"kind": "high", "price": float(row["high"]), "time": row["open_time"], "index": idx})
        elif row["low"] == neighborhood["low"].min():
            pivots.append({"kind": "low", "price": float(row["low"]), "time": row["open_time"], "index": idx})

    alternating: list[dict[str, float | str | pd.Timestamp]] = []
    for pivot in pivots:
        if not alternating:
            alternating.append(pivot)
            continue
        last = alternating[-1]
        if pivot["kind"] == last["kind"]:
            if pivot["kind"] == "high" and float(pivot["price"]) > float(last["price"]):
                alternating[-1] = pivot
            elif pivot["kind"] == "low" and float(pivot["price"]) < float(last["price"]):
                alternating[-1] = pivot
            continue
        if abs(float(pivot["price"]) - float(last["price"])) >= threshold:
            alternating.append(pivot)
    return alternating[-9:]


def market_structure(pivots: list[dict[str, float | str | pd.Timestamp]]) -> str:
    highs = [p for p in pivots if p["kind"] == "high"]
    lows = [p for p in pivots if p["kind"] == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return "forming"
    higher_highs = float(highs[-1]["price"]) > float(highs[-2]["price"])
    higher_lows = float(lows[-1]["price"]) > float(lows[-2]["price"])
    lower_highs = float(highs[-1]["price"]) < float(highs[-2]["price"])
    lower_lows = float(lows[-1]["price"]) < float(lows[-2]["price"])
    if higher_highs and higher_lows:
        return "higher-high / higher-low"
    if lower_highs and lower_lows:
        return "lower-high / lower-low"
    return "mixed"
