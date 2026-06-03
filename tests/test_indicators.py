import unittest

import pandas as pd

from src.indicators import add_indicators, fibonacci_levels, find_pivots
from src.signals import analyze_timeframe


def sample_frame() -> pd.DataFrame:
    values = []
    base = 100.0
    for idx in range(260):
        drift = idx * 0.2
        wave = ((idx % 20) - 10) * 0.15
        close = base + drift + wave
        open_ = close - 0.4
        high = close + 1.0
        low = close - 1.0
        values.append(
            {
                "open_time": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=idx),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000 + idx * 2,
                "quote_volume": (1000 + idx * 2) * close,
                "trade_count": 100 + idx,
                "taker_buy_base_volume": 500,
                "taker_buy_quote_volume": 500 * close,
            }
        )
    return pd.DataFrame(values)


class IndicatorTest(unittest.TestCase):
    def test_add_indicators_produces_core_columns(self) -> None:
        df = add_indicators(sample_frame())
        for column in ["ema_9", "ema_200", "bb_upper", "bb_lower", "rsi", "atr", "relative_volume"]:
            self.assertIn(column, df.columns)
        self.assertGreater(df["rsi"].iloc[-1], 0)

    def test_fibonacci_levels_and_signal_analysis(self) -> None:
        df = add_indicators(sample_frame())
        fib = fibonacci_levels(df)
        self.assertIn("0.618", fib.levels)
        analysis = analyze_timeframe("TESTUSDT", "6h", df)
        self.assertEqual(analysis["symbol"], "TESTUSDT")
        self.assertIn(analysis["confidence"], {"low", "medium", "high"})
        self.assertIsInstance(analysis["alerts"], list)

    def test_find_pivots_returns_list(self) -> None:
        df = add_indicators(sample_frame())
        self.assertIsInstance(find_pivots(df), list)


if __name__ == "__main__":
    unittest.main()
