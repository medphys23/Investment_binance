import unittest
from pathlib import Path

import pandas as pd

from src.paper_trading import load_paper_trades, update_and_open_paper_trades
from src.strategy import PaperTradeCandidate


class PaperTradingTest(unittest.TestCase):
    def test_opens_one_active_paper_trade_per_symbol(self) -> None:
        candidate = PaperTradeCandidate(
            symbol="BTCUSDC",
            action="paper_long",
            confidence="medium",
            entry=100.0,
            invalidation=95.0,
            target_1=110.0,
            target_2=120.0,
            score=4,
            reasons=["test"],
            blockers=[],
        )
        trades = update_and_open_paper_trades(
            load_paper_trades(path=Path("missing.csv")),
            [candidate, candidate],
            {"BTCUSDC": 101.0},
            pd.Timestamp("2026-01-01T00:00:00Z"),
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["status"], "open")
        self.assertEqual(trades.iloc[0]["side"], "long")

    def test_closes_long_when_target_hit(self) -> None:
        candidate = PaperTradeCandidate(
            symbol="BTCUSDC",
            action="paper_long",
            confidence="high",
            entry=100.0,
            invalidation=95.0,
            target_1=105.0,
            target_2=110.0,
            score=6,
            reasons=[],
            blockers=[],
        )
        opened = update_and_open_paper_trades(
            pd.DataFrame(),
            [candidate],
            {"BTCUSDC": 100.0},
            pd.Timestamp("2026-01-01T00:00:00Z"),
        )
        closed = update_and_open_paper_trades(
            opened,
            [],
            {"BTCUSDC": 106.0},
            pd.Timestamp("2026-01-01T01:00:00Z"),
        )
        self.assertEqual(closed.iloc[0]["status"], "closed")
        self.assertEqual(closed.iloc[0]["close_reason"], "target hit")
        self.assertGreater(float(closed.iloc[0]["realized_return_pct"]), 0)


if __name__ == "__main__":
    unittest.main()
