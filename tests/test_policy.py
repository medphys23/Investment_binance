import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src import ml
from src.config import (
    PAPER_POLICY_BASE_THRESHOLD,
    PAPER_POLICY_MAX_SIZE_MULTIPLIER,
    PAPER_POLICY_MAX_THRESHOLD,
    PAPER_POLICY_MIN_SIZE_MULTIPLIER,
    PAPER_POLICY_MIN_THRESHOLD,
    PAPER_RISK_PER_TRADE,
)
from src.ml import get_active_policy, should_update_policy, update_policy
from src.paper_trading import insert_signal_feature, open_eligible_paper_trade
from src.risk import position_size
from src.storage import connect, get_state, initialize_database, latest_policy_state
from src.strategy import PaperTradeCandidate


def _analysis(symbol: str = "BTCUSDC", rsi: float = 55.0) -> dict:
    return {
        "symbol": symbol,
        "timeframe": "12h",
        "price": 100.0,
        "ema_state": "bullish stack",
        "rsi": rsi,
        "rsi_state": "bullish",
        "bollinger_state": "normal",
        "relative_volume": 1.2,
        "atr": 2.0,
        "market_structure": "higher highs",
        "nearest_fib": "0.618",
        "fib_distance_pct": 0.5,
        "elliott_state": "no clear wave",
        "elliott_confidence": "low",
        "scenario": "bullish continuation",
        "confidence": "medium",
        "macd_hist": 0.5,
        "adx": 27.0,
        "vwap_distance_pct": 0.4,
        "volatility_regime": "trending-expansion",
    }


def _insert_closed_trade(conn, idx: int, win: bool, side: str = "long") -> None:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    captured = (base + pd.Timedelta(hours=idx)).isoformat()
    rsi = 60.0 if win else 40.0
    feature_id = conn.execute(
        """
        INSERT INTO signal_features
        (captured_at, symbol, timeframe, price, ema_state, rsi, rsi_state, bollinger_state,
         relative_volume, atr, market_structure, nearest_fib, fib_distance_pct, elliott_state,
         elliott_confidence, scenario, confidence, side, macd_hist, adx, vwap_distance_pct, volatility_regime)
        VALUES (?, 'BTCUSDC', '12h', 100, ?, ?, 'bullish', 'normal', 1.2, 2.0, 'higher highs', '0.618',
                0.5, 'no clear wave', 'low', 'bullish continuation', 'medium', ?, ?, ?, 0.4, 'normal')
        """,
        ("bullish stack" if win else "bearish stack", rsi, "bullish" if win else "bearish", side, 0.5 if win else -0.5, 27.0),
    ).lastrowid
    realized = 5.0 if win else -5.0
    pnl = 50.0 if win else -50.0
    conn.execute(
        """
        INSERT INTO paper_trades
        (trade_id, feature_id, symbol, side, status, opened_at, closed_at, entry, last_price, invalidation,
         target_1, target_2, quantity, notional, risk_amount, leverage, margin, unrealized_return_pct,
         realized_return_pct, unrealized_pnl, realized_pnl, confidence, score)
        VALUES (?, ?, 'BTCUSDC', ?, 'closed', ?, ?, 100, 100, 95, 110, 120, 1, 100, 10, 1.0, 100, 0,
                ?, 0, ?, 'medium', 4)
        """,
        (f"T-{idx}", feature_id, side, captured, captured, realized, pnl),
    )
    conn.commit()


class PositionSizingTest(unittest.TestCase):
    def test_size_multiplier_scales_within_risk_ceiling(self) -> None:
        _, _, full_risk = position_size(100.0, 95.0, 10_000.0, side="long", size_multiplier=1.0)
        _, _, half_risk = position_size(100.0, 95.0, 10_000.0, side="long", size_multiplier=0.5)
        self.assertAlmostEqual(full_risk, 10_000.0 * PAPER_RISK_PER_TRADE)
        self.assertAlmostEqual(half_risk, full_risk * 0.5)

    def test_size_multiplier_clamped_to_ceiling(self) -> None:
        _, _, full_risk = position_size(100.0, 95.0, 10_000.0, side="long", size_multiplier=1.0)
        _, _, over_risk = position_size(100.0, 95.0, 10_000.0, side="long", size_multiplier=2.0)
        self.assertAlmostEqual(over_risk, full_risk)


class PolicyCadenceTest(unittest.TestCase):
    def test_cold_start_is_pass_through(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "bot.sqlite")
            initialize_database(conn)
            self.assertFalse(should_update_policy(conn))
            result = update_policy(conn)
            self.assertEqual(result["status"], "cold_start")
            policy = get_active_policy(conn)
            self.assertEqual(policy["entry_prob_threshold"], PAPER_POLICY_BASE_THRESHOLD)
            self.assertEqual(policy["size_multiplier"], 1.0)
            conn.close()

    def test_policy_updates_after_enough_trades(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "bot.sqlite")
            initialize_database(conn)
            for idx in range(60):
                _insert_closed_trade(conn, idx, win=(idx % 2 == 0), side="long" if idx % 3 else "short")
            self.assertTrue(should_update_policy(conn))
            model_path = Path(tmp) / "model.pkl"
            with mock.patch.object(ml, "PAPER_POLICY_MODEL_PATH", str(model_path)):
                result = update_policy(conn, pd.Timestamp("2026-02-01T00:00:00Z"))
            self.assertIn(result["status"], {"trained", "skipped"})
            state = latest_policy_state(conn)
            self.assertIsNotNone(state)
            policy = get_active_policy(conn)
            self.assertGreaterEqual(policy["entry_prob_threshold"], PAPER_POLICY_MIN_THRESHOLD)
            self.assertLessEqual(policy["entry_prob_threshold"], PAPER_POLICY_MAX_THRESHOLD)
            self.assertGreaterEqual(policy["size_multiplier"], PAPER_POLICY_MIN_SIZE_MULTIPLIER)
            self.assertLessEqual(policy["size_multiplier"], PAPER_POLICY_MAX_SIZE_MULTIPLIER)
            if result["status"] == "trained":
                self.assertTrue(model_path.exists())
            conn.close()


class EntryGatingTest(unittest.TestCase):
    def _setup(self, tmp: str):
        conn = connect(Path(tmp) / "bot.sqlite")
        initialize_database(conn)
        feature_id = insert_signal_feature(conn, pd.Timestamp("2026-01-01T00:00:00Z"), _analysis(), "long")
        conn.commit()
        candidate = PaperTradeCandidate(
            symbol="BTCUSDC",
            action="paper_long",
            confidence="medium",
            entry=100.0,
            invalidation=95.0,
            target_1=110.0,
            target_2=120.0,
            score=4,
            reasons=[],
            blockers=[],
        )
        return conn, feature_id, candidate

    def test_cold_start_opens_without_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            conn, feature_id, candidate = self._setup(tmp)
            with mock.patch.object(ml, "PAPER_POLICY_MODEL_PATH", str(Path(tmp) / "absent.pkl")):
                opened = open_eligible_paper_trade(conn, candidate, feature_id, 100.0, pd.Timestamp("2026-01-01T01:00:00Z"))
            conn.close()
            self.assertTrue(opened)

    def test_low_probability_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            conn, feature_id, candidate = self._setup(tmp)
            with mock.patch.object(ml, "predict_win_probability", return_value=0.1), mock.patch.object(
                ml, "get_active_policy", return_value={"entry_prob_threshold": 0.6, "size_multiplier": 1.0, "policy_version": 1}
            ):
                opened = open_eligible_paper_trade(conn, candidate, feature_id, 100.0, pd.Timestamp("2026-01-01T01:00:00Z"))
            conn.close()
            self.assertFalse(opened)

    def test_high_probability_opens_and_stores_p_win(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            conn, feature_id, candidate = self._setup(tmp)
            with mock.patch.object(ml, "predict_win_probability", return_value=0.9), mock.patch.object(
                ml, "get_active_policy", return_value={"entry_prob_threshold": 0.5, "size_multiplier": 0.5, "policy_version": 2}
            ):
                opened = open_eligible_paper_trade(conn, candidate, feature_id, 100.0, pd.Timestamp("2026-01-01T01:00:00Z"))
            row = conn.execute("SELECT p_win, policy_version FROM paper_trades WHERE status='open'").fetchone()
            p_win = float(row["p_win"])
            version = int(row["policy_version"])
            conn.close()
            self.assertTrue(opened)
            self.assertAlmostEqual(p_win, 0.9)
            self.assertEqual(version, 2)


class FeaturePersistenceTest(unittest.TestCase):
    def test_new_feature_columns_persist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "bot.sqlite")
            initialize_database(conn)
            derivatives = {
                "funding_rate": 0.0001,
                "futures_spot_ratio": 2.5,
                "taker_buy_ratio": 1.1,
                "long_short_ratio": 1.3,
            }
            feature_id = insert_signal_feature(
                conn, pd.Timestamp("2026-01-01T00:00:00Z"), _analysis(), "long", derivatives, btc_corr=0.85
            )
            conn.commit()
            row = conn.execute("SELECT * FROM signal_features WHERE id = ?", (feature_id,)).fetchone()
            self.assertAlmostEqual(float(row["macd_hist"]), 0.5)
            self.assertAlmostEqual(float(row["adx"]), 27.0)
            self.assertAlmostEqual(float(row["funding_rate"]), 0.0001)
            self.assertAlmostEqual(float(row["futures_spot_ratio"]), 2.5)
            self.assertAlmostEqual(float(row["btc_corr"]), 0.85)
            self.assertEqual(row["volatility_regime"], "trending-expansion")
            conn.close()


if __name__ == "__main__":
    unittest.main()
