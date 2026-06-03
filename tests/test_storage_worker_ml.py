import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.config import KEY_TIMEFRAMES, SYMBOLS
from src.indicators import fibonacci_levels
from src.ml import train_batch_model
from src.paper_worker import run_cycle
from src.risk import position_size, update_equity_and_leverage_state
from src.storage import connect, get_state, initialize_database, set_state
from src.strategy import generate_paper_trade_candidate


def frame(up: bool = True) -> pd.DataFrame:
    rows = []
    for idx in range(260):
        drift = idx * 0.2 if up else -idx * 0.2
        close = 100 + drift
        rows.append(
            {
                "open_time": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=idx),
                "open": close - 0.4,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1000 + idx,
                "quote_volume": (1000 + idx) * close,
                "trade_count": 100 + idx,
                "taker_buy_base_volume": 500,
                "taker_buy_quote_volume": 500 * close,
            }
        )
    return pd.DataFrame(rows)


class FakeClient:
    def spot_24h(self, symbol: str) -> dict[str, str]:
        return {
            "symbol": symbol,
            "lastPrice": "150",
            "priceChangePercent": "1.5",
            "volume": "1000",
            "quoteVolume": "150000",
        }

    def book_ticker(self, symbol: str) -> dict[str, str]:
        return {"bidPrice": "149.9", "askPrice": "150.1"}

    def spot_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        return frame(up=True).tail(limit)

    def futures_24h(self, symbol: str) -> dict[str, str]:
        return {"quoteVolume": "300000"}

    def open_interest(self, symbol: str) -> dict[str, str]:
        return {"openInterest": "1000"}

    def mark_price(self, symbol: str) -> dict[str, str]:
        return {"lastFundingRate": "0.0001", "markPrice": "150"}

    def taker_buy_sell_volume(self, symbol: str, period: str, limit: int = 30) -> list[dict[str, str]]:
        return [{"buySellRatio": "1.1"}]

    def long_short_ratio(self, symbol: str, period: str, limit: int = 30) -> list[dict[str, str]]:
        return [{"longShortRatio": "1.2"}]


class StorageWorkerMlTest(unittest.TestCase):
    def test_sqlite_schema_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "bot.sqlite")
            initialize_database(conn)
            tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertIn("market_snapshots", tables)
            self.assertIn("signal_features", tables)
            self.assertIn("paper_trades", tables)
            self.assertIn("model_runs", tables)
            self.assertIn("bot_state", tables)
            conn.close()

    def test_worker_one_cycle_updates_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "bot.sqlite")
            result = run_cycle(db_path=db_path, client=FakeClient(), train_model=False)
            conn = connect(db_path)
            self.assertEqual(result["snapshots"], len(SYMBOLS))
            self.assertGreater(result["features"], 0)
            self.assertEqual(get_state(conn, "worker_status"), "ok")
            self.assertTrue(get_state(conn, "last_heartbeat"))
            conn.close()

    def test_bearish_candidate_becomes_paper_short(self) -> None:
        fib = fibonacci_levels(frame(up=False))
        analyses = {
            timeframe: {
                "ema_state": "bearish stack",
                "rsi": 35.0,
                "scenario": "bearish continuation",
                "confidence": "high",
                "relative_volume": 1.3,
                "bollinger_state": "lower expansion",
                "price": 100.0,
                "atr": 2.0,
                "fib": fib,
            }
            for timeframe in KEY_TIMEFRAMES
        }
        candidate = generate_paper_trade_candidate("BTCUSDC", analyses)
        self.assertEqual(candidate.action, "paper_short")
        self.assertGreaterEqual(candidate.score, 3)
        self.assertIsNotNone(candidate.invalidation)
        self.assertGreater(candidate.invalidation, candidate.entry)

    def test_risk_sizing_uses_equity_and_invalidation_distance(self) -> None:
        quantity, notional, risk_amount = position_size(100.0, 95.0, 10_000.0)
        self.assertEqual(risk_amount, 100.0)
        self.assertEqual(quantity, 20.0)
        self.assertEqual(notional, 2000.0)

    def test_risk_sizing_for_short_uses_stop_above_entry(self) -> None:
        quantity, notional, risk_amount = position_size(100.0, 105.0, 10_000.0, side="short")
        self.assertEqual(risk_amount, 100.0)
        self.assertEqual(quantity, 20.0)
        self.assertEqual(notional, 2000.0)

    def test_ml_skips_and_then_trains_with_enough_closed_trades(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "bot.sqlite")
            initialize_database(conn)
            skipped = train_batch_model(conn, min_closed=3)
            self.assertEqual(skipped["status"], "skipped")
            for idx in range(8):
                feature_id = conn.execute(
                    """
                    INSERT INTO signal_features
                    (captured_at, symbol, timeframe, price, rsi, relative_volume, atr, fib_distance_pct, confidence, side)
                    VALUES (?, 'BTCUSDC', '12h', 100, ?, 1.2, 2.0, 1.0, 'medium', 'bullish')
                    """,
                    (f"2026-01-01T0{idx}:00:00Z", 45 + idx),
                ).lastrowid
                conn.execute(
                    """
                    INSERT INTO paper_trades
                    (trade_id, feature_id, symbol, side, status, opened_at, closed_at, entry, last_price,
                     quantity, notional, risk_amount, leverage, margin, unrealized_return_pct, realized_return_pct,
                     unrealized_pnl, realized_pnl, confidence, score)
                    VALUES (?, ?, 'BTCUSDC', 'long', 'closed', ?, ?, 100, 100, 1, 100, 10, 1, 100, 0, ?, 0, ?, 'medium', ?)
                    """,
                    (f"t{idx}", feature_id, "2026-01-01", "2026-01-02", 1 if idx % 2 else -1, 1 if idx % 2 else -1, idx),
                )
            conn.commit()
            trained = train_batch_model(conn, min_closed=3)
            self.assertEqual(trained["status"], "trained")
            self.assertIn("accuracy", trained)
            conn.close()

    def test_simulated_leverage_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "bot.sqlite")
            initialize_database(conn)
            now = pd.Timestamp("2026-01-01T00:00:00Z")
            set_state(conn, "simulated_leverage", 10.0, now)
            for idx in range(30):
                conn.execute(
                    """
                    INSERT INTO paper_trades
                    (trade_id, symbol, side, status, opened_at, closed_at, entry, last_price,
                     quantity, notional, risk_amount, leverage, margin, unrealized_return_pct, realized_return_pct,
                     unrealized_pnl, realized_pnl, confidence, score, close_reason)
                    VALUES (?, 'BTCUSDC', 'long', 'closed', ?, ?, 100, 110, 1, 100, 10, 1, 100, 10, 10, 10, 10, 'high', 5, 'target hit')
                    """,
                    (f"lev{idx}", "2026-01-01", "2026-01-02"),
                )
            conn.commit()
            state = update_equity_and_leverage_state(conn, now)
            self.assertEqual(state["simulated_leverage"], 10.0)
            conn.close()


if __name__ == "__main__":
    unittest.main()
