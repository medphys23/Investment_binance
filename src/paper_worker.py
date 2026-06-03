"""Always-on local paper bot worker.

This worker uses public spot market data only. It never connects to a Binance
account and never places real orders.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from .analysis_engine import build_dashboard_snapshot, gather_cycle_data, trend_side
from .binance_client import BinanceClient
from .config import PAPER_MAX_HOLD_HOURS, PAPER_WORKER_INTERVAL_SECONDS, SYMBOLS
from .ml import maybe_update_policy, predict_win_probability
from .paper_trading import (
    insert_market_snapshot,
    insert_signal_feature,
    open_eligible_paper_trade,
    update_open_paper_trades_db,
)
from .risk import update_equity_and_leverage_state
from .storage import connect, initialize_database, set_state
from .strategy import candidate_to_row, generate_paper_trade_candidate


def run_cycle(db_path: str | None = None, client: BinanceClient | None = None, train_model: bool = True) -> dict[str, int]:
    client = client or BinanceClient()
    conn = connect(db_path) if db_path else connect()
    initialize_database(conn)
    now = pd.Timestamp.now(tz="UTC")
    opened = 0
    features = 0
    snapshots = 0

    try:
        data = gather_cycle_data(client)

        latest_prices: dict[str, float] = {}
        for symbol, bundle in data.items():
            snapshot = bundle["snapshot"]
            if snapshot["price"] is None:
                continue
            latest_prices[symbol] = snapshot["price"]
            insert_market_snapshot(conn, now, snapshot)
            snapshots += 1
        conn.commit()

        update_open_paper_trades_db(conn, latest_prices, now, PAPER_MAX_HOLD_HOURS)
        conn.commit()

        candidate_rows = []
        for symbol, bundle in data.items():
            analyses = bundle["analyses"]
            derivatives = bundle["derivatives"]
            btc_corr = bundle["btc_corr"]
            feature_ids: dict[str, int] = {}
            for timeframe, analysis in analyses.items():
                feature_ids[timeframe] = insert_signal_feature(
                    conn, now, analysis, trend_side(analysis), derivatives, btc_corr
                )
                features += 1
            candidate = generate_paper_trade_candidate(symbol, analyses)
            primary_feature_id = feature_ids.get("12h")
            row = candidate_to_row(candidate)
            row["p_win"] = predict_win_probability(conn, primary_feature_id, candidate.score, candidate.confidence)
            candidate_rows.append(row)
            latest_price = latest_prices.get(symbol, candidate.entry)
            if open_eligible_paper_trade(conn, candidate, primary_feature_id, latest_price, now):
                opened += 1
            conn.commit()

        snapshot_payload = build_dashboard_snapshot(now, data, candidate_rows)
        set_state(conn, "dashboard_snapshot", snapshot_payload, now)
        conn.commit()

        state = update_equity_and_leverage_state(conn, now)
        conn.commit()
        if train_model:
            maybe_update_policy(conn, now)
        set_state(conn, "last_heartbeat", now.isoformat(), now)
        set_state(conn, "worker_status", "ok", now)
        set_state(conn, "last_cycle_error", "", now)
        conn.commit()
        return {"snapshots": snapshots, "features": features, "opened": opened, "paper_equity": int(state["paper_equity"])}
    except Exception as exc:
        set_state(conn, "worker_status", "error", now)
        set_state(conn, "last_cycle_error", str(exc), now)
        conn.commit()
        raise
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Binance paper bot worker.")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit.")
    parser.add_argument("--interval", type=int, default=PAPER_WORKER_INTERVAL_SECONDS, help="Polling interval in seconds.")
    args = parser.parse_args()
    while True:
        run_cycle()
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
