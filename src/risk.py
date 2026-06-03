"""Risk and simulated leverage rules for local paper trading."""

from __future__ import annotations

import sqlite3

import pandas as pd

from .config import (
    PAPER_MAX_OPEN_TRADES,
    PAPER_RISK_PER_TRADE,
    PAPER_STARTING_EQUITY,
    SIMULATED_MAX_LEVERAGE,
)
from .storage import get_state, set_state


def current_paper_equity(conn: sqlite3.Connection) -> float:
    realized = conn.execute(
        "SELECT COALESCE(SUM(realized_pnl), 0) AS total FROM paper_trades WHERE status = 'closed'"
    ).fetchone()["total"]
    return round(PAPER_STARTING_EQUITY + float(realized), 4)


def position_size(entry: float, invalidation: float | None, equity: float) -> tuple[float, float, float]:
    risk_amount = equity * PAPER_RISK_PER_TRADE
    if invalidation is None or invalidation >= entry:
        stop_distance = max(entry * 0.02, 0.00000001)
    else:
        stop_distance = max(entry - invalidation, 0.00000001)
    quantity = risk_amount / stop_distance
    max_spot_quantity = equity / entry
    quantity = min(quantity, max_spot_quantity)
    notional = quantity * entry
    return round(quantity, 8), round(notional, 4), round(risk_amount, 4)


def can_open_trade(conn: sqlite3.Connection, symbol: str) -> bool:
    active_count = conn.execute("SELECT COUNT(*) AS count FROM paper_trades WHERE status = 'open'").fetchone()["count"]
    active_symbol = conn.execute(
        "SELECT COUNT(*) AS count FROM paper_trades WHERE status = 'open' AND symbol = ?",
        (symbol,),
    ).fetchone()["count"]
    return int(active_count) < PAPER_MAX_OPEN_TRADES and int(active_symbol) == 0


def update_equity_and_leverage_state(conn: sqlite3.Connection, now: pd.Timestamp) -> dict[str, float]:
    equity = current_paper_equity(conn)
    peak = max(float(get_state(conn, "paper_peak_equity", PAPER_STARTING_EQUITY)), equity)
    drawdown = 0.0 if peak == 0 else max((peak - equity) / peak * 100, 0.0)
    leverage = _simulated_leverage(conn, drawdown)
    set_state(conn, "paper_equity", equity, now)
    set_state(conn, "paper_peak_equity", peak, now)
    set_state(conn, "drawdown_pct", round(drawdown, 4), now)
    set_state(conn, "simulated_leverage", leverage, now)
    return {"paper_equity": equity, "paper_peak_equity": peak, "drawdown_pct": drawdown, "simulated_leverage": leverage}


def _simulated_leverage(conn: sqlite3.Connection, drawdown_pct: float) -> float:
    current = float(get_state(conn, "simulated_leverage", 1.0))
    recent = pd.read_sql_query(
        """
        SELECT realized_return_pct, realized_pnl, close_reason
        FROM paper_trades
        WHERE status = 'closed'
        ORDER BY closed_at DESC
        LIMIT 30
        """,
        conn,
    )
    if len(recent) < 30:
        return min(max(current, 1.0), SIMULATED_MAX_LEVERAGE)
    invalidations = recent.head(3)["close_reason"].eq("invalidation hit").sum()
    if drawdown_pct > 8 or invalidations == 3:
        return max(1.0, current - 1.0)
    win_rate = (recent["realized_return_pct"] > 0).mean() * 100
    wins = recent[recent["realized_pnl"] > 0]["realized_pnl"].sum()
    losses = abs(recent[recent["realized_pnl"] < 0]["realized_pnl"].sum())
    profit_factor = wins / losses if losses else float("inf")
    if win_rate >= 55 and profit_factor >= 1.3 and drawdown_pct <= 5:
        return min(SIMULATED_MAX_LEVERAGE, current + 1.0)
    return min(max(current, 1.0), SIMULATED_MAX_LEVERAGE)
