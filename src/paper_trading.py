"""Paper-trade tracking based on live market data, without exchange execution."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd

from .risk import can_open_trade, current_paper_equity, position_size
from .strategy import PaperTradeCandidate


PAPER_TRADES_PATH = Path("data/paper_trades.csv")
PAPER_TRADE_COLUMNS = [
    "trade_id",
    "symbol",
    "side",
    "status",
    "opened_at",
    "closed_at",
    "entry",
    "last_price",
    "invalidation",
    "target_1",
    "target_2",
    "unrealized_return_pct",
    "realized_return_pct",
    "confidence",
    "score",
    "close_reason",
    "reasons",
    "blockers",
]


def load_paper_trades(path: Path = PAPER_TRADES_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=PAPER_TRADE_COLUMNS)
    trades = pd.read_csv(path)
    for column in PAPER_TRADE_COLUMNS:
        if column not in trades.columns:
            trades[column] = pd.NA
    return trades[PAPER_TRADE_COLUMNS]


def save_paper_trades(trades: pd.DataFrame, path: Path = PAPER_TRADES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trades[PAPER_TRADE_COLUMNS].to_csv(path, index=False)


def update_and_open_paper_trades(
    trades: pd.DataFrame,
    candidates: list[PaperTradeCandidate],
    latest_prices: dict[str, float],
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    now = now or pd.Timestamp.now(tz="UTC")
    updated = trades.copy()
    if updated.empty:
        updated = pd.DataFrame(columns=PAPER_TRADE_COLUMNS)

    updated = _mark_active_trades(updated, latest_prices, now)
    for candidate in candidates:
        if candidate.action != "paper_long":
            continue
        if candidate.confidence not in {"medium", "high"}:
            continue
        if _has_active_trade(updated, candidate.symbol):
            continue
        new_row = pd.DataFrame([_open_trade_row(candidate, latest_prices, now)])
        updated = new_row if updated.empty else pd.concat([updated, new_row], ignore_index=True)
    return updated[PAPER_TRADE_COLUMNS]


def insert_market_snapshot(conn: sqlite3.Connection, captured_at: pd.Timestamp, snapshot: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO market_snapshots
        (captured_at, symbol, price, price_change_pct, volume, quote_volume, bid, ask, spread_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            captured_at.isoformat(),
            snapshot["symbol"],
            snapshot["price"],
            snapshot.get("price_change_pct"),
            snapshot.get("volume"),
            snapshot.get("quote_volume"),
            snapshot.get("bid"),
            snapshot.get("ask"),
            snapshot.get("spread_pct"),
        ),
    )


def insert_signal_feature(
    conn: sqlite3.Connection,
    captured_at: pd.Timestamp,
    analysis: dict[str, Any],
    side: str,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO signal_features
        (captured_at, symbol, timeframe, price, ema_state, rsi, rsi_state, bollinger_state, relative_volume, atr,
         market_structure, nearest_fib, fib_distance_pct, elliott_state, elliott_confidence, scenario, confidence, side)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            captured_at.isoformat(),
            analysis["symbol"],
            analysis["timeframe"],
            analysis["price"],
            analysis["ema_state"],
            analysis["rsi"],
            analysis["rsi_state"],
            analysis["bollinger_state"],
            analysis["relative_volume"],
            analysis["atr"],
            analysis["market_structure"],
            analysis["nearest_fib"],
            analysis["fib_distance_pct"],
            analysis["elliott_state"],
            analysis["elliott_confidence"],
            analysis["scenario"],
            analysis["confidence"],
            side,
        ),
    )
    return int(cursor.lastrowid)


def open_eligible_paper_trade(
    conn: sqlite3.Connection,
    candidate: PaperTradeCandidate,
    feature_id: int | None,
    latest_price: float,
    now: pd.Timestamp,
) -> bool:
    if candidate.action != "paper_long" or candidate.confidence not in {"medium", "high"}:
        return False
    if not can_open_trade(conn, candidate.symbol):
        return False
    equity = current_paper_equity(conn)
    quantity, notional, risk_amount = position_size(latest_price, candidate.invalidation, equity)
    if quantity <= 0 or notional <= 0:
        return False
    conn.execute(
        """
        INSERT INTO paper_trades
        (trade_id, feature_id, symbol, side, status, opened_at, closed_at, entry, last_price, invalidation,
         target_1, target_2, quantity, notional, risk_amount, leverage, margin, unrealized_return_pct,
         realized_return_pct, unrealized_pnl, realized_pnl, confidence, score, close_reason, reasons, blockers)
        VALUES (?, ?, ?, 'long', 'open', ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, 1.0, ?, 0.0, NULL, 0.0, NULL, ?, ?, NULL, ?, ?)
        """,
        (
            f"{candidate.symbol}-{now.strftime('%Y%m%d%H%M%S')}",
            feature_id,
            candidate.symbol,
            now.isoformat(),
            latest_price,
            latest_price,
            candidate.invalidation,
            candidate.target_1,
            candidate.target_2,
            quantity,
            notional,
            risk_amount,
            notional,
            candidate.confidence,
            candidate.score,
            " | ".join(candidate.reasons),
            " | ".join(candidate.blockers),
        ),
    )
    return True


def update_open_paper_trades_db(
    conn: sqlite3.Connection,
    latest_prices: dict[str, float],
    now: pd.Timestamp,
    max_hold_hours: int,
) -> None:
    rows = conn.execute("SELECT * FROM paper_trades WHERE status = 'open'").fetchall()
    for row in rows:
        symbol = row["symbol"]
        if symbol not in latest_prices:
            continue
        price = float(latest_prices[symbol])
        entry = float(row["entry"])
        quantity = float(row["quantity"])
        return_pct = _return_pct("long", entry, price)
        unrealized_pnl = round((price - entry) * quantity, 4)
        close_reason = _close_reason("long", price, _optional_float(row["invalidation"]), _optional_float(row["target_1"]), _optional_float(row["target_2"]))
        opened_at = pd.Timestamp(row["opened_at"])
        if close_reason is None and now - opened_at >= pd.Timedelta(hours=max_hold_hours):
            close_reason = "timeout"
        if close_reason:
            conn.execute(
                """
                UPDATE paper_trades
                SET status = 'closed', closed_at = ?, last_price = ?, unrealized_return_pct = ?,
                    realized_return_pct = ?, unrealized_pnl = ?, realized_pnl = ?, close_reason = ?
                WHERE trade_id = ?
                """,
                (now.isoformat(), price, return_pct, return_pct, unrealized_pnl, unrealized_pnl, close_reason, row["trade_id"]),
            )
        else:
            conn.execute(
                "UPDATE paper_trades SET last_price = ?, unrealized_return_pct = ?, unrealized_pnl = ? WHERE trade_id = ?",
                (price, return_pct, unrealized_pnl, row["trade_id"]),
            )


def _mark_active_trades(trades: pd.DataFrame, latest_prices: dict[str, float], now: pd.Timestamp) -> pd.DataFrame:
    for idx, row in trades[trades["status"].eq("open")].iterrows():
        symbol = str(row["symbol"])
        if symbol not in latest_prices:
            continue
        price = float(latest_prices[symbol])
        side = str(row["side"])
        entry = float(row["entry"])
        invalidation = _optional_float(row["invalidation"])
        target_1 = _optional_float(row["target_1"])
        target_2 = _optional_float(row["target_2"])
        unrealized = _return_pct(side, entry, price)
        close_reason = _close_reason(side, price, invalidation, target_1, target_2)

        trades.at[idx, "last_price"] = price
        trades.at[idx, "unrealized_return_pct"] = unrealized
        if close_reason:
            trades.at[idx, "status"] = "closed"
            trades.at[idx, "closed_at"] = now.isoformat()
            trades.at[idx, "realized_return_pct"] = unrealized
            trades.at[idx, "close_reason"] = close_reason
    return trades


def _open_trade_row(candidate: PaperTradeCandidate, latest_prices: dict[str, float], now: pd.Timestamp) -> dict[str, Any]:
    side = "long" if candidate.action == "paper_long" else "short"
    entry = latest_prices.get(candidate.symbol, candidate.entry)
    row = {
        "trade_id": f"{candidate.symbol}-{now.strftime('%Y%m%d%H%M%S')}",
        "symbol": candidate.symbol,
        "side": side,
        "status": "open",
        "opened_at": now.isoformat(),
        "closed_at": "",
        "entry": entry,
        "last_price": entry,
        "invalidation": candidate.invalidation,
        "target_1": candidate.target_1,
        "target_2": candidate.target_2,
        "unrealized_return_pct": 0.0,
        "realized_return_pct": pd.NA,
        "confidence": candidate.confidence,
        "score": candidate.score,
        "close_reason": "",
        "reasons": " | ".join(candidate.reasons),
        "blockers": " | ".join(candidate.blockers),
    }
    return row


def _has_active_trade(trades: pd.DataFrame, symbol: str) -> bool:
    return bool((trades["symbol"].eq(symbol) & trades["status"].eq("open")).any())


def _optional_float(value: Any) -> float | None:
    if pd.isna(value) or value == "":
        return None
    return float(value)


def _return_pct(side: str, entry: float, price: float) -> float:
    if side == "short":
        return round((entry - price) / entry * 100, 4)
    return round((price - entry) / entry * 100, 4)


def _close_reason(side: str, price: float, invalidation: float | None, target_1: float | None, target_2: float | None) -> str | None:
    targets = [target for target in [target_1, target_2] if target is not None]
    if side == "long":
        if invalidation is not None and price <= invalidation:
            return "invalidation hit"
        if targets and price >= min(targets):
            return "target hit"
    else:
        if invalidation is not None and price >= invalidation:
            return "invalidation hit"
        if targets and price <= max(targets):
            return "target hit"
    return None
