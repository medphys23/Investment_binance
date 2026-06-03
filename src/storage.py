"""SQLite storage for the always-on paper bot."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PAPER_DB_PATH, PAPER_STARTING_EQUITY


SCHEMA = """
CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    price REAL NOT NULL,
    price_change_pct REAL,
    volume REAL,
    quote_volume REAL,
    bid REAL,
    ask REAL,
    spread_pct REAL
);

CREATE TABLE IF NOT EXISTS signal_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    price REAL NOT NULL,
    ema_state TEXT,
    rsi REAL,
    rsi_state TEXT,
    bollinger_state TEXT,
    relative_volume REAL,
    atr REAL,
    market_structure TEXT,
    nearest_fib TEXT,
    fib_distance_pct REAL,
    elliott_state TEXT,
    elliott_confidence TEXT,
    scenario TEXT,
    confidence TEXT,
    side TEXT
);

CREATE TABLE IF NOT EXISTS paper_trades (
    trade_id TEXT PRIMARY KEY,
    feature_id INTEGER,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    status TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    entry REAL NOT NULL,
    last_price REAL NOT NULL,
    invalidation REAL,
    target_1 REAL,
    target_2 REAL,
    quantity REAL NOT NULL,
    notional REAL NOT NULL,
    risk_amount REAL NOT NULL,
    leverage REAL NOT NULL,
    margin REAL NOT NULL,
    unrealized_return_pct REAL NOT NULL,
    realized_return_pct REAL,
    unrealized_pnl REAL NOT NULL,
    realized_pnl REAL,
    confidence TEXT,
    score INTEGER,
    close_reason TEXT,
    reasons TEXT,
    blockers TEXT
);

CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trained_at TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    model_type TEXT NOT NULL,
    accuracy REAL,
    precision REAL,
    recall REAL,
    top_features TEXT,
    skipped_reason TEXT
);

CREATE TABLE IF NOT EXISTS bot_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def connect(db_path: str | Path = PAPER_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def initialize_database(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    now = pd.Timestamp.now(tz="UTC").isoformat()
    defaults = {
        "paper_equity": PAPER_STARTING_EQUITY,
        "paper_peak_equity": PAPER_STARTING_EQUITY,
        "drawdown_pct": 0.0,
        "simulated_leverage": 1.0,
        "worker_status": "not_started",
        "last_heartbeat": "",
        "last_cycle_error": "",
    }
    for key, value in defaults.items():
        conn.execute(
            "INSERT OR IGNORE INTO bot_state (key, value, updated_at) VALUES (?, ?, ?)",
            (key, json.dumps(value), now),
        )
    conn.commit()


def set_state(conn: sqlite3.Connection, key: str, value: Any, now: pd.Timestamp | None = None) -> None:
    timestamp = (now or pd.Timestamp.now(tz="UTC")).isoformat()
    conn.execute(
        """
        INSERT INTO bot_state (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """,
        (key, json.dumps(value), timestamp),
    )


def get_state(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = conn.execute("SELECT value FROM bot_state WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    return json.loads(row["value"])


def table(conn: sqlite3.Connection, name: str, limit: int | None = None) -> pd.DataFrame:
    order = " ORDER BY id DESC" if name != "paper_trades" else " ORDER BY opened_at DESC"
    query = f"SELECT * FROM {name}{order}"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    return pd.read_sql_query(query, conn)


def latest_model_run(conn: sqlite3.Connection) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM model_runs ORDER BY id DESC LIMIT 1").fetchone()
    return dict(row) if row else None
