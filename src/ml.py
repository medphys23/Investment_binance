"""Reinforcement-style batch learning from closed local paper trades.

Every PAPER_POLICY_BATCH closed trades, the bot retrains a calibrated model on
its recent experience and derives an advisory entry-probability threshold and a
position-size multiplier for the next batch. This is a periodic, reward-driven
policy (batched contextual-bandit style), not full online RL, and it never
guarantees outcomes. All adjustments stay within the existing risk ceiling and
the simulated leverage cap defined in config.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from .config import (
    PAPER_MIN_ML_CLOSED_TRADES,
    PAPER_POLICY_BASE_THRESHOLD,
    PAPER_POLICY_BATCH,
    PAPER_POLICY_MAX_SIZE_MULTIPLIER,
    PAPER_POLICY_MAX_THRESHOLD,
    PAPER_POLICY_MIN_SIZE_MULTIPLIER,
    PAPER_POLICY_MIN_THRESHOLD,
    PAPER_POLICY_MODEL_PATH,
    PAPER_POLICY_RECENT_WINDOW,
)
from .storage import get_state, latest_policy_state, set_state


_FEATURE_QUERY = """
    SELECT
        p.realized_return_pct,
        p.realized_pnl,
        p.closed_at,
        p.side AS trade_side,
        p.score,
        p.confidence AS trade_confidence,
        f.timeframe,
        f.ema_state,
        f.rsi,
        f.rsi_state,
        f.bollinger_state,
        f.relative_volume,
        f.atr,
        f.fib_distance_pct,
        f.elliott_confidence,
        f.scenario,
        f.confidence AS signal_confidence,
        f.side,
        f.macd_hist,
        f.adx,
        f.vwap_distance_pct,
        f.volatility_regime,
        f.funding_rate,
        f.futures_spot_ratio,
        f.taker_buy_ratio,
        f.long_short_ratio,
        f.btc_corr
    FROM paper_trades p
    LEFT JOIN signal_features f ON p.feature_id = f.id
    WHERE p.status = 'closed' AND p.realized_return_pct IS NOT NULL
    ORDER BY p.closed_at ASC
"""

_NON_FEATURE_COLUMNS = ["realized_return_pct", "realized_pnl", "closed_at", "trade_side"]


def _prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.drop(columns=[c for c in _NON_FEATURE_COLUMNS if c in frame.columns]).copy()
    for column in features.columns:
        if pd.api.types.is_numeric_dtype(features[column]):
            features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0)
        else:
            features[column] = features[column].fillna("missing").astype(str)
    return pd.get_dummies(features, drop_first=False)


def _closed_trade_count(conn: sqlite3.Connection) -> int:
    return int(
        conn.execute(
            "SELECT COUNT(*) AS count FROM paper_trades WHERE status = 'closed' AND realized_return_pct IS NOT NULL"
        ).fetchone()["count"]
    )


def should_update_policy(conn: sqlite3.Connection) -> bool:
    closed = _closed_trade_count(conn)
    if closed < PAPER_MIN_ML_CLOSED_TRADES:
        return False
    if latest_policy_state(conn) is None:
        return True
    current_version = closed // PAPER_POLICY_BATCH
    last_version = int(get_state(conn, "policy_version", 0) or 0)
    return current_version > last_version


def get_active_policy(conn: sqlite3.Connection) -> dict[str, float]:
    return {
        "entry_prob_threshold": float(get_state(conn, "entry_prob_threshold", PAPER_POLICY_BASE_THRESHOLD)),
        "size_multiplier": float(get_state(conn, "size_multiplier", 1.0)),
        "policy_version": int(get_state(conn, "policy_version", 0) or 0),
    }


def _batch_metrics(data: pd.DataFrame) -> dict[str, float]:
    batch = data.tail(PAPER_POLICY_BATCH)
    returns = batch["realized_return_pct"].astype(float)
    pnl = batch["realized_pnl"].astype(float) if "realized_pnl" in batch else returns
    wins = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    profit_factor = float(wins / losses) if losses else float("inf")
    win_rate = float((returns > 0).mean() * 100) if len(returns) else 0.0

    def side_win_rate(side: str) -> float | None:
        subset = batch[batch["trade_side"] == side] if "trade_side" in batch else batch.iloc[0:0]
        if subset.empty:
            return None
        return float((subset["realized_return_pct"].astype(float) > 0).mean() * 100)

    return {
        "win_rate": win_rate,
        "avg_return_pct": float(returns.mean()) if len(returns) else 0.0,
        "profit_factor": profit_factor if profit_factor != float("inf") else 999.0,
        "reward": float(pnl.sum()),
        "long_win_rate": side_win_rate("long"),
        "short_win_rate": side_win_rate("short"),
    }


def _derive_controls(metrics: dict[str, float]) -> tuple[float, float]:
    """Reward-driven adjustment: do more of what worked, less of what didn't.

    Strong recent batches relax the entry threshold and raise size toward the
    cap; weak batches tighten the threshold and cut size. Always bounded.
    """
    win_rate = metrics["win_rate"]
    profit_factor = metrics["profit_factor"]
    threshold = PAPER_POLICY_BASE_THRESHOLD
    multiplier = 1.0
    if win_rate >= 55 and profit_factor >= 1.3:
        threshold = PAPER_POLICY_MIN_THRESHOLD
        multiplier = PAPER_POLICY_MAX_SIZE_MULTIPLIER
    elif win_rate >= 50 and profit_factor >= 1.1:
        threshold = PAPER_POLICY_BASE_THRESHOLD
        multiplier = 0.75
    else:
        threshold = PAPER_POLICY_MAX_THRESHOLD
        multiplier = PAPER_POLICY_MIN_SIZE_MULTIPLIER
    threshold = min(PAPER_POLICY_MAX_THRESHOLD, max(PAPER_POLICY_MIN_THRESHOLD, threshold))
    multiplier = min(PAPER_POLICY_MAX_SIZE_MULTIPLIER, max(PAPER_POLICY_MIN_SIZE_MULTIPLIER, multiplier))
    return round(threshold, 4), round(multiplier, 4)


def _train_model(data: pd.DataFrame) -> dict[str, Any] | None:
    window = data.tail(PAPER_POLICY_RECENT_WINDOW).reset_index(drop=True)
    labels = (window["realized_return_pct"] > 0).astype(int)
    if labels.nunique() < 2:
        return None
    features = _prepare_features(window)
    split = max(int(len(window) * 0.75), 1)
    if split >= len(window):
        split = len(window) - 1
    x_train, x_test = features.iloc[:split], features.iloc[split:]
    y_train, y_test = labels.iloc[:split], labels.iloc[split:]
    if y_train.nunique() < 2 or len(x_test) == 0:
        return None

    base = RandomForestClassifier(n_estimators=120, max_depth=5, random_state=42)
    n_train = len(x_train)
    cv = min(3, int(y_train.value_counts().min()))
    if cv >= 2:
        model: Any = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
    else:
        model = base
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    importances: list[tuple[str, float]] = []
    fitted = base
    if hasattr(model, "calibrated_classifiers_"):
        fitted.fit(x_train, y_train)
    if hasattr(fitted, "feature_importances_"):
        importances = sorted(
            zip(features.columns, fitted.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
    top_features = [{"feature": name, "importance": round(float(value), 5)} for name, value in importances]

    return {
        "model": model,
        "columns": list(features.columns),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "top_features": top_features,
        "sample_count": int(len(window)),
    }


def _save_model(payload: dict[str, Any]) -> None:
    path = Path(PAPER_POLICY_MODEL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"model": payload["model"], "columns": payload["columns"]}, handle)


def _load_model() -> dict[str, Any] | None:
    path = Path(PAPER_POLICY_MODEL_PATH)
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def predict_win_probability(
    conn: sqlite3.Connection,
    feature_id: int | None,
    score: int | None,
    trade_confidence: str | None,
) -> float | None:
    """Calibrated probability that a candidate becomes a winning paper trade.

    Returns None when no trained model exists yet (cold start), so the caller
    treats the entry as pass-through rather than blocking it.
    """
    if feature_id is None:
        return None
    bundle = _load_model()
    if bundle is None:
        return None
    row = conn.execute("SELECT * FROM signal_features WHERE id = ?", (feature_id,)).fetchone()
    if row is None:
        return None

    record = dict(row)
    frame = pd.DataFrame(
        [
            {
                "score": score if score is not None else 0,
                "trade_confidence": trade_confidence or "missing",
                "timeframe": record.get("timeframe"),
                "ema_state": record.get("ema_state"),
                "rsi": record.get("rsi"),
                "rsi_state": record.get("rsi_state"),
                "bollinger_state": record.get("bollinger_state"),
                "relative_volume": record.get("relative_volume"),
                "atr": record.get("atr"),
                "fib_distance_pct": record.get("fib_distance_pct"),
                "elliott_confidence": record.get("elliott_confidence"),
                "scenario": record.get("scenario"),
                "signal_confidence": record.get("confidence"),
                "side": record.get("side"),
                "macd_hist": record.get("macd_hist"),
                "adx": record.get("adx"),
                "vwap_distance_pct": record.get("vwap_distance_pct"),
                "volatility_regime": record.get("volatility_regime"),
                "funding_rate": record.get("funding_rate"),
                "futures_spot_ratio": record.get("futures_spot_ratio"),
                "taker_buy_ratio": record.get("taker_buy_ratio"),
                "long_short_ratio": record.get("long_short_ratio"),
                "btc_corr": record.get("btc_corr"),
            }
        ]
    )
    encoded = _prepare_features(frame).reindex(columns=bundle["columns"], fill_value=0)
    try:
        proba = bundle["model"].predict_proba(encoded)[0]
        classes = list(bundle["model"].classes_)
        if 1 in classes:
            return float(proba[classes.index(1)])
        return float(proba[-1])
    except Exception:
        return None


def update_policy(conn: sqlite3.Connection, now: pd.Timestamp | None = None) -> dict[str, Any]:
    """Retrain on recent experience and set controls for the next batch."""
    timestamp = (now or pd.Timestamp.now(tz="UTC")).isoformat()
    data = pd.read_sql_query(_FEATURE_QUERY, conn)
    closed = len(data)
    version = closed // PAPER_POLICY_BATCH

    if closed < PAPER_MIN_ML_CLOSED_TRADES:
        _persist_policy(
            conn,
            timestamp,
            version,
            closed,
            PAPER_POLICY_BASE_THRESHOLD,
            1.0,
            metrics=None,
            note=f"cold start: need {PAPER_MIN_ML_CLOSED_TRADES} closed trades",
        )
        _insert_model_run(conn, timestamp, closed, "policy", None, None, None, [], "cold start")
        return {"status": "cold_start", "closed_trades": closed}

    metrics = _batch_metrics(data)
    threshold, multiplier = _derive_controls(metrics)
    trained = _train_model(data)

    if trained is None:
        _persist_policy(conn, timestamp, version, closed, threshold, multiplier, metrics, "single-class window")
        _insert_model_run(conn, timestamp, closed, "policy", None, None, None, [], "single outcome class in window")
        return {"status": "skipped", "closed_trades": closed, "reason": "single-class window"}

    _save_model(trained)
    _persist_policy(conn, timestamp, version, closed, threshold, multiplier, metrics, "trained")
    _insert_model_run(
        conn,
        timestamp,
        trained["sample_count"],
        "RandomForestClassifier+calibrated",
        trained["accuracy"],
        trained["precision"],
        trained["recall"],
        trained["top_features"],
        None,
    )
    return {
        "status": "trained",
        "closed_trades": closed,
        "policy_version": version,
        "entry_prob_threshold": threshold,
        "size_multiplier": multiplier,
        "metrics": metrics,
    }


def maybe_update_policy(conn: sqlite3.Connection, now: pd.Timestamp | None = None) -> dict[str, Any] | None:
    if not should_update_policy(conn):
        return None
    return update_policy(conn, now)


def _persist_policy(
    conn: sqlite3.Connection,
    timestamp: str,
    version: int,
    closed: int,
    threshold: float,
    multiplier: float,
    metrics: dict[str, float] | None,
    note: str,
) -> None:
    conn.execute(
        """
        INSERT INTO policy_state
        (updated_at, policy_version, closed_trades, entry_prob_threshold, size_multiplier,
         batch_win_rate, batch_avg_return_pct, batch_profit_factor, batch_reward,
         long_win_rate, short_win_rate, note)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            timestamp,
            version,
            closed,
            threshold,
            multiplier,
            metrics.get("win_rate") if metrics else None,
            metrics.get("avg_return_pct") if metrics else None,
            metrics.get("profit_factor") if metrics else None,
            metrics.get("reward") if metrics else None,
            metrics.get("long_win_rate") if metrics else None,
            metrics.get("short_win_rate") if metrics else None,
            note,
        ),
    )
    set_state(conn, "entry_prob_threshold", threshold)
    set_state(conn, "size_multiplier", multiplier)
    set_state(conn, "policy_version", version)
    conn.commit()


def _insert_model_run(
    conn: sqlite3.Connection,
    trained_at: str,
    sample_count: int,
    model_type: str,
    accuracy: float | None,
    precision: float | None,
    recall: float | None,
    top_features: list[dict[str, Any]],
    skipped_reason: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO model_runs
        (trained_at, sample_count, model_type, accuracy, precision, recall, top_features, skipped_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (trained_at, sample_count, model_type, accuracy, precision, recall, json.dumps(top_features), skipped_reason),
    )
    conn.commit()


def train_batch_model(conn: sqlite3.Connection, min_closed: int = PAPER_MIN_ML_CLOSED_TRADES) -> dict[str, Any]:
    """Backward-compatible entry point used by tests and ad-hoc runs.

    Delegates to the policy update so a single call still trains and records a
    model run when enough closed trades exist.
    """
    data = pd.read_sql_query(_FEATURE_QUERY, conn)
    now = pd.Timestamp.now(tz="UTC").isoformat()
    if len(data) < min_closed:
        result = {"status": "skipped", "sample_count": len(data), "skipped_reason": f"need at least {min_closed} closed trades"}
        _insert_model_run(conn, now, len(data), "policy", None, None, None, [], result["skipped_reason"])
        return result
    trained = _train_model(data)
    if trained is None:
        result = {"status": "skipped", "sample_count": len(data), "skipped_reason": "closed trades only contain one outcome class"}
        _insert_model_run(conn, now, len(data), "policy", None, None, None, [], result["skipped_reason"])
        return result
    _save_model(trained)
    _insert_model_run(
        conn,
        now,
        trained["sample_count"],
        "RandomForestClassifier+calibrated",
        trained["accuracy"],
        trained["precision"],
        trained["recall"],
        trained["top_features"],
        None,
    )
    return {
        "status": "trained",
        "sample_count": trained["sample_count"],
        "accuracy": trained["accuracy"],
        "precision": trained["precision"],
        "recall": trained["recall"],
        "top_features": trained["top_features"],
    }
