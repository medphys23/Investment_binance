"""Batch learning from closed local paper trades."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from .config import PAPER_MIN_ML_CLOSED_TRADES


def train_batch_model(conn: sqlite3.Connection, min_closed: int = PAPER_MIN_ML_CLOSED_TRADES) -> dict[str, Any]:
    now = pd.Timestamp.now(tz="UTC").isoformat()
    data = pd.read_sql_query(
        """
        SELECT
            p.realized_return_pct,
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
            f.side
        FROM paper_trades p
        LEFT JOIN signal_features f ON p.feature_id = f.id
        WHERE p.status = 'closed' AND p.realized_return_pct IS NOT NULL
        """,
        conn,
    )
    if len(data) < min_closed:
        result = {"status": "skipped", "sample_count": len(data), "skipped_reason": f"need at least {min_closed} closed trades"}
        _insert_model_run(conn, now, len(data), "RandomForestClassifier", None, None, None, [], result["skipped_reason"])
        return result
    labels = (data["realized_return_pct"] > 0).astype(int)
    if labels.nunique() < 2:
        result = {"status": "skipped", "sample_count": len(data), "skipped_reason": "closed trades only contain one outcome class"}
        _insert_model_run(conn, now, len(data), "RandomForestClassifier", None, None, None, [], result["skipped_reason"])
        return result

    features = data.drop(columns=["realized_return_pct"]).copy()
    for column in features.columns:
        if pd.api.types.is_numeric_dtype(features[column]):
            features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0)
        else:
            features[column] = features[column].fillna("missing").astype(str)
    features = pd.get_dummies(features, drop_first=False)
    test_size = 0.25 if len(data) >= 80 else 0.4
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42, stratify=labels)
    model = RandomForestClassifier(n_estimators=80, max_depth=4, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    importances = sorted(
        zip(features.columns, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )[:10]
    top_features = [{"feature": name, "importance": round(float(value), 5)} for name, value in importances]
    accuracy = float(accuracy_score(y_test, predictions))
    precision = float(precision_score(y_test, predictions, zero_division=0))
    recall = float(recall_score(y_test, predictions, zero_division=0))
    _insert_model_run(conn, now, len(data), "RandomForestClassifier", accuracy, precision, recall, top_features, None)
    return {
        "status": "trained",
        "sample_count": len(data),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "top_features": top_features,
    }


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
