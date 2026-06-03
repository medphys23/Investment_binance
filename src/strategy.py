"""Non-executing paper strategy layer built from dashboard signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import KEY_TIMEFRAMES


@dataclass(frozen=True)
class PaperTradeCandidate:
    symbol: str
    action: str
    confidence: str
    entry: float
    invalidation: float | None
    target_1: float | None
    target_2: float | None
    score: int
    reasons: list[str]
    blockers: list[str]


def _side(analysis: dict[str, Any]) -> str:
    ema = analysis["ema_state"]
    rsi = analysis["rsi"]
    scenario = analysis["scenario"]
    if ema.startswith("bullish") and rsi >= 50 and "bearish" not in scenario:
        return "bullish"
    if ema.startswith("bearish") and rsi <= 50 and "bullish" not in scenario:
        return "bearish"
    return "mixed"


def _risk_levels(action: str, analysis: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    if action not in {"paper_long", "paper_short"}:
        return None, None, None
    price = float(analysis["price"])
    atr = float(analysis["atr"])
    fib = analysis["fib"]
    if action == "paper_long":
        nearby_supports = [level for level in fib.levels.values() if level < price]
        invalidation = max(nearby_supports) if nearby_supports else price - 1.5 * atr
        target_1 = fib.extensions.get("1.272")
        target_2 = fib.extensions.get("1.618")
    elif action == "paper_short":
        nearby_resistances = [level for level in fib.levels.values() if level > price]
        invalidation = min(nearby_resistances) if nearby_resistances else price + 1.5 * atr
        below_price = sorted(
            {level for level in {**fib.levels, **fib.extensions}.values() if level < price},
            reverse=True,
        )
        if len(below_price) >= 2:
            target_1, target_2 = below_price[0], below_price[-1]
        elif len(below_price) == 1:
            target_1, target_2 = below_price[0], price - 2.5 * atr
        else:
            target_1, target_2 = price - 1.5 * atr, price - 2.5 * atr
    return invalidation, target_1, target_2


def generate_paper_trade_candidate(symbol: str, analyses: dict[str, dict[str, Any]]) -> PaperTradeCandidate:
    """Create a paper-trade candidate from key timeframe confluence.

    This deliberately does not place orders or access account state.
    """

    sides = {timeframe: _side(analyses[timeframe]) for timeframe in KEY_TIMEFRAMES if timeframe in analyses}
    bullish = sum(1 for side in sides.values() if side == "bullish")
    bearish = sum(1 for side in sides.values() if side == "bearish")
    mixed = sum(1 for side in sides.values() if side == "mixed")
    primary = analyses.get("12h") or analyses[KEY_TIMEFRAMES[0]]
    reasons: list[str] = []
    blockers: list[str] = []
    score = 0

    if bullish >= 4:
        action = "paper_long"
        score += 3
        reasons.append("at least four key timeframes lean bullish")
    elif bearish >= 4:
        action = "paper_short"
        score += 3
        reasons.append("at least four key timeframes lean bearish")
    elif bullish >= 3 and sides.get("12h") == "bullish":
        action = "paper_long"
        score += 2
        reasons.append("12h regime and majority of key timeframes lean bullish")
    elif bearish >= 3 and sides.get("12h") == "bearish":
        action = "paper_short"
        score += 2
        reasons.append("12h regime and majority of key timeframes lean bearish")
    else:
        action = "observe"
        blockers.append("key timeframes do not have enough directional agreement")

    if sides.get("1d") != "mixed" and sides.get("1w") != "mixed" and sides.get("1d") != sides.get("1w"):
        score -= 1
        blockers.append("daily and weekly disagree")
    if sides.get("1h") != "mixed" and sides.get("12h") != "mixed" and sides.get("1h") != sides.get("12h"):
        score -= 1
        blockers.append("1h conflicts with 12h regime")
    if mixed >= 3:
        score -= 1
        blockers.append("too many mixed timeframes")

    for timeframe, analysis in analyses.items():
        if analysis.get("confidence") == "high":
            score += 1
            reasons.append(f"{timeframe} has high signal confidence")
        if analysis.get("relative_volume", 0) >= 1.2:
            score += 1
            reasons.append(f"{timeframe} has elevated relative volume")
        if analysis.get("bollinger_state") == "squeeze":
            reasons.append(f"{timeframe} is in Bollinger squeeze")
        if "mean-reversion" in analysis.get("scenario", ""):
            blockers.append(f"{timeframe} flags mean-reversion risk")

    if action in {"paper_long", "paper_short"} and score < 3:
        action = "observe"
        blockers.append("confluence score is too low for a paper candidate")

    confidence = "high" if score >= 6 else "medium" if score >= 3 else "low"
    invalidation, target_1, target_2 = _risk_levels(action, primary)
    return PaperTradeCandidate(
        symbol=symbol,
        action=action,
        confidence=confidence,
        entry=float(primary["price"]),
        invalidation=invalidation,
        target_1=target_1,
        target_2=target_2,
        score=score,
        reasons=reasons[:8],
        blockers=blockers[:8],
    )


def candidate_to_row(candidate: PaperTradeCandidate) -> dict[str, Any]:
    return {
        "Symbol": candidate.symbol,
        "Action": candidate.action,
        "Confidence": candidate.confidence,
        "Score": candidate.score,
        "Entry": candidate.entry,
        "Invalidation": candidate.invalidation,
        "Target 1": candidate.target_1,
        "Target 2": candidate.target_2,
        "Reasons": " | ".join(candidate.reasons),
        "Blockers": " | ".join(candidate.blockers),
    }
