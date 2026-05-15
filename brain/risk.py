from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


TRADE_ACTIONS = {"BUY", "SELL"}


@dataclass(frozen=True)
class RiskPolicy:
    max_position_size: float = 0.10
    min_confidence_to_trade: float = 0.60
    max_expected_risk: float = 0.05
    stop_loss: float = 0.02
    take_profit: float = 0.04
    allow_short: bool = True


def apply_risk_policy(predictions: pd.DataFrame, policy: RiskPolicy | None = None) -> pd.DataFrame:
    """Apply position sizing and risk blocks to model predictions."""
    policy = policy or RiskPolicy()
    if predictions.empty:
        return predictions.copy()
    required = {"action", "confidence", "metadata"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")

    adjusted = predictions.copy()
    rows = []
    for _, row in adjusted.iterrows():
        action = str(row["action"])
        confidence = float(row["confidence"])
        expected_risk = _safe_float(row.get("expected_risk"))
        blocked_reasons = _blocked_reasons(action, confidence, expected_risk, policy)
        final_action = "HOLD" if blocked_reasons else action
        position_size = 0.0 if final_action not in TRADE_ACTIONS else _position_size(confidence, policy)
        metadata = dict(row.get("metadata") or {})
        metadata["risk"] = {
            "policy": asdict(policy),
            "blocked_reasons": blocked_reasons,
            "position_size": position_size,
            "stop_loss": policy.stop_loss if position_size > 0 else None,
            "take_profit": policy.take_profit if position_size > 0 else None,
            "pre_risk_action": action,
        }
        updated = row.copy()
        updated["action"] = final_action
        updated["metadata"] = metadata
        rows.append(updated)

    return pd.DataFrame(rows).reset_index(drop=True)


def _blocked_reasons(
    action: str,
    confidence: float,
    expected_risk: float | None,
    policy: RiskPolicy,
) -> list[str]:
    reasons = []
    if action not in TRADE_ACTIONS:
        return reasons
    if action == "SELL" and not policy.allow_short:
        reasons.append("short_disabled")
    if confidence < policy.min_confidence_to_trade:
        reasons.append("confidence_below_trade_threshold")
    if expected_risk is not None and expected_risk > policy.max_expected_risk:
        reasons.append("expected_risk_above_limit")
    return reasons


def _position_size(confidence: float, policy: RiskPolicy) -> float:
    edge_scale = (confidence - policy.min_confidence_to_trade) / (1 - policy.min_confidence_to_trade)
    return round(max(0.0, min(policy.max_position_size, policy.max_position_size * edge_scale)), 6)


def _safe_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
