from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeedbackReport:
    summary: dict
    by_action: list[dict]
    by_confidence_bucket: list[dict]


def analyze_prediction_feedback(feedback: pd.DataFrame) -> FeedbackReport:
    """Summarize evaluated predictions for model monitoring and retraining decisions."""
    if feedback.empty:
        return FeedbackReport(
            summary={
                "evaluated_predictions": 0,
                "accuracy": None,
                "mean_outcome_return": None,
                "total_outcome_return": None,
            },
            by_action=[],
            by_confidence_bucket=[],
        )

    required = {"predicted_action", "actual_label", "is_correct", "confidence", "outcome_return"}
    missing = required - set(feedback.columns)
    if missing:
        raise ValueError(f"feedback missing columns: {sorted(missing)}")

    evaluated = feedback.dropna(subset=["actual_label"]).copy()
    if evaluated.empty:
        return analyze_prediction_feedback(pd.DataFrame())

    evaluated["is_correct"] = evaluated["is_correct"].astype(bool)
    evaluated["confidence"] = pd.to_numeric(evaluated["confidence"], errors="coerce")
    evaluated["outcome_return"] = pd.to_numeric(evaluated["outcome_return"], errors="coerce")

    summary = {
        "evaluated_predictions": int(len(evaluated)),
        "accuracy": float(evaluated["is_correct"].mean()),
        "mean_confidence": _nullable_float(evaluated["confidence"].mean()),
        "mean_outcome_return": _nullable_float(evaluated["outcome_return"].mean()),
        "total_outcome_return": _nullable_float(evaluated["outcome_return"].sum()),
    }

    by_action = _group_metrics(evaluated, "predicted_action")
    evaluated["confidence_bucket"] = evaluated["confidence"].map(_confidence_bucket)
    by_confidence_bucket = _group_metrics(evaluated, "confidence_bucket")

    return FeedbackReport(
        summary=summary,
        by_action=by_action,
        by_confidence_bucket=by_confidence_bucket,
    )


def _group_metrics(df: pd.DataFrame, column: str) -> list[dict]:
    rows = []
    for value, group in df.groupby(column, dropna=False):
        rows.append(
            {
                column: value,
                "count": int(len(group)),
                "accuracy": float(group["is_correct"].mean()),
                "mean_confidence": _nullable_float(group["confidence"].mean()),
                "mean_outcome_return": _nullable_float(group["outcome_return"].mean()),
                "total_outcome_return": _nullable_float(group["outcome_return"].sum()),
            }
        )
    return rows


def _confidence_bucket(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    if value < 0.55:
        return "<0.55"
    if value < 0.65:
        return "0.55-0.65"
    if value < 0.75:
        return "0.65-0.75"
    return ">=0.75"


def _nullable_float(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)
