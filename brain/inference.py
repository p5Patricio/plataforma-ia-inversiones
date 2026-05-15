from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from brain.features import FEATURE_COLUMNS


@dataclass(frozen=True)
class PredictionPolicy:
    min_confidence: float = 0.55
    hold_action: str = "HOLD"


def predict_actions(
    model,
    feature_frame: pd.DataFrame,
    policy: PredictionPolicy | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Generate action predictions and class probabilities from a trained model."""
    policy = policy or PredictionPolicy()
    columns = feature_columns or FEATURE_COLUMNS
    missing = set(["timestamp", *columns]) - set(feature_frame.columns)
    if missing:
        raise ValueError(f"feature_frame missing columns: {sorted(missing)}")

    X = feature_frame[columns]
    classes = [str(label) for label in _model_classes(model)]
    probabilities = model.predict_proba(X)

    rows = []
    for row_index, (_, row) in enumerate(feature_frame.iterrows()):
        probability_map = {
            label: float(probabilities[row_index][class_index])
            for class_index, label in enumerate(classes)
        }
        raw_action = max(probability_map, key=probability_map.get)
        confidence = probability_map[raw_action]
        action = raw_action if confidence >= policy.min_confidence else policy.hold_action

        rows.append(
            {
                "timestamp": row["timestamp"],
                "action": action,
                "confidence": confidence,
                "expected_return": None,
                "expected_risk": None,
                "probabilities": probability_map,
                "metadata": {
                    "raw_action": raw_action,
                    "min_confidence": policy.min_confidence,
                    "feature_columns": columns,
                },
            }
        )

    return pd.DataFrame(rows)


def _model_classes(model) -> list:
    if hasattr(model, "classes_"):
        return list(model.classes_)
    if hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")
        if classifier is not None and hasattr(classifier, "classes_"):
            return list(classifier.classes_)
    raise ValueError("Model does not expose classifier classes")
