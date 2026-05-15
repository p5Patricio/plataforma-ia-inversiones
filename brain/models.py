from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from brain.datasets import split_features_target
from brain.features import FEATURE_COLUMNS


@dataclass
class WalkForwardResult:
    fold_metrics: list[dict]
    summary: dict


def create_baseline_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=200,
                    l2_regularization=0.01,
                    random_state=random_state,
                ),
            ),
        ]
    )


def walk_forward_evaluate(
    dataset: pd.DataFrame,
    n_splits: int = 5,
    test_size: int | None = None,
) -> WalkForwardResult:
    """Evaluate a classifier with chronological train/test folds."""
    if len(dataset) < max(30, n_splits + 2):
        raise ValueError("Not enough rows for walk-forward evaluation")

    X, y = split_features_target(dataset)
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        model = create_baseline_model()
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        fold_metrics.append(
            {
                "fold": fold,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "accuracy": float(accuracy_score(y_test, predictions)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
                "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            }
        )

    metrics_df = pd.DataFrame(fold_metrics)
    summary = {
        "rows": len(dataset),
        "features": FEATURE_COLUMNS,
        "mean_accuracy": float(metrics_df["accuracy"].mean()),
        "mean_balanced_accuracy": float(metrics_df["balanced_accuracy"].mean()),
        "mean_f1_macro": float(metrics_df["f1_macro"].mean()),
    }
    return WalkForwardResult(fold_metrics=fold_metrics, summary=summary)


def train_final_model(dataset: pd.DataFrame) -> Pipeline:
    X, y = split_features_target(dataset)
    model = create_baseline_model()
    model.fit(X, y)
    return model
