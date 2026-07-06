from __future__ import annotations

import numpy as np
import pandas as pd

from brain.backtesting import BacktestConfig, run_prediction_backtest
from brain.backtesting import run_walk_forward_model_backtest
from brain.datasets import build_dataset_from_materialized, build_supervised_dataset
from brain.feedback import analyze_prediction_feedback
from brain.features import FEATURE_COLUMNS, build_features
from brain.inference import PredictionPolicy, predict_actions
from brain.labeling import BUY, HOLD, SELL, fixed_horizon_labels, triple_barrier_labels
from brain.models import available_model_names, create_model, walk_forward_evaluate
from brain.risk import RiskPolicy, apply_risk_policy


def make_prices(rows: int = 120) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    wave = np.sin(np.arange(rows) / 3) * 2.5
    trend = np.arange(rows) * 0.03
    close = 100 + wave + trend
    open_ = close + np.cos(np.arange(rows)) * 0.2
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    volume = 1_000_000 + (np.arange(rows) % 9) * 10_000

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_features_do_not_change_when_future_price_changes() -> None:
    prices = make_prices(80)
    changed = prices.copy()
    changed.loc[79, "close"] = changed.loc[79, "close"] * 10
    changed.loc[79, "high"] = changed.loc[79, "high"] * 10

    original_features = build_features(prices).loc[:78, FEATURE_COLUMNS]
    changed_features = build_features(changed).loc[:78, FEATURE_COLUMNS]

    pd.testing.assert_frame_equal(original_features, changed_features)


def test_fixed_horizon_labels_create_expected_classes() -> None:
    labels = fixed_horizon_labels(
        make_prices(80),
        horizon=3,
        buy_threshold=0.005,
        sell_threshold=-0.005,
    )

    assert {BUY, SELL, HOLD}.intersection(set(labels["label"]))
    assert labels["future_return"].notna().sum() == 77
    assert labels["label"].isna().sum() == 3


def test_triple_barrier_labels_include_exit_metadata() -> None:
    labels = triple_barrier_labels(
        make_prices(60),
        horizon=4,
        profit_take=0.01,
        stop_loss=0.01,
    )

    assert {"label", "outcome_return", "label_exit_timestamp"}.issubset(labels.columns)
    assert set(labels["label"].dropna()).issubset({BUY, SELL, HOLD})
    assert labels["label"].isna().sum() == 4


def test_build_supervised_dataset_has_training_columns() -> None:
    dataset = build_supervised_dataset(
        make_prices(100),
        label_method="fixed_horizon",
        horizon=3,
        buy_threshold=0.005,
        sell_threshold=-0.005,
    )

    assert set(FEATURE_COLUMNS + ["label"]).issubset(dataset.columns)
    assert dataset[FEATURE_COLUMNS + ["label"]].isna().sum().sum() == 0


def test_walk_forward_evaluate_returns_fold_metrics() -> None:
    dataset = build_supervised_dataset(
        make_prices(140),
        label_method="fixed_horizon",
        horizon=3,
        buy_threshold=0.003,
        sell_threshold=-0.003,
    )

    result = walk_forward_evaluate(dataset, n_splits=3)

    assert len(result.fold_metrics) == 3
    assert result.summary["rows"] == len(dataset)
    assert 0 <= result.summary["mean_f1_macro"] <= 1


def test_model_registry_exposes_comparable_candidates() -> None:
    names = available_model_names()

    assert "baseline_hist_gradient_boosting" in names
    assert "logistic_regression" in names
    assert "random_forest" in names
    assert create_model("extra_trees").named_steps["classifier"].__class__.__name__ == "ExtraTreesClassifier"


def test_walk_forward_evaluate_accepts_registered_model_name() -> None:
    dataset = build_supervised_dataset(
        make_prices(140),
        label_method="fixed_horizon",
        horizon=3,
        buy_threshold=0.003,
        sell_threshold=-0.003,
    )

    result = walk_forward_evaluate(dataset, n_splits=3, model_name="logistic_regression")

    assert result.summary["model_name"] == "logistic_regression"
    assert result.summary["estimator"] == "LogisticRegression"


def test_build_dataset_from_materialized_expands_feature_json() -> None:
    feature_rows = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
            "features": [
                {column: 0.1 for column in FEATURE_COLUMNS},
                {column: 0.2 for column in FEATURE_COLUMNS},
            ],
        }
    )
    label_rows = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
            "label": ["BUY", "HOLD"],
            "outcome_return": [0.03, 0.0],
        }
    )

    dataset = build_dataset_from_materialized(feature_rows, label_rows)

    assert list(dataset[FEATURE_COLUMNS].iloc[0]) == [0.1] * len(FEATURE_COLUMNS)
    assert dataset["label"].tolist() == ["BUY", "HOLD"]


class FakeClassifier:
    classes_ = ["BUY", "HOLD", "SELL"]

    def predict_proba(self, X):
        return [
            [0.7, 0.2, 0.1],
            [0.4, 0.45, 0.15],
        ]


def test_predict_actions_uses_confidence_threshold() -> None:
    feature_frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
            **{column: [0.1, 0.2] for column in FEATURE_COLUMNS},
        }
    )

    predictions = predict_actions(
        FakeClassifier(),
        feature_frame,
        policy=PredictionPolicy(min_confidence=0.6),
    )

    assert predictions["action"].tolist() == ["BUY", "HOLD"]
    assert predictions.loc[0, "confidence"] == 0.7
    assert predictions.loc[1, "metadata"]["raw_action"] == "HOLD"


def test_analyze_prediction_feedback_summarizes_errors_and_returns() -> None:
    feedback = pd.DataFrame(
        {
            "predicted_action": ["BUY", "BUY", "SELL", "HOLD"],
            "actual_label": ["BUY", "SELL", "SELL", "HOLD"],
            "is_correct": [True, False, True, True],
            "confidence": [0.8, 0.6, 0.7, 0.5],
            "outcome_return": [0.03, -0.02, 0.01, 0.0],
        }
    )

    report = analyze_prediction_feedback(feedback)

    assert report.summary["evaluated_predictions"] == 4
    assert report.summary["accuracy"] == 0.75
    assert np.isclose(report.summary["total_outcome_return"], 0.02)
    assert {row["predicted_action"] for row in report.by_action} == {"BUY", "SELL", "HOLD"}
    assert report.by_confidence_bucket


def test_analyze_prediction_feedback_handles_empty_data() -> None:
    report = analyze_prediction_feedback(pd.DataFrame())

    assert report.summary["evaluated_predictions"] == 0
    assert report.summary["accuracy"] is None


def test_run_prediction_backtest_applies_costs_and_equity_curve() -> None:
    feedback = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
            "predicted_action": ["BUY", "SELL", "HOLD"],
            "actual_label": ["BUY", "SELL", "HOLD"],
            "confidence": [0.8, 0.7, 0.6],
            "outcome_return": [0.03, -0.02, 0.01],
            "model_name": ["baseline", "baseline", "baseline"],
            "model_version": ["v1", "v1", "v1"],
        }
    )

    result = run_prediction_backtest(
        feedback,
        BacktestConfig(initial_capital=1000, fee_bps=5, slippage_bps=5),
    )

    assert result.metrics["trade_count"] == 3
    assert result.metrics["active_trade_count"] == 2
    assert result.metrics["final_equity"] > 1000
    assert np.isclose(result.trades.loc[0, "net_return"], 0.028)
    assert result.trades.loc[1, "gross_return"] == 0.02


def test_run_prediction_backtest_handles_empty_feedback() -> None:
    result = run_prediction_backtest(pd.DataFrame(), BacktestConfig(initial_capital=5000))

    assert result.metrics["final_equity"] == 5000
    assert result.metrics["trade_count"] == 0
    assert result.trades.empty


def test_run_walk_forward_model_backtest_compares_baselines() -> None:
    dataset = build_supervised_dataset(
        make_prices(180),
        label_method="triple_barrier",
        horizon=3,
        profit_take=0.01,
        stop_loss=0.01,
    )

    result = run_walk_forward_model_backtest(
        dataset,
        n_splits=3,
        model_name="random_forest",
        config=BacktestConfig(initial_capital=1000, fee_bps=5, slippage_bps=5),
    )

    assert result.summary["evaluated_rows"] == len(result.predictions)
    assert result.summary["model_name"] == "random_forest"
    assert result.summary["embargo_rows"] == 0
    assert result.summary["trade_stride"] == 1
    assert len(result.folds) == 3
    assert "model" in result.summary
    assert {"no_trade", "always_buy", "always_sell"}.issubset(result.baselines)
    assert result.model_backtest.metrics["trade_count"] == len(result.predictions)
    assert result.baselines["no_trade"].metrics["final_equity"] == 1000


def test_run_walk_forward_model_backtest_supports_embargo_and_trade_stride() -> None:
    dataset = build_supervised_dataset(
        make_prices(180),
        label_method="triple_barrier",
        horizon=3,
        profit_take=0.01,
        stop_loss=0.01,
    )

    result = run_walk_forward_model_backtest(dataset, n_splits=3, embargo_rows=3, trade_stride=3)

    assert result.summary["embargo_rows"] == 3
    assert result.summary["trade_stride"] == 3
    assert result.summary["evaluated_rows"] < len(dataset)


def test_apply_risk_policy_sizes_confident_trade() -> None:
    predictions = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC"),
            "action": ["BUY"],
            "confidence": [0.8],
            "expected_risk": [0.02],
            "metadata": [{}],
        }
    )

    adjusted = apply_risk_policy(predictions, RiskPolicy(max_position_size=0.2, min_confidence_to_trade=0.6))

    assert adjusted.loc[0, "action"] == "BUY"
    assert adjusted.loc[0, "metadata"]["risk"]["position_size"] == 0.1
    assert adjusted.loc[0, "metadata"]["risk"]["blocked_reasons"] == []


def test_apply_risk_policy_blocks_low_confidence_trade() -> None:
    predictions = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC"),
            "action": ["BUY"],
            "confidence": [0.55],
            "expected_risk": [0.02],
            "metadata": [{}],
        }
    )

    adjusted = apply_risk_policy(predictions, RiskPolicy(min_confidence_to_trade=0.6))

    assert adjusted.loc[0, "action"] == "HOLD"
    assert "confidence_below_trade_threshold" in adjusted.loc[0, "metadata"]["risk"]["blocked_reasons"]


def test_apply_risk_policy_blocks_short_when_disabled() -> None:
    predictions = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC"),
            "action": ["SELL"],
            "confidence": [0.9],
            "expected_risk": [0.02],
            "metadata": [{}],
        }
    )

    adjusted = apply_risk_policy(predictions, RiskPolicy(allow_short=False))

    assert adjusted.loc[0, "action"] == "HOLD"
    assert "short_disabled" in adjusted.loc[0, "metadata"]["risk"]["blocked_reasons"]
