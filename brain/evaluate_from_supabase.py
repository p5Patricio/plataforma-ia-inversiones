from __future__ import annotations

import argparse
import json
from pathlib import Path

from brain.backtesting import BacktestConfig, run_walk_forward_model_backtest
from brain.datasets import build_dataset_from_materialized
from brain.inference import PredictionPolicy
from brain.models import DEFAULT_MODEL_NAME, available_model_names
from collector.supabase_repository import SupabaseConfig, SupabaseRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward model backtesting from Supabase datasets")
    parser.add_argument("--ticker", required=True, help="Asset ticker stored in Supabase")
    parser.add_argument("--feature-set", default="technical_v1")
    parser.add_argument("--label-method", choices=["fixed_horizon", "triple_barrier"], default="triple_barrier")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--model-name", choices=available_model_names(), default=DEFAULT_MODEL_NAME)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--embargo-rows", type=int, help="Rows skipped between train and test. Defaults to horizon")
    parser.add_argument("--trade-stride", type=int, help="Evaluate one trade every N rows. Defaults to horizon")
    parser.add_argument("--limit", type=int, help="Optional max number of materialized rows to read")
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--position-size", type=float, default=1.0)
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--no-short", action="store_true")
    parser.add_argument("--out", help="Optional JSON report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = SupabaseRepository(SupabaseConfig.from_env())
    asset_id = repository.get_asset_id(args.ticker)
    features = repository.get_features(asset_id, args.feature_set, limit=args.limit)
    labels = repository.get_labels(asset_id, args.label_method, args.horizon, limit=args.limit)
    dataset = build_dataset_from_materialized(features, labels)

    result = run_walk_forward_model_backtest(
        dataset,
        n_splits=args.splits,
        test_size=args.test_size,
        embargo_rows=args.embargo_rows if args.embargo_rows is not None else args.horizon,
        trade_stride=args.trade_stride if args.trade_stride is not None else args.horizon,
        model_name=args.model_name,
        prediction_policy=PredictionPolicy(min_confidence=args.min_confidence),
        config=BacktestConfig(
            initial_capital=args.initial_capital,
            position_size=args.position_size,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            allow_short=not args.no_short,
        ),
    )

    payload = {
        "ticker": args.ticker.upper(),
        "asset_id": asset_id,
        "feature_set": args.feature_set,
        "label_method": args.label_method,
        "horizon": args.horizon,
        "model_name": args.model_name,
        "summary": result.summary,
        "folds": result.folds,
        "predictions": result.predictions.to_dict(orient="records"),
    }

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
