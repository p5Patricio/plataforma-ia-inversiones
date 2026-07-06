from __future__ import annotations

import argparse
import json

from brain.features import build_features, feature_columns_for_set
from brain.labeling import fixed_horizon_labels, triple_barrier_labels
from collector.supabase_repository import SupabaseConfig, SupabaseRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize ML features and labels from Supabase prices")
    parser.add_argument("--ticker", required=True, help="Asset ticker stored in Supabase, e.g. AAPL")
    parser.add_argument("--feature-set", default="technical_v1")
    parser.add_argument("--label-method", choices=["fixed_horizon", "triple_barrier"], default="triple_barrier")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--buy-threshold", type=float, default=0.015)
    parser.add_argument("--sell-threshold", type=float, default=-0.015)
    parser.add_argument("--profit-take", type=float, default=0.03)
    parser.add_argument("--stop-loss", type=float, default=0.015)
    parser.add_argument("--limit", type=int, help="Optional max number of price rows")
    parser.add_argument("--batch-size", type=int, default=500)
    return parser.parse_args()


def build_labels(args: argparse.Namespace, prices):
    if args.label_method == "fixed_horizon":
        return fixed_horizon_labels(
            prices,
            horizon=args.horizon,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
        )

    return triple_barrier_labels(
        prices,
        horizon=args.horizon,
        profit_take=args.profit_take,
        stop_loss=args.stop_loss,
    )


def main() -> None:
    args = parse_args()
    repository = SupabaseRepository(SupabaseConfig.from_env())
    asset_id = repository.get_asset_id(args.ticker)
    prices = repository.get_prices(asset_id, limit=args.limit)

    features = build_features(prices)
    labels = build_labels(args, prices)
    feature_columns = feature_columns_for_set(args.feature_set)

    features_loaded = repository.upsert_features(
        asset_id=asset_id,
        features=features,
        feature_columns=feature_columns,
        feature_set=args.feature_set,
        batch_size=args.batch_size,
    )
    labels_loaded = repository.upsert_labels(
        asset_id=asset_id,
        labels=labels,
        label_method=args.label_method,
        horizon=args.horizon,
        batch_size=args.batch_size,
    )

    print(
        json.dumps(
            {
                "ticker": args.ticker.upper(),
                "asset_id": asset_id,
                "price_rows": len(prices),
                "feature_rows_loaded": features_loaded,
                "label_rows_loaded": labels_loaded,
                "feature_set": args.feature_set,
                "label_method": args.label_method,
                "horizon": args.horizon,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
