from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from brain.feedback import analyze_prediction_feedback
from collector.supabase_repository import SupabaseConfig, SupabaseRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze evaluated prediction feedback")
    parser.add_argument("--ticker", help="Optional asset ticker stored in Supabase")
    parser.add_argument("--model-name")
    parser.add_argument("--model-version")
    parser.add_argument("--include-pending", action="store_true", help="Include predictions without labels")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", help="Optional JSON report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = SupabaseRepository(SupabaseConfig.from_env())
    asset_id = repository.get_asset_id(args.ticker) if args.ticker else None
    feedback = repository.get_prediction_feedback(
        model_name=args.model_name,
        model_version=args.model_version,
        asset_id=asset_id,
        only_evaluated=not args.include_pending,
        limit=args.limit,
    )
    report = analyze_prediction_feedback(feedback)
    payload = asdict(report)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
