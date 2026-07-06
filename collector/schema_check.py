from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from typing import Iterable

from requests import RequestException

from collector.supabase_repository import SupabaseConfig, SupabaseRepository


REQUIRED_ML_RELATIONS = (
    "features_daily",
    "labels_daily",
    "model_runs",
    "predictions",
    "prediction_feedback",
    "backtests",
    "backtest_trades",
    "risk_limits",
)


@dataclass(frozen=True)
class RelationStatus:
    name: str
    available: bool
    status_code: int | None = None
    error: str | None = None


def check_relations(
    repository: SupabaseRepository,
    relations: Iterable[str] = REQUIRED_ML_RELATIONS,
) -> list[RelationStatus]:
    statuses: list[RelationStatus] = []
    for relation in relations:
        try:
            response = repository._session.get(
                f"{repository.config.url}/rest/v1/{relation}",
                headers=repository.headers,
                params={"select": "*", "limit": "1"},
                timeout=30,
            )
            statuses.append(
                RelationStatus(
                    name=relation,
                    available=response.status_code < 400,
                    status_code=response.status_code,
                    error=None if response.status_code < 400 else response.text[:300],
                )
            )
        except RequestException as exc:
            statuses.append(RelationStatus(name=relation, available=False, error=str(exc)))
    return statuses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check required Supabase ML schema relations")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = SupabaseRepository(SupabaseConfig.from_env())
    statuses = check_relations(repository)
    missing = [status for status in statuses if not status.available]

    if args.json:
        print(json.dumps([asdict(status) for status in statuses], indent=2))
    else:
        for status in statuses:
            state = "OK" if status.available else "MISSING"
            suffix = f" ({status.status_code})" if status.status_code else ""
            print(f"{state}\t{status.name}{suffix}")

        if missing:
            print()
            print("Apply supabase/migrations/20260705000100_ml_pipeline_tables.sql in Supabase SQL Editor.")

    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
