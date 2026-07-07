from __future__ import annotations

import argparse
import json
from pathlib import Path

from brain.artifacts import DEFAULT_MODEL_ARTIFACT_BUCKET, is_supabase_artifact_uri, upload_supabase_artifact
from collector.supabase_repository import SupabaseConfig, SupabaseRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a local model artifact to Supabase Storage")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--artifact-path", help="Local .joblib path. Defaults to current model_runs.artifact_uri")
    parser.add_argument("--bucket", default=DEFAULT_MODEL_ARTIFACT_BUCKET)
    parser.add_argument("--object-path", help="Storage object path. Defaults to models/<local filename>")
    parser.add_argument("--no-create-bucket", action="store_true")
    parser.add_argument("--out", help="Optional JSON summary path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SupabaseConfig.from_env()
    repository = SupabaseRepository(config)
    model_run = repository.get_model_run(args.model_name, args.model_version)
    artifact_path = args.artifact_path or model_run.get("artifact_uri")
    if not artifact_path:
        raise RuntimeError("artifact_path_required")
    if is_supabase_artifact_uri(str(artifact_path)):
        raise RuntimeError("model_run_already_points_to_supabase_artifact")

    object_path = args.object_path or f"models/{Path(str(artifact_path)).name}"
    artifact_uri = upload_supabase_artifact(
        artifact_path,
        config=config,
        bucket=args.bucket,
        object_path=object_path,
        create_bucket=not args.no_create_bucket,
    )
    updated = repository.update_model_run_artifact_uri(model_run["id"], str(artifact_uri))
    payload = {
        "model_run_id": model_run["id"],
        "model_name": args.model_name,
        "model_version": args.model_version,
        "artifact_uri": str(artifact_uri),
        "updated": updated,
    }

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
