from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests


def build_notification_payload(
    reports_dir: str | Path,
    *,
    status: str,
    job_mode: str | None = None,
    run_url: str | None = None,
    repository: str | None = None,
    ref: str | None = None,
) -> dict[str, Any]:
    reports = []
    for report_path in sorted(Path(reports_dir).glob("*.json")):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            reports.append({"name": report_path.name, "error": f"unreadable_report:{error}"})
            continue
        reports.append(summarize_report(report_path.name, report))

    failures = sum(int(report.get("failed") or 0) for report in reports)
    skipped = sum(int(report.get("skipped_count") or 0) for report in reports)
    title_status = "failed" if status.lower() != "success" or failures else "completed"
    return {
        "title": f"IA Inversiones operational job {title_status}",
        "status": status,
        "job_mode": job_mode,
        "repository": repository,
        "ref": ref,
        "run_url": run_url,
        "failed": failures,
        "skipped": skipped,
        "reports": reports,
    }


def summarize_report(name: str, report: dict[str, Any]) -> dict[str, Any]:
    errors = report.get("errors") or []
    skipped_items = report.get("skipped") or []
    return {
        "name": name,
        "attempted": report.get("attempted"),
        "succeeded": report.get("succeeded"),
        "failed": report.get("failed", len(errors)),
        "skipped_count": len(skipped_items),
        "error_summaries": [summarize_issue(error) for error in errors[:5]],
        "skipped_summaries": [summarize_issue(item) for item in skipped_items[:5]],
    }


def summarize_issue(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": issue.get("ticker"),
        "reason": issue.get("reason") or issue.get("error") or issue.get("detail"),
    }


def send_notification(payload: dict[str, Any], webhook_url: str | None, *, session=requests) -> dict[str, Any]:
    if not webhook_url:
        return {"sent": False, "reason": "missing_webhook_url"}
    response = session.post(webhook_url, json=payload, timeout=15)
    response.raise_for_status()
    return {"sent": True, "status_code": response.status_code}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send an optional webhook summary for operational jobs")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--status", default=os.getenv("GITHUB_JOB_STATUS", "unknown"))
    parser.add_argument("--job-mode", default=os.getenv("JOB_MODE"))
    parser.add_argument("--webhook-url", default=os.getenv("OPERATIONAL_WEBHOOK_URL"))
    parser.add_argument("--run-url", default=os.getenv("GITHUB_RUN_URL"))
    parser.add_argument("--repository", default=os.getenv("GITHUB_REPOSITORY"))
    parser.add_argument("--ref", default=os.getenv("GITHUB_REF_NAME"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_notification_payload(
        args.reports_dir,
        status=args.status,
        job_mode=args.job_mode,
        run_url=args.run_url,
        repository=args.repository,
        ref=args.ref,
    )
    result = send_notification(payload, args.webhook_url)
    print(json.dumps({"notification": result, "payload": payload}, indent=2))


if __name__ == "__main__":
    main()
