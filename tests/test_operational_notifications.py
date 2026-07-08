from __future__ import annotations

import json

from ops.notify_operational_job import build_notification_payload, send_notification


def test_build_notification_payload_summarizes_reports(tmp_path) -> None:
    (tmp_path / "retraining_job.json").write_text(
        json.dumps(
            {
                "attempted": 2,
                "succeeded": 1,
                "failed": 0,
                "skipped": [
                    {
                        "ticker": "BTC-USD",
                        "reason": "candidate_not_better_than_incumbent",
                    }
                ],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )

    payload = build_notification_payload(
        tmp_path,
        status="success",
        job_mode="full_retrain",
        run_url="https://github.com/example/actions/runs/1",
        repository="owner/repo",
        ref="main",
    )

    assert payload["title"] == "IA Inversiones operational job completed"
    assert payload["job_mode"] == "full_retrain"
    assert payload["repository"] == "owner/repo"
    assert payload["skipped"] == 1
    assert payload["failed"] == 0
    assert payload["reports"][0]["skipped_summaries"][0]["ticker"] == "BTC-USD"


def test_build_notification_payload_marks_failed_when_report_has_errors(tmp_path) -> None:
    (tmp_path / "market_data_job.json").write_text(
        json.dumps({"attempted": 1, "succeeded": 0, "failed": 1, "errors": [{"ticker": "AAPL", "error": "timeout"}]}),
        encoding="utf-8",
    )

    payload = build_notification_payload(tmp_path, status="success")

    assert payload["title"] == "IA Inversiones operational job failed"
    assert payload["failed"] == 1
    assert payload["reports"][0]["error_summaries"][0]["reason"] == "timeout"


def test_send_notification_skips_when_webhook_is_missing() -> None:
    result = send_notification({"status": "success"}, None)

    assert result == {"sent": False, "reason": "missing_webhook_url"}
