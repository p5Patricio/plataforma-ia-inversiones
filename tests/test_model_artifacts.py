from __future__ import annotations

from pathlib import Path

from brain.artifacts import (
    download_supabase_artifact,
    parse_supabase_artifact_uri,
    resolve_model_artifact,
    upload_supabase_artifact,
)
from collector.supabase_repository import SupabaseConfig


class FakeStorageResponse:
    def __init__(
        self,
        content: bytes = b"",
        status_code: int = 200,
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.content = content
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeStorageSession:
    def __init__(self) -> None:
        self.get_calls: list[dict] = []
        self.post_calls: list[dict] = []
        self.patch_calls: list[dict] = []

    def get(self, url: str, headers: dict, timeout: int) -> FakeStorageResponse:
        self.get_calls.append({"url": url, "headers": headers, "timeout": timeout})
        return FakeStorageResponse(content=b"model-bytes")

    def post(
        self,
        url: str,
        headers: dict,
        timeout: int,
        json: dict | None = None,
        data: bytes | None = None,
    ) -> FakeStorageResponse:
        self.post_calls.append({"url": url, "headers": headers, "json": json, "data": data, "timeout": timeout})
        if "upload/resumable" in url:
            return FakeStorageResponse(status_code=201, headers={"Location": "/storage/v1/upload/resumable/upload-id"})
        return FakeStorageResponse(status_code=201)

    def patch(
        self,
        url: str,
        headers: dict,
        data: bytes,
        timeout: int,
    ) -> FakeStorageResponse:
        self.patch_calls.append({"url": url, "headers": headers, "data": data, "timeout": timeout})
        return FakeStorageResponse(status_code=204)


def test_parse_supabase_artifact_uri() -> None:
    parsed = parse_supabase_artifact_uri("supabase://model-artifacts/models/btc.joblib")

    assert parsed.bucket == "model-artifacts"
    assert parsed.path == "models/btc.joblib"
    assert str(parsed) == "supabase://model-artifacts/models/btc.joblib"


def test_resolve_model_artifact_accepts_normalized_local_path(tmp_path: Path) -> None:
    artifact = tmp_path / "models" / "btc.joblib"
    artifact.parent.mkdir()
    artifact.write_text("ok", encoding="utf-8")

    resolved = resolve_model_artifact(str(artifact).replace("/", "\\"))

    assert resolved.exists()


def test_download_supabase_artifact_writes_cache_file(tmp_path: Path) -> None:
    session = FakeStorageSession()
    config = SupabaseConfig(url="https://example.supabase.co", key="test-key")

    resolved = download_supabase_artifact(
        "supabase://model-artifacts/models/btc.joblib",
        config=config,
        cache_dir=tmp_path,
        session=session,  # type: ignore[arg-type]
    )

    assert resolved.read_bytes() == b"model-bytes"
    assert session.get_calls[0]["url"] == "https://example.supabase.co/storage/v1/object/model-artifacts/models/btc.joblib"
    assert session.get_calls[0]["headers"]["Authorization"] == "Bearer test-key"


def test_upload_supabase_artifact_creates_bucket_and_uploads_bytes(tmp_path: Path) -> None:
    session = FakeStorageSession()
    config = SupabaseConfig(url="https://example.supabase.co", key="test-key")
    artifact = tmp_path / "btc.joblib"
    artifact.write_bytes(b"model-bytes")

    uri = upload_supabase_artifact(
        artifact,
        config=config,
        bucket="model-artifacts",
        object_path="models/btc.joblib",
        session=session,  # type: ignore[arg-type]
    )

    assert str(uri) == "supabase://model-artifacts/models/btc.joblib"
    assert session.post_calls[0]["url"] == "https://example.supabase.co/storage/v1/bucket"
    assert session.post_calls[0]["json"]["public"] is False
    assert session.post_calls[1]["url"] == "https://example.supabase.co/storage/v1/object/model-artifacts/models/btc.joblib"
    assert session.post_calls[1]["headers"]["x-upsert"] == "true"
    assert session.post_calls[1]["data"] == b"model-bytes"


def test_upload_supabase_artifact_uses_resumable_upload_for_large_files(tmp_path: Path) -> None:
    session = FakeStorageSession()
    config = SupabaseConfig(url="https://example.supabase.co", key="test-key")
    artifact = tmp_path / "large.joblib"
    artifact.write_bytes(b"abcdef")

    uri = upload_supabase_artifact(
        artifact,
        config=config,
        bucket="model-artifacts",
        object_path="models/large.joblib",
        resumable_threshold_bytes=1,
        session=session,  # type: ignore[arg-type]
    )

    assert str(uri) == "supabase://model-artifacts/models/large.joblib"
    assert session.post_calls[1]["url"] == "https://example.storage.supabase.co/storage/v1/upload/resumable"
    assert "bucketName" in session.post_calls[1]["headers"]["Upload-Metadata"]
    assert session.post_calls[1]["headers"]["Tus-Resumable"] == "1.0.0"
    assert session.patch_calls[0]["url"] == "https://example.storage.supabase.co/storage/v1/upload/resumable/upload-id"
    assert session.patch_calls[0]["headers"]["Upload-Offset"] == "0"
    assert session.patch_calls[0]["data"] == b"abcdef"
