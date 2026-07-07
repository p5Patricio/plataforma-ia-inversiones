from __future__ import annotations

from dataclasses import dataclass
import base64
from pathlib import Path
from tempfile import gettempdir
from urllib.parse import quote, urljoin, urlparse

import requests

from collector.supabase_repository import SupabaseConfig


DEFAULT_MODEL_ARTIFACT_BUCKET = "model-artifacts"
RESUMABLE_UPLOAD_THRESHOLD_BYTES = 6 * 1024 * 1024
TUS_CHUNK_SIZE_BYTES = 6 * 1024 * 1024
TUS_VERSION = "1.0.0"


@dataclass(frozen=True)
class SupabaseArtifactUri:
    bucket: str
    path: str

    def __str__(self) -> str:
        return f"supabase://{self.bucket}/{self.path}"


def is_supabase_artifact_uri(uri: str) -> bool:
    return uri.startswith("supabase://")


def parse_supabase_artifact_uri(uri: str) -> SupabaseArtifactUri:
    if not is_supabase_artifact_uri(uri):
        raise ValueError(f"Not a Supabase artifact URI: {uri}")

    raw = uri.removeprefix("supabase://")
    bucket, separator, path = raw.partition("/")
    if not bucket or not separator or not path:
        raise ValueError("Supabase artifact URI must look like supabase://bucket/path")
    return SupabaseArtifactUri(bucket=bucket, path=path.strip("/"))


def resolve_model_artifact(
    artifact_uri: str,
    config: SupabaseConfig | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    if is_supabase_artifact_uri(artifact_uri):
        return download_supabase_artifact(
            artifact_uri,
            config=config or SupabaseConfig.from_env(),
            cache_dir=cache_dir,
        )

    path = Path(artifact_uri)
    if path.exists():
        return path

    normalized_path = Path(artifact_uri.replace("\\", "/"))
    if normalized_path.exists():
        return normalized_path

    raise ValueError(f"artifact_not_found:{artifact_uri}")


def download_supabase_artifact(
    artifact_uri: str,
    config: SupabaseConfig,
    cache_dir: str | Path | None = None,
    session: requests.Session | None = None,
) -> Path:
    parsed = parse_supabase_artifact_uri(artifact_uri)
    target_dir = Path(cache_dir) if cache_dir else Path(gettempdir()) / "ia-inversiones-model-artifacts"
    target_path = target_dir / parsed.bucket / parsed.path
    if target_path.exists():
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    client = session or requests.Session()
    response = client.get(
        storage_object_url(config, parsed.bucket, parsed.path),
        headers=storage_headers(config),
        timeout=60,
    )
    response.raise_for_status()
    target_path.write_bytes(response.content)
    return target_path


def upload_supabase_artifact(
    local_path: str | Path,
    config: SupabaseConfig,
    bucket: str = DEFAULT_MODEL_ARTIFACT_BUCKET,
    object_path: str | None = None,
    create_bucket: bool = True,
    upsert: bool = True,
    resumable_threshold_bytes: int = RESUMABLE_UPLOAD_THRESHOLD_BYTES,
    session: requests.Session | None = None,
) -> SupabaseArtifactUri:
    artifact_path = Path(local_path)
    if not artifact_path.exists():
        raise ValueError(f"artifact_not_found:{artifact_path}")

    client = session or requests.Session()
    if create_bucket:
        ensure_artifact_bucket(config=config, bucket=bucket, session=client)

    normalized_object_path = (object_path or artifact_path.name).replace("\\", "/").strip("/")
    if artifact_path.stat().st_size > resumable_threshold_bytes:
        upload_supabase_artifact_resumable(
            artifact_path,
            config=config,
            bucket=bucket,
            object_path=normalized_object_path,
            upsert=upsert,
            session=client,
        )
        return SupabaseArtifactUri(bucket=bucket, path=normalized_object_path)

    response = client.post(
        storage_object_url(config, bucket, normalized_object_path),
        headers=storage_headers(config)
        | {
            "Content-Type": "application/octet-stream",
            "x-upsert": "true" if upsert else "false",
        },
        data=artifact_path.read_bytes(),
        timeout=120,
    )
    response.raise_for_status()
    return SupabaseArtifactUri(bucket=bucket, path=normalized_object_path)


def upload_supabase_artifact_resumable(
    local_path: Path,
    config: SupabaseConfig,
    bucket: str,
    object_path: str,
    upsert: bool = True,
    session: requests.Session | None = None,
) -> None:
    client = session or requests.Session()
    upload_url = create_resumable_upload_url(
        client=client,
        config=config,
        bucket=bucket,
        object_path=object_path,
        size_bytes=local_path.stat().st_size,
        upsert=upsert,
    )
    offset = 0
    with local_path.open("rb") as artifact:
        while chunk := artifact.read(TUS_CHUNK_SIZE_BYTES):
            response = client.patch(
                upload_url,
                headers=storage_headers(config)
                | {
                    "Tus-Resumable": TUS_VERSION,
                    "Upload-Offset": str(offset),
                    "Content-Type": "application/offset+octet-stream",
                },
                data=chunk,
                timeout=120,
            )
            response.raise_for_status()
            offset += len(chunk)


def create_resumable_upload_url(
    client: requests.Session,
    config: SupabaseConfig,
    bucket: str,
    object_path: str,
    size_bytes: int,
    upsert: bool = True,
) -> str:
    response = client.post(
        resumable_upload_endpoint(config),
        headers=storage_headers(config)
        | {
            "Tus-Resumable": TUS_VERSION,
            "Upload-Length": str(size_bytes),
            "Upload-Metadata": encode_tus_metadata(
                {
                    "bucketName": bucket,
                    "objectName": object_path,
                    "contentType": "application/octet-stream",
                    "cacheControl": "3600",
                }
            ),
            "x-upsert": "true" if upsert else "false",
        },
        timeout=30,
    )
    response.raise_for_status()
    location = response.headers.get("Location")
    if not location:
        raise RuntimeError("Supabase resumable upload did not return a Location header")
    return urljoin(resumable_upload_endpoint(config), location)


def ensure_artifact_bucket(
    config: SupabaseConfig,
    bucket: str = DEFAULT_MODEL_ARTIFACT_BUCKET,
    session: requests.Session | None = None,
) -> None:
    client = session or requests.Session()
    response = client.post(
        f"{config.url}/storage/v1/bucket",
        headers=storage_headers(config) | {"Content-Type": "application/json"},
        json={"id": bucket, "name": bucket, "public": False},
        timeout=30,
    )
    if response.status_code in {200, 201, 409}:
        return
    if response.status_code == 400 and "already" in response.text.lower():
        return
    response.raise_for_status()


def storage_headers(config: SupabaseConfig) -> dict[str, str]:
    return {
        "apikey": config.key,
        "Authorization": f"Bearer {config.key}",
    }


def storage_object_url(config: SupabaseConfig, bucket: str, path: str) -> str:
    safe_path = quote(path.strip("/"), safe="/")
    return f"{config.url}/storage/v1/object/{bucket}/{safe_path}"


def resumable_upload_endpoint(config: SupabaseConfig) -> str:
    hostname = urlparse(config.url).hostname
    if hostname and hostname.endswith(".supabase.co"):
        project_ref = hostname.split(".")[0]
        return f"https://{project_ref}.storage.supabase.co/storage/v1/upload/resumable"
    return f"{config.url}/storage/v1/upload/resumable"


def encode_tus_metadata(metadata: dict[str, str]) -> str:
    return ",".join(
        f"{key} {base64.b64encode(value.encode('utf-8')).decode('ascii')}" for key, value in metadata.items()
    )
