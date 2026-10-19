"""S3 for the Textract parser: the async job's input PDF and the figure crops."""
import hashlib
import logging
import mimetypes
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import boto3
from botocore.config import Config as BotoConfig
from ...config import settings

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_client():
    """The process-wide boto3 client, built on first use."""
    return boto3.client(
        "s3",
        region_name=settings.s3_region,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
        config=BotoConfig(signature_version="s3v4"),
    )

def require_bucket() -> str:
    """The configured bucket, or a clear error instead of a NoneType crash in boto3."""
    if not settings.s3_bucket:
        raise ValueError("S3_BUCKET belum diset — upload ke S3 tidak bisa jalan.")
    return settings.s3_bucket

def upload_fileobj(fileobj, key: str, content_type: str | None = None) -> None:
    get_client().upload_fileobj(
        fileobj,
        require_bucket(),
        key,
        ExtraArgs={"ContentType": content_type or "application/octet-stream"},
    )

def delete_object(key: str) -> None:
    get_client().delete_object(Bucket=require_bucket(), Key=key)

def object_url(key: str) -> str:
    """Public URL of one object. The VLM fetches figures from it directly."""
    return f"https://{require_bucket()}.s3.{settings.s3_region}.amazonaws.com/{key}"

def figure_key(file_path: str) -> str:
    """Object key of one figure crop, under ``S3_PREFIX``."""
    ext = Path(file_path).suffix.lower() or ".jpg"
    stamp = time.strftime("%y%m%d%H%M%S")
    short_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
    return f"{settings.s3_prefix.strip('/')}/{stamp}_{short_hash}{ext}"

def upload_figures_to_s3(items: list[dict], max_workers: int = 5) -> None:
    """Upload every figure in parallel and replace its ``img_path`` with the URL.

    All or nothing: when an upload still fails after boto3's own retries, no
    ``img_path`` is rewritten and the error is raised. The document then fails at
    parse and can be retried, instead of being stored with a local path that is
    gone once the tmp dir is cleaned up.
    """
    if not items:
        return
    def _put(path: str) -> str:
        key = figure_key(path)
        try:
            with open(path, "rb") as fh:
                upload_fileobj(fh, key, mimetypes.guess_type(path)[0])
        except Exception as e:
            raise RuntimeError(f"Upload gambar ke S3 gagal: {Path(path).name}") from e
        return object_url(key)
    get_client()
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        urls = list(pool.map(_put, [item["img_path"] for item in items]))
    for item, url in zip(items, urls):
        item["img_path"] = url
    logger.info(f"[S3] {len(items)} figure diupload")
