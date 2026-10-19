"""Panggilan AWS Textract: client bersama + AnalyzeDocument per halaman."""
import io
import logging
import threading

import boto3
from botocore.config import Config

from ....config import settings
from .pages import render_page_image

logger = logging.getLogger(__name__)

FEATURE_TYPES = ["LAYOUT", "TABLES"]

# Batas sync AnalyzeDocument: 10 MB per dokumen (Set Quotas in Amazon Textract).
TEXTRACT_SYNC_MAX_BYTES = 10 * 1024 * 1024


class TextractPageError(Exception):
    """Raised ketika satu atau lebih halaman gagal di Textract (setelah retry habis).

    Dokumen digagalkan utuh: melewati halaman diam-diam membuat isi dokumen
    hilang dari RAG padahal statusnya sukses.
    """
    def __init__(self, failed_pages: list[dict]):
        self.failed_pages = failed_pages
        page_list = "; ".join(f"halaman {p['page_idx']+1}: {p['error']}" for p in failed_pages)
        super().__init__(f"Textract gagal pada {len(failed_pages)} halaman: {page_list}")


_client_lock = threading.Lock()
_client = None


def get_client():
    """Satu client Textract per proses (client boto3 thread-safe, pembuatannya tidak)."""
    global _client
    with _client_lock:
        if _client is None:
            _client = boto3.client(
                "textract",
                aws_access_key_id=settings.textract_access_key_id,
                aws_secret_access_key=settings.textract_secret_access_key,
                region_name=settings.textract_region,
                config=Config(
                    retries={"mode": "adaptive", "max_attempts": 8},
                    max_pool_connections=50,
                ),
            )
        return _client


def page_document_bytes(page: dict) -> bytes:
    """PDF satu halaman; fallback ke PNG render bila melewati batas sync 10 MB."""
    if len(page["doc_bytes"]) <= TEXTRACT_SYNC_MAX_BYTES:
        return page["doc_bytes"]
    logger.warning(f"[Textract] Halaman {page['page_idx'] + 1} PDF >10 MB, fallback ke PNG render")
    buf = io.BytesIO()
    # Gunakan enhance=False pada fallback bila ingin karakter piksel sedekat mungkin dengan PDF asli
    render_page_image(page["source_pdf"], page["page_idx"]).save(buf, "PNG")
    return buf.getvalue()


def analyze_page(textract_client, page: dict) -> dict:
    """Proses satu halaman: panggil Textract lalu simpan hasil + metadata. Tidak pernah raise.

    Struktur dict & validasi BlockType == 'PAGE' disamakan persis dengan _raw_pages (async).
    """
    try:
        response = textract_client.analyze_document(
            Document={"Bytes": page_document_bytes(page)},
            FeatureTypes=FEATURE_TYPES,
        )
        blocks = response.get("Blocks", [])
        
        # Samakan dengan validasi di _raw_pages (Async):
        if not any(b.get("BlockType") == "PAGE" for b in blocks):
            error = "tidak ada hasil Textract untuk halaman ini"
            blocks = []
        else:
            error = None

        result = {
            "page_idx": page["page_idx"],
            "img_w": page["img_w"],
            "img_h": page["img_h"],
            "DocumentMetadata": response.get("DocumentMetadata"),
            "AnalyzeDocumentModelVersion": response.get("AnalyzeDocumentModelVersion"),
            "Blocks": blocks,
            "error": error,
        }
    except Exception as e:
        result = {
            "page_idx": page["page_idx"],
            "img_w": page["img_w"],
            "img_h": page["img_h"],
            "DocumentMetadata": None,
            "AnalyzeDocumentModelVersion": None,
            "Blocks": [],
            "error": str(e),
        }
    finally:
        page["doc_bytes"] = None
    return result