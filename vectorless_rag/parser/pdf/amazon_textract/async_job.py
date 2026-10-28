"""Textract mode async: satu job StartDocumentAnalysis lewat S3 + polling GetDocumentAnalysis.

Menghasilkan halaman berbentuk sama dengan ``analyze_page`` (sync); dipanggil dari ``parse``.
"""
import asyncio
import io
import random
import time
import uuid
import pymupdf as fitz
from ....config import settings
import logging
from .... import progress
from ....storage.s3 import delete_object, require_bucket, upload_fileobj
from .pages import RENDER_DPI
from .textract_api import FEATURE_TYPES, get_client

logger = logging.getLogger(__name__)

POLL_FIRST_S = 2.0
POLL_MAX_S = 20.0  # 50 job / 20 s = 2,5 TPS polling; sisa kuota Get 5 TPS untuk ambil hasil
POLL_TIMEOUT_S = 30 * 60


def _normalized_pdf(pdf_path: str, dpi: int = RENDER_DPI) -> tuple[bytes, list[tuple[int, int]]]:
    """PDF utuh dengan /Rotate dibakar ke konten + ukuran piksel render ``dpi`` per halaman.

    Sama dengan ``split_pdf_pages``: bbox Textract = orientasi render yang dipakai crop
    figure, dan ``img_w``/``img_h`` memakai rumus yang identik.
    """
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    try:
        sizes = []
        for page in doc:
            irect = (page.rect * mat).irect  # dihitung sebelum rotasi dibakar, seperti sync
            sizes.append((irect.width, irect.height))
            page.remove_rotation()
        return doc.tobytes(garbage=3, deflate=True), sizes
    finally:
        doc.close()


_job_slots: asyncio.Semaphore | None = None


def _slots() -> asyncio.Semaphore:
    """Semaphore job Textract per proses, dibuat saat pertama dipakai (nilai dari settings)."""
    global _job_slots
    if _job_slots is None:
        _job_slots = asyncio.Semaphore(settings.textract_max_concurrent_jobs)
    return _job_slots


async def _wait_for_job(textract, job_id: str) -> dict:
    """Polling sampai job tidak ``IN_PROGRESS``; return respons Get pertama (halaman hasil 1).

    ponytail: polling memakan kuota Get (5 TPS default di ap-southeast-1) — cukup sampai
    ~50 job bersamaan; di atas itu naikkan kuota Get atau ganti SNS→SQS.
    """
    delay, deadline = POLL_FIRST_S, time.monotonic() + POLL_TIMEOUT_S
    while True:
        response = await asyncio.to_thread(textract.get_document_analysis, JobId=job_id, MaxResults=1000)
        if response["JobStatus"] != "IN_PROGRESS":
            return response
        if time.monotonic() > deadline:
            raise TimeoutError(f"Job Textract {job_id} belum selesai setelah {POLL_TIMEOUT_S} detik")
        # Jitter: job yang dimulai bersamaan tidak polling serempak.
        await asyncio.sleep(delay * random.uniform(0.8, 1.2))
        delay = min(delay * 1.5, POLL_MAX_S)


def _collect_blocks(textract, job_id: str, first: dict) -> list:
    """Kumpulkan semua Blocks lewat ``NextToken`` (pola contoh Python dokumentasi AWS)."""
    blocks, response = list(first.get("Blocks", [])), first
    while response.get("NextToken"):
        response = textract.get_document_analysis(JobId=job_id, MaxResults=1000, NextToken=response["NextToken"])
        blocks.extend(response.get("Blocks", []))
    return blocks


def _raw_pages(first: dict, blocks: list, sizes: list) -> list:
    """Blocks satu job → daftar per halaman berbentuk sama dengan ``analyze_page`` sync."""
    by_page: dict[int, list] = {}
    for b in blocks:
        by_page.setdefault(b.get("Page", 1) - 1, []).append(b)

    status = first["JobStatus"]
    warned = {p - 1: w.get("ErrorCode") for w in first.get("Warnings") or [] for p in w.get("Pages") or []}
    raw_pages = []
    for idx, (img_w, img_h) in enumerate(sizes):
        page_blocks = by_page.get(idx, [])
        if status == "FAILED":
            error = first.get("StatusMessage") or "Job Textract FAILED"
        elif idx in warned:
            error = warned[idx] or "PARTIAL_SUCCESS"
        elif not any(b["BlockType"] == "PAGE" for b in page_blocks):
            error = "tidak ada hasil Textract untuk halaman ini"
        else:
            error = None
        raw_pages.append({
            "page_idx": idx,
            "img_w": img_w,
            "img_h": img_h,
            "DocumentMetadata": first.get("DocumentMetadata"),
            "AnalyzeDocumentModelVersion": first.get("AnalyzeDocumentModelVersion"),
            "Blocks": [] if error else page_blocks,
            "error": error,
        })
    return raw_pages


async def _run_job(pdf_bytes: bytes, n_pages: int) -> tuple[dict, list]:
    """Upload → Start → polling → kumpulkan Blocks; input S3 selalu dihapus."""
    token = uuid.uuid4().hex
    key = f"{settings.s3_prefix.strip('/')}/textract-input/{token}.pdf"
    await asyncio.to_thread(upload_fileobj, io.BytesIO(pdf_bytes), key, "application/pdf")
    try:
        textract = get_client()
        job_id = (await asyncio.to_thread(
            textract.start_document_analysis,
            DocumentLocation={"S3Object": {"Bucket": require_bucket(), "Name": key}},
            FeatureTypes=FEATURE_TYPES,
            ClientRequestToken=token,  # retry boto atas Start tidak memulai job ganda
        ))["JobId"]
        logger.info(f"[Textract] Job async {job_id} dimulai ({n_pages} halaman)")
        progress.activity(f"Textract async: memproses {n_pages} halaman")
        first = await _wait_for_job(textract, job_id)
        blocks = await asyncio.to_thread(_collect_blocks, textract, job_id, first) if first["JobStatus"] != "FAILED" else []
        logger.info(f"[Textract] Job {job_id} {first['JobStatus']}: {len(blocks)} blocks")
        progress.activity(f"Textract async selesai: {first['JobStatus']}")
        return first, blocks
    finally:
        try:
            await asyncio.to_thread(delete_object, key)
        except Exception:
            logger.warning(f"[Textract] Gagal menghapus input S3 {key}", exc_info=True)


async def analyze_job(file_path: str) -> list:
    """Blocks per halaman lewat satu job async; antre bila slot job Textract penuh."""
    if settings.textract_region != settings.resolved_s3_region:
        raise ValueError(
            f"Textract async wajib satu region dengan bucket S3: "
            f"TEXTRACT_REGION={settings.textract_region}, S3_REGION={settings.resolved_s3_region}"
        )
    pdf_bytes, sizes = await asyncio.to_thread(_normalized_pdf, file_path)
    slots = _slots()
    if slots.locked():
        progress.activity("Menunggu antrean Textract")
    async with slots:
        first, blocks = await _run_job(pdf_bytes, len(sizes))
    return _raw_pages(first, blocks, sizes)
