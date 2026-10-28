"""Orkestrasi PDF → raw JSON Textract: cek kualitas → Blocks per halaman (sync atau async) → raw JSON."""
import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .... import progress
from ....config import settings
from .async_job import analyze_job
from .pages import split_pdf_pages
from .quality import check_effective_dpi
from .textract_api import TextractPageError, analyze_page, get_client

logger = logging.getLogger(__name__)

MODES = ("sync", "async")
SYNC_MAX_WORKERS = 5


def _analyze_pages(textract, file_path: str, n_pages: int, max_workers: int) -> list:
    """Kirim semua halaman ke Textract; return hasil per halaman sesuai urutan."""
    raw_pages = [None] * n_pages
    progress.activity(f"Textract sync: memproses {n_pages} halaman")

    if max_workers > 1:
        logger.info(f"[Textract] Concurrent processing dengan {max_workers} workers ({n_pages} halaman)")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(analyze_page, textract, page): page["page_idx"]
                for page in split_pdf_pages(file_path)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                raw_pages[idx] = future.result()  # analyze_page tidak pernah raise
                if raw_pages[idx]["error"]:
                    logger.warning(f"[Textract] Halaman {idx + 1}/{n_pages} selesai (ERROR: {raw_pages[idx]['error']})")
                else:
                    logger.info(f"[Textract] Halaman {idx + 1}/{n_pages} selesai")
    else:
        logger.info(f"[Textract] Sequential processing ({n_pages} halaman)")
        for page in split_pdf_pages(file_path):
            idx = page["page_idx"]
            raw_pages[idx] = analyze_page(textract, page)
            if raw_pages[idx]["error"]:
                logger.warning(f"[Textract] Halaman {idx + 1}/{n_pages} error: {raw_pages[idx]['error']}")
            else:
                logger.info(f"[Textract] Halaman {idx + 1}/{n_pages} selesai")

    progress.activity("Textract sync selesai")
    return raw_pages


async def parse_document_to_json(
    file_path: str,
    output_file: str = "raw_output.json",
    mode: str | None = None,
) -> dict:
    """Pipeline PDF -> raw Output JSON via Textract; raw JSON-nya berbentuk sama di kedua mode.

    ``mode`` (default ``TEXTRACT_MODE``) hanya menentukan cara Blocks per halaman diambil:
    ``sync`` = AnalyzeDocument per halaman secara paralel, ``async`` = satu job
    StartDocumentAnalysis lewat S3 (wajib satu region dengan bucket). ``output_file``
    hanya menentukan working directory: crop figure masuk ke ``images/`` di sebelahnya.
    """
    mode = mode or settings.textract_mode
    if mode not in MODES:
        raise ValueError(f"TEXTRACT_MODE harus 'sync' atau 'async', bukan {mode!r}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File tidak ditemukan: {file_path}")

    all_dpi = await asyncio.to_thread(check_effective_dpi, file_path)

    images_dir = os.path.join(os.path.dirname(os.path.abspath(output_file)), "images")
    os.makedirs(images_dir, exist_ok=True)

    if mode == "sync":
        raw_pages = await asyncio.to_thread(_analyze_pages, get_client(), file_path, len(all_dpi), SYNC_MAX_WORKERS)
    else:
        raw_pages = await analyze_job(file_path)

    raw_data = {
        "source_pdf": os.path.abspath(file_path),
        "images_dir": images_dir,
        "pages": raw_pages,
    }

    failed_pages = [p for p in raw_pages if p["error"]]
    if failed_pages:
        raise TextractPageError(failed_pages)
    return raw_data
