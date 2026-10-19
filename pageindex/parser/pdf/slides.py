"""PDF slide deck → satu item teks per slide, ditulis VLM dari gambar slide utuh.

Parser teks meratakan tata letak slide: tanggal lepas dari langkah timeline-nya,
legenda warna hilang, teks di logo tidak terbaca. VLM melihat slide utuh, dengan
text layer PDF sebagai referensi ejaan. Hasilnya item ``text`` biasa, jadi ia
lewat jalur teks (chunking, ekstraksi KG, sitasi halaman), bukan jalur
multimodal. ``img_path`` item hanya membawa gambar slide ke S3 (``host_figures``)
dan ke hasil retrieval; isinya tidak pernah masuk ke teks chunk.
"""
import asyncio
import logging
import re
from pathlib import Path

import pymupdf

from hybrid_rag.config import settings
from hybrid_rag.ingestion.multimodal.content import encode_image_to_base64
from hybrid_rag.ingestion.multimodal.chain import slide_transcription_chain
from hybrid_rag.ingestion.parser.figures import FIGURE_MAX_PIXELS, save_for_vlm
from hybrid_rag.ingestion.parser.pdf.amazon_textract.pages import render_page_image

logger = logging.getLogger(__name__)

# shortcut: 16:10 dan 16:9 (≥1.5) saja; dek 4:3 (1.33) tetap ke Textract, tambah override eksplisit bila sering muncul.
# A4 landscape (1.41) dan Letter landscape (1.29) sengaja di bawah ambang.
SLIDE_MIN_ASPECT = 1.5
SLIDE_MIN_SHARE = 0.8
MIN_COVERAGE = 0.8


def is_slide_deck(path: str | Path) -> bool:
    """Sebagian besar halaman berbentuk layar lebar → PDF ini ekspor presentasi.

    Metadata (creator/producer) tidak dipakai: dek hasil iLovePDF atau Canva
    tidak menyebut PowerPoint di sana. PDF yang tidak terbaca PyMuPDF dianggap
    bukan dek, supaya jalur Textract tetap yang menangani (dan melaporkan) PDF itu.
    """
    try:
        doc = pymupdf.open(path)
    except pymupdf.FileDataError:
        return False
    with doc:
        if not doc.page_count:
            return False
        wide = sum(page.rect.width / page.rect.height >= SLIDE_MIN_ASPECT for page in doc)
        return wide / doc.page_count >= SLIDE_MIN_SHARE


def _coverage(reference: str, transcript: str) -> float:
    """Porsi kata text layer (> 3 huruf) yang ikut tertulis di transkripsi."""
    words = set(re.findall(r"\w{4,}", reference.lower()))
    if not words:
        return 1.0
    found = set(re.findall(r"\w{4,}", transcript.lower()))
    return len(words & found) / len(words)


async def parse_slides(path: Path, tmp_dir: Path) -> list[dict]:
    """Satu item ``text`` per slide, berurutan, dengan ``page_idx`` dan gambar slidenya."""
    with pymupdf.open(path) as doc:
        references = [page.get_text(sort=True).strip() for page in doc]
        # Render langsung ke FIGURE_MAX_PIXELS di sisi panjang: teks kecil (spanduk di foto) masih terbaca.
        dpis = [FIGURE_MAX_PIXELS * 72 / max(page.rect.width, page.rect.height) for page in doc]

    semaphore = asyncio.Semaphore(max(1, settings.vlm_max_async))
    chain = slide_transcription_chain()

    async def _transcribe(page_idx: int) -> dict:
        image = await asyncio.to_thread(render_page_image, str(path), page_idx, dpis[page_idx], False)
        image_path = await asyncio.to_thread(save_for_vlm, image, tmp_dir / f"slide_{page_idx}.jpg")
        async with semaphore:
            text = await chain.ainvoke({
                "page_number": page_idx + 1,
                "reference_text": references[page_idx] or "(kosong)",
                "image_url": f"data:image/jpeg;base64,{encode_image_to_base64(str(image_path))}",
            })
        text = (text or "").strip()
        # Gagal keras, bukan jatuh ke teks mentah: slide tanpa transkripsi berarti isinya hilang diam-diam.
        if not text:
            raise RuntimeError(f"{path.name}: slide {page_idx + 1} tidak menghasilkan transkripsi")
        coverage = _coverage(references[page_idx], text)
        if coverage < MIN_COVERAGE:
            logger.warning(f"[slides] {path.name} slide {page_idx + 1}: hanya {coverage:.0%} kata text layer ikut tertulis")
        return {"type": "text", "text": text, "page_idx": page_idx, "img_path": str(image_path)}

    logger.info(f"[slides] {path.name}: {len(references)} slide → VLM")
    return list(await asyncio.gather(*(_transcribe(i) for i in range(len(references)))))
