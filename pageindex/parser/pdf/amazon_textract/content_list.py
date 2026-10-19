"""Raw JSON Textract → content list (text / image / table) untuk chunker."""
import functools
import json
import os
import uuid
import warnings

from ...figures import is_large_enough, save_for_vlm
from .blocks import (
    bbox_iou,
    bbox_to_pixels,
    build_block_map,
    get_children,
    get_plain_text,
    is_vertical_text,
)
from .pages import render_page_image
from .tables import build_table_markdown, get_table_caption_footnote

LAYOUT_TEXT_TYPES = {
    "LAYOUT_TEXT",
    "LAYOUT_TITLE",
    "LAYOUT_SECTION_HEADER",
    "LAYOUT_LIST",
    "LAYOUT_KEY_VALUE",
}

IGNORED_LAYOUT_TYPES = {
    "LAYOUT_HEADER",
    "LAYOUT_FOOTER",
    "LAYOUT_PAGE_NUMBER",
}

HEADING_LAYOUT_LEVELS = {
    "LAYOUT_TITLE": 1,
    "LAYOUT_SECTION_HEADER": 2,
}

FIGURE_CROP_PAD_PX = 3

def _text_item(block: dict, block_map: dict, page_idx: int) -> dict | None:
    """LAYOUT teks → item ``text``; None untuk teks vertikal / kosong."""
    if is_vertical_text(block, block_map):
        return None
    text = get_plain_text(block, block_map)
    if not text:
        return None
    item = {
        "type": "text",
        "text": text,
        "page_idx": page_idx,
    }
    # Heading ditandai, bukan dipisahkan: blocknya tetap ikut
    # sebagai teks biasa di ``LAYOUT_TEXT_TYPES`` supaya body
    # dokumen utuh. ``text_level`` cuma metadata tambahan —
    # ``separate_content`` mengabaikannya, jadi teks yang
    # sampai ke chunker sama persis seperti sebelumnya.
    level = HEADING_LAYOUT_LEVELS.get(block["BlockType"])
    if level:
        item["text_level"] = level
    return item


def _figure_item(bbox: dict, page_idx: int, img_w: int, img_h: int, page_image, images_dir: str) -> dict | None:
    """LAYOUT_FIGURE → crop JPEG + item ``image``; None bila figure terlalu kecil.

    ``page_image`` dipanggil (lazy, di-cache per halaman) hanya untuk figure yang lolos filter.
    """
    area_ratio = bbox["Width"] * bbox["Height"]
    left, top, right, bottom = bbox_to_pixels(bbox, img_w, img_h)
    crop_w_px, crop_h_px = right - left, bottom - top
    if not is_large_enough(crop_w_px, crop_h_px, area_ratio):
        return None
    pad = FIGURE_CROP_PAD_PX
    crop_box = (
        max(0, left - pad), max(0, top - pad),
        min(img_w, right + pad), min(img_h, bottom + pad)
    )
    fname = f"figure_p{page_idx}_{uuid.uuid4().hex[:8]}.jpg"
    fpath = os.path.join(images_dir, fname) if images_dir else fname
    if images_dir:
        os.makedirs(images_dir, exist_ok=True)
    fpath = save_for_vlm(page_image().crop(crop_box), fpath)
    return {
        "type": "image",
        "img_path": os.path.abspath(fpath),
        "image_caption": None,
        "image_footnote": None,
        "page_idx": page_idx,
    }


def _table_item(block: dict, bbox: dict, table_blocks: list, used_table_ids: set, block_map: dict,
                header_bboxes: list, footer_bboxes: list, page_idx: int) -> dict:
    """LAYOUT_TABLE → item ``table``: pasangkan ke TABLE dengan IoU terbaik (>0.3), else teks polos."""
    best_table, best_iou = None, 0.0
    for t in table_blocks:
        if t["Id"] in used_table_ids: continue
        iou = bbox_iou(bbox, t["Geometry"]["BoundingBox"])
        if iou > best_iou:
            best_iou, best_table = iou, t
    if best_table is not None and best_iou > 0.3:
        used_table_ids.add(best_table["Id"])
        table_md = build_table_markdown(best_table, block_map)
        captions, footnotes = get_table_caption_footnote(best_table, block_map, header_bboxes, footer_bboxes)
    else:
        table_md = get_plain_text(block, block_map)
        captions, footnotes = [], []

    return {
        "type": "table",
        "table_body": table_md,
        "table_caption": captions if captions else None,
        "table_footnote": footnotes if footnotes else None,
        "page_idx": page_idx,
    }


def _page_items(page_data: dict, data: dict, images_dir: str) -> list:
    """Satu halaman raw Textract → item content list sesuai urutan baca PAGE."""
    page_idx = page_data["page_idx"]
    if page_data.get("error"):
        warnings.warn(f"Halaman {page_idx} di-skip (error: {page_data['error']})")
        return []
    blocks = page_data["Blocks"]
    if not blocks:
        return []
    img_w = page_data["img_w"]
    img_h = page_data["img_h"]
    block_map = build_block_map(blocks)
    page_block = next((b for b in blocks if b["BlockType"] == "PAGE"), None)
    if page_block is None:
        return []

    ordered_children = get_children(page_block, block_map)
    header_bboxes = [b["Geometry"]["BoundingBox"] for b in ordered_children if b["BlockType"] == "LAYOUT_HEADER"]
    footer_bboxes = [b["Geometry"]["BoundingBox"] for b in ordered_children if b["BlockType"] == "LAYOUT_FOOTER"]
    table_blocks = [b for b in blocks if b["BlockType"] == "TABLE"]
    used_table_ids = set()
    # Render hanya halaman yang punya figure lolos filter (piksel = render lama).
    # enhance=False: crop untuk VLM tetap berwarna, tanpa halo unsharp (lihat render_page_image).
    page_image = functools.cache(lambda: render_page_image(data["source_pdf"], page_idx, enhance=False).convert("RGB"))

    items = []
    for block in ordered_children:
        btype = block["BlockType"]
        bbox = block["Geometry"]["BoundingBox"]
        if btype in IGNORED_LAYOUT_TYPES:
            continue
        if btype in LAYOUT_TEXT_TYPES:
            item = _text_item(block, block_map, page_idx)
        elif btype == "LAYOUT_FIGURE":
            item = _figure_item(bbox, page_idx, img_w, img_h, page_image, images_dir)
        elif btype == "LAYOUT_TABLE":
            item = _table_item(block, bbox, table_blocks, used_table_ids, block_map,
                               header_bboxes, footer_bboxes, page_idx)
        else:
            item = None
        if item:
            items.append(item)
    return items


def reconstruct_content_list(data: dict, output_file: str = "formatted_output.json") -> list:
    """Format raw JSON output Textract menjadi content list.

    Gambar figure di-crop ke ``data["images_dir"]``; ``img_path`` masih berupa file
    lokal. Mengunggahnya ke S3 dikerjakan ``parse_file`` untuk semua parser.
    """
    images_dir = data.get("images_dir", "")
    content_list = []
    for page_data in data.get("pages", []):
        content_list.extend(_page_items(page_data, data, images_dir))

    if output_file is None:
        if images_dir:
            tmp_dir = os.path.dirname(images_dir)
            output_file = os.path.join(tmp_dir, "formatted_parsing_result.json")
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(content_list, f, ensure_ascii=False, indent=2, default=str)
        print(f"[Textract] Formatted JSON disimpan ke: {output_file}")
    return content_list
