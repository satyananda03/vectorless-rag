"""Navigasi graf Block Textract + geometri bounding box (koordinat ternormalisasi 0-1)."""


def build_block_map(blocks: list) -> dict:
    return {b["Id"]: b for b in blocks}


def get_children(block: dict, block_map: dict, rel_type: str = "CHILD") -> list:
    ids = []
    for rel in block.get("Relationships", []):
        if rel["Type"] == rel_type:
            ids.extend(rel["Ids"])
    return [block_map[i] for i in ids if i in block_map]


def get_plain_text(block: dict, block_map: dict) -> str:
    """Rekursive Loop sampai WORD/LINE, untuk merangkai teks."""
    if block["BlockType"] in ("WORD", "LINE"):
        return block.get("Text", "")
    parts = []
    for child in get_children(block, block_map):
        txt = get_plain_text(child, block_map)
        if txt:
            parts.append(txt)
    return " ".join(parts).strip()


def get_word_blocks(block: dict, block_map: dict) -> list:
    """Rekursive Loop mengumpulkan semua block WORD dari suatu parent block."""
    if block["BlockType"] == "WORD":
        return [block]
    words = []
    for child in get_children(block, block_map):
        words.extend(get_word_blocks(child, block_map))
    return words


def is_vertical_text(block: dict, block_map: dict, angle_threshold: float = 45.0, ratio_threshold: float = 0.5) -> bool:
    words = get_word_blocks(block, block_map)
    if not words:
        return False
    rotated_count = sum(1 for w in words if abs(w.get("Geometry", {}).get("RotationAngle", 0.0)) > angle_threshold)
    return (rotated_count / len(words)) > ratio_threshold


def bbox_to_pixels(bbox: dict, img_w: int, img_h: int):
    left = bbox["Left"] * img_w
    top = bbox["Top"] * img_h
    width = bbox["Width"] * img_w
    height = bbox["Height"] * img_h
    return left, top, left + width, top + height


def bbox_iou(b1: dict, b2: dict) -> float:
    ax1, ay1 = b1["Left"], b1["Top"]
    ax2, ay2 = ax1 + b1["Width"], ay1 + b1["Height"]
    bx1, by1 = b2["Left"], b2["Top"]
    bx2, by2 = bx1 + b2["Width"], by1 + b2["Height"]

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = b1["Width"] * b1["Height"]
    area_b = b2["Width"] * b2["Height"]
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def bbox_overlap_ratio(bbox1: dict, bbox2: dict) -> float:
    """Menghitung seberapa persen sebuah bbox1 tumpang tindih dengan bbox2."""
    ax1, ay1 = bbox1["Left"], bbox1["Top"]
    ax2, ay2 = ax1 + bbox1["Width"], ay1 + bbox1["Height"]
    bx1, by1 = bbox2["Left"], bbox2["Top"]
    bx2, by2 = bx1 + bbox2["Width"], by1 + bbox2["Height"]

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = bbox1["Width"] * bbox1["Height"]
    return inter_area / area1 if area1 > 0 else 0.0
