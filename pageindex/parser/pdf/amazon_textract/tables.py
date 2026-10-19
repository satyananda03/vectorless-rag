"""TABLE Textract → markdown, caption & footnote."""
from .blocks import bbox_overlap_ratio, get_children, get_plain_text


def build_table_markdown(table_block: dict, block_map: dict) -> str:
    cell_blocks = [
        c for c in get_children(table_block, block_map)
        if c["BlockType"] in ("CELL", "MERGED_CELL")
    ]
    if not cell_blocks:
        return ""

    max_row = max(c.get("RowIndex", 1) + c.get("RowSpan", 1) - 1 for c in cell_blocks)
    max_col = max(c.get("ColumnIndex", 1) + c.get("ColumnSpan", 1) - 1 for c in cell_blocks)
    grid = [["" for _ in range(max_col)] for _ in range(max_row)]

    for c in cell_blocks:
        r = c.get("RowIndex", 1) - 1
        col = c.get("ColumnIndex", 1) - 1
        if 0 <= r < max_row and 0 <= col < max_col:
            text = get_plain_text(c, block_map).replace("|", "/").replace("\n", " ")
            entity_types = c.get("EntityTypes", [])
            if not text:
                pass  # sel kosong: tanpa penanda bold, bukan "****"
            elif "TABLE_SECTION_TITLE" in entity_types:
                # Penanda Bold untuk Table Section Title
                text = f"**{text}**"
            elif "COLUMN_HEADER" in entity_types:
                # Penanda Bold untuk Header Column
                text = f"**{text}**"
            grid[r][col] = text
    lines = []
    lines.append("| " + " | ".join(grid[0]) + " |")
    lines.append("|" + "|".join(["---"] * max_col) + "|")
    for row in grid[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def get_table_caption_footnote(table_block: dict, block_map: dict, header_bboxes: list = None, footer_bboxes: list = None) -> tuple:
    """Ekstrak caption & footnote dari TABLE block. Buang caption/footnote yang >50% berada di area header/footer."""
    captions, footnotes = [], []
    header_bboxes = header_bboxes or []
    footer_bboxes = footer_bboxes or []

    for rel in table_block.get("Relationships", []):
        if rel["Type"] == "TABLE_TITLE":
            for bid in rel["Ids"]:
                if bid in block_map:
                    cap_block = block_map[bid]
                    cap_bbox = cap_block["Geometry"]["BoundingBox"]
                    is_header = any(
                        bbox_overlap_ratio(cap_bbox, h_bbox) > 0.5
                        for h_bbox in header_bboxes
                    )
                    if not is_header:
                        captions.append(get_plain_text(cap_block, block_map))

        elif rel["Type"] == "TABLE_FOOTER":
            for bid in rel["Ids"]:
                if bid in block_map:
                    foot_block = block_map[bid]
                    foot_bbox = foot_block["Geometry"]["BoundingBox"]
                    is_footer = any(
                        bbox_overlap_ratio(foot_bbox, f_bbox) > 0.5
                        for f_bbox in footer_bboxes
                    )
                    if not is_footer:
                        footnotes.append(get_plain_text(foot_block, block_map))
    return captions, footnotes
