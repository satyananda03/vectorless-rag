from io import BytesIO
from pathlib import Path
from PIL import Image
from pageindex.parser.office.docx import Document
from docx.oxml.ns import qn
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph
from unstructured.partition.docx import partition_docx
from unstructured.documents.elements import Table as UnstructuredTable
from html_table_rescuer import TableParser

from hybrid_rag.ingestion.parser.figures import is_large_image, save_for_vlm

# ============================================================
# NAMESPACE
# ============================================================
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"

# ============================================================
# FOOTNOTE / ENDNOTE
# ============================================================
def get_footnotes_map(doc):
    footnotes = {}
    try:
        part = doc.part.package.part_related_by(
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footnotes"
        )
        root = part.element
        ns = {"w": W_NS}
        for fn in root.findall(".//w:footnote", ns):
            fn_id = fn.get(qn("w:id"))
            if fn_id in ("-1", "0"):
                continue
            texts = [t.text or "" for t in fn.findall(".//w:t", ns)]
            footnotes[fn_id] = " ".join(texts).strip()
    except Exception:
        pass
    return footnotes


def get_endnotes_map(doc):
    endnotes = {}
    try:
        part = doc.part.package.part_related_by(
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/endnotes"
        )
        root = part.element
        ns = {"w": W_NS}
        for en in root.findall(".//w:endnote", ns):
            en_id = en.get(qn("w:id"))
            if en_id in ("-1", "0"):
                continue
            texts = [t.text or "" for t in en.findall(".//w:t", ns)]
            endnotes[en_id] = " ".join(texts).strip()
    except Exception:
        pass
    return endnotes

def extract_footnote_refs(paragraph):
    refs = []
    for ref in paragraph._element.findall(f".//{{{W_NS}}}footnoteReference"):
        refs.append(("footnote", ref.get(qn("w:id"))))
    for ref in paragraph._element.findall(f".//{{{W_NS}}}endnoteReference"):
        refs.append(("endnote", ref.get(qn("w:id"))))
    return refs


# ============================================================
# RUMUS OMML
# ============================================================
def extract_omml_equations(paragraph):
    equations = []
    for omath in paragraph._element.findall(f".//{{{M_NS}}}oMath"):
        texts = [t.text or "" for t in omath.findall(f".//{{{M_NS}}}t")]
        equations.append({
            "latex": "",
            "text": " ".join(texts).strip(),
            "is_display": False,
        })
    for omathpara in paragraph._element.findall(f".//{{{M_NS}}}oMathPara"):
        for omath in omathpara.findall(f".//{{{M_NS}}}oMath"):
            texts = [t.text or "" for t in omath.findall(f".//{{{M_NS}}}t")]
            equations.append({
                "latex": "",
                "text": " ".join(texts).strip(),
                "is_display": True,
            })
    return equations


# ============================================================
# GAMBAR (inline + floating)
# ============================================================
def extract_images_from_paragraph(paragraph, doc, img_dir):
    images = []
    for drawing in paragraph._element.findall(f".//{{{W_NS}}}drawing"):
        for blip in drawing.findall(f".//{{{A_NS}}}blip"):
            rId = blip.get(qn("r:embed"))
            if rId is None:
                continue
            try:
                image_part = doc.part.related_parts[rId]
                ext = image_part.partname.ext
                filename = f"img_{rId}.{ext}"
                if not is_large_image(image_part.blob, filename):
                    continue
                path = Path(img_dir) / filename
                with Image.open(BytesIO(image_part.blob)) as pil_img:
                    path = save_for_vlm(pil_img, path, original=image_part.blob)
                images.append({"img_path": str(path.resolve()), "rId": rId})
            except Exception:
                continue
    return images


def has_page_break(paragraph):
    for br in paragraph._element.findall(f".//{{{W_NS}}}br"):
        if br.get(qn("w:type")) == "page":
            return True
    return False


# ============================================================
# DETEKSI CAPTION — STYLE-BASED, TANPA REGEX
# ============================================================
def is_caption_paragraph(paragraph) -> bool:
    """True jika paragraph memakai style yang mengandung kata 'caption'.

    Fleksibel untuk 'Caption', 'Caption2', 'Figure Caption', dll.
    Tidak bergantung pada isi teks, sehingga aman untuk berbagai penamaan.
    """
    try:
        style = paragraph.style
        if style is None:
            return False
        return "caption" in style.name.lower()
    except Exception:
        return False


# ============================================================
# KONVERSI TABEL HTML -> MARKDOWN
# ============================================================
def html_table_to_markdown(html: str) -> str | None:
    if not html:
        return None
    try:
        parser = TableParser(html)
        tables = parser.parse()
        if not tables:
            return None
        return "\n\n".join(t.to_markdown() for t in tables)
    except Exception:
        return None


def docx_table_to_html(table: DocxTable) -> str:
    rows = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            text = " ".join(p.text for p in cell.paragraphs).strip()
            cells.append(f"<td>{text}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table>" + "".join(rows) + "</table>"


# ============================================================
# ITERASI BLOCK
# ============================================================
def iter_block_items(doc):
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield DocxTable(child, doc)


def classify_block(block, doc, images_dir):
    """Klasifikasikan satu block menjadi tuple (type, payload).

    Type:
      - "image"    : payload = list of {img_path, rId}
      - "table"    : payload = DocxTable
      - "equation" : payload = list of {latex, text, is_display}
      - "caption"  : payload = str  (style-based, tanpa regex)
      - "text"     : payload = str
      - "empty"    : payload = None
    """
    if isinstance(block, DocxTable):
        return ("table", block)

    if not isinstance(block, Paragraph):
        return ("empty", None)

    # 1. Gambar
    imgs = extract_images_from_paragraph(block, doc, images_dir)
    if imgs:
        return ("image", imgs)

    # 2. Rumus
    eqs = extract_omml_equations(block)
    if eqs:
        return ("equation", eqs)

    # 3. Teks / Caption
    txt = (block.text or "").strip()
    if not txt:
        return ("empty", None)

    if is_caption_paragraph(block):
        return ("caption", txt)

    return ("text", txt)


# ============================================================
# CAPTION MATCHING — AMBIL SATU CAPTION TERAKHIR
# ============================================================
def find_caption_for_element(classified, idx, consumed):
    """Ambil SATU caption untuk elemen di `idx`.

    Strategi:
      - Kumpulkan semua caption berurutan yang berdekatan (skip empty).
      - Hentikan walk saat menemukan non-caption.
      - Tandai SEMUA caption yang terkumpul sebagai `consumed`
        (agar tidak muncul sebagai text).
      - Attach HANYA caption terakhir (paling deskriptif, biasanya
        'Gambar N: ...' atau 'Tabel N: ...').

    Return: list of caption string (0 atau 1 elemen).
    """
    candidates = []   # list of (index, text) sesuai urutan dokumen

    # --- Walk BACKWARD ---
    j = idx - 1
    while j >= 0:
        if j in consumed:
            j -= 1
            continue
        t, p = classified[j]
        if t == "empty":
            j -= 1
            continue
        if t == "caption":
            candidates.insert(0, (j, p))
            j -= 1
        else:
            break

    # --- Walk FORWARD ---
    j = idx + 1
    while j < len(classified):
        if j in consumed:
            j += 1
            continue
        t, p = classified[j]
        if t == "empty":
            j += 1
            continue
        if t == "caption":
            candidates.append((j, p))
            j += 1
        else:
            break

    if not candidates:
        return []

    # Tandai semua caption yang terkumpul sebagai consumed
    for ci, _ in candidates:
        consumed.add(ci)

    # Attach HANYA caption terakhir
    return [candidates[-1][1]]


# ============================================================
# MAP TABEL DARI UNSTRUCTURED
# ============================================================
def build_unstructured_table_map(elements):
    mapping = {}
    idx = 0
    for el in elements:
        if isinstance(el, UnstructuredTable):
            html = getattr(el.metadata, "text_as_html", None)
            if html:
                mapping[idx] = html
            idx += 1
    return mapping


# ============================================================
# PARSE
# ============================================================
def parse_docx(path: str | Path, images_dir: str | Path) -> list[dict]:
    """A .docx file → content list (text / image / table / equation).

    Embedded images are written to ``images_dir`` and returned as ``image`` items
    with a local ``img_path``; images under the ``FIGURE_MIN_*`` thresholds are
    skipped. Hosting them (S3) is the caller's job. ``page_idx`` counts explicit
    page breaks, since a .docx has no fixed pagination.
    """
    path = str(path)
    Path(images_dir).mkdir(parents=True, exist_ok=True)
    doc = Document(path)

    # 1. Unstructured untuk tabel HTML
    us_elements = partition_docx(filename=path, infer_table_structure=True)
    us_table_html_map = build_unstructured_table_map(us_elements)

    # 2. Footnote & endnote
    footnotes_map = get_footnotes_map(doc)
    endnotes_map = get_endnotes_map(doc)

    # 3. Klasifikasi semua block
    blocks = list(iter_block_items(doc))
    classified = [classify_block(b, doc, images_dir) for b in blocks]

    # 4. Iterasi
    output = []
    consumed = set()
    page_idx = 0
    table_counter = 0

    for idx, (btype, payload) in enumerate(classified):
        # Update page_idx kalau ada page break
        if isinstance(blocks[idx], Paragraph) and has_page_break(blocks[idx]):
            page_idx += 1

        if idx in consumed:
            continue

        # ----- IMAGE -----
        if btype == "image":
            caption = find_caption_for_element(classified, idx, consumed)
            for img in payload:
                output.append({
                    "type": "image",
                    "img_path": img["img_path"],
                    "image_caption": caption,
                    "image_footnote": [],
                    "page_idx": page_idx,
                })

        # ----- TABLE -----
        elif btype == "table":
            html = us_table_html_map.get(table_counter)
            if not html:
                html = docx_table_to_html(payload)
            md_table = html_table_to_markdown(html) or html

            caption = find_caption_for_element(classified, idx, consumed)

            output.append({
                "type": "table",
                "table_body": md_table,
                "table_caption": caption,
                "table_footnote": [],
                "page_idx": page_idx,
            })
            table_counter += 1

        # ----- EQUATION -----
        elif btype == "equation":
            for eq in payload:
                output.append({
                    "type": "equation",
                    "latex": eq["latex"],
                    "text": eq["text"],
                    "page_idx": page_idx,
                })

        # ----- TEXT -----
        elif btype == "text":
            refs = extract_footnote_refs(blocks[idx])
            fn_texts = []
            for kind, fid in refs:
                if kind == "footnote" and fid in footnotes_map:
                    fn_texts.append(footnotes_map[fid])
                elif kind == "endnote" and fid in endnotes_map:
                    fn_texts.append(endnotes_map[fid])

            item = {"type": "text", "text": payload, "page_idx": page_idx}
            if fn_texts:
                item["footnote"] = fn_texts
            output.append(item)

        # ----- CAPTION ORPHAN (tidak terpakai) -----
        elif btype == "caption":
            output.append({
                "type": "text",
                "text": payload,
                "page_idx": page_idx,
            })

        # ----- EMPTY: skip -----

    return output
