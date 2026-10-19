"""PDF → halaman: split per halaman untuk Textract, render hanya bila butuh piksel."""
import pymupdf as fitz
from PIL import Image, ImageFilter

RENDER_DPI = 400


def enhance_image(
    pil_img: Image.Image,
    apply_otsu: bool = False,
    denoise: bool = True,
) -> Image.Image:
    """Tingkatkan kualitas gambar menggunakan PIL (Grayscale + Unsharp Masking)."""
    gray = pil_img.convert("L")
    if denoise:
        gray = gray.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    if apply_otsu:
        gray = gray.point(lambda p: 255 if p > 128 else 0)
    return gray


def render_page_image(
    pdf_path: str,
    page_idx: int,
    dpi: int = RENDER_DPI,
    enhance: bool = True,
) -> Image.Image:
    """Render satu halaman → PIL Image.

    Bila dipakai untuk crop figure berwarna, panggil dengan ``enhance=False``
    agar warna RGB tidak hilang menjadi grayscale.
    """
    doc = fitz.open(pdf_path)
    try:
        pix = doc[page_idx].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return enhance_image(img) if enhance else img
    finally:
        doc.close()


def split_pdf_pages(pdf_path: str, dpi: int = RENDER_DPI):
    """Pecah PDF jadi PDF satu halaman tanpa rasterisasi (~ms/halaman)."""
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    try:
        for page_idx, page in enumerate(doc):
            # Dihitung di awal sebelum manipulasi rotasi, identik dengan _normalized_pdf (async)
            irect = (page.rect * mat).irect

            one = fitz.open()
            try:
                one.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
                one[0].remove_rotation()
                pdf_bytes = one.tobytes(garbage=3, deflate=True)
            finally:
                one.close()

            yield {
                "page_idx": page_idx,
                "source_pdf": pdf_path,
                "doc_bytes": pdf_bytes,
                "img_w": irect.width,
                "img_h": irect.height,
            }
    finally:
        doc.close()