"""Gerbang kualitas input: DPI efektif halaman scan (dari metadata, tanpa render)."""
import pymupdf as fitz

DPI_EFFECTIVE_THRESHOLD = 1

class LowDpiPageError(Exception):
    """Raised ketika satu atau lebih halaman punya DPI efektif di bawah threshold."""
    def __init__(self, low_dpi_pages: list[dict]):
        self.low_dpi_pages = low_dpi_pages
        page_list = ", ".join(
            f"halaman {p['page_idx']+1} (DPI_efektif={p['dpi']:.1f})"
            for p in low_dpi_pages
        )
        super().__init__(f"Dokumen ditolak: {len(low_dpi_pages)} halaman DPI rendah: {page_list}")


def get_effective_dpi_all_pages(pdf_path: str) -> list:
    """
    Hitung DPI efektif untuk SEMUA halaman sekaligus, buka file PDF
    hanya sekali. Return list of dict {page_idx, dpi} (dpi=None jika
    halaman tidak punya gambar embedded / PDF native).
    """
    doc = fitz.open(pdf_path)
    results = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_width_in = page.rect.width / 72
        images = page.get_images(full=True)
        biggest = None
        for img in images:
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                area = base_image["width"] * base_image["height"]
                if biggest is None or area > biggest[1]:
                    biggest = (base_image, area)
            except Exception:
                pass
        dpi = biggest[0]["width"] / page_width_in if biggest and page_width_in > 0 else None
        results.append({"page_idx": page_idx, "dpi": dpi})
    doc.close()
    return results


def check_effective_dpi(pdf_path: str) -> list:
    """Tolak dokumen bila ada halaman di bawah threshold; return hasil DPI semua halaman."""
    print(f"[Textract] Mengecek DPI efektif setiap halaman (threshold={DPI_EFFECTIVE_THRESHOLD})...")
    all_dpi = get_effective_dpi_all_pages(pdf_path)
    low_dpi_pages = [p for p in all_dpi if p["dpi"] is not None and p["dpi"] < DPI_EFFECTIVE_THRESHOLD]

    if low_dpi_pages:
        print(f"[Textract] ❌ {len(low_dpi_pages)} halaman DPI di bawah threshold, dokumen ditolak.")
        for p in low_dpi_pages:
            print(f"    - Halaman {p['page_idx']+1}: DPI_efektif={p['dpi']:.1f}")
        raise LowDpiPageError(low_dpi_pages)

    print(f"[Textract] Semua {len(all_dpi)} halaman lolos cek DPI efektif.")
    return all_dpi
