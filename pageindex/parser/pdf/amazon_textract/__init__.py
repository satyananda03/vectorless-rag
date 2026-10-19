"""Parser AWS Textract (mode sync atau async): PDF → raw JSON → content list.

Alur & modul:

* ``quality``       — gerbang DPI efektif (metadata, tanpa render)
* ``pages``         — split PDF per halaman; render + enhance hanya untuk crop figure / fallback
* ``textract_api``  — client Textract bersama + AnalyzeDocument per halaman (mode sync)
* ``async_job``     — StartDocumentAnalysis lewat S3 + polling (mode async)
* ``parse``         — orkestrasi ``parse_document_to_json``, satu pintu untuk kedua mode
* ``blocks``        — navigasi graf Block + geometri bbox
* ``tables``        — TABLE → markdown, caption, footnote
* ``content_list``  — ``reconstruct_content_list`` (text / image / table)
"""
from .content_list import reconstruct_content_list
from .parse import parse_document_to_json
from .quality import LowDpiPageError
from .textract_api import TextractPageError

__all__ = [
    "LowDpiPageError",
    "TextractPageError",
    "parse_document_to_json",
    "reconstruct_content_list",
]
