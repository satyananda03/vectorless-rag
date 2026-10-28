"""PageIndex multimodal: Textract + VLM for scanned PDFs, images, office files,
spreadsheets, tables and figures, on top of the upstream ``pageindex`` package.

Importing this package patches ``pageindex`` in place (see ``install``): Flash's
extraction, the summary and description prompts (Bahasa Indonesia), and
``get_page_content``'s citations.
"""
from pageindex.errors import PageIndexAPIError

from . import citation, language, multimodal_api
from .client import MultimodalClient
from .langgraph_tools import pageindex_tools


def install() -> None:
    """Apply every patch to ``pageindex``; idempotent, run on import."""
    multimodal_api.install()
    language.install()
    citation.install()


install()


__all__ = ["MultimodalClient", "PageIndexAPIError", "install", "pageindex_tools"]
