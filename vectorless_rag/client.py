from pageindex import PageIndexLocalClient
from pageindex.local_api import LocalAPI

from .multimodal_api import MultimodalLocalAPI

# Without an output cap Anthropic models (and proxies in front of them) fall back to
# 4096 tokens, which truncates the long TOC JSON standard mode asks for. A
# `max_tokens` in index_backend overrides it.
INDEX_MAX_TOKENS = 16000


class MultimodalClient(PageIndexLocalClient):
    """``PageIndexLocalClient`` that also indexes scanned PDFs, images, office files
    and ``.xlsx``, and reads tables and figures with Textract + VLM.

    Takes ``PageIndexLocalClient``'s arguments unchanged.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if type(self._api) is LocalAPI:
            # swap the class, not the object: the constructor's arguments stay upstream's
            self._api.__class__ = MultimodalLocalAPI
            self._api._init_multimodal()
        self._api._index_backend = {"max_tokens": INDEX_MAX_TOKENS,
                                    **(self._api._index_backend or {})}
