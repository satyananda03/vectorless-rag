"""Bahasa Indonesia for every text the index writes: node summaries, merged-page
titles, the document description. The agent routes on these texts, so they must
keep the terms the documents themselves use.

PageIndex writes its prompts inline, so this patches them from outside:
``SummaryScheduler._ask`` (every node summary, flash and standard) gets the rule
appended; ``generate_doc_description`` is replaced by a copy of upstream's with the
rule in. tests/test_multimodal.py pins the source of the copied function, so an
upstream prompt change fails there instead of silently going stale.
"""
from pageindex import utils

INDEX_LANGUAGE = "BAHASA INDONESIA"
LANGUAGE_RULE = f"""
    Write every summary, title and description in {INDEX_LANGUAGE}, whatever the language of the text. Keep the document's own terms exactly as written: names, codes, numbers and abbreviations (such as Renstra, DPA, Berita Acara) are never translated.
    """


def generate_doc_description(structure, model=None):
    prompt = f"""Your are an expert in generating descriptions for a document.
    You are given a structure of a document. Your task is to generate a one-sentence description for the document, which makes it easy to distinguish the document from other documents.

    Document Structure: {structure}

    Directly return the description, do not include any other text.
    {LANGUAGE_RULE}"""
    try:
        return utils.llm_completion(model, prompt)
    except Exception as e:
        # Per-prompt 400: the unbounded whole-tree prompt overran the
        # context; the indexed document survives with no description.
        if getattr(e, "status_code", None) == 400:
            return ""
        raise


def install() -> None:
    """Patch PageIndex's summary and description prompts; idempotent."""
    from pageindex import page_index_classic  # its `from .utils import *` copied the names
    scheduler = utils.SummaryScheduler
    if not getattr(scheduler._ask, "multimodal", False):
        original_ask = scheduler._ask

        async def _ask(self, prompt, prio):
            return await original_ask(self, prompt + LANGUAGE_RULE, prio)

        _ask.multimodal = True
        scheduler._ask = _ask
    utils.generate_doc_description = generate_doc_description
    page_index_classic.generate_doc_description = generate_doc_description
