"""VLM description of the figures Textract cropped: the S3 URL plus the text around it.

The VLM client (``RetryingChat`` / ``get_vlm``) retries transient failures;
the leading/trailing context helpers are ported from hybrid-rag
``ingestion/multimodal/content.py``, with PageIndex's ``count_tokens`` as the tokenizer.
"""
import asyncio
import functools
import logging
import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import cache, lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
import openai
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import before_sleep_log, retry, retry_if_exception, stop_after_attempt

from .config import settings
from .prompts import load_prompt
from .utils import count_tokens

logger = logging.getLogger(__name__)

MULTIMODAL_PROMPTS_PATH = Path(__file__).resolve().parent / "multimodal.yaml"


# ── VLM client ──

def is_transient(exc: BaseException) -> bool:
    """True kalau mengirim ulang permintaan yang sama bisa berhasil."""
    if isinstance(exc, httpx.HTTPStatusError):  # jalur rerank
        return exc.response.status_code in (408, 429) or exc.response.status_code >= 500
    if isinstance(exc, (httpx.TransportError, TimeoutError)):  # TimeoutError: stream macet di langchain-openai
        return True
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    return isinstance(exc, openai.APIStatusError) and (exc.status_code in (408, 409, 429) or exc.status_code >= 500)


def seconds(value) -> float | None:
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            delay = (parsedate_to_datetime(str(value)).astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
    return max(0.0, float(math.ceil(delay)))


def wait_seconds(state) -> float:
    """`retry_after` dari provider (body JSON, lalu header) kalau ada; kalau tidak 1s, 2s, 4s, maksimal 8s."""
    exc = state.outcome.exception()
    body = getattr(exc, "body", None)
    headers = getattr(getattr(exc, "response", None), "headers", None)
    delay = seconds(body.get("retry_after")) if isinstance(body, dict) else None
    if delay is None and headers is not None:
        delay = seconds(headers.get("retry-after"))
    return delay if delay is not None else min(8.0, 2.0 ** (state.attempt_number - 1))


def with_retry(attempts):
    def decorate(func):
        def policy(*args):
            return retry(
                stop=stop_after_attempt(max(1, attempts(*args))),
                wait=wait_seconds,
                retry=retry_if_exception(is_transient),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            )(func)

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def awrapper(*args, **kwargs):
                return await policy(*args)(*args, **kwargs)
            return awrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return policy(*args)(*args, **kwargs)
        return wrapper

    return decorate


per_instance = with_retry(lambda self, *_: self.retry_attempts)


class RetryingChat(ChatOpenAI):
    retry_attempts: int = 1

    @per_instance
    def _generate_with_cache(self, *args, **kwargs):
        return super()._generate_with_cache(*args, **kwargs)

    @per_instance
    async def _agenerate_with_cache(self, *args, **kwargs):
        return await super()._agenerate_with_cache(*args, **kwargs)


def get_llm(model_id, *, base_url, api_key, timeout, max_retries, max_tokens, **model_kwargs):
    return RetryingChat(
        model=model_id, base_url=base_url, api_key=api_key, timeout=timeout, max_tokens=max_tokens,
        max_retries=0, retry_attempts=max_retries, **model_kwargs,
    )


def get_vlm(model_id: str, *, base_url: str | None, api_key: str | None, timeout: float | None = None):
    """Same signature as hybrid_rag ``providers.get_vlm``: retries come from ``MAX_RETRIES``."""
    return get_llm(model_id, base_url=base_url, api_key=api_key, timeout=timeout,
                   max_retries=settings.max_retries, max_tokens=None)


# ── surrounding context (hybrid-rag ingestion/multimodal/content.py) ──

#: Hierarki separator dari kasar ke halus. Konteks dipotong di batas
#: paragraf/baris/kalimat, tidak pernah putus di tengah kata.
_SPLIT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def extract_section_path_from_content_list(content_list: List[Dict[str, Any]], current_index: int) -> str:
    """``BAB I > 1.2 …`` from the headings (``text_level``) before ``current_index``."""
    if not content_list or current_index is None:
        return ""
    try:
        limit = max(0, int(current_index))
    except (TypeError, ValueError):
        return ""

    heading_chain: List[Tuple[int, str]] = []
    for item in content_list[:limit]:
        if not isinstance(item, dict) or item.get("type", "text") != "text":
            continue
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        try:
            level = int(item.get("text_level", 0) or 0)
        except (TypeError, ValueError):
            continue
        if level <= 0:
            continue
        while heading_chain and heading_chain[-1][0] >= level:
            heading_chain.pop()
        heading_chain.append((level, text))
    return " > ".join(text for _, text in heading_chain)


def _token_count(text: str) -> int:
    return count_tokens(text)


@lru_cache(maxsize=4)
def _splitter(max_tokens: int) -> RecursiveCharacterTextSplitter:
    """Splitter yang menghitung ukuran dalam token, bukan karakter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=0,
        separators=_SPLIT_SEPARATORS,
        length_function=_token_count,
    )


def _take_nearest(blocks: List[str], max_tokens: int, from_end: bool) -> str:
    """Blok terdekat item, sebanyak yang muat dalam budget token.

    Batas potongnya adalah batas blok (satu paragraf utuh dari parser). Baru kalau
    satu blok sendirian sudah melewati budget, isinya dipotong
    :class:`RecursiveCharacterTextSplitter` mengikuti ``_SPLIT_SEPARATORS``.
    """
    if not blocks or max_tokens <= 0:
        return ""

    picked: List[str] = []
    used = 0
    for block in reversed(blocks) if from_end else blocks:
        length = _token_count(block)
        if used + length <= max_tokens:
            picked.append(block)
            used += length
            continue
        if not picked:
            # Blok tetangga langsung pun sudah kebesaran: ambil ujungnya yang menempel ke item.
            chunks = _splitter(max_tokens).split_text(block)
            if chunks:
                chunk = (chunks[-1] if from_end else chunks[0]).strip()
                picked.append(chunk.lstrip(". ") if from_end else chunk)
        break

    if from_end:
        picked.reverse()
    return "\n\n".join(picked)


def extract_leading_trailing_from_content_list(
    content_list: List[Dict[str, Any]],
    current_index: int,
    leading_tokens: int | None = None,
    trailing_tokens: int | None = None,
) -> Tuple[str, str]:
    """Teks sebelum dan sesudah satu item, masing-masing dibatasi token (lintas halaman)."""
    if not content_list or current_index is None:
        return "", ""
    try:
        idx = int(current_index)
    except (TypeError, ValueError):
        return "", ""
    if idx < 0 or idx >= len(content_list):
        return "", ""

    def texts(positions) -> List[str]:
        out = []
        for pos in positions:
            item = content_list[pos]
            if not isinstance(item, dict) or item.get("type", "text") != "text":
                continue
            text = str(item.get("text", "") or "").strip()
            if text:
                out.append(text)
        return out

    leading = _take_nearest(
        texts(range(0, idx)),
        leading_tokens if leading_tokens is not None else settings.leading_context_tokens,
        from_end=True,
    )
    trailing = _take_nearest(
        texts(range(idx + 1, len(content_list))),
        trailing_tokens if trailing_tokens is not None else settings.trailing_context_tokens,
        from_end=False,
    )
    return leading, trailing


def surrounding_context_block(leading: str, trailing: str) -> str:
    """Render the neighbours block, or "" when the item has no neighbours."""
    if not (leading or trailing):
        return ""
    template = load_prompt("surrounding_context", MULTIMODAL_PROMPTS_PATH)["user_prompt"]
    return template.format(leading=leading or "None", trailing=trailing or "None")


# ── image analysis ──

@cache
def image_analysis_chain() -> Runnable:
    """Prompt + S3 image URL → VLM → plain text."""
    settings.require("vlm")
    cfg = load_prompt("image_analysis", MULTIMODAL_PROMPTS_PATH)
    prompt = ChatPromptTemplate.from_messages([
        ("system", cfg["system_prompt"]),
        ("human", [
            {"type": "text", "text": cfg["user_prompt"]},
            # detail=high: an OpenAI-format endpoint may otherwise downscale the image.
            {"type": "image_url", "image_url": {"url": "{image_url}", "detail": "high"}},
        ]),
    ])
    vlm = get_vlm(settings.vlm_model, base_url=settings.vlm_binding_host,
                  api_key=settings.vlm_binding_api_key, timeout=settings.vlm_timeout)
    return prompt | vlm | StrOutputParser()


def _caption_text(value) -> str:
    if isinstance(value, list):
        value = "; ".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip() if value else "None"


async def describe_figures(content_list: List[Dict[str, Any]]) -> None:
    """Write a VLM description into every ``image`` item (``img_description``), in place.

    Runs after ``host_figures``: the VLM gets the S3 URL, never a local file. A figure
    that still fails after the retries fails the document, so it is never indexed
    without its image.
    """
    figures = [i for i, item in enumerate(content_list) if item.get("type") == "image"]
    for i in figures:
        if not str(content_list[i].get("img_path") or "").startswith(("http://", "https://")):
            raise RuntimeError(f"Figure belum di-upload ke S3: {content_list[i].get('img_path')}")
    if not figures:
        return

    # Context before any item changes, so every figure sees the same parser output.
    inputs = {}
    for i in figures:
        item = content_list[i]
        leading, trailing = extract_leading_trailing_from_content_list(content_list, i)
        inputs[i] = {
            "context_block": surrounding_context_block(leading, trailing),
            "section_path": extract_section_path_from_content_list(content_list, i) or "None",
            "image_path": item["img_path"],
            "captions": _caption_text(item.get("image_caption")),
            "footnotes": _caption_text(item.get("image_footnote")),
            "image_url": item["img_path"],
        }

    chain = image_analysis_chain()
    semaphore = asyncio.Semaphore(max(1, settings.vlm_max_async))

    async def _describe(i: int) -> str:
        async with semaphore:
            logger.info(f"[VLM] {content_list[i]['img_path']}")
            text = (await chain.ainvoke(inputs[i]) or "").strip()
        if not text:
            raise RuntimeError(f"VLM tidak menghasilkan deskripsi: {content_list[i]['img_path']}")
        return text

    descriptions = await asyncio.gather(*(_describe(i) for i in figures))
    for i, text in zip(figures, descriptions):
        content_list[i]["img_description"] = text
