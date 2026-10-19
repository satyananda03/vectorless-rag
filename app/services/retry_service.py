"""Retry policy for model calls: transient failures only, honouring the provider's ``retry_after``."""
import asyncio
import functools
import logging
import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx
import openai
from tenacity import before_sleep_log, retry, retry_if_exception, stop_after_attempt

logger = logging.getLogger(__name__)


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
