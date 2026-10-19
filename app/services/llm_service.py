"""Chat models for the agent: an OpenAI-compatible client that retries transient failures."""
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.services.retry_service import per_instance


class RetryingChat(ChatOpenAI):
    retry_attempts: int = 1

    @per_instance
    def _generate_with_cache(self, *args, **kwargs):
        return super()._generate_with_cache(*args, **kwargs)

    @per_instance
    async def _agenerate_with_cache(self, *args, **kwargs):
        return await super()._agenerate_with_cache(*args, **kwargs)


def get_llm(model_id, *, base_url=None, api_key=None, timeout=None, max_retries=None,
            max_tokens=None, **model_kwargs):
    """A chat model on the configured OpenAI-compatible endpoint; unset arguments come from ``settings``."""
    return RetryingChat(
        model=model_id,
        base_url=base_url or settings.openai_base_url,
        api_key=api_key or settings.openai_api_key,
        timeout=timeout or settings.llm_timeout,
        max_tokens=max_tokens,
        max_retries=0,
        retry_attempts=settings.max_retries if max_retries is None else max_retries,
        **model_kwargs,
    )
