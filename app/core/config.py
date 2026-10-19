"""Settings for the agent app, read from the repository's ``.env``."""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore", env_ignore_empty=True)

    pageindex_chat_model: str = Field(validation_alias="PAGEINDEX_CHAT_MODEL")
    openai_api_key: str = Field(validation_alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")
    max_retries: int = Field(default=3, validation_alias="MAX_RETRIES")
    llm_timeout: float = Field(default=120.0, validation_alias="LLM_TIMEOUT")
    # The PageIndex library the agent's tools read.
    pageindex_storage_path: str = Field(default=".pageindex", validation_alias="PAGEINDEX_STORAGE_PATH")


settings = Settings()
