from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

MODEL_ROLES: dict[str, tuple[str, str]] = {
    "vlm": ("vlm", "VLM"),
}

class MultimodalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)  # `KEY=` means unset
    vlm_model: Optional[str] = Field(default=None, validation_alias="VLM_MODEL")
    vlm_binding_host: Optional[str] = Field(default=None, validation_alias="VLM_BINDING_HOST")
    vlm_binding_api_key: Optional[str] = Field(default=None, validation_alias="VLM_BINDING_API_KEY")
    vlm_max_async: int = Field(default=3, validation_alias="VLM_MAX_ASYNC")
    vlm_timeout: float = Field(default=120.0, validation_alias="VLM_TIMEOUT")
    max_retries: int = Field(default=5, validation_alias="MAX_RETRIES")
    leading_context_tokens: int = Field(default=500, validation_alias="LEADING_CONTEXT_TOKENS")
    trailing_context_tokens: int = Field(default=500, validation_alias="TRAILING_CONTEXT_TOKENS")
    textract_access_key_id: Optional[str] = Field(default=None, validation_alias="TEXTRACT_ACCESS_KEY_ID")
    textract_secret_access_key: Optional[str] = Field(default=None, validation_alias="TEXTRACT_SECRET_ACCESS_KEY")
    textract_region: Optional[str] = Field(default=None, validation_alias="TEXTRACT_REGION")
    textract_max_concurrent_jobs: int = Field(default=40, validation_alias="TEXTRACT_MAX_CONCURRENT_JOBS")
    textract_mode: str = Field(default="async", validation_alias="TEXTRACT_MODE")  # "async" | "sync"
    figure_min_width_px: int = Field(default=600, validation_alias="FIGURE_MIN_WIDTH_PX")
    figure_min_height_px: int = Field(default=600, validation_alias="FIGURE_MIN_HEIGHT_PX")
    figure_min_area_ratio: float = Field(default=0.01, validation_alias="FIGURE_MIN_AREA_RATIO")
    s3_bucket: Optional[str] = Field(default=None, validation_alias="S3_BUCKET")
    s3_access_key_id: Optional[str] = Field(default=None, validation_alias="S3_ACCESS_KEY_ID")
    s3_secret_access_key: Optional[str] = Field(default=None, validation_alias="S3_SECRET_ACCESS_KEY")
    s3_region: Optional[str] = Field(default=None, validation_alias="S3_REGION")
    s3_prefix: str = Field(default="pdf-extractions", validation_alias="S3_PREFIX")
    flat_tree_max_nodes: int = Field(default=15, validation_alias="FLAT_TREE_MAX_NODES")
    max_parallel_office_convert: int = Field(default=2, validation_alias="MAX_PARALLEL_OFFICE_CONVERT")
    unoserver_host: str = Field(default="unoserver", validation_alias="UNOSERVER_HOST")
    unoserver_port: int = Field(default=2003, validation_alias="UNOSERVER_PORT")
    office_convert_timeout: int = Field(default=120, validation_alias="OFFICE_CONVERT_TIMEOUT")

    @property
    def resolved_s3_region(self) -> Optional[str]:
        return self.s3_region

    def require(self, *roles: str) -> None:
        missing = []
        for role in roles:
            field, env = MODEL_ROLES[role]
            for suffix in ("model", "binding_host", "binding_api_key"):
                if not getattr(self, f"{field}_{suffix}"):
                    missing.append(f"{env}_{suffix.upper()}")
        if missing:
            raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

settings = MultimodalSettings()
