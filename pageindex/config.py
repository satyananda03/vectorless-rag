from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

NOVA_MODELS: frozenset[str] = frozenset({"mjai.nova-2-lite-v1:0"})

MODEL_ROLES: dict[str, tuple[str, str]] = {
    "extract": ("extract_llm", "EXTRACT_LLM"),
    "contextualize": ("contextualize_llm", "CONTEXTUALIZE_LLM"),
    "keyword": ("keyword_llm", "KEYWORD_LLM"),
    "vlm": ("vlm", "VLM"),
    "embedding": ("embedding", "EMBEDDING"),
}

class RagSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)  # `KEY=` means unset
    extract_llm_max_async: int = Field(default=4, validation_alias="EXTRACT_LLM_MAX_ASYNC")
    keyword_llm_model: Optional[str] = Field(default=None, validation_alias="KEYWORD_LLM_MODEL")
    keyword_llm_binding_host: Optional[str] = Field(default=None, validation_alias="KEYWORD_LLM_BINDING_HOST")
    keyword_llm_binding_api_key: Optional[str] = Field(default=None, validation_alias="KEYWORD_LLM_BINDING_API_KEY")
    keyword_llm_timeout: Optional[float] = Field(default=None, validation_alias="KEYWORD_LLM_TIMEOUT")
    extract_llm_model: Optional[str] = Field(default=None, validation_alias="EXTRACT_LLM_MODEL")
    extract_llm_binding_host: Optional[str] = Field(default=None, validation_alias="EXTRACT_LLM_BINDING_HOST")
    extract_llm_binding_api_key: Optional[str] = Field(default=None, validation_alias="EXTRACT_LLM_BINDING_API_KEY")
    extract_llm_timeout: Optional[float] = Field(default=None, validation_alias="EXTRACT_LLM_TIMEOUT")
    vlm_model: Optional[str] = Field(default=None, validation_alias="VLM_MODEL")
    vlm_binding_host: Optional[str] = Field(default=None, validation_alias="VLM_BINDING_HOST")
    vlm_binding_api_key: Optional[str] = Field(default=None, validation_alias="VLM_BINDING_API_KEY")
    vlm_max_async: int = Field(default=3, validation_alias="VLM_MAX_ASYNC")
    vlm_timeout: float = Field(default=120.0, validation_alias="VLM_TIMEOUT")
    max_retries: int = Field(default=5, validation_alias="MAX_RETRIES")
    embedding_binding_host: Optional[str] = Field(default=None, validation_alias="EMBEDDING_BINDING_HOST")
    embedding_binding_api_key: Optional[str] = Field(default=None, validation_alias="EMBEDDING_BINDING_API_KEY")
    embedding_model: str = Field(default="mjai.titan-embed-text-v2:0", validation_alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=1024, validation_alias="EMBEDDING_DIM")
    embedding_token_limit: int = Field(default=8192, validation_alias="EMBEDDING_TOKEN_LIMIT")
    embedding_batch_num: int = Field(default=1, validation_alias="EMBEDDING_BATCH_NUM")
    embedding_func_max_async: int = Field(default=8, validation_alias="EMBEDDING_FUNC_MAX_ASYNC")
    embedding_timeout: float = Field(default=60.0, validation_alias="EMBEDDING_TIMEOUT")
    top_k: int = Field(default=10, validation_alias="TOP_K")
    chunk_top_k: int = Field(default=10, validation_alias="CHUNK_TOP_K")
    parent_top_k: int = Field(default=10, validation_alias="PARENT_TOP_K")
    cosine_threshold: float = Field(default=0.2, validation_alias="COSINE_THRESHOLD")
    max_entity_tokens: int = Field(default=6000, validation_alias="MAX_ENTITY_TOKENS")
    max_relation_tokens: int = Field(default=8000, validation_alias="MAX_RELATION_TOKENS")
    enable_content_headings: bool = Field(default=True, validation_alias="ENABLE_CONTENT_HEADINGS")
    related_chunk_number: int = Field(default=5, validation_alias="RELATED_CHUNK_NUMBER")
    kg_chunk_pick_method: str = Field(default="VECTOR", validation_alias="KG_CHUNK_PICK_METHOD")
    max_graph_nodes: int = Field(default=1000, validation_alias="MAX_GRAPH_NODES")
    hybrid_search_chunk_candidates: int = Field(default=30, validation_alias="HYBRID_SEARCH_CHUNK_CANDIDATES")
    hybrid_search_w_semantic: float = Field(default=1.0, validation_alias="HYBRID_SEARCH_W_SEMANTIC")
    hybrid_search_w_keyword: float = Field(default=0.8, validation_alias="HYBRID_SEARCH_W_KEYWORD")
    rerank_model: Optional[str] = Field(default=None, validation_alias="RERANK_MODEL")
    rerank_binding_host: Optional[str] = Field(default=None, validation_alias="RERANK_BINDING_HOST")
    rerank_binding_api_key: Optional[str] = Field(default=None, validation_alias="RERANK_BINDING_API_KEY")
    rerank_max_async: int = Field(default=4, validation_alias="RERANK_MAX_ASYNC")
    rerank_timeout: float = Field(default=30.0, validation_alias="RERANK_TIMEOUT")
    min_rerank_score: float = Field(default=0.0, validation_alias="MIN_RERANK_SCORE")
    rerank_by_default: bool = Field(default=True, validation_alias="RERANK_BY_DEFAULT")
    enable_llm_cache: bool = Field(default=True, validation_alias="ENABLE_LLM_CACHE")
    parent_chunk_size: int = Field(default=2500, validation_alias="PARENT_CHUNK_SIZE")
    parent_chunk_overlap: int = Field(default=200, validation_alias="PARENT_CHUNK_OVERLAP")
    child_chunk_size: int = Field(default=310, validation_alias="CHILD_CHUNK_SIZE")
    child_chunk_overlap: int = Field(default=50, validation_alias="CHILD_CHUNK_OVERLAP")
    contextualize_llm_model: Optional[str] = Field(default=None, validation_alias="CONTEXTUALIZE_LLM_MODEL")
    contextualize_llm_binding_host: Optional[str] = Field(default=None, validation_alias="CONTEXTUALIZE_LLM_BINDING_HOST")
    contextualize_llm_binding_api_key: Optional[str] = Field(default=None, validation_alias="CONTEXTUALIZE_LLM_BINDING_API_KEY")
    contextualize_llm_timeout: Optional[float] = Field(default=None, validation_alias="CONTEXTUALIZE_LLM_TIMEOUT")
    contextualize_max_async: int = Field(default=10, validation_alias="CONTEXTUALIZE_MAX_ASYNC")
    leading_context_tokens: int = Field(default=500, validation_alias="LEADING_CONTEXT_TOKENS")
    trailing_context_tokens: int = Field(default=500, validation_alias="TRAILING_CONTEXT_TOKENS")
    max_gleaning: int = Field(default=1, validation_alias="MAX_GLEANING")
    max_extract_input_tokens: int = Field(default=20480, validation_alias="MAX_EXTRACT_INPUT_TOKENS")
    max_extraction_records: int = Field(default=100, validation_alias="MAX_EXTRACTION_RECORDS")
    max_extraction_entities: int = Field(default=40, validation_alias="MAX_EXTRACTION_ENTITIES")
    force_llm_summary_on_merge: int = Field(default=8, validation_alias="FORCE_LLM_SUMMARY_ON_MERGE")
    summary_max_tokens: int = Field(default=1200, validation_alias="SUMMARY_MAX_TOKENS")
    summary_context_size: int = Field(default=12000, validation_alias="SUMMARY_CONTEXT_SIZE")
    summary_length_recommended: int = Field(default=1000, validation_alias="SUMMARY_LENGTH_RECOMMENDED")
    summary_max_async: int = Field(default=5, validation_alias="SUMMARY_MAX_ASYNC")
    max_source_ids_per_entity: int = Field(default=200, validation_alias="MAX_SOURCE_IDS_PER_ENTITY")
    max_source_ids_per_relation: int = Field(default=200, validation_alias="MAX_SOURCE_IDS_PER_RELATION")
    source_ids_limit_method: str = Field(default="KEEP", validation_alias="SOURCE_IDS_LIMIT_METHOD")
    max_file_paths: int = Field(default=80, validation_alias="MAX_FILE_PATHS")
    max_parallel_ingestion: int = Field(default=40, validation_alias="MAX_PARALLEL_INGESTION")
    textract_access_key_id: Optional[str] = Field(default=None, validation_alias="TEXTRACT_ACCESS_KEY_ID")
    textract_secret_access_key: Optional[str] = Field(default=None, validation_alias="TEXTRACT_SECRET_ACCESS_KEY")
    textract_region: Optional[str] = Field(default=None, validation_alias="TEXTRACT_REGION")
    textract_max_concurrent_jobs: int = Field(default=50, validation_alias="TEXTRACT_MAX_CONCURRENT_JOBS")
    textract_mode: str = Field(default="async", validation_alias="TEXTRACT_MODE")  # "async" | "sync"
    figure_min_width_px: int = Field(default=500, validation_alias="FIGURE_MIN_WIDTH_PX")
    figure_min_height_px: int = Field(default=500, validation_alias="FIGURE_MIN_HEIGHT_PX")
    figure_min_area_ratio: float = Field(default=0.01, validation_alias="FIGURE_MIN_AREA_RATIO")
    s3_bucket: Optional[str] = Field(default=None, validation_alias="S3_BUCKET")
    s3_access_key_id: Optional[str] = Field(default=None, validation_alias="S3_ACCESS_KEY_ID")
    s3_secret_access_key: Optional[str] = Field(default=None, validation_alias="S3_SECRET_ACCESS_KEY")
    s3_region: Optional[str] = Field(default=None, validation_alias="S3_REGION")
    s3_prefix: str = Field(default="pdf-extractions", validation_alias="S3_PREFIX")
    max_parallel_office_convert: int = Field(default=2, validation_alias="MAX_PARALLEL_OFFICE_CONVERT")
    unoserver_host: str = Field(default="unoserver", validation_alias="UNOSERVER_HOST")
    unoserver_port: int = Field(default=2003, validation_alias="UNOSERVER_PORT")
    office_convert_timeout: int = Field(default=120, validation_alias="OFFICE_CONVERT_TIMEOUT")
    postgres_host: str = Field(default="localhost", validation_alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, validation_alias="POSTGRES_PORT")
    postgres_database: str = Field(default="postgres", validation_alias="POSTGRES_DATABASE")
    postgres_user: str = Field(default="postgres", validation_alias="POSTGRES_USER")
    postgres_password: str = Field(default="", validation_alias="POSTGRES_PASSWORD")
    postgres_max_connections: int = Field(default=25, validation_alias="POSTGRES_MAX_CONNECTIONS")
    postgres_schema: str = Field(default="rag", validation_alias="POSTGRES_SCHEMA")
    postgres_ssl_mode: str = Field(default="prefer", validation_alias="POSTGRES_SSL_MODE")
    postgres_vector_index_type: str = Field(default="HNSW", validation_alias="POSTGRES_VECTOR_INDEX_TYPE")
    postgres_hnsw_m: int = Field(default=16, validation_alias="POSTGRES_HNSW_M")
    postgres_hnsw_ef: int = Field(default=256, validation_alias="POSTGRES_HNSW_EF")
    postgres_connection_retries: int = Field(default=10, validation_alias="POSTGRES_CONNECTION_RETRIES")
    postgres_connection_retry_backoff: float = Field(default=3.0, validation_alias="POSTGRES_CONNECTION_RETRY_BACKOFF")
    postgres_connection_retry_backoff_max: float = Field(default=30.0, validation_alias="POSTGRES_CONNECTION_RETRY_BACKOFF_MAX")
    postgres_pool_close_timeout: float = Field(default=5.0, validation_alias="POSTGRES_POOL_CLOSE_TIMEOUT")
    postgres_upsert_max_payload_bytes: int = Field(default=16 * 1024 * 1024, validation_alias="POSTGRES_UPSERT_MAX_PAYLOAD_BYTES")
    postgres_upsert_max_records_per_batch: int = Field(default=200, validation_alias="POSTGRES_UPSERT_MAX_RECORDS_PER_BATCH")
    postgres_delete_max_records_per_batch: int = Field(default=1000, validation_alias="POSTGRES_DELETE_MAX_RECORDS_PER_BATCH")
    neo4j_uri: Optional[str] = Field(default=None, validation_alias="NEO4J_URI")
    neo4j_username: Optional[str] = Field(default=None, validation_alias="NEO4J_USERNAME")
    neo4j_password: Optional[str] = Field(default=None, validation_alias="NEO4J_PASSWORD")
    neo4j_database: Optional[str] = Field(default=None, validation_alias="NEO4J_DATABASE")
    neo4j_max_connection_pool_size: int = Field(default=100, validation_alias="NEO4J_MAX_CONNECTION_POOL_SIZE")
    neo4j_connection_timeout: float = Field(default=30.0, validation_alias="NEO4J_CONNECTION_TIMEOUT")
    neo4j_connection_acquisition_timeout: float = Field(default=30.0, validation_alias="NEO4J_CONNECTION_ACQUISITION_TIMEOUT")
    neo4j_max_transaction_retry_time: float = Field(default=30.0, validation_alias="NEO4J_MAX_TRANSACTION_RETRY_TIME")
    neo4j_max_connection_lifetime: float = Field(default=300.0, validation_alias="NEO4J_MAX_CONNECTION_LIFETIME")
    neo4j_liveness_check_timeout: float = Field(default=30.0, validation_alias="NEO4J_LIVENESS_CHECK_TIMEOUT")
    neo4j_keep_alive: bool = Field(default=True, validation_alias="NEO4J_KEEP_ALIVE")

    @property
    def resolved_s3_region(self) -> Optional[str]:
        return self.s3_region

    @property
    def pg_async_pool_size(self) -> int:
        return max(4, self.postgres_max_connections // 5)

    def require(self, *roles: str) -> None:
        missing = []
        for role in roles:
            field, env = MODEL_ROLES[role]
            for suffix in ("model", "binding_host", "binding_api_key"):
                if not getattr(self, f"{field}_{suffix}"):
                    missing.append(f"{env}_{suffix.upper()}")
        if missing:
            raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

settings = RagSettings()