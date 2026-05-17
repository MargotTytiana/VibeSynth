# =============================================================================
# File: backend/app/config.py
# Purpose: Centralised application configuration loaded from environment
#          variables. All other modules import settings from here — never
#          read os.environ directly elsewhere in the codebase.
# =============================================================================

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings resolved from environment variables or a .env file.
    Pydantic-settings validates types and raises on startup if required
    variables are missing, preventing silent misconfiguration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # LLM provider
    # ------------------------------------------------------------------
    llm_provider: str = "openai"           # "openai" | "anthropic" | "local"
    llm_model: str = "gpt-4o"             # model identifier passed to the API
    llm_api_key: str = ""                  # loaded from LLM_API_KEY in .env
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.2           # low temp for deterministic code gen

    # ------------------------------------------------------------------
    # Compilation sandbox
    # ------------------------------------------------------------------
    wasm_compile_timeout_ms: int = 5000    # abort compilation after this limit
    sandbox_max_loop_iterations: int = 512 # loop-unroll ceiling for shaders

    # ------------------------------------------------------------------
    # Vector cache (similarity search)
    # ------------------------------------------------------------------
    cache_similarity_threshold: float = 0.92  # cosine distance cutoff for a hit
    cache_max_entries: int = 1000
    cache_index_path: str = "cache/cache_index.json"

    # ------------------------------------------------------------------
    # API server
    # ------------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    api_cors_origins: list[str] = ["http://localhost:5173"]  # Vite dev server

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = "INFO"     # DEBUG | INFO | WARNING | ERROR
    log_compile_csv: str = "compiler/compile_log.csv"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton Settings instance.
    Use this in FastAPI dependency injection:

        from app.config import get_settings
        settings = get_settings()
    """
    return Settings()