"""Central settings, loaded from environment / .env."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # LLM (optional). Without a key the agents run in deterministic mode.
    google_api_key: str | None = None
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.2
    use_llm: bool = True  # set False to force deterministic agents even with a key

    # Vector store
    chroma_dir: str = str(BASE_DIR / ".chroma")
    embedding_backend: str = "hash"  # "hash" (offline) or "gemini"
    knowledge_file: str = str(BASE_DIR / "app" / "knowledge" / "schemes.json")

    # Self-correction loop
    max_iterations: int = 4
    min_dscr: float = 1.25
    nbfc_share_warning: float = 0.30
    cost_tolerance_pct: float = 1.0

    # Langfuse (optional)
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.use_llm and self.google_api_key)

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
