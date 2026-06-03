"""Single configuration module for the Customer Care Bot backend.

All runtime configuration is read from environment variables (and an optional
local ``.env`` file) via ``pydantic-settings``. Nothing here is hardcoded as a
secret -- the only literals are safe local-first defaults.

Forward-compat: keys for later phases (Neo4j graph, vector store, long-term
memory DB) are present now so feature code can plug in without touching wiring.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, populated from the environment / ``.env``.

    Field names are lowercase; env vars are matched case-insensitively, so
    ``GEMINI_API_KEY`` in the environment maps to ``gemini_api_key`` here.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # tolerate forward-declared keys not yet modeled
    )

    # ---- App / environment -------------------------------------------------
    app_env: str = Field(default="local", description="local | staging | production")
    app_name: str = Field(default="customer-care-bot")
    log_level: str = Field(default="INFO")

    # ---- LLM (Gemini Flash) ------------------------------------------------
    # No default for the key: absence is detected at runtime by /health/llm so
    # the app still boots (and /health stays green) without a key configured.
    gemini_api_key: str = Field(default="", description="Google AI Studio API key")
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description=(
            "Flash model id for the current google-genai SDK. "
            "Alternatives: 'gemini-flash-latest' (tracks newest), "
            "'gemini-2.0-flash'. Override via env."
        ),
    )
    # 2-LLM-calls-per-turn discipline + free-tier (~15 rpm) backoff knobs.
    llm_request_timeout_s: float = Field(default=30.0)
    llm_max_retries: int = Field(default=4)
    llm_backoff_base_s: float = Field(default=1.0)

    # ---- Relational / mock-business store (Phase 3) ------------------------
    # Local-first default: a file-based SQLite DB. Swap to Postgres/Supabase by
    # changing this URL only.
    database_url: str = Field(default="sqlite:///./data/app.db")

    # ---- Customer operational graph (Phase 4: Neo4j-in-Docker, later Aura) --
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="")

    # ---- Document retrieval (Phase 2: LightRAG / Chroma local) -------------
    vector_store: str = Field(default="chroma", description="chroma | lightrag")
    corpus_dir: str = Field(default="./data/corpus")

    # ---- CORS --------------------------------------------------------------
    # Comma-separated list in the env, e.g.
    #   CORS_ORIGINS=http://localhost:3000,https://my-app.vercel.app
    # ``NoDecode`` stops pydantic-settings from JSON-decoding the raw env value
    # so our validator below can accept a plain comma-separated string.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        description="Allowed browser origins for the Next.js frontend.",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors(cls, value: object) -> object:
        """Accept either a JSON list or a plain comma-separated string."""
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            # Already JSON-list-ish? Let pydantic handle it.
            if stripped.startswith("["):
                return value
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return value

    @property
    def gemini_configured(self) -> bool:
        """True when a non-placeholder API key is present."""
        key = self.gemini_api_key.strip()
        return bool(key) and not key.lower().startswith("your-")


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide ``Settings`` instance.

    Use as a FastAPI dependency (``Depends(get_settings)``) or import-and-call.
    The cache means the ``.env`` file is parsed exactly once per process.
    """
    return Settings()


# Convenience module-level singleton for simple imports.
settings = get_settings()
