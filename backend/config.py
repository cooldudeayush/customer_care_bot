"""Single configuration module for the Customer Care Bot backend.

All runtime configuration is read from environment variables (and an optional
local ``.env`` file) via ``pydantic-settings``. Nothing here is hardcoded as a
secret -- the only literals are safe local-first defaults.

Forward-compat: keys for later phases (Neo4j graph, vector store, long-term
memory DB) are present now so feature code can plug in without touching wiring.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Anchor for resolving relative data paths. Computed from this file's location so
# paths work regardless of the process's current working directory (a subprocess,
# IDE, or deploy launcher could start us from anywhere).
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))


def _anchor(path: str) -> str:
    """Resolve a possibly-relative path against the backend/ directory."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(_BACKEND_DIR, path))


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

    # ---- Embeddings (Phase 2 retrieval) ------------------------------------
    # Current unified-SDK embedding model. 768 dims (truncated + normalized) keeps
    # storage/compute lean for a small policy corpus; cosine sim needs normalized
    # vectors anyway. Bump dims via env if you want more fidelity.
    gemini_embed_model: str = Field(default="gemini-embedding-001")
    embed_dim: int = Field(default=768)

    # ---- Relational / mock-business store (Phase 3) ------------------------
    # Local-first default: a file-based SQLite DB. Swap to Postgres/Supabase by
    # changing this URL only.
    database_url: str = Field(default="sqlite:///./data/app.db")

    # ---- Customer operational graph (Phase 4: Neo4j-in-Docker, later Aura) --
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="")

    # ---- Document retrieval (Phase 2) --------------------------------------
    # 'vector' = reliable local Gemini-embeddings + cosine store (default, no
    # heavy deps). 'lightrag' is a documented swap-in behind the same Retriever
    # interface if/when its heavier ingestion is worth it.
    vector_store: str = Field(default="vector", description="vector | lightrag")
    corpus_dir: str = Field(default="../data/corpus")
    vector_index_path: str = Field(default="../data/vectorstore/index.json")
    retrieval_top_k: int = Field(default=4)

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

    @property
    def corpus_path(self) -> str:
        """Absolute corpus dir (cwd-independent)."""
        return _anchor(self.corpus_dir)

    @property
    def index_path(self) -> str:
        """Absolute vector-index path (cwd-independent)."""
        return _anchor(self.vector_index_path)


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide ``Settings`` instance.

    Use as a FastAPI dependency (``Depends(get_settings)``) or import-and-call.
    The cache means the ``.env`` file is parsed exactly once per process.
    """
    return Settings()


# Convenience module-level singleton for simple imports.
settings = get_settings()
