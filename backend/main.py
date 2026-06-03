"""FastAPI application entry point for the Customer Care Bot backend.

Phase 0 surface:
  * GET  /            -> service banner
  * GET  /health      -> liveness (always cheap, no external deps)
  * GET  /health/llm  -> tiny Gemini round-trip to verify key + SDK
  * POST /chat        -> stub placeholder reply (real agent loop lands in Phase 1)

The app is stateless: any persistent state (sessions, memory, graph) is
externalized in later phases. CORS origins are read from config so the deployed
Next.js frontend (Vercel) can call the deployed backend (Render).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import get_settings
from llm import LLMError, LLMNotConfigured, gemini_client

settings = get_settings()

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger("ccb.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Modern startup/shutdown hook (replaces deprecated on_event)."""
    logger.info(
        "Starting %s [env=%s, model=%s, gemini_configured=%s]",
        settings.app_name,
        settings.app_env,
        settings.gemini_model,
        settings.gemini_configured,
    )
    # Future phases: open DB pool, Neo4j driver, vector store, warm caches here.
    yield
    # Future phases: close drivers / flush traces here.
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title="Customer Care Bot API",
    version="0.0.1",
    description="Agentic customer care bot backend (Phase 0 skeleton).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """Incoming chat turn. Matches the Phase 1 contract so the shape is stable."""

    message: str = Field(..., min_length=1, description="The user's message.")
    session_id: str | None = Field(
        default=None, description="Existing session id; omit to start a new chat."
    )
    customer_id: str | None = Field(
        default=None, description="Known customer id, if authenticated."
    )


class ChatResponse(BaseModel):
    reply: str
    session_id: str | None = None
    placeholder: bool = True


class HealthResponse(BaseModel):
    status: str
    app_env: str
    model: str


class LLMHealthResponse(BaseModel):
    status: str  # ok | not_configured | error
    model: str
    detail: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["meta"])
async def root() -> dict[str, str]:
    return {"service": settings.app_name, "status": "ok", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Cheap liveness check -- never calls any external service."""
    return HealthResponse(
        status="ok",
        app_env=settings.app_env,
        model=settings.gemini_model,
    )


@app.get("/health/llm", response_model=LLMHealthResponse, tags=["health"])
async def health_llm() -> LLMHealthResponse:
    """Verify the Gemini key + SDK with a tiny generation call.

    Never raises: reports configuration/upstream problems in the body so a
    monitoring probe gets a 200 with a clear status string.
    """
    if not settings.gemini_configured:
        return LLMHealthResponse(
            status="not_configured",
            model=settings.gemini_model,
            detail="GEMINI_API_KEY is missing or a placeholder.",
        )
    try:
        reply = await gemini_client.ping()
        return LLMHealthResponse(status="ok", model=settings.gemini_model, detail=reply)
    except LLMNotConfigured as exc:
        return LLMHealthResponse(
            status="not_configured", model=settings.gemini_model, detail=str(exc)
        )
    except LLMError as exc:
        logger.exception("LLM health check failed")
        return LLMHealthResponse(
            status="error", model=settings.gemini_model, detail=str(exc)
        )


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """Phase 0 stub.

    Returns a fixed placeholder so the frontend can wire end-to-end. The real
    PERCEIVE->RETRIEVE->DECIDE->ACT->RESPOND->REMEMBER loop lands in Phase 1 and
    will be invoked from here without changing this signature.
    """
    logger.info(
        "POST /chat session=%s customer=%s len=%d",
        request.session_id,
        request.customer_id,
        len(request.message),
    )
    return ChatResponse(
        reply=(
            "Hi! I'm the Customer Care Bot (Phase 0 skeleton). "
            "I can't help for real just yet -- the agent loop ships in Phase 1."
        ),
        session_id=request.session_id,
        placeholder=True,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
