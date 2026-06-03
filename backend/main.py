"""FastAPI application entry point for the Customer Care Bot backend.

Phase 1 surface:
  * GET  /                    -> service banner
  * GET  /health              -> liveness (cheap, no external deps)
  * GET  /health/llm          -> tiny Gemini round-trip to verify key + SDK
  * POST /chat                -> streamed (SSE) agent turn; persists the session
  * GET  /sessions            -> list a customer's chat sessions (sidebar)
  * GET  /sessions/{id}       -> full transcript of one session (reopen a chat)

The app is stateless: conversation state lives in SQLite (memory/store.py) and is
rehydrated each turn. CORS origins come from config so the deployed Next.js
frontend (Vercel) can call the deployed backend (Render).
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.loop import agent_loop
from config import get_settings
from knowledge.retriever import retriever
from llm import LLMError, LLMNotConfigured, gemini_client
from memory.store import store

settings = get_settings()

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger("ccb.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Modern startup/shutdown hook (replaces deprecated on_event)."""
    store.init()  # create SQLite tables if missing
    logger.info(
        "Starting %s [env=%s, model=%s, gemini_configured=%s]",
        settings.app_name,
        settings.app_env,
        settings.gemini_model,
        settings.gemini_configured,
    )

    # Phase 2 grounding: load the policy retrieval index. If it's missing but a
    # key is present, best-effort build it so a fresh deploy self-ingests. Never
    # fatal — if grounding is unavailable, the bot just declines to guess.
    if retriever.load():
        logger.info("Retrieval index loaded (%d chunks).", retriever.size)
    elif settings.gemini_configured:
        try:
            n = await retriever.build()
            if n:
                retriever.save()
                logger.info("Built retrieval index at startup (%d chunks).", n)
        except Exception:  # noqa: BLE001
            logger.exception("Startup auto-ingest failed; continuing without grounding.")
    else:
        logger.warning(
            "No retrieval index and no API key — grounding disabled until "
            "`python -m knowledge.ingest` is run with a key set."
        )

    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title="Customer Care Bot API",
    version="0.1.0",
    description="Agentic customer care bot backend (Phase 1: streaming chat + sessions).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    # No cookie/credential auth yet (Phase 6). Keep this False so we never widen
    # the attack surface implicitly; flip to True only alongside real auth.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's message.")
    session_id: str | None = Field(
        default=None, description="Existing session id; omit to start a new chat."
    )
    customer_id: str | None = Field(
        default=None, description="Known customer id, if available."
    )


class HealthResponse(BaseModel):
    status: str
    app_env: str
    model: str
    retrieval_chunks: int


class LLMHealthResponse(BaseModel):
    status: str  # ok | not_configured | error
    model: str
    detail: str | None = None


class SessionSummaryOut(BaseModel):
    session_id: str
    customer_id: str | None
    title: str
    started_at: str
    updated_at: str
    message_count: int


class MessageOut(BaseModel):
    role: str
    content: str
    created_at: str | None = None


class SessionDetailOut(BaseModel):
    session_id: str
    title: str
    messages: list[MessageOut]


# ---------------------------------------------------------------------------
# Meta / health
# ---------------------------------------------------------------------------
@app.get("/", tags=["meta"])
async def root() -> dict[str, str]:
    return {"service": settings.app_name, "status": "ok", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_env=settings.app_env,
        model=settings.gemini_model,
        retrieval_chunks=retriever.size,
    )


@app.get("/health/llm", response_model=LLMHealthResponse, tags=["health"])
async def health_llm() -> LLMHealthResponse:
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
        return LLMHealthResponse(status="error", model=settings.gemini_model, detail=str(exc))


# ---------------------------------------------------------------------------
# Chat (streaming)
# ---------------------------------------------------------------------------
@app.post("/chat", tags=["chat"])
async def chat(request: ChatRequest) -> StreamingResponse:
    """Run one agent turn and stream the reply back as Server-Sent Events.

    Each SSE frame is ``data: {json}\\n\\n`` where json is one of:
      {"type":"token","content":"..."}            one chunk of the reply
      {"type":"done","session_id":..,"title":..}  end of turn
      {"type":"error","message":"..."}            something went wrong

    The frontend reads this with a fetch ReadableStream (not EventSource, since
    this is a POST). The user message is persisted before generation, so an
    interrupted turn still shows up in the session transcript.
    """
    session_id = request.session_id or f"sess_{uuid.uuid4().hex[:12]}"
    logger.info(
        "POST /chat session=%s customer=%s len=%d",
        session_id,
        request.customer_id,
        len(request.message),
    )

    async def event_stream() -> AsyncIterator[bytes]:
        agen = agent_loop.stream(
            session_id=session_id,
            customer_id=request.customer_id,
            message=request.message,
        )
        try:
            async for event in agen:
                yield f"data: {json.dumps(event)}\n\n".encode("utf-8")
        except Exception as exc:  # noqa: BLE001 - never leak a raw stack to the client
            logger.exception("Unhandled error during chat stream")
            err = {"type": "error", "message": f"Unexpected error: {exc}"}
            yield f"data: {json.dumps(err)}\n\n".encode("utf-8")
        finally:
            # Ensure the loop (and the underlying LLM stream) is closed even if
            # the client disconnects mid-stream — propagates aclose() downstream.
            await agen.aclose()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Disable proxy buffering (nginx/Render) so tokens flush immediately.
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Sessions (sidebar)
# ---------------------------------------------------------------------------
@app.get("/sessions", response_model=list[SessionSummaryOut], tags=["sessions"])
async def list_sessions(customer_id: str | None = None) -> list[SessionSummaryOut]:
    """List chat sessions (most recently updated first) for the sidebar."""
    sessions = await store.list_sessions(customer_id)
    return [SessionSummaryOut(**vars(s)) for s in sessions]


@app.get("/sessions/{session_id}", response_model=SessionDetailOut, tags=["sessions"])
async def get_session(session_id: str) -> SessionDetailOut:
    """Full transcript for one session (used when reopening a chat).

    NOTE (Phase 6): this does not yet enforce that the caller owns the session —
    there is no authentication layer in Phase 1 (single demo customer). Real
    multi-tenant isolation lands with auth in Phase 6; until then, adding a
    customer_id check here would be cosmetic (the client supplies its own id).
    """
    title = await store.get_title(session_id)
    if title is None:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = await store.get_messages(session_id)
    return SessionDetailOut(
        session_id=session_id,
        title=title,
        messages=[MessageOut(role=m.role, content=m.content, created_at=m.created_at) for m in messages],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
