"""Thin Gemini client wrapper (unified google-genai SDK).

Goals for Phase 0:
  * A single place that configures the **current** unified Google GenAI SDK
    (``from google import genai`` / ``pip install google-genai``).
    NOTE: the older ``google-generativeai`` package is DEPRECATED -- do not use it.
  * ``generate()`` -> plain text, ``generate_json()`` -> parsed dict (ready for the
    structured PERCEIVE/DECIDE call in Phase 3 via response_schema).
  * ``stream()`` -> async token iterator (used by the SSE endpoint in Phase 1).
  * Exponential backoff + jitter on rate-limit (HTTP 429 / RESOURCE_EXHAUSTED),
    since the Gemini free tier is ~15 req/min and the turn design budgets 2 calls.
  * Async-friendly: the SDK's sync calls are offloaded to threads so they never
    block the FastAPI event loop.

This wrapper owns transport + resilience only; agent/emotion phases own prompts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any, AsyncIterator

from google import genai
from google.genai import types

from config import Settings, get_settings

logger = logging.getLogger("ccb.llm")


class LLMError(RuntimeError):
    """Raised when the LLM cannot satisfy a request (config or upstream)."""


class LLMNotConfigured(LLMError):
    """Raised when no usable Gemini API key is configured."""


class LLMRateLimited(LLMError):
    """Raised when the model is rate-limited / out of quota (HTTP 429).

    ``retry_after`` is the suggested wait in seconds (parsed from the error),
    so the UI can tell the customer exactly how long to wait.
    """

    def __init__(self, message: str, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _retry_after_seconds(exc: Exception) -> int | None:
    """Extract the suggested retry delay (seconds) from a Gemini 429 error."""
    import re

    s = str(exc)
    for pattern in (
        r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s",  # 'retryDelay': '56s'
        r"retry in (\d+(?:\.\d+)?)\s*s",  # Please retry in 56.5s
    ):
        m = re.search(pattern, s)
        if m:
            try:
                return int(float(m.group(1))) + 1  # round up a touch
            except ValueError:
                return None
    return None


def _is_rate_limit(exc: Exception) -> bool:
    """Best-effort detection of a 429 / quota-exhausted error.

    The unified SDK raises ``google.genai.errors.ClientError`` with a 429 status
    for quota issues. We sniff the type name + message to stay dependency-light
    and resilient to SDK internals.
    """
    name = type(exc).__name__.lower()
    if "resourceexhausted" in name or "toomanyrequests" in name:
        return True
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "resource_exhausted" in text or "rate limit" in text or "quota" in text


class GeminiClient:
    """Minimal async wrapper around a configured Gemini model (google-genai)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: genai.Client | None = None

    # -- lifecycle ----------------------------------------------------------
    def _ensure_client(self) -> genai.Client:
        if not self._settings.gemini_configured:
            raise LLMNotConfigured(
                "GEMINI_API_KEY is missing or still a placeholder. "
                "Set it in backend/.env (see backend/.env.example)."
            )
        if self._client is None:
            self._client = genai.Client(api_key=self._settings.gemini_api_key)
        return self._client

    def _build_config(
        self,
        *,
        system_instruction: str | None,
        temperature: float,
        max_output_tokens: int | None,
        response_mime_type: str | None,
        response_schema: Any | None,
    ) -> types.GenerateContentConfig:
        kwargs: dict[str, Any] = {"temperature": temperature}
        if system_instruction:
            kwargs["system_instruction"] = system_instruction
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        if response_mime_type is not None:
            kwargs["response_mime_type"] = response_mime_type
        if response_schema is not None:
            kwargs["response_schema"] = response_schema
        return types.GenerateContentConfig(**kwargs)

    async def _with_backoff(self, fn) -> Any:
        """Run a zero-arg callable with exponential backoff + jitter on 429s."""
        last_exc: Exception | None = None
        for attempt in range(self._settings.llm_max_retries + 1):
            try:
                return await asyncio.to_thread(fn)
            except Exception as exc:  # noqa: BLE001 - normalize SDK errors
                last_exc = exc
                if _is_rate_limit(exc) and attempt < self._settings.llm_max_retries:
                    delay = self._settings.llm_backoff_base_s * (2**attempt)
                    delay += random.uniform(0, delay * 0.25)  # jitter
                    logger.warning(
                        "Gemini rate-limited (attempt %d/%d); retrying in %.2fs",
                        attempt + 1,
                        self._settings.llm_max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                break
        if last_exc is not None and _is_rate_limit(last_exc):
            raise LLMRateLimited(
                f"Gemini rate limit: {last_exc}",
                retry_after=_retry_after_seconds(last_exc),
            ) from last_exc
        raise LLMError(f"Gemini call failed: {last_exc}") from last_exc

    # -- core generation ----------------------------------------------------
    async def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.7,
        max_output_tokens: int | None = None,
        response_mime_type: str | None = None,
        response_schema: Any | None = None,
    ) -> str:
        """Generate text for ``prompt`` and return the model's reply string."""
        client = self._ensure_client()
        config = self._build_config(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
        )

        def _call():
            return client.models.generate_content(
                model=self._settings.gemini_model,
                contents=prompt,
                config=config,
            )

        response = await self._with_backoff(_call)
        return (getattr(response, "text", None) or "").strip()

    async def generate_json(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.2,
        response_schema: Any | None = None,
    ) -> dict[str, Any]:
        """Structured-output helper for the PERCEIVE/DECIDE call.

        Asks Gemini for ``application/json`` (optionally constrained by a
        ``response_schema`` -- a pydantic model or genai Schema) and parses the
        result into a dict. Raises ``LLMError`` on invalid JSON.
        """
        raw = await self.generate(
            prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=response_schema,
        )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Model did not return valid JSON: {raw[:200]!r}") from exc
        if not isinstance(parsed, dict):
            raise LLMError(f"Expected a JSON object, got {type(parsed).__name__}.")
        return parsed

    async def generate_structured(
        self,
        contents: Any,
        *,
        response_schema: Any,
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> Any:
        """Controlled generation against a Pydantic ``response_schema``.

        Returns an instance of ``response_schema`` (via the SDK's ``.parsed``),
        falling back to validating ``.text`` if needed. Used by the single
        PERCEIVE/DECIDE call to get {emotion, intents, entities, tool plan}.
        """
        client = self._ensure_client()
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        def _call():
            return client.models.generate_content(
                model=self._settings.gemini_model, contents=contents, config=config
            )

        resp = await self._with_backoff(_call)
        parsed = getattr(resp, "parsed", None)
        if parsed is not None:
            return parsed
        text = getattr(resp, "text", None) or ""
        if hasattr(response_schema, "model_validate_json"):
            try:
                return response_schema.model_validate_json(text)
            except Exception as exc:  # noqa: BLE001
                raise LLMError(f"Structured output did not match schema: {text[:200]!r}") from exc
        raise LLMError("Structured generation returned no parsable output.")

    async def stream(
        self,
        contents: Any,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Yield reply text chunk-by-chunk (for SSE streaming in Phase 1).

        ``contents`` may be a plain prompt string OR a multi-turn list of
        ``{"role": "user"|"model", "parts": [{"text": ...}]}`` dicts (the SDK
        coerces both). The agent loop passes the full transcript this way so the
        stateless backend gets proper multi-turn context every call.

        The SDK's streaming generator is sync; we drain it on a worker thread and
        hand chunks back to the event loop one at a time. Rate-limit backoff is
        applied to establishing the stream (the first chunk).
        """
        client = self._ensure_client()
        config = self._build_config(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=None,
            response_mime_type=None,
            response_schema=None,
        )

        sentinel = object()

        def _next(gen):
            try:
                return next(gen)
            except StopIteration:
                return sentinel

        def _open_and_first():
            # Opening the stream is lazy — the HTTP request actually fires on the
            # first next(). So we open AND pull the first chunk under one backoff
            # guard: a 429 here retries by cleanly re-opening the whole stream.
            gen = client.models.generate_content_stream(
                model=self._settings.gemini_model,
                contents=contents,
                config=config,
            )
            return gen, _next(gen)

        generator, chunk = await self._with_backoff(_open_and_first)
        try:
            while chunk is not sentinel:
                text = getattr(chunk, "text", None)
                if text:
                    yield text
                chunk = await asyncio.to_thread(_next, generator)
        finally:
            # Close the sync SDK generator even if the client disconnects mid-
            # stream (otherwise the worker thread / SDK resources can leak).
            close = getattr(generator, "close", None)
            if callable(close):
                await asyncio.to_thread(close)

    async def embed_texts(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Embed a list of texts. Returns one raw vector per input (in order).

        Batched (free-tier friendly) with the same 429 backoff as generation.
        ``task_type`` should be 'RETRIEVAL_DOCUMENT' for corpus chunks and
        'RETRIEVAL_QUERY' for the user's query (improves retrieval quality).
        Vectors are returned un-normalized; the retriever normalizes for cosine.
        """
        if not texts:
            return []
        client = self._ensure_client()
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            def _call(b: list[str] = batch):
                return client.models.embed_content(
                    model=self._settings.gemini_embed_model,
                    contents=b,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self._settings.embed_dim,
                    ),
                )

            resp = await self._with_backoff(_call)
            embeddings = getattr(resp, "embeddings", None)
            if embeddings is None:  # some SDK paths return singular .embedding
                single = getattr(resp, "embedding", None)
                embeddings = [single] if single is not None else []
            out.extend([list(e.values) for e in embeddings])
        return out

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (RETRIEVAL_QUERY task type)."""
        vecs = await self.embed_texts([text], task_type="RETRIEVAL_QUERY")
        return vecs[0] if vecs else []

    async def ping(self) -> str:
        """Tiny liveness call used by GET /health/llm to verify key + SDK."""
        return await self.generate(
            "Reply with the single word: pong",
            temperature=0.0,
            max_output_tokens=8,
        )


# Process-wide default client. Cheap to construct; configures the SDK lazily on
# first real call, so importing this never requires a key to be present.
gemini_client = GeminiClient()
