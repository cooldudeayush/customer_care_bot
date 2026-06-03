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
