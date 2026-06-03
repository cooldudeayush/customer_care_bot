"""Session summarization — the write side of long-term memory (Phase 6).

When a customer returns and starts a new chat, the previous session is digested
into a concise, evolving memory keyed by customer_id (one structured Gemini call).
On the next session it's injected as "what we know about this customer" so the bot
can proactively follow up ("Last time you reached out about a delayed order…").
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from llm import GeminiClient, LLMError, LLMNotConfigured, gemini_client

from .store import ConversationStore, CustomerMemoryRecord
from .store import store as default_store

logger = logging.getLogger("ccb.memory")


class CustomerMemory(BaseModel):
    summary: str = Field(
        description="A 2-4 sentence running profile of this customer and their history with us."
    )
    open_items: list[str] = Field(
        default_factory=list,
        description="Unresolved things to follow up on next time (empty if none).",
    )
    preferences: list[str] = Field(
        default_factory=list,
        description="Stated preferences or useful notes about the customer.",
    )
    sentiment: str = Field(default="neutral", description="Overall recent sentiment.")


_SYSTEM = """\
You maintain a concise long-term memory about a customer across support chats. Given
the EXISTING memory and a NEW conversation, produce an UPDATED memory that MERGES both:
- summary: who they are + the gist of their history (2-4 sentences, no fluff).
- open_items: anything still unresolved to follow up on. Remove items that the new
  conversation resolved; add new unresolved ones.
- preferences: durable preferences/notes worth remembering.
- sentiment: their overall recent sentiment.
Be factual and brief. Do not invent details that aren't in the conversation."""


def _existing_text(mem: CustomerMemoryRecord | None) -> str:
    if mem is None:
        return "(no existing memory)"
    parts = [f"Summary: {mem.summary or '(none)'}"]
    if mem.open_items:
        parts.append("Open items: " + "; ".join(mem.open_items))
    if mem.preferences:
        parts.append("Preferences: " + "; ".join(mem.preferences))
    if mem.sentiment:
        parts.append(f"Sentiment: {mem.sentiment}")
    return "\n".join(parts)


async def summarize_session(
    session_id: str,
    customer_id: str,
    *,
    store: ConversationStore | None = None,
    llm: GeminiClient | None = None,
) -> bool:
    """Digest one session into the customer's long-term memory. Returns True on
    success. Marks the session summarized either way (so we don't retry forever)."""
    store = store or default_store
    llm = llm or gemini_client

    messages = await store.get_messages(session_id)
    if not messages:
        await store.mark_summarized(session_id)
        return False

    transcript = "\n".join(
        f"{'Customer' if m.role == 'user' else 'Agent'}: {m.content}" for m in messages
    )
    existing = await store.get_customer_memory(customer_id)
    prompt = (
        f"EXISTING MEMORY:\n{_existing_text(existing)}\n\n"
        f"NEW CONVERSATION:\n{transcript}\n\n"
        "Produce the UPDATED memory."
    )

    try:
        mem = await llm.generate_structured(
            prompt, response_schema=CustomerMemory, system_instruction=_SYSTEM, temperature=0.3
        )
    except (LLMError, LLMNotConfigured):
        logger.info("Session summarization skipped (LLM unavailable) for %s", session_id)
        return False
    except Exception:  # noqa: BLE001
        logger.exception("Session summarization failed for %s", session_id)
        return False

    await store.upsert_customer_memory(
        customer_id, mem.summary, mem.open_items, mem.preferences, mem.sentiment
    )
    await store.mark_summarized(session_id)
    logger.info("Summarized session %s into memory for %s", session_id, customer_id)
    return True
