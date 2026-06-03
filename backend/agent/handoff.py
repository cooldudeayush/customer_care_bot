"""Graceful escalation — the handoff packet (Phase 7).

When the bot reaches its limits (out of scope, a legal/edge case, repeated
dissatisfaction, an explicit request for a human, or unresolving high anger), it
hands off to a human specialist WITH a full packet so the customer never has to
repeat themselves. The packet = {who the customer is, the issue, a conversation
summary, actions already taken, a suggested next step, current sentiment} plus a
warm, tone-aware message to show the customer.

Degrades gracefully: if the LLM is unavailable, a deterministic fallback packet
is built from the transcript so escalation always works.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from llm import GeminiClient, gemini_client
from memory.store import Message

logger = logging.getLogger("ccb.handoff")


class HandoffPacket(BaseModel):
    issue: str = Field(description="The core problem in one clear line.")
    conversation_summary: str = Field(
        description="A concise summary of what happened in this chat."
    )
    actions_taken: list[str] = Field(
        default_factory=list,
        description="Actions the bot already took this session (refunds, lookups, etc.).",
    )
    suggested_next_step: str = Field(
        description="What the human specialist should do next."
    )
    sentiment: str = Field(default="neutral", description="The customer's current sentiment.")
    customer_message: str = Field(
        description="A warm, tone-matched message to show the customer about the handoff."
    )


_SYSTEM = """\
You are preparing a HANDOFF PACKET to pass a customer to a human specialist. From
the conversation and customer context, produce a packet so the specialist has full
context and the customer NEVER has to repeat themselves. Be concise and factual.
The customer_message should reassure them warmly (match their mood) and make clear
a specialist is taking over with full context — do NOT ask them to repeat anything."""


def _transcript(history: list[Message]) -> str:
    return "\n".join(
        f"{'Customer' if m.role == 'user' else 'Agent'}: {m.content}" for m in history
    )


def build_customer_summary(snapshot: dict, memory) -> str:
    """Deterministic 'who is this customer' line from the snapshot + memory."""
    parts: list[str] = []
    if snapshot and snapshot.get("known"):
        parts.append(f"{snapshot['name']} (id={snapshot['customer_id']}, tier={snapshot['tier']})")
        n = len(snapshot.get("orders", []))
        if n:
            parts.append(f"{n} order(s) on file")
    summary = getattr(memory, "summary", None)
    if summary:
        parts.append(f"History: {summary}")
    return ". ".join(parts) if parts else "Customer details unavailable."


async def build_handoff_packet(
    *,
    history: list[Message],
    snapshot: dict,
    memory,
    emotion,
    reason: str,
    llm: GeminiClient | None = None,
) -> HandoffPacket:
    """Generate the handoff packet via one structured call."""
    client = llm or gemini_client
    em = (
        f"{getattr(emotion, 'state', 'neutral')} (intensity {getattr(emotion, 'intensity', 1)}/5)"
    )
    prompt = (
        f"CUSTOMER: {build_customer_summary(snapshot, memory)}\n"
        f"CURRENT EMOTION: {em}\n"
        f"REASON FOR ESCALATION: {reason}\n\n"
        f"CONVERSATION:\n{_transcript(history)}\n\n"
        "Produce the handoff packet."
    )
    return await client.generate_structured(
        prompt, response_schema=HandoffPacket, system_instruction=_SYSTEM, temperature=0.3
    )


def fallback_packet(message: str, history: list[Message]) -> HandoffPacket:
    """Deterministic packet when the LLM is unavailable — escalation still works."""
    return HandoffPacket(
        issue=(message or "Customer needs specialist assistance").strip()[:140],
        conversation_summary=_transcript(history)[:1500],
        actions_taken=[],
        suggested_next_step="Review the conversation and assist the customer directly.",
        sentiment="unknown",
        customer_message=(
            "I'm connecting you with a specialist who can take this further. "
            "I've shared the full context of our chat, so you won't need to repeat anything."
        ),
    )
