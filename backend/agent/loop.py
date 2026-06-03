"""The agent loop — the controller that runs on every user turn.

Phase 1 implements a thin slice of the full loop:

    PERCEIVE -> RETRIEVE -> DECIDE -> ACT -> RESPOND -> REMEMBER

Only RESPOND does real work right now (a single streamed Gemini call grounded in
the running transcript). The other stages are explicit no-op markers so later
phases drop in without reshaping this function:
  * PERCEIVE  (Phase 3/5): one structured call -> {emotion, intents, entities, tools}
  * RETRIEVE  (Phase 2/4): LightRAG doc chunks + Neo4j graph traversal
  * DECIDE    (Phase 3):   ANSWER | ACT | CLARIFY | CONFIRM | ESCALATE
  * ACT       (Phase 3):   execute tools, write back to the graph
  * REMEMBER  (Phase 6):   long-term session summary at session end

The loop yields a stream of event dicts so the endpoint can forward them as SSE:
  {"type": "token",  "content": "..."}    # one chunk of the reply
  {"type": "done",   "session_id", "title"}
  {"type": "error",  "message": "..."}
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from llm import GeminiClient, LLMError, LLMNotConfigured, gemini_client
from memory.store import ConversationStore, Message, store

from .prompts import SYSTEM_PROMPT

logger = logging.getLogger("ccb.agent")


def _to_contents(history: list[Message]) -> list[dict[str, Any]]:
    """Convert stored transcript -> Gemini multi-turn ``contents`` list.

    Roles map: our 'user' -> 'user', our 'bot' -> 'model'.
    """
    contents: list[dict[str, Any]] = []
    for m in history:
        role = "user" if m.role == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m.content}]})
    return contents


def _autotitle(message: str) -> str:
    title = " ".join(message.strip().split())
    return (title[:40] or "New chat")


class AgentLoop:
    def __init__(
        self,
        conversation_store: ConversationStore | None = None,
        llm: GeminiClient | None = None,
    ) -> None:
        self.store = conversation_store or store
        self.llm = llm or gemini_client

    async def stream(
        self, *, session_id: str, customer_id: str | None, message: str
    ) -> AsyncIterator[dict[str, Any]]:
        """Run one turn, yielding SSE-ready event dicts."""
        # --- persist the incoming turn -------------------------------------
        await self.store.ensure_session(session_id, customer_id)
        await self.store.add_message(session_id, "user", message)

        # Title the chat from its first user message (drives the sidebar label).
        is_first_turn = (await self.store.count_user_messages(session_id)) == 1
        if is_first_turn:
            title = _autotitle(message)
            await self.store.set_title(session_id, title)
        else:
            title = await self.store.get_title(session_id) or "New chat"

        # --- PERCEIVE / RETRIEVE / DECIDE / ACT --------------------------------
        # (Phase 2-5) — no-ops in Phase 1. The RESPOND step below is grounded
        # only in the conversation transcript for now.

        # --- RESPOND: one streamed Gemini call ------------------------------
        history = await self.store.get_messages(session_id)
        contents = _to_contents(history)

        chunks: list[str] = []
        try:
            async for token in self.llm.stream(
                contents, system_instruction=SYSTEM_PROMPT
            ):
                chunks.append(token)
                yield {"type": "token", "content": token}
        except LLMNotConfigured:
            logger.info("Chat attempted without a configured GEMINI_API_KEY")
            yield {
                "type": "error",
                "message": (
                    "The bot isn't connected to Gemini yet. Add your GEMINI_API_KEY "
                    "to backend/.env and restart the server."
                ),
            }
            return
        except LLMError as exc:
            logger.exception("LLM error during RESPOND for session %s", session_id)
            yield {"type": "error", "message": f"Sorry — I hit a problem: {exc}"}
            return

        reply = "".join(chunks).strip()
        if reply:
            await self.store.add_message(session_id, "bot", reply)

        # --- REMEMBER (Phase 6): long-term summary happens at session end. ---
        yield {"type": "done", "session_id": session_id, "title": title}


# Process-wide loop instance used by the /chat endpoint.
agent_loop = AgentLoop()
