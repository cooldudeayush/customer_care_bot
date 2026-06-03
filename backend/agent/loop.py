"""The agent loop — the controller that runs on every user turn (Phase 3).

Flow per turn (~2 LLM calls — PERCEIVE + RESPOND — to respect the free tier):

  PERCEIVE  : one structured Gemini call -> {emotion, intents, action plan}
  DECIDE    : route ANSWER | ACT | CLARIFY | CONFIRM | ESCALATE; enforce the
              CONFIRM gate for money/irreversible tools; resolve a pending yes/no
  ACT       : execute the planned tools against the mock business systems
  RESPOND   : one streamed Gemini call, grounded in policy excerpts + tool results
  REMEMBER  : (Phase 6) long-term summary at session end

SSE events yielded:
  {"type":"token","content":...}                  reply text chunk
  {"type":"sources","sources":[...]}              policy docs that grounded the reply
  {"type":"tool","name","status","success"?}      a tool firing / finished
  {"type":"done","session_id","title","awaiting_confirmation"}
  {"type":"error","message":...}
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from knowledge.retriever import Retriever, retriever as default_retriever
from llm import GeminiClient, LLMError, LLMNotConfigured, gemini_client
from memory.store import ConversationStore, Message, store
from memory.summarize import summarize_session
from tools import registry

from .perception import Emotion, Perception, ToolCall, parse_args, perceive
from .prompts import build_respond_instruction, memory_block

logger = logging.getLogger("ccb.agent")

_NO_KEY_MSG = (
    "The bot isn't connected to Gemini yet. Add your GEMINI_API_KEY to "
    "backend/.env and restart the server."
)


def _to_contents(history: list[Message]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for m in history:
        role = "user" if m.role == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m.content}]})
    return contents


def _autotitle(message: str) -> str:
    title = " ".join(message.strip().split())
    return title[:40] or "New chat"


def _synthesize_confirm(tools: list[ToolCall]) -> str:
    """Fallback confirmation text if the model didn't supply confirm_message."""
    bits = []
    for tc in tools:
        args = parse_args(tc)
        if tc.name == "issue_refund":
            amt = args.get("amount")
            oid = args.get("order_id")
            amt_s = f" of Rs.{float(amt):.0f}" if amt is not None else ""
            bits.append(f"issue a refund{amt_s} for order #{oid}")
        elif tc.name == "cancel_order":
            bits.append(f"cancel order #{args.get('order_id')} and refund it")
        else:
            bits.append(f"run {tc.name}")
    action = " and ".join(bits) if bits else "make this change"
    return f"Just to confirm — I'll {action}. Shall I go ahead?"


class AgentLoop:
    def __init__(
        self,
        conversation_store: ConversationStore | None = None,
        llm: GeminiClient | None = None,
        retriever: Retriever | None = None,
    ) -> None:
        self.store = conversation_store or store
        self.llm = llm or gemini_client
        self.retriever = retriever or default_retriever

    # -- helpers ------------------------------------------------------------
    async def _retrieve(self, query: str) -> list:
        try:
            return await self.retriever.search(query)
        except Exception:  # noqa: BLE001 - retrieval must never break a turn
            logger.exception("Retrieval failed")
            return []

    # -- main ---------------------------------------------------------------
    async def stream(
        self, *, session_id: str, customer_id: str | None, message: str
    ) -> AsyncIterator[dict[str, Any]]:
        await self.store.ensure_session(session_id, customer_id)
        await self.store.add_message(session_id, "user", message)

        history = await self.store.get_messages(session_id)
        user_msgs = [m.content for m in history if m.role == "user"]
        if len(user_msgs) == 1:
            title = _autotitle(message)
            await self.store.set_title(session_id, title)
        else:
            title = await self.store.get_title(session_id) or "New chat"

        pending = await self.store.get_pending_action(session_id)

        # --- REMEMBER (load): cross-session memory (Phase 6) ----------------
        # On a brand-new session, first digest the customer's previous session
        # into long-term memory, so we can greet them with that context.
        if len(user_msgs) == 1 and customer_id:
            prior = await self.store.get_prior_unsummarized_session(customer_id, session_id)
            if prior:
                try:
                    await summarize_session(prior, customer_id, store=self.store, llm=self.llm)
                except Exception:  # noqa: BLE001 - memory is best-effort
                    logger.exception("Prior-session summarization failed")
        memory_rec = (
            await self.store.get_customer_memory(customer_id) if customer_id else None
        )
        memory_text = memory_block(memory_rec)

        # --- PERCEIVE -------------------------------------------------------
        try:
            perception = await perceive(
                history=history,
                customer_id=customer_id,
                pending_action=pending,
                memory_text=memory_text,
                llm=self.llm,
            )
        except LLMNotConfigured:
            logger.info("Chat attempted without a configured GEMINI_API_KEY")
            yield {"type": "error", "message": _NO_KEY_MSG}
            return
        except Exception:  # noqa: BLE001 - degrade to a plain answer on parse/LLM error
            logger.exception("PERCEIVE failed; falling back to ANSWER")
            perception = Perception(emotion=Emotion(), intents=[], action_type="ANSWER", tools=[])

        # Surface the detected emotion (the adaptive-tone read drives RESPOND).
        yield {
            "type": "emotion",
            "state": perception.emotion.state,
            "intensity": perception.emotion.intensity,
        }

        # --- DECIDE: resolve a pending confirmation, then route -------------
        tools_to_run: list[ToolCall] = []
        effective: str | None = None
        route = "fresh"  # "fresh" | "confirmed" — for the tool audit log

        if pending:
            await self.store.clear_pending_action(session_id)
            if perception.pending_decision == "confirm":
                try:
                    tools_to_run = [
                        ToolCall.model_validate(t) for t in pending.get("tools", [])
                    ]
                    effective = "ACT"
                    route = "confirmed"
                except Exception:  # noqa: BLE001 - corrupt pending shouldn't crash the turn
                    logger.exception("Could not restore pending action for %s", session_id)
                    tools_to_run = []
                    effective = "ANSWER"
            elif perception.pending_decision == "deny":
                effective = "DENY"
            # "none" -> fall through and handle the fresh request below

        if effective is None:
            effective = perception.action_type
            # Enforce the CONFIRM gate: money/irreversible tools must confirm first.
            if effective == "ACT" and any(
                registry.requires_confirmation(t.name) for t in perception.tools
            ):
                effective = "CONFIRM"
            elif effective == "CONFIRM" and not perception.tools:
                effective = "ANSWER"  # nothing concrete to confirm
            if effective == "ACT":
                tools_to_run = list(perception.tools)

        # --- Execute the route ---------------------------------------------
        reply_parts: list[str] = []
        awaiting_confirmation = False
        errored = False

        if effective == "CONFIRM":
            tools_to_run = list(perception.tools)
            confirm_text = perception.confirm_message or _synthesize_confirm(tools_to_run)
            await self.store.set_pending_action(
                session_id,
                {"tools": [t.model_dump() for t in tools_to_run], "summary": confirm_text},
            )
            awaiting_confirmation = True
            reply_parts.append(confirm_text)
            yield {"type": "token", "content": confirm_text}

        elif effective == "CLARIFY":
            text = perception.clarification_question or "Could you share a bit more detail so I can help?"
            reply_parts.append(text)
            yield {"type": "token", "content": text}

        elif effective == "DENY":
            text = "No problem — I won't make that change. Is there anything else I can help with?"
            reply_parts.append(text)
            yield {"type": "token", "content": text}

        elif effective == "ESCALATE":
            # Phase 7 builds the full handoff packet; here we log a ticket + reply.
            await asyncio.to_thread(
                registry.execute,
                "create_ticket",
                {
                    "customer_id": customer_id or "unknown",
                    "subject": "Escalation to a specialist",
                    "body": perception.reasoning or message,
                    "priority": "high",
                },
            )
            text = (
                "I'm connecting you with a specialist who can take this further. "
                "I've shared the full context, so you won't need to repeat anything."
            )
            reply_parts.append(text)
            yield {"type": "token", "content": text}

        else:
            # ANSWER or ACT -> run any tools, then a grounded RESPOND call.
            # tool_results is a LOCAL (not instance) var so concurrent requests
            # to the shared loop singleton never clobber each other's results.
            tool_results: list[dict[str, Any]] = []
            if effective == "ACT" and tools_to_run:
                for tc in tools_to_run:
                    args = parse_args(tc)
                    yield {"type": "tool", "name": tc.name, "status": "running", "args": args}
                    result = await asyncio.to_thread(registry.execute, tc.name, args)
                    # Audit log: every action (esp. money/irreversible) leaves a record.
                    logger.info(
                        "TOOL_EXEC session=%s route=%s tool=%s args=%s success=%s",
                        session_id, route, tc.name, args, result.get("success"),
                    )
                    yield {
                        "type": "tool",
                        "name": tc.name,
                        "status": "done",
                        "success": bool(result.get("success")),
                    }
                    tool_results.append({"name": tc.name, "args": args, "result": result})

            retrieval_query = " ".join(user_msgs[-2:]) if user_msgs else message
            doc_chunks = await self._retrieve(retrieval_query)
            if doc_chunks:
                yield {
                    "type": "sources",
                    "sources": [{"source": c.source, "heading": c.heading} for c in doc_chunks],
                }

            system_instruction = build_respond_instruction(
                doc_chunks,
                tool_results=tool_results,
                emotion=perception.emotion,
                memory_text=memory_text,
            )
            contents = _to_contents(history)
            try:
                async for token in self.llm.stream(contents, system_instruction=system_instruction):
                    reply_parts.append(token)
                    yield {"type": "token", "content": token}
            except LLMNotConfigured:
                yield {"type": "error", "message": _NO_KEY_MSG}
                errored = True
            except LLMError as exc:
                logger.exception("LLM error during RESPOND for session %s", session_id)
                yield {"type": "error", "message": f"Sorry — I hit a problem: {exc}"}
                errored = True

        if errored:
            return

        reply = "".join(reply_parts).strip()
        if reply:
            await self.store.add_message(session_id, "bot", reply)

        yield {
            "type": "done",
            "session_id": session_id,
            "title": title,
            "awaiting_confirmation": awaiting_confirmation,
        }


# Process-wide loop instance used by the /chat endpoint.
agent_loop = AgentLoop()
