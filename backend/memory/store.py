"""Conversation persistence — the short-term memory store.

Backs the chat sidebar (list past sessions, reopen a transcript) and supplies
the agent loop with the running transcript for multi-turn context.

Async methods wrap the synchronous sqlite layer (db.py) via ``asyncio.to_thread``
so the FastAPI event loop stays responsive. Phase 6 will add long-term
cross-session summaries alongside this.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from .db import connect, init_db

logger = logging.getLogger("ccb.store")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _loads_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        data = json.loads(value)
        return data if isinstance(data, list) else []
    except (ValueError, TypeError):
        logger.warning("Corrupt JSON list in customer memory; ignoring: %.80r", value)
        return []


@dataclass
class Message:
    role: str  # 'user' | 'bot'
    content: str
    created_at: str | None = None


@dataclass
class SessionSummary:
    session_id: str
    customer_id: str | None
    title: str
    started_at: str
    updated_at: str
    message_count: int = 0


@dataclass
class CustomerMemoryRecord:
    customer_id: str
    summary: str | None
    open_items: list[str]
    preferences: list[str]
    sentiment: str | None
    last_interaction: str | None


@dataclass
class HandoffRecord:
    id: int
    session_id: str | None
    customer_id: str | None
    customer_summary: str | None
    issue: str | None
    conversation_summary: str | None
    actions_taken: list[str]
    suggested_next_step: str | None
    sentiment: str | None
    status: str
    created_at: str


@dataclass
class TraceRecord:
    id: int
    session_id: str | None
    customer_id: str | None
    turn_no: int | None
    emotion_state: str | None
    emotion_intensity: int | None
    action: str | None
    intents: list[str]
    tools: list[dict]
    sources: list[str]
    reply_len: int | None
    latency_ms: int | None
    errored: bool
    created_at: str


class ConversationStore:
    """CRUD over conversations + messages. All public methods are async."""

    def init(self) -> None:
        """Create the schema. Run once at startup (sync — fast)."""
        init_db()

    # -- sessions -----------------------------------------------------------
    async def ensure_session(
        self, session_id: str, customer_id: str | None
    ) -> None:
        """Insert the conversation row if it doesn't exist yet (idempotent)."""

        def _op() -> None:
            with connect() as conn:
                row = conn.execute(
                    "SELECT 1 FROM conversations WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if row is None:
                    now = _now()
                    conn.execute(
                        "INSERT INTO conversations "
                        "(session_id, customer_id, title, started_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (session_id, customer_id, "New chat", now, now),
                    )
                    conn.commit()

        await asyncio.to_thread(_op)

    async def set_title(self, session_id: str, title: str) -> None:
        def _op() -> None:
            with connect() as conn:
                conn.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? "
                    "WHERE session_id = ?",
                    (title, _now(), session_id),
                )
                conn.commit()

        await asyncio.to_thread(_op)

    async def get_title(self, session_id: str) -> str | None:
        def _op() -> str | None:
            with connect() as conn:
                row = conn.execute(
                    "SELECT title FROM conversations WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                return row["title"] if row else None

        return await asyncio.to_thread(_op)

    async def list_sessions(self, customer_id: str | None) -> list[SessionSummary]:
        """Most-recently-updated first — drives the sidebar order."""

        def _op() -> list[SessionSummary]:
            with connect() as conn:
                if customer_id is None:
                    rows = conn.execute(
                        "SELECT c.*, "
                        "(SELECT COUNT(*) FROM messages m "
                        " WHERE m.session_id = c.session_id) AS message_count "
                        "FROM conversations c ORDER BY c.updated_at DESC"
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT c.*, "
                        "(SELECT COUNT(*) FROM messages m "
                        " WHERE m.session_id = c.session_id) AS message_count "
                        "FROM conversations c WHERE c.customer_id = ? "
                        "ORDER BY c.updated_at DESC",
                        (customer_id,),
                    ).fetchall()
                return [
                    SessionSummary(
                        session_id=r["session_id"],
                        customer_id=r["customer_id"],
                        title=r["title"],
                        started_at=r["started_at"],
                        updated_at=r["updated_at"],
                        message_count=r["message_count"],
                    )
                    for r in rows
                ]

        return await asyncio.to_thread(_op)

    # -- messages -----------------------------------------------------------
    async def add_message(self, session_id: str, role: str, content: str) -> None:
        def _op() -> None:
            with connect() as conn:
                now = _now()
                conn.execute(
                    "INSERT INTO messages (session_id, role, content, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (session_id, role, content, now),
                )
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )
                conn.commit()

        await asyncio.to_thread(_op)

    async def get_messages(self, session_id: str) -> list[Message]:
        def _op() -> list[Message]:
            with connect() as conn:
                rows = conn.execute(
                    "SELECT role, content, created_at FROM messages "
                    "WHERE session_id = ? ORDER BY id ASC",
                    (session_id,),
                ).fetchall()
                return [
                    Message(role=r["role"], content=r["content"], created_at=r["created_at"])
                    for r in rows
                ]

        return await asyncio.to_thread(_op)

    # -- pending action (CONFIRM gate) --------------------------------------
    async def set_pending_action(self, session_id: str, payload: dict) -> None:
        data = json.dumps(payload)

        def _op() -> None:
            with connect() as conn:
                conn.execute(
                    "INSERT INTO pending_actions (session_id, payload, created_at) "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT(session_id) DO UPDATE SET payload = excluded.payload, "
                    "created_at = excluded.created_at",
                    (session_id, data, _now()),
                )
                conn.commit()

        await asyncio.to_thread(_op)

    async def get_pending_action(self, session_id: str) -> dict | None:
        def _op() -> dict | None:
            with connect() as conn:
                row = conn.execute(
                    "SELECT payload FROM pending_actions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if row is None:
                    return None
                try:
                    return json.loads(row["payload"])
                except (ValueError, TypeError):
                    return None

        return await asyncio.to_thread(_op)

    async def clear_pending_action(self, session_id: str) -> None:
        def _op() -> None:
            with connect() as conn:
                conn.execute(
                    "DELETE FROM pending_actions WHERE session_id = ?", (session_id,)
                )
                conn.commit()

        await asyncio.to_thread(_op)

    async def count_user_messages(self, session_id: str) -> int:
        def _op() -> int:
            with connect() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM messages "
                    "WHERE session_id = ? AND role = 'user'",
                    (session_id,),
                ).fetchone()
                return int(row["n"]) if row else 0

        return await asyncio.to_thread(_op)

    # -- cross-session memory (Phase 6) -------------------------------------
    async def get_customer_memory(self, customer_id: str) -> CustomerMemoryRecord | None:
        def _op() -> CustomerMemoryRecord | None:
            with connect() as conn:
                row = conn.execute(
                    "SELECT * FROM customer_memory WHERE customer_id = ?", (customer_id,)
                ).fetchone()
                if row is None:
                    return None
                return CustomerMemoryRecord(
                    customer_id=row["customer_id"],
                    summary=row["summary"],
                    open_items=_loads_list(row["open_items"]),
                    preferences=_loads_list(row["preferences"]),
                    sentiment=row["sentiment"],
                    last_interaction=row["last_interaction"],
                )

        return await asyncio.to_thread(_op)

    async def upsert_customer_memory(
        self,
        customer_id: str,
        summary: str,
        open_items: list[str],
        preferences: list[str],
        sentiment: str,
    ) -> None:
        # Bound the stored memory so it can't grow unbounded over many sessions
        # (it's injected into every prompt).
        summary = (summary or "")[:1200]
        open_items = [str(x)[:200] for x in (open_items or [])][:10]
        preferences = [str(x)[:200] for x in (preferences or [])][:8]

        def _op() -> None:
            with connect() as conn:
                conn.execute(
                    "INSERT INTO customer_memory "
                    "(customer_id, summary, open_items, preferences, sentiment, last_interaction) "
                    "VALUES (?,?,?,?,?,?) "
                    "ON CONFLICT(customer_id) DO UPDATE SET summary=excluded.summary, "
                    "open_items=excluded.open_items, preferences=excluded.preferences, "
                    "sentiment=excluded.sentiment, last_interaction=excluded.last_interaction",
                    (
                        customer_id,
                        summary,
                        json.dumps(open_items),
                        json.dumps(preferences),
                        sentiment,
                        _now(),
                    ),
                )
                conn.commit()

        await asyncio.to_thread(_op)

    async def get_prior_unsummarized_session(
        self, customer_id: str, exclude_session_id: str
    ) -> str | None:
        """Most-recent prior session for this customer that has messages and
        hasn't been summarized yet (the one to digest when they return)."""

        def _op() -> str | None:
            with connect() as conn:
                row = conn.execute(
                    "SELECT c.session_id FROM conversations c "
                    "WHERE c.customer_id = ? AND c.session_id != ? AND c.summarized = 0 "
                    "AND EXISTS (SELECT 1 FROM messages m WHERE m.session_id = c.session_id) "
                    "ORDER BY c.updated_at DESC, c.session_id DESC LIMIT 1",
                    (customer_id, exclude_session_id),
                ).fetchone()
                return row["session_id"] if row else None

        return await asyncio.to_thread(_op)

    async def mark_summarized(self, session_id: str) -> None:
        def _op() -> None:
            with connect() as conn:
                conn.execute(
                    "UPDATE conversations SET summarized = 1 WHERE session_id = ?",
                    (session_id,),
                )
                conn.commit()

        await asyncio.to_thread(_op)

    def seed_demo_memory(
        self,
        customer_id: str,
        summary: str,
        open_items: list[str],
        preferences: list[str],
        sentiment: str,
    ) -> None:
        """Seed a starter memory ONLY if the customer has none (sync; startup use).
        Makes the cross-session recall demo work on the very first session.
        Atomic via ON CONFLICT DO NOTHING (won't clobber real memory)."""
        with connect() as conn:
            conn.execute(
                "INSERT INTO customer_memory "
                "(customer_id, summary, open_items, preferences, sentiment, last_interaction) "
                "VALUES (?,?,?,?,?,?) ON CONFLICT(customer_id) DO NOTHING",
                (
                    customer_id,
                    summary,
                    json.dumps(open_items),
                    json.dumps(preferences),
                    sentiment,
                    _now(),
                ),
            )
            conn.commit()

    # -- escalation / handoff packets (Phase 7) -----------------------------
    async def create_handoff(
        self,
        session_id: str,
        customer_id: str | None,
        customer_summary: str,
        issue: str,
        conversation_summary: str,
        actions_taken: list[str],
        suggested_next_step: str,
        sentiment: str,
    ) -> int:
        def _op() -> int:
            with connect() as conn:
                cur = conn.execute(
                    "INSERT INTO handoff_packets "
                    "(session_id, customer_id, customer_summary, issue, conversation_summary, "
                    "actions_taken, suggested_next_step, sentiment, status, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,'open',?)",
                    (
                        session_id,
                        customer_id,
                        customer_summary,
                        issue,
                        conversation_summary,
                        json.dumps(actions_taken or []),
                        suggested_next_step,
                        sentiment,
                        _now(),
                    ),
                )
                conn.commit()
                return int(cur.lastrowid)

        return await asyncio.to_thread(_op)

    def _row_to_handoff(self, r) -> HandoffRecord:
        return HandoffRecord(
            id=r["id"],
            session_id=r["session_id"],
            customer_id=r["customer_id"],
            customer_summary=r["customer_summary"],
            issue=r["issue"],
            conversation_summary=r["conversation_summary"],
            actions_taken=_loads_list(r["actions_taken"]),
            suggested_next_step=r["suggested_next_step"],
            sentiment=r["sentiment"],
            status=r["status"],
            created_at=r["created_at"],
        )

    async def list_handoffs(self, status: str | None = None) -> list[HandoffRecord]:
        def _op() -> list[HandoffRecord]:
            with connect() as conn:
                if status:
                    rows = conn.execute(
                        "SELECT * FROM handoff_packets WHERE status = ? ORDER BY id DESC",
                        (status,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM handoff_packets ORDER BY id DESC"
                    ).fetchall()
                return [self._row_to_handoff(r) for r in rows]

        return await asyncio.to_thread(_op)

    async def get_handoff(self, handoff_id: int) -> HandoffRecord | None:
        def _op() -> HandoffRecord | None:
            with connect() as conn:
                row = conn.execute(
                    "SELECT * FROM handoff_packets WHERE id = ?", (handoff_id,)
                ).fetchone()
                return self._row_to_handoff(row) if row else None

        return await asyncio.to_thread(_op)

    # -- observability traces (Phase 9) -------------------------------------
    async def add_trace(
        self,
        *,
        session_id: str,
        customer_id: str | None,
        turn_no: int,
        emotion_state: str | None,
        emotion_intensity: int | None,
        action: str | None,
        intents: list[str],
        tools: list[dict],
        sources: list[str],
        reply_len: int,
        latency_ms: int,
        errored: bool,
    ) -> None:
        def _op() -> None:
            with connect() as conn:
                conn.execute(
                    "INSERT INTO turn_traces "
                    "(session_id, customer_id, turn_no, emotion_state, emotion_intensity, "
                    "action, intents, tools, sources, reply_len, latency_ms, errored, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        session_id, customer_id, turn_no, emotion_state, emotion_intensity,
                        action, json.dumps(intents or []), json.dumps(tools or []),
                        json.dumps(sources or []), reply_len, latency_ms,
                        1 if errored else 0, _now(),
                    ),
                )
                conn.commit()

        await asyncio.to_thread(_op)

    async def list_traces(self, session_id: str | None = None, limit: int = 50) -> list[TraceRecord]:
        def _op() -> list[TraceRecord]:
            with connect() as conn:
                if session_id:
                    rows = conn.execute(
                        "SELECT * FROM turn_traces WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                        (session_id, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM turn_traces ORDER BY id DESC LIMIT ?", (limit,)
                    ).fetchall()
                return [
                    TraceRecord(
                        id=r["id"], session_id=r["session_id"], customer_id=r["customer_id"],
                        turn_no=r["turn_no"], emotion_state=r["emotion_state"],
                        emotion_intensity=r["emotion_intensity"], action=r["action"],
                        intents=_loads_list(r["intents"]), tools=_loads_list(r["tools"]),
                        sources=_loads_list(r["sources"]), reply_len=r["reply_len"],
                        latency_ms=r["latency_ms"], errored=bool(r["errored"]),
                        created_at=r["created_at"],
                    )
                    for r in rows
                ]

        return await asyncio.to_thread(_op)


# Process-wide store instance.
store = ConversationStore()
