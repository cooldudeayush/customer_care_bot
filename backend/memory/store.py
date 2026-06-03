"""Conversation persistence — the short-term memory store.

Backs the chat sidebar (list past sessions, reopen a transcript) and supplies
the agent loop with the running transcript for multi-turn context.

Async methods wrap the synchronous sqlite layer (db.py) via ``asyncio.to_thread``
so the FastAPI event loop stays responsive. Phase 6 will add long-term
cross-session summaries alongside this.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

from .db import connect, init_db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


# Process-wide store instance.
store = ConversationStore()
