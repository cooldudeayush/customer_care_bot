"""Low-level SQLite access for the Customer Care Bot.

Phase 1 home of conversation persistence: the ``conversations`` + ``messages``
tables that back the ChatGPT-style chat sidebar and give the agent loop the
running transcript for context.

This is the thin, swappable data-access layer the architecture calls for — swap
the connection here for Postgres/Supabase later without touching the store API
that sits above it (memory/store.py).

``sqlite3`` is synchronous; callers offload these helpers to threads (see
ConversationStore) so the FastAPI event loop is never blocked.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator

from config import _anchor, get_settings


def _resolve_path(database_url: str) -> str:
    """Turn a ``sqlite:///./data/app.db`` URL into a filesystem path."""
    if database_url.startswith("sqlite:///"):
        return database_url[len("sqlite:///") :]
    if database_url.startswith("sqlite://"):
        return database_url[len("sqlite://") :]
    return database_url


# Anchor to backend/ so the DB lives in a stable place regardless of cwd.
DB_PATH = _anchor(_resolve_path(get_settings().database_url))


SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    session_id  TEXT PRIMARY KEY,
    customer_id TEXT,
    title       TEXT NOT NULL DEFAULT 'New chat',
    started_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,          -- 'user' | 'bot'
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES conversations(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Open a short-lived SQLite connection (one per logical operation).

    Connection-per-operation keeps things thread-safe under the to_thread
    offloading we use, and SQLite connections are cheap to open.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        # foreign_keys is a per-CONNECTION pragma (not persistent), so it must be
        # set on every connection for ON DELETE CASCADE to take effect. WAL mode,
        # by contrast, is a persistent DB property set once in init_db().
        conn.execute("PRAGMA foreign_keys=ON;")
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist. Called once at app startup."""
    parent = os.path.dirname(os.path.abspath(DB_PATH))
    os.makedirs(parent or ".", exist_ok=True)
    with connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL;")  # persistent; set once
        conn.executescript(SCHEMA)
        conn.commit()
