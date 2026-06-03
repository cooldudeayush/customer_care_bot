"""Mock business database (Phase 3) — the systems the tools read and write.

A separate SQLite store from the conversation memory: it holds the simulated
operational data (customers, products, orders, items, payments, tickets) that the
agentic tools act on. Kept distinct so it maps cleanly to "real" business systems
and so memory can be swapped (Postgres) without touching business data.

Phase 4 mirrors this same data into the Neo4j customer graph for relational
reasoning; the tools write back to both.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator

from config import _anchor, _resolve_sqlite_path, get_settings

BUSINESS_DB_PATH = _anchor(_resolve_sqlite_path(get_settings().business_database_url))


SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    tier          TEXT NOT NULL DEFAULT 'regular',   -- regular | premium
    language_pref TEXT DEFAULT 'en',
    email         TEXT,
    address       TEXT
);

CREATE TABLE IF NOT EXISTS products (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    category     TEXT NOT NULL,                       -- electronics | apparel | accessory | ...
    price        REAL NOT NULL,
    warranty_days INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS orders (
    id             TEXT PRIMARY KEY,
    customer_id    TEXT NOT NULL,
    status         TEXT NOT NULL,                     -- placed | shipped | delivered | cancelled | refunded
    total          REAL NOT NULL,
    channel        TEXT DEFAULT 'web',
    placed_at      TEXT,
    shipped_at     TEXT,
    delivered_at   TEXT,
    reschedule_date TEXT,
    address        TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);

CREATE TABLE IF NOT EXISTS order_items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id      TEXT NOT NULL,
    product_id    TEXT NOT NULL,
    qty           INTEGER NOT NULL DEFAULT 1,
    price         REAL NOT NULL,
    return_status TEXT DEFAULT 'none',                -- none | requested | returned | refunded
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);

CREATE TABLE IF NOT EXISTS payments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id   TEXT NOT NULL,
    method     TEXT NOT NULL,                         -- card | upi | netbanking | wallet
    amount     REAL NOT NULL,
    status     TEXT NOT NULL,                         -- captured | refunded
    last4      TEXT,
    is_refund  INTEGER NOT NULL DEFAULT 0,
    created_at TEXT,
    FOREIGN KEY (order_id) REFERENCES orders(id)
);

CREATE TABLE IF NOT EXISTS tickets (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT,
    order_id   TEXT,
    subject    TEXT NOT NULL,
    body       TEXT,
    priority   TEXT DEFAULT 'normal',                 -- low | normal | high | urgent
    status     TEXT NOT NULL DEFAULT 'open',          -- open | resolved | escalated
    created_at TEXT
);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(BUSINESS_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=ON;")
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    parent = os.path.dirname(os.path.abspath(BUSINESS_DB_PATH))
    os.makedirs(parent or ".", exist_ok=True)
    with connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript(SCHEMA)
        conn.commit()


def is_seeded() -> bool:
    try:
        with connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM customers").fetchone()
            return bool(row and row["n"] > 0)
    except sqlite3.Error:
        return False
