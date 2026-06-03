"""Seed the mock business DB with realistic demo data.

Run from the backend/ directory:

    python -m tools.seed

Idempotent: clears and repopulates so you always get a clean demo state. Dates
are relative to "now" so refund/return windows stay correct whenever you seed.

The data is crafted for the demo scenario (Section 12): customer ``cust_demo``
(the default frontend customer) has order #1234 with a DUPLICATE charge, a
within-window electronics item, a window-expired item, an in-transit order, and a
cancellable not-yet-shipped order.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tools.business_db import connect, init_db


def _days_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).isoformat()


CUSTOMERS = [
    # id, name, tier, lang, email, address
    ("cust_demo", "Aarav Sharma", "regular", "en", "aarav@example.com", "12 MG Road, Bengaluru 560001"),
    ("cust_002", "Priya Nair", "premium", "en", "priya@example.com", "44 Anna Salai, Chennai 600002"),
    ("cust_003", "Rohan Mehta", "regular", "hi", "rohan@example.com", "8 Park Street, Kolkata 700016"),
]

PRODUCTS = [
    # id, name, category, price, warranty_days
    ("p_head", "Wireless Headphones", "electronics", 1499.0, 365),
    ("p_speaker", "Bluetooth Speaker", "electronics", 2999.0, 365),
    ("p_tee", "Cotton T-Shirt", "apparel", 699.0, 0),
    ("p_case", "Phone Case", "accessory", 299.0, 180),
    ("p_charger", "USB-C Charger", "accessory", 499.0, 180),
    ("p_mug", "Ceramic Mug", "home", 349.0, 0),
]

# orders: id, customer_id, status, total, channel, placed_at, shipped_at, delivered_at, address
ORDERS = [
    ("1234", "cust_demo", "delivered", 1499.0, "web", _days_ago(15), _days_ago(14), _days_ago(12), "12 MG Road, Bengaluru 560001"),
    ("1190", "cust_demo", "delivered", 699.0, "web", _days_ago(45), _days_ago(44), _days_ago(40), "12 MG Road, Bengaluru 560001"),
    ("1255", "cust_demo", "shipped", 299.0, "app", _days_ago(2), _days_ago(1), None, "12 MG Road, Bengaluru 560001"),
    ("1260", "cust_demo", "placed", 2999.0, "web", _days_ago(0), None, None, "12 MG Road, Bengaluru 560001"),
    ("1300", "cust_002", "delivered", 499.0, "web", _days_ago(8), _days_ago(7), _days_ago(5), "44 Anna Salai, Chennai 600002"),
    ("1350", "cust_003", "delivered", 349.0, "app", _days_ago(24), _days_ago(23), _days_ago(20), "8 Park Street, Kolkata 700016"),
]

# order_items: order_id, product_id, qty, price, return_status
ORDER_ITEMS = [
    ("1234", "p_head", 1, 1499.0, "none"),
    ("1190", "p_tee", 1, 699.0, "none"),
    ("1255", "p_case", 1, 299.0, "none"),
    ("1260", "p_speaker", 1, 2999.0, "none"),
    ("1300", "p_charger", 1, 499.0, "none"),
    ("1350", "p_mug", 1, 349.0, "none"),
]

# payments: order_id, method, amount, status, last4, is_refund, created_at
# NOTE: order 1234 has TWO captured charges of 1499 — the demo double-charge.
PAYMENTS = [
    ("1234", "card", 1499.0, "captured", "4242", 0, _days_ago(15)),
    ("1234", "card", 1499.0, "captured", "4242", 0, _days_ago(15)),  # duplicate!
    ("1190", "upi", 699.0, "captured", None, 0, _days_ago(45)),
    ("1255", "card", 299.0, "captured", "1881", 0, _days_ago(2)),
    ("1260", "card", 2999.0, "captured", "7702", 0, _days_ago(0)),
    ("1300", "card", 499.0, "captured", "9001", 0, _days_ago(8)),
    ("1350", "upi", 349.0, "captured", None, 0, _days_ago(24)),
]


def seed() -> None:
    init_db()
    with connect() as conn:
        for table in ("payments", "order_items", "tickets", "orders", "products", "customers"):
            conn.execute(f"DELETE FROM {table}")
        conn.executemany(
            "INSERT INTO customers (id, name, tier, language_pref, email, address) VALUES (?,?,?,?,?,?)",
            CUSTOMERS,
        )
        conn.executemany(
            "INSERT INTO products (id, name, category, price, warranty_days) VALUES (?,?,?,?,?)",
            PRODUCTS,
        )
        conn.executemany(
            "INSERT INTO orders (id, customer_id, status, total, channel, placed_at, shipped_at, delivered_at, address) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            ORDERS,
        )
        conn.executemany(
            "INSERT INTO order_items (order_id, product_id, qty, price, return_status) VALUES (?,?,?,?,?)",
            ORDER_ITEMS,
        )
        conn.executemany(
            "INSERT INTO payments (order_id, method, amount, status, last4, is_refund, created_at) VALUES (?,?,?,?,?,?,?)",
            PAYMENTS,
        )
        conn.commit()


if __name__ == "__main__":
    seed()
    with connect() as conn:
        c = conn.execute("SELECT COUNT(*) AS n FROM customers").fetchone()["n"]
        o = conn.execute("SELECT COUNT(*) AS n FROM orders").fetchone()["n"]
        p = conn.execute("SELECT COUNT(*) AS n FROM payments").fetchone()["n"]
    print(f"[OK] Seeded business DB: {c} customers, {o} orders, {p} payments.")
    print("  Demo: cust_demo order #1234 has a duplicate Rs.1499 charge.")
