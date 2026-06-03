"""Business actions — the real (mock) operations the agent can take.

Each function is a deterministic operation over the mock business DB and returns
a STRUCTURED result: {"success": bool, "data": {...} | None, "error": str | None}.
The RESPOND step turns these into natural language — it never invents an outcome
a tool didn't return.

Reads are safe; writes (issue_refund, cancel_order, ...) change DB state. Phase 4
will additionally write back to the Neo4j customer graph here.

PII discipline: payment card numbers are never returned in full — only a masked
"card ending 1234" form (Section 9.3).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from knowledge.graph import graph_client

from .business_db import connect

# --- Policy constants (mirror data/corpus so tools + docs agree) -------------
REFUND_WINDOW_DAYS = {"electronics": 15}
DEFAULT_REFUND_WINDOW_DAYS = 30
NON_REFUNDABLE = {"perishable", "gift_card"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso() -> str:
    return _now().isoformat()


def _ok(data: dict) -> dict:
    return {"success": True, "data": data, "error": None}


def _err(message: str) -> dict:
    return {"success": False, "data": None, "error": message}


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _mask_payment(method: str, last4: str | None) -> str:
    if method == "card" and last4:
        return f"card ending {last4}"
    return method


def _window_days(category: str) -> int:
    return REFUND_WINDOW_DAYS.get(category, DEFAULT_REFUND_WINDOW_DAYS)


def _get_order(conn: sqlite3.Connection, order_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()


def _order_items(conn: sqlite3.Connection, order_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT oi.qty, oi.price, oi.return_status, p.name, p.category, p.warranty_days "
        "FROM order_items oi JOIN products p ON p.id = oi.product_id "
        "WHERE oi.order_id = ?",
        (order_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _captured_payments(conn: sqlite3.Connection, order_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM payments WHERE order_id = ? AND is_refund = 0 AND status = 'captured'",
        (order_id,),
    ).fetchall()


def _duplicate_amount(payments: list[sqlite3.Row]) -> float | None:
    """Return the amount of a duplicated captured charge, if any."""
    return _duplicate_amount_from_list([p["amount"] for p in payments])


def _duplicate_amount_from_list(amounts: list[float]) -> float | None:
    seen: dict[float, int] = {}
    for a in amounts:
        seen[a] = seen.get(a, 0) + 1
    for amount, count in seen.items():
        if count >= 2:
            return float(amount)
    return None


def compute_refund_eligibility(
    order_id: str,
    *,
    status: str | None,
    delivered_at: str | None,
    category: str | None,
    window_days: int | None,
    payment_amounts: list[float],
    source: str,
) -> dict:
    """Pure eligibility decision from the fields a refund check needs.

    Shared by the graph path (fields from a Cypher traversal) and the SQLite
    fallback (fields from the tables), so both stay perfectly consistent.
    ``source`` records which path produced the answer ('graph' | 'sqlite').
    """
    category = category or "unknown"
    window = int(window_days) if window_days is not None else _window_days(category)
    dup = _duplicate_amount_from_list(payment_amounts or [])
    delivered = _parse_dt(delivered_at)
    days_since = (_now() - delivered).days if delivered else None

    if status == "refunded":
        eligible, reason = False, "This order has already been refunded."
    elif status == "cancelled":
        eligible, reason = False, "This order was cancelled."
    elif category in NON_REFUNDABLE:
        eligible, reason = False, f"{category} items are non-refundable."
    elif delivered is None:
        eligible, reason = (
            False,
            "The order hasn't been delivered yet — it can be cancelled instead of refunded.",
        )
    elif days_since is not None and days_since <= window:
        eligible, reason = (
            True,
            f"Within the {window}-day refund window ({days_since} days since delivery).",
        )
    else:
        eligible, reason = (
            False,
            f"Outside the {window}-day refund window ({days_since} days since delivery).",
        )

    legit_total = (
        float(sum(payment_amounts) - (dup or 0.0)) if payment_amounts else None
    )
    return _ok(
        {
            "order_id": order_id,
            "category": category,
            "status": status,
            "window_days": window,
            "days_since_delivery": days_since,
            "eligible": eligible,
            "reason": reason,
            "refundable_amount": legit_total if eligible else None,
            "duplicate_charge": {"detected": dup is not None, "amount": dup},
            "source": source,
        }
    )


# ---------------------------------------------------------------------------
# READ actions (safe)
# ---------------------------------------------------------------------------
def check_order_status(order_id: str) -> dict:
    with connect() as conn:
        order = _get_order(conn, order_id)
        if order is None:
            return _err(f"No order found with id {order_id}.")
        items = _order_items(conn, order_id)
        payments = _captured_payments(conn, order_id)
        dup = _duplicate_amount(payments)
        return _ok(
            {
                "order_id": order_id,
                "status": order["status"],
                "total": order["total"],
                "placed_at": order["placed_at"],
                "shipped_at": order["shipped_at"],
                "delivered_at": order["delivered_at"],
                "items": [
                    {
                        "name": it["name"],
                        "category": it["category"],
                        "qty": it["qty"],
                        "price": it["price"],
                        "return_status": it["return_status"],
                    }
                    for it in items
                ],
                "payments": [
                    {"amount": p["amount"], "method": _mask_payment(p["method"], p["last4"])}
                    for p in payments
                ],
                "duplicate_charge": {"detected": dup is not None, "amount": dup},
            }
        )


def get_refund_eligibility(order_id: str) -> dict:
    # PRIMARY: the Neo4j graph traversal (Order->Item->Product->Policy) — the
    # relational-reasoning showcase. Returns None if the graph is off/unreachable
    # or the order isn't in it, in which case we fall back to SQLite below.
    data = graph_client.refund_eligibility_data(order_id)
    if data is not None:
        return compute_refund_eligibility(order_id, source="graph", **data)

    # FALLBACK: SQLite (same decision logic, same result shape).
    with connect() as conn:
        order = _get_order(conn, order_id)
        if order is None:
            return _err(f"No order found with id {order_id}.")
        items = _order_items(conn, order_id)
        category = items[0]["category"] if items else "unknown"
        payments = _captured_payments(conn, order_id)
        return compute_refund_eligibility(
            order_id,
            source="sqlite",
            status=order["status"],
            delivered_at=order["delivered_at"],
            category=category,
            window_days=None,
            payment_amounts=[p["amount"] for p in payments],
        )


def track_shipment(order_id: str) -> dict:
    with connect() as conn:
        order = _get_order(conn, order_id)
        if order is None:
            return _err(f"No order found with id {order_id}.")
        status = order["status"]
        stage = {
            "placed": "Order placed — preparing for shipment.",
            "shipped": "In transit — expected within 1–2 business days.",
            "delivered": f"Delivered on {order['delivered_at']}.",
            "cancelled": "This order was cancelled.",
            "refunded": "This order was refunded.",
        }.get(status, status)
        return _ok(
            {
                "order_id": order_id,
                "status": status,
                "stage": stage,
                "reschedule_date": order["reschedule_date"],
            }
        )


# ---------------------------------------------------------------------------
# WRITE actions
# ---------------------------------------------------------------------------
def issue_refund(order_id: str, amount: float | None = None, reason: str | None = None) -> dict:
    """Issue a refund. Confirmation-gated upstream (Section 8).

    If ``reason`` mentions a duplicate (or ``amount`` matches a duplicated
    charge), only the duplicate charge is reversed and the order itself stands.
    Otherwise the order is fully refunded.
    """
    with connect() as conn:
        order = _get_order(conn, order_id)
        if order is None:
            return _err(f"No order found with id {order_id}.")
        if order["status"] == "refunded":
            return _err(f"Order {order_id} has already been refunded.")

        payments = _captured_payments(conn, order_id)
        if not payments:
            return _err(f"No captured payment found for order {order_id} to refund.")

        dup = _duplicate_amount(payments)
        reason_says_dup = bool(reason and "duplicate" in reason.lower())
        # Guard: if the user/model claims a duplicate but none exists, do NOT
        # silently fall through to a full refund — say so and stop.
        if reason_says_dup and dup is None:
            return _err(
                f"No duplicate charge was found on order {order_id}, so there's "
                "nothing to reverse. Let me know if you'd like a different refund."
            )
        is_duplicate = bool(
            reason_says_dup
            or (dup is not None and amount is not None and abs(amount - dup) < 0.01)
        )

        if is_duplicate and dup is not None:
            refund_amount = dup
            # Mark ONE of the duplicate captured charges as refunded; order stands.
            dupe_row = next((p for p in payments if abs(p["amount"] - dup) < 0.01), None)
            if dupe_row is not None:
                conn.execute(
                    "UPDATE payments SET status = 'refunded' WHERE id = ?", (dupe_row["id"],)
                )
            new_status = order["status"]
            note = "duplicate charge reversed"
        else:
            refund_amount = float(amount) if amount is not None else float(order["total"])
            conn.execute("UPDATE orders SET status = 'refunded' WHERE id = ?", (order_id,))
            conn.execute(
                "UPDATE order_items SET return_status = 'refunded' WHERE order_id = ?",
                (order_id,),
            )
            new_status = "refunded"
            note = "order refunded"

        ref = payments[0]
        conn.execute(
            "INSERT INTO payments (order_id, method, amount, status, last4, is_refund, created_at) "
            "VALUES (?,?,?,?,?,1,?)",
            (order_id, ref["method"], refund_amount, "refunded", ref["last4"], _iso()),
        )
        conn.commit()
        # Mirror to the customer graph (best-effort; SQLite is the source of truth).
        graph_client.record_refund(order_id, new_status, refund_amount, ref["method"], ref["last4"])
        return _ok(
            {
                "order_id": order_id,
                "refund_amount": refund_amount,
                "to": _mask_payment(ref["method"], ref["last4"]),
                "eta": "3–5 business days",
                "order_status": new_status,
                "note": note,
            }
        )


def cancel_order(order_id: str) -> dict:
    """Cancel an order (confirmation-gated upstream). Only before shipment."""
    with connect() as conn:
        order = _get_order(conn, order_id)
        if order is None:
            return _err(f"No order found with id {order_id}.")
        status = order["status"]
        if status in ("cancelled", "refunded"):
            return _err(f"Order {order_id} is already {status}.")
        if status in ("shipped", "delivered"):
            return _err(
                f"Order {order_id} has already {status}; it can't be cancelled — "
                "please start a return instead."
            )
        # status == 'placed'
        conn.execute("UPDATE orders SET status = 'cancelled' WHERE id = ?", (order_id,))
        payments = _captured_payments(conn, order_id)
        refunded = 0.0
        for p in payments:
            conn.execute("UPDATE payments SET status = 'refunded' WHERE id = ?", (p["id"],))
            conn.execute(
                "INSERT INTO payments (order_id, method, amount, status, last4, is_refund, created_at) "
                "VALUES (?,?,?,?,?,1,?)",
                (order_id, p["method"], p["amount"], "refunded", p["last4"], _iso()),
            )
            refunded += float(p["amount"])
        conn.commit()
        # Mirror the status flip to the customer graph (best-effort).
        graph_client.flip_order_status(order_id, "cancelled")
        return _ok(
            {
                "order_id": order_id,
                "order_status": "cancelled",
                "refunded_amount": refunded,
                "eta": "3–5 business days",
            }
        )


def update_address(customer_id: str, new_address: str) -> dict:
    with connect() as conn:
        cust = conn.execute("SELECT id FROM customers WHERE id = ?", (customer_id,)).fetchone()
        if cust is None:
            return _err(f"No customer found with id {customer_id}.")
        conn.execute("UPDATE customers SET address = ? WHERE id = ?", (new_address, customer_id))
        # Also update not-yet-shipped orders so the change actually takes effect.
        conn.execute(
            "UPDATE orders SET address = ? WHERE customer_id = ? AND status = 'placed'",
            (new_address, customer_id),
        )
        conn.commit()
        return _ok({"customer_id": customer_id, "new_address": new_address})


def reschedule_delivery(order_id: str, new_date: str) -> dict:
    with connect() as conn:
        order = _get_order(conn, order_id)
        if order is None:
            return _err(f"No order found with id {order_id}.")
        if order["status"] not in ("placed", "shipped"):
            return _err(
                f"Order {order_id} is {order['status']}; delivery can only be "
                "rescheduled while it's in transit or being prepared."
            )
        conn.execute("UPDATE orders SET reschedule_date = ? WHERE id = ?", (new_date, order_id))
        conn.commit()
        return _ok({"order_id": order_id, "new_date": new_date})


def create_ticket(
    customer_id: str,
    subject: str,
    body: str = "",
    priority: str = "normal",
    order_id: str | None = None,
) -> dict:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO tickets (customer_id, order_id, subject, body, priority, status, created_at) "
            "VALUES (?,?,?,?,?,'open',?)",
            (customer_id, order_id, subject, body, priority, _iso()),
        )
        conn.commit()
        return _ok(
            {
                "ticket_id": cur.lastrowid,
                "status": "open",
                "subject": subject,
                "priority": priority,
            }
        )


# ---------------------------------------------------------------------------
# Customer snapshot (not a tool) — context for the PERCEIVE step
# ---------------------------------------------------------------------------
def customer_snapshot(customer_id: str) -> dict:
    """Compact view of a customer + their orders, injected into PERCEIVE so the
    model can resolve entities (order ids) and decide actions accurately.
    """
    with connect() as conn:
        cust = conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
        if cust is None:
            return {"known": False}
        orders = conn.execute(
            "SELECT id, status, total, delivered_at FROM orders WHERE customer_id = ? ORDER BY id",
            (customer_id,),
        ).fetchall()
        order_views = []
        for o in orders:
            items = _order_items(conn, o["id"])
            payments = _captured_payments(conn, o["id"])
            dup = _duplicate_amount(payments)
            order_views.append(
                {
                    "order_id": o["id"],
                    "status": o["status"],
                    "total": o["total"],
                    "items": ", ".join(f"{it['qty']}x {it['name']} ({it['category']})" for it in items),
                    "duplicate_charge": dup,
                }
            )
        return {
            "known": True,
            "customer_id": customer_id,
            "name": cust["name"],
            "tier": cust["tier"],
            "address": cust["address"],
            "orders": order_views,
        }
