"""Customer operational graph (Neo4j) — relational reasoning (Phase 4).

The novelty layer: the SQLite business data is mirrored into a Neo4j graph so the
agent can REASON over a customer's situation with a real traversal:

    (Order)-[:CONTAINS]->(OrderItem)-[:OF_PRODUCT]->(Product)-[:GOVERNED_BY]->(Policy)

`get_refund_eligibility` uses this traversal (graph + policy) to decide eligibility —
that's the showcase. Writes (refund/cancel) flip the order's status in the graph too.

DEGRADES GRACEFULLY: SQLite is the source of truth. If Neo4j is unreachable or
disabled (no password configured), every method here no-ops/returns None and the
caller falls back to the SQLite path (architecture §9.4). Same driver works against
Docker locally or Neo4j Aura in the cloud — only NEO4J_URI changes.
"""

from __future__ import annotations

import logging

from config import Settings, get_settings

logger = logging.getLogger("ccb.graph")

try:  # neo4j is optional at runtime
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - import guard
    GraphDatabase = None  # type: ignore


# Refund policy windows mirrored into Policy nodes (must match data/corpus + actions.py).
_POLICIES = [
    {"id": "refund_electronics", "type": "refund", "window_days": 15, "category": "electronics"},
    {"id": "refund_standard", "type": "refund", "window_days": 30, "category": "standard"},
]


def _policy_for(category: str) -> str:
    return "refund_electronics" if category == "electronics" else "refund_standard"


class GraphClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._driver = None
        self._enabled = False

    # -- lifecycle ----------------------------------------------------------
    def _build_driver(self):
        if GraphDatabase is None or not self._settings.neo4j_configured:
            return None
        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(
                    self._settings.neo4j_uri,
                    auth=(self._settings.neo4j_user, self._settings.neo4j_password),
                    connection_timeout=5,
                )
            except Exception:  # noqa: BLE001
                logger.warning("Could not create Neo4j driver", exc_info=True)
                self._driver = None
        return self._driver

    def init(self) -> bool:
        """Connect + verify once at startup. Returns True if the graph is usable.

        Done once so per-turn calls never pay a connection timeout when Neo4j is
        down — they just see _enabled=False and the caller uses SQLite.
        """
        self._enabled = False
        driver = self._build_driver()
        if driver is None:
            return False
        try:
            driver.verify_connectivity()
            self._enabled = True
        except Exception:  # noqa: BLE001
            logger.info("Neo4j not reachable; running in SQLite-only (degraded) mode.")
            self._enabled = False
        return self._enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def close(self) -> None:
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:  # noqa: BLE001
                pass
            self._driver = None

    # -- low-level ----------------------------------------------------------
    def _run(self, cypher: str, **params):
        if not self._enabled:
            return None
        driver = self._build_driver()
        if driver is None:
            return None
        try:
            with driver.session() as session:
                return list(session.run(cypher, **params))
        except Exception:  # noqa: BLE001
            logger.debug("Neo4j read failed", exc_info=True)
            return None

    def _write(self, cypher: str, **params) -> bool:
        if not self._enabled:
            return False
        driver = self._build_driver()
        if driver is None:
            return False
        try:
            with driver.session() as session:
                session.run(cypher, **params).consume()
            return True
        except Exception:  # noqa: BLE001
            logger.debug("Neo4j write failed", exc_info=True)
            return False

    # -- sync from SQLite ---------------------------------------------------
    def sync_from_business_db(self) -> int | None:
        """Rebuild the whole graph from the SQLite business DB. Returns #orders."""
        if not self._enabled:
            return None
        driver = self._build_driver()
        if driver is None:
            return None
        from tools.business_db import connect as biz_connect

        try:
            with biz_connect() as bz:
                customers = [dict(r) for r in bz.execute("SELECT * FROM customers")]
                products = [dict(r) for r in bz.execute("SELECT * FROM products")]
                orders = [dict(r) for r in bz.execute("SELECT * FROM orders")]
                items = [dict(r) for r in bz.execute("SELECT * FROM order_items")]
                payments = [dict(r) for r in bz.execute("SELECT * FROM payments")]
                tickets = [dict(r) for r in bz.execute("SELECT * FROM tickets")]
            with driver.session() as session:
                session.execute_write(
                    self._build_graph, customers, products, orders, items, payments, tickets
                )
            logger.info("Neo4j graph synced: %d orders.", len(orders))
            return len(orders)
        except Exception:  # noqa: BLE001
            logger.warning("Graph sync failed; continuing in degraded mode.", exc_info=True)
            return None

    @staticmethod
    def _build_graph(tx, customers, products, orders, items, payments, tickets) -> None:
        tx.run("MATCH (n) DETACH DELETE n")
        for pol in _POLICIES:
            tx.run(
                "MERGE (p:Policy {id:$id}) SET p.type=$type, p.window_days=$window_days, p.category=$category",
                **pol,
            )
        for c in customers:
            tx.run(
                "MERGE (c:Customer {id:$id}) SET c.name=$name, c.tier=$tier, c.language_pref=$lang",
                id=c["id"], name=c["name"], tier=c["tier"], lang=c["language_pref"],
            )
        for p in products:
            tx.run(
                "MERGE (pr:Product {id:$id}) SET pr.name=$name, pr.category=$cat, "
                "pr.price=$price, pr.warranty_days=$wd",
                id=p["id"], name=p["name"], cat=p["category"], price=p["price"], wd=p["warranty_days"],
            )
            tx.run(
                "MATCH (pr:Product {id:$pid}),(pol:Policy {id:$pol}) MERGE (pr)-[:GOVERNED_BY]->(pol)",
                pid=p["id"], pol=_policy_for(p["category"]),
            )
        for o in orders:
            tx.run(
                "MERGE (o:Order {id:$id}) SET o.status=$st, o.total=$total, o.channel=$ch, "
                "o.placed_at=$pa, o.shipped_at=$sa, o.delivered_at=$da",
                id=o["id"], st=o["status"], total=o["total"], ch=o["channel"],
                pa=o["placed_at"], sa=o["shipped_at"], da=o["delivered_at"],
            )
            tx.run(
                "MATCH (c:Customer {id:$cid}),(o:Order {id:$oid}) MERGE (c)-[:PLACED]->(o)",
                cid=o["customer_id"], oid=o["id"],
            )
        for it in items:
            tx.run(
                "MATCH (o:Order {id:$oid}),(pr:Product {id:$pid}) "
                "CREATE (i:OrderItem {qty:$qty, price:$price, return_status:$rs}) "
                "MERGE (o)-[:CONTAINS]->(i) MERGE (i)-[:OF_PRODUCT]->(pr)",
                oid=it["order_id"], pid=it["product_id"], qty=it["qty"], price=it["price"], rs=it["return_status"],
            )
        for pay in payments:
            tx.run(
                "MATCH (o:Order {id:$oid}) "
                "CREATE (p:Payment {amount:$amt, method:$method, status:$st, last4:$last4, is_refund:$isref}) "
                "MERGE (o)-[:PAID_WITH]->(p)",
                oid=pay["order_id"], amt=pay["amount"], method=pay["method"],
                st=pay["status"], last4=pay["last4"], isref=bool(pay["is_refund"]),
            )
        for t in tickets:
            tx.run(
                "MATCH (c:Customer {id:$cid}) "
                "CREATE (tk:Ticket {subject:$subj, status:$st, priority:$pri}) "
                "MERGE (c)-[:RAISED]->(tk)",
                cid=t["customer_id"], subj=t["subject"], st=t["status"], pri=t["priority"],
            )

    # -- reasoning queries --------------------------------------------------
    def refund_eligibility_data(self, order_id: str) -> dict | None:
        """Traverse Order->Item->Product->Policy to fetch the fields needed to
        decide refund eligibility. Returns None if the graph is off or the order
        isn't present (caller falls back to SQLite)."""
        recs = self._run(
            "MATCH (o:Order {id:$oid}) "
            "OPTIONAL MATCH (o)-[:CONTAINS]->(:OrderItem)-[:OF_PRODUCT]->"
            "(p:Product)-[:GOVERNED_BY]->(pol:Policy {type:'refund'}) "
            "RETURN o.status AS status, o.delivered_at AS delivered_at, "
            "p.category AS category, pol.window_days AS window_days LIMIT 1",
            oid=order_id,
        )
        if not recs or recs[0]["status"] is None:
            return None
        rec = recs[0]
        pays = self._run(
            "MATCH (o:Order {id:$oid})-[:PAID_WITH]->(pay:Payment) "
            "WHERE pay.is_refund = false AND pay.status = 'captured' "
            "RETURN pay.amount AS amount",
            oid=order_id,
        )
        amounts = [r["amount"] for r in pays] if pays else []
        return {
            "status": rec["status"],
            "delivered_at": rec["delivered_at"],
            "category": rec["category"],
            "window_days": rec["window_days"],
            "payment_amounts": amounts,
        }

    # -- writebacks (best-effort mirror) ------------------------------------
    def flip_order_status(self, order_id: str, status: str) -> bool:
        return self._write(
            "MATCH (o:Order {id:$oid}) SET o.status=$st", oid=order_id, st=status
        )

    def record_refund(
        self, order_id: str, new_status: str, amount: float, method: str = "card", last4: str | None = None
    ) -> bool:
        return self._write(
            "MATCH (o:Order {id:$oid}) SET o.status=$st "
            "CREATE (o)-[:REFUNDED_BY]->(:Payment {amount:$amt, method:$method, "
            "last4:$last4, status:'refunded', is_refund:true})",
            oid=order_id, st=new_status, amt=amount, method=method, last4=last4,
        )


# Process-wide graph client.
graph_client = GraphClient()
