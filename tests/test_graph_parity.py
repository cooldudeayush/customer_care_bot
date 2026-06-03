"""Phase 4 graph parity test.

Proves the Neo4j Cypher traversal produces the SAME refund-eligibility decisions
as the SQLite fallback for every order. Skips gracefully if Neo4j isn't running
(so it's safe to run anytime; the degradation path is tested separately).

Setup, then run from the repo root:
    docker compose up -d
    # set NEO4J_PASSWORD=password123 in backend/.env
    backend/.venv/Scripts/python.exe tests/test_graph_parity.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from knowledge.graph import graph_client  # noqa: E402
from tools import actions as A  # noqa: E402
from tools.business_db import connect  # noqa: E402
from tools.seed import seed  # noqa: E402

seed()

if not graph_client.init():
    print(
        "[SKIP] Neo4j is not reachable. Start it with `docker compose up -d` and set "
        "NEO4J_PASSWORD=password123 in backend/.env to run the graph parity test.\n"
        "       (The SQLite degradation path is verified by test_agentic_flow.py.)"
    )
    sys.exit(0)

n = graph_client.sync_from_business_db()
print(f"Synced {n} orders into Neo4j.")

with connect() as c:
    order_ids = [r["id"] for r in c.execute("SELECT id FROM orders ORDER BY id")]

COMPARE = ("eligible", "window_days", "days_since_delivery", "status", "duplicate_charge")
for oid in order_ids:
    graph_client._enabled = True
    g = A.get_refund_eligibility(oid)["data"]
    graph_client._enabled = False
    s = A.get_refund_eligibility(oid)["data"]
    graph_client._enabled = True

    assert g["source"] == "graph", f"{oid}: expected graph path, got {g['source']}"
    assert s["source"] == "sqlite", f"{oid}: expected sqlite path, got {s['source']}"
    for k in COMPARE:
        assert g[k] == s[k], f"PARITY MISMATCH {oid}.{k}: graph={g[k]} sqlite={s[k]}"
    print(f"  #{oid}: eligible={g['eligible']} window={g['window_days']} "
          f"days={g['days_since_delivery']} (graph == sqlite)")

# Writeback check: issue a refund and confirm the graph status reflects it.
graph_client.flip_order_status("1260", "cancelled")
rec = graph_client._run("MATCH (o:Order {id:'1260'}) RETURN o.status AS s")
assert rec and rec[0]["s"] == "cancelled", "graph writeback (flip status) failed"
print("  writeback: graph order #1260 status -> cancelled (ok)")

graph_client.close()
print("\n=== GRAPH PARITY VERIFIED: Cypher traversal == SQLite decisions ===")
