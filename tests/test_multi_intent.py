"""Phase 8 multi-intent + language tests (no API key — LLM faked).

Verifies the MIXED split: a compound request ("change my address AND refund the
duplicate") runs the SAFE action immediately, gates the money action behind
CONFIRM (gate invariant held), gives one combined reply, and the language-mirroring
directive is present in the RESPOND prompt. Then 'yes' executes the gated refund.
"""

import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

for f in ("app.db", "app.db-wal", "app.db-shm"):
    try:
        os.remove(os.path.join(ROOT, "backend", "data", f))
    except OSError:
        pass

import agent.loop as loop_mod  # noqa: E402
from agent.loop import agent_loop  # noqa: E402
from agent.perception import Emotion, Perception, ToolCall  # noqa: E402
from knowledge.retriever import retriever  # noqa: E402
from llm import gemini_client  # noqa: E402
from memory.store import store  # noqa: E402
from tools.business_db import connect  # noqa: E402
from tools.seed import seed  # noqa: E402

store.init()
seed()
CUST = "cust_demo"
CAPTURED = {}
PLAN = {}


async def fake_perceive(*, history, customer_id, pending_action, memory_text="", llm=None):
    return PLAN[history[-1].content]


async def fake_stream(contents, *, system_instruction=None, temperature=0.7):
    CAPTURED["si"] = system_instruction
    yield "Done — address updated."


async def fake_search(query, top_k=None):
    return []


loop_mod.perceive = fake_perceive
gemini_client.stream = fake_stream
retriever.search = fake_search


async def run_turn(sid, msg):
    out = []
    async for ev in agent_loop.stream(session_id=sid, customer_id=CUST, message=msg):
        out.append(ev)
    return out


def captured_count(order_id):
    with connect() as c:
        return c.execute(
            "SELECT COUNT(*) n FROM payments WHERE order_id=? AND is_refund=0 AND status='captured'",
            (order_id,),
        ).fetchone()["n"]


def address():
    with connect() as c:
        return c.execute("SELECT address FROM customers WHERE id=?", (CUST,)).fetchone()["address"]


async def main():
    before = captured_count("1234")  # 2 (duplicate)

    # MIXED compound: safe (update_address) + money (issue_refund duplicate).
    PLAN["change my address to 99 New Lane Pune and refund the duplicate on 1234"] = Perception(
        emotion=Emotion(state="frustrated", intensity=3),
        intents=["address_change", "refund"],
        action_type="ACT",
        tools=[
            ToolCall(name="update_address", args_json='{"customer_id":"cust_demo","new_address":"99 New Lane, Pune 411001"}'),
            ToolCall(name="issue_refund", args_json='{"order_id":"1234","amount":1499,"reason":"duplicate"}'),
        ],
    )
    ev = await run_turn("m1", "change my address to 99 New Lane Pune and refund the duplicate on 1234")

    tool_names = [e["name"] for e in ev if e["type"] == "tool" and e["status"] == "done"]
    assert "update_address" in tool_names, "safe action should have run"
    assert "issue_refund" not in tool_names, "money action must NOT run before confirmation"
    assert "99 New Lane, Pune 411001" in (address() or ""), "address change should be applied"
    assert captured_count("1234") == before, "refund must NOT have happened yet"

    done = next(e for e in ev if e["type"] == "done")
    assert done["awaiting_confirmation"] is True
    pend = await store.get_pending_action("m1")
    assert pend and len(pend["tools"]) == 1 and pend["tools"][0]["name"] == "issue_refund", pend
    assert "STILL NEEDS THE CUSTOMER'S CONFIRMATION" in CAPTURED["si"], "combined proposal missing"
    assert "LANGUAGE" in CAPTURED["si"] and "Hinglish" in CAPTURED["si"], "language directive missing"
    print("1 MIXED: safe action ran, money action gated, combined reply + language directive: ok")

    # Confirm 'yes' -> the gated refund executes.
    PLAN["yes go ahead"] = Perception(
        emotion=Emotion(), intents=["confirm"], action_type="ANSWER", tools=[], pending_decision="confirm"
    )
    ev2 = await run_turn("m1", "yes go ahead")
    done2 = [e["name"] for e in ev2 if e["type"] == "tool" and e["status"] == "done"]
    assert "issue_refund" in done2, "refund should run after confirmation"
    assert captured_count("1234") == before - 1, "one duplicate charge should now be refunded"
    assert await store.get_pending_action("m1") is None, "pending should be cleared"
    print("2 confirm 'yes' executes the gated refund: ok")

    print("\n=== ALL PHASE 8 MULTI-INTENT TESTS PASS ===")


if __name__ == "__main__":
    asyncio.run(main())
