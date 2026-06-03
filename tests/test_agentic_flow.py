"""Phase 3 scenario tests — the agentic loop, with PERCEIVE scripted.

Runs the real DECIDE/ACT/CONFIRM/RESPOND pipeline against the real mock business
DB, but injects scripted Perception objects (and a fake token stream) so it needs
NO Gemini API key. This is the end-to-end proof that agentic resolution works.

Run from the repo root:  backend/.venv/Scripts/python.exe tests/test_agentic_flow.py
"""

import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

# Fresh conversation DB for the test run.
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
seed()  # fresh business state

# --- patches: no real LLM / retrieval -------------------------------------
PLAN: dict[str, Perception] = {}


async def fake_perceive(*, history, customer_id, pending_action, llm=None):
    return PLAN[history[-1].content]


async def fake_stream(contents, *, system_instruction=None, temperature=0.7):
    for t in ["Okay — ", "all done."]:
        yield t


async def fake_search(query, top_k=None):
    return []


loop_mod.perceive = fake_perceive
gemini_client.stream = fake_stream
retriever.search = fake_search


# --- helpers ---------------------------------------------------------------
async def run_turn(session_id, message, customer_id="cust_demo"):
    events = []
    async for ev in agent_loop.stream(
        session_id=session_id, customer_id=customer_id, message=message
    ):
        events.append(ev)
    return events


def types(events):
    return [e["type"] for e in events]


def text(events):
    return "".join(e["content"] for e in events if e["type"] == "token")


def tools(events):
    return [e for e in events if e["type"] == "tool"]


def done(events):
    return next(e for e in events if e["type"] == "done")


def captured_count(order_id):
    with connect() as c:
        return c.execute(
            "SELECT COUNT(*) n FROM payments WHERE order_id=? AND is_refund=0 AND status='captured'",
            (order_id,),
        ).fetchone()["n"]


def order_status(order_id):
    with connect() as c:
        return c.execute("SELECT status FROM orders WHERE id=?", (order_id,)).fetchone()["status"]


def ticket_count():
    with connect() as c:
        return c.execute("SELECT COUNT(*) n FROM tickets").fetchone()["n"]


def E(state="neutral", i=1):
    return Emotion(state=state, intensity=i)


async def main():
    # A) ANSWER
    PLAN["what is your refund policy?"] = Perception(
        emotion=E(), intents=["policy"], action_type="ANSWER", tools=[]
    )
    ev = await run_turn("sA", "what is your refund policy?")
    assert "token" in types(ev) and "done" in types(ev), types(ev)
    assert not tools(ev)
    assert text(ev) == "Okay — all done."
    print("A ANSWER: ok")

    # B) ACT with a safe read tool
    PLAN["where is order 1234"] = Perception(
        emotion=E(), intents=["track"], action_type="ACT",
        tools=[ToolCall(name="check_order_status", args_json='{"order_id":"1234"}')],
    )
    ev = await run_turn("sB", "where is order 1234")
    te = tools(ev)
    assert any(t["status"] == "running" for t in te) and any(t["status"] == "done" for t in te)
    assert all(t.get("success", True) for t in te if t["status"] == "done")
    assert "token" in types(ev)
    print("B ACT(read): ok")

    # C) CONFIRM gate on a refund — proposes, does NOT execute
    before = captured_count("1234")  # 2 (duplicate)
    PLAN["refund the duplicate charge on 1234"] = Perception(
        emotion=E("frustrated", 4), intents=["refund"], action_type="CONFIRM",
        tools=[ToolCall(name="issue_refund", args_json='{"order_id":"1234","amount":1499,"reason":"duplicate"}')],
        confirm_message="I can refund the duplicate Rs.1499 on order #1234 — shall I go ahead?",
    )
    ev = await run_turn("sC", "refund the duplicate charge on 1234")
    assert not tools(ev), "no tool should run before confirmation"
    assert "shall i go ahead" in text(ev).lower()
    assert done(ev)["awaiting_confirmation"] is True
    assert await store.get_pending_action("sC") is not None
    assert captured_count("1234") == before, "refund must NOT have happened yet"
    print("C CONFIRM gate (refund proposed, not executed): ok")

    # C2) Gate ENFORCEMENT: model says ACT for a refund -> forced to CONFIRM
    PLAN["just refund order 1260"] = Perception(
        emotion=E(), intents=["refund"], action_type="ACT",
        tools=[ToolCall(name="cancel_order", args_json='{"order_id":"1260"}')],
    )
    ev = await run_turn("sC2", "just refund order 1260")
    assert not tools(ev), "confirmation gate must block direct execution"
    assert done(ev)["awaiting_confirmation"] is True
    assert order_status("1260") == "placed", "order must not be cancelled yet"
    print("C2 CONFIRM gate enforcement (ACT->CONFIRM): ok")

    # D) Confirm 'yes' executes the pending refund
    PLAN["yes go ahead"] = Perception(
        emotion=E(), intents=["confirm"], action_type="ANSWER", tools=[], pending_decision="confirm"
    )
    ev = await run_turn("sC", "yes go ahead")
    te = tools(ev)
    assert any(t["name"] == "issue_refund" and t["status"] == "done" and t["success"] for t in te), te
    assert done(ev)["awaiting_confirmation"] is False
    assert await store.get_pending_action("sC") is None, "pending must be cleared"
    assert captured_count("1234") == before - 1, "one duplicate charge should now be refunded"
    assert order_status("1234") == "delivered", "duplicate refund must not refund the whole order"
    print("D Confirm 'yes' executes refund (duplicate reversed): ok")

    # E) Deny cancels the pending action
    PLAN["cancel order 1260 please"] = Perception(
        emotion=E(), intents=["cancel"], action_type="CONFIRM",
        tools=[ToolCall(name="cancel_order", args_json='{"order_id":"1260"}')],
        confirm_message="I can cancel order #1260 and refund Rs.2999 — shall I go ahead?",
    )
    await run_turn("sE", "cancel order 1260 please")
    assert await store.get_pending_action("sE") is not None
    PLAN["no don't"] = Perception(
        emotion=E(), intents=["deny"], action_type="ANSWER", tools=[], pending_decision="deny"
    )
    ev = await run_turn("sE", "no don't")
    assert await store.get_pending_action("sE") is None
    assert order_status("1260") == "placed", "order must NOT be cancelled after a deny"
    assert "won't" in text(ev).lower() or "no problem" in text(ev).lower()
    print("E Deny cancels pending action: ok")

    # F) CLARIFY
    PLAN["i have a problem"] = Perception(
        emotion=E("confused", 3), intents=["unknown"], action_type="CLARIFY",
        needs_clarification=True, clarification_question="Could you tell me which order this is about?",
    )
    ev = await run_turn("sF", "i have a problem")
    assert not tools(ev)
    assert "which order" in text(ev).lower()
    print("F CLARIFY: ok")

    # G) ESCALATE creates a ticket
    t0 = ticket_count()
    PLAN["i'll take legal action"] = Perception(
        emotion=E("angry", 5), intents=["legal"], action_type="ESCALATE", reasoning="legal threat"
    )
    ev = await run_turn("sG", "i'll take legal action")
    assert "specialist" in text(ev).lower()
    assert ticket_count() == t0 + 1, "escalation should open a ticket"
    print("G ESCALATE (ticket opened): ok")

    print("\n=== ALL PHASE 3 AGENTIC SCENARIOS PASS ===")


if __name__ == "__main__":
    asyncio.run(main())
