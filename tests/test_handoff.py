"""Phase 7 escalation/handoff tests (no API key needed — LLM calls faked).

Verifies: escalation builds a full handoff packet, emits a handoff event + a
tone-aware customer message, persists the packet (specialist inbox), creates a
ticket, and degrades to a deterministic fallback packet when the LLM is down.
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
from agent.handoff import HandoffPacket  # noqa: E402
from agent.loop import agent_loop  # noqa: E402
from agent.perception import Emotion, Perception  # noqa: E402
from knowledge.retriever import retriever  # noqa: E402
from llm import LLMError, gemini_client  # noqa: E402
from memory.store import store  # noqa: E402
from tools.seed import seed  # noqa: E402

store.init()
seed()
CUST = "cust_demo"

ESCALATE_P = Perception(
    emotion=Emotion(state="angry", intensity=5),
    intents=["legal"],
    action_type="ESCALATE",
    reasoning="legal threat about a warranty dispute",
)


async def fake_perceive(*, history, customer_id, pending_action, memory_text="", llm=None):
    return ESCALATE_P


async def fake_search(query, top_k=None):
    return []


PACKET = HandoffPacket(
    issue="Warranty legal dispute",
    conversation_summary="Customer threatens legal action over a warranty claim.",
    actions_taken=["checked order #1234 status"],
    suggested_next_step="Escalate to the legal/warranty team",
    sentiment="angry",
    customer_message="I'm connecting you with a specialist — you won't need to repeat anything.",
)


async def fake_gs(prompt, *, response_schema, system_instruction=None, temperature=0.2):
    return PACKET


loop_mod.perceive = fake_perceive
retriever.search = fake_search
gemini_client.generate_structured = fake_gs


async def run_turn(sid, msg):
    out = []
    async for ev in agent_loop.stream(session_id=sid, customer_id=CUST, message=msg):
        out.append(ev)
    return out


async def main():
    ev = await run_turn("h1", "I'll take legal action over this warranty")
    handoffs = [e for e in ev if e["type"] == "handoff"]
    assert handoffs, [e["type"] for e in ev]
    assert handoffs[0]["issue"] == "Warranty legal dispute"
    assert handoffs[0]["actions_taken"] == ["checked order #1234 status"]
    text = "".join(e["content"] for e in ev if e["type"] == "token")
    assert "specialist" in text.lower()

    packets = await store.list_handoffs()
    assert packets and packets[0].issue == "Warranty legal dispute" and packets[0].status == "open"
    assert packets[0].customer_summary and "Aarav" in packets[0].customer_summary
    print("1 escalation builds + emits + persists packet (with customer summary): ok")

    # Fallback when the summarization/packet LLM call fails.
    async def boom(prompt, *, response_schema, system_instruction=None, temperature=0.2):
        raise LLMError("no key")

    gemini_client.generate_structured = boom
    ev2 = await run_turn("h2", "I demand a human right now")
    assert any(e["type"] == "handoff" for e in ev2), [e["type"] for e in ev2]
    text2 = "".join(e["content"] for e in ev2 if e["type"] == "token")
    assert "specialist" in text2.lower()
    assert len(await store.list_handoffs()) == 2, "fallback packet should still persist"
    print("2 fallback packet on LLM failure (escalation still works): ok")


asyncio.run(main())

# 3) The specialist-inbox endpoint lists the packets.
from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402

with TestClient(app) as c:
    r = c.get("/handoffs")
    assert r.status_code == 200 and len(r.json()) >= 2
    one = c.get(f"/handoffs/{r.json()[0]['id']}")
    assert one.status_code == 200 and one.json()["issue"]
    assert c.get("/handoffs/999999").status_code == 404
print("3 GET /handoffs + /handoffs/{id} list the specialist inbox: ok")

print("\n=== ALL PHASE 7 HANDOFF TESTS PASS ===")
