"""Phase 6 cross-session memory tests (no API key needed — LLM calls faked).

Verifies: (1) a session is digested into long-term memory, (2) the memory is
injected into a later session's prompt, (3) the loop auto-summarizes a prior
session when the customer returns, and (4) seeded demo memory loads.
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
from agent.perception import Emotion, Perception  # noqa: E402
from knowledge.retriever import retriever  # noqa: E402
from llm import gemini_client  # noqa: E402
from memory.store import store  # noqa: E402
from memory.summarize import CustomerMemory, summarize_session  # noqa: E402
from tools.seed import seed  # noqa: E402

store.init()
seed()
CUST = "cust_demo"

# --- fakes -----------------------------------------------------------------
CAPTURED = {}


async def fake_perceive(*, history, customer_id, pending_action, memory_text="", llm=None):
    CAPTURED["perceive_memory"] = memory_text
    return Perception(emotion=Emotion(), intents=["chat"], action_type="ANSWER", tools=[])


async def fake_stream(contents, *, system_instruction=None, temperature=0.7):
    CAPTURED["respond_si"] = system_instruction
    yield "Hello again!"


async def fake_search(query, top_k=None):
    return []


async def fake_generate_structured(prompt, *, response_schema, system_instruction=None, temperature=0.2):
    return CustomerMemory(
        summary="Customer Aarav contacted us about delayed order #1190; now resolved.",
        open_items=["confirm order #1190 arrived"],
        preferences=["prefers UPI"],
        sentiment="relieved",
    )


loop_mod.perceive = fake_perceive
gemini_client.stream = fake_stream
gemini_client.generate_structured = fake_generate_structured
retriever.search = fake_search


async def run_turn(session_id, message):
    out = []
    async for ev in agent_loop.stream(session_id=session_id, customer_id=CUST, message=message):
        out.append(ev)
    return out


async def main():
    # 1) Direct summarization of a session into memory.
    await store.ensure_session("s1", CUST)
    await store.add_message("s1", "user", "My order 1190 is really late, I'm frustrated.")
    await store.add_message("s1", "bot", "So sorry — I see #1190 was delayed; it's now delivered.")
    ok = await summarize_session("s1", CUST)
    assert ok is True
    mem = await store.get_customer_memory(CUST)
    assert mem and "1190" in mem.summary and "confirm order #1190 arrived" in mem.open_items
    print("1 summarize_session -> memory written: ok")

    # 2) Memory is injected into a later session's RESPOND + PERCEIVE prompts.
    await run_turn("s2", "hi there")
    assert "1190" in CAPTURED["respond_si"], "memory not in RESPOND prompt"
    assert "Last time you reached out" in CAPTURED["respond_si"], "recall nudge missing"
    assert "1190" in CAPTURED["perceive_memory"], "memory not in PERCEIVE prompt"
    print("2 memory injected into next session's prompts: ok")

    # 3) Loop auto-summarizes a prior un-summarized session when the customer returns.
    await store.ensure_session("s3", CUST)
    await store.add_message("s3", "user", "Also, can you update my address?")
    await store.add_message("s3", "bot", "Done — address updated.")
    assert await store.get_prior_unsummarized_session(CUST, "s4") == "s3"
    await run_turn("s4", "hello")  # first turn of s4 digests the most-recent prior (s3)
    from memory.db import connect as mconnect

    with mconnect() as mc:
        row = mc.execute(
            "SELECT summarized FROM conversations WHERE session_id='s3'"
        ).fetchone()
    assert row and row["summarized"] == 1, "s3 should be summarized after the customer returned"
    print("3 loop auto-summarized prior session on return: ok")

    # 4) Seeded demo memory (only if absent).
    store.seed_demo_memory("cust_777", "Seeded summary about cust_777.", ["open A"], [], "neutral")
    seeded = await store.get_customer_memory("cust_777")
    assert seeded and "cust_777" in seeded.summary and seeded.open_items == ["open A"]
    # seeding again must NOT overwrite
    store.seed_demo_memory("cust_777", "DIFFERENT", [], [], "x")
    again = await store.get_customer_memory("cust_777")
    assert again.summary == "Seeded summary about cust_777.", "seed must not clobber existing memory"
    print("4 seeded demo memory (no clobber): ok")

    print("\n=== ALL PHASE 6 MEMORY TESTS PASS ===")


if __name__ == "__main__":
    asyncio.run(main())
