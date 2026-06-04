# Demo Script

A single ~90-second flow that shows off every differentiator. Uses the seeded demo
customer **`cust_demo` (Aarav Sharma)**. Script it, rehearse it, record it.

## Setup (once)
1. `GEMINI_API_KEY` set in `backend/.env`; run `python -m knowledge.ingest`.
2. Start backend (`uvicorn main:app --reload`) and frontend (`npm run dev`).
3. (Optional) `docker compose up -d` + `NEO4J_PASSWORD=password123` to light up the graph.
4. Open the app — note the **"What we know about you"** panel already shows Aarav's
   orders and the seeded memory (a previously-delayed order #1190).

## Seeded data you can rely on
| Order | Item | State | Hook |
|---|---|---|---|
| **#1234** | Wireless Headphones (electronics) | delivered 12 days ago, **duplicate ₹1,499 charge** | refund the duplicate |
| #1190 | Cotton T-Shirt | delivered 40 days ago | window expired + the "delayed last month" memory |
| #1255 | Phone Case | in transit | track / reschedule |
| #1260 | Bluetooth Speaker | placed (not shipped) | cancellable |

---

## The flow

**1. Angry + Hinglish + multi-intent** — type:
> *"I was charged TWICE for order 1234 and I'm furious. Also I need my delivery address changed to 50 MG Road Bengaluru, kuch toh karo."*

Watch for:
- 😠 **emotion badge** ("sensed: angry/frustrated") and a calm, apologetic tone.
- 🔧 **tool chips** as it checks the order and finds the **duplicate ₹1,499** charge.
- The reply **mirrors light Hinglish** and handles **both intents**: it updates the
  address immediately and **asks to confirm the refund** (the money action is gated).

**2. Confirm the refund** — click **Yes, go ahead** (or type "haan kar do").
- 🔧 **Issuing refund** chip → success. The reply confirms ₹1,499 back in 3–5 days.
- The **panel updates** (and, with Neo4j on, the order's graph status flips).

**3. Grounded policy question** — type:
> *"What's your refund policy for electronics?"*
- Answers **15 days** from the real docs, with a **"📄 Grounded in: refund_policy.md"** footer.

**4. Cross-session memory** — type:
> *"Did my earlier delayed order ever arrive?"*
- Recalls order **#1190** from long-term memory ("that's been delivered now…").

**5. Graceful escalation** — type something out of scope:
> *"I want to take legal action over a warranty dispute."*
- 🤝 **Handoff card** appears (issue · actions taken · next step · ticket #), and the
  bot reassures: *"I've shared the full context with a specialist — you won't need to
  repeat anything."* Show the specialist inbox at **`GET /handoffs`**.

**End state:** angry customer leaves calm · double charge refunded · address fixed ·
nothing repeated · clean human handoff.

---

## Bonus things to show judges
- **`/docs`** — the FastAPI OpenAPI explorer (all endpoints).
- **`/traces`** — per-turn observability (emotion, action, tools, latency).
- **`/health`** — `graph_enabled`, `retrieval_chunks`, model.
- The **CONFIRM gate**: try "refund order 1234" and *decline* — nothing happens.

> Record this as a video **and** keep it runnable live. Put both links in the README.
