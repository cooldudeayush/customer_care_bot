# Customer Care Bot — 1-Day Execution Plan

> Senior-architect, ruthlessly-prioritized plan for a **single-day** hackathon build.
> Locked decisions: **Next.js frontend** · **Custom async agent loop** · **Local-first
> swappable infra (Neo4j-in-Docker + SQLite/Chroma), cloud as later swap** · **Gemini Flash**
> · **ChatGPT-style sidebar of past chat sessions** (committed feature, see Phase 1).

---

## The North Star (do not forget under time pressure)
A bot that **talks → grounds in real docs → ACTS (issues a refund) → reasons over a
customer graph** is a *winning* demo. Everything after that is bonus. If the clock
runs out, a working Phase-0→4 bot beats a half-built everything.

**Two non-negotiables regardless of time:**
1. A **working public demo link** (deploy the empty skeleton FIRST, hour 1).
2. A **README** with the link, setup, architecture, and the demo scenario.

---

## Time Budget (≈ 1 working day, 10–12 hrs)

| Block | Phases | Hours | Outcome |
|------|--------|-------|---------|
| **A. Foundation + Talk** | 0, 1 | ~2.5h | Deployed skeleton + multi-turn streaming chat |
| **B. Grounding** | 2 | ~1.5h | Answers from real policy docs |
| **C. Agentic core** | 3 | ~2h | **Bot issues a refund (the 30% headline)** |
| **D. Graph reasoning** | 4 | ~2h | The showcase sentence works end-to-end |
| **E. Differentiator (pick ONE)** | 5 *or* 6 | ~1h | Emotion tone OR cross-session memory |
| **F. Demo + Docs + Deploy** | 10 | ~1.5h | README, demo script, verified live link |

**HARD CUT LINE:** Phases **0–4 + 10** are the committed build. Phases 5–9 are
**stretch**, taken only if a block finishes early. **Phase 10 is never skipped.**

---

## PHASE 0 — FOUNDATION  *(hour 1 — do the deploy FIRST)*
**Goal:** repo skeleton + a live public URL before any real feature.

- [ ] Repo structure (see `docs/` / `backend/` / `frontend/` layout).
- [ ] FastAPI skeleton with a `GET /health` and stub `POST /chat`.
- [ ] `.env` + `.env.example` + a single `config.py` (no hardcoded secrets).
- [ ] Gemini "hello world" call working (verifies the key + SDK).
- [ ] Lean Next.js app: one chat page (Tailwind).
- [ ] **DEPLOY the empty skeleton:** backend → Render/Railway, frontend → Vercel.
      Confirm the public URL loads in an **incognito** window.

**Manual (you):** create Gemini API key; create Render + Vercel accounts; install
Docker Desktop + Node.js. (Full checklist at the bottom.)

**Milestone:** a public URL that loads, even if the bot says nothing useful yet.

---

## PHASE 1 — CORE CONVERSATIONAL LOOP  *(a bot that talks + holds the chat)*
**Goal:** coherent multi-turn streaming chat. This is your **fallback demo**.

- [ ] `POST /chat {session_id, customer_id, message}` → streamed (SSE) reply.
- [ ] Short-term conversation state (in-memory dict keyed by session, persisted to SQLite).
- [ ] Single Gemini generation call with a system prompt (care-agent persona).
- [ ] Next.js: SSE token streaming + typing indicator.

### ChatGPT-style session sidebar  *(committed — reuses the `conversations` store)*
- [ ] `conversations` table persists every session (`session_id, customer_id, title, transcript, started_at`).
- [ ] `GET /sessions?customer_id=X` → list past sessions for the sidebar (id, title, timestamp).
- [ ] `GET /sessions/{session_id}` → full transcript to reopen a chat.
- [ ] `POST /chat` supports starting a **new** session ("+ New chat" button).
- [ ] Auto-title each session from its first user message (e.g. "Refund for order #1234").
- [ ] Next.js: left **sidebar** listing chats; click to reopen; "+ New chat" button.
      *(Pairs with Phase 6: sidebar SHOWS past chats; memory makes the bot RECALL them.)*

**Milestone:** a real, coherent multi-turn conversation in the deployed UI, with a
clickable sidebar of past chat sessions (new chat / reopen old chat both work).

---

## PHASE 2 — GROUNDING  *(make it factual, no hallucinated policies)*
**Goal:** answers come from real docs, not vibes.

- [ ] Write a **small, clean** corpus: `refund_policy.md`, `shipping_faq.md`,
      `returns_guide.md`, `warranty_terms.md` (quality > volume).
- [ ] Stand up retrieval behind a `Retriever` interface.
      **Primary: LightRAG.** **Fallback ready: Chroma** (swap if LightRAG eats >45 min).
- [ ] Inject retrieved chunks + the grounding instruction into the RESPOND prompt:
      *"Only state facts present in context; if unknown, say you'll check / escalate."*

**Milestone:** bot answers "what's the refund policy for electronics?" from the docs.

> ⚠️ **Risk gate:** if LightRAG install/ingest isn't working in 45 min, switch to
> Chroma immediately. Same interface, zero downstream changes. Don't lose the day here.

---

## PHASE 3 — AGENTIC RESOLUTION  *(THE HEADLINE — wins the 30%)*
**Goal:** the bot DOES things, not just tells.

- [ ] Seed **SQLite** mock business DB: 5–10 customers, orders, items, products, payments.
- [ ] Mock biz API functions (Python) over that DB.
- [ ] Define starred tools as Gemini function-calling schemas:
      `check_order_status`, `get_refund_eligibility`, `issue_refund`,
      `cancel_order`, `create_ticket`.
- [ ] Collapse PERCEIVE+DECIDE into **one structured Gemini call** returning
      `{emotion, intents, entities, tool_decision}` (the free-tier discipline).
- [ ] **CONFIRM gate** before `issue_refund` / `cancel_order` (state action, await "yes").
- [ ] ACT step executes tools; RESPOND turns the structured result into tone-right NL.

**Milestone:** bot checks an order and **issues a refund** with a confirmation gate.

---

## PHASE 4 — CUSTOMER KNOWLEDGE GRAPH  *(the novelty / relational reasoning)*
**Goal:** the showcase sentence works end-to-end.

- [ ] Neo4j in **Docker** (local). Load schema (Customer/Order/Item/Product/Payment/
      Ticket/Policy) and mirror the SQLite seed data into the graph.
- [ ] `GraphClient` interface (Cypher driver) — same code path as Aura later.
- [ ] `get_refund_eligibility` runs the Cypher traversal (order→item→product→policy)
      + compares `today - order.date` to `policy.window_days`.
- [ ] Tool writebacks update the graph (refund → order status flips to `refunded`).

**Milestone:** *"Your order #1234 contains item X, purchased 12 days ago — within our
30-day window — so I can refund it now."* Traversal + policy + action, end-to-end.

---

## STRETCH (only if a block above finished early) — pick the highest-impact one

### PHASE 5 — EMOTIONAL ADAPTATION  *(cheap; folds into existing call)*
- [ ] Emotion read already in the structured PERCEIVE output → append a **tone directive**
      to the RESPOND prompt (angry→calm/apologetic; confused→step-by-step; etc.).
- **Milestone:** angry vs confused customer get visibly different tone. *(Highest ROI stretch — ~30 min, big demo payoff.)*

### PHASE 6 — CROSS-SESSION MEMORY
- [ ] Session-end summary (1 Gemini call) → store vs `customer_id`.
- [ ] Session-start: inject "what we know about this customer" into the system prompt.
- **Milestone:** "Last time you contacted us about a delayed order…"

### PHASE 7 — ESCALATION / HANDOFF
- [ ] Triggers + handoff packet (summary, actions taken, sentiment) → ticket / mock view.

### PHASE 8 — MULTI-INTENT + HINGLISH
- [ ] Action queue for compound requests; language-mirroring directive in RESPOND.

### PHASE 9 — GUARDRAILS / POLISH / OBSERVABILITY
- [ ] PII masking, tool-failure fallbacks, turn traces, "what we know" UI panel.

---

## PHASE 10 — DEMO + DOCS + DEPLOY  *(NEVER skip — 20% of score + compliance)*
**Goal:** judges can see it work and read how it's built.

- [ ] **README:** one-line problem, the **WORKING PUBLIC LINK**, clean-clone setup,
      architecture diagram (from the doc's Section 2), what's novel, screenshots/GIF.
- [ ] `docs/demo_script.md`: the angry double-charge + address-change scenario.
- [ ] Record a 90-sec demo video (live + recorded as fallback).
- [ ] Final deploy; verify the public link in **incognito**. Repo **PUBLIC**, all links live.

---

## Risk Register (the things that actually kill 1-day builds)
1. **Broken demo link at submission** → mitigated by deploying empty skeleton hour 1.
2. **LightRAG install/ingest stall** → 45-min timebox, Chroma fallback behind same interface.
3. **Neo4j Aura auto-pause mid-demo** → mitigated by running Neo4j in Docker locally.
4. **Gemini 429 rate-limit during demo** → 2 calls/turn by design + exponential backoff.
5. **Next.js eating frontend time** → keep it to ONE lean chat page; don't gold-plate.
6. **Phase 4 (graph) overrunning** → it's the last *committed* phase; if it stalls,
   the bot already issues refunds from SQLite (Phase 3) — graph becomes a stretch.

---

## What I need from you MANUALLY (do these before/while I code)
> Detailed step-by-step provided when we start Phase 0. High level:
1. **Gemini API key** — get from Google AI Studio, paste into `.env`.
2. **Install:** Docker Desktop, Node.js 18+, Python 3.10+. Confirm `docker`, `node`, `python` run.
3. **Accounts:** Render (or Railway) for backend, Vercel for frontend, GitHub (public repo).
4. **(Later swap, optional):** Neo4j AuraDB + Supabase accounts — only if we go cloud after the demo works.
