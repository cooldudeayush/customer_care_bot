# 🤝 Agentic Customer Care Bot

> A care agent that **resolves** problems by taking real actions — not a chatbot that
> tells customers how to solve problems themselves.

**Cliché bot:** *"To get a refund, please go to Orders › Returns and…"*
**This bot:** *"I've checked order #1234 — it's within the 30-day window, so I've issued
your refund of ₹1,499. You'll see it in 3–5 days. Anything else?"*

---

## 🔗 Live Demo

- **App (frontend):** _<add Vercel URL here>_
- **API (backend):** _<add Render URL here>_ · health check: `/health`
- **Demo video:** _<add link here>_

> ⚠️ These links are filled in during Phase 0/10 deploy. The repo is public and all
> links must be live at submission.

---

## ✨ What's novel (the pitch)

A **hybrid knowledge layer** + **agentic resolution**, combined:

1. **Agentic resolution** — calls real (mock) business tools to *do* things (issue a
   refund, cancel an order, update an address), gated by confirmation for money moves.
2. **Hybrid knowledge** — **LightRAG** over policy docs (unstructured) **+ Neo4j**
   customer operational graph (structured), reasoned over together:
   *"Your order #1234 contains item X, purchased 12 days ago — within our 30-day window
   — so I can refund it now."* (graph traversal + policy retrieval + a tool call).
3. **Adaptive emotional tone** — detects frustration/confusion and modulates tone
   (not naive cheerfulness).
4. **Cross-session memory** — remembers the *customer*, not just the chat.
5. **Graceful escalation** — hands off to a human *with a full summary*.

Plus a **ChatGPT-style chat sidebar** (browse past sessions) and Hinglish/code-switching support.

---

## 🏗️ Architecture (high level)

```
 Next.js + Tailwind chat UI  ──HTTP/SSE──▶  FastAPI (stateless)
                                              │
                                  Agent loop (custom async):
                                  PERCEIVE → RETRIEVE → DECIDE → ACT → RESPOND → REMEMBER
                                              │
       ┌──────────────┬──────────────┬───────┴───────┬──────────────┐
   Gemini Flash    LightRAG       Neo4j graph     Mock biz APIs   SQLite / (Supabase)
   (LLM, 2         (doc RAG)      (customer ops    (real actions)  (memory + traces)
    calls/turn)                    reasoning)
```

- **Stateless backend** — all state externalized (scalable, swappable stores).
- **~2 LLM calls/turn** by design — survives Gemini's free-tier (~15 req/min).
- See [`docs/architecture.md`](docs/architecture.md) for the full design.

---

## 🚀 Setup (from a clean clone)

### Prerequisites
- Python 3.10+ · Node.js 18+ · (Docker — for Neo4j, Phase 4) · A Gemini API key
  ([get one](https://aistudio.google.com/apikey))

### 1. Backend (FastAPI)
```bash
cd backend
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt

# configure your key
cp .env.example .env        # then edit .env and paste your GEMINI_API_KEY

uvicorn main:app --reload --port 8000
```
Verify: <http://localhost:8000/health> · <http://localhost:8000/health/llm> · <http://localhost:8000/docs>

### 2. Frontend (Next.js)
```bash
cd frontend
npm install
cp .env.example .env.local   # NEXT_PUBLIC_API_URL defaults to http://localhost:8000
npm run dev
```
Open <http://localhost:3000>.

---

## 🧪 Project status

Built in phases (see [`EXECUTION_PLAN.md`](EXECUTION_PLAN.md)).
**Phase 0 (foundation) is complete:** repo scaffold, FastAPI skeleton with health +
stub `/chat`, Gemini wrapper (unified `google-genai` SDK), and a Next.js chat UI with
the session sidebar. Phases 1–10 layer the conversational loop, grounding, agentic
tools, the customer graph, emotion, memory, escalation, and polish.

---

## 📁 Repo structure

```
customer-care-bot/
├── README.md
├── EXECUTION_PLAN.md          # the phase-by-phase build plan
├── render.yaml                # Render deploy blueprint (backend)
├── docs/                      # architecture.md, demo_script.md
├── frontend/                  # Next.js + Tailwind chat UI (sidebar + chat)
├── backend/
│   ├── main.py                # FastAPI app, /chat endpoint
│   ├── config.py              # single config module (.env driven)
│   ├── llm.py                 # Gemini client (google-genai, backoff, streaming)
│   ├── agent/                 # the loop: perceive, decide, act, respond, remember
│   ├── knowledge/             # LightRAG + Neo4j client + retrieval
│   ├── tools/                 # tool defs + mock biz APIs
│   ├── memory/                # short-term state + long-term store
│   └── emotion/               # emotion read + tone policy
├── data/
│   ├── corpus/                # policy/FAQ markdown for LightRAG
│   └── seed/                  # SQLite seed + Neo4j load script
└── tests/                     # tool tests + scenario tests
```

---

## 🔐 Configuration

All config is environment-driven (`backend/.env`, `frontend/.env.local`). No secrets in
code. See `*.env.example` files for the full list of variables.
