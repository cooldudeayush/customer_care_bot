"use client";

/**
 * Customer Care Bot — main chat page.
 *
 * Streams replies token-by-token from the backend (SSE over fetch) and wires the
 * ChatGPT-style sidebar to real persisted sessions:
 *   - on load: GET /sessions populates the sidebar
 *   - send:    POST /chat streams the reply; the session is persisted + titled
 *   - click a past chat: GET /sessions/{id} reopens its transcript
 *   - "+ New chat": starts a fresh session id
 *
 * The presentation layer is a "premium agentic-support console": dark gradient
 * sidebar, glass header, animated message bubbles, a live profile panel. All
 * motion is CSS-only (transform/opacity), so the richer UI stays fast.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  checkHealth,
  getCustomerInfo,
  getSession,
  listSessions,
  streamChat,
  type CustomerInfo,
  type SessionSummary,
  type Source,
  type ToolEvent,
  type EmotionEvent,
  type HandoffEvent,
} from "@/lib/api";

const CUSTOMER_ID = "cust_demo";

// One-click starter prompts (shown on an empty chat) — each kicks off a real
// flow against the seeded demo data, while keeping the natural-language UX.
const SUGGESTIONS: { icon: string; text: string }[] = [
  { icon: "💳", text: "I was charged twice for order 1234 and I'm upset" },
  { icon: "📋", text: "What's your refund policy for electronics?" },
  { icon: "🚚", text: "Where is my order 1255?" },
  { icon: "🛑", text: "Cancel order 1260" },
  { icon: "⏱️", text: "Did my earlier delayed order get sorted?" },
];

// Turn a corpus filename into a clean title: refund_policy.md -> "Refund Policy".
const prettySource = (f: string) =>
  f
    .replace(/\.md$/i, "")
    .split("_")
    .map((w) => (w.toLowerCase() === "faq" ? "FAQ" : w.charAt(0).toUpperCase() + w.slice(1)))
    .join(" ");

// Emoji + styling for the detected-emotion badge (neutral is hidden).
const EMOTION_BADGE: Record<string, { emoji: string; cls: string }> = {
  angry: { emoji: "😠", cls: "border-red-200 bg-red-50 text-red-700" },
  frustrated: { emoji: "😟", cls: "border-orange-200 bg-orange-50 text-orange-700" },
  confused: { emoji: "😕", cls: "border-amber-200 bg-amber-50 text-amber-700" },
  happy: { emoji: "😊", cls: "border-emerald-200 bg-emerald-50 text-emerald-700" },
};

// Friendly labels for the tool-activity chips.
const TOOL_LABELS: Record<string, string> = {
  check_order_status: "Checking order",
  get_refund_eligibility: "Checking refund eligibility",
  issue_refund: "Issuing refund",
  cancel_order: "Cancelling order",
  track_shipment: "Tracking shipment",
  update_address: "Updating address",
  reschedule_delivery: "Rescheduling delivery",
  create_ticket: "Opening a ticket",
};

type Role = "user" | "bot";

interface ToolActivity {
  name: string;
  status: "running" | "done";
  success?: boolean;
}

interface Message {
  id: string;
  role: Role;
  text: string;
  sources?: Source[];
  tools?: ToolActivity[];
  emotion?: EmotionEvent;
  handoff?: HandoffEvent;
}

const uuid = () =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.floor(Math.random() * 1e9)}`;

// A stable per-browser token so each visitor sees ONLY their own chats (privacy).
// The bot still acts on the shared demo customer (CUSTOMER_ID) for the orders/data.
function getOwner(): string {
  if (typeof window === "undefined") return "anon";
  let o = window.localStorage.getItem("ccb_owner");
  if (!o) {
    o = uuid();
    window.localStorage.setItem("ccb_owner", o);
  }
  return o;
}

const initialsOf = (name?: string) =>
  (name || "?")
    .split(/\s+/)
    .map((w) => w[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();

// Color-code an order status pill.
function orderStatusStyle(status: string): string {
  const s = status.toLowerCase();
  if (s.includes("deliver")) return "border-emerald-200 bg-emerald-50 text-emerald-700";
  if (s.includes("transit") || s.includes("ship")) return "border-sky-200 bg-sky-50 text-sky-700";
  if (s.includes("cancel")) return "border-slate-200 bg-slate-100 text-slate-500";
  if (s.includes("placed") || s.includes("process") || s.includes("pending"))
    return "border-amber-200 bg-amber-50 text-amber-700";
  return "border-slate-200 bg-slate-50 text-slate-600";
}

/* ---------------- Inline icons (no icon lib — keeps the bundle lean) ---------------- */
function SparklesIcon({ className = "" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M9.94 15.5A2 2 0 0 0 8.5 14.06l-6.13-1.58a.5.5 0 0 1 0-.96L8.5 9.94A2 2 0 0 0 9.94 8.5l1.58-6.14a.5.5 0 0 1 .96 0L14.06 8.5A2 2 0 0 0 15.5 9.94l6.14 1.58a.5.5 0 0 1 0 .96L15.5 14.06a2 2 0 0 0-1.44 1.44l-1.58 6.14a.5.5 0 0 1-.96 0z" />
      <path d="M20 3v4M22 5h-4M4 17v2M5 18H3" />
    </svg>
  );
}
function PlusIcon({ className = "" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M5 12h14M12 5v14" />
    </svg>
  );
}
function SendIcon({ className = "" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M14.54 21.69a.5.5 0 0 0 .94-.02l6.5-19a.5.5 0 0 0-.64-.64l-19 6.5a.5.5 0 0 0-.02.94l7.93 3.18a2 2 0 0 1 1.11 1.11z" />
      <path d="m21.85 2.15-10.94 10.94" />
    </svg>
  );
}

export default function ChatPage() {
  const [owner] = useState<string>(() => getOwner());
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeId, setActiveId] = useState<string>(() => uuid());
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [backendUp, setBackendUp] = useState<boolean | null>(null);
  const [awaitingConfirm, setAwaitingConfirm] = useState(false);
  const [customerInfo, setCustomerInfo] = useState<CustomerInfo | null>(null);
  const [composerFocused, setComposerFocused] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  const refreshCustomerInfo = useCallback(async () => {
    setCustomerInfo(await getCustomerInfo(CUSTOMER_ID));
  }, []);

  // Initial load: backend health + existing sessions + what-we-know panel.
  // setState runs inside async .then callbacks (not synchronously in the effect
  // body), so it never cascades renders; `cancelled` guards a fast unmount.
  useEffect(() => {
    let cancelled = false;
    checkHealth().then((up) => {
      if (!cancelled) setBackendUp(up);
    });
    listSessions(owner).then((s) => {
      if (!cancelled) setSessions(s);
    });
    getCustomerInfo(CUSTOMER_ID).then((c) => {
      if (!cancelled) setCustomerInfo(c);
    });
    return () => {
      cancelled = true;
    };
  }, [owner]);

  // Auto-scroll to the newest content.
  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, sending]);

  const activeInList = sessions.some((s) => s.session_id === activeId);
  const draftTitle =
    messages.find((m) => m.role === "user")?.text.slice(0, 40) || "New chat";
  const sidebarItems = activeInList
    ? sessions
    : [
        {
          session_id: activeId,
          title: draftTitle,
        } as Pick<SessionSummary, "session_id" | "title">,
        ...sessions,
      ];

  function resetComposerHeight() {
    if (taRef.current) taRef.current.style.height = "auto";
  }

  function handleInputChange(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setInput(e.target.value);
    const el = taRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 168)}px`;
    }
  }

  function handleNewChat() {
    if (sending) return;
    setActiveId(uuid());
    setMessages([]);
    setInput("");
    setAwaitingConfirm(false);
    resetComposerHeight();
  }

  async function handleSelect(id: string) {
    if (sending || id === activeId) return;
    setActiveId(id);
    setMessages([]);
    setAwaitingConfirm(false);
    setLoadingHistory(true);
    const detail = await getSession(id);
    setLoadingHistory(false);
    if (detail) {
      setMessages(
        detail.messages.map((m, i) => ({
          // Stable key across reloads (index within this session) so React
          // doesn't remount every message bubble each time a chat is reopened.
          id: `${id}:${i}`,
          role: m.role === "bot" ? "bot" : "user",
          text: m.content,
        })),
      );
    }
  }

  async function handleSend(textArg?: string) {
    const text = (textArg ?? input).trim();
    // Don't allow a send while a stream is in flight or while a past chat's
    // history is still loading (otherwise the turn lands on the wrong session).
    if (!text || sending || loadingHistory) return;

    const botId = uuid();
    setMessages((prev) => [
      ...prev,
      { id: uuid(), role: "user", text },
      { id: botId, role: "bot", text: "" },
    ]);
    setInput("");
    resetComposerHeight();
    setSending(true);
    setAwaitingConfirm(false);

    const appendToBot = (chunk: string) =>
      setMessages((prev) =>
        prev.map((m) => (m.id === botId ? { ...m, text: m.text + chunk } : m)),
      );
    const setBot = (value: string) =>
      setMessages((prev) =>
        prev.map((m) => (m.id === botId ? { ...m, text: value } : m)),
      );
    const setSources = (sources: Source[]) =>
      setMessages((prev) =>
        prev.map((m) => (m.id === botId ? { ...m, sources } : m)),
      );
    const addTool = (t: ToolEvent) =>
      setMessages((prev) =>
        prev.map((m) => {
          if (m.id !== botId) return m;
          const tools = [...(m.tools ?? [])];
          if (t.status === "running") {
            tools.push({ name: t.name, status: "running" });
          } else {
            // mark the last running entry for this tool as done
            for (let i = tools.length - 1; i >= 0; i--) {
              if (tools[i].name === t.name && tools[i].status === "running") {
                tools[i] = { name: t.name, status: "done", success: t.success };
                break;
              }
            }
          }
          return { ...m, tools };
        }),
      );

    await streamChat(
      { message: text, session_id: activeId, customer_id: CUSTOMER_ID, owner },
      {
        onToken: appendToBot,
        onSources: setSources,
        onTool: addTool,
        onEmotion: (em) =>
          setMessages((prev) =>
            prev.map((m) => (m.id === botId ? { ...m, emotion: em } : m)),
          ),
        onHandoff: (h) =>
          setMessages((prev) =>
            prev.map((m) => (m.id === botId ? { ...m, handoff: h } : m)),
          ),
        onDone: (e) => {
          // Only update confirm state if this stream is for the active session
          // (guards against a late stream landing after a session switch).
          if (e.session_id === activeId) {
            setAwaitingConfirm(Boolean(e.awaiting_confirmation));
          }
          // Reflect the (possibly new) session + title in the sidebar.
          setSessions((prev) => {
            const exists = prev.some((s) => s.session_id === e.session_id);
            if (exists) {
              return prev.map((s) =>
                s.session_id === e.session_id ? { ...s, title: e.title } : s,
              );
            }
            const now = new Date().toISOString();
            return [
              {
                session_id: e.session_id,
                customer_id: CUSTOMER_ID,
                title: e.title,
                started_at: now,
                updated_at: now,
                message_count: 2,
              },
              ...prev,
            ];
          });
        },
        onError: (msg) => setBot(`⚠️ ${msg}`),
      },
    );

    setSending(false);
    refreshCustomerInfo(); // orders/memory may have changed this turn
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  const statusLabel =
    backendUp == null ? "Connecting…" : backendUp ? "Online" : "Offline";
  const statusDot =
    backendUp == null ? "bg-slate-400" : backendUp ? "bg-emerald-500" : "bg-red-500";
  // The currently-streaming bot message (last one while sending) gets the
  // live typing dots / blinking caret.
  const liveId = sending ? messages[messages.length - 1]?.id : null;

  return (
    <div className="app-bg flex h-screen w-full overflow-hidden text-slate-900">
      {/* ---------------- SIDEBAR ---------------- */}
      <aside className="sidebar-bg scroll-dark hidden w-72 shrink-0 flex-col md:flex">
        {/* Brand */}
        <div className="flex items-center gap-3 px-5 py-5">
          <div className="brand-grad-animated grid h-10 w-10 place-items-center rounded-xl text-white shadow-lg shadow-indigo-900/40">
            <SparklesIcon className="h-5 w-5" />
          </div>
          <div className="leading-tight">
            <div className="text-sm font-semibold text-white">Care Agent</div>
            <div className="text-[11px] text-slate-400">Agentic support</div>
          </div>
        </div>

        {/* New chat */}
        <div className="px-3">
          <button
            onClick={handleNewChat}
            disabled={sending}
            className="brand-grad lift flex w-full items-center justify-center gap-2 rounded-xl px-3 py-2.5 text-sm font-semibold text-white shadow-lg shadow-indigo-900/30 hover:shadow-indigo-900/50 disabled:opacity-40 disabled:hover:translate-y-0"
          >
            <PlusIcon className="h-4 w-4" />
            New chat
          </button>
        </div>

        {/* Chats */}
        <div className="px-5 pb-2 pt-6 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
          Recent chats
        </div>
        <nav className="scroll-dark flex-1 overflow-y-auto px-2 pb-3">
          {sidebarItems.length === 0 && (
            <p className="px-3 py-2 text-sm text-slate-500">No chats yet.</p>
          )}
          {sidebarItems.map((s) => {
            const active = s.session_id === activeId;
            return (
              <button
                key={s.session_id}
                onClick={() => handleSelect(s.session_id)}
                className={`group relative mb-0.5 flex w-full items-center gap-2 truncate rounded-lg px-3 py-2 text-left text-[13px] transition ${
                  active
                    ? "bg-white/10 font-medium text-white"
                    : "text-slate-400 hover:bg-white/5 hover:text-slate-200"
                }`}
                title={s.title}
              >
                {active && (
                  <span className="brand-grad absolute left-0 top-1/2 h-5 w-[3px] -translate-y-1/2 rounded-full" />
                )}
                <span className="truncate">{s.title || "New chat"}</span>
              </button>
            );
          })}
        </nav>

        {/* Status */}
        <div className="border-t border-white/5 px-4 py-3.5">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <span className="relative flex h-2 w-2">
              <span className={`${backendUp ? "pulse-ring" : ""} inline-block h-2 w-2 rounded-full ${statusDot}`} />
            </span>
            Backend {statusLabel}
          </div>
        </div>
      </aside>

      {/* ---------------- MAIN CHAT AREA ---------------- */}
      <main className="relative flex flex-1 flex-col">
        <header className="glass z-10 flex items-center justify-between border-b border-slate-200/70 px-5 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="brand-grad grid h-9 w-9 place-items-center rounded-xl text-white shadow-md shadow-indigo-500/30 md:hidden">
              <SparklesIcon className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-[15px] font-semibold tracking-tight text-slate-800">
                Customer Care Bot
              </h1>
              <p className="hidden text-xs text-slate-500 sm:block">
                Resolves problems by taking real actions — not just telling you how.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 rounded-full border border-slate-200 bg-white/70 px-3 py-1 text-xs font-medium text-slate-500">
            <span className={`inline-block h-2 w-2 rounded-full ${statusDot} ${backendUp ? "pulse-ring" : ""}`} />
            {statusLabel}
          </div>
        </header>

        <div ref={scrollRef} className="scroll-thin flex-1 overflow-y-auto px-4 py-6 sm:px-6">
          <div className="mx-auto flex max-w-3xl flex-col gap-5">
            {loadingHistory && (
              <div className="mt-10 flex flex-col items-center gap-2 text-sm text-slate-400">
                <div className="flex gap-1">
                  <span className="typing-dot" />
                  <span className="typing-dot" style={{ animationDelay: "0.15s" }} />
                  <span className="typing-dot" style={{ animationDelay: "0.3s" }} />
                </div>
                Loading conversation…
              </div>
            )}

            {/* Empty-state hero */}
            {!loadingHistory && messages.length === 0 && (
              <div className="flex flex-col items-center gap-6 pt-8 animate-fade-up sm:pt-12">
                <div className="brand-grad-animated animate-float grid h-16 w-16 place-items-center rounded-2xl text-white shadow-xl shadow-indigo-500/30">
                  <SparklesIcon className="h-8 w-8" />
                </div>
                <div className="text-center">
                  <h2 className="text-2xl font-bold tracking-tight text-slate-800">
                    Hi! I&apos;m your <span className="text-brand-grad">Care Agent</span>
                  </h2>
                  <p className="mx-auto mt-2 max-w-md text-sm leading-relaxed text-slate-500">
                    I don&apos;t just answer questions — I take real actions: refunds,
                    order tracking, address changes &amp; more. Try one:
                  </p>
                </div>
                <div className="grid w-full max-w-xl grid-cols-1 gap-2.5 sm:grid-cols-2">
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s.text}
                      onClick={() => handleSend(s.text)}
                      disabled={sending}
                      className="lift group flex items-center gap-3 rounded-xl border border-slate-200 bg-white/80 px-4 py-3 text-left text-sm text-slate-700 shadow-sm hover:border-indigo-300 hover:shadow-md disabled:opacity-40 disabled:hover:translate-y-0"
                    >
                      <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-indigo-50 text-base transition group-hover:bg-indigo-100">
                        {s.icon}
                      </span>
                      <span className="leading-snug">{s.text}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Messages */}
            {messages.map((m) => {
              const usedSources = m.sources
                ? Array.from(new Set(m.sources.map((s) => s.source)))
                : [];
              const isBot = m.role === "bot";
              const isLive = isBot && m.id === liveId;
              return (
                <div
                  key={m.id}
                  className={`flex items-start gap-3 animate-message-in ${
                    isBot ? "" : "flex-row-reverse"
                  }`}
                >
                  {/* Avatar */}
                  {isBot ? (
                    <div className="brand-grad grid h-8 w-8 shrink-0 place-items-center rounded-lg text-white shadow-md shadow-indigo-500/25">
                      <SparklesIcon className="h-4 w-4" />
                    </div>
                  ) : (
                    <div className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-slate-200 text-[11px] font-semibold text-slate-600">
                      You
                    </div>
                  )}

                  {/* Content column */}
                  <div
                    className={`flex min-w-0 max-w-[80%] flex-col ${
                      isBot ? "items-start" : "items-end"
                    }`}
                  >
                    {/* Emotion badge */}
                    {isBot &&
                      m.emotion &&
                      m.emotion.state !== "neutral" &&
                      EMOTION_BADGE[m.emotion.state] && (
                        <div
                          className={`mb-1.5 inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium ${EMOTION_BADGE[m.emotion.state].cls}`}
                        >
                          {EMOTION_BADGE[m.emotion.state].emoji} sensed: {m.emotion.state}
                          {m.emotion.intensity >= 4 ? " (high)" : ""}
                        </div>
                      )}

                    {/* Handoff card */}
                    {isBot && m.handoff && (
                      <div className="mb-1.5 w-full rounded-xl border border-indigo-200 bg-gradient-to-br from-indigo-50 to-violet-50 px-3.5 py-3 text-xs text-indigo-900 shadow-sm">
                        <div className="flex items-center gap-1.5 font-semibold">
                          🤝 Handed off to a specialist
                          <span className="rounded-full bg-indigo-100 px-1.5 py-0.5 text-[10px] font-medium text-indigo-700">
                            ticket #{m.handoff.packet_id}
                          </span>
                        </div>
                        {m.handoff.issue && (
                          <div className="mt-1.5">
                            <span className="font-medium">Issue:</span> {m.handoff.issue}
                          </div>
                        )}
                        {m.handoff.actions_taken.length > 0 && (
                          <div className="mt-1">
                            <span className="font-medium">Actions taken:</span>{" "}
                            {m.handoff.actions_taken.join("; ")}
                          </div>
                        )}
                        {m.handoff.suggested_next_step && (
                          <div className="mt-1">
                            <span className="font-medium">Next step:</span>{" "}
                            {m.handoff.suggested_next_step}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Tool activity chips */}
                    {isBot && m.tools && m.tools.length > 0 && (
                      <div className="mb-1.5 flex flex-wrap gap-1.5">
                        {m.tools.map((t, i) => {
                          const label = TOOL_LABELS[t.name] ?? t.name;
                          if (t.status === "running") {
                            return (
                              <span
                                key={i}
                                className="inline-flex items-center gap-1.5 rounded-full border border-amber-200 bg-amber-50 px-2.5 py-0.5 text-xs font-medium text-amber-700"
                              >
                                <span className="spinner h-3 w-3 rounded-full border-2 border-amber-300 border-t-transparent" />
                                {label}…
                              </span>
                            );
                          }
                          return (
                            <span
                              key={i}
                              className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium ${
                                t.success
                                  ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                                  : "border-red-200 bg-red-50 text-red-700"
                              }`}
                            >
                              {t.success ? `✓ ${label}` : `⚠ ${label} failed`}
                            </span>
                          );
                        })}
                      </div>
                    )}

                    {/* Bubble */}
                    <div
                      className={`whitespace-pre-wrap break-words text-sm leading-relaxed shadow-sm ${
                        isBot
                          ? "rounded-2xl rounded-tl-md border border-slate-200/80 bg-white px-4 py-2.5 text-slate-800"
                          : "brand-grad rounded-2xl rounded-tr-md px-4 py-2.5 text-white shadow-indigo-500/20"
                      }`}
                    >
                      {isLive && !m.text ? (
                        <span className="flex items-center gap-1 py-1">
                          <span className="typing-dot" />
                          <span className="typing-dot" style={{ animationDelay: "0.15s" }} />
                          <span className="typing-dot" style={{ animationDelay: "0.3s" }} />
                        </span>
                      ) : (
                        <>
                          {m.text}
                          {isLive && m.text ? <span className="caret" /> : null}
                        </>
                      )}
                    </div>

                    {/* Sources footer */}
                    {isBot && usedSources.length > 0 && (
                      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                        <span className="text-[11px] text-slate-400">📄 Grounded in</span>
                        {usedSources.map((src) => (
                          <span
                            key={src}
                            className="rounded-md border border-slate-200 bg-slate-50 px-1.5 py-0.5 text-[11px] font-medium text-slate-500"
                          >
                            {prettySource(src)}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* ---------------- COMPOSER ---------------- */}
        <div className="px-4 pb-5 pt-1 sm:px-6">
          <div className="mx-auto max-w-3xl">
            {awaitingConfirm && !sending && (
              <div className="mb-3 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-amber-200 bg-amber-50/90 px-4 py-3 animate-message-in">
                <div className="flex items-center gap-2 text-sm text-amber-900">
                  <span className="text-base">🛡️</span>
                  <span className="font-medium">Confirm this action?</span>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleSend("Yes, go ahead.")}
                    className="rounded-lg bg-emerald-600 px-4 py-1.5 text-sm font-semibold text-white shadow-sm transition hover:bg-emerald-700 active:scale-95"
                  >
                    Yes, go ahead
                  </button>
                  <button
                    onClick={() => handleSend("No, please don't.")}
                    className="rounded-lg border border-slate-300 bg-white px-4 py-1.5 text-sm font-medium text-slate-600 transition hover:bg-slate-50 active:scale-95"
                  >
                    No
                  </button>
                </div>
              </div>
            )}

            <div
              className={`relative rounded-2xl border bg-white shadow-lg transition ${
                composerFocused
                  ? "border-indigo-400 ring-4 ring-indigo-100"
                  : "border-slate-200"
              }`}
            >
              <textarea
                ref={taRef}
                value={input}
                onChange={handleInputChange}
                onKeyDown={onKeyDown}
                onFocus={() => setComposerFocused(true)}
                onBlur={() => setComposerFocused(false)}
                rows={1}
                disabled={awaitingConfirm}
                placeholder={
                  awaitingConfirm
                    ? "Use the Yes / No buttons above to confirm…"
                    : "Message the Care Agent…"
                }
                className="max-h-44 w-full resize-none rounded-2xl bg-transparent px-4 py-3.5 pr-14 text-sm leading-relaxed text-slate-800 outline-none placeholder:text-slate-400 disabled:opacity-60"
              />
              <button
                onClick={() => handleSend()}
                disabled={sending || !input.trim() || awaitingConfirm}
                aria-label="Send message"
                className="brand-grad absolute bottom-2.5 right-2.5 grid h-9 w-9 place-items-center rounded-xl text-white shadow-md transition hover:scale-105 active:scale-95 disabled:opacity-40 disabled:hover:scale-100"
              >
                {sending ? (
                  <span className="spinner h-4 w-4 rounded-full border-2 border-white/40 border-t-white" />
                ) : (
                  <SendIcon className="h-4 w-4" />
                )}
              </button>
            </div>
            <p className="mt-2 text-center text-[11px] text-slate-400">
              Press <span className="font-medium text-slate-500">Enter</span> to send ·
              {" "}
              <span className="font-medium text-slate-500">Shift + Enter</span> for a new line
            </p>
          </div>
        </div>
      </main>

      {/* ---------------- "WHAT WE KNOW ABOUT YOU" PANEL ---------------- */}
      <aside className="scroll-thin hidden w-80 shrink-0 flex-col gap-4 overflow-y-auto border-l border-slate-200/70 bg-white/60 p-5 lg:flex">
        <div className="flex items-center gap-2">
          <span className="text-base">🪪</span>
          <h2 className="text-sm font-semibold text-slate-700">What we know about you</h2>
        </div>

        {!customerInfo || !customerInfo.snapshot.known ? (
          <p className="text-xs text-slate-400">No customer record.</p>
        ) : (
          <>
            {/* Profile card */}
            <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="brand-grad grid h-11 w-11 place-items-center rounded-full text-sm font-bold text-white shadow-md shadow-indigo-500/25">
                  {initialsOf(customerInfo.snapshot.name)}
                </div>
                <div className="min-w-0">
                  <div className="truncate text-sm font-semibold text-slate-800">
                    {customerInfo.snapshot.name}
                  </div>
                  <div className="flex items-center gap-1.5 text-[11px] text-slate-500">
                    {customerInfo.snapshot.tier && (
                      <span className="rounded-full bg-indigo-50 px-1.5 py-0.5 font-medium capitalize text-indigo-600">
                        {customerInfo.snapshot.tier}
                      </span>
                    )}
                    <span className="truncate font-mono">{customerInfo.snapshot.customer_id}</span>
                  </div>
                </div>
              </div>
              {customerInfo.snapshot.address && (
                <div className="mt-3 flex items-start gap-1.5 rounded-lg bg-slate-50 px-2.5 py-2 text-xs text-slate-600">
                  <span>📍</span>
                  <span>{customerInfo.snapshot.address}</span>
                </div>
              )}
            </div>

            {/* Orders */}
            <div>
              <div className="mb-2 flex items-center justify-between">
                <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                  Orders
                </div>
                <span className="rounded-full bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium text-slate-500">
                  {customerInfo.snapshot.orders?.length ?? 0}
                </span>
              </div>
              <div className="flex flex-col gap-2">
                {customerInfo.snapshot.orders?.map((o) => (
                  <div
                    key={o.order_id}
                    className="lift rounded-xl border border-slate-200 bg-white p-3 shadow-sm"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-slate-700">#{o.order_id}</span>
                      <span
                        className={`rounded-full border px-2 py-0.5 text-[10px] font-medium capitalize ${orderStatusStyle(o.status)}`}
                      >
                        {o.status}
                      </span>
                    </div>
                    <div className="mt-1 text-xs text-slate-500">{o.items}</div>
                    <div className="mt-0.5 text-xs font-semibold text-slate-700">₹{o.total}</div>
                    {o.duplicate_charge ? (
                      <div className="mt-1.5 flex items-center gap-1 rounded-md bg-red-50 px-2 py-1 text-[11px] font-medium text-red-600">
                        ⚠ Duplicate charge ₹{o.duplicate_charge}
                      </div>
                    ) : null}
                  </div>
                ))}
              </div>
            </div>

            {/* Memory */}
            {customerInfo.memory &&
              (customerInfo.memory.summary ||
                customerInfo.memory.open_items.length > 0 ||
                customerInfo.memory.preferences.length > 0) && (
                <div>
                  <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                    🧠 Memory
                  </div>
                  <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
                    {customerInfo.memory.summary && (
                      <p className="text-xs leading-relaxed text-slate-600">
                        {customerInfo.memory.summary}
                      </p>
                    )}
                    {customerInfo.memory.open_items.length > 0 && (
                      <div className="mt-2.5">
                        <div className="mb-1 text-[11px] font-medium text-slate-500">Open items</div>
                        <ul className="flex flex-col gap-1">
                          {customerInfo.memory.open_items.map((it, i) => (
                            <li key={i} className="flex items-start gap-1.5 text-xs text-slate-600">
                              <span className="mt-0.5 text-amber-500">•</span>
                              {it}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {customerInfo.memory.preferences.length > 0 && (
                      <div className="mt-2.5 flex flex-wrap gap-1">
                        {customerInfo.memory.preferences.map((p, i) => (
                          <span
                            key={i}
                            className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] text-slate-600"
                          >
                            {p}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}
          </>
        )}
      </aside>
    </div>
  );
}
