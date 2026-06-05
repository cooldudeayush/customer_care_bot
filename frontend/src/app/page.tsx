"use client";

/**
 * Customer Care Bot — main chat page.
 *
 * Streams replies token-by-token from the backend (SSE over fetch) and wires the
 * sidebar to real persisted sessions:
 *   - on load: GET /sessions populates the sidebar
 *   - send:    POST /chat streams the reply; the session is persisted + titled
 *   - click a past chat: GET /sessions/{id} reopens its transcript
 *   - "+ New chat": starts a fresh session id
 *
 * Presentation: a warm, editorial "care desk" — slate sidebar, gold accents,
 * maroon highlights on white. All motion is CSS-only, so the richer look is free.
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

// Emoji + chip tone for the detected-emotion badge (neutral is hidden).
const EMOTION_BADGE: Record<string, { emoji: string; tone: string }> = {
  angry: { emoji: "😠", tone: "neg" },
  frustrated: { emoji: "😟", tone: "neg" },
  confused: { emoji: "😕", tone: "warn" },
  happy: { emoji: "😊", tone: "pos" },
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

// Map an order status onto a badge style.
function badgeClass(status: string): string {
  const s = status.toLowerCase();
  if (s.includes("deliver")) return "delivered";
  if (s.includes("cancel")) return "cancelled";
  if (s.includes("placed") || s.includes("process") || s.includes("pending")) return "placed";
  return "shipped";
}

/* ---------------- Inline icons (no icon lib — keeps the bundle lean) ---------------- */
// Headset: the universal "customer care" mark — warmer + clearer than a sparkle.
function HeadsetIcon({ className = "" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M3 12a9 9 0 0 1 18 0" />
      <path d="M21 16v1a3 3 0 0 1-3 3h-4" />
      <rect x="2.5" y="12" width="4" height="7" rx="1.6" />
      <rect x="17.5" y="12" width="4" height="7" rx="1.6" />
    </svg>
  );
}
function PlusIcon({ className = "" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2.4" strokeLinecap="round" aria-hidden>
      <path d="M12 5v14M5 12h14" />
    </svg>
  );
}
function SendIcon({ className = "" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <path d="M4 12 20 4l-4.5 16-4-7-7.5-1Z" />
    </svg>
  );
}
function ChevronIcon({ className = "" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden width="18" height="18">
      <path d="m9 6 6 6-6 6" />
    </svg>
  );
}
function PinIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
      <path d="M12 21s7-5.4 7-11a7 7 0 1 0-14 0c0 5.6 7 11 7 11Z" />
      <circle cx="12" cy="10" r="2.6" />
    </svg>
  );
}
function ListIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" aria-hidden>
      <path d="M4 7h16M4 12h16M4 17h10" />
    </svg>
  );
}
function BrainIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden>
      <path d="M9 3a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 6 0V6a3 3 0 0 0-3-3Z" />
      <path d="M15 3a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-6 0" />
    </svg>
  );
}
function WarnIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
      strokeLinejoin="round" aria-hidden>
      <path d="M12 3 2 20h20L12 3Z" />
      <path d="M12 10v4" strokeLinecap="round" />
      <circle cx="12" cy="17" r=".5" fill="currentColor" />
    </svg>
  );
}

/* ---------------- Lightweight, XSS-safe Markdown for bot replies ----------------
 * Claude returns Markdown (**bold**, *italic*, lists). We render it as real React
 * nodes — never raw HTML — so there's no injection risk and no extra dependency.
 * Handles: **bold**, __bold__, *italic*, _italic_, `code`, and "- " / "1." lists. */
function renderInline(text: string): React.ReactNode[] {
  const out: React.ReactNode[] = [];
  const re = /(\*\*([^*]+?)\*\*|__([^_]+?)__|\*([^*\n]+?)\*|_([^_\n]+?)_|`([^`]+?)`)/g;
  let last = 0;
  let k = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    if (m[2] != null || m[3] != null) out.push(<strong key={k++}>{m[2] ?? m[3]}</strong>);
    else if (m[4] != null || m[5] != null) out.push(<em key={k++}>{m[4] ?? m[5]}</em>);
    else if (m[6] != null) out.push(<code key={k++}>{m[6]}</code>);
    last = re.lastIndex;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

function Markdown({ text }: { text: string }) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const blocks: React.ReactNode[] = [];
  let para: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;
  let key = 0;

  const flushPara = () => {
    if (!para.length) return;
    const buf = para;
    blocks.push(
      <p key={`p${key++}`}>
        {buf.map((ln, j) => (
          <span key={j}>
            {j > 0 ? <br /> : null}
            {renderInline(ln)}
          </span>
        ))}
      </p>,
    );
    para = [];
  };
  const flushList = () => {
    if (!list) return;
    const cur = list;
    const items = cur.items.map((it, j) => <li key={j}>{renderInline(it)}</li>);
    blocks.push(
      cur.ordered ? <ol key={`l${key++}`}>{items}</ol> : <ul key={`l${key++}`}>{items}</ul>,
    );
    list = null;
  };

  for (const line of lines) {
    const lm = line.match(/^\s*([-*•]|\d+[.)])\s+(.*)$/);
    if (lm) {
      flushPara();
      const ordered = /\d/.test(lm[1]);
      if (!list || list.ordered !== ordered) {
        flushList();
        list = { ordered, items: [] };
      }
      list.items.push(lm[2]);
    } else if (line.trim() === "") {
      flushPara();
      flushList();
    } else {
      flushList();
      para.push(line);
    }
  }
  flushPara();
  flushList();
  return <>{blocks}</>;
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
    const probe = () =>
      checkHealth().then((up) => {
        if (!cancelled) setBackendUp(up);
      });
    probe();
    // Re-check on an interval so the badge recovers once Render's free tier
    // wakes from cold start (the first probe can run before the backend is up).
    const id = window.setInterval(probe, 25000);
    listSessions(owner).then((s) => {
      if (!cancelled) setSessions(s);
    });
    getCustomerInfo(CUSTOMER_ID).then((c) => {
      if (!cancelled) {
        setCustomerInfo(c);
        if (c) setBackendUp(true); // a real response means we're online
      }
    });
    return () => {
      cancelled = true;
      window.clearInterval(id);
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
      el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
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
        onToken: (chunk) => {
          setBackendUp(true); // a streaming reply proves the backend is online
          appendToBot(chunk);
        },
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
          setBackendUp(true);
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

  const statusKind = backendUp == null ? "idle" : backendUp ? "online" : "offline";
  const statusLabel =
    backendUp == null ? "Connecting…" : backendUp ? "Online" : "Offline";
  const pillClass = backendUp == null ? "idle" : backendUp ? "" : "off";
  // The currently-streaming bot message (last one while sending) gets the live
  // typing dots / blinking caret.
  const liveId = sending ? messages[messages.length - 1]?.id : null;
  const snap = customerInfo?.snapshot;
  const mem = customerInfo?.memory;

  return (
    <div className="app">
      {/* ============ LEFT SIDEBAR ============ */}
      <aside className="side">
        <div className="brand">
          <span className="logo">
            <HeadsetIcon />
          </span>
          <div>
            <div className="brand-name">Care Agent</div>
            <div className="brand-sub">Agentic support</div>
          </div>
        </div>

        <button className="new-chat" onClick={handleNewChat} disabled={sending}>
          <PlusIcon />
          New chat
        </button>

        <div className="rail-label">Recent chats</div>
        <nav className="rail scroll-dark">
          {sidebarItems.length === 0 && <div className="side-empty">No chats yet.</div>}
          {sidebarItems.map((s) => (
            <button
              key={s.session_id}
              className={`chat-item ${s.session_id === activeId ? "active" : ""}`}
              onClick={() => handleSelect(s.session_id)}
              title={s.title}
            >
              {s.title || "New chat"}
            </button>
          ))}
        </nav>

        <div className="side-foot">
          <span className={`dot ${statusKind}`} /> Backend {statusLabel}
        </div>
      </aside>

      {/* ============ MAIN ============ */}
      <main className="main">
        <header className="topbar">
          <div className="topbar-id">
            <span className="topbar-logo">
              <HeadsetIcon />
            </span>
            <div>
              <h1>Customer Care Bot</h1>
              <p>Resolves problems by taking real actions — not just telling you how.</p>
            </div>
          </div>
          <span className={`status-pill ${pillClass}`}>
            <span className="dot" /> {statusLabel}
          </span>
        </header>

        <section className="thread scroll-thin" ref={scrollRef}>
          {loadingHistory && (
            <div className="thread-hint">
              <span className="tdot" />
              <span className="tdot" style={{ animationDelay: "0.15s" }} />
              <span className="tdot" style={{ animationDelay: "0.3s" }} />
              <span style={{ marginLeft: 4 }}>Loading conversation…</span>
            </div>
          )}

          {/* Empty-state hero + quiet suggestion list */}
          {!loadingHistory && messages.length === 0 && (
            <div className="empty fade-up">
              <span className="hero">
                <HeadsetIcon />
              </span>
              <h2>
                Hi! I&apos;m your <span className="accent">Care Agent</span>
              </h2>
              <p>
                I don&apos;t just answer questions — I take real actions: refunds, order
                tracking, address changes &amp; more. Try one:
              </p>
              <div className="suggests">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s.text}
                    className="suggest"
                    onClick={() => handleSend(s.text)}
                    disabled={sending}
                  >
                    <span className="s-ico">{s.icon}</span>
                    <span>{s.text}</span>
                    <ChevronIcon className="s-arrow" />
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

            if (!isBot) {
              return (
                <div key={m.id} className="row user msg-in">
                  <span className="you-tag">You</span>
                  <div className="bubble-user">{m.text}</div>
                </div>
              );
            }

            const emo =
              m.emotion && m.emotion.state !== "neutral"
                ? EMOTION_BADGE[m.emotion.state]
                : undefined;

            return (
              <div key={m.id} className="row bot msg-in">
                <span className="avatar">
                  <HeadsetIcon />
                </span>
                <div className="msg-col">
                  {emo && (
                    <div className="meta-row">
                      <span className={`chip ${emo.tone}`}>
                        {emo.emoji} sensed: {m.emotion!.state}
                        {m.emotion!.intensity >= 4 ? " (high)" : ""}
                      </span>
                    </div>
                  )}

                  {m.handoff && (
                    <div className="handoff">
                      <div className="h-title">
                        🤝 Handed off to a specialist
                        <span className="h-ticket">ticket #{m.handoff.packet_id}</span>
                      </div>
                      {m.handoff.issue && (
                        <div className="h-row">
                          <b>Issue:</b> {m.handoff.issue}
                        </div>
                      )}
                      {m.handoff.actions_taken.length > 0 && (
                        <div className="h-row">
                          <b>Actions taken:</b> {m.handoff.actions_taken.join("; ")}
                        </div>
                      )}
                      {m.handoff.suggested_next_step && (
                        <div className="h-row">
                          <b>Next step:</b> {m.handoff.suggested_next_step}
                        </div>
                      )}
                    </div>
                  )}

                  {m.tools && m.tools.length > 0 && (
                    <div className="meta-row">
                      {m.tools.map((t, i) => {
                        const label = TOOL_LABELS[t.name] ?? t.name;
                        if (t.status === "running") {
                          return (
                            <span key={i} className="chip warn">
                              <span className="spin-g" />
                              {label}…
                            </span>
                          );
                        }
                        return (
                          <span key={i} className={`chip ${t.success ? "pos" : "neg"}`}>
                            {t.success ? `✓ ${label}` : `⚠ ${label} failed`}
                          </span>
                        );
                      })}
                    </div>
                  )}

                  <div className="bubble-bot">
                    {isLive ? (
                      !m.text ? (
                        <span style={{ display: "inline-flex", gap: 5, padding: "3px 0" }}>
                          <span className="tdot" />
                          <span className="tdot" style={{ animationDelay: "0.15s" }} />
                          <span className="tdot" style={{ animationDelay: "0.3s" }} />
                        </span>
                      ) : (
                        // While streaming, show raw text + caret (Markdown is
                        // applied once the reply is complete, below).
                        <>
                          <span style={{ whiteSpace: "pre-wrap" }}>{m.text}</span>
                          <span className="caret" />
                        </>
                      )
                    ) : (
                      <Markdown text={m.text} />
                    )}
                  </div>

                  {usedSources.length > 0 && (
                    <div className="sources">
                      📄 Grounded in
                      {usedSources.map((src) => (
                        <span key={src} className="source-pill">
                          {prettySource(src)}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </section>

        {/* ============ COMPOSER ============ */}
        <div className="composer-wrap">
          {awaitingConfirm && !sending && (
            <div className="confirm msg-in">
              <div className="c-label">
                <span>🛡️</span> Confirm this action?
              </div>
              <div className="c-btns">
                <button className="btn-yes" onClick={() => handleSend("Yes, go ahead.")}>
                  Yes, go ahead
                </button>
                <button className="btn-no" onClick={() => handleSend("No, please don't.")}>
                  No
                </button>
              </div>
            </div>
          )}

          <div className={`composer ${composerFocused ? "focus" : ""}`}>
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
            />
            <button
              className="send"
              onClick={() => handleSend()}
              disabled={sending || !input.trim() || awaitingConfirm}
              aria-label="Send message"
            >
              {sending ? <span className="spin-w" /> : <SendIcon />}
            </button>
          </div>
          <div className="composer-hint">
            Press <b>Enter</b> to send · <b>Shift + Enter</b> for a new line
          </div>
        </div>
      </main>

      {/* ============ RIGHT PANEL ============ */}
      <aside className="panel scroll-thin">
        {!snap || !snap.known ? (
          <div className="panel-empty">No customer record.</div>
        ) : (
          <>
            <div className="profile">
              <span className="pa">{initialsOf(snap.name)}</span>
              <div>
                <div className="pname">{snap.name}</div>
                <div className="pmeta">
                  {snap.tier && <span className="tier-pill">{snap.tier}</span>}
                  <span>{snap.customer_id}</span>
                </div>
              </div>
            </div>

            {snap.address && (
              <div className="loc">
                <PinIcon />
                {snap.address}
              </div>
            )}

            <div>
              <div className="section-head">
                <ListIcon />
                Orders
                <span className="count">{snap.orders?.length ?? 0}</span>
              </div>
              <div className="orders">
                {snap.orders?.map((o) => (
                  <div key={o.order_id} className="order">
                    <div className="order-top">
                      <span className="order-id">#{o.order_id}</span>
                      <span className={`badge ${badgeClass(o.status)}`}>{o.status}</span>
                    </div>
                    <div className="order-item">{o.items}</div>
                    <div className="order-price">₹{o.total}</div>
                    {o.duplicate_charge ? (
                      <div className="flag">
                        <WarnIcon />
                        Duplicate charge ₹{o.duplicate_charge}
                      </div>
                    ) : null}
                  </div>
                ))}
              </div>
            </div>

            {mem && (mem.summary || mem.open_items.length > 0 || mem.preferences.length > 0) && (
              <div>
                <div className="section-head">
                  <BrainIcon />
                  Memory
                </div>
                <div className="memory">
                  {mem.summary && <div>{mem.summary}</div>}
                  {mem.open_items.length > 0 && (
                    <>
                      <div className="open-label">Open items</div>
                      <ul>
                        {mem.open_items.map((it, i) => (
                          <li key={i}>{it}</li>
                        ))}
                      </ul>
                    </>
                  )}
                  {mem.preferences.length > 0 && (
                    <div className="prefs">
                      {mem.preferences.map((p, i) => (
                        <span key={i} className="pref">
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
