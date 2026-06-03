"use client";

/**
 * Customer Care Bot — main chat page (Phase 1).
 *
 * Streams replies token-by-token from the backend (SSE over fetch) and wires the
 * ChatGPT-style sidebar to real persisted sessions:
 *   - on load: GET /sessions populates the sidebar
 *   - send:    POST /chat streams the reply; the session is persisted + titled
 *   - click a past chat: GET /sessions/{id} reopens its transcript
 *   - "+ New chat": starts a fresh session id
 *
 * Customer auth isn't built yet, so all sessions belong to a fixed demo customer
 * (Phase 6 makes this real).
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  checkHealth,
  getSession,
  listSessions,
  streamChat,
  type SessionSummary,
  type Source,
  type ToolEvent,
  type EmotionEvent,
  type HandoffEvent,
} from "@/lib/api";

const CUSTOMER_ID = "cust_demo";

// Emoji + styling for the detected-emotion badge (neutral is hidden).
const EMOTION_BADGE: Record<string, { emoji: string; cls: string }> = {
  angry: { emoji: "😠", cls: "border-red-300 bg-red-50 text-red-700" },
  frustrated: { emoji: "😟", cls: "border-orange-300 bg-orange-50 text-orange-700" },
  confused: { emoji: "😕", cls: "border-amber-300 bg-amber-50 text-amber-700" },
  happy: { emoji: "😊", cls: "border-green-300 bg-green-50 text-green-700" },
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

export default function ChatPage() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeId, setActiveId] = useState<string>(() => uuid());
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [backendUp, setBackendUp] = useState<boolean | null>(null);
  const [awaitingConfirm, setAwaitingConfirm] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const refreshSessions = useCallback(async () => {
    setSessions(await listSessions(CUSTOMER_ID));
  }, []);

  // Initial load: backend health + existing sessions.
  useEffect(() => {
    checkHealth().then(setBackendUp);
    refreshSessions();
  }, [refreshSessions]);

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

  function handleNewChat() {
    if (sending) return;
    setActiveId(uuid());
    setMessages([]);
    setInput("");
    setAwaitingConfirm(false);
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
      { message: text, session_id: activeId, customer_id: CUSTOMER_ID },
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
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="flex h-screen w-full bg-slate-50 text-slate-900">
      {/* ---------------- SIDEBAR (ChatGPT-style) ---------------- */}
      <aside className="flex w-72 shrink-0 flex-col border-r border-slate-200 bg-white">
        <div className="p-3">
          <button
            onClick={handleNewChat}
            disabled={sending}
            className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-medium transition hover:bg-slate-100 disabled:opacity-40"
          >
            + New chat
          </button>
        </div>
        <div className="px-3 pb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
          Chats
        </div>
        <nav className="flex-1 overflow-y-auto px-2 pb-3">
          {sidebarItems.length === 0 && (
            <p className="px-3 py-2 text-sm text-slate-400">No chats yet.</p>
          )}
          {sidebarItems.map((s) => (
            <button
              key={s.session_id}
              onClick={() => handleSelect(s.session_id)}
              className={`mb-1 block w-full truncate rounded-lg px-3 py-2 text-left text-sm transition ${
                s.session_id === activeId
                  ? "bg-slate-200 font-medium"
                  : "text-slate-600 hover:bg-slate-100"
              }`}
              title={s.title}
            >
              {s.title || "New chat"}
            </button>
          ))}
        </nav>
        <div className="border-t border-slate-200 p-3 text-xs text-slate-500">
          <div className="flex items-center gap-2">
            <span
              className={`inline-block h-2 w-2 rounded-full ${
                backendUp == null
                  ? "bg-slate-300"
                  : backendUp
                    ? "bg-green-500"
                    : "bg-red-500"
              }`}
            />
            {backendUp == null
              ? "Checking backend…"
              : backendUp
                ? "Backend online"
                : "Backend offline"}
          </div>
        </div>
      </aside>

      {/* ---------------- MAIN CHAT AREA ---------------- */}
      <main className="flex flex-1 flex-col">
        <header className="border-b border-slate-200 bg-white px-6 py-4">
          <h1 className="text-lg font-semibold">Customer Care Bot</h1>
          <p className="text-xs text-slate-500">
            Agentic support — resolves problems by taking real actions.
          </p>
        </header>

        <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-6">
          <div className="mx-auto flex max-w-2xl flex-col gap-4">
            {loadingHistory && (
              <p className="mt-10 text-center text-sm text-slate-400">
                Loading conversation…
              </p>
            )}
            {!loadingHistory && messages.length === 0 && (
              <div className="mt-20 text-center text-slate-400">
                <p className="text-sm">
                  Start a conversation. Try: “What is your refund policy?”
                </p>
              </div>
            )}
            {messages.map((m) => {
              const usedSources = m.sources
                ? Array.from(new Set(m.sources.map((s) => s.source)))
                : [];
              return (
                <div
                  key={m.id}
                  className={`flex flex-col ${
                    m.role === "user" ? "items-end" : "items-start"
                  }`}
                >
                  {m.role === "bot" &&
                    m.emotion &&
                    m.emotion.state !== "neutral" &&
                    EMOTION_BADGE[m.emotion.state] && (
                      <div
                        className={`mb-1 inline-flex max-w-[80%] items-center self-start rounded-full border px-2 py-0.5 text-xs ${EMOTION_BADGE[m.emotion.state].cls}`}
                      >
                        {EMOTION_BADGE[m.emotion.state].emoji} sensed:{" "}
                        {m.emotion.state}
                        {m.emotion.intensity >= 4 ? " (high)" : ""}
                      </div>
                    )}
                  {m.role === "bot" && m.handoff && (
                    <div className="mb-1 max-w-[80%] rounded-xl border border-indigo-200 bg-indigo-50 px-3 py-2 text-xs text-indigo-900">
                      <div className="font-semibold">
                        🤝 Handed off to a specialist (ticket #{m.handoff.packet_id})
                      </div>
                      {m.handoff.issue && (
                        <div className="mt-1">
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
                  {m.role === "bot" && m.tools && m.tools.length > 0 && (
                    <div className="mb-1 flex max-w-[80%] flex-wrap gap-1 px-1">
                      {m.tools.map((t, i) => (
                        <span
                          key={i}
                          className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs ${
                            t.status === "running"
                              ? "border-amber-300 bg-amber-50 text-amber-700"
                              : t.success
                                ? "border-green-300 bg-green-50 text-green-700"
                                : "border-red-300 bg-red-50 text-red-700"
                          }`}
                        >
                          {t.status === "running"
                            ? `⏳ ${TOOL_LABELS[t.name] ?? t.name}…`
                            : t.success
                              ? `✅ ${TOOL_LABELS[t.name] ?? t.name}`
                              : `⚠️ ${TOOL_LABELS[t.name] ?? t.name} failed`}
                        </span>
                      ))}
                    </div>
                  )}
                  <div
                    className={`max-w-[80%] whitespace-pre-wrap rounded-2xl px-4 py-2 text-sm ${
                      m.role === "user"
                        ? "bg-blue-600 text-white"
                        : "border border-slate-200 bg-white text-slate-800"
                    }`}
                  >
                    {m.text || (m.role === "bot" && sending ? "…" : "")}
                  </div>
                  {m.role === "bot" && usedSources.length > 0 && (
                    <div className="mt-1 max-w-[80%] px-1 text-xs text-slate-400">
                      📄 Grounded in: {usedSources.join(", ")}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        <div className="border-t border-slate-200 bg-white px-6 py-4">
          {awaitingConfirm && !sending && (
            <div className="mx-auto mb-3 flex max-w-2xl items-center gap-2">
              <span className="text-xs text-slate-500">Confirm this action?</span>
              <button
                onClick={() => handleSend("Yes, go ahead.")}
                className="rounded-lg bg-green-600 px-3 py-1.5 text-sm font-medium text-white transition hover:bg-green-700"
              >
                Yes, go ahead
              </button>
              <button
                onClick={() => handleSend("No, please don't.")}
                className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 transition hover:bg-slate-100"
              >
                No
              </button>
            </div>
          )}
          <div className="mx-auto flex max-w-2xl items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              rows={1}
              disabled={awaitingConfirm}
              placeholder={
                awaitingConfirm
                  ? "Use the Yes/No buttons above to confirm…"
                  : "Type your message… (Enter to send, Shift+Enter for newline)"
              }
              className="flex-1 resize-none rounded-xl border border-slate-300 px-4 py-2 text-sm focus:border-blue-500 focus:outline-none disabled:bg-slate-50"
            />
            <button
              onClick={() => handleSend()}
              disabled={sending || !input.trim()}
              className="rounded-xl bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-700 disabled:opacity-40"
            >
              {sending ? "…" : "Send"}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
