"use client";

/**
 * Customer Care Bot — main chat page (Phase 0 skeleton).
 *
 * Layout: ChatGPT-style left SIDEBAR (list of past chat sessions + "New chat")
 * and a main chat area (message bubbles, typing indicator, input box).
 *
 * Phase 0 keeps sessions in client state and calls the backend's placeholder
 * POST /chat. Phase 1 will: stream tokens (SSE), and persist/list sessions from
 * the backend's `conversations` table (GET /sessions) — the sidebar UI here is
 * already shaped for that swap.
 */

import { useEffect, useRef, useState } from "react";
import { sendChat, checkHealth } from "@/lib/api";

type Role = "user" | "bot";

interface Message {
  id: string;
  role: Role;
  text: string;
}

interface Session {
  id: string;
  title: string;
  messages: Message[];
}

let idCounter = 0;
const nextId = () => `${Date.now()}-${idCounter++}`;

function newSession(): Session {
  return { id: nextId(), title: "New chat", messages: [] };
}

export default function ChatPage() {
  const [sessions, setSessions] = useState<Session[]>(() => [newSession()]);
  const [activeId, setActiveId] = useState<string>(() => sessions[0].id);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [backendUp, setBackendUp] = useState<boolean | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const active = sessions.find((s) => s.id === activeId) ?? sessions[0];

  // Backend liveness badge.
  useEffect(() => {
    checkHealth().then(setBackendUp);
  }, []);

  // Auto-scroll to the newest message.
  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [active.messages.length, sending]);

  function patchActive(update: (s: Session) => Session) {
    setSessions((prev) => prev.map((s) => (s.id === activeId ? update(s) : s)));
  }

  function handleNewChat() {
    const s = newSession();
    setSessions((prev) => [s, ...prev]);
    setActiveId(s.id);
    setInput("");
  }

  async function handleSend() {
    const text = input.trim();
    if (!text || sending) return;

    const userMsg: Message = { id: nextId(), role: "user", text };
    // Title a fresh chat from its first message.
    patchActive((s) => ({
      ...s,
      title: s.messages.length === 0 ? text.slice(0, 40) : s.title,
      messages: [...s.messages, userMsg],
    }));
    setInput("");
    setSending(true);

    try {
      const res = await sendChat({ message: text, session_id: activeId });
      const botMsg: Message = { id: nextId(), role: "bot", text: res.reply };
      patchActive((s) => ({ ...s, messages: [...s.messages, botMsg] }));
    } catch (err) {
      const botMsg: Message = {
        id: nextId(),
        role: "bot",
        text:
          "⚠️ I couldn't reach the backend. Make sure it's running on " +
          "http://localhost:8000 (uvicorn). Details: " +
          (err instanceof Error ? err.message : String(err)),
      };
      patchActive((s) => ({ ...s, messages: [...s.messages, botMsg] }));
    } finally {
      setSending(false);
    }
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
            className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-medium transition hover:bg-slate-100"
          >
            + New chat
          </button>
        </div>
        <div className="px-3 pb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
          Chats
        </div>
        <nav className="flex-1 overflow-y-auto px-2 pb-3">
          {sessions.map((s) => (
            <button
              key={s.id}
              onClick={() => setActiveId(s.id)}
              className={`mb-1 block w-full truncate rounded-lg px-3 py-2 text-left text-sm transition ${
                s.id === activeId
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
            {active.messages.length === 0 && (
              <div className="mt-20 text-center text-slate-400">
                <p className="text-sm">
                  Start a conversation. Try: “What is your refund policy?”
                </p>
              </div>
            )}
            {active.messages.map((m) => (
              <div
                key={m.id}
                className={`flex ${
                  m.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[80%] whitespace-pre-wrap rounded-2xl px-4 py-2 text-sm ${
                    m.role === "user"
                      ? "bg-blue-600 text-white"
                      : "border border-slate-200 bg-white text-slate-800"
                  }`}
                >
                  {m.text}
                </div>
              </div>
            ))}
            {sending && (
              <div className="flex justify-start">
                <div className="flex items-center gap-1 rounded-2xl border border-slate-200 bg-white px-4 py-3">
                  <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.3s]" />
                  <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.15s]" />
                  <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400" />
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="border-t border-slate-200 bg-white px-6 py-4">
          <div className="mx-auto flex max-w-2xl items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              rows={1}
              placeholder="Type your message… (Enter to send, Shift+Enter for newline)"
              className="flex-1 resize-none rounded-xl border border-slate-300 px-4 py-2 text-sm focus:border-blue-500 focus:outline-none"
            />
            <button
              onClick={handleSend}
              disabled={sending || !input.trim()}
              className="rounded-xl bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-700 disabled:opacity-40"
            >
              Send
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
