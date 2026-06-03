/**
 * API client for the Customer Care Bot backend.
 *
 * Phase 1: streaming chat over Server-Sent Events (read via a fetch
 * ReadableStream, since /chat is a POST), plus the session endpoints that drive
 * the ChatGPT-style sidebar.
 */

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

export interface ChatStreamRequest {
  message: string;
  session_id: string;
  customer_id: string;
}

export interface DoneEvent {
  type: "done";
  session_id: string;
  title: string;
}

export interface Source {
  source: string;
  heading: string;
}

export interface StreamHandlers {
  onToken: (text: string) => void;
  onDone: (e: DoneEvent) => void;
  onError: (message: string) => void;
  onSources?: (sources: Source[]) => void;
}

export interface SessionSummary {
  session_id: string;
  customer_id: string | null;
  title: string;
  started_at: string;
  updated_at: string;
  message_count: number;
}

export interface ApiMessage {
  role: "user" | "bot";
  content: string;
  created_at: string | null;
}

export interface SessionDetail {
  session_id: string;
  title: string;
  messages: ApiMessage[];
}

/**
 * Send one chat turn and stream the reply. Resolves when the stream completes.
 * Parses SSE frames (`data: {json}\n\n`) from the fetch body reader.
 */
export async function streamChat(
  req: ChatStreamRequest,
  handlers: StreamHandlers,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
  } catch {
    handlers.onError(
      "Couldn't reach the backend. Is it running on " + API_URL + "?",
    );
    return;
  }

  if (!res.ok || !res.body) {
    handlers.onError(`Backend error ${res.status}: ${res.statusText}`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let doneFired = false; // guard against duplicate 'done' frames

  const handleFrame = (frame: string) => {
    const line = frame.trim();
    if (!line.startsWith("data:")) return;
    const payload = line.slice(line.indexOf(":") + 1).trim();
    if (!payload) return;
    let data: {
      type?: string;
      content?: string;
      message?: string;
      session_id?: string;
      title?: string;
      sources?: Source[];
    };
    try {
      data = JSON.parse(payload);
    } catch {
      return;
    }
    if (data.type === "token" && data.content) {
      handlers.onToken(data.content);
    } else if (data.type === "sources" && data.sources) {
      handlers.onSources?.(data.sources);
    } else if (data.type === "done") {
      if (doneFired) return;
      doneFired = true;
      handlers.onDone({
        type: "done",
        session_id: data.session_id ?? req.session_id,
        title: data.title ?? "New chat",
      });
    } else if (data.type === "error") {
      handlers.onError(data.message ?? "Unknown error");
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? ""; // keep the trailing partial frame
    for (const frame of frames) handleFrame(frame);
  }

  // Flush any bytes the decoder is still holding (e.g. a multi-byte UTF-8 char
  // split across the final packet), then process any remaining complete frame.
  buffer += decoder.decode();
  if (buffer.trim()) handleFrame(buffer);
}

/** List a customer's chat sessions for the sidebar (most recent first). */
export async function listSessions(customerId: string): Promise<SessionSummary[]> {
  try {
    const res = await fetch(
      `${API_URL}/sessions?customer_id=${encodeURIComponent(customerId)}`,
      { cache: "no-store" },
    );
    if (!res.ok) return [];
    return (await res.json()) as SessionSummary[];
  } catch {
    return [];
  }
}

/** Load the full transcript of one session (when reopening a chat). */
export async function getSession(sessionId: string): Promise<SessionDetail | null> {
  try {
    const res = await fetch(`${API_URL}/sessions/${encodeURIComponent(sessionId)}`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return (await res.json()) as SessionDetail;
  } catch {
    return null;
  }
}

/** Quick liveness check used by the UI to show backend status. */
export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
    return res.ok;
  } catch {
    return false;
  }
}
