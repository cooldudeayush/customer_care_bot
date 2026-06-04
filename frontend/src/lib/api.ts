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
  owner: string;
}

export interface DoneEvent {
  type: "done";
  session_id: string;
  title: string;
  awaiting_confirmation?: boolean;
}

export interface Source {
  source: string;
  heading: string;
}

export interface ToolEvent {
  name: string;
  status: "running" | "done";
  success?: boolean;
}

export interface EmotionEvent {
  state: string;
  intensity: number;
}

export interface HandoffEvent {
  packet_id: number;
  issue: string;
  actions_taken: string[];
  suggested_next_step: string;
}

export interface StreamHandlers {
  onToken: (text: string) => void;
  onDone: (e: DoneEvent) => void;
  onError: (message: string) => void;
  onSources?: (sources: Source[]) => void;
  onTool?: (t: ToolEvent) => void;
  onEmotion?: (e: EmotionEvent) => void;
  onHandoff?: (h: HandoffEvent) => void;
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
      name?: string;
      status?: "running" | "done";
      success?: boolean;
      awaiting_confirmation?: boolean;
      state?: string;
      intensity?: number;
      packet_id?: number;
      issue?: string;
      actions_taken?: string[];
      suggested_next_step?: string;
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
    } else if (data.type === "tool" && data.name && data.status) {
      handlers.onTool?.({ name: data.name, status: data.status, success: data.success });
    } else if (data.type === "emotion" && data.state) {
      handlers.onEmotion?.({ state: data.state, intensity: data.intensity ?? 1 });
    } else if (data.type === "handoff" && data.packet_id != null) {
      handlers.onHandoff?.({
        packet_id: data.packet_id,
        issue: data.issue ?? "",
        actions_taken: data.actions_taken ?? [],
        suggested_next_step: data.suggested_next_step ?? "",
      });
    } else if (data.type === "done") {
      if (doneFired) return;
      doneFired = true;
      handlers.onDone({
        type: "done",
        session_id: data.session_id ?? req.session_id,
        title: data.title ?? "New chat",
        awaiting_confirmation: data.awaiting_confirmation,
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

/** List this browser's chat sessions for the sidebar (most recent first). */
export async function listSessions(owner: string): Promise<SessionSummary[]> {
  try {
    const res = await fetch(
      `${API_URL}/sessions?owner=${encodeURIComponent(owner)}`,
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

// ---- "What we know about you" panel (Phase 9) ----
export interface CustomerOrder {
  order_id: string;
  status: string;
  total: number;
  items: string;
  duplicate_charge: number | null;
}

export interface CustomerSnapshot {
  known: boolean;
  customer_id?: string;
  name?: string;
  tier?: string;
  address?: string;
  orders?: CustomerOrder[];
}

export interface CustomerMemoryView {
  summary: string | null;
  open_items: string[];
  preferences: string[];
  sentiment: string | null;
}

export interface CustomerInfo {
  snapshot: CustomerSnapshot;
  memory: CustomerMemoryView | null;
}

/** What the bot knows about a customer (live orders + long-term memory). */
export async function getCustomerInfo(customerId: string): Promise<CustomerInfo | null> {
  try {
    const res = await fetch(`${API_URL}/customer/${encodeURIComponent(customerId)}`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return (await res.json()) as CustomerInfo;
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
