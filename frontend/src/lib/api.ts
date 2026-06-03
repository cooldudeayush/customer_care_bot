/**
 * Thin API client for the Customer Care Bot backend.
 *
 * Phase 0: a simple non-streaming POST /chat that returns the placeholder reply.
 * Phase 1 will add an SSE streaming variant (the backend already exposes a
 * streaming generator) — this module is the single place that knows the wire
 * format, so the UI never hardcodes the backend URL.
 */

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

export interface ChatRequest {
  message: string;
  session_id?: string | null;
  customer_id?: string | null;
}

export interface ChatResponse {
  reply: string;
  session_id: string | null;
  placeholder: boolean;
}

/** Send one chat turn and get the bot's reply (non-streaming, Phase 0). */
export async function sendChat(req: ChatRequest): Promise<ChatResponse> {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Backend error ${res.status}: ${text || res.statusText}`);
  }
  return (await res.json()) as ChatResponse;
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
