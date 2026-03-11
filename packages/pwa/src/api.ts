/** API client for Animus Bootstrap dashboard backend. */

const BASE = "/api";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// Health
export interface HealthResponse {
  status: string;
  version: string;
  components: {
    memory: boolean;
    tools: boolean;
    proactive: boolean;
    automations: boolean;
  };
}

export async function getHealth(): Promise<HealthResponse> {
  return request("/health");
}

// Chat (REST fallback)
export async function sendMessage(text: string): Promise<{ text: string }> {
  return request("/conversations/messages", {
    method: "POST",
    body: JSON.stringify({ text }),
  });
}

// WebSocket chat
export type WSMessage = {
  id: string;
  channel: string;
  text: string;
  timestamp: string;
  sender: string;
  metadata: Record<string, unknown>;
};

export type OnWSMessage = (msg: WSMessage) => void;

export function connectChat(onMessage: OnWSMessage): {
  send: (text: string) => void;
  close: () => void;
  getState: () => number;
} {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws/chat`);

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data) as WSMessage;
      onMessage(msg);
    } catch {
      // ignore malformed messages
    }
  };

  return {
    send: (text: string) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ text, sender_id: "pwa-user", sender_name: "User" }));
      }
    },
    close: () => ws.close(),
    getState: () => ws.readyState,
  };
}

// Personas
export interface Persona {
  id: string;
  name: string;
  description: string;
  tone: string;
  active: boolean;
  is_default: boolean;
  knowledge_domains: string[];
}

export async function listPersonas(): Promise<Persona[]> {
  return request("/personas");
}

export async function createPersona(data: Partial<Persona>): Promise<{ id: string }> {
  return request("/personas", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function deletePersona(id: string): Promise<void> {
  await request(`/personas/${id}`, { method: "DELETE" });
}

// Feedback
export async function submitFeedback(rating: number, messageText: string): Promise<void> {
  await request("/feedback", {
    method: "POST",
    body: JSON.stringify({ rating, message_text: messageText }),
  });
}
