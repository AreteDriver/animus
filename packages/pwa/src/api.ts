/** API client for Animus Bootstrap dashboard backend. */

import { AuthError, clearToken, getToken } from "./auth";

const BASE = "/api";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...((options?.headers as Record<string, string>) ?? {}),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${BASE}${path}`, { ...options, headers });
  if (res.status === 401) {
    clearToken();
    window.dispatchEvent(new Event("animus:unauthorized"));
    throw new AuthError();
  }
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
  role?: string;
  metadata: Record<string, unknown>;
};

export type OnWSMessage = (msg: WSMessage) => void;

export function connectChat(onMessage: OnWSMessage): {
  send: (text: string) => void;
  close: () => void;
  getState: () => number;
  getPendingCount: () => number;
} {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const token = getToken();
  const query = token ? `?token=${encodeURIComponent(token)}` : "";
  const ws = new WebSocket(`${proto}//${location.host}/ws/chat${query}`);
  const pending: string[] = [];

  ws.onopen = () => {
    while (pending.length > 0) {
      const text = pending.shift()!;
      ws.send(JSON.stringify({ text, sender_id: "pwa-user", sender_name: "User" }));
    }
  };

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
      } else {
        pending.push(text);
      }
    },
    close: () => ws.close(),
    getState: () => ws.readyState,
    getPendingCount: () => pending.length,
  };
}

// Conversation history (persisted server-side)
export async function getHistory(limit = 50): Promise<WSMessage[]> {
  return request(`/conversations/history?limit=${limit}`);
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

// Quick capture
export async function captureNote(text: string): Promise<{ ok: boolean; message: string }> {
  return request("/capture", {
    method: "POST",
    body: JSON.stringify({ text }),
  });
}

// Web Push
export async function getVapidPublicKey(): Promise<string> {
  const res = await request<{ publicKey: string }>("/push/vapid-public-key");
  return res.publicKey;
}

export async function subscribePush(subscription: PushSubscriptionJSON): Promise<void> {
  await request("/push/subscribe", {
    method: "POST",
    body: JSON.stringify({ subscription }),
  });
}

export async function unsubscribePush(endpoint: string): Promise<void> {
  await request("/push/unsubscribe", {
    method: "POST",
    body: JSON.stringify({ endpoint }),
  });
}
