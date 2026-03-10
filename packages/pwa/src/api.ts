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
export async function getHealth(): Promise<{ status: string }> {
  return request("/health");
}

// Chat
export async function sendMessage(text: string): Promise<{ text: string }> {
  return request("/conversations/messages", {
    method: "POST",
    body: JSON.stringify({ text }),
  });
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
