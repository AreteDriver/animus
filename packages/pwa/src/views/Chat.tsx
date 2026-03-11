import { useState, useRef, useEffect, useCallback } from "react";
import { connectChat, type WSMessage } from "../api";
import "./Chat.css";

interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  timestamp: Date;
  pending?: boolean;
}

export function ChatView() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [connected, setConnected] = useState(false);
  const [sending, setSending] = useState(false);
  const [pendingCount, setPendingCount] = useState(0);
  const bottomRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<ReturnType<typeof connectChat> | null>(null);

  const handleIncoming = useCallback((msg: WSMessage) => {
    setSending(false);
    setMessages((prev) => {
      // Clear pending flags on all user messages (they've been flushed)
      const cleared = prev.map((m) =>
        m.pending ? { ...m, pending: false } : m,
      );
      return [
        ...cleared,
        {
          id: msg.id,
          role: "assistant" as const,
          text: msg.text,
          timestamp: new Date(msg.timestamp),
        },
      ];
    });
    setPendingCount(0);
  }, []);

  useEffect(() => {
    let reconnectTimer: ReturnType<typeof setTimeout>;

    function connect() {
      const ws = connectChat(handleIncoming);
      wsRef.current = ws;

      // Poll connection state briefly to detect open/close
      const check = setInterval(() => {
        const state = ws.getState();
        if (state === WebSocket.OPEN) {
          setConnected(true);
          setPendingCount(ws.getPendingCount());
          clearInterval(check);
        } else if (state === WebSocket.CLOSED) {
          setConnected(false);
          clearInterval(check);
          reconnectTimer = setTimeout(connect, 3000);
        }
      }, 200);
    }

    connect();

    return () => {
      clearTimeout(reconnectTimer);
      wsRef.current?.close();
    };
  }, [handleIncoming]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function handleSend() {
    const text = input.trim();
    if (!text || sending || !wsRef.current) return;

    const isOffline = !connected;
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      text,
      timestamp: new Date(),
      pending: isOffline,
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    if (!isOffline) setSending(true);

    wsRef.current.send(text);
    if (isOffline) setPendingCount(wsRef.current.getPendingCount());
  }

  return (
    <div className="chat">
      <h1 className="chat-title">
        Animus
        <span className={`chat-status ${connected ? "chat-status--ok" : "chat-status--off"}`}>
          {connected ? "connected" : "offline"}
        </span>
        {pendingCount > 0 && (
          <span className="chat-status chat-status--off">
            {pendingCount} pending
          </span>
        )}
      </h1>
      <div className="chat-messages">
        {messages.length === 0 && (
          <p className="chat-empty">Start a conversation.</p>
        )}
        {messages.map((msg) => (
          <div key={msg.id} className={`chat-bubble chat-bubble--${msg.role}${msg.pending ? " chat-msg--pending" : ""}`}>
            {msg.text}
          </div>
        ))}
        {sending && (
          <div className="chat-bubble chat-bubble--assistant chat-bubble--thinking">
            Thinking...
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <div className="chat-input-row">
        <input
          className="chat-input"
          type="text"
          placeholder="Message Animus..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          disabled={sending}
        />
        <button
          className="chat-send"
          onClick={handleSend}
          disabled={sending || !input.trim()}
        >
          Send
        </button>
      </div>
    </div>
  );
}
