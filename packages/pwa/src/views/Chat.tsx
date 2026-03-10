import { useState, useRef, useEffect } from "react";
import { sendMessage } from "../api";
import "./Chat.css";

interface Message {
  id: number;
  role: "user" | "assistant";
  text: string;
  timestamp: Date;
}

export function ChatView() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend() {
    const text = input.trim();
    if (!text || sending) return;

    const userMsg: Message = {
      id: Date.now(),
      role: "user",
      text,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setSending(true);

    try {
      const response = await sendMessage(text);
      const assistantMsg: Message = {
        id: Date.now() + 1,
        role: "assistant",
        text: response.text,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const errorMsg: Message = {
        id: Date.now() + 1,
        role: "assistant",
        text: `Error: ${err instanceof Error ? err.message : "Connection failed"}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="chat">
      <h1 className="chat-title">Animus</h1>
      <div className="chat-messages">
        {messages.length === 0 && (
          <p className="chat-empty">Start a conversation.</p>
        )}
        {messages.map((msg) => (
          <div key={msg.id} className={`chat-bubble chat-bubble--${msg.role}`}>
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
