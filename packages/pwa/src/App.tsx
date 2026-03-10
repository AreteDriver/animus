import { useState } from "react";
import { ChatView } from "./views/Chat";
import { StatusView } from "./views/Status";
import { PersonasView } from "./views/Personas";
import "./App.css";

type View = "chat" | "status" | "personas";

export function App() {
  const [view, setView] = useState<View>("chat");

  return (
    <div className="app">
      <main className="app-main">
        {view === "chat" && <ChatView />}
        {view === "status" && <StatusView />}
        {view === "personas" && <PersonasView />}
      </main>

      <nav className="app-nav">
        <NavButton
          icon="💬"
          label="Chat"
          active={view === "chat"}
          onClick={() => setView("chat")}
        />
        <NavButton
          icon="📊"
          label="Status"
          active={view === "status"}
          onClick={() => setView("status")}
        />
        <NavButton
          icon="🎭"
          label="Personas"
          active={view === "personas"}
          onClick={() => setView("personas")}
        />
      </nav>
    </div>
  );
}

function NavButton({
  icon,
  label,
  active,
  onClick,
}: {
  icon: string;
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      className={`nav-btn ${active ? "nav-btn--active" : ""}`}
      onClick={onClick}
    >
      <span className="nav-btn-icon">{icon}</span>
      <span className="nav-btn-label">{label}</span>
    </button>
  );
}
