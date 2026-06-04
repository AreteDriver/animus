import { useEffect, useState } from "react";
import { ChatView } from "./views/Chat";
import { StatusView } from "./views/Status";
import { PersonasView } from "./views/Personas";
import { CaptureView } from "./views/Capture";
import { LoginView } from "./views/Login";
import { getToken } from "./auth";
import "./App.css";

type View = "chat" | "capture" | "status" | "personas";

/** Read text shared into the app via the manifest share_target (?title/text/url). */
function readSharedText(): string {
  const params = new URLSearchParams(window.location.search);
  const parts = [params.get("title"), params.get("text"), params.get("url")].filter(
    (p): p is string => !!p,
  );
  return parts.join("\n");
}

export function App() {
  const [sharedText] = useState(readSharedText);
  const [view, setView] = useState<View>(() => (sharedText ? "capture" : "chat"));
  const [authed, setAuthed] = useState<boolean>(() => getToken() !== null);

  useEffect(() => {
    // The API client clears the token and fires this event on a 401.
    const onUnauthorized = () => setAuthed(false);
    window.addEventListener("animus:unauthorized", onUnauthorized);
    return () => window.removeEventListener("animus:unauthorized", onUnauthorized);
  }, []);

  if (!authed) {
    return (
      <div className="app">
        <main className="app-main">
          <LoginView onAuthed={() => setAuthed(true)} />
        </main>
      </div>
    );
  }

  return (
    <div className="app">
      <main className="app-main">
        {view === "chat" && <ChatView />}
        {view === "capture" && <CaptureView initialText={sharedText} />}
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
          icon="📝"
          label="Capture"
          active={view === "capture"}
          onClick={() => setView("capture")}
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
