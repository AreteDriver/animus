import { useState } from "react";
import { getHealth } from "../api";
import { setToken } from "../auth";
import "./Login.css";

export function LoginView({ onAuthed }: { onAuthed: () => void }) {
  const [token, setTokenInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);

  async function handleSubmit() {
    const trimmed = token.trim();
    if (!trimmed || checking) return;
    setChecking(true);
    setError(null);
    // Store first so the validation request carries the token.
    setToken(trimmed);
    try {
      await getHealth();
      onAuthed();
    } catch {
      setError("That token was rejected. Check it and try again.");
    } finally {
      setChecking(false);
    }
  }

  return (
    <div className="login">
      <h1 className="login-title">Animus</h1>
      <p className="login-hint">
        Paste the access token shown when the dashboard started.
      </p>
      <input
        className="login-input"
        type="password"
        placeholder="Access token"
        value={token}
        autoFocus
        onChange={(e) => setTokenInput(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
      />
      {error && <p className="login-error">{error}</p>}
      <button
        className="login-button"
        onClick={handleSubmit}
        disabled={checking || !token.trim()}
      >
        {checking ? "Checking..." : "Connect"}
      </button>
    </div>
  );
}
