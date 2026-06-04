import { useState } from "react";
import { captureNote } from "../api";
import "./Capture.css";

export function CaptureView({ initialText = "" }: { initialText?: string }) {
  const [text, setText] = useState(initialText);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSave() {
    const trimmed = text.trim();
    if (!trimmed || saving) return;
    setSaving(true);
    setError(null);
    try {
      await captureNote(trimmed);
      setText("");
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch {
      setError("Could not save. Try again.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="capture">
      <h1 className="capture-title">Quick Capture</h1>
      <p className="capture-hint">Jot a thought — it's stored in memory.</p>
      <textarea
        className="capture-input"
        placeholder="What's on your mind?"
        value={text}
        autoFocus
        onChange={(e) => setText(e.target.value)}
      />
      {error && <p className="capture-error">{error}</p>}
      {saved && <p className="capture-saved">Saved ✓</p>}
      <button
        className="capture-button"
        onClick={handleSave}
        disabled={saving || !text.trim()}
      >
        {saving ? "Saving..." : "Save"}
      </button>
    </div>
  );
}
