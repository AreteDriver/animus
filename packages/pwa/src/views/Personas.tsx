import { useState, useEffect } from "react";
import { listPersonas, createPersona, deletePersona, type Persona } from "../api";
import "./Personas.css";

const TONE_OPTIONS = ["professional", "casual", "technical", "friendly"];

export function PersonasView() {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [tone, setTone] = useState(TONE_OPTIONS[0]);
  const [creating, setCreating] = useState(false);

  async function load() {
    try {
      setLoading(true);
      const data = await listPersonas();
      setPersonas(data);
      setError(null);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    try {
      setCreating(true);
      setError(null);
      await createPersona({ name: name.trim(), description: description.trim(), tone });
      setName("");
      setDescription("");
      setTone(TONE_OPTIONS[0]);
      await load();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Create failed");
    } finally {
      setCreating(false);
    }
  }

  async function handleDelete(id: string) {
    try {
      await deletePersona(id);
      setPersonas((prev) => prev.filter((p) => p.id !== id));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Delete failed");
    }
  }

  return (
    <div className="personas">
      <h1 className="personas-title">Personas</h1>

      <form className="persona-form" onSubmit={handleCreate}>
        <input
          className="persona-input"
          type="text"
          placeholder="Name (required)"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
        />
        <input
          className="persona-input"
          type="text"
          placeholder="Description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
        />
        <select
          className="persona-select"
          value={tone}
          onChange={(e) => setTone(e.target.value)}
        >
          {TONE_OPTIONS.map((t) => (
            <option key={t} value={t}>
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </option>
          ))}
        </select>
        <button className="persona-submit" type="submit" disabled={creating || !name.trim()}>
          {creating ? "Creating..." : "Create"}
        </button>
      </form>

      {error && <p className="personas-error">{error}</p>}
      {loading && <p className="personas-loading">Loading...</p>}

      {!loading && personas.length === 0 && (
        <p className="personas-empty">
          No personas configured. Create one above or use the Bootstrap CLI.
        </p>
      )}

      <div className="personas-list">
        {personas.map((p) => (
          <div key={p.id} className="persona-card">
            <div className="persona-header">
              <span className="persona-name">{p.name}</span>
              {p.is_default && (
                <span className="persona-badge">Default</span>
              )}
            </div>
            <p className="persona-desc">{p.description}</p>
            <div className="persona-meta">
              <span className="persona-tone">{p.tone}</span>
              {p.knowledge_domains.length > 0 && (
                <span className="persona-domains">
                  {p.knowledge_domains.join(", ")}
                </span>
              )}
            </div>
            <div className="persona-actions">
              <button
                className="persona-delete"
                onClick={() => handleDelete(p.id)}
              >
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
