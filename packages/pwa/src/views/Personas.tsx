import { useState, useEffect } from "react";
import { listPersonas, deletePersona, type Persona } from "../api";
import "./Personas.css";

export function PersonasView() {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

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

      {loading && <p className="personas-loading">Loading...</p>}
      {error && <p className="personas-error">{error}</p>}

      {!loading && personas.length === 0 && (
        <p className="personas-empty">
          No personas configured. Use the Bootstrap CLI or dashboard to create
          one.
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
