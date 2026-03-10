import { useState, useEffect } from "react";
import { getHealth } from "../api";
import "./Status.css";

export function StatusView() {
  const [health, setHealth] = useState<{ status: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((err: unknown) =>
        setError(err instanceof Error ? err.message : "Connection failed"),
      );
  }, []);

  return (
    <div className="status">
      <h1 className="status-title">System Status</h1>

      <div className="status-card">
        <div className="status-row">
          <span className="status-label">Backend</span>
          {error ? (
            <span className="status-badge status-badge--error">Offline</span>
          ) : health ? (
            <span className="status-badge status-badge--ok">Online</span>
          ) : (
            <span className="status-badge status-badge--loading">Checking...</span>
          )}
        </div>

        <div className="status-row">
          <span className="status-label">Dashboard</span>
          <span className="status-value">localhost:7700</span>
        </div>

        <div className="status-row">
          <span className="status-label">PWA Version</span>
          <span className="status-value">0.1.0</span>
        </div>
      </div>

      {error && (
        <p className="status-error">
          Cannot reach backend: {error}. Make sure Bootstrap dashboard is running.
        </p>
      )}
    </div>
  );
}
