import { useState, useEffect, useCallback } from "react";
import { getHealth, type HealthResponse } from "../api";
import "./Status.css";

const REFRESH_INTERVAL_MS = 10_000;

const COMPONENT_LABELS: Record<string, string> = {
  memory: "Memory",
  tools: "Tools",
  proactive: "Proactive Engine",
  automations: "Automations",
};

export function StatusView() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  const refresh = useCallback(() => {
    getHealth()
      .then((data) => {
        setHealth(data);
        setError(null);
        setLastChecked(new Date());
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Connection failed");
        setHealth(null);
        setLastChecked(new Date());
      });
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, REFRESH_INTERVAL_MS);
    return () => clearInterval(id);
  }, [refresh]);

  return (
    <div className="status">
      <h1 className="status-title">System Status</h1>

      <div className="status-card">
        <div className="status-row">
          <span className="status-label">Backend</span>
          {error ? (
            <span className="status-badge status-badge--error">Offline</span>
          ) : health ? (
            <span className="status-badge status-badge--ok">
              {health.status === "ok" ? "Online" : "Degraded"}
            </span>
          ) : (
            <span className="status-badge status-badge--loading">
              Checking...
            </span>
          )}
        </div>

        {health && (
          <div className="status-row">
            <span className="status-label">Backend Version</span>
            <span className="status-value">{health.version}</span>
          </div>
        )}

        <div className="status-row">
          <span className="status-label">Dashboard</span>
          <span className="status-value">localhost:7700</span>
        </div>

        <div className="status-row">
          <span className="status-label">PWA Version</span>
          <span className="status-value">0.1.0</span>
        </div>

        {lastChecked && (
          <div className="status-row">
            <span className="status-label">Last Checked</span>
            <span className="status-value">
              {lastChecked.toLocaleTimeString()}
            </span>
          </div>
        )}
      </div>

      {health && (
        <div className="status-card status-components">
          <h2 className="status-section-title">Components</h2>
          {Object.entries(health.components).map(([key, ok]) => (
            <div key={key} className="status-row">
              <span className="status-label">
                {COMPONENT_LABELS[key] ?? key}
              </span>
              <span
                className={`status-badge ${ok ? "status-badge--ok" : "status-badge--error"}`}
              >
                {ok ? "Active" : "Inactive"}
              </span>
            </div>
          ))}
        </div>
      )}

      {error && (
        <p className="status-error">
          Cannot reach backend: {error}. Make sure Bootstrap dashboard is
          running.
        </p>
      )}
    </div>
  );
}
