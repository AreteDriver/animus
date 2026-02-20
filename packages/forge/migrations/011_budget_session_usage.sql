-- Migration 011: Session-level budget usage tracking
-- Persists BudgetManager token usage across restarts

CREATE TABLE IF NOT EXISTS budget_session_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    tokens INTEGER NOT NULL DEFAULT 0,
    operation TEXT,
    recorded_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_budget_session_usage_session
    ON budget_session_usage(session_id);
