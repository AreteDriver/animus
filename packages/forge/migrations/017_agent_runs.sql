-- Agent run persistence for SubAgentManager.
-- Stores completed agent runs so they survive process restarts
-- and can be queried for analytics and debugging.

CREATE TABLE IF NOT EXISTS agent_runs (
    run_id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    task TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    result TEXT,
    error TEXT,
    started_at REAL NOT NULL DEFAULT 0.0,
    completed_at REAL NOT NULL DEFAULT 0.0,
    parent_id TEXT,
    children TEXT NOT NULL DEFAULT '[]',  -- JSON array of run_ids
    config_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_status
    ON agent_runs(status);
CREATE INDEX IF NOT EXISTS idx_agent_runs_agent
    ON agent_runs(agent);
CREATE INDEX IF NOT EXISTS idx_agent_runs_parent
    ON agent_runs(parent_id);
CREATE INDEX IF NOT EXISTS idx_agent_runs_started
    ON agent_runs(started_at);
