-- Migration 010: Task history for analytics and phi-weighted scoring
-- Denormalized task records, agent performance scores, and daily budget rollups

-- Task history — denormalized record of completed/failed tasks
CREATE TABLE IF NOT EXISTS task_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    workflow_id TEXT NOT NULL,
    status TEXT NOT NULL,
    agent_role TEXT,
    model TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    duration_ms INTEGER DEFAULT 0,
    error TEXT,
    metadata TEXT,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_task_history_job_id ON task_history(job_id);
CREATE INDEX IF NOT EXISTS idx_task_history_workflow_id ON task_history(workflow_id);
CREATE INDEX IF NOT EXISTS idx_task_history_status ON task_history(status);
CREATE INDEX IF NOT EXISTS idx_task_history_agent_role ON task_history(agent_role);
CREATE INDEX IF NOT EXISTS idx_task_history_completed_at ON task_history(completed_at DESC);

-- Agent scores — aggregated per-agent performance
CREATE TABLE IF NOT EXISTS agent_scores (
    agent_role TEXT PRIMARY KEY,
    total_tasks INTEGER DEFAULT 0,
    successful_tasks INTEGER DEFAULT 0,
    failed_tasks INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_cost_usd REAL DEFAULT 0.0,
    avg_duration_ms REAL DEFAULT 0.0,
    success_rate REAL DEFAULT 0.0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Budget log — daily token/cost rollups
CREATE TABLE IF NOT EXISTS budget_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,  -- YYYY-MM-DD
    agent_role TEXT,
    total_tokens INTEGER DEFAULT 0,
    total_cost_usd REAL DEFAULT 0.0,
    task_count INTEGER DEFAULT 0,
    UNIQUE(date, agent_role)
);

CREATE INDEX IF NOT EXISTS idx_budget_log_date ON budget_log(date DESC);
