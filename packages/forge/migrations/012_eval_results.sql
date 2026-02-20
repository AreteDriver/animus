-- Migration 012: Evaluation benchmark results
-- Stores eval suite runs and per-case results for quality tracking

CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY,
    suite_name TEXT NOT NULL,
    agent_role TEXT,
    model TEXT,
    run_mode TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    duration_ms REAL NOT NULL,
    total_cases INTEGER DEFAULT 0,
    passed INTEGER DEFAULT 0,
    failed INTEGER DEFAULT 0,
    errors INTEGER DEFAULT 0,
    skipped INTEGER DEFAULT 0,
    avg_score REAL DEFAULT 0.0,
    pass_rate REAL DEFAULT 0.0,
    total_tokens INTEGER DEFAULT 0,
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_suite ON eval_runs(suite_name);
CREATE INDEX IF NOT EXISTS idx_eval_runs_completed ON eval_runs(completed_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_runs_agent ON eval_runs(agent_role);

CREATE TABLE IF NOT EXISTS eval_case_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    case_name TEXT NOT NULL,
    status TEXT NOT NULL,
    score REAL NOT NULL,
    output TEXT,
    error TEXT,
    latency_ms REAL DEFAULT 0,
    tokens_used INTEGER DEFAULT 0,
    metrics_json TEXT,
    FOREIGN KEY (run_id) REFERENCES eval_runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_eval_case_run ON eval_case_results(run_id);
