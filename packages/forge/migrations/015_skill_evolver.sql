-- Skill evolution: metrics, versions, experiments, deprecation tracking.

-- Ensure outcome_records exists (normally created by OutcomeTracker, but
-- migrations may run before it initialises).
CREATE TABLE IF NOT EXISTS outcome_records (
    step_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    agent_role TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    success INTEGER NOT NULL,
    quality_score REAL NOT NULL,
    cost_usd REAL NOT NULL,
    tokens_used INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_outcome_agent_role ON outcome_records(agent_role);
CREATE INDEX IF NOT EXISTS idx_outcome_provider ON outcome_records(provider);
CREATE INDEX IF NOT EXISTS idx_outcome_workflow ON outcome_records(workflow_id);
CREATE INDEX IF NOT EXISTS idx_outcome_timestamp ON outcome_records(timestamp);

-- Add skill tracking columns to outcome_records
ALTER TABLE outcome_records ADD COLUMN skill_name TEXT DEFAULT '';
ALTER TABLE outcome_records ADD COLUMN skill_version TEXT DEFAULT '';
CREATE INDEX IF NOT EXISTS idx_outcome_skill_name ON outcome_records(skill_name);

-- Aggregated skill metrics (materialized periodically)
CREATE TABLE IF NOT EXISTS skill_metrics (
    skill_name TEXT NOT NULL,
    skill_version TEXT NOT NULL,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    total_invocations INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0.0,
    avg_quality_score REAL DEFAULT 0.0,
    avg_cost_usd REAL DEFAULT 0.0,
    avg_latency_ms REAL DEFAULT 0.0,
    total_cost_usd REAL DEFAULT 0.0,
    trend TEXT DEFAULT 'stable',
    computed_at TEXT NOT NULL,
    PRIMARY KEY (skill_name, skill_version, period_start)
);

-- Skill version history (full YAML snapshots)
CREATE TABLE IF NOT EXISTS skill_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_name TEXT NOT NULL,
    version TEXT NOT NULL,
    previous_version TEXT,
    change_type TEXT NOT NULL,
    change_description TEXT,
    schema_snapshot TEXT NOT NULL,
    diff_summary TEXT,
    approval_id TEXT,
    created_at TEXT NOT NULL,
    created_by TEXT DEFAULT 'skill_evolver',
    UNIQUE(skill_name, version)
);

-- A/B test experiments
CREATE TABLE IF NOT EXISTS skill_experiments (
    id TEXT PRIMARY KEY,
    skill_name TEXT NOT NULL,
    control_version TEXT NOT NULL,
    variant_version TEXT NOT NULL,
    traffic_split REAL DEFAULT 0.5,
    status TEXT DEFAULT 'active',
    min_invocations INTEGER DEFAULT 100,
    start_date TEXT NOT NULL,
    end_date TEXT,
    winner TEXT,
    conclusion_reason TEXT,
    created_at TEXT NOT NULL,
    concluded_at TEXT
);

-- Deprecation tracking
CREATE TABLE IF NOT EXISTS skill_deprecations (
    skill_name TEXT PRIMARY KEY,
    status TEXT DEFAULT 'flagged',
    flagged_at TEXT NOT NULL,
    deprecated_at TEXT,
    retired_at TEXT,
    reason TEXT,
    success_rate_at_flag REAL,
    invocations_at_flag INTEGER,
    replacement_skill TEXT,
    approval_id TEXT
);
