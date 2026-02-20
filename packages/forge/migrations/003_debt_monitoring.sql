-- Migration 003: Technical debt monitoring and audit baselines
-- Supports Zorya Polunochnaya's system health auditing role

CREATE TABLE IF NOT EXISTS audit_baselines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    task_completion_time_avg REAL,
    agent_spawn_time_avg REAL,
    idle_cpu_percent REAL,
    idle_memory_percent REAL,
    skill_hashes TEXT,
    config_snapshots TEXT,
    package_versions TEXT,
    is_active INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS technical_debt (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    title TEXT NOT NULL,
    description TEXT,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source TEXT NOT NULL CHECK (source IN ('audit', 'manual', 'incident')),
    estimated_effort TEXT,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'acknowledged', 'in_progress', 'resolved')),
    resolution TEXT,
    resolved_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_technical_debt_status ON technical_debt(status);
CREATE INDEX IF NOT EXISTS idx_technical_debt_severity ON technical_debt(severity);
CREATE INDEX IF NOT EXISTS idx_technical_debt_category ON technical_debt(category);
CREATE INDEX IF NOT EXISTS idx_technical_debt_detected_at ON technical_debt(detected_at);

CREATE TABLE IF NOT EXISTS audit_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    check_name TEXT NOT NULL,
    category TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ok', 'warning', 'critical')),
    result_data TEXT,
    run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_results_check ON audit_results(check_name);
CREATE INDEX IF NOT EXISTS idx_audit_results_run_at ON audit_results(run_at);
