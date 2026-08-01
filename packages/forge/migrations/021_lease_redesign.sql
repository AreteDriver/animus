-- Migration 021: Lease redesign for retry, audit history, and atomic dispatch.
--
-- Replaces the old task_leases table (permanent UNIQUE on task_id) with:
--   - task_lease_current: one mutable row per task while a lease is active
--   - task_lease_history: append-only record of every lease event
--   - task_attempts: per-attempt metadata used for fencing and retry tracking
--
-- Timestamps are stored as UTC ISO-8601 strings (TEXT) for SQLite/Postgres parity.

-- New current-lease table: exactly one active row per task at any time.
CREATE TABLE IF NOT EXISTS task_lease_current (
    task_id TEXT PRIMARY KEY,
    lease_id TEXT NOT NULL,
    mission_id TEXT NOT NULL,
    citizen_role TEXT NOT NULL,
    worker_id TEXT NOT NULL,
    generation INTEGER NOT NULL DEFAULT 1,
    acquired_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    heartbeat_at TEXT,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'expired', 'released')),
    attempt_id TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_task_lease_current_expires ON task_lease_current(expires_at);
CREATE INDEX IF NOT EXISTS idx_task_lease_current_mission ON task_lease_current(mission_id);

-- Append-only lease history for audit and recovery analysis.
CREATE TABLE IF NOT EXISTS task_lease_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    lease_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    mission_id TEXT NOT NULL,
    citizen_role TEXT NOT NULL,
    worker_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    acquired_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    heartbeat_at TEXT,
    status TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    outcome TEXT,
    recorded_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lease_history_task ON task_lease_history(task_id, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_lease_history_lease ON task_lease_history(lease_id);

-- Attempt records created at dispatch time.
CREATE TABLE IF NOT EXISTS task_attempts (
    attempt_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    mission_id TEXT NOT NULL,
    citizen_role TEXT NOT NULL,
    lease_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'started'
        CHECK (status IN ('started', 'completed', 'failed', 'cancelled')),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    cost_usd TEXT DEFAULT '0.00',
    FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_attempts_task ON task_attempts(task_id, started_at DESC);

-- Data migration from old task_leases table, if it exists.
-- Copy all rows into history, then create current rows for still-active leases.
-- Active old leases get generation 1 and a synthetic attempt_id equal to lease_id.
-- The guard table makes this idempotent on fresh databases that never had task_leases.
CREATE TABLE IF NOT EXISTS task_leases (
    lease_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL UNIQUE,
    mission_id TEXT NOT NULL,
    citizen_role TEXT NOT NULL,
    worker_id TEXT NOT NULL,
    acquired_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    heartbeat_at TEXT,
    outcome TEXT
);

INSERT INTO task_lease_history (
    lease_id, task_id, mission_id, citizen_role, worker_id, generation,
    acquired_at, expires_at, heartbeat_at, status, attempt_id, outcome, recorded_at
)
SELECT
    lease_id, task_id, mission_id, citizen_role, worker_id, 1,
    acquired_at, expires_at, heartbeat_at, status, lease_id, outcome,
    CURRENT_TIMESTAMP
FROM task_leases
WHERE TRUE
ON CONFLICT DO NOTHING;

INSERT INTO task_lease_current (
    task_id, lease_id, mission_id, citizen_role, worker_id, generation,
    acquired_at, expires_at, heartbeat_at, status, attempt_id
)
SELECT
    task_id, lease_id, mission_id, citizen_role, worker_id, 1,
    acquired_at, expires_at, heartbeat_at, 'active', lease_id
FROM task_leases
WHERE status = 'active'
ON CONFLICT DO NOTHING;

INSERT INTO task_attempts (
    attempt_id, task_id, mission_id, citizen_role, lease_id, generation,
    status, started_at, completed_at, cost_usd
)
SELECT
    lease_id, task_id, mission_id, citizen_role, lease_id, 1,
    'started', acquired_at, NULL, '0.00'
FROM task_leases
WHERE status = 'active'
ON CONFLICT DO NOTHING;

-- Drop the old table now that its data has been preserved.
DROP TABLE IF EXISTS task_leases;
