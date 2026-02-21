-- Self-improvement approval persistence.
-- Tracks approval requests across restarts so the orchestrator
-- can resume waiting for human decisions.

CREATE TABLE IF NOT EXISTS self_improve_approvals (
    id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,          -- plan | apply | merge
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    details TEXT NOT NULL DEFAULT '{}',  -- JSON
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | approved | rejected | expired
    created_at TEXT NOT NULL,
    decided_at TEXT,
    decided_by TEXT,
    reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_si_approvals_status
    ON self_improve_approvals(status);
CREATE INDEX IF NOT EXISTS idx_si_approvals_stage
    ON self_improve_approvals(stage);
