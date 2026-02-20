-- Resume tokens for approval gates in workflow execution.
-- When a workflow hits an approval step, execution halts and a
-- compact token is returned. External callers resume with the token.

CREATE TABLE IF NOT EXISTS approval_tokens (
    token TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL,
    workflow_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    next_step_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    prompt TEXT,
    preview TEXT,
    context TEXT,
    timeout_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    decided_at TIMESTAMP,
    decided_by TEXT
);

CREATE INDEX IF NOT EXISTS idx_approval_tokens_execution
    ON approval_tokens(execution_id);
