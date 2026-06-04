-- Migration 012: persist cost-weighted Effective-Tokens per usage record
-- so BudgetManager._total_effective survives a restart (the A1 flip makes ET
-- an enforced ceiling; restore must not under-count it). Raw `tokens` alone
-- loses the model/breakdown that ET is computed from, so we store ET directly.

ALTER TABLE budget_session_usage
    ADD COLUMN effective_tokens REAL NOT NULL DEFAULT 0;
