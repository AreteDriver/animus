-- Migration 018: Rubric + failure taxonomy + cost awareness on eval results
-- Extends 012_eval_results.sql with structured scoring and triage fields.

-- Runs
ALTER TABLE eval_runs ADD COLUMN rubric_name TEXT;
ALTER TABLE eval_runs ADD COLUMN rubric_version TEXT;
ALTER TABLE eval_runs ADD COLUMN config_hash TEXT;
ALTER TABLE eval_runs ADD COLUMN prompt_version TEXT;
ALTER TABLE eval_runs ADD COLUMN total_cost_usd REAL DEFAULT 0.0;

-- Case results
ALTER TABLE eval_case_results ADD COLUMN failure_mode TEXT;
ALTER TABLE eval_case_results ADD COLUMN rubric_band TEXT;
ALTER TABLE eval_case_results ADD COLUMN rubric_scores_json TEXT;
ALTER TABLE eval_case_results ADD COLUMN cost_usd REAL DEFAULT 0.0;

-- Indexes for triage queries
CREATE INDEX IF NOT EXISTS idx_eval_runs_rubric ON eval_runs(rubric_name, rubric_version);
CREATE INDEX IF NOT EXISTS idx_eval_case_failure ON eval_case_results(failure_mode);
CREATE INDEX IF NOT EXISTS idx_eval_case_band ON eval_case_results(rubric_band);
