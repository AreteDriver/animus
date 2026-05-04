-- Migration 019: Persist content failure modes per case result
-- Complements 018's technical failure_mode with a JSON list of F1-F8 content tags.

ALTER TABLE eval_case_results ADD COLUMN content_failure_modes_json TEXT;

CREATE INDEX IF NOT EXISTS idx_eval_case_content_failures
    ON eval_case_results(content_failure_modes_json);
