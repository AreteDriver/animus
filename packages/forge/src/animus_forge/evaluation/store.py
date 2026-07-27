"""Persistent evaluation results store.

Follows the TaskStore pattern (shared DatabaseBackend singleton,
transaction wrapping) for storing eval suite runs and case results.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any

from animus_forge.state.backends import DatabaseBackend

from .runner import SuiteResult

logger = logging.getLogger(__name__)

_eval_store: EvalStore | None = None


class EvalStore:
    """Analytics store for evaluation benchmark results.

    Provides CRUD and aggregation for eval suite runs,
    with an optional bridge to OutcomeTracker for quality scoring.
    """

    def __init__(self, backend: DatabaseBackend):
        self.backend = backend

    # =========================================================================
    # Record
    # =========================================================================

    def record_run(
        self,
        suite_name: str,
        result: SuiteResult,
        *,
        agent_role: str | None = None,
        model: str | None = None,
        run_mode: str = "live",
        metadata: dict[str, Any] | None = None,
        rubric_name: str | None = None,
        rubric_version: str | None = None,
        config_hash: str | None = None,
        prompt_version: str | None = None,
    ) -> str:
        """Record a completed evaluation run with all case results.

        Returns the generated run UUID.  If an equivalent run was recorded
        within the last 60 minutes (same suite, agent, model, mode and an
        identical pass_rate within 0.01) the existing run_id is returned
        instead of writing a duplicate row.
        """
        now = datetime.now()
        started_at = (result.timestamp or now).isoformat()
        completed_at = now.isoformat()
        total_tokens = sum(r.tokens_used for r in result.results)
        total_cost_usd = sum(getattr(r, "cost_usd", 0.0) or 0.0 for r in result.results)
        meta_json = json.dumps(metadata) if metadata else None

        # Deduplication: look for a near-identical run in the last hour.
        cutoff = (now - timedelta(hours=1)).isoformat()
        dup_row = self.backend.fetchone(
            """
            SELECT id, pass_rate
            FROM eval_runs
            WHERE suite_name = ?
              AND (agent_role IS NULL OR agent_role = ?)
              AND (model IS NULL OR model = ?)
              AND run_mode = ?
              AND completed_at >= ?
            ORDER BY completed_at DESC
            LIMIT 1
            """,
            (suite_name, agent_role, model, run_mode, cutoff),
        )
        if dup_row is not None:
            stored_pass = float(dup_row["pass_rate"] or 0.0)
            if abs(stored_pass - result.pass_rate) <= 0.01:
                logger.info(
                    "Deduplicated eval run for suite '%s' (pass_rate=%.2f, existing=%s)",
                    suite_name,
                    result.pass_rate,
                    dup_row["id"][:8],
                )
                return dup_row["id"]

        run_id = str(uuid.uuid4())

        with self.backend.transaction():
            self.backend.execute(
                """
                INSERT INTO eval_runs
                    (id, suite_name, agent_role, model, run_mode,
                     started_at, completed_at, duration_ms,
                     total_cases, passed, failed, errors, skipped,
                     avg_score, pass_rate, score_variance, total_tokens, metadata,
                     rubric_name, rubric_version, config_hash,
                     prompt_version, total_cost_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    suite_name,
                    agent_role,
                    model,
                    run_mode,
                    started_at,
                    completed_at,
                    result.duration_ms,
                    result.total,
                    result.passed,
                    result.failed,
                    result.errors,
                    result.skipped,
                    result.total_score,
                    result.pass_rate,
                    result.score_variance,
                    total_tokens,
                    meta_json,
                    rubric_name,
                    rubric_version,
                    config_hash,
                    prompt_version,
                    total_cost_usd,
                ),
            )

            for case_result in result.results:
                metrics_json = json.dumps(case_result.metrics) if case_result.metrics else None
                output_text = str(case_result.output)[:2000] if case_result.output else None
                rubric_scores_json = (
                    json.dumps(case_result.rubric_scores)
                    if getattr(case_result, "rubric_scores", None)
                    else None
                )
                content_failure_modes = (
                    case_result.metadata.get("content_failure_modes")
                    if case_result.metadata
                    else None
                )
                content_failures_json = (
                    json.dumps(content_failure_modes) if content_failure_modes else None
                )
                self.backend.execute(
                    """
                    INSERT INTO eval_case_results
                        (run_id, case_name, status, score, output, error,
                         latency_ms, tokens_used, metrics_json,
                         failure_mode, rubric_band, rubric_scores_json,
                         cost_usd, content_failure_modes_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        case_result.case.name,
                        case_result.status.value,
                        case_result.score,
                        output_text,
                        case_result.error,
                        case_result.latency_ms,
                        case_result.tokens_used,
                        metrics_json,
                        getattr(case_result, "failure_mode", None),
                        getattr(case_result, "rubric_band", None),
                        rubric_scores_json,
                        getattr(case_result, "cost_usd", 0.0) or 0.0,
                        content_failures_json,
                    ),
                )

        logger.info(
            "Recorded eval run %s for suite '%s': %d/%d passed (%.0f%%)",
            run_id[:8],
            suite_name,
            result.passed,
            result.total,
            result.pass_rate * 100,
        )
        return run_id

    # =========================================================================
    # Query
    # =========================================================================

    def query_runs(
        self,
        *,
        suite_name: str | None = None,
        agent_role: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Query recent eval runs with optional filters."""
        query = """
            SELECT id, suite_name, agent_role, model, run_mode,
                   started_at, completed_at, duration_ms,
                   total_cases, passed, failed, errors, skipped,
                   avg_score, pass_rate, total_tokens, metadata,
                   rubric_name, rubric_version, config_hash,
                   prompt_version, total_cost_usd
            FROM eval_runs
            WHERE 1=1
        """
        params: list[Any] = []

        if suite_name is not None:
            query += " AND suite_name = ?"
            params.append(suite_name)
        if agent_role is not None:
            query += " AND agent_role = ?"
            params.append(agent_role)

        query += " ORDER BY completed_at DESC LIMIT ?"
        params.append(limit)

        rows = self.backend.fetchall(query, tuple(params))
        return [self._parse_run_row(row) for row in rows]

    def get_run(self, run_id: str) -> dict | None:
        """Get a single run with its case results."""
        row = self.backend.fetchone("SELECT * FROM eval_runs WHERE id = ?", (run_id,))
        if not row:
            return None

        run = self._parse_run_row(row)

        case_rows = self.backend.fetchall(
            """
            SELECT case_name, status, score, output, error,
                   latency_ms, tokens_used, metrics_json
            FROM eval_case_results
            WHERE run_id = ?
            ORDER BY id
            """,
            (run_id,),
        )
        run["case_results"] = [self._parse_case_row(r) for r in case_rows]
        return run

    def get_suite_trend(self, suite_name: str, days: int = 30) -> list[dict]:
        """Get time-series data for a suite's quality trend."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        rows = self.backend.fetchall(
            """
            SELECT id, completed_at, avg_score, pass_rate,
                   total_cases, passed, duration_ms, model
            FROM eval_runs
            WHERE suite_name = ? AND completed_at >= ?
            ORDER BY completed_at ASC
            """,
            (suite_name, cutoff),
        )
        return [dict(r) for r in rows]

    def get_agent_summary(self, agent_role: str) -> dict:
        """Get aggregate eval stats for an agent role."""
        row = self.backend.fetchone(
            """
            SELECT
                COUNT(*) as total_runs,
                COALESCE(AVG(avg_score), 0.0) as avg_score,
                COALESCE(AVG(pass_rate), 0.0) as avg_pass_rate,
                COALESCE(SUM(total_cases), 0) as total_cases,
                COALESCE(SUM(passed), 0) as total_passed,
                COALESCE(SUM(failed), 0) as total_failed,
                COALESCE(SUM(errors), 0) as total_errors
            FROM eval_runs
            WHERE agent_role = ?
            """,
            (agent_role,),
        )

        if not row or row["total_runs"] == 0:
            return {
                "agent_role": agent_role,
                "total_runs": 0,
                "avg_score": 0.0,
                "avg_pass_rate": 0.0,
                "total_cases": 0,
                "total_passed": 0,
                "total_failed": 0,
                "total_errors": 0,
            }

        return {
            "agent_role": agent_role,
            "total_runs": row["total_runs"],
            "avg_score": round(float(row["avg_score"]), 4),
            "avg_pass_rate": round(float(row["avg_pass_rate"]), 4),
            "total_cases": row["total_cases"],
            "total_passed": row["total_passed"],
            "total_failed": row["total_failed"],
            "total_errors": row["total_errors"],
        }

    # =========================================================================
    # Baseline / Regression
    # =========================================================================

    def set_baseline(self, suite_name: str, run_id: str) -> None:
        """Mark a specific eval run as the gold baseline for a suite."""
        with self.backend.transaction():
            # Unset any existing baseline for this suite
            self.backend.execute(
                """
                UPDATE eval_runs
                SET metadata = json_set(COALESCE(metadata, '{}'), '$.is_baseline', 0)
                WHERE suite_name = ? AND json_extract(COALESCE(metadata, '{}'), '$.is_baseline') = 1
                """,
                (suite_name,),
            )
            # Set new baseline
            self.backend.execute(
                """
                UPDATE eval_runs
                SET metadata = json_set(COALESCE(metadata, '{}'), '$.is_baseline', 1)
                WHERE id = ?
                """,
                (run_id,),
            )
        logger.info("Baseline set for suite '%s': run=%s", suite_name, run_id[:8])

    def get_baseline(self, suite_name: str) -> dict | None:
        """Get the current baseline run for a suite."""
        row = self.backend.fetchone(
            """
            SELECT id, pass_rate, avg_score, score_variance, metadata, completed_at
            FROM eval_runs
            WHERE suite_name = ?
              AND json_extract(COALESCE(metadata, '{}'), '$.is_baseline') = 1
            ORDER BY completed_at DESC
            LIMIT 1
            """,
            (suite_name,),
        )
        if not row:
            return None
        result = dict(row)
        if result.get("metadata"):
            try:
                result["metadata"] = json.loads(result["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
        return result

    def check_regression(
        self,
        suite_name: str,
        current_pass_rate: float,
        *,
        delta_threshold: float = 0.2,
    ) -> dict:
        """Compare current pass_rate against the stored baseline.

        Returns:
            Dict with ``regression_detected``, ``delta``, ``baseline_pass_rate``.
        """
        baseline = self.get_baseline(suite_name)
        if baseline is None:
            return {
                "regression_detected": False,
                "delta": 0.0,
                "baseline_pass_rate": None,
                "reason": "no_baseline",
            }

        baseline_pass_rate = float(baseline.get("pass_rate", 1.0))
        delta = baseline_pass_rate - current_pass_rate
        detected = delta > delta_threshold

        return {
            "regression_detected": detected,
            "delta": round(delta, 4),
            "baseline_pass_rate": baseline_pass_rate,
            "reason": "regression" if detected else "within_tolerance",
        }

    # =========================================================================
    # OutcomeTracker bridge
    # =========================================================================

    def feed_to_outcome_tracker(
        self,
        run_id: str,
        workflow_id: str,
        *,
        provider: str = "eval",
        model: str = "mock",
    ) -> int:
        """Bridge eval results to OutcomeTracker for quality scoring.

        Creates one OutcomeRecord per case result in the run.
        Returns count of records fed.
        """
        run = self.get_run(run_id)
        if not run or not run.get("case_results"):
            return 0

        try:
            from animus_forge.intelligence.outcome_tracker import (
                OutcomeRecord,
                OutcomeTracker,
            )
            from animus_forge.state.database import get_database

            tracker = OutcomeTracker(get_database())
            records = []
            for cr in run["case_results"]:
                records.append(
                    OutcomeRecord(
                        step_id=f"eval-{run_id[:8]}-{cr['case_name']}",
                        workflow_id=workflow_id,
                        agent_role=run.get("agent_role") or "unknown",
                        provider=provider,
                        model=run.get("model") or model,
                        success=cr["status"] == "passed",
                        quality_score=cr["score"],
                        cost_usd=0.0,
                        tokens_used=cr.get("tokens_used", 0),
                        latency_ms=cr.get("latency_ms", 0.0),
                        metadata={"source": "eval", "suite": run["suite_name"]},
                    )
                )

            tracker.record_many(records)
            return len(records)

        except Exception as e:
            logger.warning("Failed to feed eval results to OutcomeTracker: %s", e)
            return 0

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _parse_run_row(self, row: dict) -> dict:
        result = dict(row)
        if result.get("metadata"):
            try:
                result["metadata"] = json.loads(result["metadata"])
            except (json.JSONDecodeError, TypeError):
                logger.debug("Failed to parse run metadata as JSON, keeping raw value")
        return result

    def _parse_case_row(self, row: dict) -> dict:
        result = dict(row)
        if result.get("metrics_json"):
            try:
                result["metrics"] = json.loads(result["metrics_json"])
            except (json.JSONDecodeError, TypeError):
                result["metrics"] = {}
        else:
            result["metrics"] = {}
        result.pop("metrics_json", None)

        if result.get("rubric_scores_json"):
            try:
                result["rubric_scores"] = json.loads(result["rubric_scores_json"])
            except (json.JSONDecodeError, TypeError):
                result["rubric_scores"] = {}
        else:
            result["rubric_scores"] = {}
        result.pop("rubric_scores_json", None)

        if result.get("content_failure_modes_json"):
            try:
                result["content_failure_modes"] = json.loads(result["content_failure_modes_json"])
            except (json.JSONDecodeError, TypeError):
                result["content_failure_modes"] = []
        else:
            result["content_failure_modes"] = []
        result.pop("content_failure_modes_json", None)
        return result


# =============================================================================
# Global access
# =============================================================================


def get_eval_store() -> EvalStore:
    """Get or create the global EvalStore singleton."""
    global _eval_store
    if _eval_store is None:
        from animus_forge.state.database import get_database

        _eval_store = EvalStore(get_database())
    return _eval_store


def reset_eval_store() -> None:
    """Reset the global EvalStore singleton (for testing)."""
    global _eval_store
    _eval_store = None
