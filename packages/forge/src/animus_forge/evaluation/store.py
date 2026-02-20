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
    ) -> str:
        """Record a completed evaluation run with all case results.

        Returns the generated run UUID.
        """
        run_id = str(uuid.uuid4())
        now = datetime.now()
        started_at = (result.timestamp or now).isoformat()
        completed_at = now.isoformat()
        total_tokens = sum(r.tokens_used for r in result.results)
        meta_json = json.dumps(metadata) if metadata else None

        with self.backend.transaction():
            self.backend.execute(
                """
                INSERT INTO eval_runs
                    (id, suite_name, agent_role, model, run_mode,
                     started_at, completed_at, duration_ms,
                     total_cases, passed, failed, errors, skipped,
                     avg_score, pass_rate, total_tokens, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    total_tokens,
                    meta_json,
                ),
            )

            for case_result in result.results:
                metrics_json = json.dumps(case_result.metrics) if case_result.metrics else None
                output_text = str(case_result.output)[:2000] if case_result.output else None
                self.backend.execute(
                    """
                    INSERT INTO eval_case_results
                        (run_id, case_name, status, score, output, error,
                         latency_ms, tokens_used, metrics_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                   avg_score, pass_rate, total_tokens, metadata
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
