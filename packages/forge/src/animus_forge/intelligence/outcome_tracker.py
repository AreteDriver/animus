"""Outcome tracking for agent step executions.

Records whether agent outputs actually worked and provides aggregate
queries to close the feedback loop in the orchestration pipeline.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from animus_forge.state.backends import DatabaseBackend


@dataclass(frozen=True)
class OutcomeRecord:
    """A single recorded outcome for an agent step execution.

    Attributes:
        step_id: Unique identifier for this step execution.
        workflow_id: Identifier of the parent workflow run.
        agent_role: The agent role that produced the output (e.g. "builder").
        provider: AI provider used (e.g. "openai", "anthropic").
        model: Specific model identifier (e.g. "gpt-4o").
        success: Whether the step produced a passing result.
        quality_score: Quality rating from 0.0 (worst) to 1.0 (best).
        cost_usd: Cost of the API call in USD.
        tokens_used: Total tokens consumed (prompt + completion).
        latency_ms: Wall-clock latency of the step in milliseconds.
        metadata: Arbitrary key-value metadata for the step.
        timestamp: UTC timestamp when the outcome was recorded.
    """

    step_id: str
    workflow_id: str
    agent_role: str
    provider: str
    model: str
    success: bool
    quality_score: float
    cost_usd: float
    tokens_used: int
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self) -> None:
        """Set default timestamp if not provided."""
        if not self.timestamp:
            object.__setattr__(self, "timestamp", datetime.now(UTC).isoformat())


@dataclass(frozen=True)
class ProviderStats:
    """Aggregate statistics for a provider/model combination.

    Attributes:
        success_rate: Fraction of calls that succeeded (0.0-1.0).
        avg_latency_ms: Mean latency across calls in milliseconds.
        avg_cost_usd: Mean cost per call in USD.
        total_calls: Number of calls in the aggregation window.
    """

    success_rate: float
    avg_latency_ms: float
    avg_cost_usd: float
    total_calls: int


_SCHEMA = """
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
"""


class OutcomeTracker:
    """Records and queries agent step outcomes.

    Thread-safe tracker backed by a ``DatabaseBackend``. Initialises its
    schema on first use and exposes both recording and aggregate query
    methods.

    Args:
        backend: A ``DatabaseBackend`` instance (e.g. ``SQLiteBackend``).

    Example::

        from animus_forge.state.backends import SQLiteBackend
        tracker = OutcomeTracker(SQLiteBackend("outcomes.db"))
        tracker.record(OutcomeRecord(
            step_id="s1", workflow_id="w1", agent_role="builder",
            provider="openai", model="gpt-4o", success=True,
            quality_score=0.95, cost_usd=0.03, tokens_used=1500,
            latency_ms=1200, metadata={"file": "api.py"},
        ))
        rate = tracker.get_agent_success_rate("builder")
    """

    def __init__(self, backend: DatabaseBackend) -> None:
        self._backend = backend
        self._lock = threading.Lock()
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """Create the outcome_records table if it does not exist."""
        with self._lock:
            self._backend.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, outcome: OutcomeRecord) -> None:
        """Persist a single outcome record.

        Args:
            outcome: The outcome to store. If ``step_id`` is empty a UUID
                is generated automatically.
        """
        step_id = outcome.step_id or str(uuid.uuid4())
        meta_json = json.dumps(outcome.metadata) if outcome.metadata else "{}"

        query = (
            "INSERT INTO outcome_records "
            "(step_id, workflow_id, agent_role, provider, model, success, "
            "quality_score, cost_usd, tokens_used, latency_ms, metadata, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        params = (
            step_id,
            outcome.workflow_id,
            outcome.agent_role,
            outcome.provider,
            outcome.model,
            1 if outcome.success else 0,
            outcome.quality_score,
            outcome.cost_usd,
            outcome.tokens_used,
            outcome.latency_ms,
            meta_json,
            outcome.timestamp,
        )

        with self._lock:
            with self._backend.transaction():
                self._backend.execute(query, params)

    def record_many(self, outcomes: list[OutcomeRecord]) -> None:
        """Persist multiple outcome records in a single transaction.

        Args:
            outcomes: List of outcomes to store.
        """
        if not outcomes:
            return

        query = (
            "INSERT INTO outcome_records "
            "(step_id, workflow_id, agent_role, provider, model, success, "
            "quality_score, cost_usd, tokens_used, latency_ms, metadata, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )

        rows: list[tuple[Any, ...]] = []
        for o in outcomes:
            rows.append(
                (
                    o.step_id or str(uuid.uuid4()),
                    o.workflow_id,
                    o.agent_role,
                    o.provider,
                    o.model,
                    1 if o.success else 0,
                    o.quality_score,
                    o.cost_usd,
                    o.tokens_used,
                    o.latency_ms,
                    json.dumps(o.metadata) if o.metadata else "{}",
                    o.timestamp,
                )
            )

        with self._lock:
            with self._backend.transaction():
                self._backend.executemany(query, rows)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def _cutoff_iso(self, days: int) -> str:
        """Return an ISO-8601 timestamp ``days`` ago from now (UTC)."""
        from datetime import timedelta

        return (datetime.now(UTC) - timedelta(days=days)).isoformat()

    def get_agent_success_rate(self, agent_role: str, days: int = 30) -> float:
        """Return the success rate for an agent role over a time window.

        Args:
            agent_role: The role to query (e.g. ``"builder"``).
            days: Look-back window in days. Defaults to 30.

        Returns:
            Success rate as a float between 0.0 and 1.0.  Returns 0.0 if
            there are no records in the window.
        """
        cutoff = self._cutoff_iso(days)
        query = (
            "SELECT AVG(success) AS rate FROM outcome_records "
            "WHERE agent_role = ? AND timestamp >= ?"
        )
        with self._lock:
            row = self._backend.fetchone(query, (agent_role, cutoff))
        if row and row.get("rate") is not None:
            return float(row["rate"])
        return 0.0

    def get_provider_stats(
        self,
        provider: str,
        model: str | None = None,
        days: int = 30,
    ) -> ProviderStats:
        """Return aggregate statistics for a provider (and optional model).

        Args:
            provider: Provider name (e.g. ``"openai"``).
            model: Optional model filter (e.g. ``"gpt-4o"``).
            days: Look-back window in days. Defaults to 30.

        Returns:
            A ``ProviderStats`` dataclass with the aggregated values.
            All-zero stats are returned when no records match.
        """
        cutoff = self._cutoff_iso(days)
        if model is not None:
            query = (
                "SELECT AVG(success) AS success_rate, "
                "AVG(latency_ms) AS avg_latency_ms, "
                "AVG(cost_usd) AS avg_cost_usd, "
                "COUNT(*) AS total_calls "
                "FROM outcome_records "
                "WHERE provider = ? AND model = ? AND timestamp >= ?"
            )
            params: tuple[Any, ...] = (provider, model, cutoff)
        else:
            query = (
                "SELECT AVG(success) AS success_rate, "
                "AVG(latency_ms) AS avg_latency_ms, "
                "AVG(cost_usd) AS avg_cost_usd, "
                "COUNT(*) AS total_calls "
                "FROM outcome_records "
                "WHERE provider = ? AND timestamp >= ?"
            )
            params = (provider, cutoff)

        with self._lock:
            row = self._backend.fetchone(query, params)

        if not row or row.get("total_calls", 0) == 0:
            return ProviderStats(
                success_rate=0.0,
                avg_latency_ms=0.0,
                avg_cost_usd=0.0,
                total_calls=0,
            )

        return ProviderStats(
            success_rate=float(row["success_rate"]),
            avg_latency_ms=float(row["avg_latency_ms"]),
            avg_cost_usd=float(row["avg_cost_usd"]),
            total_calls=int(row["total_calls"]),
        )

    def get_best_provider_for_role(self, agent_role: str, days: int = 30) -> tuple[str, str]:
        """Return the best provider/model combo for a given agent role.

        "Best" is defined as the highest ``quality_score * success_rate``
        product, computed per (provider, model) pair.

        Args:
            agent_role: The agent role to query.
            days: Look-back window in days. Defaults to 30.

        Returns:
            A ``(provider, model)`` tuple. Returns ``("", "")`` when no
            records exist.
        """
        cutoff = self._cutoff_iso(days)
        query = (
            "SELECT provider, model, "
            "AVG(quality_score) * AVG(success) AS score "
            "FROM outcome_records "
            "WHERE agent_role = ? AND timestamp >= ? "
            "GROUP BY provider, model "
            "ORDER BY score DESC "
            "LIMIT 1"
        )
        with self._lock:
            row = self._backend.fetchone(query, (agent_role, cutoff))

        if not row:
            return ("", "")
        return (str(row["provider"]), str(row["model"]))

    def get_workflow_outcomes(self, workflow_id: str) -> list[OutcomeRecord]:
        """Return all outcome records for a workflow run.

        Args:
            workflow_id: The workflow identifier.

        Returns:
            List of ``OutcomeRecord`` instances ordered by timestamp.
        """
        query = "SELECT * FROM outcome_records WHERE workflow_id = ? ORDER BY timestamp"
        with self._lock:
            rows = self._backend.fetchall(query, (workflow_id,))

        return [self._row_to_record(r) for r in rows]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: dict[str, Any]) -> OutcomeRecord:
        """Convert a database row dict to an ``OutcomeRecord``."""
        meta = row.get("metadata", "{}")
        if isinstance(meta, str):
            meta = json.loads(meta)

        return OutcomeRecord(
            step_id=str(row["step_id"]),
            workflow_id=str(row["workflow_id"]),
            agent_role=str(row["agent_role"]),
            provider=str(row["provider"]),
            model=str(row["model"]),
            success=bool(row["success"]),
            quality_score=float(row["quality_score"]),
            cost_usd=float(row["cost_usd"]),
            tokens_used=int(row["tokens_used"]),
            latency_ms=float(row["latency_ms"]),
            metadata=meta,
            timestamp=str(row["timestamp"]),
        )
