"""Token Budget Manager."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from animus_forge.state.backends import DatabaseBackend

logger = logging.getLogger(__name__)


class BudgetStatus(Enum):
    """Budget status indicators."""

    OK = "ok"
    WARNING = "warning"  # > 75% used
    CRITICAL = "critical"  # > 90% used
    EXCEEDED = "exceeded"  # > 100% used


@dataclass
class UsageRecord:
    """Record of token usage.

    ``tokens`` is the rolled-up total (kept for back-compat). When the
    underlying provider returns a breakdown, ``input_tokens`` /
    ``output_tokens`` / ``cache_read_tokens`` carry it so we can score
    workflows with the *Effective Tokens* cost metric (see ``effective_tokens``).
    Breakdown fields default to 0 when the caller doesn't supply them.
    """

    agent_id: str
    tokens: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    operation: str = ""
    metadata: dict = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    model: str | None = None


# Effective-Tokens weighting — input/cache/output have very different costs.
# GitHub's "Improving token efficiency in agentic workflows" (2026-05) reports
# the same weighting under the name ET = m × (1.0·I + 0.1·C + 4.0·O).
ET_INPUT_WEIGHT = 1.0
ET_CACHE_READ_WEIGHT = 0.1
ET_OUTPUT_WEIGHT = 4.0


# Default model-tier multipliers (m), normalised so Sonnet = 1.0. Reflect
# Anthropic's published price ratios — Haiku is the cheap tier, Opus the
# expensive one. Override via ``BudgetConfig.model_multipliers``; unknown
# models fall back to 1.0 (Sonnet-equivalent).
DEFAULT_MODEL_MULTIPLIERS: dict[str, float] = {
    "haiku": 0.08,
    "claude-haiku": 0.08,
    "claude-haiku-4-5": 0.08,
    "sonnet": 1.0,
    "claude-sonnet": 1.0,
    "claude-sonnet-4-6": 1.0,
    "opus": 5.0,
    "claude-opus": 5.0,
    "claude-opus-4-7": 5.0,
    "gpt-4o-mini": 0.05,
    "gpt-4o": 0.5,
    "gpt-5": 0.7,
    "ollama": 0.0,  # local — no $ cost (compute is the cost, not tokens)
}


def _resolve_model_multiplier(
    model: str | None,
    table: dict[str, float] | None,
) -> float:
    """Look up a model's tier multiplier, with a substring fallback for
    versioned ids (e.g. "claude-sonnet-4-6-1m" → "claude-sonnet")."""
    if not model:
        return 1.0
    table = table if table is not None else DEFAULT_MODEL_MULTIPLIERS
    m = model.lower()
    if m in table:
        return table[m]
    for key, val in table.items():
        if key in m:
            return val
    return 1.0


def effective_tokens(
    record: UsageRecord,
    model_multipliers: dict[str, float] | None = None,
) -> float:
    """The cost-weighted Effective-Tokens value for one usage record.

    ``ET = m × (1.0·I + 0.1·C + 4.0·O)`` when an input/output breakdown is
    present. When the record carries no breakdown (callers that only pass
    ``tokens``), the output weight cannot be applied to cost we never
    observed, so ET is model-weighted only (``m × tokens``) — neutral for an
    unknown model. This keeps ET enforcement non-breaking on the executor's
    raw-only record path while still penalizing a known-expensive tier.
    """
    m = _resolve_model_multiplier(record.model, model_multipliers)
    if record.input_tokens or record.output_tokens or record.cache_read_tokens:
        return m * (
            ET_INPUT_WEIGHT * record.input_tokens
            + ET_CACHE_READ_WEIGHT * record.cache_read_tokens
            + ET_OUTPUT_WEIGHT * record.output_tokens
        )
    # No in/out breakdown: apply the model-tier multiplier but NOT the output
    # weight. We can't claim output cost we didn't observe, so a raw-only
    # record stays neutral (ET == raw for an unknown model), which keeps ET
    # enforcement non-breaking on the executor's raw-only record path while
    # still penalizing a known-expensive tier (e.g. opus, m=5).
    return m * record.tokens


@dataclass
class BudgetConfig:
    """Budget configuration."""

    total_budget: int = 100000
    warning_threshold: float = 0.75
    critical_threshold: float = 0.90
    per_agent_limit: int | None = None
    per_step_limit: int | None = None
    reserve_tokens: int = 5000  # Reserved for retries/overhead
    daily_token_limit: int = 0  # 0 = disabled
    model_multipliers: dict[str, float] | None = None  # None → DEFAULT_MODEL_MULTIPLIERS
    # Effective-Tokens enforcement. As of the A1 "flip", ET is an ENFORCED
    # ceiling by default (not reporting-only). ``effective_token_budget``:
    #   - None  → derive the ceiling from ``total_budget`` (same number, but
    #             measured in cost-weighted ET; the worse of raw/ET governs)
    #   - float → explicit ET ceiling
    # Set ``enforce_effective_tokens=False`` to opt out and get pure raw-token
    # behavior (escape hatch; the default is enforce).
    effective_token_budget: float | None = None
    enforce_effective_tokens: bool = True


class BudgetManager:
    """Manages token budgets across workflow execution.

    Tracks usage, enforces limits, and provides allocation strategies.
    """

    def __init__(
        self,
        config: BudgetConfig = None,
        on_threshold_callback: Callable[[BudgetStatus, dict], None] = None,
        backend: DatabaseBackend | None = None,
        session_id: str | None = None,
    ):
        """Initialize budget manager.

        Args:
            config: Budget configuration
            on_threshold_callback: Called when budget thresholds are crossed
            backend: Optional DatabaseBackend for persistence
            session_id: Session identifier (required if backend is provided)
        """
        self.config = config or BudgetConfig()
        self._on_threshold = on_threshold_callback
        self._backend = backend
        self._session_id = session_id
        self._usage_history: list[UsageRecord] = []
        self._agent_usage: dict[str, int] = {}
        self._total_used: int = 0
        # Cost-weighted accumulator, maintained in lockstep with _total_used.
        # Governs enforcement via effective_ceiling (on by default post-flip).
        self._total_effective: float = 0.0
        self._last_status: BudgetStatus = BudgetStatus.OK

        if self._backend and self._session_id:
            self._restore_from_db()

    def _restore_from_db(self) -> None:
        """Restore usage state from the database.

        Restores both raw ``_total_used`` and the cost-weighted
        ``_total_effective`` (migration 020). Falls back to a raw-only restore
        if the ``effective_tokens`` column is absent (un-migrated DB), so an
        old database still restores raw usage rather than failing outright.
        """
        try:
            rows = self._backend.fetchall(
                "SELECT agent_id, SUM(tokens) as total, "
                "SUM(effective_tokens) as eff "
                "FROM budget_session_usage WHERE session_id = ? "
                "GROUP BY agent_id",
                (self._session_id,),
            )
            for row in rows:
                agent_id = row["agent_id"]
                tokens = int(row["total"])
                self._agent_usage[agent_id] = tokens
                self._total_used += tokens
                self._total_effective += float(row["eff"] or 0.0)
        except Exception:
            logger.warning("ET-aware budget restore failed; trying raw-only", exc_info=True)
            self._restore_from_db_raw_only()

    def _restore_from_db_raw_only(self) -> None:
        """Fallback restore for databases without the effective_tokens column."""
        try:
            rows = self._backend.fetchall(
                "SELECT agent_id, SUM(tokens) as total "
                "FROM budget_session_usage WHERE session_id = ? "
                "GROUP BY agent_id",
                (self._session_id,),
            )
            for row in rows:
                agent_id = row["agent_id"]
                tokens = int(row["total"])
                self._agent_usage[agent_id] = tokens
                self._total_used += tokens
                # No persisted ET: neutral fallback (ET == raw), consistent
                # with the no-breakdown rule.
                self._total_effective += float(tokens)
        except Exception:
            logger.warning("Failed to restore budget from DB", exc_info=True)

    def _persist_usage(
        self,
        agent_id: str,
        tokens: int,
        operation: str,
        effective: float = 0.0,
    ) -> None:
        """Persist a usage record (raw + cost-weighted) to the database."""
        try:
            self._backend.execute(
                "INSERT INTO budget_session_usage "
                "(session_id, agent_id, tokens, operation, effective_tokens) "
                "VALUES (?, ?, ?, ?, ?)",
                (self._session_id, agent_id, tokens, operation, effective),
            )
        except Exception:
            logger.warning("Failed to persist budget usage", exc_info=True)

    @property
    def total_budget(self) -> int:
        """Get total budget."""
        return self.config.total_budget

    @property
    def used(self) -> int:
        """Get total tokens used."""
        return self._total_used

    @property
    def remaining(self) -> int:
        """Get remaining tokens."""
        return max(0, self.config.total_budget - self._total_used)

    @property
    def available(self) -> int:
        """Get available tokens (excluding reserve)."""
        return max(0, self.remaining - self.config.reserve_tokens)

    @property
    def usage_percent(self) -> float:
        """Get usage as percentage."""
        if self.config.total_budget == 0:
            return 100.0
        return (self._total_used / self.config.total_budget) * 100

    @property
    def effective_used(self) -> float:
        """Cumulative cost-weighted Effective-Tokens consumed so far."""
        return self._total_effective

    @property
    def effective_ceiling(self) -> float:
        """The active Effective-Tokens ceiling.

        ``inf`` when enforcement is off. Otherwise the explicit
        ``effective_token_budget`` if set, else derived from ``total_budget``
        (the A1 default: the same budget number, enforced on the cost-weighted
        axis so the worse of raw/ET governs).
        """
        if not self.config.enforce_effective_tokens:
            return float("inf")
        if self.config.effective_token_budget is not None:
            return self.config.effective_token_budget
        return float(self.config.total_budget)

    @property
    def effective_remaining(self) -> float:
        """Remaining Effective-Tokens, or ``inf`` when ET enforcement is off."""
        ceiling = self.effective_ceiling
        if ceiling == float("inf"):
            return float("inf")
        return max(0.0, ceiling - self._total_effective)

    @property
    def status(self) -> BudgetStatus:
        """Get current budget status.

        Governed by the raw-token budget AND the cost-weighted Effective-Tokens
        ceiling (enforced by default; see ``effective_ceiling``). The
        more-constrained of the two ratios wins, so an output-heavy or
        opus-tier run that is cheap in raw tokens but expensive in real cost
        cannot read OK while overspending.
        """
        ratio = self._total_used / self.config.total_budget if self.config.total_budget > 0 else 1.0

        et_ceiling = self.effective_ceiling
        if et_ceiling not in (0.0, float("inf")):
            ratio = max(ratio, self._total_effective / et_ceiling)

        if ratio > 1.0:
            return BudgetStatus.EXCEEDED
        elif ratio > self.config.critical_threshold:
            return BudgetStatus.CRITICAL
        elif ratio > self.config.warning_threshold:
            return BudgetStatus.WARNING
        return BudgetStatus.OK

    def can_allocate(self, tokens: int, agent_id: str = None) -> bool:
        """Check if tokens can be allocated.

        Args:
            tokens: Number of tokens to allocate
            agent_id: Optional agent identifier for per-agent limits

        Returns:
            True if allocation is possible
        """
        # Check total budget
        if self._total_used + tokens > self.config.total_budget:
            return False

        # Check available (accounting for reserve)
        if tokens > self.available:
            return False

        # Check per-agent limit
        if agent_id and self.config.per_agent_limit:
            agent_total = self._agent_usage.get(agent_id, 0) + tokens
            if agent_total > self.config.per_agent_limit:
                return False

        # Check per-step limit
        if self.config.per_step_limit and tokens > self.config.per_step_limit:
            return False

        return True

    def allocate(self, tokens: int, agent_id: str = None) -> bool:
        """Attempt to allocate tokens.

        Args:
            tokens: Number of tokens to allocate
            agent_id: Optional agent identifier

        Returns:
            True if allocation succeeded
        """
        if not self.can_allocate(tokens, agent_id):
            return False

        # Record pending allocation (not yet recorded as used)
        return True

    def record_usage(
        self,
        agent_id: str,
        tokens: int = 0,
        operation: str = "",
        metadata: dict = None,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        model: str | None = None,
    ) -> UsageRecord:
        """Record actual token usage.

        Args:
            agent_id: Agent identifier.
            tokens: Rolled-up total. If 0 *and* a breakdown is supplied below,
                the breakdown sum is used.
            operation: Operation description.
            metadata: Additional metadata.
            input_tokens / output_tokens / cache_read_tokens: optional
                per-direction breakdown. When supplied, lets ``effective_tokens``
                / ``total_effective_tokens`` compute a cost-weighted score —
                cache reads count ~0.1× input, output ~4× input.
            model: optional model id (e.g. ``"claude-sonnet-4-6"``); used to
                pick a tier multiplier for Effective-Tokens.

        Returns:
            Usage record.
        """
        # If only the breakdown is supplied, derive the rolled-up total.
        if tokens == 0 and (input_tokens or output_tokens or cache_read_tokens):
            tokens = input_tokens + output_tokens + cache_read_tokens

        record = UsageRecord(
            agent_id=agent_id,
            tokens=tokens,
            operation=operation,
            metadata=metadata or {},
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            model=model,
        )

        record_effective = effective_tokens(record, self.config.model_multipliers)

        self._usage_history.append(record)
        self._total_used += tokens
        self._total_effective += record_effective
        self._agent_usage[agent_id] = self._agent_usage.get(agent_id, 0) + tokens

        # Persist to database if backend is available
        if self._backend and self._session_id:
            self._persist_usage(agent_id, tokens, operation, record_effective)

        # Check for status change
        new_status = self.status
        if new_status != self._last_status:
            self._last_status = new_status
            if self._on_threshold:
                self._on_threshold(
                    new_status,
                    {
                        "used": self._total_used,
                        "remaining": self.remaining,
                        "percent": self.usage_percent,
                        "agent_id": agent_id,
                    },
                )

        return record

    def get_agent_usage(self, agent_id: str) -> int:
        """Get total usage for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Total tokens used by agent
        """
        return self._agent_usage.get(agent_id, 0)

    def get_agent_remaining(self, agent_id: str) -> int | None:
        """Get remaining tokens for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Remaining tokens or None if no per-agent limit
        """
        if not self.config.per_agent_limit:
            return None
        used = self._agent_usage.get(agent_id, 0)
        return max(0, self.config.per_agent_limit - used)

    def total_effective_tokens(self) -> float:
        """Sum the Effective-Tokens score across the in-memory usage history.

        Useful for comparing workflows on a single cost axis across model
        tiers and input/cache/output mix — see ``effective_tokens``.
        """
        weights = self.config.model_multipliers
        return sum(effective_tokens(r, weights) for r in self._usage_history)

    def effective_tokens_by_agent(self) -> dict[str, float]:
        """Effective-Tokens score grouped by ``agent_id`` (one entry per
        agent observed in the in-memory history)."""
        weights = self.config.model_multipliers
        out: dict[str, float] = {}
        for r in self._usage_history:
            out[r.agent_id] = out.get(r.agent_id, 0.0) + effective_tokens(r, weights)
        return out

    def get_usage_history(
        self,
        agent_id: str = None,
        limit: int = 50,
    ) -> list[UsageRecord]:
        """Get usage history.

        Args:
            agent_id: Filter by agent (optional)
            limit: Maximum records to return

        Returns:
            List of usage records
        """
        records = self._usage_history
        if agent_id:
            records = [r for r in records if r.agent_id == agent_id]
        return records[-limit:]

    def get_stats(self) -> dict:
        """Get budget statistics.

        Returns:
            Dictionary with budget stats
        """
        return {
            "total_budget": self.config.total_budget,
            "used": self._total_used,
            "remaining": self.remaining,
            "available": self.available,
            "reserve": self.config.reserve_tokens,
            "percent_used": round(self.usage_percent, 1),
            "status": self.status.value,
            "total_operations": len(self._usage_history),
            "agents": {
                agent_id: {
                    "used": used,
                    "remaining": self.get_agent_remaining(agent_id),
                }
                for agent_id, used in self._agent_usage.items()
            },
        }

    def estimate_cost(self, tokens: int, model: str = "claude-3-opus") -> float:
        """Estimate cost for token usage.

        Args:
            tokens: Number of tokens
            model: Model name for pricing

        Returns:
            Estimated cost in USD
        """
        # Approximate pricing per 1M tokens (as of late 2024)
        pricing = {
            "claude-3-opus": {"input": 15.0, "output": 75.0},
            "claude-3-sonnet": {"input": 3.0, "output": 15.0},
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        }

        if model not in pricing:
            model = "claude-3-opus"  # Default to opus pricing

        # Assume 50/50 input/output split for estimation
        prices = pricing[model]
        avg_price = (prices["input"] + prices["output"]) / 2
        cost = (tokens / 1_000_000) * avg_price

        return round(cost, 4)

    def reset(self) -> None:
        """Reset budget tracking."""
        self._usage_history = []
        self._agent_usage = {}
        self._total_used = 0
        self._total_effective = 0.0
        self._last_status = BudgetStatus.OK

        if self._backend and self._session_id:
            try:
                self._backend.execute(
                    "DELETE FROM budget_session_usage WHERE session_id = ?",
                    (self._session_id,),
                )
            except Exception:
                logger.warning("Failed to clear budget from DB", exc_info=True)

    def get_budget_context(self) -> str:
        """Return a formatted budget constraint string for prompt injection.

        Returns empty string if budget is effectively unlimited (total_budget == 0).
        """
        if self.config.total_budget <= 0:
            return ""
        return (
            "[Budget Constraint]\n"
            f"Remaining session budget: {self.remaining:,} / "
            f"{self.config.total_budget:,} tokens.\n"
            "Be concise and efficient with token usage."
        )

    def set_budget(self, total_budget: int) -> None:
        """Update total budget.

        Args:
            total_budget: New total budget
        """
        self.config.total_budget = total_budget
