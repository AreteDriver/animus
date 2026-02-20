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
    """Record of token usage."""

    agent_id: str
    tokens: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    operation: str = ""
    metadata: dict = field(default_factory=dict)


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
        self._last_status: BudgetStatus = BudgetStatus.OK

        if self._backend and self._session_id:
            self._restore_from_db()

    def _restore_from_db(self) -> None:
        """Restore usage state from the database."""
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
        except Exception:
            logger.warning("Failed to restore budget from DB", exc_info=True)

    def _persist_usage(self, agent_id: str, tokens: int, operation: str) -> None:
        """Persist a usage record to the database."""
        try:
            self._backend.execute(
                "INSERT INTO budget_session_usage "
                "(session_id, agent_id, tokens, operation) "
                "VALUES (?, ?, ?, ?)",
                (self._session_id, agent_id, tokens, operation),
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
    def status(self) -> BudgetStatus:
        """Get current budget status."""
        ratio = self._total_used / self.config.total_budget if self.config.total_budget > 0 else 1.0

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
        tokens: int,
        operation: str = "",
        metadata: dict = None,
    ) -> UsageRecord:
        """Record actual token usage.

        Args:
            agent_id: Agent identifier
            tokens: Tokens consumed
            operation: Operation description
            metadata: Additional metadata

        Returns:
            Usage record
        """
        record = UsageRecord(
            agent_id=agent_id,
            tokens=tokens,
            operation=operation,
            metadata=metadata or {},
        )

        self._usage_history.append(record)
        self._total_used += tokens
        self._agent_usage[agent_id] = self._agent_usage.get(agent_id, 0) + tokens

        # Persist to database if backend is available
        if self._backend and self._session_id:
            self._persist_usage(agent_id, tokens, operation)

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
