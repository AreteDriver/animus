"""Cost tracking for API usage and workflow execution."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any


class Provider(Enum):
    """API providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GITHUB = "github"
    NOTION = "notion"
    SLACK = "slack"


@dataclass
class TokenUsage:
    """Token usage for a single API call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class CostEntry:
    """A single cost entry for tracking."""

    timestamp: datetime
    provider: Provider
    model: str
    tokens: TokenUsage
    cost_usd: float
    workflow_id: str | None = None
    step_id: str | None = None
    agent_role: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider.value,
            "model": self.model,
            "tokens": {
                "input": self.tokens.input_tokens,
                "output": self.tokens.output_tokens,
                "total": self.tokens.total_tokens,
            },
            "cost_usd": self.cost_usd,
            "workflow_id": self.workflow_id,
            "step_id": self.step_id,
            "agent_role": self.agent_role,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostEntry:
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            provider=Provider(data["provider"]),
            model=data["model"],
            tokens=TokenUsage(
                input_tokens=data["tokens"]["input"],
                output_tokens=data["tokens"]["output"],
                total_tokens=data["tokens"]["total"],
            ),
            cost_usd=data["cost_usd"],
            workflow_id=data.get("workflow_id"),
            step_id=data.get("step_id"),
            agent_role=data.get("agent_role"),
            metadata=data.get("metadata", {}),
        )


class CostTracker:
    """Tracks API costs across workflows and agents.

    Provides:
    - Per-call cost tracking
    - Aggregation by workflow, agent, model, and time period
    - Budget alerts and limits
    - Cost reporting and export
    """

    # Pricing per 1M tokens (as of 2024)
    PRICING = {
        # OpenAI
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        # Anthropic
        "claude-3-opus": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet": {"input": 3.00, "output": 15.00},
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    }

    def __init__(
        self,
        storage_path: Path | None = None,
        budget_limit_usd: float | None = None,
        alert_threshold_percent: float = 80.0,
    ):
        """Initialize cost tracker.

        Args:
            storage_path: Path to store cost data (JSON file)
            budget_limit_usd: Optional monthly budget limit in USD
            alert_threshold_percent: Percentage of budget to trigger alert
        """
        self.storage_path = storage_path
        self.budget_limit_usd = budget_limit_usd
        self.alert_threshold_percent = alert_threshold_percent
        self.entries: list[CostEntry] = []
        self._alerts: list[dict[str, Any]] = []

        if storage_path and storage_path.exists():
            self._load()

    def _load(self) -> None:
        """Load cost data from storage."""
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
            self.entries = [CostEntry.from_dict(e) for e in data.get("entries", [])]
        except Exception:
            self.entries = []

    def _save(self) -> None:
        """Save cost data to storage."""
        if not self.storage_path:
            return

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({"entries": [e.to_dict() for e in self.entries]}, f, indent=2)

    def calculate_cost(self, model: str, tokens: TokenUsage) -> float:
        """Calculate cost for token usage.

        Args:
            model: Model name
            tokens: Token usage

        Returns:
            Cost in USD
        """
        # Try exact match first, then prefix match
        pricing = self.PRICING.get(model)
        if not pricing:
            for model_prefix, model_pricing in self.PRICING.items():
                if model.startswith(model_prefix):
                    pricing = model_pricing
                    break

        if not pricing:
            # Default fallback pricing
            pricing = {"input": 1.00, "output": 2.00}

        input_cost = (tokens.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (tokens.output_tokens / 1_000_000) * pricing["output"]

        return round(input_cost + output_cost, 6)

    def track(
        self,
        provider: Provider,
        model: str,
        input_tokens: int,
        output_tokens: int,
        workflow_id: str | None = None,
        step_id: str | None = None,
        agent_role: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CostEntry:
        """Track a new API call.

        Args:
            provider: API provider
            model: Model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            workflow_id: Optional workflow ID
            step_id: Optional step ID
            agent_role: Optional agent role
            metadata: Optional additional metadata

        Returns:
            The created CostEntry
        """
        tokens = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
        cost = self.calculate_cost(model, tokens)

        entry = CostEntry(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            tokens=tokens,
            cost_usd=cost,
            workflow_id=workflow_id,
            step_id=step_id,
            agent_role=agent_role,
            metadata=metadata or {},
        )

        self.entries.append(entry)
        self._save()
        self._check_budget()

        return entry

    def _check_budget(self) -> None:
        """Check if budget threshold has been reached."""
        if not self.budget_limit_usd:
            return

        monthly_cost = self.get_monthly_cost()
        percent_used = (monthly_cost / self.budget_limit_usd) * 100

        if percent_used >= self.alert_threshold_percent:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "type": "budget_alert",
                "percent_used": percent_used,
                "monthly_cost": monthly_cost,
                "budget_limit": self.budget_limit_usd,
            }
            self._alerts.append(alert)

    def get_monthly_cost(self, year: int | None = None, month: int | None = None) -> float:
        """Get total cost for a month.

        Args:
            year: Year (defaults to current)
            month: Month (defaults to current)

        Returns:
            Total cost in USD
        """
        now = datetime.now()
        year = year or now.year
        month = month or now.month

        return sum(
            e.cost_usd
            for e in self.entries
            if e.timestamp.year == year and e.timestamp.month == month
        )

    def get_daily_cost(self, date: datetime | None = None) -> float:
        """Get total cost for a day.

        Args:
            date: Date (defaults to today)

        Returns:
            Total cost in USD
        """
        date = date or datetime.now()

        return sum(e.cost_usd for e in self.entries if e.timestamp.date() == date.date())

    def get_workflow_cost(self, workflow_id: str) -> dict[str, Any]:
        """Get cost breakdown for a workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            Cost breakdown with totals and per-step details
        """
        entries = [e for e in self.entries if e.workflow_id == workflow_id]

        by_step = defaultdict(lambda: {"cost": 0.0, "tokens": 0, "calls": 0})
        by_agent = defaultdict(lambda: {"cost": 0.0, "tokens": 0, "calls": 0})

        for e in entries:
            if e.step_id:
                by_step[e.step_id]["cost"] += e.cost_usd
                by_step[e.step_id]["tokens"] += e.tokens.total_tokens
                by_step[e.step_id]["calls"] += 1

            if e.agent_role:
                by_agent[e.agent_role]["cost"] += e.cost_usd
                by_agent[e.agent_role]["tokens"] += e.tokens.total_tokens
                by_agent[e.agent_role]["calls"] += 1

        return {
            "workflow_id": workflow_id,
            "total_cost": sum(e.cost_usd for e in entries),
            "total_tokens": sum(e.tokens.total_tokens for e in entries),
            "total_calls": len(entries),
            "by_step": dict(by_step),
            "by_agent": dict(by_agent),
        }

    def get_agent_costs(self, days: int = 30) -> dict[str, dict[str, float]]:
        """Get cost breakdown by agent role.

        Args:
            days: Number of days to include

        Returns:
            Cost breakdown by agent role
        """
        cutoff = datetime.now() - timedelta(days=days)
        entries = [e for e in self.entries if e.timestamp >= cutoff]

        by_agent = defaultdict(lambda: {"cost": 0.0, "tokens": 0, "calls": 0})

        for e in entries:
            role = e.agent_role or "unknown"
            by_agent[role]["cost"] += e.cost_usd
            by_agent[role]["tokens"] += e.tokens.total_tokens
            by_agent[role]["calls"] += 1

        return dict(by_agent)

    def get_model_costs(self, days: int = 30) -> dict[str, dict[str, float]]:
        """Get cost breakdown by model.

        Args:
            days: Number of days to include

        Returns:
            Cost breakdown by model
        """
        cutoff = datetime.now() - timedelta(days=days)
        entries = [e for e in self.entries if e.timestamp >= cutoff]

        by_model = defaultdict(lambda: {"cost": 0.0, "tokens": 0, "calls": 0})

        for e in entries:
            by_model[e.model]["cost"] += e.cost_usd
            by_model[e.model]["tokens"] += e.tokens.total_tokens
            by_model[e.model]["calls"] += 1

        return dict(by_model)

    def get_summary(self, days: int = 30) -> dict[str, Any]:
        """Get a summary of costs.

        Args:
            days: Number of days to include

        Returns:
            Cost summary
        """
        cutoff = datetime.now() - timedelta(days=days)
        entries = [e for e in self.entries if e.timestamp >= cutoff]

        total_cost = sum(e.cost_usd for e in entries)
        total_tokens = sum(e.tokens.total_tokens for e in entries)

        return {
            "period_days": days,
            "total_cost_usd": round(total_cost, 4),
            "total_tokens": total_tokens,
            "total_calls": len(entries),
            "avg_cost_per_call": round(total_cost / len(entries), 6) if entries else 0,
            "avg_tokens_per_call": total_tokens // len(entries) if entries else 0,
            "by_provider": self._group_by_provider(entries),
            "by_model": self.get_model_costs(days),
            "by_agent": self.get_agent_costs(days),
            "budget": {
                "limit_usd": self.budget_limit_usd,
                "monthly_used_usd": self.get_monthly_cost(),
                "percent_used": (
                    (self.get_monthly_cost() / self.budget_limit_usd * 100)
                    if self.budget_limit_usd
                    else None
                ),
            },
            "alerts": self._alerts[-10:],  # Last 10 alerts
        }

    def _group_by_provider(self, entries: list[CostEntry]) -> dict[str, dict[str, float]]:
        """Group entries by provider."""
        by_provider = defaultdict(lambda: {"cost": 0.0, "tokens": 0, "calls": 0})

        for e in entries:
            by_provider[e.provider.value]["cost"] += e.cost_usd
            by_provider[e.provider.value]["tokens"] += e.tokens.total_tokens
            by_provider[e.provider.value]["calls"] += 1

        return dict(by_provider)

    def export_csv(self, filepath: Path, days: int | None = None) -> None:
        """Export cost data to CSV.

        Args:
            filepath: Output file path
            days: Optional number of days to include
        """
        import csv

        entries = self.entries
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            entries = [e for e in entries if e.timestamp >= cutoff]

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "provider",
                    "model",
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "cost_usd",
                    "workflow_id",
                    "step_id",
                    "agent_role",
                ]
            )

            for e in entries:
                writer.writerow(
                    [
                        e.timestamp.isoformat(),
                        e.provider.value,
                        e.model,
                        e.tokens.input_tokens,
                        e.tokens.output_tokens,
                        e.tokens.total_tokens,
                        e.cost_usd,
                        e.workflow_id or "",
                        e.step_id or "",
                        e.agent_role or "",
                    ]
                )

    def clear_old_entries(self, days: int = 90) -> int:
        """Clear entries older than specified days.

        Args:
            days: Number of days to keep

        Returns:
            Number of entries removed
        """
        cutoff = datetime.now() - timedelta(days=days)
        original_count = len(self.entries)
        self.entries = [e for e in self.entries if e.timestamp >= cutoff]
        removed = original_count - len(self.entries)
        self._save()
        return removed


# Global tracker instance
_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = CostTracker()
    return _tracker


def initialize_cost_tracker(
    storage_path: Path | None = None,
    budget_limit_usd: float | None = None,
) -> CostTracker:
    """Initialize the global cost tracker.

    Args:
        storage_path: Path to store cost data
        budget_limit_usd: Optional monthly budget limit

    Returns:
        The initialized CostTracker
    """
    global _tracker
    _tracker = CostTracker(storage_path=storage_path, budget_limit_usd=budget_limit_usd)
    return _tracker
