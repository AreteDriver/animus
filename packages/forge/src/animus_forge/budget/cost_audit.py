"""Cost-audit anomaly detector for Effective-Tokens spend.

Lifted from the GitHub Agentic Workflows token-efficiency pattern §4
(see `docs/patterns/token-optimization-from-github-2026-05.md`):
GitHub runs a *Daily Token Usage Auditor* over their per-run JSONL
log, flagging anomalous workflows on a sigma threshold against a
trailing baseline. This is the Forge analogue, operating on the
``BudgetManager.get_usage_history()`` stream rather than a separate
JSONL — Forge already has structured spend records.

Pure-logic module; no I/O. Inputs are passed in, results are returned
as a dataclass. The matching step handler in
``workflow/executor_cost_audit.py`` is the thin wrapper that wires
this into a workflow run.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from animus_forge.budget.manager import UsageRecord

from animus_forge.budget.manager import effective_tokens

DEFAULT_SIGMA_THRESHOLD = 2.0
DEFAULT_RATIO_THRESHOLD = 1.5
DEFAULT_BASELINE_DAYS = 7
DEFAULT_WINDOW_HOURS = 24


@dataclass
class CostAnomaly:
    """One agent's ET spend judged anomalous against its baseline."""

    agent_id: str
    window_et: float
    baseline_mean: float
    baseline_stdev: float
    ratio: float  # window_et / baseline_mean (∞ if baseline is 0)
    sigma: float  # (window_et - baseline_mean) / baseline_stdev (0 if stdev is 0)
    reason: str  # "sigma_exceeded" | "ratio_exceeded" | "both"


@dataclass
class CostAuditReport:
    """Output of one cost-audit pass."""

    window_start: datetime
    window_end: datetime
    baseline_start: datetime
    baseline_end: datetime
    window_total_et: float = 0.0
    baseline_total_et: float = 0.0
    window_by_agent: dict[str, float] = field(default_factory=dict)
    baseline_by_agent: dict[str, float] = field(default_factory=dict)
    anomalies: list[CostAnomaly] = field(default_factory=list)

    @property
    def has_anomalies(self) -> bool:
        return bool(self.anomalies)

    def to_dict(self) -> dict:
        """Workflow-step-handler-friendly serialization."""
        return {
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "baseline_start": self.baseline_start.isoformat(),
            "baseline_end": self.baseline_end.isoformat(),
            "window_total_et": self.window_total_et,
            "baseline_total_et": self.baseline_total_et,
            "window_by_agent": self.window_by_agent,
            "baseline_by_agent": self.baseline_by_agent,
            "anomalies": [
                {
                    "agent_id": a.agent_id,
                    "window_et": a.window_et,
                    "baseline_mean": a.baseline_mean,
                    "baseline_stdev": a.baseline_stdev,
                    "ratio": a.ratio,
                    "sigma": a.sigma,
                    "reason": a.reason,
                }
                for a in self.anomalies
            ],
            "anomaly_count": len(self.anomalies),
        }


def _et_by_agent(
    records: Iterable[UsageRecord],
    model_multipliers: dict[str, float] | None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in records:
        out[r.agent_id] = out.get(r.agent_id, 0.0) + effective_tokens(r, model_multipliers)
    return out


def _records_in(
    records: list[UsageRecord],
    start: datetime,
    end: datetime,
) -> list[UsageRecord]:
    return [r for r in records if start <= r.timestamp < end]


def _bucket_by_day(
    records: list[UsageRecord],
    model_multipliers: dict[str, float] | None,
) -> dict[str, list[float]]:
    """Per-agent daily ET totals — the unit of variance for the baseline."""
    buckets: dict[str, dict[str, float]] = {}
    for r in records:
        day = r.timestamp.date().isoformat()
        et = effective_tokens(r, model_multipliers)
        buckets.setdefault(r.agent_id, {})
        buckets[r.agent_id][day] = buckets[r.agent_id].get(day, 0.0) + et
    return {agent: list(daily.values()) for agent, daily in buckets.items()}


def _mean_stdev(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def audit_cost(
    history: list[UsageRecord],
    now: datetime | None = None,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    baseline_days: int = DEFAULT_BASELINE_DAYS,
    sigma_threshold: float = DEFAULT_SIGMA_THRESHOLD,
    ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
    model_multipliers: dict[str, float] | None = None,
) -> CostAuditReport:
    """Run a single cost-audit pass over a usage-record stream.

    Args:
        history: Full usage history (e.g. ``BudgetManager.get_usage_history()``
            called without filters). Records older than the baseline window
            are ignored.
        now: Anchor "right now" for the window. Defaults to the timestamp of
            the most recent record (or current UTC if history is empty),
            chosen because tests + replay want determinism, not wall clock.
        window_hours: Size of the current-window analysis bucket. Defaults
            to 24 hours (the GitHub auditor cadence).
        baseline_days: Size of the trailing baseline immediately before the
            window. Defaults to 7 days.
        sigma_threshold: Flag if (window_et - baseline_mean) / stdev exceeds
            this. Defaults to 2σ.
        ratio_threshold: Flag if window_et / baseline_mean exceeds this.
            Defaults to 1.5× — protects against the stdev=0 edge case where
            sigma is undefined but spend obviously jumped.
        model_multipliers: Per-model ET multipliers (forwarded to
            ``effective_tokens``). When omitted, defaults from
            ``DEFAULT_MODEL_MULTIPLIERS`` apply.

    Returns:
        CostAuditReport. Empty history → empty report (no error).
    """
    if not history:
        # C1-12: tz-aware to match UsageRecord.timestamp (datetime.now(UTC));
        # a naive anchor would TypeError if later compared to record timestamps.
        anchor = now or datetime.now(UTC)
        return CostAuditReport(
            window_start=anchor - timedelta(hours=window_hours),
            window_end=anchor,
            baseline_start=anchor - timedelta(hours=window_hours) - timedelta(days=baseline_days),
            baseline_end=anchor - timedelta(hours=window_hours),
        )

    if now is None:
        now = max(r.timestamp for r in history)

    window_start = now - timedelta(hours=window_hours)
    baseline_start = window_start - timedelta(days=baseline_days)

    window_records = _records_in(history, window_start, now + timedelta(microseconds=1))
    baseline_records = _records_in(history, baseline_start, window_start)

    window_by_agent = _et_by_agent(window_records, model_multipliers)
    baseline_by_agent = _et_by_agent(baseline_records, model_multipliers)
    baseline_daily = _bucket_by_day(baseline_records, model_multipliers)

    # window_hours can be != 24; normalize comparison to a per-day rate
    window_rate_multiplier = 24.0 / max(window_hours, 1)

    anomalies: list[CostAnomaly] = []
    seen_agents = set(window_by_agent) | set(baseline_by_agent)
    for agent in sorted(seen_agents):
        window_et = window_by_agent.get(agent, 0.0)
        window_daily_rate = window_et * window_rate_multiplier
        daily_samples = baseline_daily.get(agent, [])
        baseline_mean, baseline_stdev = _mean_stdev(daily_samples)

        if baseline_mean == 0.0 and window_daily_rate == 0.0:
            continue  # no spend either window — nothing to flag

        ratio = (
            float("inf")
            if baseline_mean == 0.0 and window_daily_rate > 0.0
            else (window_daily_rate / baseline_mean if baseline_mean else 0.0)
        )
        sigma = (window_daily_rate - baseline_mean) / baseline_stdev if baseline_stdev > 0 else 0.0

        triggered = []
        if baseline_stdev > 0 and sigma >= sigma_threshold:
            triggered.append("sigma_exceeded")
        if baseline_mean > 0 and ratio >= ratio_threshold:
            triggered.append("ratio_exceeded")
        # Cold-start: no baseline at all, but spend appeared in window.
        if baseline_mean == 0.0 and window_daily_rate > 0.0:
            triggered.append("cold_start_spend")

        if triggered:
            anomalies.append(
                CostAnomaly(
                    agent_id=agent,
                    window_et=window_et,
                    baseline_mean=baseline_mean,
                    baseline_stdev=baseline_stdev,
                    ratio=ratio,
                    sigma=sigma,
                    reason="+".join(triggered),
                )
            )

    return CostAuditReport(
        window_start=window_start,
        window_end=now,
        baseline_start=baseline_start,
        baseline_end=window_start,
        window_total_et=sum(window_by_agent.values()),
        baseline_total_et=sum(baseline_by_agent.values()),
        window_by_agent=window_by_agent,
        baseline_by_agent=baseline_by_agent,
        anomalies=anomalies,
    )
