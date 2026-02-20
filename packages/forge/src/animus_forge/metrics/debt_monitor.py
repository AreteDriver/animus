"""Technical debt monitoring and system drift detection.

Implements Zorya Polunochnaya's auditing role: scheduled self-audits,
drift detection, technical debt tracking, and remediation recommendations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ..state.backends import DatabaseBackend

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


class AuditFrequency(Enum):
    """How often an audit check should run."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

    @property
    def seconds(self) -> int:
        """Return the frequency in seconds."""
        return {
            "hourly": 3600,
            "daily": 86400,
            "weekly": 604800,
            "monthly": 2592000,
        }[self.value]


class AuditStatus(Enum):
    """Result status of an audit check."""

    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


class DebtSeverity(Enum):
    """Severity level of a technical debt item."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DebtStatus(Enum):
    """Status of a technical debt item."""

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"


class DebtSource(Enum):
    """How the debt was detected."""

    AUDIT = "audit"
    MANUAL = "manual"
    INCIDENT = "incident"


@dataclass
class AuditCheck:
    """Definition of a single audit check."""

    name: str
    category: str
    frequency: AuditFrequency
    check_function: str
    threshold_warning: float
    threshold_critical: float
    last_run: datetime | None = None
    last_result: dict[str, Any] | None = None


@dataclass
class AuditResult:
    """Result of running an audit check."""

    check_name: str
    category: str
    status: AuditStatus
    value: float
    baseline: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    recommendation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "check_name": self.check_name,
            "category": self.category,
            "status": self.status.value,
            "value": self.value,
            "baseline": self.baseline,
            "details": self.details,
            "recommendation": self.recommendation,
        }


@dataclass
class TechnicalDebt:
    """A tracked technical debt item."""

    id: str
    category: str
    severity: DebtSeverity
    title: str
    description: str
    detected_at: datetime
    source: DebtSource
    estimated_effort: str | None = None
    status: DebtStatus = DebtStatus.OPEN
    resolution: str | None = None
    resolved_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "category": self.category,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
            "source": self.source.value,
            "estimated_effort": self.estimated_effort,
            "status": self.status.value,
            "resolution": self.resolution,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


@dataclass
class SystemBaseline:
    """Captured system state used as a reference for drift detection."""

    captured_at: datetime
    task_completion_time_avg: float
    agent_spawn_time_avg: float
    idle_cpu_percent: float
    idle_memory_percent: float
    skill_hashes: dict[str, str] = field(default_factory=dict)
    config_snapshots: dict[str, dict] = field(default_factory=dict)
    package_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "captured_at": self.captured_at.isoformat(),
            "task_completion_time_avg": self.task_completion_time_avg,
            "agent_spawn_time_avg": self.agent_spawn_time_avg,
            "idle_cpu_percent": self.idle_cpu_percent,
            "idle_memory_percent": self.idle_memory_percent,
            "skill_hashes": self.skill_hashes,
            "config_snapshots": self.config_snapshots,
            "package_versions": self.package_versions,
        }


# Default audit checks
DEFAULT_AUDIT_CHECKS: list[AuditCheck] = [
    AuditCheck(
        name="task_completion_time",
        category="performance",
        frequency=AuditFrequency.DAILY,
        check_function="check_task_completion_time",
        threshold_warning=1.2,
        threshold_critical=1.5,
    ),
    AuditCheck(
        name="error_rate",
        category="reliability",
        frequency=AuditFrequency.DAILY,
        check_function="check_error_rate",
        threshold_warning=0.05,
        threshold_critical=0.10,
    ),
    AuditCheck(
        name="dependency_versions",
        category="dependencies",
        frequency=AuditFrequency.WEEKLY,
        check_function="check_dependencies",
        threshold_warning=5,
        threshold_critical=10,
    ),
    AuditCheck(
        name="skill_integrity",
        category="skills",
        frequency=AuditFrequency.DAILY,
        check_function="check_skill_integrity",
        threshold_warning=1,
        threshold_critical=3,
    ),
    AuditCheck(
        name="config_drift",
        category="configuration",
        frequency=AuditFrequency.DAILY,
        check_function="check_config_drift",
        threshold_warning=1,
        threshold_critical=3,
    ),
    AuditCheck(
        name="resource_baseline",
        category="resources",
        frequency=AuditFrequency.HOURLY,
        check_function="check_resource_usage",
        threshold_warning=1.3,
        threshold_critical=1.5,
    ),
]


class TechnicalDebtRegistry:
    """Manages the technical debt registry backed by the database."""

    def __init__(self, backend: DatabaseBackend) -> None:
        self.backend = backend

    def register(self, debt: TechnicalDebt) -> None:
        """Register a new technical debt item.

        Args:
            debt: The technical debt item to register.
        """
        with self.backend.transaction():
            self.backend.execute(
                """INSERT OR IGNORE INTO technical_debt
                   (id, category, severity, title, description,
                    detected_at, source, estimated_effort, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    debt.id,
                    debt.category,
                    debt.severity.value,
                    debt.title,
                    debt.description,
                    debt.detected_at.isoformat(),
                    debt.source.value,
                    debt.estimated_effort,
                    debt.status.value,
                ),
            )

    def resolve(self, debt_id: str, resolution: str) -> None:
        """Mark a debt item as resolved.

        Args:
            debt_id: ID of the debt item.
            resolution: Description of the resolution.
        """
        now = datetime.now(UTC).isoformat()
        with self.backend.transaction():
            self.backend.execute(
                """UPDATE technical_debt
                   SET status = ?, resolution = ?, resolved_at = ?
                   WHERE id = ?""",
                (DebtStatus.RESOLVED.value, resolution, now, debt_id),
            )

    def update_status(self, debt_id: str, status: DebtStatus) -> None:
        """Update the status of a debt item.

        Args:
            debt_id: ID of the debt item.
            status: New status.
        """
        with self.backend.transaction():
            self.backend.execute(
                "UPDATE technical_debt SET status = ? WHERE id = ?",
                (status.value, debt_id),
            )

    def list_open(self) -> list[dict[str, Any]]:
        """List all open technical debt items.

        Returns:
            List of debt items as dictionaries.
        """
        return self.backend.fetchall(
            """SELECT * FROM technical_debt
               WHERE status IN ('open', 'acknowledged', 'in_progress')
               ORDER BY
                 CASE severity
                   WHEN 'critical' THEN 0
                   WHEN 'high' THEN 1
                   WHEN 'medium' THEN 2
                   WHEN 'low' THEN 3
                 END,
                 detected_at DESC"""
        )

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of technical debt.

        Returns:
            Summary with counts by severity and category.
        """
        rows = self.backend.fetchall(
            """SELECT severity, COUNT(*) as count
               FROM technical_debt
               WHERE status != 'resolved'
               GROUP BY severity"""
        )
        by_severity = {r["severity"]: r["count"] for r in rows}

        rows = self.backend.fetchall(
            """SELECT category, COUNT(*) as count
               FROM technical_debt
               WHERE status != 'resolved'
               GROUP BY category"""
        )
        by_category = {r["category"]: r["count"] for r in rows}

        total = sum(by_severity.values())
        return {
            "total_open": total,
            "by_severity": by_severity,
            "by_category": by_category,
        }


class SystemAuditor:
    """Runs scheduled audit checks and detects system drift.

    This is Zorya Polunochnaya's technical debt and drift detection system.
    It compares current system metrics against a captured baseline and
    registers any detected issues into the technical debt registry.
    """

    def __init__(
        self,
        backend: DatabaseBackend,
        baseline: SystemBaseline | None = None,
        checks: list[AuditCheck] | None = None,
        skills_path: Path | None = None,
        config_path: Path | None = None,
    ) -> None:
        self.backend = backend
        self.baseline = baseline
        self.checks = checks or list(DEFAULT_AUDIT_CHECKS)
        self.debt_registry = TechnicalDebtRegistry(backend)
        self.skills_path = skills_path or Path.home() / ".gorgon" / "skills"
        self.config_path = config_path or Path.home() / ".gorgon" / "config"
        self._check_functions: dict[str, Callable[..., Coroutine[Any, Any, AuditResult]]] = {}

    def register_check(
        self,
        name: str,
        fn: Callable[..., Coroutine[Any, Any, AuditResult]],
    ) -> None:
        """Register a custom audit check function.

        Args:
            name: The check name matching AuditCheck.check_function.
            fn: Async callable that returns an AuditResult.
        """
        self._check_functions[name] = fn

    def evaluate_threshold(
        self,
        value: float,
        warning: float,
        critical: float,
    ) -> AuditStatus:
        """Evaluate a value against warning and critical thresholds.

        Args:
            value: The measured value.
            warning: Warning threshold.
            critical: Critical threshold.

        Returns:
            The resulting audit status.
        """
        if value >= critical:
            return AuditStatus.CRITICAL
        if value >= warning:
            return AuditStatus.WARNING
        return AuditStatus.OK

    def _is_due(self, check: AuditCheck, now: datetime) -> bool:
        """Determine if an audit check is due to run.

        Args:
            check: The audit check definition.
            now: Current time.

        Returns:
            True if the check should run now.
        """
        if check.last_run is None:
            return True
        elapsed = (now - check.last_run).total_seconds()
        return elapsed >= check.frequency.seconds

    async def run_scheduled_audits(self) -> list[AuditResult]:
        """Run all audit checks that are due.

        Returns:
            List of audit results for checks that ran.
        """
        now = datetime.now(UTC)
        results: list[AuditResult] = []

        for check in self.checks:
            if not self._is_due(check, now):
                continue

            fn = self._check_functions.get(check.check_function)
            if fn is None:
                logger.warning(
                    "No implementation for audit check '%s'",
                    check.check_function,
                )
                continue

            try:
                result = await fn(check)
                check.last_run = now
                check.last_result = result.to_dict()
                results.append(result)

                # Persist the result
                self._store_result(result)

                # Auto-register debt for non-OK results
                if result.status != AuditStatus.OK:
                    self._register_debt_from_result(result)

            except Exception:
                logger.exception("Audit check '%s' failed", check.check_function)

        return results

    async def run_all_checks(self) -> list[AuditResult]:
        """Run all registered checks regardless of schedule.

        Returns:
            List of all audit results.
        """
        now = datetime.now(UTC)
        results: list[AuditResult] = []

        for check in self.checks:
            fn = self._check_functions.get(check.check_function)
            if fn is None:
                continue

            try:
                result = await fn(check)
                check.last_run = now
                check.last_result = result.to_dict()
                results.append(result)
                self._store_result(result)

                if result.status != AuditStatus.OK:
                    self._register_debt_from_result(result)
            except Exception:
                logger.exception("Audit check '%s' failed", check.check_function)

        return results

    def _store_result(self, result: AuditResult) -> None:
        """Persist an audit result to the database."""
        with self.backend.transaction():
            self.backend.execute(
                """INSERT INTO audit_results (check_name, category, status, result_data)
                   VALUES (?, ?, ?, ?)""",
                (
                    result.check_name,
                    result.category,
                    result.status.value,
                    json.dumps(result.to_dict()),
                ),
            )

    def _register_debt_from_result(self, result: AuditResult) -> None:
        """Register a technical debt item from an audit result."""
        severity = (
            DebtSeverity.CRITICAL if result.status == AuditStatus.CRITICAL else DebtSeverity.HIGH
        )
        debt = TechnicalDebt(
            id=f"audit-{result.check_name}-{uuid.uuid4().hex[:8]}",
            category=result.category,
            severity=severity,
            title=f"{result.check_name} threshold exceeded",
            description=json.dumps(result.to_dict()),
            detected_at=datetime.now(UTC),
            source=DebtSource.AUDIT,
            estimated_effort=self._estimate_effort(result),
            status=DebtStatus.OPEN,
        )
        self.debt_registry.register(debt)

    def _estimate_effort(self, result: AuditResult) -> str:
        """Estimate remediation effort from an audit result."""
        effort_map: dict[str, str] = {
            "performance": "2h",
            "reliability": "4h",
            "dependencies": "30m",
            "skills": "1h",
            "configuration": "30m",
            "resources": "1h",
        }
        return effort_map.get(result.category, "1h")

    def get_audit_history(
        self,
        check_name: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Retrieve past audit results.

        Args:
            check_name: Filter by check name. None returns all.
            limit: Maximum results to return.

        Returns:
            List of audit result records.
        """
        if check_name:
            return self.backend.fetchall(
                """SELECT * FROM audit_results
                   WHERE check_name = ?
                   ORDER BY run_at DESC LIMIT ?""",
                (check_name, limit),
            )
        return self.backend.fetchall(
            "SELECT * FROM audit_results ORDER BY run_at DESC LIMIT ?",
            (limit,),
        )

    def generate_report(self, results: list[AuditResult]) -> str:
        """Generate a human-readable audit report.

        Args:
            results: List of audit results to include.

        Returns:
            Formatted report string.
        """
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        lines: list[str] = [
            "=" * 63,
            "  GORGON SYSTEM AUDIT - Zorya Polunochnaya",
            f"  {now}",
            "=" * 63,
            "",
        ]

        # Group by category
        by_category: dict[str, list[AuditResult]] = {}
        for r in results:
            by_category.setdefault(r.category, []).append(r)

        status_icons = {
            AuditStatus.OK: "OK",
            AuditStatus.WARNING: "WARNING",
            AuditStatus.CRITICAL: "CRITICAL",
        }

        warnings = 0
        criticals = 0

        for category, cat_results in sorted(by_category.items()):
            lines.append(f"  {category.upper()}")
            for r in cat_results:
                icon = status_icons[r.status]
                lines.append(f"  - {r.check_name}: [{icon}]")
                if r.baseline is not None:
                    lines.append(f"    Baseline: {r.baseline:.2f} -> Current: {r.value:.2f}")
                if r.recommendation:
                    lines.append(f"    Recommendation: {r.recommendation}")

                if r.status == AuditStatus.WARNING:
                    warnings += 1
                elif r.status == AuditStatus.CRITICAL:
                    criticals += 1
            lines.append("")

        lines.append("-" * 63)
        lines.append(f"  SUMMARY: {warnings} warning(s), {criticals} critical(s)")
        lines.append("=" * 63)

        return "\n".join(lines)


def capture_baseline(
    skills_path: Path | None = None,
    config_path: Path | None = None,
    task_completion_time: float = 0.0,
    agent_spawn_time: float = 0.0,
) -> SystemBaseline:
    """Capture the current system state as a baseline.

    Args:
        skills_path: Path to skill files directory.
        config_path: Path to config files directory.
        task_completion_time: Average task completion time in seconds.
        agent_spawn_time: Average agent spawn time in seconds.

    Returns:
        A SystemBaseline snapshot.
    """
    skills_path = skills_path or Path.home() / ".gorgon" / "skills"
    config_path = config_path or Path.home() / ".gorgon" / "config"

    # Hash skill files
    skill_hashes: dict[str, str] = {}
    if skills_path.exists():
        for skill_file in skills_path.rglob("*.md"):
            if skill_file.stat().st_size > MAX_FILE_SIZE:
                logger.warning("Skipping oversized file: %s", skill_file)
                continue
            skill_hashes[str(skill_file)] = hashlib.sha256(skill_file.read_bytes()).hexdigest()

    # Snapshot config files
    config_snapshots: dict[str, dict] = {}
    if config_path.exists():
        for config_file in config_path.glob("*.yaml"):
            if config_file.stat().st_size > MAX_FILE_SIZE:
                logger.warning("Skipping oversized config: %s", config_file)
                continue
            try:
                import yaml

                config_snapshots[config_file.name] = yaml.safe_load(config_file.read_text())
            except Exception:
                logger.warning("Could not parse config file %s", config_file.name)

    # Capture package versions
    package_versions: dict[str, str] = {}
    try:
        import importlib.metadata

        for dist in importlib.metadata.distributions():
            package_versions[dist.metadata["Name"]] = dist.metadata["Version"]
    except Exception:
        logger.warning("Could not capture package versions")

    # Resource usage
    cpu_percent = 0.0
    memory_percent = 0.0
    try:
        import psutil

        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
    except ImportError:
        logger.info("psutil not available, skipping resource baseline")

    return SystemBaseline(
        captured_at=datetime.now(UTC),
        task_completion_time_avg=task_completion_time,
        agent_spawn_time_avg=agent_spawn_time,
        idle_cpu_percent=cpu_percent,
        idle_memory_percent=memory_percent,
        skill_hashes=skill_hashes,
        config_snapshots=config_snapshots,
        package_versions=package_versions,
    )


def save_baseline(backend: DatabaseBackend, baseline: SystemBaseline) -> None:
    """Save a baseline to the database.

    Args:
        backend: Database backend.
        baseline: The baseline to persist.
    """
    with backend.transaction():
        # Deactivate previous baselines
        backend.execute("UPDATE audit_baselines SET is_active = 0")

        backend.execute(
            """INSERT INTO audit_baselines
               (captured_at, task_completion_time_avg, agent_spawn_time_avg,
                idle_cpu_percent, idle_memory_percent,
                skill_hashes, config_snapshots, package_versions)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                baseline.captured_at.isoformat(),
                baseline.task_completion_time_avg,
                baseline.agent_spawn_time_avg,
                baseline.idle_cpu_percent,
                baseline.idle_memory_percent,
                json.dumps(baseline.skill_hashes),
                json.dumps(baseline.config_snapshots),
                json.dumps(baseline.package_versions),
            ),
        )


def load_active_baseline(backend: DatabaseBackend) -> SystemBaseline | None:
    """Load the currently active baseline from the database.

    Args:
        backend: Database backend.

    Returns:
        The active SystemBaseline or None if no baseline exists.
    """
    row = backend.fetchone(
        "SELECT * FROM audit_baselines WHERE is_active = 1 ORDER BY id DESC LIMIT 1"
    )
    if not row:
        return None

    return SystemBaseline(
        captured_at=datetime.fromisoformat(row["captured_at"]),
        task_completion_time_avg=row["task_completion_time_avg"],
        agent_spawn_time_avg=row["agent_spawn_time_avg"],
        idle_cpu_percent=row["idle_cpu_percent"],
        idle_memory_percent=row["idle_memory_percent"],
        skill_hashes=json.loads(row["skill_hashes"]) if row["skill_hashes"] else {},
        config_snapshots=json.loads(row["config_snapshots"]) if row["config_snapshots"] else {},
        package_versions=json.loads(row["package_versions"]) if row["package_versions"] else {},
    )
