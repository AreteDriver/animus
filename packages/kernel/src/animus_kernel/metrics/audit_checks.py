"""Built-in audit check implementations for system drift detection.

Each check function takes an AuditCheck and returns an AuditResult.
These are registered with the SystemAuditor at startup.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from ..state.backends import DatabaseBackend
from .debt_monitor import (
    AuditCheck,
    AuditResult,
    AuditStatus,
    SystemAuditor,
    SystemBaseline,
)

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def _evaluate(
    value: float,
    warning: float,
    critical: float,
) -> AuditStatus:
    """Evaluate a value against thresholds."""
    if value >= critical:
        return AuditStatus.CRITICAL
    if value >= warning:
        return AuditStatus.WARNING
    return AuditStatus.OK


async def check_task_completion_time(
    check: AuditCheck,
    *,
    backend: DatabaseBackend,
    baseline: SystemBaseline,
) -> AuditResult:
    """Check if average task completion time has drifted from baseline.

    Compares the rolling 7-day average against the baseline value.
    """
    row = backend.fetchone(
        """SELECT AVG(
             CAST(
               (julianday(ended_at) - julianday(started_at)) * 86400
             AS REAL)
           ) as avg_duration
           FROM jobs
           WHERE ended_at IS NOT NULL
             AND status = 'completed'
             AND ended_at > datetime('now', '-7 days')"""
    )

    current_avg = (row or {}).get("avg_duration") or 0.0
    baseline_avg = baseline.task_completion_time_avg

    if baseline_avg <= 0:
        return AuditResult(
            check_name=check.name,
            category=check.category,
            status=AuditStatus.OK,
            value=current_avg,
            baseline=0.0,
            details={"note": "No baseline set for task completion time"},
        )

    ratio = current_avg / baseline_avg
    status = _evaluate(ratio, check.threshold_warning, check.threshold_critical)

    recommendation = None
    if status != AuditStatus.OK:
        pct = int((ratio - 1) * 100)
        recommendation = (
            f"Task completion time is {pct}% above baseline. "
            "Review recent skill or workflow changes."
        )

    return AuditResult(
        check_name=check.name,
        category=check.category,
        status=status,
        value=current_avg,
        baseline=baseline_avg,
        details={"ratio": ratio},
        recommendation=recommendation,
    )


async def check_error_rate(
    check: AuditCheck,
    *,
    backend: DatabaseBackend,
    **_: Any,
) -> AuditResult:
    """Check the 7-day error rate for agent runs."""
    row = backend.fetchone(
        """SELECT
             COUNT(*) as total,
             SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failures
           FROM jobs
           WHERE created_at > datetime('now', '-7 days')"""
    )

    total = (row or {}).get("total") or 0
    failures = (row or {}).get("failures") or 0
    rate = failures / max(total, 1)

    status = _evaluate(rate, check.threshold_warning, check.threshold_critical)

    recommendation = None
    if status != AuditStatus.OK:
        recommendation = f"Error rate is {rate:.1%}. Investigate top failure reasons in job logs."

    return AuditResult(
        check_name=check.name,
        category=check.category,
        status=status,
        value=rate,
        details={"total": total, "failures": failures},
        recommendation=recommendation,
    )


async def check_dependencies(
    check: AuditCheck,
    **_: Any,
) -> AuditResult:
    """Check for outdated Python packages."""
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["pip", "list", "--outdated", "--format=json"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        outdated = json.loads(result.stdout) if result.returncode == 0 else []
    except Exception:
        logger.warning("Could not check outdated packages")
        outdated = []

    count = len(outdated)
    status = _evaluate(count, check.threshold_warning, check.threshold_critical)

    recommendation = None
    if outdated:
        names = [p["name"] for p in outdated[:5]]
        recommendation = f"Outdated packages: {', '.join(names)}"

    return AuditResult(
        check_name=check.name,
        category=check.category,
        status=status,
        value=float(count),
        details={"outdated": outdated[:10]},
        recommendation=recommendation,
    )


async def check_skill_integrity(
    check: AuditCheck,
    *,
    baseline: SystemBaseline,
    skills_path: Path | None = None,
    **_: Any,
) -> AuditResult:
    """Check skill file hashes against the baseline."""
    skills_path = skills_path or Path.home() / ".gorgon" / "skills"
    modified: list[dict[str, str]] = []

    if skills_path.exists() and baseline.skill_hashes:
        for file_path_str, expected_hash in baseline.skill_hashes.items():
            path = Path(file_path_str)
            if not path.exists():
                modified.append({"file": file_path_str, "issue": "missing"})
                continue
            if path.stat().st_size > MAX_FILE_SIZE:
                logger.warning("Skipping oversized file: %s", path)
                continue
            content = await asyncio.to_thread(path.read_bytes)
            current_hash = hashlib.sha256(content).hexdigest()
            if current_hash != expected_hash:
                modified.append(
                    {
                        "file": file_path_str,
                        "expected": expected_hash[:8],
                        "actual": current_hash[:8],
                    }
                )

    count = len(modified)
    status = _evaluate(count, check.threshold_warning, check.threshold_critical)

    recommendation = None
    if modified:
        recommendation = "Review modified skills or restore from git"

    return AuditResult(
        check_name=check.name,
        category=check.category,
        status=status,
        value=float(count),
        details={"modified": modified},
        recommendation=recommendation,
    )


async def check_config_drift(
    check: AuditCheck,
    *,
    baseline: SystemBaseline,
    config_path: Path | None = None,
    **_: Any,
) -> AuditResult:
    """Detect configuration drift from the documented baseline state."""
    config_path = config_path or Path.home() / ".gorgon" / "config"
    drift: list[dict[str, Any]] = []

    if config_path.exists() and baseline.config_snapshots:
        try:
            import yaml
        except ImportError:
            return AuditResult(
                check_name=check.name,
                category=check.category,
                status=AuditStatus.OK,
                value=0.0,
                details={"note": "yaml not available"},
            )

        for config_name, documented in baseline.config_snapshots.items():
            config_file = config_path / config_name
            if not config_file.exists():
                drift.append({"file": config_name, "issue": "missing"})
                continue
            if config_file.stat().st_size > MAX_FILE_SIZE:
                logger.warning("Skipping oversized config: %s", config_file)
                continue
            try:
                text = await asyncio.to_thread(config_file.read_text)
                current = yaml.safe_load(text)
            except Exception:
                drift.append({"file": config_name, "issue": "parse_error"})
                continue

            differences = _diff_configs(documented, current)
            if differences:
                drift.append({"file": config_name, "differences": differences})

    count = len(drift)
    status = _evaluate(count, check.threshold_warning, check.threshold_critical)

    recommendation = None
    if drift:
        files = [d.get("file", "unknown") for d in drift]
        recommendation = f"Config drift detected in: {', '.join(files)}"

    return AuditResult(
        check_name=check.name,
        category=check.category,
        status=status,
        value=float(count),
        details={"drift": drift},
        recommendation=recommendation,
    )


async def check_resource_usage(
    check: AuditCheck,
    *,
    baseline: SystemBaseline,
    **_: Any,
) -> AuditResult:
    """Compare current resource usage against the baseline."""
    try:
        import psutil

        cpu = await asyncio.to_thread(psutil.cpu_percent, 1)
        mem = psutil.virtual_memory().percent
    except ImportError:
        return AuditResult(
            check_name=check.name,
            category=check.category,
            status=AuditStatus.OK,
            value=0.0,
            details={"note": "psutil not available"},
        )

    baseline_cpu = baseline.idle_cpu_percent or 1.0
    baseline_mem = baseline.idle_memory_percent or 1.0

    cpu_ratio = cpu / max(baseline_cpu, 0.1)
    mem_ratio = mem / max(baseline_mem, 0.1)
    worst_ratio = max(cpu_ratio, mem_ratio)

    status = _evaluate(worst_ratio, check.threshold_warning, check.threshold_critical)

    recommendation = None
    if status != AuditStatus.OK:
        recommendation = (
            f"Resource usage elevated: CPU {cpu:.1f}% "
            f"(baseline {baseline_cpu:.1f}%), "
            f"Memory {mem:.1f}% (baseline {baseline_mem:.1f}%)"
        )

    return AuditResult(
        check_name=check.name,
        category=check.category,
        status=status,
        value=worst_ratio,
        baseline=1.0,
        details={
            "cpu_percent": cpu,
            "memory_percent": mem,
            "cpu_ratio": cpu_ratio,
            "memory_ratio": mem_ratio,
        },
        recommendation=recommendation,
    )


def _diff_configs(
    expected: dict | Any,
    actual: dict | Any,
    path: str = "",
) -> list[dict[str, Any]]:
    """Recursively diff two config dictionaries.

    Args:
        expected: The baseline config.
        actual: The current config.
        path: Dot-separated key path for nested diffs.

    Returns:
        List of difference records.
    """
    diffs: list[dict[str, Any]] = []

    if not isinstance(expected, dict) or not isinstance(actual, dict):
        if expected != actual:
            diffs.append({"path": path or "<root>", "expected": expected, "actual": actual})
        return diffs

    all_keys = set(expected.keys()) | set(actual.keys())
    for key in sorted(all_keys):
        key_path = f"{path}.{key}" if path else key
        if key not in actual:
            diffs.append({"path": key_path, "issue": "removed"})
        elif key not in expected:
            diffs.append({"path": key_path, "issue": "added", "value": actual[key]})
        else:
            diffs.extend(_diff_configs(expected[key], actual[key], key_path))

    return diffs


def register_default_checks(
    auditor: SystemAuditor,
    backend: DatabaseBackend,
    baseline: SystemBaseline,
    skills_path: Path | None = None,
    config_path: Path | None = None,
) -> None:
    """Register all built-in audit checks with an auditor.

    Args:
        auditor: The SystemAuditor to register checks on.
        backend: Database backend for queries.
        baseline: Active system baseline.
        skills_path: Override skills directory path.
        config_path: Override config directory path.
    """

    async def _check_task_time(check: AuditCheck) -> AuditResult:
        return await check_task_completion_time(check, backend=backend, baseline=baseline)

    async def _check_errors(check: AuditCheck) -> AuditResult:
        return await check_error_rate(check, backend=backend)

    async def _check_deps(check: AuditCheck) -> AuditResult:
        return await check_dependencies(check)

    async def _check_skills(check: AuditCheck) -> AuditResult:
        return await check_skill_integrity(check, baseline=baseline, skills_path=skills_path)

    async def _check_config(check: AuditCheck) -> AuditResult:
        return await check_config_drift(check, baseline=baseline, config_path=config_path)

    async def _check_resources(check: AuditCheck) -> AuditResult:
        return await check_resource_usage(check, baseline=baseline)

    auditor.register_check("check_task_completion_time", _check_task_time)
    auditor.register_check("check_error_rate", _check_errors)
    auditor.register_check("check_dependencies", _check_deps)
    auditor.register_check("check_skill_integrity", _check_skills)
    auditor.register_check("check_config_drift", _check_config)
    auditor.register_check("check_resource_usage", _check_resources)
