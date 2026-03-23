"""
Harvest Watchlist — competition monitoring for repos.

Stores a list of repos to periodically scan, tracks score changes,
and generates diff reports between scans.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

WATCHLIST_FILE = Path("~/.animus/harvest_watchlist.json").expanduser()

# Default scan interval: 7 days
DEFAULT_SCAN_INTERVAL_HOURS = 168


def _load_watchlist() -> dict[str, Any]:
    """Load watchlist from disk. Returns default structure if missing."""
    if not WATCHLIST_FILE.exists():
        return {"repos": [], "scan_interval_hours": DEFAULT_SCAN_INTERVAL_HOURS}
    try:
        return json.loads(WATCHLIST_FILE.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read watchlist: %s", e)
        return {"repos": [], "scan_interval_hours": DEFAULT_SCAN_INTERVAL_HOURS}


def _save_watchlist(data: dict[str, Any]) -> None:
    """Save watchlist to disk."""
    WATCHLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    WATCHLIST_FILE.write_text(json.dumps(data, indent=2))


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def add_to_watchlist(
    target: str,
    tags: list[str] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Add a repo to the watchlist.

    Args:
        target: GitHub repo as 'user/repo' or full URL.
        tags: Optional tags for categorization.
        notes: Optional notes about why this repo matters.

    Returns:
        The created watchlist entry.
    """
    from animus.harvest import _extract_repo_name

    repo_name = _extract_repo_name(target)
    data = _load_watchlist()

    # Check for duplicates
    for entry in data["repos"]:
        if entry["target"] == repo_name:
            raise ValueError(f"'{repo_name}' is already on the watchlist")

    entry = {
        "target": repo_name,
        "added_at": _now_iso(),
        "last_scanned": None,
        "last_score": None,
        "last_findings": None,
        "tags": tags or [],
        "notes": notes or "",
    }
    data["repos"].append(entry)
    _save_watchlist(data)
    logger.info("Added %s to watchlist", repo_name)
    return entry


def remove_from_watchlist(target: str) -> bool:
    """Remove a repo from the watchlist.

    Args:
        target: GitHub repo as 'user/repo' or full URL.

    Returns:
        True if removed, False if not found.
    """
    from animus.harvest import _extract_repo_name

    repo_name = _extract_repo_name(target)
    data = _load_watchlist()

    original_len = len(data["repos"])
    data["repos"] = [e for e in data["repos"] if e["target"] != repo_name]

    if len(data["repos"]) < original_len:
        _save_watchlist(data)
        logger.info("Removed %s from watchlist", repo_name)
        return True
    return False


def get_watchlist() -> list[dict[str, Any]]:
    """Return the full watchlist."""
    data = _load_watchlist()
    return data["repos"]


def get_due_repos(interval_hours: int | None = None) -> list[dict[str, Any]]:
    """Return repos that haven't been scanned within interval_hours.

    Args:
        interval_hours: Override default interval. None uses watchlist default.

    Returns:
        List of watchlist entries due for scanning.
    """
    data = _load_watchlist()
    interval = (
        interval_hours
        if interval_hours is not None
        else data.get("scan_interval_hours", DEFAULT_SCAN_INTERVAL_HOURS)
    )

    now = datetime.now(timezone.utc)
    due: list[dict[str, Any]] = []

    for entry in data["repos"]:
        last = entry.get("last_scanned")
        if last is None:
            due.append(entry)
            continue

        try:
            last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
            hours_since = (now - last_dt).total_seconds() / 3600
            if hours_since >= interval:
                due.append(entry)
        except (ValueError, TypeError):
            due.append(entry)

    return due


def update_scan_result(
    target: str,
    score: int,
    findings: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Update a watchlist entry with scan results.

    Args:
        target: GitHub repo as 'user/repo'.
        score: New score from harvest scan.
        findings: Optional dict of scan findings to store.

    Returns:
        Updated entry or None if not found.
    """
    from animus.harvest import _extract_repo_name

    repo_name = _extract_repo_name(target)
    data = _load_watchlist()

    for entry in data["repos"]:
        if entry["target"] == repo_name:
            entry["last_scanned"] = _now_iso()
            entry["last_score"] = score
            if findings is not None:
                entry["last_findings"] = findings
            _save_watchlist(data)
            return entry

    return None


def get_changes_report(
    target: str,
    current_result: dict[str, Any],
) -> dict[str, Any]:
    """Compare current scan against stored last scan for a repo.

    Args:
        target: GitHub repo as 'user/repo'.
        current_result: Current HarvestResult.to_dict() output.

    Returns:
        Change report dict with score_change, new/removed patterns, etc.
    """
    from animus.harvest import _extract_repo_name

    repo_name = _extract_repo_name(target)
    data = _load_watchlist()

    entry = None
    for e in data["repos"]:
        if e["target"] == repo_name:
            entry = e
            break

    report: dict[str, Any] = {
        "repo": repo_name,
        "score_change": None,
        "new_patterns": [],
        "removed_patterns": [],
        "new_dependencies": [],
        "removed_dependencies": [],
        "alert": None,
    }

    if entry is None:
        report["alert"] = "Repo not on watchlist"
        return report

    prev_score = entry.get("last_score")
    curr_score = current_result.get("score", 0)

    if prev_score is not None:
        diff = curr_score - prev_score
        sign = "+" if diff > 0 else ""
        report["score_change"] = f"{sign}{diff} ({prev_score} -> {curr_score})"
    else:
        report["score_change"] = f"Initial scan: {curr_score}"

    # Compare patterns
    prev_findings = entry.get("last_findings") or {}
    prev_patterns = set(prev_findings.get("notable_patterns", []))
    curr_patterns = set(current_result.get("notable_patterns", []))

    report["new_patterns"] = sorted(curr_patterns - prev_patterns)
    report["removed_patterns"] = sorted(prev_patterns - curr_patterns)

    # Compare tools/dependencies
    prev_tools = set(prev_findings.get("tools_worth_adopting", []))
    curr_tools = set(current_result.get("tools_worth_adopting", []))

    report["new_dependencies"] = sorted(curr_tools - prev_tools)
    report["removed_dependencies"] = sorted(prev_tools - curr_tools)

    # Generate alert if significant change
    if prev_score is not None:
        diff = curr_score - prev_score
        if abs(diff) >= 10:
            direction = "improved" if diff > 0 else "declined"
            report["alert"] = f"Significant score change: {direction} by {abs(diff)} points"
        elif report["new_patterns"]:
            report["alert"] = f"New patterns detected: {', '.join(report['new_patterns'][:3])}"

    return report


async def run_watchlist_scan(
    memory: Any | None = None,
    interval_hours: int | None = None,
) -> dict[str, Any]:
    """Run harvest scans on all due repos and generate a changes report.

    Runs sequentially to respect GitHub rate limits.

    Args:
        memory: Optional MemoryLayer for storing findings.
        interval_hours: Override scan interval.

    Returns:
        Structured report with scanned count, changes, and no_changes.
    """
    from animus.harvest import harvest_repo

    due = get_due_repos(interval_hours=interval_hours)

    report: dict[str, Any] = {
        "scanned": 0,
        "changes": [],
        "no_changes": [],
        "errors": [],
    }

    for entry in due:
        target = entry["target"]
        try:
            result = harvest_repo(
                target=target,
                compare=True,
                depth="quick",
                memory_layer=memory,
            )
            result_dict = result.to_dict()

            # Get changes
            change = get_changes_report(target, result_dict)

            # Update stored scan data
            update_scan_result(
                target=target,
                score=result.score,
                findings=result_dict,
            )

            # Classify as changed or no_changes
            has_changes = (
                change.get("new_patterns")
                or change.get("removed_patterns")
                or change.get("new_dependencies")
                or change.get("removed_dependencies")
                or (
                    change.get("score_change")
                    and "Initial" not in str(change.get("score_change", ""))
                )
            )

            if has_changes:
                report["changes"].append(change)
            else:
                report["no_changes"].append(target)

            report["scanned"] += 1

        except Exception as e:
            logger.warning("Failed to scan %s: %s", target, e)
            report["errors"].append({"repo": target, "error": str(e)})

    return report
