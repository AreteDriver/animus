"""Eval evidence integration for Animus Citizens.

Provides a unified interface for citizens to query the Forge eval system
and incorporate eval results into ImprovementProposal evidence.

Gracefully degrades when Forge is not installed or eval store is unavailable.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from animus.logging import get_logger

logger = get_logger("citizens.eval_evidence")

# Module-level cache to suppress repeated "no eval DB" warnings per session.
_eval_db_available: bool | None = None


def _try_import_eval_store() -> Any | None:
    """Attempt to import Forge EvalStore."""
    try:
        from animus_forge.evaluation.store import EvalStore  # boundary-ok: citizen degrades gracefully without Forge
        return EvalStore
    except ImportError:
        return None


def _try_create_backend(db_path: str = "") -> Any | None:
    """Create a Forge database backend for EvalStore."""
    try:
        from animus_forge.state.backends import create_backend  # boundary-ok: citizen degrades gracefully without Forge
        if db_path:
            return create_backend(db_path=db_path)
        return create_backend()
    except ImportError:
        return None


def _try_import_suite_result() -> Any | None:
    """Attempt to import Forge SuiteResult."""
    try:
        from animus_forge.evaluation.base import EvalResult  # boundary-ok: citizen degrades gracefully without Forge
        return EvalResult
    except ImportError:
        return None


def query_eval_runs(
    suite_name: str | None = None,
    limit: int = 20,
    db_path: str = "",
) -> list[dict[str, Any]]:
    """Query eval runs from the Forge eval store.

    Args:
        suite_name: Filter by suite name. If None, returns all suites.
        limit: Maximum runs to return.
        db_path: Path to eval SQLite DB. If empty, uses default.

    Returns:
        List of run dicts with keys: suite_name, score, timestamp, status,
        failure_mode, rubric_band, pass_rate, etc.
    """
    global _eval_db_available
    if _eval_db_available is False:
        return []

    EvalStore = _try_import_eval_store()
    if EvalStore is None:
        _eval_db_available = False
        return []

    try:
        backend = _try_create_backend(db_path)
        if backend is None:
            _eval_db_available = False
            return []
        store = EvalStore(backend)
        runs = store.query_runs(suite_name=suite_name, limit=limit)
        _eval_db_available = True
        # Normalize to plain dicts
        results = []
        for run in runs:
            if hasattr(run, "to_dict"):
                run = run.to_dict()
            if isinstance(run, dict):
                results.append(_normalize_run(run))
        return results
    except Exception as e:
        if "no such table" in str(e).lower():
            _eval_db_available = False
            logger.debug("Eval DB tables missing — caching unavailable state for session")
        else:
            logger.warning(f"Eval query failed: {e}")
        return []


def get_suite_trend(suite_name: str, days: int = 30, db_path: str = "") -> list[dict[str, Any]]:
    """Get trend data for a specific eval suite.

    Args:
        suite_name: Name of the eval suite.
        days: Number of days to look back.
        db_path: Path to eval SQLite DB.

    Returns:
        List of trend data points.
    """
    global _eval_db_available
    if _eval_db_available is False:
        return []

    EvalStore = _try_import_eval_store()
    if EvalStore is None:
        _eval_db_available = False
        return []

    try:
        backend = _try_create_backend(db_path)
        if backend is None:
            _eval_db_available = False
            return []
        store = EvalStore(backend)
        trend = store.get_suite_trend(suite_name=suite_name, days=days)
        _eval_db_available = True
        return [t if isinstance(t, dict) else t.to_dict() for t in trend]
    except Exception as e:
        if "no such table" in str(e).lower():
            _eval_db_available = False
            logger.debug("Eval DB tables missing — caching unavailable state for session")
        else:
            logger.warning(f"Eval trend query failed: {e}")
        return []


def build_eval_evidence_item(run: dict[str, Any]) -> dict[str, Any]:
    """Build an EvidenceItem-compatible dict from an eval run.

    Args:
        run: Eval run dict.

    Returns:
        Dict with keys: source, description, data.
    """
    suite = run.get("suite_name", run.get("suite", "unknown"))
    score = run.get("score", run.get("total_score", 0.0))
    status = run.get("status", "unknown")
    failure_mode = run.get("failure_mode", "")
    rubric_band = run.get("rubric_band", "")
    pass_rate = run.get("pass_rate", run.get("passed", 0))

    description = f"Eval suite '{suite}': score={score:.2f}, status={status}"
    if failure_mode:
        description += f", failure_mode={failure_mode}"
    if rubric_band:
        description += f", rubric_band={rubric_band}"

    return {
        "source": "eval_system",
        "description": description,
        "data": {
            "suite": suite,
            "score": score,
            "status": status,
            "failure_mode": failure_mode,
            "rubric_band": rubric_band,
            "pass_rate": pass_rate,
            "timestamp": run.get("timestamp", ""),
        },
    }


def _normalize_run(run: dict[str, Any]) -> dict[str, Any]:
    """Normalize an eval run dict to a consistent schema."""
    # Handle different field names across versions
    normalized = {
        "suite_name": run.get("suite_name", run.get("suite", "unknown")),
        "score": run.get("score", run.get("total_score", 0.0)),
        "status": run.get("status", "unknown"),
        "timestamp": run.get("timestamp", ""),
        "failure_mode": run.get("failure_mode", ""),
        "rubric_band": run.get("rubric_band", ""),
        "pass_rate": run.get("pass_rate", run.get("passed", 0)),
    }
    return normalized


def read_eval_results_from_memory(memory_layer: Any, limit: int = 50) -> list[dict[str, Any]]:
    """Read eval results stored in Animus memory.

    Fallback when Forge eval store is not directly accessible.

    Args:
        memory_layer: Animus MemoryLayer instance.
        limit: Maximum results.

    Returns:
        List of eval result dicts.
    """
    if memory_layer is None:
        return []

    results = []
    try:
        from animus.memory import MemoryType
        memories = memory_layer.search(
            query="eval suite result score pass_rate",
            memory_type=MemoryType.PROCEDURAL,
            limit=limit,
        )
        for mem in memories:
            if hasattr(mem, "to_dict"):
                mem_dict = mem.to_dict()
            elif isinstance(mem, dict):
                mem_dict = mem
            else:
                continue
            meta = mem_dict.get("metadata", {})
            if meta.get("score") is not None or meta.get("suite") is not None:
                results.append(_normalize_run(meta))
    except Exception as e:
        logger.debug(f"Memory eval query failed: {e}")

    return results


def read_eval_results_from_dir(eval_dir: Path | str, limit: int = 20) -> list[dict[str, Any]]:
    """Read eval results from JSON files in a directory.

    Args:
        eval_dir: Directory containing eval result JSON files.
        limit: Maximum files to read.

    Returns:
        List of eval result dicts.
    """
    path = Path(eval_dir).expanduser() if eval_dir else None
    if not path or not path.exists():
        return []

    results = []
    for json_file in sorted(path.glob("*.json"))[-limit:]:
        try:
            data = json.loads(json_file.read_text())
            if isinstance(data, list):
                for item in data:
                    results.append(_normalize_run(item))
            elif isinstance(data, dict):
                results.append(_normalize_run(data))
        except Exception:
            continue

    return results
