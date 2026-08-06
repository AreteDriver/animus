"""Path resolution helpers for the Governor state layout.

The Governor writes its run state under
``<repo>/.animus-loop-governor/runs/<run-id>/``. These helpers are the
only sanctioned way to compute those paths — call sites never build
the path by hand.

The functions accept ``str | Path`` for ergonomics but always return
``pathlib.Path``.
"""

from __future__ import annotations

from pathlib import Path

from animus_forge.governor.errors import RunNotFoundError

GOVERNOR_DIRNAME = ".animus-loop-governor"
RUNS_DIRNAME = "runs"


def runs_root(repository: str | Path) -> Path:
    """``<repository>/.animus-loop-governor`` — Governor state root."""
    return Path(repository).resolve() / GOVERNOR_DIRNAME


def run_dir(repository: str | Path, run_id: str) -> Path:
    """``<runs_root>/runs/<run_id>`` — canonical run directory."""
    return runs_root(repository) / RUNS_DIRNAME / run_id


def run_dir_or_raise(repository: str | Path, run_id: str) -> Path:
    """Return run dir, raising :class:`RunNotFoundError` if absent."""
    path = run_dir(repository, run_id)
    if not path.is_dir():
        raise RunNotFoundError(f"Run directory not found: {path}")
    return path


def find_active_run(repository: str | Path) -> Path | None:
    """Most-recently-modified run dir under ``runs/``; ``None`` if absent.

    Used only as a *hint* during :meth:`adapter.ensure_run` resolution.
    The adapter always validates any returned dir against the
    compatibility key before reuse; it never trusts a run found here
    blindly.

    "Most recent" is by ``Path.stat().st_mtime`` — matches user intuition
    when sorting ``runs/`` in a file manager.
    """
    runs = runs_root(repository) / RUNS_DIRNAME
    if not runs.is_dir():
        return None

    candidates = [entry for entry in runs.iterdir() if entry.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda entry: entry.stat().st_mtime)


__all__ = [
    "GOVERNOR_DIRNAME",
    "RUNS_DIRNAME",
    "find_active_run",
    "run_dir",
    "run_dir_or_raise",
    "runs_root",
]
