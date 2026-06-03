"""Grounding — verify ``animus_gap`` claims by actually reading the repo.

The rule that separates Ogma from a summary bot (per ``docs/OGMA.md``
§Architecture): *no Animus file path appears in output unless it was
actually read during synthesis.* This module enforces that by exposing a
single entry point that returns the list of paths it touched.

``verify_animus_gap`` shells out to ``grep -rn`` (already a hard dep — Lugh
uses ``yt-dlp`` the same way), then reads the surrounding 50 lines of each
hit. No new Python deps, no scanning the whole repo into memory.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

DEFAULT_REPO_ROOT = Path("~/projects/animus").expanduser()
DEFAULT_SEARCH_GLOB = "packages/"
GREP_TIMEOUT_SECONDS = 30
CONTEXT_LINES = 25  # lines on either side of a hit, so we read 50 lines per file


class GroundingError(RuntimeError):
    """Raised when grounding cannot complete (timeout, missing repo, etc)."""


@dataclass(frozen=True)
class GapResult:
    """The outcome of one ``verify_animus_gap`` call.

    Attributes:
        status: ``NONE`` (no hits), ``PARTIAL`` (some hits across few files),
            ``FULL`` (broad coverage across multiple subsystems).
        paths_read: ``relpath:line_start-line_end`` references for every file
            actually opened during this call. Non-empty whenever status is
            not NONE.
        notes: One paragraph for the LLM consumer summarizing what was found.
    """

    status: Literal["NONE", "PARTIAL", "FULL"]
    paths_read: list[str] = field(default_factory=list)
    notes: str = ""


def verify_animus_gap(
    concept: str,
    repo_root: Path | None = None,
    *,
    max_files: int = 5,
    extra_terms: list[str] | None = None,
) -> GapResult:
    """Check whether the animus repo already implements ``concept``.

    Args:
        concept: the concept to search for (e.g. ``"learned test-time memory"``).
        repo_root: defaults to ``~/projects/animus`` — must be a git repo.
        max_files: cap on distinct files we open per call. Hits past the cap
            are still surfaced in ``notes`` but not read into ``paths_read``.
        extra_terms: additional query terms; if omitted, derive 1–2 obvious
            synonyms from the concept itself (word splits).

    Returns:
        A ``GapResult`` whose ``paths_read`` lists every file:line range
        actually loaded — the synthesis prompt then has provenance for any
        path it cites.

    Raises:
        ValueError: empty ``concept``.
        GroundingError: ``repo_root`` is not a git repo, or grep times out.
    """
    if not concept or not concept.strip():
        raise ValueError("verify_animus_gap: concept must be non-empty")
    root = (repo_root or DEFAULT_REPO_ROOT).resolve()
    if not (root / ".git").is_dir():
        raise GroundingError(f"not a git repo: {root}")

    terms = _query_terms(concept, extra_terms)
    hits: dict[Path, list[int]] = {}
    for term in terms:
        for path, line_no in _grep(term, root):
            hits.setdefault(path, []).append(line_no)

    if not hits:
        return GapResult(status="NONE", notes=f"no matches for {concept!r} across {root}")

    paths_read: list[str] = []
    for path, line_nos in list(hits.items())[:max_files]:
        line_nos = sorted(set(line_nos))
        start = max(1, line_nos[0] - CONTEXT_LINES)
        end = line_nos[-1] + CONTEXT_LINES
        try:
            _ = _read_lines(path, start, end)
        except OSError as e:
            logger.warning("verify_animus_gap: read failed for %s: %s", path, e)
            continue
        rel = path.relative_to(root) if path.is_absolute() else path
        paths_read.append(f"{rel}:{start}-{end}")

    if not paths_read:
        return GapResult(
            status="NONE",
            notes=f"grep matched {len(hits)} files but none were readable",
        )

    n_files = len(hits)
    status: Literal["PARTIAL", "FULL"] = "FULL" if n_files >= 3 else "PARTIAL"
    notes = (
        f"{n_files} file(s) contain matches for {concept!r}; "
        f"read {len(paths_read)} file(s) for context."
    )
    return GapResult(status=status, paths_read=paths_read, notes=notes)


def _query_terms(concept: str, extra: list[str] | None) -> list[str]:
    """Yield the search terms — the concept itself + caller-supplied + word splits."""
    seen: set[str] = set()
    terms: list[str] = []

    def add(t: str) -> None:
        t = t.strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            terms.append(t)

    add(concept)
    for e in extra or []:
        add(e)
    # Word splits longer than 4 chars (avoid noise like "the"/"and").
    for word in re.split(r"\s+", concept):
        if len(word) > 4:
            add(word)
    return terms


def _grep(term: str, root: Path) -> list[tuple[Path, int]]:
    """Run ``grep -rn`` for ``term`` under ``root/packages/``. Returns ``(path, line_no)`` pairs.

    Failures (grep absent, timeout, nonzero exit on no-match) are downgraded
    to "no hits" rather than raised — caller decides what to do with an empty
    result. The exception is grep timeout, which surfaces because it
    indicates a pathological tree.
    """
    search_root = root / DEFAULT_SEARCH_GLOB
    cmd = [
        "grep",
        "-rn",
        "-I",
        "--include=*.py",
        "--include=*.md",
        "-F",
        term,
        str(search_root),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=GREP_TIMEOUT_SECONDS, check=False
        )
    except FileNotFoundError:
        logger.warning("grep not found on PATH")
        return []
    except subprocess.TimeoutExpired as e:
        raise GroundingError(f"grep timeout after {GREP_TIMEOUT_SECONDS}s") from e
    rows: list[tuple[Path, int]] = []
    for line in (proc.stdout or "").splitlines():
        # grep -rn output: <path>:<line>:<text>
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        try:
            rows.append((Path(parts[0]), int(parts[1])))
        except ValueError:
            continue
    return rows


def _read_lines(path: Path, start: int, end: int) -> str:
    """Read ``path``'s line range (1-indexed, inclusive). Caps at file length."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    lo = max(0, start - 1)
    hi = min(len(lines), end)
    return "\n".join(lines[lo:hi])
