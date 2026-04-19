"""
SQLite cache for parsed Claude Code session summaries.

Keyed by (session_file, mtime_ns). A cache hit returns a fully-hydrated
SessionSummary (turns included) without re-parsing JSONL. A miss re-parses
and writes back.

Default location: ~/.animus/lugh_cache.sqlite
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from animus.lugh.models import SessionSummary, Turn

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path("~/.animus/lugh_cache.sqlite").expanduser()

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_file TEXT PRIMARY KEY,
    mtime_ns    INTEGER NOT NULL,
    parsed_at   TEXT    NOT NULL,
    summary_json TEXT   NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_mtime ON sessions(mtime_ns);
"""


class TranscriptCache:
    """Thin SQLite cache for SessionSummary objects."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_CACHE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False per Animus CLAUDE.md cross-thread pattern
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def get(self, session_file: str, mtime_ns: int) -> SessionSummary | None:
        row = self._conn.execute(
            "SELECT summary_json FROM sessions WHERE session_file = ? AND mtime_ns = ?",
            (session_file, mtime_ns),
        ).fetchone()
        if row is None:
            return None
        try:
            return _summary_from_json(row["summary_json"])
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("cache row corrupt for %s: %s — invalidating", session_file, e)
            self._conn.execute("DELETE FROM sessions WHERE session_file = ?", (session_file,))
            self._conn.commit()
            return None

    def put(self, session_file: str, mtime_ns: int, summary: SessionSummary) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions (session_file, mtime_ns, parsed_at, summary_json) "
            "VALUES (?, ?, ?, ?)",
            (
                session_file,
                mtime_ns,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                _summary_to_json(summary),
            ),
        )
        self._conn.commit()

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()
        return int(row["n"]) if row else 0

    def vacuum(self, valid_files: set[str]) -> int:
        """Drop rows whose session_file isn't in valid_files. Returns rows deleted."""
        if not valid_files:
            return 0
        rows = self._conn.execute("SELECT session_file FROM sessions").fetchall()
        stale = [r["session_file"] for r in rows if r["session_file"] not in valid_files]
        if not stale:
            return 0
        self._conn.executemany("DELETE FROM sessions WHERE session_file = ?", ((s,) for s in stale))
        self._conn.commit()
        return len(stale)


# ---------------------------------------------------------------------------
# (de)serialization
# ---------------------------------------------------------------------------


def _summary_to_json(s: SessionSummary) -> str:
    d = asdict(s)
    # datetime is not JSON-serializable; convert per-turn timestamps
    for t in d["turns"]:
        ts = t.get("timestamp")
        if isinstance(ts, datetime):
            t["timestamp"] = ts.isoformat()
    return json.dumps(d, separators=(",", ":"))


def _summary_from_json(blob: str) -> SessionSummary:
    d = json.loads(blob)
    turns = [_turn_from_dict(t) for t in d.pop("turns", [])]
    return SessionSummary(turns=turns, **d)


def _turn_from_dict(d: dict) -> Turn:
    ts = d.get("timestamp")
    if isinstance(ts, str):
        try:
            d["timestamp"] = datetime.fromisoformat(ts)
        except ValueError:
            d["timestamp"] = None
    return Turn(**d)
