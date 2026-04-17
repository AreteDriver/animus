"""Tests for the Lugh SQLite transcript cache."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from animus.lugh.cache import TranscriptCache
from animus.lugh.models import SessionSummary, Turn


def _fake_summary(session_file: str = "proj/a.jsonl", turn_count: int = 2) -> SessionSummary:
    turns = [
        Turn(
            session_id="sess",
            turn_index=i,
            timestamp=datetime(2026, 4, 10, 9, 0, i),
            role="assistant" if i % 2 else "user",
            is_sidechain=False,
            agent_id=None,
            cwd="/home/arete/projects",
            model="claude-opus-4-6",
            input_tokens=10,
            output_tokens=20,
            cache_read_tokens=100,
            cache_write_tokens=50,
            tool_calls=["Edit"] if i % 2 else [],
            mcp_servers=[],
            activity="coding" if i % 2 else "conversation",
            cost_usd=0.01,
        )
        for i in range(turn_count)
    ]
    return SessionSummary(
        session_id="sess",
        session_file=session_file,
        is_sidechain=False,
        agent_id=None,
        project_dir="-home-arete-projects",
        primary_cwd="/home/arete/projects",
        date="2026-04-10",
        duration_seconds=60.0,
        active_seconds=60.0,
        turn_count=turn_count,
        tool_call_count=1,
        mcp_call_count=0,
        model_distribution={"claude-opus-4-6": turn_count},
        total_input_tokens=20,
        total_output_tokens=40,
        total_cache_read_tokens=200,
        total_cache_write_tokens=100,
        total_cost_usd=0.02,
        activity_breakdown={"coding": 50.0, "conversation": 50.0},
        cwd_breakdown={"/home/arete/projects": turn_count},
        unknown_line_types=[],
        unknown_models=[],
        turns=turns,
    )


@pytest.fixture
def cache(tmp_path):
    c = TranscriptCache(path=tmp_path / "cache.sqlite")
    yield c
    c.close()


class TestTranscriptCache:
    def test_round_trip_preserves_summary(self, cache):
        s = _fake_summary()
        cache.put(s.session_file, 111, s)
        got = cache.get(s.session_file, 111)
        assert got is not None
        assert got.session_id == s.session_id
        assert got.turn_count == s.turn_count
        assert got.activity_breakdown == s.activity_breakdown
        assert got.turns[0].model == "claude-opus-4-6"
        assert got.turns[0].activity in ("conversation", "coding")
        assert isinstance(got.turns[0].timestamp, datetime)

    def test_miss_on_different_mtime(self, cache):
        s = _fake_summary()
        cache.put(s.session_file, 100, s)
        assert cache.get(s.session_file, 101) is None

    def test_miss_on_unknown_file(self, cache):
        assert cache.get("never/written.jsonl", 0) is None

    def test_replace_on_mtime_change(self, cache):
        s1 = _fake_summary(turn_count=2)
        cache.put(s1.session_file, 100, s1)
        s2 = _fake_summary(turn_count=5)
        cache.put(s2.session_file, 200, s2)
        # old mtime row still there? No — put uses INSERT OR REPLACE on the primary key
        assert cache.get(s2.session_file, 100) is None
        got = cache.get(s2.session_file, 200)
        assert got is not None
        assert got.turn_count == 5

    def test_vacuum_drops_stale_rows(self, cache):
        for i in range(3):
            s = _fake_summary(session_file=f"proj/s{i}.jsonl")
            cache.put(s.session_file, 100 + i, s)
        assert cache.count() == 3
        dropped = cache.vacuum({"proj/s0.jsonl", "proj/s2.jsonl"})
        assert dropped == 1
        assert cache.get("proj/s1.jsonl", 101) is None
        assert cache.get("proj/s0.jsonl", 100) is not None

    def test_vacuum_noop_when_empty_set(self, cache):
        s = _fake_summary()
        cache.put(s.session_file, 100, s)
        # Empty valid_files set is treated as no-op to avoid deleting everything
        # on a harvest with no files discovered (edge case safety)
        assert cache.vacuum(set()) == 0
        assert cache.count() == 1

    def test_corrupt_row_is_invalidated(self, cache, tmp_path):
        # Directly insert garbage JSON into the DB and confirm get() returns None
        import sqlite3

        conn = sqlite3.connect(cache.path)
        conn.execute(
            "INSERT INTO sessions (session_file, mtime_ns, parsed_at, summary_json) "
            "VALUES (?, ?, ?, ?)",
            ("bad.jsonl", 1, "2026-04-10T00:00:00Z", "{not valid json"),
        )
        conn.commit()
        conn.close()
        assert cache.get("bad.jsonl", 1) is None
        # After invalidation, the row is gone
        conn = sqlite3.connect(cache.path)
        rows = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_file = ?", ("bad.jsonl",)
        ).fetchone()
        conn.close()
        assert rows[0] == 0

    def test_harvest_uses_cache(self, tmp_path, monkeypatch):
        """harvest_transcripts should skip re-parsing when mtime unchanged."""
        from animus.lugh.transcripts import harvest_transcripts, parse_session

        base = tmp_path / "projects"
        d = base / "-home-arete-projects"
        d.mkdir(parents=True)
        jsonl = d / "sess.jsonl"
        jsonl.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "user",
                            "sessionId": "sess",
                            "timestamp": "2026-04-10T09:03:58Z",
                            "cwd": "/home/arete/projects",
                            "message": {
                                "role": "user",
                                "content": [{"type": "text", "text": "hi"}],
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "assistant",
                            "sessionId": "sess",
                            "timestamp": "2026-04-10T09:04:00Z",
                            "cwd": "/home/arete/projects",
                            "message": {
                                "role": "assistant",
                                "model": "claude-opus-4-6",
                                "content": [{"type": "text", "text": "ok"}],
                                "usage": {"input_tokens": 1, "output_tokens": 1},
                            },
                        }
                    ),
                ]
            )
        )

        cache = TranscriptCache(path=tmp_path / "cache.sqlite")

        # First pass populates the cache
        list(harvest_transcripts(claude_dir=base, cache=cache))
        assert cache.count() == 1

        # Second pass: monkeypatch parse_session to raise if called — proves cache hit
        def _boom(*a, **kw):
            raise AssertionError("parse_session should not be called on cache hit")

        monkeypatch.setattr("animus.lugh.transcripts.parse_session", _boom)
        sessions = list(harvest_transcripts(claude_dir=base, cache=cache))
        assert len(sessions) == 1

        # Touching the file changes mtime_ns → cache miss → parse is called again
        import os

        orig_parse = parse_session
        monkeypatch.setattr("animus.lugh.transcripts.parse_session", orig_parse)
        os.utime(jsonl, None)  # bump mtime
        # Overwrite with a slightly different byte to really ensure mtime_ns changed
        content = jsonl.read_text()
        jsonl.write_text(content + "\n")
        assert jsonl.stat().st_mtime_ns != 0  # sanity
        list(harvest_transcripts(claude_dir=base, cache=cache))
        # Row still one session but it's been replaced
        assert cache.count() == 1
        cache.close()
