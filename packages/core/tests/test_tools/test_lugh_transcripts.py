"""Tests for the Claude Code JSONL transcript harvester (animus.lugh.transcripts)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from animus.lugh.transcripts import (
    KNOWN_LINE_TYPES,
    MODEL_PRICING,
    _base_activity,
    _decode_project_dir,
    _normalize_model,
    _parse_ts,
    _turn_cost,
    drift_signals,
    harvest_transcripts,
    parse_session,
    rollup_by_cwd,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _user_line(session_id: str, text: str, cwd: str = "/home/arete/projects") -> str:
    return json.dumps(
        {
            "type": "user",
            "sessionId": session_id,
            "timestamp": "2026-04-10T09:03:58.166Z",
            "cwd": cwd,
            "message": {"role": "user", "content": [{"type": "text", "text": text}]},
        }
    )


def _assistant_line(
    session_id: str,
    text: str = "",
    *,
    model: str = "claude-opus-4-6",
    tool_calls: list[dict] | None = None,
    usage: dict | None = None,
    cwd: str = "/home/arete/projects",
    ts: str = "2026-04-10T09:04:02.000Z",
) -> str:
    content: list[dict] = []
    if text:
        content.append({"type": "text", "text": text})
    if tool_calls:
        content.extend(tool_calls)
    return json.dumps(
        {
            "type": "assistant",
            "sessionId": session_id,
            "timestamp": ts,
            "cwd": cwd,
            "message": {
                "role": "assistant",
                "model": model,
                "content": content,
                "usage": usage or {"input_tokens": 10, "output_tokens": 20},
            },
        }
    )


def _write_session(dir_: Path, name: str, lines: list[str]) -> Path:
    path = dir_ / name
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_decode_project_dir(self):
        assert _decode_project_dir("-home-arete-projects") == "/home/arete/projects"
        assert _decode_project_dir("-home-arete") == "/home/arete"
        assert _decode_project_dir("literal-name") == "literal-name"

    def test_normalize_model_longest_match(self):
        assert _normalize_model("claude-opus-4-6") == ("claude-opus-4-6", None)
        assert _normalize_model("claude-opus-4-7-20260901") == ("claude-opus-4-7", None)
        assert _normalize_model("claude-sonnet-4") == ("claude-sonnet-4", None)
        # Unknown model returns default plus the original string for tracking
        key, unknown = _normalize_model("future-model-v9")
        assert key == "default"
        assert unknown == "future-model-v9"
        assert _normalize_model(None) == ("default", None)
        assert _normalize_model("") == ("default", None)

    def test_parse_ts(self):
        assert _parse_ts("2026-04-10T09:03:58Z") is not None
        assert _parse_ts("not-a-date") is None
        assert _parse_ts(None) is None

    def test_base_activity_tool_priority(self):
        assert _base_activity(["Edit"], "") == "coding"
        assert _base_activity(["Read"], "") == "file_read"
        assert _base_activity(["Grep"], "") == "search"
        assert _base_activity(["mcp__x"], "") == "mcp_call"
        assert _base_activity(["Agent"], "") == "exploration"
        assert _base_activity(["Bash"], "") == "shell"

    def test_base_activity_text_fallback(self):
        assert _base_activity([], "Traceback: bad") == "debugging"
        assert _base_activity([], "code reviewed by lead") == "review"
        assert _base_activity([], "just chatting") == "conversation"

    def test_turn_cost_uses_pricing(self):
        # 1M input tokens at opus-4 = $15.00
        cost = _turn_cost("claude-opus-4", 1_000_000, 0, 0, 0)
        assert cost == pytest.approx(15.0)
        # unknown model falls back to default sonnet pricing
        cost = _turn_cost("nonexistent", 1_000_000, 0, 0, 0)
        assert cost == pytest.approx(MODEL_PRICING["default"]["input"])


# ---------------------------------------------------------------------------
# parse_session
# ---------------------------------------------------------------------------


class TestParseSession:
    def test_empty_file_returns_none(self, tmp_path):
        f = tmp_path / "projects" / "-home-arete-projects" / "abc.jsonl"
        f.parent.mkdir(parents=True)
        f.write_text("")
        assert parse_session(f, claude_base=tmp_path / "projects") is None

    def test_non_turn_lines_skipped(self, tmp_path):
        base = tmp_path / "projects"
        dir_ = base / "-home-arete-projects"
        dir_.mkdir(parents=True)
        lines = [
            json.dumps({"type": "permission-mode"}),
            json.dumps({"type": "file-history-snapshot"}),
            _user_line("sess-1", "hi"),
            _assistant_line("sess-1", "hello"),
            json.dumps({"type": "attachment"}),
        ]
        path = _write_session(dir_, "sess-1.jsonl", lines)
        s = parse_session(path, claude_base=base)
        assert s is not None
        assert s.turn_count == 2

    def test_cost_and_activity_rollup(self, tmp_path):
        base = tmp_path / "projects"
        dir_ = base / "-home-arete-projects"
        dir_.mkdir(parents=True)
        usage = {
            "input_tokens": 100,
            "output_tokens": 200,
            "cache_read_input_tokens": 1000,
            "cache_creation_input_tokens": 500,
        }
        lines = [
            _user_line("sess-1", "edit this file"),
            _assistant_line(
                "sess-1",
                tool_calls=[{"type": "tool_use", "name": "Edit"}],
                usage=usage,
            ),
        ]
        path = _write_session(dir_, "sess-1.jsonl", lines)
        s = parse_session(path, claude_base=base)
        assert s is not None
        assert s.total_input_tokens == 100
        assert s.total_output_tokens == 200
        assert s.total_cache_read_tokens == 1000
        assert s.total_cache_write_tokens == 500
        assert s.total_cost_usd > 0
        assert s.tool_call_count == 1
        assert s.activity_breakdown["coding"] == 50.0  # 1 of 2 turns
        assert s.primary_cwd == "/home/arete/projects"

    def test_multi_cwd_session(self, tmp_path):
        base = tmp_path / "projects"
        dir_ = base / "-home-arete-projects"
        dir_.mkdir(parents=True)
        lines = [
            _user_line("sess-1", "a", cwd="/home/arete/projects"),
            _assistant_line("sess-1", "b", cwd="/home/arete/projects"),
            _user_line("sess-1", "c", cwd="/home/arete/projects/monolith"),
            _assistant_line("sess-1", "d", cwd="/home/arete/projects/monolith"),
            _assistant_line("sess-1", "e", cwd="/home/arete/projects/monolith"),
        ]
        path = _write_session(dir_, "sess-1.jsonl", lines)
        s = parse_session(path, claude_base=base)
        assert s is not None
        assert s.primary_cwd == "/home/arete/projects/monolith"
        assert s.cwd_breakdown["/home/arete/projects"] == 2
        assert s.cwd_breakdown["/home/arete/projects/monolith"] == 3

    def test_subagent_marked_sidechain(self, tmp_path):
        base = tmp_path / "projects"
        dir_ = base / "-home-arete-projects" / "sess-1" / "subagents"
        dir_.mkdir(parents=True)
        lines = [
            _user_line("sess-1", "hi"),
            _assistant_line("sess-1", "hello"),
        ]
        path = _write_session(dir_, "agent-xyz.jsonl", lines)
        s = parse_session(path, claude_base=base)
        assert s is not None
        assert s.is_sidechain is True
        assert s.agent_id == "xyz"

    def test_malformed_lines_tolerated(self, tmp_path):
        base = tmp_path / "projects"
        dir_ = base / "-home-arete-projects"
        dir_.mkdir(parents=True)
        lines = [
            "not json at all",
            _user_line("sess-1", "hi"),
            "{{{bad",
            _assistant_line("sess-1", "ok"),
        ]
        path = _write_session(dir_, "sess-1.jsonl", lines)
        s = parse_session(path, claude_base=base)
        assert s is not None
        assert s.turn_count == 2


# ---------------------------------------------------------------------------
# harvest_transcripts + rollup
# ---------------------------------------------------------------------------


class TestHarvest:
    def test_filters(self, tmp_path):
        base = tmp_path / "projects"
        d1 = base / "-home-arete-projects"
        d2 = base / "-home-arete-otherproject"
        d1.mkdir(parents=True)
        d2.mkdir(parents=True)
        _write_session(d1, "a.jsonl", [_user_line("a", "hi"), _assistant_line("a", "ok")])
        _write_session(d2, "b.jsonl", [_user_line("b", "hi"), _assistant_line("b", "ok")])
        all_s = list(harvest_transcripts(claude_dir=base, use_cache=False))
        assert len(all_s) == 2
        filtered = list(
            harvest_transcripts(claude_dir=base, project="otherproject", use_cache=False)
        )
        assert len(filtered) == 1
        assert filtered[0].session_id == "b"

    def test_exclude_sidechains(self, tmp_path):
        base = tmp_path / "projects"
        d = base / "-home-arete-projects"
        sub = d / "sess-a" / "subagents"
        d.mkdir(parents=True)
        sub.mkdir(parents=True)
        _write_session(
            d, "sess-a.jsonl", [_user_line("sess-a", "hi"), _assistant_line("sess-a", "ok")]
        )
        _write_session(
            sub, "agent-1.jsonl", [_user_line("sess-a", "sub"), _assistant_line("sess-a", "r")]
        )
        with_sub = list(harvest_transcripts(claude_dir=base, use_cache=False))
        without = list(
            harvest_transcripts(claude_dir=base, include_sidechains=False, use_cache=False)
        )
        assert len(with_sub) == 2
        assert len(without) == 1
        assert all(not s.is_sidechain for s in without)

    def test_rollup_by_cwd_attributes_per_turn(self, tmp_path):
        base = tmp_path / "projects"
        d = base / "-home-arete-projects"
        d.mkdir(parents=True)
        lines = [
            _user_line("sess-1", "a", cwd="/root/alpha"),
            _assistant_line("sess-1", "b", cwd="/root/alpha"),
            _user_line("sess-1", "c", cwd="/root/beta"),
            _assistant_line("sess-1", "d", cwd="/root/beta"),
        ]
        _write_session(d, "sess-1.jsonl", lines)
        sessions = list(harvest_transcripts(claude_dir=base, use_cache=False))
        roll = rollup_by_cwd(sessions)
        assert set(roll.keys()) == {"/root/alpha", "/root/beta"}
        assert roll["/root/alpha"]["turns"] == 2
        assert roll["/root/beta"]["turns"] == 2


# ---------------------------------------------------------------------------
# drift signals
# ---------------------------------------------------------------------------


class TestDriftSignals:
    def test_payload_shape(self, tmp_path):
        base = tmp_path / "projects"
        d = base / "-home-arete-projects"
        d.mkdir(parents=True)
        _write_session(
            d,
            "sess-1.jsonl",
            [
                _user_line("sess-1", "hi"),
                _assistant_line("sess-1", "ok"),
            ],
        )
        s = next(iter(harvest_transcripts(claude_dir=base, use_cache=False)))
        payload = drift_signals(s)
        assert payload["source"] == "claude_transcripts"
        assert payload["session_id"] == "sess-1"
        assert "efficiency_drift" in payload["signals"]
        assert "total_cost_usd" in payload["signals"]
        assert "wall_seconds" in payload["signals"]
        assert "active_seconds" in payload["signals"]
        assert "unknown_line_types" in payload["signals"]
        assert "unknown_models" in payload["signals"]
        assert "activity_breakdown" in payload


# ---------------------------------------------------------------------------
# sequence-aware activity reclassification
# ---------------------------------------------------------------------------


def _tool_use(name: str) -> dict:
    return {"type": "tool_use", "name": name}


def _tool_result_error(msg: str = "boom") -> dict:
    return {"type": "tool_result", "is_error": True, "content": msg}


class TestSequenceActivity:
    def test_debug_loop_upgrades_shell_to_debugging(self, tmp_path):
        base = tmp_path / "projects"
        d = base / "-home-arete-projects"
        d.mkdir(parents=True)
        # shell call, then a user turn carrying a tool_result error, then shell again
        lines = [
            _user_line("sess", "run it"),
            _assistant_line("sess", tool_calls=[_tool_use("Bash")]),
            # user turn with tool_result is_error=True
            json.dumps(
                {
                    "type": "user",
                    "sessionId": "sess",
                    "timestamp": "2026-04-10T09:03:58Z",
                    "cwd": "/home/arete/projects",
                    "message": {"role": "user", "content": [_tool_result_error("cmd failed")]},
                }
            ),
            _assistant_line("sess", tool_calls=[_tool_use("Bash")]),
        ]
        _write_session(d, "sess.jsonl", lines)
        s = parse_session(d / "sess.jsonl", claude_base=base)
        # Both shell turns should be reclassified as debugging
        assert s is not None
        assert "debugging" in s.activity_breakdown
        assert s.activity_breakdown["debugging"] > 0

    def test_exploration_burst_of_reads(self, tmp_path):
        base = tmp_path / "projects"
        d = base / "-home-arete-projects"
        d.mkdir(parents=True)
        # 5 consecutive assistant Reads between user turns
        lines: list[str] = []
        for i in range(6):
            lines.append(_user_line("sess", f"q{i}"))
            lines.append(_assistant_line("sess", tool_calls=[_tool_use("Read")]))
        _write_session(d, "sess.jsonl", lines)
        s = parse_session(d / "sess.jsonl", claude_base=base)
        assert s is not None
        # The read burst should upgrade file_read turns to exploration
        assert s.activity_breakdown.get("exploration", 0) > 0


# ---------------------------------------------------------------------------
# schema guard + pricing staleness
# ---------------------------------------------------------------------------


class TestSchemaAndPricing:
    def test_unknown_line_types_tracked(self, tmp_path):
        base = tmp_path / "projects"
        d = base / "-home-arete-projects"
        d.mkdir(parents=True)
        lines = [
            _user_line("sess", "hi"),
            json.dumps({"type": "new-cc-line-type-v2", "timestamp": "2026-04-10T09:03:58Z"}),
            _assistant_line("sess", "ok"),
        ]
        _write_session(d, "sess.jsonl", lines)
        s = parse_session(d / "sess.jsonl", claude_base=base)
        assert s is not None
        assert "new-cc-line-type-v2" in s.unknown_line_types
        # Known types should not be flagged
        assert all(t not in KNOWN_LINE_TYPES for t in s.unknown_line_types)

    def test_unpriced_model_tracked(self, tmp_path):
        base = tmp_path / "projects"
        d = base / "-home-arete-projects"
        d.mkdir(parents=True)
        lines = [
            _user_line("sess", "hi"),
            _assistant_line("sess", model="claude-omega-9-1-future", text="ok"),
        ]
        _write_session(d, "sess.jsonl", lines)
        s = parse_session(d / "sess.jsonl", claude_base=base)
        assert s is not None
        assert "claude-omega-9-1-future" in s.unknown_models

    def test_known_models_not_flagged(self, tmp_path):
        base = tmp_path / "projects"
        d = base / "-home-arete-projects"
        d.mkdir(parents=True)
        lines = [
            _user_line("sess", "hi"),
            _assistant_line("sess", model="claude-opus-4-7"),
        ]
        _write_session(d, "sess.jsonl", lines)
        s = parse_session(d / "sess.jsonl", claude_base=base)
        assert s is not None
        assert s.unknown_models == []


# ---------------------------------------------------------------------------
# active vs wall duration
# ---------------------------------------------------------------------------


def _assistant_line_ts(sid: str, ts: str) -> str:
    return json.dumps(
        {
            "type": "assistant",
            "sessionId": sid,
            "timestamp": ts,
            "cwd": "/home/arete/projects",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        }
    )


class TestDuration:
    def test_active_seconds_excludes_long_idle(self, tmp_path):
        base = tmp_path / "projects"
        d = base / "-home-arete-projects"
        d.mkdir(parents=True)
        # 4 turns: 0s, 10s, +30min idle, +5s — wall = ~30m5s, active ~15s
        lines = [
            _assistant_line_ts("sess", "2026-04-10T09:00:00Z"),
            _assistant_line_ts("sess", "2026-04-10T09:00:10Z"),
            _assistant_line_ts("sess", "2026-04-10T09:30:10Z"),
            _assistant_line_ts("sess", "2026-04-10T09:30:15Z"),
        ]
        _write_session(d, "sess.jsonl", lines)
        s = parse_session(d / "sess.jsonl", claude_base=base)
        assert s is not None
        # Wall should be ~1815s, active ~15s (two gaps of ≤5min summed)
        assert s.duration_seconds is not None
        assert s.active_seconds is not None
        assert s.duration_seconds > 1800  # 30 min
        assert s.active_seconds < 60  # 15s (10s + 5s)
        assert s.active_seconds < s.duration_seconds

    def test_active_seconds_none_for_single_turn(self, tmp_path):
        base = tmp_path / "projects"
        d = base / "-home-arete-projects"
        d.mkdir(parents=True)
        lines = [_assistant_line_ts("sess", "2026-04-10T09:00:00Z")]
        _write_session(d, "sess.jsonl", lines)
        s = parse_session(d / "sess.jsonl", claude_base=base)
        assert s is not None
        assert s.active_seconds is None
        assert s.duration_seconds is None
