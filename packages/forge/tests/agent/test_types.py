"""Tests for animus_forge.agent.types — receipt schema, exit codes, hook decisions."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from animus_forge.agent.types import (
    EXIT_CODES,
    AgentContext,
    Allow,
    Deny,
    Receipt,
    ReceiptStatus,
    Skill,
    ToolCallRecord,
    ToolError,
    ToolResult,
)


class TestReceiptStatus:
    def test_all_statuses_have_exit_codes(self):
        """Every ReceiptStatus must map to an exit code (spec R9)."""
        for status in ReceiptStatus:
            assert status in EXIT_CODES

    def test_exit_codes_match_spec_r9(self):
        """Exact mapping required by spec R9."""
        assert EXIT_CODES[ReceiptStatus.SUCCESS] == 0
        assert EXIT_CODES[ReceiptStatus.MAX_TURNS] == 1
        assert EXIT_CODES[ReceiptStatus.BUDGET_EXCEEDED] == 2
        assert EXIT_CODES[ReceiptStatus.WALL_TIME_EXCEEDED] == 3
        assert EXIT_CODES[ReceiptStatus.TOOL_ERROR] == 4
        assert EXIT_CODES[ReceiptStatus.AGENT_QUIT] == 5
        assert EXIT_CODES[ReceiptStatus.INTERNAL_ERROR] == 6

    def test_status_values_are_strings(self):
        """Status enum values must be lowercase string identifiers for JSON."""
        for status in ReceiptStatus:
            assert isinstance(status.value, str)
            assert status.value == status.value.lower()


class TestHookDecisions:
    def test_allow_default_reason_empty(self):
        assert Allow().reason == ""

    def test_allow_with_reason(self):
        a = Allow("test ok")
        assert a.reason == "test ok"

    def test_allow_is_frozen(self):
        a = Allow()
        with pytest.raises((AttributeError, Exception)):
            a.reason = "mutated"

    def test_deny_requires_reason(self):
        d = Deny("policy violation")
        assert d.reason == "policy violation"

    def test_deny_is_frozen(self):
        d = Deny("x")
        with pytest.raises((AttributeError, Exception)):
            d.reason = "mutated"

    def test_allow_and_deny_distinguishable(self):
        """isinstance checks must work for the loop's branching."""
        a, d = Allow(), Deny("no")
        assert isinstance(a, Allow) and not isinstance(a, Deny)
        assert isinstance(d, Deny) and not isinstance(d, Allow)


class TestToolError:
    def test_tool_error_is_exception(self):
        assert issubclass(ToolError, Exception)

    def test_tool_error_carries_message(self):
        err = ToolError("file not found: foo")
        assert str(err) == "file not found: foo"


class TestToolResult:
    def test_tool_result_ok(self):
        r = ToolResult(tool="ReadFile", status="ok", output="contents", wall_ms=1.2)
        assert r.tool == "ReadFile"
        assert r.status == "ok"
        assert r.error is None

    def test_tool_result_with_error(self):
        r = ToolResult(
            tool="Bash", status="error", output=None, wall_ms=0.5, error="exit 1"
        )
        assert r.error == "exit 1"


class TestToolCallRecord:
    def test_round_trip_fields(self):
        rec = ToolCallRecord(
            turn=3, tool="Grep", args_summary="pattern='foo'", status="ok", wall_ms=15.0
        )
        assert rec.turn == 3
        assert rec.tool == "Grep"
        assert rec.status == "ok"


class TestSkill:
    def test_skill_construction(self, tmp_path: Path):
        src = tmp_path / "SKILL.md"
        skill = Skill(
            name="demo",
            description="a demo skill",
            body="# Body",
            invokable=True,
            source_path=src,
        )
        assert skill.name == "demo"
        assert skill.invokable is True
        assert skill.source_path == src


class TestAgentContext:
    def test_minimal_context(self, tmp_path: Path):
        ctx = AgentContext(
            task="echo",
            project_root=tmp_path,
            provider="llamacpp",
            model="qwen3.6-research",
        )
        assert ctx.task == "echo"
        assert ctx.turn == 0
        assert ctx.max_turns == 50  # spec §4.1 default
        assert ctx.budget_tokens == 200_000
        assert ctx.wall_seconds == 1800
        assert ctx.metadata == {}

    def test_started_at_default_is_utc(self, tmp_path: Path):
        ctx = AgentContext(
            task="x", project_root=tmp_path, provider="llamacpp", model="m"
        )
        assert ctx.started_at.tzinfo is UTC

    def test_metadata_is_per_instance(self, tmp_path: Path):
        """field(default_factory=dict) must not be shared across instances."""
        a = AgentContext(task="a", project_root=tmp_path, provider="p", model="m")
        b = AgentContext(task="b", project_root=tmp_path, provider="p", model="m")
        a.metadata["key"] = "value"
        assert "key" not in b.metadata


class TestReceipt:
    @pytest.fixture
    def sample_receipt(self) -> Receipt:
        return Receipt(
            task="echo hello",
            status=ReceiptStatus.SUCCESS,
            turns_taken=2,
            tokens_in=42,
            tokens_out=17,
            wall_seconds=3.14,
            model="qwen3.6-research",
            tool_calls=[
                ToolCallRecord(
                    turn=1,
                    tool="WriteFile",
                    args_summary="path='out.txt'",
                    status="ok",
                    wall_ms=12.0,
                ),
                ToolCallRecord(
                    turn=2,
                    tool="ReadFile",
                    args_summary="path='out.txt'",
                    status="ok",
                    wall_ms=4.5,
                ),
            ],
            skill_load_warnings=["skill_x failed to parse: bad frontmatter"],
            hook_errors=[],
            final_message="done",
            error=None,
            started_at=datetime(2026, 5, 24, 12, 0, 0, tzinfo=UTC),
            ended_at=datetime(2026, 5, 24, 12, 0, 3, tzinfo=UTC),
        )

    def test_exit_code_routes_through_status(self, sample_receipt: Receipt):
        assert sample_receipt.exit_code() == 0

    def test_exit_code_nonzero_on_failure(self):
        r = Receipt(
            task="x",
            status=ReceiptStatus.BUDGET_EXCEEDED,
            turns_taken=10,
            tokens_in=0,
            tokens_out=0,
            wall_seconds=1.0,
            model="m",
            tool_calls=[],
            skill_load_warnings=[],
            hook_errors=[],
            final_message="",
            error="over budget",
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
        )
        assert r.exit_code() == 2

    def test_to_dict_has_all_schema_keys(self, sample_receipt: Receipt):
        """Receipt.to_dict() must produce every key in spec §4.9."""
        d = sample_receipt.to_dict()
        required = {
            "task",
            "status",
            "turns_taken",
            "tokens_in",
            "tokens_out",
            "wall_seconds",
            "model",
            "tool_calls",
            "skill_load_warnings",
            "hook_errors",
            "final_message",
            "error",
            "started_at",
            "ended_at",
        }
        assert set(d.keys()) == required

    def test_to_dict_status_is_string(self, sample_receipt: Receipt):
        """JSON schema requires status as string value, not Enum."""
        d = sample_receipt.to_dict()
        assert d["status"] == "success"
        assert isinstance(d["status"], str)

    def test_to_dict_timestamps_iso_format(self, sample_receipt: Receipt):
        d = sample_receipt.to_dict()
        assert d["started_at"] == "2026-05-24T12:00:00+00:00"
        assert d["ended_at"] == "2026-05-24T12:00:03+00:00"

    def test_to_dict_tool_calls_are_dicts(self, sample_receipt: Receipt):
        """tool_calls must serialize to plain dicts (not dataclass reprs)."""
        d = sample_receipt.to_dict()
        assert len(d["tool_calls"]) == 2
        first = d["tool_calls"][0]
        assert first == {
            "turn": 1,
            "tool": "WriteFile",
            "args_summary": "path='out.txt'",
            "status": "ok",
            "wall_ms": 12.0,
        }

    def test_to_dict_lists_are_copies(self, sample_receipt: Receipt):
        """skill_load_warnings/hook_errors lists must not be aliases."""
        d = sample_receipt.to_dict()
        d["skill_load_warnings"].append("mutated")
        assert "mutated" not in sample_receipt.skill_load_warnings

    def test_to_dict_is_json_serializable(self, sample_receipt: Receipt):
        import json

        json.dumps(sample_receipt.to_dict())  # raises if not serializable
