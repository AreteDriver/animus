"""Tests for the MCP tool-usage recorder + reader.

Append-only JSONL log under the forge data directory. Used by the
mcp_tool_pruner to decide which registered tools are actually called.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from animus_forge.monitoring.mcp_tool_usage import (
    ENV_ENABLED,
    aggregate_tool_calls,
    is_recording_enabled,
    read_usage_records,
    record_mcp_tool_usage,
)


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "mcp_tool_usage.jsonl"


class TestRecordAndRead:
    def test_record_writes_jsonl_line(self, log_path: Path):
        record_mcp_tool_usage(
            workflow_id="wf1",
            step_id="s1",
            server="aurora-query",
            tool="search_lore",
            log_path=log_path,
        )
        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["workflow_id"] == "wf1"
        assert data["tool"] == "search_lore"
        assert data["is_error"] is False

    def test_record_appends(self, log_path: Path):
        for i in range(3):
            record_mcp_tool_usage(
                workflow_id="wf1",
                step_id=f"s{i}",
                server="srv",
                tool=f"t{i}",
                log_path=log_path,
            )
        records = read_usage_records(log_path=log_path)
        assert len(records) == 3
        assert {r.tool for r in records} == {"t0", "t1", "t2"}

    def test_read_filters_by_workflow(self, log_path: Path):
        record_mcp_tool_usage("wf-A", "s1", "srv", "t1", log_path=log_path)
        record_mcp_tool_usage("wf-B", "s1", "srv", "t2", log_path=log_path)
        records = read_usage_records(log_path=log_path, workflow_id="wf-A")
        assert len(records) == 1
        assert records[0].tool == "t1"

    def test_read_missing_log_returns_empty(self, tmp_path: Path):
        records = read_usage_records(log_path=tmp_path / "nope.jsonl")
        assert records == []

    def test_read_skips_malformed_lines(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "{not json}\n"
            + json.dumps(
                {
                    "workflow_id": "wf",
                    "step_id": "s",
                    "server": "srv",
                    "tool": "t",
                    "timestamp": "2026-05-13T00:00:00+00:00",
                }
            )
            + "\n"
        )
        records = read_usage_records(log_path=log_path)
        assert len(records) == 1
        assert records[0].tool == "t"


class TestRecordError:
    def test_error_records_with_is_error_true(self, log_path: Path):
        record_mcp_tool_usage(
            workflow_id="wf",
            step_id="s",
            server="srv",
            tool="flaky",
            log_path=log_path,
            is_error=True,
            duration_ms=42,
        )
        records = read_usage_records(log_path=log_path)
        assert records[0].is_error is True
        assert records[0].duration_ms == 42

    def test_record_write_failure_is_swallowed(self, tmp_path: Path):
        # Point at a path whose parent cannot be created (a file blocks dir)
        blocker = tmp_path / "blocker"
        blocker.write_text("not a dir")
        target = blocker / "log.jsonl"  # parent is a file → mkdir will fail
        # Should not raise even though writing is impossible
        record_mcp_tool_usage("wf", "s", "srv", "t", log_path=target)


class TestRecordingToggle:
    def test_disable_via_env(self, log_path: Path, monkeypatch):
        monkeypatch.setenv(ENV_ENABLED, "0")
        assert is_recording_enabled() is False
        record_mcp_tool_usage("wf", "s", "srv", "t", log_path=log_path)
        assert not log_path.exists()  # No write happened

    def test_default_is_on(self, monkeypatch):
        monkeypatch.delenv(ENV_ENABLED, raising=False)
        assert is_recording_enabled() is True


class TestAggregate:
    def test_aggregate_counts_calls_per_tool(self):
        from animus_forge.monitoring.mcp_tool_usage import MCPToolUsageRecord

        records = [
            MCPToolUsageRecord("wf", "s1", "srvA", "alpha"),
            MCPToolUsageRecord("wf", "s2", "srvA", "alpha"),
            MCPToolUsageRecord("wf", "s3", "srvA", "beta"),
            MCPToolUsageRecord("wf", "s4", "srvB", "gamma"),
        ]
        agg = aggregate_tool_calls(records)
        assert agg == {
            "srvA": {"alpha": 2, "beta": 1},
            "srvB": {"gamma": 1},
        }

    def test_aggregate_empty(self):
        assert aggregate_tool_calls([]) == {}
