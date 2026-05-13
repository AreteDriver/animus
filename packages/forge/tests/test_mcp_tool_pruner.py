"""Tests for the MCP tool-usage pruner analyzer.

Reads the JSONL log and proposes a pruned allowlist per workflow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from animus_forge.monitoring.mcp_tool_pruner import (
    analyze_workflow_usage,
    format_proposal_summary,
)
from animus_forge.monitoring.mcp_tool_usage import record_mcp_tool_usage


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "usage.jsonl"


def _bulk_record(log_path: Path, workflow_id: str, calls: list[tuple[str, str]]) -> None:
    """Helper: write (server, tool) pairs as recorded usage."""
    for i, (server, tool) in enumerate(calls):
        record_mcp_tool_usage(
            workflow_id=workflow_id,
            step_id=f"step-{i}",
            server=server,
            tool=tool,
            log_path=log_path,
        )


class TestEmptyLog:
    def test_empty_returns_empty_proposal(self, log_path: Path):
        proposal = analyze_workflow_usage("wf", log_path=log_path)
        assert proposal.sample_size == 0
        assert proposal.used == {}
        assert proposal.unused == {}
        assert not proposal.has_pruning_opportunity


class TestUsedSet:
    def test_used_set_reflects_actual_calls(self, log_path: Path):
        _bulk_record(
            log_path,
            "wf",
            [
                ("aurora", "search_lore"),
                ("aurora", "get_node"),
                ("aurora", "search_lore"),  # repeat
            ]
            * 4,
        )  # 12 calls — above min_samples
        proposal = analyze_workflow_usage("wf", log_path=log_path, min_samples=10)
        assert proposal.used == {"aurora": ["get_node", "search_lore"]}

    def test_call_counts_aggregate_correctly(self, log_path: Path):
        _bulk_record(
            log_path,
            "wf",
            [
                ("srv", "alpha"),
                ("srv", "alpha"),
                ("srv", "beta"),
            ]
            * 4,
        )  # alpha=8, beta=4
        proposal = analyze_workflow_usage("wf", log_path=log_path, min_samples=10)
        assert proposal.call_counts["srv"]["alpha"] == 8
        assert proposal.call_counts["srv"]["beta"] == 4


class TestUnusedSet:
    def test_unused_set_requires_configured_tools(self, log_path: Path):
        _bulk_record(log_path, "wf", [("aurora", "search_lore")] * 12)
        # No configured_tools → unused stays empty
        p1 = analyze_workflow_usage("wf", log_path=log_path)
        assert p1.unused == {}
        # With configured_tools — the never-called ones get flagged
        p2 = analyze_workflow_usage(
            "wf",
            log_path=log_path,
            configured_tools={"aurora": ["search_lore", "get_node", "search_meta"]},
        )
        assert p2.unused == {"aurora": ["get_node", "search_meta"]}
        assert p2.has_pruning_opportunity

    def test_no_unused_when_everything_called(self, log_path: Path):
        _bulk_record(
            log_path,
            "wf",
            [
                ("aurora", "search_lore"),
                ("aurora", "get_node"),
            ]
            * 6,
        )  # 12 calls — above min_samples
        proposal = analyze_workflow_usage(
            "wf",
            log_path=log_path,
            configured_tools={"aurora": ["search_lore", "get_node"]},
        )
        assert proposal.unused == {}
        assert not proposal.has_pruning_opportunity


class TestMinSamples:
    def test_below_min_samples_yields_call_counts_only(self, log_path: Path):
        _bulk_record(log_path, "wf", [("srv", "t")] * 5)
        proposal = analyze_workflow_usage("wf", log_path=log_path, min_samples=10)
        assert proposal.sample_size == 5
        assert proposal.call_counts == {"srv": {"t": 5}}
        # used/unused stay empty — "not enough data" signal
        assert proposal.used == {}
        assert proposal.unused == {}


class TestWorkflowIsolation:
    def test_other_workflow_records_ignored(self, log_path: Path):
        _bulk_record(log_path, "wf-A", [("srv", "alpha")] * 12)
        _bulk_record(log_path, "wf-B", [("srv", "beta")] * 12)
        a = analyze_workflow_usage("wf-A", log_path=log_path, min_samples=10)
        b = analyze_workflow_usage("wf-B", log_path=log_path, min_samples=10)
        assert a.used == {"srv": ["alpha"]}
        assert b.used == {"srv": ["beta"]}


class TestFormatSummary:
    def test_summary_for_empty_proposal(self, log_path: Path):
        proposal = analyze_workflow_usage("wf", log_path=log_path)
        assert "no recorded MCP tool calls" in format_proposal_summary(proposal)

    def test_summary_mentions_prunable_count(self, log_path: Path):
        _bulk_record(log_path, "wf", [("srv", "used")] * 12)
        proposal = analyze_workflow_usage(
            "wf",
            log_path=log_path,
            configured_tools={"srv": ["used", "unused1", "unused2"]},
            min_samples=10,
        )
        summary = format_proposal_summary(proposal)
        assert "2 unused tool" in summary
