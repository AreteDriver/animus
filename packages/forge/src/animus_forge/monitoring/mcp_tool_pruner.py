"""MCP tool-usage analyzer — proposes pruned allowlists.

Pairs with ``mcp_tool_usage`` (the recorder). Reads the JSONL log,
aggregates per-workflow tool calls, and produces a pruning proposal
for each workflow:

- ``used``: ``{server: [tool, ...]}`` — tools that were actually
  called at least once in the analysis window.
- ``unused``: tools registered in the workflow's configured tool-set
  but never observed in the log (only populated when the caller
  supplies the configured set).
- ``call_counts``: ``{server: {tool: int}}`` — raw aggregate so a
  human reviewer can sanity-check the proposal before approving.

The analyzer does *not* mutate any workflow config. It returns a
plain dict that a self-improvement source — or a human running the
CLI — can turn into a config patch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .mcp_tool_usage import (
    MCPToolUsageRecord,
    aggregate_tool_calls,
    read_usage_records,
)


@dataclass
class PruneProposal:
    """One workflow's pruning proposal."""

    workflow_id: str
    sample_size: int  # Total record count in the analysis window
    used: dict[str, list[str]] = field(default_factory=dict)
    unused: dict[str, list[str]] = field(default_factory=dict)
    call_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def has_pruning_opportunity(self) -> bool:
        """True if any registered tool was never called in-window."""
        return any(tools for tools in self.unused.values())


def analyze_workflow_usage(
    workflow_id: str,
    log_path: Path | None = None,
    configured_tools: dict[str, list[str]] | None = None,
    min_samples: int = 10,
) -> PruneProposal:
    """Build a pruning proposal for a single workflow.

    Args:
        workflow_id: Workflow ID to filter on.
        log_path: Override the default usage-log path (test injection).
        configured_tools: ``{server: [tool, ...]}`` of tools the workflow
            *could* call. When supplied, the proposal lists unused tools
            as well as used ones. When omitted, ``unused`` stays empty
            and the proposal is "what got called" only.
        min_samples: Refuse to propose pruning below this record count
            to avoid acting on a sparse sample. Returns the partial
            proposal so the caller can still inspect call_counts.

    Returns:
        PruneProposal — never raises; empty proposal when the log has
        no records for this workflow.
    """
    records: list[MCPToolUsageRecord] = read_usage_records(
        log_path=log_path, workflow_id=workflow_id
    )
    proposal = PruneProposal(workflow_id=workflow_id, sample_size=len(records))

    if not records:
        return proposal

    proposal.call_counts = aggregate_tool_calls(records)

    if len(records) < min_samples:
        # Not enough signal — caller sees the call_counts but no
        # used/unused split (the recommendation is "need more data").
        return proposal

    # Populate used set
    for server, tools in proposal.call_counts.items():
        proposal.used[server] = sorted(tools.keys())

    # Populate unused set (requires configured_tools)
    if configured_tools:
        for server, tools in configured_tools.items():
            called = proposal.call_counts.get(server, {})
            unused_here = [t for t in tools if t not in called]
            if unused_here:
                proposal.unused[server] = sorted(unused_here)

    return proposal


def format_proposal_summary(proposal: PruneProposal) -> str:
    """Human-readable single-paragraph summary for CLI / dashboard."""
    if proposal.sample_size == 0:
        return f"workflow {proposal.workflow_id!r}: no recorded MCP tool calls"
    used_count = sum(len(tools) for tools in proposal.used.values())
    unused_count = sum(len(tools) for tools in proposal.unused.values())
    parts = [
        f"workflow {proposal.workflow_id!r}: {proposal.sample_size} record(s),",
        f"{used_count} tool(s) actually called across {len(proposal.used)} server(s)",
    ]
    if unused_count:
        parts.append(f"— {unused_count} unused tool(s) safe to prune")
    return " ".join(parts)
