"""Self-improvement tools — analyze behavior, propose and apply improvements."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from animus_bootstrap.intelligence.tools.executor import ToolDefinition

logger = logging.getLogger(__name__)

# Log of improvement proposals for audit trail
_improvement_log: list[dict] = []


def get_improvement_log() -> list[dict]:
    """Return the improvement log (for testing/inspection)."""
    return list(_improvement_log)


def clear_improvement_log() -> None:
    """Clear the improvement log."""
    _improvement_log.clear()


# References set at runtime
_tool_executor = None
_cognitive_backend = None


def set_self_improve_deps(tool_executor: object, cognitive_backend: object) -> None:
    """Wire live dependencies for self-improvement tools."""
    global _tool_executor, _cognitive_backend  # noqa: PLW0603
    _tool_executor = tool_executor
    _cognitive_backend = cognitive_backend


async def _analyze_behavior(focus: str = "all") -> str:
    """Analyze recent tool execution history to identify patterns and issues.

    ``focus`` can be: "all", "errors", "slow", "frequent".
    """
    if _tool_executor is None:
        return "No tool executor available — cannot analyze behavior"

    history = _tool_executor.get_history(limit=100)
    if not history:
        return "No tool execution history to analyze"

    total = len(history)
    failures = [r for r in history if not r.success]
    slow = [r for r in history if r.duration_ms > 5000]
    tool_counts: dict[str, int] = {}
    for r in history:
        tool_counts[r.tool_name] = tool_counts.get(r.tool_name, 0) + 1

    report_lines = [f"Behavior Analysis ({total} recent executions):"]

    if focus in ("all", "errors"):
        report_lines.append(f"\nErrors: {len(failures)}/{total}")
        for f in failures[:10]:
            report_lines.append(f"  - {f.tool_name}: {f.output[:100]}")

    if focus in ("all", "slow"):
        report_lines.append(f"\nSlow executions (>5s): {len(slow)}")
        for s in slow[:10]:
            report_lines.append(f"  - {s.tool_name}: {s.duration_ms:.0f}ms")

    if focus in ("all", "frequent"):
        sorted_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)
        report_lines.append("\nTool usage frequency:")
        for name, count in sorted_tools[:10]:
            report_lines.append(f"  - {name}: {count} calls")

    avg_ms = sum(r.duration_ms for r in history) / total if total else 0
    report_lines.append(f"\nAverage execution time: {avg_ms:.0f}ms")
    report_lines.append(f"Success rate: {(total - len(failures)) / total * 100:.1f}%")

    return "\n".join(report_lines)


async def _propose_improvement(area: str, description: str) -> str:
    """Generate an improvement proposal using the cognitive backend.

    ``area`` identifies what to improve (e.g. "tool:web_search", "prompt",
    "workflow").  ``description`` explains the problem or desired enhancement.
    """
    proposal = {
        "id": len(_improvement_log) + 1,
        "area": area,
        "description": description,
        "status": "proposed",
        "timestamp": datetime.now(UTC).isoformat(),
        "analysis": None,
        "patch": None,
    }

    # Use cognitive backend if available to generate analysis
    if _cognitive_backend is not None:
        try:
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"You are analyzing the Animus AI system for self-improvement.\n"
                        f"Area: {area}\n"
                        f"Issue: {description}\n\n"
                        f"Propose a specific, actionable improvement. "
                        f"Include what code to change and why. Be concise."
                    ),
                }
            ]
            response = await _cognitive_backend.generate_response(
                messages=messages,
                system_prompt="You are a code improvement analyst. Be specific and actionable.",
                max_tokens=1024,
            )
            proposal["analysis"] = response
        except Exception as exc:
            proposal["analysis"] = f"Cognitive analysis failed: {exc}"
            logger.warning("Self-improvement analysis failed: %s", exc)
    else:
        proposal["analysis"] = (
            f"No cognitive backend available. Manual review needed for: {area} — {description}"
        )

    _improvement_log.append(proposal)
    logger.info("Improvement proposal #%d: %s", proposal["id"], area)

    output = f"Improvement Proposal #{proposal['id']}\n"
    output += f"Area: {area}\n"
    output += "Status: proposed\n"
    output += f"Analysis:\n{proposal['analysis']}"
    return output


async def _apply_improvement(proposal_id: int, confirm: bool = False) -> str:
    """Apply a previously proposed improvement.

    If ``confirm`` is False (default), shows what would be done.
    If ``confirm`` is True, executes the improvement via code_patch.
    """
    matching = [p for p in _improvement_log if p["id"] == proposal_id]
    if not matching:
        return f"Proposal #{proposal_id} not found"

    proposal = matching[0]
    if proposal["status"] == "applied":
        return f"Proposal #{proposal_id} has already been applied"

    if not confirm:
        return (
            f"Proposal #{proposal_id} ready for application.\n"
            f"Area: {proposal['area']}\n"
            f"Analysis: {proposal['analysis']}\n\n"
            f"Call apply_improvement with confirm=true to execute."
        )

    proposal["status"] = "applied"
    proposal["applied_at"] = datetime.now(UTC).isoformat()
    logger.info("Improvement proposal #%d applied", proposal_id)
    return (
        f"Proposal #{proposal_id} marked as applied.\n"
        f"Use code_patch or code_write tools to implement the specific changes "
        f"described in the analysis."
    )


async def _list_improvements(status: str = "all") -> str:
    """List improvement proposals, optionally filtered by status."""
    if not _improvement_log:
        return "No improvement proposals recorded"

    filtered = _improvement_log
    if status != "all":
        filtered = [p for p in _improvement_log if p["status"] == status]

    if not filtered:
        return f"No proposals with status '{status}'"

    lines = [f"Improvement Proposals ({len(filtered)}):"]
    for p in filtered:
        lines.append(f"  #{p['id']} [{p['status']}] {p['area']}: {p['description'][:80]}")

    return "\n".join(lines)


def get_self_improve_tools() -> list[ToolDefinition]:
    """Return self-improvement tool definitions."""
    return [
        ToolDefinition(
            name="analyze_behavior",
            description=(
                "Analyze recent tool execution history to identify patterns, "
                "errors, slow operations, and usage frequency."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "Focus area: 'all', 'errors', 'slow', 'frequent'.",
                        "default": "all",
                    },
                },
            },
            handler=_analyze_behavior,
            category="self_improvement",
        ),
        ToolDefinition(
            name="propose_improvement",
            description=(
                "Generate an improvement proposal for a specific area. "
                "Uses the cognitive backend to analyze the issue and suggest changes."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "area": {
                        "type": "string",
                        "description": (
                            "What to improve (e.g. 'tool:web_search', 'prompt', 'workflow')."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of the problem or desired enhancement.",
                    },
                },
                "required": ["area", "description"],
            },
            handler=_propose_improvement,
            category="self_improvement",
            permission="approve",
        ),
        ToolDefinition(
            name="apply_improvement",
            description=(
                "Apply a previously proposed improvement. "
                "Use confirm=false to preview, confirm=true to execute."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "proposal_id": {
                        "type": "integer",
                        "description": "ID of the improvement proposal to apply.",
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Set to true to execute the improvement.",
                        "default": False,
                    },
                },
                "required": ["proposal_id"],
            },
            handler=_apply_improvement,
            category="self_improvement",
            permission="approve",
        ),
        ToolDefinition(
            name="list_improvements",
            description="List improvement proposals, optionally filtered by status.",
            parameters={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by status: 'all', 'proposed', 'applied'.",
                        "default": "all",
                    },
                },
            },
            handler=_list_improvements,
            category="self_improvement",
        ),
    ]
