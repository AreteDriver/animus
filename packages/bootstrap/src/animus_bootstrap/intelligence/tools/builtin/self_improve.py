"""Self-improvement tools — analyze behavior, propose and apply improvements."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from animus_bootstrap.intelligence.tools.executor import ToolDefinition

logger = logging.getLogger(__name__)

# Log of improvement proposals for audit trail (in-memory fallback)
_improvement_log: list[dict] = []

# Persistent store (set at runtime)
_improvement_store = None


def set_improvement_store(store: object | None) -> None:
    """Wire the persistent improvement store."""
    global _improvement_store  # noqa: PLW0603
    _improvement_store = store


def get_improvement_log() -> list[dict]:
    """Return the improvement log (for testing/inspection)."""
    if _improvement_store is not None:
        return _improvement_store.list_all()
    return list(_improvement_log)


def clear_improvement_log() -> None:
    """Clear the in-memory improvement log."""
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
        except (ConnectionError, TimeoutError, RuntimeError, ValueError) as exc:
            proposal["analysis"] = f"Cognitive analysis failed: {exc}"
            logger.warning("Self-improvement analysis failed: %s", exc)
    else:
        proposal["analysis"] = (
            f"No cognitive backend available. Manual review needed for: {area} — {description}"
        )

    # Persist to store or in-memory log
    if _improvement_store is not None:
        proposal["id"] = _improvement_store.save(proposal)
    else:
        proposal["id"] = len(_improvement_log) + 1
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
    # Look up proposal from store or in-memory log
    if _improvement_store is not None:
        proposal = _improvement_store.get(proposal_id)
    else:
        matching = [p for p in _improvement_log if p["id"] == proposal_id]
        proposal = matching[0] if matching else None

    if proposal is None:
        return f"Proposal #{proposal_id} not found"

    if proposal["status"] == "applied":
        return f"Proposal #{proposal_id} has already been applied"

    if not confirm:
        return (
            f"Proposal #{proposal_id} ready for application.\n"
            f"Area: {proposal['area']}\n"
            f"Analysis: {proposal['analysis']}\n\n"
            f"Call apply_improvement with confirm=true to execute."
        )

    applied_at = datetime.now(UTC).isoformat()
    if _improvement_store is not None:
        _improvement_store.update_status(proposal_id, "applied", applied_at)
    else:
        proposal["status"] = "applied"
        proposal["applied_at"] = applied_at

    logger.info("Improvement proposal #%d applied", proposal_id)
    return (
        f"Proposal #{proposal_id} marked as applied.\n"
        f"Use code_patch or code_write tools to implement the specific changes "
        f"described in the analysis."
    )


async def _list_improvements(status: str = "all") -> str:
    """List improvement proposals, optionally filtered by status."""
    if _improvement_store is not None:
        filtered = _improvement_store.list_all(status=status)
    else:
        filtered = _improvement_log
        if status != "all":
            filtered = [p for p in _improvement_log if p["status"] == status]

    if not filtered:
        if status == "all":
            return "No improvement proposals recorded"
        return f"No proposals with status '{status}'"

    lines = [f"Improvement Proposals ({len(filtered)}):"]
    for p in filtered:
        lines.append(f"  #{p['id']} [{p['status']}] {p['area']}: {p['description'][:80]}")

    return "\n".join(lines)


async def _self_improve_loop(area: str, description: str) -> str:
    """Run a multi-turn self-improvement loop.

    Steps:
    1. Analyze recent behavior for context
    2. Propose an improvement using the cognitive backend
    3. Report findings for human review

    The loop does NOT auto-apply patches — it stops after proposal
    so a human can review and approve via apply_improvement.
    """
    steps: list[str] = []

    # Step 1: Analyze
    steps.append("=== Step 1: Behavior Analysis ===")
    analysis = await _analyze_behavior(focus="all")
    steps.append(analysis)

    # Step 2: Propose
    steps.append("\n=== Step 2: Improvement Proposal ===")
    proposal_result = await _propose_improvement(area, description)
    steps.append(proposal_result)

    # Step 3: Report
    steps.append("\n=== Step 3: Summary ===")
    log = get_improvement_log()
    if log:
        latest = log[-1]
        steps.append(f"Proposal #{latest['id']} created for area '{area}'.")
        steps.append(f"Status: {latest['status']}")
        if latest.get("analysis"):
            steps.append(f"Analysis preview: {latest['analysis'][:200]}...")
        steps.append(
            f"\nTo apply: call apply_improvement with proposal_id={latest['id']} and confirm=true."
        )
    else:
        steps.append("No proposal was created.")

    return "\n".join(steps)


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
        ToolDefinition(
            name="self_improve_loop",
            description=(
                "Run a multi-turn self-improvement loop: analyze behavior, "
                "propose an improvement, and report findings. Does NOT auto-apply "
                "— stops after proposal for human review."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "area": {
                        "type": "string",
                        "description": ("What to improve (e.g. 'tool:web_search', 'prompt')."),
                    },
                    "description": {
                        "type": "string",
                        "description": "Problem description or desired enhancement.",
                    },
                },
                "required": ["area", "description"],
            },
            handler=_self_improve_loop,
            category="self_improvement",
            permission="approve",
        ),
    ]
