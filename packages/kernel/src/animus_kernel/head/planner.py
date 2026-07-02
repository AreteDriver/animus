"""Planner for Animus Head — generates tool-call plans from parsed intents.

Takes a ParsedIntent and produces an ordered sequence of tool calls.
Handles dependencies (e.g., list_files before read_file) and
early-stops when a plan is impossible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from animus_kernel.head.intent_parser import IntentType, ParsedIntent


@dataclass
class ToolPlanStep:
    """A single step in a tool execution plan."""

    tool_name: str
    arguments: dict = field(default_factory=dict)
    reason: str = ""
    depends_on: list[int] = field(default_factory=list)  # Indices of prior steps


@dataclass
class ToolPlan:
    """A generated plan for executing a user request."""

    steps: list[ToolPlanStep] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    requires_clarification: bool = False
    clarification_prompt: str = ""


class HeadPlanner:
    """Generates tool-call plans from parsed intents.

    Uses rule-based templates for common task patterns.
    No model call required for plan generation.
    """

    # Tool dependencies: some tools need prior steps to gather context.
    # These are only added when the extracted args don't already provide
    # sufficient context (e.g., edit_file always needs read_file first).
    DEPENDENCIES: dict[str, list[str]] = {
        "edit_file": ["read_file"],  # Must read before editing
        "search_code": [],  # Independent
        "run_tests": [],  # Independent
        "run_linter": [],  # Independent
    }

    def plan(self, intent: ParsedIntent, project_root: str = ".") -> ToolPlan:
        """Generate a tool execution plan from a parsed intent.

        Args:
            intent: ParsedIntent from HeadIntentParser
            project_root: Project root path for default arguments

        Returns:
            ToolPlan with ordered steps
        """
        if intent.intent_type == IntentType.CONVERSATIONAL:
            return ToolPlan(
                steps=[],
                confidence=1.0,
                reason="conversational — no tools needed",
            )

        if intent.intent_type == IntentType.CLARIFICATION_NEEDED:
            tools_str = ", ".join(intent.suggested_tools[:3])
            return ToolPlan(
                steps=[],
                confidence=0.5,
                requires_clarification=True,
                clarification_prompt=(
                    f"I can help with that, but I'm not sure which approach to take. "
                    f"Possible tools: {tools_str}. Could you be more specific?"
                ),
                reason="ambiguous intent",
            )

        if intent.intent_type == IntentType.DIRECT_COMMAND:
            return self._plan_direct_command(intent, project_root)

        # Vague request — generate a heuristic plan
        return self._plan_vague_request(intent, project_root)

    def _plan_direct_command(self, intent: ParsedIntent, project_root: str) -> ToolPlan:
        """Generate plan for a direct command intent."""
        steps: list[ToolPlanStep] = []
        seen_tools: set[str] = set()

        for tool_name in intent.suggested_tools:
            if tool_name in seen_tools:
                continue
            seen_tools.add(tool_name)

            # Build arguments from extracted values or defaults
            args = self._build_args(tool_name, intent.extracted_args, project_root)

            # Check if we need dependency steps
            for dep in self.DEPENDENCIES.get(tool_name, []):
                if dep not in seen_tools:
                    seen_tools.add(dep)
                    dep_args = self._build_args(dep, intent.extracted_args, project_root)
                    steps.append(
                        ToolPlanStep(
                            tool_name=dep,
                            arguments=dep_args,
                            reason=f"gather context before {tool_name}",
                        )
                    )

            steps.append(
                ToolPlanStep(
                    tool_name=tool_name,
                    arguments=args,
                    reason=f"execute direct command: {tool_name}",
                )
            )

        return ToolPlan(
            steps=steps,
            confidence=intent.confidence,
            reason="direct command plan",
        )

    def _plan_vague_request(self, intent: ParsedIntent, project_root: str) -> ToolPlan:
        """Generate plan for a vague request using heuristics."""
        steps: list[ToolPlanStep] = []
        seen_tools: set[str] = set()

        # For vague requests, be conservative — start with discovery
        discovery_tools = ["list_files", "search_code", "recall"]
        action_tools = [t for t in intent.suggested_tools if t not in discovery_tools]

        # Add discovery steps first
        for tool_name in discovery_tools:
            if tool_name in intent.suggested_tools and tool_name not in seen_tools:
                seen_tools.add(tool_name)
                args = self._build_args(tool_name, intent.extracted_args, project_root)
                steps.append(
                    ToolPlanStep(
                        tool_name=tool_name,
                        arguments=args,
                        reason="discover context for vague request",
                    )
                )

        # Then add action steps
        for tool_name in action_tools:
            if tool_name not in seen_tools:
                seen_tools.add(tool_name)
                args = self._build_args(tool_name, intent.extracted_args, project_root)
                steps.append(
                    ToolPlanStep(
                        tool_name=tool_name,
                        arguments=args,
                        reason="execute based on discovered context",
                    )
                )

        return ToolPlan(
            steps=steps,
            confidence=intent.confidence * 0.8,  # Lower confidence for vague plans
            reason="heuristic plan for vague request",
        )

    @staticmethod
    def _build_args(tool_name: str, extracted: dict, project_root: str) -> dict:
        """Build arguments for a tool from extracted values and defaults."""
        args: dict = {}

        if tool_name == "read_file":
            if "path" in extracted:
                args["path"] = extracted["path"]
            else:
                args["path"] = "."

        elif tool_name == "write_file":
            if "path" in extracted:
                args["path"] = extracted["path"]
            if "content" in extracted:
                args["content"] = extracted["content"]

        elif tool_name == "edit_file":
            if "path" in extracted:
                args["path"] = extracted["path"]
            if "old_string" in extracted:
                args["old_string"] = extracted["old_string"]
            if "new_string" in extracted:
                args["new_string"] = extracted["new_string"]

        elif tool_name == "list_files":
            args["path"] = extracted.get("path", ".")

        elif tool_name == "search_code":
            if "query" in extracted:
                args["query"] = extracted["query"]
            if "path" in extracted:
                args["path"] = extracted["path"]

        elif tool_name == "run_shell":
            if "command" in extracted:
                args["command"] = extracted["command"]

        elif tool_name == "remember":
            if "content" in extracted:
                args["content"] = extracted["content"]
                args["tags"] = ["auto"]

        elif tool_name == "recall":
            if "query" in extracted:
                args["query"] = extracted["query"]
            else:
                args["query"] = project_root
            args["limit"] = 5

        elif tool_name == "create_task":
            if "description" in extracted:
                args["description"] = extracted["description"]

        elif tool_name == "run_tests":
            args["command"] = "pytest -x"

        elif tool_name == "run_linter":
            args["command"] = "ruff check ."

        return args

    @staticmethod
    def estimate_cost(plan: ToolPlan) -> int:
        """Estimate token cost of executing a plan (rough heuristic)."""
        # Each tool call costs roughly 500-1000 tokens in context
        return len(plan.steps) * 800
