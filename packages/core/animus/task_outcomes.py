"""
Task Outcome Tracker — learns from what worked and what broke.

Records task outcomes (request, tools used, success/failure, error details)
in MemoryLayer, recalls similar past tasks for context enrichment, and
extracts patterns across sessions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from animus.logging import get_logger

if TYPE_CHECKING:
    from animus.memory import MemoryLayer

logger = get_logger("task_outcomes")

OUTCOME_TAG = "task_outcome"
PATTERN_TAG = "task_pattern"


@dataclass
class TaskOutcome:
    """Record of a single task execution."""

    request: str
    success: bool
    tools_used: list[str] = field(default_factory=list)
    error: str | None = None
    response_summary: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_memory_content(self) -> str:
        """Format as a storable memory string."""
        status = "SUCCESS" if self.success else "FAILURE"
        parts = [f"[{status}] {self.request}"]
        if self.tools_used:
            parts.append(f"Tools: {', '.join(self.tools_used)}")
        if self.error:
            parts.append(f"Error: {self.error}")
        if self.response_summary:
            parts.append(f"Result: {self.response_summary[:200]}")
        return " | ".join(parts)

    def to_json(self) -> str:
        """Serialize to JSON for metadata storage."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> TaskOutcome:
        """Deserialize from JSON."""
        return cls(**json.loads(data))


@dataclass
class TaskPattern:
    """A pattern detected across multiple task outcomes."""

    description: str
    occurrences: int
    examples: list[str] = field(default_factory=list)
    suggestion: str = ""

    def to_memory_content(self) -> str:
        """Format as a storable memory string."""
        parts = [f"[PATTERN] {self.description} ({self.occurrences}x)"]
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)


class TaskOutcomeTracker:
    """Tracks task outcomes and learns from them.

    Uses MemoryLayer for persistence — outcomes are stored as memories
    tagged with 'task_outcome' for easy retrieval.
    """

    def __init__(self, memory: MemoryLayer):
        self.memory = memory

    def record(self, outcome: TaskOutcome) -> str:
        """Store a task outcome in memory.

        Returns:
            Memory ID of the stored outcome.
        """
        from animus.memory import MemoryType

        content = outcome.to_memory_content()
        tags = [OUTCOME_TAG]
        if outcome.success:
            tags.append("success")
        else:
            tags.append("failure")

        # Add tool names as tags for filtering
        for tool in outcome.tools_used:
            tags.append(f"tool:{tool}")

        mem = self.memory.remember(
            content=content,
            memory_type=MemoryType.PROCEDURAL,
            tags=tags,
            source="task_tracker",
            metadata={"outcome_json": outcome.to_json()},
        )
        logger.debug(f"Recorded task outcome: {outcome.request[:60]}... success={outcome.success}")
        return mem.id

    def recall_similar(self, request: str, limit: int = 3) -> list[TaskOutcome]:
        """Find past task outcomes similar to the given request.

        Uses vector similarity search via MemoryLayer.recall().
        """
        memories = self.memory.recall(
            query=request,
            tags=[OUTCOME_TAG],
            limit=limit,
        )

        outcomes = []
        for mem in memories:
            # Try to reconstruct from metadata JSON
            outcome_json = (mem.metadata or {}).get("outcome_json")
            if outcome_json:
                try:
                    outcomes.append(TaskOutcome.from_json(outcome_json))
                    continue
                except (json.JSONDecodeError, TypeError):
                    pass
            # Fallback: reconstruct from content string
            outcomes.append(
                TaskOutcome(
                    request=mem.content,
                    success="SUCCESS" in mem.content,
                    timestamp=mem.created_at.isoformat() if mem.created_at else "",
                )
            )

        return outcomes

    def get_context_for_task(self, request: str, limit: int = 3) -> str | None:
        """Build context string from similar past tasks.

        Returns None if no relevant outcomes found.
        """
        similar = self.recall_similar(request, limit=limit)
        if not similar:
            return None

        lines = ["Past similar tasks:"]
        for outcome in similar:
            status = "OK" if outcome.success else "FAILED"
            line = f"- [{status}] {outcome.request[:100]}"
            if outcome.tools_used:
                line += f" (tools: {', '.join(outcome.tools_used[:5])})"
            if outcome.error:
                line += f" — error: {outcome.error[:80]}"
            lines.append(line)

        return "\n".join(lines)

    def get_failure_patterns(self, limit: int = 20) -> list[TaskPattern]:
        """Detect common failure patterns from stored outcomes.

        Groups failures by error similarity and returns patterns
        that occur 2+ times.
        """
        failures = self.memory.recall_by_tags(tags=["task_outcome", "failure"], limit=limit)

        # Group by error substring (first 50 chars of error)
        error_groups: dict[str, list[str]] = {}
        for mem in failures:
            content = mem.content
            # Extract error from content
            if "Error:" in content:
                error = content.split("Error:")[1].strip()[:50]
            else:
                error = content[:50]

            error_groups.setdefault(error, []).append(content)

        patterns = []
        for error_key, examples in error_groups.items():
            if len(examples) >= 2:
                patterns.append(
                    TaskPattern(
                        description=f"Recurring failure: {error_key}",
                        occurrences=len(examples),
                        examples=examples[:3],
                        suggestion=_suggest_fix(error_key),
                    )
                )

        return patterns

    def get_success_rate(self) -> dict[str, int | float]:
        """Calculate overall and per-tool success rates."""
        all_outcomes = self.memory.recall_by_tags(tags=[OUTCOME_TAG], limit=1000)

        total = len(all_outcomes)
        if total == 0:
            return {"total": 0, "successes": 0, "failures": 0, "rate": 0.0}

        successes = sum(1 for m in all_outcomes if "SUCCESS" in m.content)
        failures = total - successes

        return {
            "total": total,
            "successes": successes,
            "failures": failures,
            "rate": successes / total if total > 0 else 0.0,
        }


def _suggest_fix(error_key: str) -> str:
    """Suggest a fix based on common error patterns."""
    error_lower = error_key.lower()

    if "ruff" in error_lower or "lint" in error_lower:
        return "Run 'ruff check --fix && ruff format' after edits"
    if "import" in error_lower:
        return "Check import paths and ensure module exists"
    if "test" in error_lower or "assert" in error_lower:
        return "Review test expectations — may need updating after code changes"
    if "permission" in error_lower:
        return "Check file permissions and path access"
    if "timeout" in error_lower:
        return "Increase timeout or check for infinite loops"
    if "connection" in error_lower:
        return "Verify service is running and reachable"

    return "Review error details and similar past failures"
