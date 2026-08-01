"""Execution events emitted by the Head loop and consumed by Meta-Thinker."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Event:
    """Base class for all execution events."""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    session_id: str = ""
    iteration: int = 0


@dataclass
class IterationStarted(Event):
    """Fired when a new iteration of the agentic loop begins."""

    max_iterations: int = 5
    mode: str = "quick"


@dataclass
class ToolExecution(Event):
    """Fired after a tool is executed."""

    tool_name: str = ""
    params: dict = field(default_factory=dict)
    success: bool = True
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class ResponseReceived(Event):
    """Fired when the model produces a response."""

    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    tokens_used: int = 0


@dataclass
class LoopCompleted(Event):
    """Fired when the agentic loop terminates normally."""

    final_answer: str = ""
    total_iterations: int = 0
    reason: str = "completed"


@dataclass
class MaxIterationsReached(Event):
    """Fired when the loop exhausts max_iterations without completion."""

    final_response: str = ""
    total_iterations: int = 0
