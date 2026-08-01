"""Signals emitted by Meta-Thinker to influence the Head loop."""

from dataclasses import dataclass, field
from enum import Enum


class SignalType(Enum):
    """Types of signals Meta-Thinker can emit."""

    REPLAN = "replan"
    INJECT_BRIEF = "inject_brief"
    ESCALATE = "escalate"
    HALT = "halt"


class ReplanStrategy(Enum):
    """How to replan when REPLAN signal is emitted."""

    FULL_RETHINK = "full_rethink"  # Start over with fresh context
    ADJUST_APPROACH = "adjust_approach"  # Keep context, add guidance
    SIMPLIFY_TASK = "simplify_task"  # Break into smaller subtasks
    ASK_CLARIFICATION = "ask_clarification"  # Request user input


@dataclass
class Signal:
    """Base class for all signals."""

    signal_type: SignalType
    confidence: float = 0.0  # 0.0–1.0
    reason: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class ReplanSignal(Signal):
    """Request the Head to replan its approach."""

    strategy: ReplanStrategy = ReplanStrategy.ADJUST_APPROACH
    suggested_approach: str = ""

    def __init__(
        self,
        confidence: float = 0.0,
        reason: str = "",
        strategy: ReplanStrategy = ReplanStrategy.ADJUST_APPROACH,
        suggested_approach: str = "",
        metadata: dict | None = None,
    ):
        super().__init__(
            SignalType.REPLAN,
            confidence=confidence,
            reason=reason,
            metadata=metadata or {},
        )
        self.strategy = strategy
        self.suggested_approach = suggested_approach


@dataclass
class InjectBriefSignal(Signal):
    """Inject a strategic brief into the conversation context."""

    brief_text: str = ""
    priority: str = "normal"  # normal, high, critical

    def __init__(
        self,
        confidence: float = 0.0,
        reason: str = "",
        brief_text: str = "",
        priority: str = "normal",
        metadata: dict | None = None,
    ):
        super().__init__(
            SignalType.INJECT_BRIEF,
            confidence=confidence,
            reason=reason,
            metadata=metadata or {},
        )
        self.brief_text = brief_text
        self.priority = priority


@dataclass
class EscalateSignal(Signal):
    """Escalate to a more capable model or reasoning mode."""

    target_mode: str = "deep"
    preserve_context: bool = True

    def __init__(
        self,
        confidence: float = 0.0,
        reason: str = "",
        target_mode: str = "deep",
        preserve_context: bool = True,
        metadata: dict | None = None,
    ):
        super().__init__(
            SignalType.ESCALATE,
            confidence=confidence,
            reason=reason,
            metadata=metadata or {},
        )
        self.target_mode = target_mode
        self.preserve_context = preserve_context


@dataclass
class HaltSignal(Signal):
    """Halt execution and return partial result."""

    partial_result: str = ""
    explanation: str = ""

    def __init__(
        self,
        confidence: float = 0.0,
        reason: str = "",
        partial_result: str = "",
        explanation: str = "",
        metadata: dict | None = None,
    ):
        super().__init__(
            SignalType.HALT,
            confidence=confidence,
            reason=reason,
            metadata=metadata or {},
        )
        self.partial_result = partial_result
        self.explanation = explanation
