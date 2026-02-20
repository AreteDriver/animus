"""Data models for Animus Forge workflows."""

from dataclasses import dataclass, field


class ForgeError(Exception):
    """Base exception for Forge operations."""


class BudgetExhaustedError(ForgeError):
    """Raised when an agent or workflow exceeds its token budget."""


class GateFailedError(ForgeError):
    """Raised when a quality gate fails with on_fail='halt'."""


class ReviseRequestedError(ForgeError):
    """Internal signal to trigger revision loop-back."""

    def __init__(self, target: str, gate_name: str, reason: str, max_revisions: int):
        self.target = target
        self.gate_name = gate_name
        self.reason = reason
        self.max_revisions = max_revisions
        super().__init__(f"Revise: {gate_name} -> {target}")


@dataclass
class AgentConfig:
    """Configuration for a single agent in a workflow."""

    name: str
    archetype: str
    budget_tokens: int = 10_000
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    system_prompt: str | None = None
    tools: list[str] = field(default_factory=list)


@dataclass
class GateConfig:
    """Configuration for a quality gate between pipeline stages."""

    name: str
    after: str
    type: str = "automated"
    pass_condition: str = ""
    on_fail: str = "halt"
    revise_target: str | None = None
    max_revisions: int = 3


@dataclass
class WorkflowConfig:
    """Top-level workflow configuration loaded from YAML."""

    name: str
    description: str = ""
    agents: list[AgentConfig] = field(default_factory=list)
    gates: list[GateConfig] = field(default_factory=list)
    max_cost_usd: float = 1.0
    provider: str = "ollama"
    model: str = "llama3:8b"
    execution_mode: str = "sequential"


@dataclass
class StepResult:
    """Result of executing a single agent step."""

    agent_name: str
    outputs: dict[str, str] = field(default_factory=dict)
    tokens_used: int = 0
    cost_usd: float = 0.0
    success: bool = True
    error: str | None = None


@dataclass
class WorkflowState:
    """Mutable state tracking workflow execution progress."""

    workflow_name: str
    status: str = "pending"
    current_step: int = 0
    results: list[StepResult] = field(default_factory=list)
    total_tokens: int = 0
    total_cost: float = 0.0
    revision_counts: dict[str, int] = field(default_factory=dict)
