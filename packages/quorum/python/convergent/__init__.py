"""Convergent — Multi-agent coherence and coordination for AI systems."""

__version__ = "1.0.0"

from convergent.async_backend import AsyncBackendWrapper, AsyncGraphBackend
from convergent.benchmark import (
    BenchmarkMetrics,
    BenchmarkSuite,
    ScenarioType,
    run_benchmark,
    run_scaling_suite,
)
from convergent.constraints import (
    ConstraintCheckResult,
    ConstraintEngine,
    ConstraintKind,
    GateResult,
    TypedConstraint,
)
from convergent.contract import (
    DEFAULT_CONTRACT,
    DEFAULT_RESOLUTION_POLICY,
    DEFAULT_STABILITY_WEIGHTS,
    ConflictClass,
    ContractViolation,
    EdgeType,
    GraphInvariant,
    IntentGraphContract,
    MutationType,
    ResolutionPolicy,
    StabilityWeights,
    content_hash_intent,
    content_hash_intents,
    validate_publish,
)

# Phase 3: Coordination Protocol
from convergent.coordination_config import CoordinationConfig

# Phase 4: Observability & Analysis
from convergent.cycles import DependencyCycle, DependencyGraph, find_cycles, topological_order
from convergent.economics import (
    Budget,
    CoordinationCostReport,
    CostModel,
    EscalationAction,
    EscalationDecision,
    EscalationPolicy,
)
from convergent.event_log import CoordinationEvent, EventLog, EventType, event_timeline
from convergent.flocking import FlockingCoordinator
from convergent.gates import (
    CommandGate,
    CompileGate,
    ConstraintGate,
    GateReport,
    GateRunner,
    GateRunResult,
    MypyGate,
    PytestGate,
)
from convergent.gorgon_bridge import GorgonBridge
from convergent.governor import (
    AgentBranch,
    GovernorVerdict,
    MergeGovernor,
    ProposalResult,
    VerdictKind,
)
from convergent.health import (
    CoordinationHealth,
    HealthChecker,
    IntentGraphHealth,
    ScoringHealth,
    StigmergyHealth,
    VotingHealth,
    health_report,
)
from convergent.intent import (
    Adjustment,
    ConflictReport,
    Constraint,
    ConstraintSeverity,
    Evidence,
    EvidenceKind,
    Intent,
    InterfaceKind,
    InterfaceSpec,
    ResolutionResult,
)
from convergent.matching import (
    names_overlap,
    normalize_constraint_target,
    normalize_name,
    normalize_type,
    parse_signature,
    signatures_compatible,
)
from convergent.protocol import (
    AgentIdentity,
    ConsensusRequest,
    Decision,
    DecisionOutcome,
    QuorumLevel,
    Signal,
    StigmergyMarker,
    Vote,
    VoteChoice,
)
from convergent.replay import ReplayLog, ReplayResult
from convergent.resolver import GraphBackend, IntentResolver, PythonGraphBackend
from convergent.rust_backend import HAS_RUST, RustGraphBackend
from convergent.score_store import ScoreStore
from convergent.scoring import PhiScorer
from convergent.semantic import (
    ConstraintApplicability,
    SemanticMatch,
    SemanticMatcher,
    TrajectoryPrediction,
)
from convergent.signal_backend import FilesystemSignalBackend, SignalBackend
from convergent.signal_bus import SignalBus
from convergent.sqlite_backend import SQLiteBackend
from convergent.sqlite_signal_backend import SQLiteSignalBackend
from convergent.stigmergy import StigmergyField
from convergent.triumvirate import Triumvirate
from convergent.versioning import GraphSnapshot, MergeResult, VersionedGraph
from convergent.visualization import dot_graph, html_report, overlap_matrix, text_table

__all__ = [
    # Layer 1: Constraint Engine
    "ConstraintCheckResult",
    "ConstraintEngine",
    "ConstraintKind",
    "GateResult",
    "TypedConstraint",
    # Layer 3: Economics
    "Budget",
    "CoordinationCostReport",
    "CostModel",
    "EscalationAction",
    "EscalationDecision",
    "EscalationPolicy",
    # Governor (integrates all 3 layers)
    "AgentBranch",
    "GovernorVerdict",
    "MergeGovernor",
    "ProposalResult",
    "VerdictKind",
    # Contract
    "ConflictClass",
    "ContractViolation",
    "DEFAULT_CONTRACT",
    "DEFAULT_RESOLUTION_POLICY",
    "DEFAULT_STABILITY_WEIGHTS",
    "EdgeType",
    "GraphInvariant",
    "IntentGraphContract",
    "MutationType",
    "ResolutionPolicy",
    "StabilityWeights",
    "content_hash_intent",
    "content_hash_intents",
    "validate_publish",
    # Core types
    "Adjustment",
    "ConflictReport",
    "Constraint",
    "ConstraintApplicability",
    "ConstraintSeverity",
    "Evidence",
    "EvidenceKind",
    "Intent",
    "IntentResolver",
    "InterfaceKind",
    "InterfaceSpec",
    "ResolutionResult",
    # Backends
    "AsyncBackendWrapper",
    "AsyncGraphBackend",
    "GraphBackend",
    "HAS_RUST",
    "PythonGraphBackend",
    "RustGraphBackend",
    "SQLiteBackend",
    # Replay
    "ReplayLog",
    "ReplayResult",
    # Semantic
    "SemanticMatch",
    "SemanticMatcher",
    "TrajectoryPrediction",
    # Versioning
    "GraphSnapshot",
    "MergeResult",
    "VersionedGraph",
    # Matching utilities
    "names_overlap",
    "normalize_constraint_target",
    "normalize_name",
    "normalize_type",
    "parse_signature",
    "signatures_compatible",
    # Benchmark
    "BenchmarkMetrics",
    "BenchmarkSuite",
    "ScenarioType",
    "run_benchmark",
    "run_scaling_suite",
    # Gates (subprocess-backed evidence)
    "CommandGate",
    "CompileGate",
    "ConstraintGate",
    "GateReport",
    "GateRunResult",
    "GateRunner",
    "MypyGate",
    "PytestGate",
    # Visualization
    "dot_graph",
    "html_report",
    "overlap_matrix",
    "text_table",
    # Factories
    "create_delegation_checker",
    # Phase 3: Coordination Protocol
    "AgentIdentity",
    "ConsensusRequest",
    "CoordinationConfig",
    "Decision",
    "DecisionOutcome",
    "QuorumLevel",
    "Signal",
    "StigmergyMarker",
    "Vote",
    "VoteChoice",
    # Phase 3: Phi-Weighted Scoring
    "PhiScorer",
    "ScoreStore",
    # Phase 3: Triumvirate Voting
    "Triumvirate",
    # Phase 3: Signal Bus
    "FilesystemSignalBackend",
    "SignalBackend",
    "SignalBus",
    "SQLiteSignalBackend",
    # Phase 3: Stigmergy
    "StigmergyField",
    # Phase 3: Flocking
    "FlockingCoordinator",
    # Phase 3: Gorgon Integration
    "GorgonBridge",
    # Phase 4: Health Dashboard
    "CoordinationHealth",
    "HealthChecker",
    "IntentGraphHealth",
    "ScoringHealth",
    "StigmergyHealth",
    "VotingHealth",
    "health_report",
    # Phase 4: Cycle Detection
    "DependencyCycle",
    "DependencyGraph",
    "find_cycles",
    "topological_order",
    # Phase 4: Event Log
    "CoordinationEvent",
    "EventLog",
    "EventType",
    "event_timeline",
]

# Conditional export: AnthropicSemanticMatcher (only when anthropic installed)
try:
    from convergent.semantic import AnthropicSemanticMatcher  # noqa: F401

    __all__.append("AnthropicSemanticMatcher")
except ImportError:
    pass  # anthropic not installed; AnthropicSemanticMatcher unavailable


def create_delegation_checker(
    min_stability: float = 0.0,
    backend: GraphBackend | None = None,
) -> IntentResolver:
    """Create an IntentResolver configured for delegation coherence checking.

    Convenience factory for Gorgon integration. Uses a PythonGraphBackend
    by default (in-memory, no persistence needed for delegation checks).

    Args:
        min_stability: Minimum stability threshold for overlap detection.
        backend: Optional custom backend (e.g., SQLiteBackend for persistence).

    Returns:
        Configured IntentResolver ready for delegation checking.
    """
    return IntentResolver(
        backend=backend or PythonGraphBackend(),
        min_stability=min_stability,
    )
