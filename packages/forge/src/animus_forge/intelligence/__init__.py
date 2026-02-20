"""Intelligence layer — application-layer dominance for Gorgon.

Provides feedback loops, outcome tracking, cross-workflow learning,
and intelligent provider routing that make Gorgon's orchestration
irreplaceable.

Modules:
    outcome_tracker: Records whether agent outputs actually worked
    cross_workflow_memory: Persistent learning across workflow executions
    provider_router: Intelligent provider+model selection
    feedback_engine: Closes the loop between outcomes and future behavior
"""

from animus_forge.intelligence.cost_intelligence import (
    CostIntelligence,
    SavingsRecommendation,
    SpendingAnalysis,
)
from animus_forge.intelligence.cross_workflow_memory import (
    AgentProfile,
    CrossWorkflowMemory,
    Pattern,
)
from animus_forge.intelligence.feedback_engine import (
    AgentTrajectory,
    FeedbackEngine,
    FeedbackResult,
    Suggestion,
    WorkflowFeedback,
)
from animus_forge.intelligence.integration_graph import (
    DispatchResult,
    IntegrationChain,
    IntegrationGraph,
    TriggerRule,
)
from animus_forge.intelligence.outcome_tracker import (
    OutcomeRecord,
    OutcomeTracker,
    ProviderStats,
)
from animus_forge.intelligence.prompt_evolution import (
    PromptEvolution,
    PromptVariant,
    VariantStats,
)
from animus_forge.intelligence.provider_router import (
    ProviderCapability,
    ProviderRouter,
    ProviderSelection,
    RoutingStrategy,
)

__all__ = [
    # Outcome tracking
    "OutcomeRecord",
    "OutcomeTracker",
    "ProviderStats",
    # Cross-workflow memory
    "AgentProfile",
    "CrossWorkflowMemory",
    "Pattern",
    # Provider routing
    "ProviderCapability",
    "ProviderRouter",
    "ProviderSelection",
    "RoutingStrategy",
    # Feedback engine
    "AgentTrajectory",
    "FeedbackEngine",
    "FeedbackResult",
    "Suggestion",
    "WorkflowFeedback",
    # Prompt evolution
    "PromptEvolution",
    "PromptVariant",
    "VariantStats",
    # Integration graph
    "IntegrationGraph",
    "TriggerRule",
    "DispatchResult",
    "IntegrationChain",
    # Cost intelligence
    "CostIntelligence",
    "SpendingAnalysis",
    "SavingsRecommendation",
]
