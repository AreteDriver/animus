"""History-Aware Provider Router for Animus.

Implements P-20260706-003: Replaces static semantic matching with
trajectory-based provider selection. Inspired by ACE-Router (ACL 2026).

Key design:
- RoutingGraph: tracks outcomes per (provider, task_signature) edge
- TrajectoryScorer: scores providers based on historical success + capability match
- ProviderRouter: selects best provider for each request, learns from outcomes
"""

from animus.routing.graph import (
    ProviderNode,
    RoutingEdge,
    RoutingGraph,
    TaskSignature,
)
from animus.routing.router import (
    ProviderRouter,
    RouterConfig,
    RoutingDecision,
)
from animus.routing.scorer import (
    ScoreWeights,
    TrajectoryScorer,
)

__all__ = [
    "ProviderRouter",
    "RouterConfig",
    "RoutingDecision",
    "RoutingGraph",
    "ProviderNode",
    "TaskSignature",
    "RoutingEdge",
    "TrajectoryScorer",
    "ScoreWeights",
]
