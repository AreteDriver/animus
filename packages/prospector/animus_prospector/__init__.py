"""Animus Prospector — Autonomous crypto opportunity hunter."""

__version__ = "0.1.0"

from animus_prospector.config import ProspectorConfig
from animus_prospector.evaluator import Evaluator
from animus_prospector.feed import OpportunityFeed
from animus_prospector.models import (
    Action,
    Chain,
    EvaluationResult,
    HunterResult,
    Opportunity,
    OpportunityStatus,
    OpportunityType,
    SourceType,
)
from animus_prospector.orchestrator import Prospector

__all__ = [
    "Action",
    "Chain",
    "EvaluationResult",
    "Evaluator",
    "HunterResult",
    "Opportunity",
    "OpportunityFeed",
    "OpportunityStatus",
    "OpportunityType",
    "Prospector",
    "ProspectorConfig",
    "SourceType",
    "__version__",
]
