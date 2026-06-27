"""Animus Validator — Autonomous validator and node rewards management."""

__version__ = "0.1.0"

from animus_validator.config import ValidatorConfig
from animus_validator.models import (
    CompoundAction,
    Network,
    NetworkInfo,
    Node,
    RewardRecord,
    RewardType,
    StakePosition,
    StakeStatus,
)

__all__ = [
    "CompoundAction",
    "Network",
    "NetworkInfo",
    "Node",
    "RewardRecord",
    "RewardType",
    "StakePosition",
    "StakeStatus",
    "ValidatorConfig",
    "__version__",
]
