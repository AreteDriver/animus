"""Base strategy interface."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from animus_mev.config import MEVConfig
from animus_mev.models import MempoolTx, MEVOpportunity

logger = logging.getLogger("animus_mev.strategies")


class BaseStrategy(ABC):
    """Abstract base for MEV detection strategies."""

    def __init__(self, config: MEVConfig):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @abstractmethod
    async def evaluate(self, tx: MempoolTx) -> MEVOpportunity | None:
        """Evaluate a transaction for MEV opportunity.

        Returns an MEVOpportunity if one is detected, None otherwise.
        """

    @abstractmethod
    def is_applicable(self, tx: MempoolTx) -> bool:
        """Quick check if this strategy applies to the given tx."""
