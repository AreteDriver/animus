"""Abstract base class for DEX price feeds."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from animus_arbitrage.models import DEX, Chain, PriceQuote, TokenPair

logger = logging.getLogger("animus_arbitrage.feeds")


class BaseFeed(ABC):
    """Abstract price feed for a specific DEX."""

    def __init__(self, dex: DEX, chain: Chain, rpc_url: str = ""):
        self.dex = dex
        self.chain = chain
        self.rpc_url = rpc_url
        self._healthy = True

    @abstractmethod
    async def get_quote(self, pair: TokenPair) -> PriceQuote | None:
        """Fetch a price quote for the given token pair.

        Returns None if the pair is not available on this DEX.
        """

    @abstractmethod
    async def get_supported_pairs(self) -> list[TokenPair]:
        """Return list of token pairs this feed can quote."""

    async def health_check(self) -> bool:
        """Check if the feed is operational."""
        return self._healthy

    @property
    def name(self) -> str:
        return f"{self.dex.value}@{self.chain.value}"
