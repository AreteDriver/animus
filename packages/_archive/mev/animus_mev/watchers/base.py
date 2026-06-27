"""Base watcher interface."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from animus_mev.config import ChainConfig
from animus_mev.models import MempoolTx

logger = logging.getLogger("animus_mev.watchers")


class BaseWatcher(ABC):
    """Abstract base for transaction watchers."""

    def __init__(self, chain_config: ChainConfig):
        self.chain_config = chain_config
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean up connection."""

    @abstractmethod
    async def watch(self) -> AsyncIterator[MempoolTx]:
        """Yield transactions as they appear."""
        yield  # pragma: no cover

    async def start(self) -> None:
        """Start watching."""
        self._running = True
        await self.connect()

    async def stop(self) -> None:
        """Stop watching."""
        self._running = False
        await self.disconnect()
