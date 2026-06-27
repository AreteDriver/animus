"""Base hunter interface."""

from __future__ import annotations

import abc
import logging
import time
from typing import TYPE_CHECKING

from animus_prospector.models import HunterResult

if TYPE_CHECKING:
    from animus_prospector.config import ProspectorConfig

logger = logging.getLogger("animus_prospector.hunters")


class BaseHunter(abc.ABC):
    """Abstract base for all opportunity hunters.

    Each hunter scans a specific source category (web, aggregator, social)
    and returns discovered opportunities for evaluation.
    """

    name: str = "base"

    def __init__(self, config: ProspectorConfig):
        self.config = config

    async def scan(self) -> HunterResult:
        """Run a scan and return discovered opportunities.

        Wraps _scan() with timing and error handling.
        """
        start = time.monotonic()
        try:
            result = await self._scan()
            result.scan_duration_seconds = time.monotonic() - start
            return result
        except Exception as e:
            logger.exception(f"Hunter {self.name} scan failed: {e}")
            return HunterResult(
                hunter_name=self.name,
                errors=[str(e)],
                scan_duration_seconds=time.monotonic() - start,
            )

    @abc.abstractmethod
    async def _scan(self) -> HunterResult:
        """Implement the actual scanning logic."""

    async def health_check(self) -> bool:
        """Check if this hunter's sources are reachable."""
        return True
