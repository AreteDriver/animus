"""Base platform scanner interface."""

from __future__ import annotations

import abc
import logging
import time
from typing import TYPE_CHECKING

from animus_bounty.models import Bounty

if TYPE_CHECKING:
    from animus_bounty.config import BountyConfig

logger = logging.getLogger("animus_bounty.platforms")


class ScanResult:
    """Output from a platform scan."""

    def __init__(
        self,
        platform_name: str,
        bounties: list[Bounty] | None = None,
        errors: list[str] | None = None,
        scan_duration_seconds: float = 0.0,
    ):
        self.platform_name = platform_name
        self.bounties: list[Bounty] = bounties or []
        self.errors: list[str] = errors or []
        self.scan_duration_seconds = scan_duration_seconds

    @property
    def count(self) -> int:
        return len(self.bounties)

    def to_dict(self) -> dict:
        return {
            "platform_name": self.platform_name,
            "count": self.count,
            "errors": self.errors,
            "scan_duration_seconds": self.scan_duration_seconds,
        }


class BasePlatform(abc.ABC):
    """Abstract base for all bounty platform scanners.

    Each scanner discovers bounties from a specific platform
    and returns them for skill matching and tracking.
    """

    name: str = "base"

    def __init__(self, config: BountyConfig):
        self.config = config

    async def scan(self) -> ScanResult:
        """Run a scan and return discovered bounties.

        Wraps _scan() with timing and error handling.
        """
        start = time.monotonic()
        try:
            result = await self._scan()
            result.scan_duration_seconds = time.monotonic() - start
            return result
        except Exception as e:
            logger.exception(f"Platform {self.name} scan failed: {e}")
            return ScanResult(
                platform_name=self.name,
                errors=[str(e)],
                scan_duration_seconds=time.monotonic() - start,
            )

    @abc.abstractmethod
    async def _scan(self) -> ScanResult:
        """Implement the actual scanning logic."""

    async def health_check(self) -> bool:
        """Check if this platform's API is reachable."""
        return True
