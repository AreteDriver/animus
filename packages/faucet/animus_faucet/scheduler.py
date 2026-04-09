"""Faucet claim scheduler with jitter and health-aware execution."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING

from animus_faucet.drivers.api_driver import APIDriver
from animus_faucet.drivers.browser_driver import BrowserDriver
from animus_faucet.drivers.cli_driver import CLIDriver
from animus_faucet.models import (
    ClaimResult,
    ClaimStatus,
    DriverType,
    FaucetDefinition,
    FaucetHealth,
)
from animus_faucet.store import ClaimStore, HealthStore

if TYPE_CHECKING:
    from animus_faucet.config import FaucetConfig
    from animus_faucet.drivers.base import BaseDriver
    from animus_faucet.wallet.manager import WalletManager

logger = logging.getLogger("animus_faucet.scheduler")

DRIVER_MAP: dict[DriverType, type] = {
    DriverType.API: APIDriver,
    DriverType.BROWSER: BrowserDriver,
    DriverType.CLI: CLIDriver,
}


class FaucetScheduler:
    """Orchestrates faucet claims on schedule with jitter.

    Responsibilities:
    - Creates the right driver for each faucet
    - Applies random jitter to schedules (human-like)
    - Tracks health per faucet and auto-disables failing ones
    - Records all claim results for reporting
    """

    def __init__(
        self,
        config: FaucetConfig,
        wallet_manager: WalletManager,
    ):
        self.config = config
        self.wallet_manager = wallet_manager
        self.claim_store = ClaimStore(config.data_dir)
        self.health_store = HealthStore(config.data_dir)
        self._drivers: dict[str, BaseDriver] = {}
        self._running = False

    def _get_driver(self, faucet: FaucetDefinition) -> BaseDriver:
        """Get or create a driver for a faucet."""
        if faucet.name not in self._drivers:
            driver_cls = DRIVER_MAP.get(faucet.driver_type)
            if driver_cls is None:
                raise ValueError(f"Unknown driver type: {faucet.driver_type}")
            self._drivers[faucet.name] = driver_cls(faucet, self.config)
        return self._drivers[faucet.name]

    async def claim_faucet(self, faucet: FaucetDefinition) -> ClaimResult:
        """Execute a single faucet claim."""
        health = self.health_store.get(faucet.name)
        if health and health.should_disable:
            logger.warning(f"Skipping {faucet.name} — auto-disabled (banned or too many failures)")
            return ClaimResult(
                faucet_name=faucet.name,
                chain=faucet.chain,
                status=ClaimStatus.BANNED if health.banned else ClaimStatus.RATE_LIMITED,
                error_detail="Auto-disabled due to consecutive failures",
            )

        address = self.wallet_manager.get_address(faucet.chain)
        if not address:
            return ClaimResult(
                faucet_name=faucet.name,
                chain=faucet.chain,
                status=ClaimStatus.UNKNOWN_ERROR,
                error_detail=f"No wallet address configured for {faucet.chain.value}",
            )

        # Apply jitter
        if self.config.jitter_seconds > 0:
            jitter = random.uniform(0, self.config.jitter_seconds)
            logger.debug(f"Jitter: waiting {jitter:.0f}s before claiming {faucet.name}")
            await asyncio.sleep(jitter)

        driver = self._get_driver(faucet)
        result = await driver.claim(address)

        # Update stores
        self.claim_store.record(result)
        if health is None:
            health = FaucetHealth(name=faucet.name, enabled=faucet.enabled)
        health.record_claim(result)
        self.health_store.save(health)

        if result.succeeded:
            logger.info(
                f"Claimed {faucet.name}: {result.amount or '?'} ({result.amount_usd or '?'} USD)"
            )
        else:
            logger.warning(f"Failed {faucet.name}: {result.status.value} — {result.error_detail}")

        return result

    async def claim_all(self) -> list[ClaimResult]:
        """Claim from all enabled faucets."""
        results = []
        for faucet in self.config.faucets:
            if not faucet.enabled:
                continue
            result = await self.claim_faucet(faucet)
            results.append(result)
        return results

    async def run_once(self) -> list[ClaimResult]:
        """Run a single pass through all enabled faucets."""
        logger.info(f"Starting claim pass — {len(self.config.faucets)} faucets configured")
        results = await self.claim_all()
        succeeded = sum(1 for r in results if r.succeeded)
        logger.info(f"Claim pass complete: {succeeded}/{len(results)} succeeded")
        return results

    def get_summary(self) -> dict:
        """Get a summary of all faucet health and earnings."""
        faucets = []
        total_earned = 0.0
        for faucet in self.config.faucets:
            health = self.health_store.get(faucet.name)
            if health:
                faucets.append(health.to_dict())
                total_earned += health.total_earned_usd
            else:
                faucets.append({"name": faucet.name, "enabled": faucet.enabled, "no_data": True})

        return {
            "total_faucets": len(self.config.faucets),
            "enabled": sum(1 for f in self.config.faucets if f.enabled),
            "total_earned_usd": total_earned,
            "total_claims": self.claim_store.total_claims(),
            "faucets": faucets,
        }
