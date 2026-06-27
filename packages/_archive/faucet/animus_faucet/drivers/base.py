"""Base driver interface for faucet interactions."""

from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING

from animus_faucet.models import ClaimResult, ClaimStatus, FaucetDefinition

if TYPE_CHECKING:
    from animus_faucet.config import FaucetConfig

logger = logging.getLogger("animus_faucet.drivers")


class BaseDriver(abc.ABC):
    """Abstract base for all faucet drivers.

    Each faucet gets a concrete driver subclass that knows how to:
    1. Navigate to the faucet
    2. Solve any captcha
    3. Submit the claim
    4. Parse the result
    """

    def __init__(self, faucet: FaucetDefinition, config: FaucetConfig):
        self.faucet = faucet
        self.config = config

    @abc.abstractmethod
    async def claim(self, wallet_address: str) -> ClaimResult:
        """Execute a single claim against this faucet.

        Args:
            wallet_address: The receiving wallet address for the faucet's chain.

        Returns:
            ClaimResult with status and any payout details.
        """

    async def health_check(self) -> bool:
        """Check if the faucet is reachable and functional.

        Default implementation returns True. Override for real checks.
        """
        return True

    def _success(
        self,
        amount: float | None = None,
        amount_usd: float | None = None,
        tx_hash: str | None = None,
        proxy_used: str | None = None,
    ) -> ClaimResult:
        """Helper to build a successful ClaimResult."""
        return ClaimResult(
            faucet_name=self.faucet.name,
            chain=self.faucet.chain,
            status=ClaimStatus.SUCCESS,
            amount=amount,
            amount_usd=amount_usd,
            tx_hash=tx_hash,
            proxy_used=proxy_used,
        )

    def _failure(
        self,
        status: ClaimStatus,
        error_detail: str | None = None,
        proxy_used: str | None = None,
    ) -> ClaimResult:
        """Helper to build a failed ClaimResult."""
        return ClaimResult(
            faucet_name=self.faucet.name,
            chain=self.faucet.chain,
            status=status,
            error_detail=error_detail,
            proxy_used=proxy_used,
        )
