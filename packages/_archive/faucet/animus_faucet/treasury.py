"""Treasury management — consolidation, balance checking, sweeping."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from animus_faucet.models import Chain, TreasuryBalance
from animus_faucet.wallet.manager import BalanceTracker, WalletManager

if TYPE_CHECKING:
    from animus_faucet.config import FaucetConfig

logger = logging.getLogger("animus_faucet.treasury")


class Treasury:
    """High-level treasury operations.

    Manages balance tracking and consolidation decisions.
    Actual chain interactions (balance queries, transfers) are delegated
    to chain-specific adapters.
    """

    def __init__(self, config: FaucetConfig, wallet_manager: WalletManager):
        self.config = config
        self.wallet_manager = wallet_manager
        self.balance_tracker = BalanceTracker(config.data_dir)
        self._chain_adapters: dict[Chain, ChainAdapter] = {}

    def register_adapter(self, chain: Chain, adapter: ChainAdapter) -> None:
        """Register a chain-specific adapter for balance/transfer operations."""
        self._chain_adapters[chain] = adapter

    async def refresh_balances(self) -> list[TreasuryBalance]:
        """Query current balances across all registered chains."""
        balances = []
        for chain_str, address in self.wallet_manager.list_wallets().items():
            chain = Chain(chain_str)
            adapter = self._chain_adapters.get(chain)
            if adapter is None:
                logger.debug(f"No adapter for {chain.value}, skipping balance check")
                continue

            try:
                balance = await adapter.get_balance(address)
                tb = TreasuryBalance(
                    chain=chain,
                    address=address,
                    balance=balance,
                    balance_usd=await adapter.to_usd(balance),
                )
                self.balance_tracker.update_balance(tb)
                balances.append(tb)
            except Exception as e:
                logger.warning(f"Failed to get balance for {chain.value}: {e}")

        return balances

    async def should_consolidate(self, chain: Chain) -> bool:
        """Check if a chain's balance exceeds the sweep threshold."""
        if not self.config.consolidation.enabled:
            return False

        balance = self.balance_tracker.get_balance(chain)
        if balance is None or balance.balance_usd is None:
            return False

        return balance.balance_usd >= self.config.consolidation.min_balance_usd

    async def consolidate(self, chain: Chain) -> str | None:
        """Sweep funds from hot wallet to cold wallet.

        Returns tx hash on success, None on failure or skip.
        """
        cold_address = self.config.consolidation.cold_wallet_addresses.get(chain.value)
        if not cold_address:
            logger.debug(f"No cold wallet for {chain.value}, skipping consolidation")
            return None

        adapter = self._chain_adapters.get(chain)
        if adapter is None:
            logger.debug(f"No adapter for {chain.value}, skipping consolidation")
            return None

        hot_address = self.wallet_manager.get_address(chain)
        if not hot_address:
            return None

        try:
            balance = await adapter.get_balance(hot_address)
            if balance <= 0:
                return None

            tx_hash = await adapter.transfer(hot_address, cold_address, balance)
            logger.info(f"Consolidated {balance} {chain.value} to cold wallet: {tx_hash}")
            return tx_hash
        except Exception as e:
            logger.error(f"Consolidation failed for {chain.value}: {e}")
            return None

    def get_portfolio(self) -> dict:
        """Get full portfolio summary."""
        balances = self.balance_tracker.get_all_balances()
        return {
            "balances": [b.to_dict() for b in balances],
            "total_usd": self.balance_tracker.total_usd(),
            "chains": len(balances),
            "last_updated": max(
                (b.last_updated for b in balances),
                default=datetime.now(UTC),
            ).isoformat(),
        }


class ChainAdapter:
    """Abstract interface for chain-specific operations.

    Concrete implementations (EVM, Sui, BTC) provide actual
    balance queries and transfer execution.
    """

    async def get_balance(self, address: str) -> float:
        """Get the native token balance for an address."""
        raise NotImplementedError

    async def to_usd(self, amount: float) -> float | None:
        """Convert native token amount to USD."""
        raise NotImplementedError

    async def transfer(self, from_address: str, to_address: str, amount: float) -> str:
        """Transfer tokens. Returns tx hash."""
        raise NotImplementedError

    async def get_gas_price(self) -> float | None:
        """Get current gas price (for consolidation decisions)."""
        return None
