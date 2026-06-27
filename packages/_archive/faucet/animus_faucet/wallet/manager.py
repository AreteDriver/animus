"""Multi-chain wallet manager for receiving and consolidating faucet payouts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from animus_faucet.models import Chain, TreasuryBalance

logger = logging.getLogger("animus_faucet.wallet")


class WalletManager:
    """Manages wallet addresses across multiple chains.

    Stores addresses in a JSON file. Does NOT store private keys —
    those must be managed externally (hardware wallet, env vars, etc).
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._wallets_file = data_dir / "wallets.json"
        self._wallets: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self._wallets_file.exists():
            with open(self._wallets_file) as f:
                self._wallets = json.load(f)

    def _save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self._wallets_file, "w") as f:
            json.dump(self._wallets, f, indent=2)

    def set_address(self, chain: Chain, address: str) -> None:
        """Register a wallet address for a chain."""
        self._wallets[chain.value] = address
        self._save()

    def get_address(self, chain: Chain) -> str | None:
        """Get the registered wallet address for a chain."""
        return self._wallets.get(chain.value)

    def list_wallets(self) -> dict[str, str]:
        """Return all registered chain -> address mappings."""
        return dict(self._wallets)

    def remove_address(self, chain: Chain) -> bool:
        """Remove a wallet address for a chain."""
        if chain.value in self._wallets:
            del self._wallets[chain.value]
            self._save()
            return True
        return False

    def has_address(self, chain: Chain) -> bool:
        """Check if a wallet address is registered for a chain."""
        return chain.value in self._wallets


class BalanceTracker:
    """Tracks wallet balances over time."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._balances_file = data_dir / "balances.json"
        self._balances: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self._balances_file.exists():
            with open(self._balances_file) as f:
                self._balances = json.load(f)

    def _save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self._balances_file, "w") as f:
            json.dump(self._balances, f, indent=2)

    def update_balance(self, balance: TreasuryBalance) -> None:
        """Record a balance snapshot."""
        self._balances[balance.chain.value] = balance.to_dict()
        self._save()

    def get_balance(self, chain: Chain) -> TreasuryBalance | None:
        """Get the last known balance for a chain."""
        data = self._balances.get(chain.value)
        if data:
            return TreasuryBalance.from_dict(data)
        return None

    def get_all_balances(self) -> list[TreasuryBalance]:
        """Get all last known balances."""
        return [TreasuryBalance.from_dict(d) for d in self._balances.values()]

    def total_usd(self) -> float:
        """Sum of all balances in USD."""
        return sum(b.balance_usd for b in self.get_all_balances() if b.balance_usd is not None)
