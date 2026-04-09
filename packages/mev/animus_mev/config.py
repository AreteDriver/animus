"""Configuration for Animus MEV."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from animus_mev.models import Chain

DEFAULT_DATA_DIR = Path.home() / ".animus" / "mev"

# Default RPC endpoints (public, rate-limited)
DEFAULT_RPC_ENDPOINTS: dict[str, str] = {
    "base": "https://mainnet.base.org",
    "arb": "https://arb1.arbitrum.io/rpc",
    "op": "https://mainnet.optimism.io",
    "eth": "https://eth.llamarpc.com",
}

DEFAULT_WS_ENDPOINTS: dict[str, str] = {
    "base": "wss://base-mainnet.public.blastapi.io",
    "arb": "wss://arb-mainnet.g.alchemy.com/v2/demo",
    "op": "wss://optimism-mainnet.public.blastapi.io",
    "eth": "wss://eth-mainnet.public.blastapi.io",
}


@dataclass
class ChainConfig:
    """Per-chain configuration."""

    chain: Chain = Chain.BASE
    enabled: bool = True
    rpc_url: str = ""
    ws_url: str = ""
    flashbots_relay: str = ""
    min_swap_usd: float = 1000.0  # Minimum swap size to consider
    max_gas_gwei: float = 50.0

    def __post_init__(self):
        if not self.rpc_url:
            self.rpc_url = DEFAULT_RPC_ENDPOINTS.get(self.chain.value, "")
        if not self.ws_url:
            self.ws_url = DEFAULT_WS_ENDPOINTS.get(self.chain.value, "")

        # Override from env
        env_prefix = f"MEV_{self.chain.value.upper()}"
        self.rpc_url = os.environ.get(f"{env_prefix}_RPC_URL", self.rpc_url)
        self.ws_url = os.environ.get(f"{env_prefix}_WS_URL", self.ws_url)


@dataclass
class ExecutionConfig:
    """Execution settings. DRY_RUN by default."""

    live: bool = False  # MUST be explicitly enabled
    private_tx: bool = True  # Use Flashbots/private mempool
    max_slippage_bps: int = 50  # 0.5%
    tx_timeout_seconds: int = 30

    def __post_init__(self):
        env_live = os.environ.get("MEV_EXECUTION_LIVE", "")
        if env_live.lower() in ("true", "1", "yes"):
            self.live = True


@dataclass
class RiskConfig:
    """Risk management settings."""

    max_gas_per_tx_usd: float = 5.0
    daily_gas_budget_usd: float = 50.0
    min_profit_usd: float = 0.50
    min_confidence: float = 0.7
    max_concurrent_txs: int = 3
    revert_protection: bool = True
    sandwich_execution_blocked: bool = True  # ALWAYS True — ethical constraint


@dataclass
class MonitorConfig:
    """Monitoring and alerting."""

    poll_interval_seconds: float = 1.0
    max_pending_opportunities: int = 100
    log_all_txs: bool = False  # Log every mempool tx (noisy)
    discord_webhook_url: str = ""

    def __post_init__(self):
        self.discord_webhook_url = os.environ.get("MEV_DISCORD_WEBHOOK", self.discord_webhook_url)


@dataclass
class MEVConfig:
    """Root configuration for the MEV module."""

    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    chains: dict[str, ChainConfig] = field(default_factory=dict)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)

    def __post_init__(self):
        if env_dir := os.environ.get("MEV_DATA_DIR"):
            self.data_dir = Path(env_dir)
        # Default: enable all L2 chains
        if not self.chains:
            for chain in (Chain.BASE, Chain.ARB, Chain.OP):
                self.chains[chain.value] = ChainConfig(chain=chain)

    def get_chain(self, chain: Chain) -> ChainConfig | None:
        """Get config for a specific chain."""
        return self.chains.get(chain.value)

    def enabled_chains(self) -> list[ChainConfig]:
        """Get all enabled chain configs."""
        return [c for c in self.chains.values() if c.enabled]

    @classmethod
    def from_yaml(cls, path: Path) -> MEVConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> MEVConfig:
        chains: dict[str, ChainConfig] = {}
        for name, chain_data in data.get("chains", {}).items():
            chain_data["chain"] = Chain(name)
            chains[name] = ChainConfig(**chain_data)

        execution = (
            ExecutionConfig(**data["execution"]) if "execution" in data else ExecutionConfig()
        )
        risk = RiskConfig(**data["risk"]) if "risk" in data else RiskConfig()
        monitor = MonitorConfig(**data["monitor"]) if "monitor" in data else MonitorConfig()

        config = cls(
            chains=chains,
            execution=execution,
            risk=risk,
            monitor=monitor,
        )

        if "data_dir" in data:
            config.data_dir = Path(data["data_dir"])

        return config

    def to_yaml(self, path: Path) -> None:
        """Write config to YAML file."""
        data: dict[str, Any] = {
            "data_dir": str(self.data_dir),
            "chains": {},
            "execution": {
                "live": self.execution.live,
                "private_tx": self.execution.private_tx,
                "max_slippage_bps": self.execution.max_slippage_bps,
                "tx_timeout_seconds": self.execution.tx_timeout_seconds,
            },
            "risk": {
                "max_gas_per_tx_usd": self.risk.max_gas_per_tx_usd,
                "daily_gas_budget_usd": self.risk.daily_gas_budget_usd,
                "min_profit_usd": self.risk.min_profit_usd,
                "min_confidence": self.risk.min_confidence,
                "max_concurrent_txs": self.risk.max_concurrent_txs,
                "revert_protection": self.risk.revert_protection,
                "sandwich_execution_blocked": self.risk.sandwich_execution_blocked,
            },
            "monitor": {
                "poll_interval_seconds": self.monitor.poll_interval_seconds,
                "max_pending_opportunities": self.monitor.max_pending_opportunities,
                "log_all_txs": self.monitor.log_all_txs,
            },
        }
        for name, chain_cfg in self.chains.items():
            data["chains"][name] = {
                "enabled": chain_cfg.enabled,
                "rpc_url": chain_cfg.rpc_url,
                "ws_url": chain_cfg.ws_url,
                "min_swap_usd": chain_cfg.min_swap_usd,
                "max_gas_gwei": chain_cfg.max_gas_gwei,
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def config_file(self) -> Path:
        return self.data_dir / "config.yaml"
