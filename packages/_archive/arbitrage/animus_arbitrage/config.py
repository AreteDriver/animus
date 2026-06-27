"""Configuration for Animus Arbitrage."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from animus_arbitrage.models import DEX, Chain, ExecutionMode

DEFAULT_DATA_DIR = Path.home() / ".animus" / "arbitrage"


@dataclass
class FeedConfig:
    """Per-feed settings."""

    dex: DEX = DEX.UNISWAP_V3
    chain: Chain = Chain.ETH
    enabled: bool = True
    poll_interval_seconds: float = 30.0
    rpc_url: str = ""


@dataclass
class ExecutionConfig:
    """Execution settings. DRY_RUN by default — safety gate."""

    mode: ExecutionMode = ExecutionMode.DRY_RUN
    max_trade_size_usd: float = 100.0
    slippage_tolerance_pct: float = 0.5
    confirmation_timeout_seconds: float = 120.0

    def __post_init__(self):
        env_mode = os.environ.get("ARBITRAGE_EXECUTION_MODE", "")
        if env_mode.lower() == "live":
            self.mode = ExecutionMode.LIVE
        elif env_mode.lower() == "dry_run":
            self.mode = ExecutionMode.DRY_RUN

    @property
    def is_live(self) -> bool:
        return self.mode == ExecutionMode.LIVE


@dataclass
class RiskConfig:
    """Risk management parameters."""

    max_position_usd: float = 500.0
    daily_loss_limit_usd: float = 50.0
    min_spread_pct: float = 0.3
    min_liquidity_usd: float = 10000.0
    min_confidence: float = 0.7
    cooldown_seconds: float = 60.0
    max_trades_per_hour: int = 10
    max_concurrent_trades: int = 2


@dataclass
class AlertConfig:
    """Alert settings for high-spread opportunities."""

    enabled: bool = True
    high_spread_threshold_pct: float = 1.0
    discord_webhook_url: str = ""

    def __post_init__(self):
        self.discord_webhook_url = os.environ.get(
            "ARBITRAGE_DISCORD_WEBHOOK", self.discord_webhook_url
        )


@dataclass
class WalletConfig:
    """Wallet configuration per chain."""

    chain: Chain = Chain.ETH
    address: str = ""
    # Private key loaded from env only, never from config files
    private_key_env_var: str = ""


@dataclass
class ArbitrageConfig:
    """Root configuration for the arbitrage module."""

    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    feeds: list[FeedConfig] = field(default_factory=list)
    wallets: list[WalletConfig] = field(default_factory=list)

    def __post_init__(self):
        if env_dir := os.environ.get("ARBITRAGE_DATA_DIR"):
            self.data_dir = Path(env_dir)

    @classmethod
    def from_yaml(cls, path: Path) -> ArbitrageConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> ArbitrageConfig:
        execution = (
            ExecutionConfig(**data["execution"]) if "execution" in data else ExecutionConfig()
        )
        risk = RiskConfig(**data["risk"]) if "risk" in data else RiskConfig()
        alerts = AlertConfig(**data["alerts"]) if "alerts" in data else AlertConfig()

        feeds = []
        for fd in data.get("feeds", []):
            fc = FeedConfig(**fd)
            feeds.append(fc)

        wallets = []
        for wd in data.get("wallets", []):
            wallets.append(WalletConfig(**wd))

        config = cls(
            execution=execution,
            risk=risk,
            alerts=alerts,
            feeds=feeds,
            wallets=wallets,
        )

        if "data_dir" in data:
            config.data_dir = Path(data["data_dir"])

        return config

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dict (excludes sensitive data)."""
        return {
            "data_dir": str(self.data_dir),
            "execution": {
                "mode": self.execution.mode.value,
                "max_trade_size_usd": self.execution.max_trade_size_usd,
                "slippage_tolerance_pct": self.execution.slippage_tolerance_pct,
            },
            "risk": {
                "max_position_usd": self.risk.max_position_usd,
                "daily_loss_limit_usd": self.risk.daily_loss_limit_usd,
                "min_spread_pct": self.risk.min_spread_pct,
                "min_liquidity_usd": self.risk.min_liquidity_usd,
                "min_confidence": self.risk.min_confidence,
                "cooldown_seconds": self.risk.cooldown_seconds,
                "max_trades_per_hour": self.risk.max_trades_per_hour,
            },
            "alerts": {
                "enabled": self.alerts.enabled,
                "high_spread_threshold_pct": self.alerts.high_spread_threshold_pct,
            },
            "feeds_count": len(self.feeds),
            "wallets_count": len(self.wallets),
        }

    def ensure_dirs(self) -> None:
        """Create necessary data directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def config_file(self) -> Path:
        return self.data_dir / "config.yaml"
