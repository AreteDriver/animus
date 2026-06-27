"""Configuration for Animus Validator."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DATA_DIR = Path.home() / ".animus" / "validator"


@dataclass
class CompoundConfig:
    """Auto-compound settings."""

    enabled: bool = True
    min_reward_threshold: float = 0.01  # Min rewards before compounding
    gas_buffer_multiplier: float = 2.0  # Compound only if reward > gas * multiplier
    check_interval_hours: int = 6

    def __post_init__(self):
        if env_val := os.environ.get("VALIDATOR_COMPOUND_THRESHOLD"):
            self.min_reward_threshold = float(env_val)


@dataclass
class MonitorConfig:
    """Node monitoring settings."""

    check_interval_minutes: int = 15
    uptime_alert_threshold: float = 95.0  # Alert if below this %
    slashing_alert: bool = True
    downtime_alert_minutes: int = 30


@dataclass
class AlertConfig:
    """Notification settings."""

    discord_webhook_url: str = ""
    alert_on_slashing: bool = True
    alert_on_downtime: bool = True
    alert_on_compound: bool = True
    daily_summary: bool = True

    def __post_init__(self):
        self.discord_webhook_url = os.environ.get(
            "VALIDATOR_DISCORD_WEBHOOK", self.discord_webhook_url
        )


@dataclass
class DiscoveryConfig:
    """Settings for discovering new staking opportunities."""

    enabled: bool = True
    scan_interval_hours: int = 24
    min_apy_threshold: float = 3.0  # Minimum APY to surface
    max_risk_score: float = 0.5  # Max risk (0=safe, 1=risky)


@dataclass
class ValidatorConfig:
    """Root configuration for the validator module."""

    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    compound: CompoundConfig = field(default_factory=CompoundConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)

    def __post_init__(self):
        if env_dir := os.environ.get("VALIDATOR_DATA_DIR"):
            self.data_dir = Path(env_dir)

    @classmethod
    def from_yaml(cls, path: Path) -> ValidatorConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> ValidatorConfig:
        compound = CompoundConfig(**data["compound"]) if "compound" in data else CompoundConfig()
        monitor = MonitorConfig(**data["monitor"]) if "monitor" in data else MonitorConfig()
        alerts = AlertConfig(**data["alerts"]) if "alerts" in data else AlertConfig()
        discovery = (
            DiscoveryConfig(**data["discovery"]) if "discovery" in data else DiscoveryConfig()
        )

        config = cls(
            compound=compound,
            monitor=monitor,
            alerts=alerts,
            discovery=discovery,
        )

        if "data_dir" in data:
            config.data_dir = Path(data["data_dir"])

        return config

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "positions").mkdir(exist_ok=True)
        (self.data_dir / "rewards").mkdir(exist_ok=True)
        (self.data_dir / "compounds").mkdir(exist_ok=True)

    @property
    def positions_file(self) -> Path:
        return self.data_dir / "positions" / "stakes.jsonl"

    @property
    def rewards_file(self) -> Path:
        return self.data_dir / "rewards" / "history.jsonl"

    @property
    def compounds_file(self) -> Path:
        return self.data_dir / "compounds" / "actions.jsonl"

    @property
    def nodes_file(self) -> Path:
        return self.data_dir / "nodes.jsonl"

    @property
    def config_file(self) -> Path:
        return self.data_dir / "config.yaml"
