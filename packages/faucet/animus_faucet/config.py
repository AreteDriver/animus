"""Configuration for Animus Faucet."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from animus_faucet.models import FaucetDefinition

logger = logging.getLogger("animus_faucet.config")

DEFAULT_DATA_DIR = Path.home() / ".animus" / "faucet"


@dataclass
class ProxyConfig:
    """Proxy rotation configuration."""

    enabled: bool = False
    provider: str = "brightdata"  # brightdata, oxylabs, custom
    endpoint: str = ""
    username: str = ""
    password: str = ""
    rotation_interval_seconds: int = 300

    def __post_init__(self):
        self.enabled = os.environ.get("FAUCET_PROXY_ENABLED", str(self.enabled)).lower() in (
            "true",
            "1",
            "yes",
        )
        self.endpoint = os.environ.get("FAUCET_PROXY_ENDPOINT", self.endpoint)
        self.username = os.environ.get("FAUCET_PROXY_USERNAME", self.username)
        self.password = os.environ.get("FAUCET_PROXY_PASSWORD", self.password)

    @property
    def proxy_url(self) -> str | None:
        if not self.enabled or not self.endpoint:
            return None
        if self.username:
            return f"http://{self.username}:{self.password}@{self.endpoint}"
        return f"http://{self.endpoint}"


@dataclass
class VisionConfig:
    """Vision-based captcha solving configuration."""

    enabled: bool = False
    ollama_url: str = "http://localhost:11434"
    model: str = "llava"
    whisper_model: str = "base"
    audio_fallback: bool = True

    def __post_init__(self):
        self.enabled = os.environ.get("FAUCET_VISION_ENABLED", str(self.enabled)).lower() in (
            "true",
            "1",
            "yes",
        )
        self.ollama_url = os.environ.get("FAUCET_OLLAMA_URL", self.ollama_url)
        self.model = os.environ.get("FAUCET_VISION_MODEL", self.model)


@dataclass
class ConsolidationConfig:
    """Treasury consolidation settings."""

    enabled: bool = True
    schedule_cron: str = "0 3 * * 0"  # weekly Sunday 3am
    min_balance_usd: float = 1.0  # minimum before sweeping
    gas_price_max_gwei: int = 30  # skip if gas too high
    cold_wallet_addresses: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if env_wallets := os.environ.get("FAUCET_COLD_WALLETS"):
            self.cold_wallet_addresses = json.loads(env_wallets)


@dataclass
class AlertConfig:
    """Alert/notification configuration."""

    discord_webhook_url: str = ""
    alert_on_ban: bool = True
    alert_on_consolidation: bool = True
    daily_summary: bool = True
    daily_summary_cron: str = "0 9 * * *"

    def __post_init__(self):
        self.discord_webhook_url = os.environ.get(
            "FAUCET_DISCORD_WEBHOOK", self.discord_webhook_url
        )


@dataclass
class FaucetConfig:
    """Root configuration for the faucet module."""

    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    faucets: list[FaucetDefinition] = field(default_factory=list)
    # Human-like behavior
    jitter_seconds: int = 300  # ±5min random on schedules
    user_agent: str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    headless: bool = True
    # Auto-disable after N consecutive failures
    auto_disable_threshold: int = 10

    def __post_init__(self):
        if env_dir := os.environ.get("FAUCET_DATA_DIR"):
            self.data_dir = Path(env_dir)
        if env_jitter := os.environ.get("FAUCET_JITTER_SECONDS"):
            self.jitter_seconds = int(env_jitter)

    @classmethod
    def from_yaml(cls, path: Path) -> FaucetConfig:
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> FaucetConfig:
        faucets = [FaucetDefinition.from_dict(f) for f in data.get("faucets", [])]

        proxy = ProxyConfig(**data["proxy"]) if "proxy" in data else ProxyConfig()
        vision = VisionConfig(**data["vision"]) if "vision" in data else VisionConfig()
        consolidation = (
            ConsolidationConfig(**data["consolidation"])
            if "consolidation" in data
            else ConsolidationConfig()
        )
        alerts = AlertConfig(**data["alerts"]) if "alerts" in data else AlertConfig()

        config = cls(
            proxy=proxy,
            vision=vision,
            consolidation=consolidation,
            alerts=alerts,
            faucets=faucets,
            jitter_seconds=data.get("jitter_seconds", 300),
            user_agent=data.get("user_agent", cls.user_agent),
            headless=data.get("headless", True),
            auto_disable_threshold=data.get("auto_disable_threshold", 10),
        )

        if "data_dir" in data:
            config.data_dir = Path(data["data_dir"])

        return config

    def ensure_dirs(self) -> None:
        """Create data directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "claims").mkdir(exist_ok=True)
        (self.data_dir / "health").mkdir(exist_ok=True)

    @property
    def claims_file(self) -> Path:
        return self.data_dir / "claims" / "history.json"

    @property
    def health_file(self) -> Path:
        return self.data_dir / "health" / "status.json"

    @property
    def faucets_file(self) -> Path:
        return self.data_dir / "faucets.yaml"
