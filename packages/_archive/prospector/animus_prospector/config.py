"""Configuration for Animus Prospector."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DATA_DIR = Path.home() / ".animus" / "prospector"


@dataclass
class LLMConfig:
    """LLM configuration for opportunity evaluation."""

    provider: str = "ollama"  # ollama, anthropic
    ollama_url: str = "http://localhost:11434"
    model: str = "qwen2.5:14b"
    anthropic_model: str = "claude-haiku-4-5-20251001"
    timeout: float = 60.0

    def __post_init__(self):
        self.provider = os.environ.get("PROSPECTOR_LLM_PROVIDER", self.provider)
        self.ollama_url = os.environ.get("PROSPECTOR_OLLAMA_URL", self.ollama_url)
        self.model = os.environ.get("PROSPECTOR_LLM_MODEL", self.model)


@dataclass
class HunterConfig:
    """Per-hunter settings."""

    enabled: bool = True
    scan_interval_hours: int = 6
    max_results_per_scan: int = 50

    # Source-specific settings
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class AlertConfig:
    """Notification settings."""

    discord_webhook_url: str = ""
    alert_on_high_value: bool = True
    high_value_threshold_usd: float = 10.0
    daily_digest: bool = True
    daily_digest_cron: str = "0 9 * * *"

    def __post_init__(self):
        self.discord_webhook_url = os.environ.get(
            "PROSPECTOR_DISCORD_WEBHOOK", self.discord_webhook_url
        )


@dataclass
class EvaluationConfig:
    """Evaluation thresholds."""

    auto_execute_min_roi: float = 0.5  # $0.50/min minimum
    auto_execute_max_risk: float = 0.2  # max 20% scam probability
    flag_min_value_usd: float = 1.0  # minimum value to flag for review
    skip_max_effort_minutes: int = 120  # skip if > 2 hours effort
    scam_keywords: list[str] = field(
        default_factory=lambda: [
            "send eth to receive",
            "double your",
            "guaranteed profit",
            "private key",
            "seed phrase",
            "connect wallet to claim",
            "urgent",
            "last chance",
            "limited time only",
        ]
    )


@dataclass
class ProspectorConfig:
    """Root configuration for the prospector module."""

    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    llm: LLMConfig = field(default_factory=LLMConfig)
    hunters: HunterConfig = field(default_factory=HunterConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self):
        if env_dir := os.environ.get("PROSPECTOR_DATA_DIR"):
            self.data_dir = Path(env_dir)

    @classmethod
    def from_yaml(cls, path: Path) -> ProspectorConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> ProspectorConfig:
        llm = LLMConfig(**data["llm"]) if "llm" in data else LLMConfig()
        hunters = HunterConfig(**data["hunters"]) if "hunters" in data else HunterConfig()
        alerts = AlertConfig(**data["alerts"]) if "alerts" in data else AlertConfig()
        evaluation = (
            EvaluationConfig(**data["evaluation"]) if "evaluation" in data else EvaluationConfig()
        )

        config = cls(
            llm=llm,
            hunters=hunters,
            alerts=alerts,
            evaluation=evaluation,
        )

        if "data_dir" in data:
            config.data_dir = Path(data["data_dir"])

        return config

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "opportunities").mkdir(exist_ok=True)
        (self.data_dir / "evaluations").mkdir(exist_ok=True)

    @property
    def opportunities_file(self) -> Path:
        return self.data_dir / "opportunities" / "feed.jsonl"

    @property
    def evaluations_file(self) -> Path:
        return self.data_dir / "evaluations" / "results.jsonl"

    @property
    def config_file(self) -> Path:
        return self.data_dir / "config.yaml"
