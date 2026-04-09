"""Configuration for Animus Bounty."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DATA_DIR = Path.home() / ".animus" / "bounty"


@dataclass
class LLMConfig:
    """LLM configuration for skill matching."""

    provider: str = "ollama"  # ollama, anthropic
    ollama_url: str = "http://localhost:11434"
    model: str = "qwen2.5:14b"
    anthropic_model: str = "claude-haiku-4-5-20251001"
    timeout: float = 60.0

    def __post_init__(self):
        self.provider = os.environ.get("BOUNTY_LLM_PROVIDER", self.provider)
        self.ollama_url = os.environ.get("BOUNTY_OLLAMA_URL", self.ollama_url)
        self.model = os.environ.get("BOUNTY_LLM_MODEL", self.model)


@dataclass
class SkillProfile:
    """Developer skill profile for matching bounties."""

    languages: list[str] = field(
        default_factory=lambda: ["python", "rust", "typescript", "solidity"]
    )
    frameworks: list[str] = field(default_factory=lambda: ["fastapi", "react", "nextjs", "bevy"])
    domains: list[str] = field(default_factory=lambda: ["ai", "blockchain", "devtools", "web"])

    def all_skills(self) -> list[str]:
        """Return all skills as a flat list."""
        return self.languages + self.frameworks + self.domains


@dataclass
class FilterConfig:
    """Bounty filter settings."""

    min_reward_usd: float = 5.0
    max_difficulty: str = "hard"
    auto_claim_low_effort: bool = True
    auto_claim_max_reward_usd: float = 50.0
    platforms: list[str] = field(default_factory=list)  # empty = all platforms
    exclude_labels: list[str] = field(default_factory=lambda: ["wontfix", "invalid"])


@dataclass
class AlertConfig:
    """Notification settings."""

    discord_webhook_url: str = ""
    alert_on_high_value: bool = True
    high_value_threshold_usd: float = 100.0

    def __post_init__(self):
        self.discord_webhook_url = os.environ.get(
            "BOUNTY_DISCORD_WEBHOOK", self.discord_webhook_url
        )


@dataclass
class PlatformCredentials:
    """API credentials for bounty platforms."""

    github_token: str = ""
    gitcoin_api_key: str = ""
    replit_token: str = ""

    def __post_init__(self):
        self.github_token = os.environ.get("GITHUB_TOKEN", self.github_token)
        self.gitcoin_api_key = os.environ.get("GITCOIN_API_KEY", self.gitcoin_api_key)
        self.replit_token = os.environ.get("REPLIT_TOKEN", self.replit_token)


@dataclass
class BountyConfig:
    """Root configuration for the bounty module."""

    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    llm: LLMConfig = field(default_factory=LLMConfig)
    skills: SkillProfile = field(default_factory=SkillProfile)
    filters: FilterConfig = field(default_factory=FilterConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    credentials: PlatformCredentials = field(default_factory=PlatformCredentials)
    scan_interval_hours: int = 4

    def __post_init__(self):
        if env_dir := os.environ.get("BOUNTY_DATA_DIR"):
            self.data_dir = Path(env_dir)

    @classmethod
    def from_yaml(cls, path: Path) -> BountyConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> BountyConfig:
        llm = LLMConfig(**data["llm"]) if "llm" in data else LLMConfig()
        skills = SkillProfile(**data["skills"]) if "skills" in data else SkillProfile()
        filters = FilterConfig(**data["filters"]) if "filters" in data else FilterConfig()
        alerts = AlertConfig(**data["alerts"]) if "alerts" in data else AlertConfig()
        credentials = (
            PlatformCredentials(**data["credentials"])
            if "credentials" in data
            else PlatformCredentials()
        )

        config = cls(
            llm=llm,
            skills=skills,
            filters=filters,
            alerts=alerts,
            credentials=credentials,
        )

        if "data_dir" in data:
            config.data_dir = Path(data["data_dir"])

        if "scan_interval_hours" in data:
            config.scan_interval_hours = data["scan_interval_hours"]

        return config

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def config_file(self) -> Path:
        return self.data_dir / "config.yaml"

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dict (excluding secrets)."""
        return {
            "data_dir": str(self.data_dir),
            "scan_interval_hours": self.scan_interval_hours,
            "skills": {
                "languages": self.skills.languages,
                "frameworks": self.skills.frameworks,
                "domains": self.skills.domains,
            },
            "filters": {
                "min_reward_usd": self.filters.min_reward_usd,
                "max_difficulty": self.filters.max_difficulty,
                "auto_claim_low_effort": self.filters.auto_claim_low_effort,
                "auto_claim_max_reward_usd": self.filters.auto_claim_max_reward_usd,
                "platforms": self.filters.platforms,
                "exclude_labels": self.filters.exclude_labels,
            },
        }
