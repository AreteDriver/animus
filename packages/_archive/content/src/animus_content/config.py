"""Configuration for the content module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PlatformCredentials(BaseModel):
    """API credentials for a single platform."""

    api_key: str = ""
    api_secret: str = ""
    base_url: str = ""
    extra: dict[str, str] = Field(default_factory=dict)


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = "ollama"
    model: str = "qwen2.5:14b"
    base_url: str = "http://localhost:11434"
    api_key: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7


class ContentConfig(BaseModel):
    """Top-level configuration for the content module."""

    data_dir: Path = Field(default_factory=lambda: Path.home() / ".animus" / "content")
    llm: LLMConfig = Field(default_factory=LLMConfig)
    platforms: dict[str, PlatformCredentials] = Field(default_factory=dict)
    default_tags: list[str] = Field(default_factory=lambda: ["crypto", "web3", "blockchain"])
    max_articles_per_day: int = 5
    min_word_count: int = 500
    max_word_count: int = 2500
    duplicate_topic_cooldown_days: int = 7

    @classmethod
    def load(cls, path: Path) -> ContentConfig:
        """Load configuration from a YAML file."""
        if not path.exists():
            return cls()
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save configuration to a YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)
