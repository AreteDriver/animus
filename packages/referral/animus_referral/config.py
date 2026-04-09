"""Configuration for Animus Referral."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from animus_referral.models import ReferralProgram

logger = logging.getLogger("animus_referral.config")

DEFAULT_DATA_DIR = Path.home() / ".animus" / "referral"


@dataclass
class ReferralConfig:
    """Root configuration for the referral module."""

    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    programs: list[ReferralProgram] = field(default_factory=list)
    # Alert thresholds
    high_value_threshold_usd: float = 50.0
    # Auto-discovery
    auto_discover: bool = True

    def __post_init__(self):
        if env_dir := os.environ.get("REFERRAL_DATA_DIR"):
            self.data_dir = Path(env_dir)
        if env_threshold := os.environ.get("REFERRAL_HIGH_VALUE_THRESHOLD"):
            self.high_value_threshold_usd = float(env_threshold)

    @classmethod
    def from_yaml(cls, path: Path) -> ReferralConfig:
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> ReferralConfig:
        programs = [ReferralProgram.from_dict(p) for p in data.get("programs", [])]
        config = cls(
            programs=programs,
            high_value_threshold_usd=data.get("high_value_threshold_usd", 50.0),
            auto_discover=data.get("auto_discover", True),
        )
        if "data_dir" in data:
            config.data_dir = Path(data["data_dir"])
        return config

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dict."""
        return {
            "data_dir": str(self.data_dir),
            "programs": [p.to_dict() for p in self.programs],
            "high_value_threshold_usd": self.high_value_threshold_usd,
            "auto_discover": self.auto_discover,
        }

    def save_yaml(self, path: Path) -> None:
        """Save config to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def ensure_dirs(self) -> None:
        """Create data directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def config_file(self) -> Path:
        return self.data_dir / "referral.yaml"

    def add_program(self, program: ReferralProgram) -> bool:
        """Add a program if not already present. Returns True if added."""
        if any(p.name == program.name for p in self.programs):
            return False
        self.programs.append(program)
        return True

    def get_program(self, name: str) -> ReferralProgram | None:
        """Look up a program by name."""
        return next((p for p in self.programs if p.name == name), None)

    def get_active_programs(self) -> list[ReferralProgram]:
        """Return only active programs."""
        return [p for p in self.programs if p.active]

    def get_high_value_programs(self) -> list[ReferralProgram]:
        """Return programs with referrer bonus above threshold."""
        return [
            p
            for p in self.programs
            if p.active and p.bonus_referrer_usd >= self.high_value_threshold_usd
        ]
