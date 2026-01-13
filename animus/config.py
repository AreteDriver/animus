"""
Animus Configuration Management

Loads configuration from YAML file with environment variable overrides.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def default_data_dir() -> Path:
    """Return the default data directory."""
    return Path.home() / ".animus"


@dataclass
class ModelConfig:
    """Configuration for the cognitive model."""

    provider: str = "ollama"
    name: str = "llama3:8b"
    ollama_url: str = "http://localhost:11434"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    def __post_init__(self):
        # Allow environment overrides
        self.provider = os.environ.get("ANIMUS_MODEL_PROVIDER", self.provider)
        self.name = os.environ.get("ANIMUS_MODEL_NAME", self.name)
        self.ollama_url = os.environ.get("ANIMUS_OLLAMA_URL", self.ollama_url)
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", self.anthropic_api_key)
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", self.openai_api_key)


@dataclass
class MemoryConfig:
    """Configuration for the memory system."""

    backend: str = "chroma"  # "chroma" or "json"
    collection_name: str = "animus_memories"

    def __post_init__(self):
        self.backend = os.environ.get("ANIMUS_MEMORY_BACKEND", self.backend)


@dataclass
class AnimusConfig:
    """Main configuration for Animus."""

    data_dir: Path = field(default_factory=default_data_dir)
    log_level: str = "INFO"
    log_to_file: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    def __post_init__(self):
        # Convert string to Path if needed
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)

        # Expand user path
        self.data_dir = self.data_dir.expanduser()

        # Environment overrides
        if env_data_dir := os.environ.get("ANIMUS_DATA_DIR"):
            self.data_dir = Path(env_data_dir).expanduser()
        self.log_level = os.environ.get("ANIMUS_LOG_LEVEL", self.log_level)

    @property
    def config_file(self) -> Path:
        return self.data_dir / "config.yaml"

    @property
    def log_file(self) -> Path:
        return self.data_dir / "animus.log"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def history_file(self) -> Path:
        return self.data_dir / "history"

    def ensure_dirs(self) -> None:
        """Create necessary directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "data_dir": str(self.data_dir),
            "log_level": self.log_level,
            "log_to_file": self.log_to_file,
            "model": {
                "provider": self.model.provider,
                "name": self.model.name,
                "ollama_url": self.model.ollama_url,
            },
            "memory": {
                "backend": self.memory.backend,
                "collection_name": self.memory.collection_name,
            },
        }

    def save(self) -> None:
        """Save configuration to YAML file."""
        self.ensure_dirs()
        with open(self.config_file, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, config_path: Path | None = None) -> "AnimusConfig":
        """
        Load configuration from file.

        Precedence (highest to lowest):
        1. Environment variables
        2. Config file values
        3. Default values
        """
        config = cls()

        # Determine config file path
        if config_path is None:
            config_path = config.config_file

        # Load from file if exists
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}

            # Apply file values (env vars applied in __post_init__)
            if "data_dir" in data:
                config.data_dir = Path(data["data_dir"]).expanduser()
            if "log_level" in data:
                config.log_level = data["log_level"]
            if "log_to_file" in data:
                config.log_to_file = data["log_to_file"]

            if model_data := data.get("model"):
                if "provider" in model_data:
                    config.model.provider = model_data["provider"]
                if "name" in model_data:
                    config.model.name = model_data["name"]
                if "ollama_url" in model_data:
                    config.model.ollama_url = model_data["ollama_url"]

            if memory_data := data.get("memory"):
                if "backend" in memory_data:
                    config.memory.backend = memory_data["backend"]
                if "collection_name" in memory_data:
                    config.memory.collection_name = memory_data["collection_name"]

            # Re-apply environment overrides
            config.__post_init__()
            config.model.__post_init__()
            config.memory.__post_init__()

        return config


def get_config() -> AnimusConfig:
    """Get the global configuration, loading from default location."""
    return AnimusConfig.load()
