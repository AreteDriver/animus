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
class APIConfig:
    """Configuration for the HTTP API server."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8420
    api_key: str | None = None

    def __post_init__(self):
        if env_enabled := os.environ.get("ANIMUS_API_ENABLED"):
            self.enabled = env_enabled.lower() in ("true", "1", "yes")
        self.host = os.environ.get("ANIMUS_API_HOST", self.host)
        if env_port := os.environ.get("ANIMUS_API_PORT"):
            self.port = int(env_port)
        self.api_key = os.environ.get("ANIMUS_API_KEY", self.api_key)


@dataclass
class VoiceConfig:
    """Configuration for voice input/output."""

    input_enabled: bool = False
    output_enabled: bool = False
    whisper_model: str = "base"  # tiny, base, small, medium, large
    tts_engine: str = "pyttsx3"  # pyttsx3 or edge-tts
    tts_rate: int = 150

    def __post_init__(self):
        if env_input := os.environ.get("ANIMUS_VOICE_INPUT"):
            self.input_enabled = env_input.lower() in ("true", "1", "yes")
        if env_output := os.environ.get("ANIMUS_VOICE_OUTPUT"):
            self.output_enabled = env_output.lower() in ("true", "1", "yes")
        self.whisper_model = os.environ.get("ANIMUS_WHISPER_MODEL", self.whisper_model)
        self.tts_engine = os.environ.get("ANIMUS_TTS_ENGINE", self.tts_engine)
        if env_rate := os.environ.get("ANIMUS_TTS_RATE"):
            self.tts_rate = int(env_rate)


@dataclass
class AnimusConfig:
    """Main configuration for Animus."""

    data_dir: Path = field(default_factory=default_data_dir)
    log_level: str = "INFO"
    log_to_file: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    api: APIConfig = field(default_factory=APIConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)

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
            "api": {
                "enabled": self.api.enabled,
                "host": self.api.host,
                "port": self.api.port,
            },
            "voice": {
                "input_enabled": self.voice.input_enabled,
                "output_enabled": self.voice.output_enabled,
                "whisper_model": self.voice.whisper_model,
                "tts_engine": self.voice.tts_engine,
                "tts_rate": self.voice.tts_rate,
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

            if api_data := data.get("api"):
                if "enabled" in api_data:
                    config.api.enabled = api_data["enabled"]
                if "host" in api_data:
                    config.api.host = api_data["host"]
                if "port" in api_data:
                    config.api.port = api_data["port"]
                if "api_key" in api_data:
                    config.api.api_key = api_data["api_key"]

            if voice_data := data.get("voice"):
                if "input_enabled" in voice_data:
                    config.voice.input_enabled = voice_data["input_enabled"]
                if "output_enabled" in voice_data:
                    config.voice.output_enabled = voice_data["output_enabled"]
                if "whisper_model" in voice_data:
                    config.voice.whisper_model = voice_data["whisper_model"]
                if "tts_engine" in voice_data:
                    config.voice.tts_engine = voice_data["tts_engine"]
                if "tts_rate" in voice_data:
                    config.voice.tts_rate = voice_data["tts_rate"]

            # Re-apply environment overrides
            config.__post_init__()
            config.model.__post_init__()
            config.memory.__post_init__()
            config.api.__post_init__()
            config.voice.__post_init__()

        return config


def get_config() -> AnimusConfig:
    """Get the global configuration, loading from default location."""
    return AnimusConfig.load()
