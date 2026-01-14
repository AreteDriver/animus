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
class GoogleIntegrationConfig:
    """Configuration for Google integrations (Calendar, Gmail)."""

    enabled: bool = False
    client_id: str | None = None
    client_secret: str | None = None

    def __post_init__(self):
        if env_enabled := os.environ.get("GOOGLE_INTEGRATION_ENABLED"):
            self.enabled = env_enabled.lower() in ("true", "1", "yes")
        self.client_id = os.environ.get("GOOGLE_CLIENT_ID", self.client_id)
        self.client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", self.client_secret)


@dataclass
class TodoistConfig:
    """Configuration for Todoist integration."""

    enabled: bool = False
    api_key: str | None = None

    def __post_init__(self):
        if env_enabled := os.environ.get("TODOIST_ENABLED"):
            self.enabled = env_enabled.lower() in ("true", "1", "yes")
        self.api_key = os.environ.get("TODOIST_API_KEY", self.api_key)


@dataclass
class FilesystemConfig:
    """Configuration for filesystem integration."""

    enabled: bool = False
    indexed_paths: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(
        default_factory=lambda: ["*.pyc", "__pycache__", ".git", "node_modules", ".venv"]
    )

    def __post_init__(self):
        if env_enabled := os.environ.get("FILESYSTEM_INTEGRATION_ENABLED"):
            self.enabled = env_enabled.lower() in ("true", "1", "yes")


@dataclass
class WebhookConfig:
    """Configuration for webhook receiver."""

    enabled: bool = False
    port: int = 8421
    secret: str | None = None

    def __post_init__(self):
        if env_enabled := os.environ.get("WEBHOOK_ENABLED"):
            self.enabled = env_enabled.lower() in ("true", "1", "yes")
        if env_port := os.environ.get("WEBHOOK_PORT"):
            self.port = int(env_port)
        self.secret = os.environ.get("WEBHOOK_SECRET", self.secret)


@dataclass
class IntegrationConfig:
    """Configuration for all external integrations."""

    google: GoogleIntegrationConfig = field(default_factory=GoogleIntegrationConfig)
    todoist: TodoistConfig = field(default_factory=TodoistConfig)
    filesystem: FilesystemConfig = field(default_factory=FilesystemConfig)
    webhooks: WebhookConfig = field(default_factory=WebhookConfig)


@dataclass
class ToolsSecurityConfig:
    """Security configuration for tool execution."""

    # File system access restrictions
    allowed_paths: list[str] = field(default_factory=lambda: [str(Path.home())])
    blocked_paths: list[str] = field(
        default_factory=lambda: [
            "/etc/shadow",
            "/etc/passwd",
            "/etc/sudoers",
            "~/.ssh/id_*",
            "~/.gnupg",
            "~/.aws/credentials",
            "~/.config/gcloud",
        ]
    )
    max_file_size_kb: int = 1000  # 1MB default

    # Command execution restrictions
    command_enabled: bool = True
    command_blocklist: list[str] = field(
        default_factory=lambda: [
            "rm -rf /",
            "rm -rf ~",
            "dd if=",
            "mkfs",
            ":(){:|:&};:",  # Fork bomb
            "chmod -R 777 /",
            "curl.*|.*sh",
            "wget.*|.*sh",
        ]
    )
    command_timeout_seconds: int = 30


@dataclass
class LearningConfig:
    """Configuration for the self-learning system."""

    enabled: bool = True
    auto_scan_enabled: bool = True
    auto_scan_interval_hours: int = 24
    min_pattern_occurrences: int = 3
    min_pattern_confidence: float = 0.6
    lookback_days: int = 30
    max_pending_approvals: int = 50

    def __post_init__(self):
        if env_enabled := os.environ.get("ANIMUS_LEARNING_ENABLED"):
            self.enabled = env_enabled.lower() in ("true", "1", "yes")
        if env_auto_scan := os.environ.get("ANIMUS_LEARNING_AUTO_SCAN"):
            self.auto_scan_enabled = env_auto_scan.lower() in ("true", "1", "yes")


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
    integrations: IntegrationConfig = field(default_factory=IntegrationConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    tools_security: ToolsSecurityConfig = field(default_factory=ToolsSecurityConfig)

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
            "integrations": {
                "google": {
                    "enabled": self.integrations.google.enabled,
                },
                "todoist": {
                    "enabled": self.integrations.todoist.enabled,
                },
                "filesystem": {
                    "enabled": self.integrations.filesystem.enabled,
                    "indexed_paths": self.integrations.filesystem.indexed_paths,
                    "exclude_patterns": self.integrations.filesystem.exclude_patterns,
                },
                "webhooks": {
                    "enabled": self.integrations.webhooks.enabled,
                    "port": self.integrations.webhooks.port,
                },
            },
            "learning": {
                "enabled": self.learning.enabled,
                "auto_scan_enabled": self.learning.auto_scan_enabled,
                "auto_scan_interval_hours": self.learning.auto_scan_interval_hours,
                "min_pattern_occurrences": self.learning.min_pattern_occurrences,
                "min_pattern_confidence": self.learning.min_pattern_confidence,
                "lookback_days": self.learning.lookback_days,
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

            if integrations_data := data.get("integrations"):
                if google_data := integrations_data.get("google"):
                    if "enabled" in google_data:
                        config.integrations.google.enabled = google_data["enabled"]
                    if "client_id" in google_data:
                        config.integrations.google.client_id = google_data["client_id"]
                    if "client_secret" in google_data:
                        config.integrations.google.client_secret = google_data["client_secret"]
                if todoist_data := integrations_data.get("todoist"):
                    if "enabled" in todoist_data:
                        config.integrations.todoist.enabled = todoist_data["enabled"]
                    if "api_key" in todoist_data:
                        config.integrations.todoist.api_key = todoist_data["api_key"]
                if fs_data := integrations_data.get("filesystem"):
                    if "enabled" in fs_data:
                        config.integrations.filesystem.enabled = fs_data["enabled"]
                    if "indexed_paths" in fs_data:
                        config.integrations.filesystem.indexed_paths = fs_data["indexed_paths"]
                    if "exclude_patterns" in fs_data:
                        config.integrations.filesystem.exclude_patterns = fs_data[
                            "exclude_patterns"
                        ]
                if webhook_data := integrations_data.get("webhooks"):
                    if "enabled" in webhook_data:
                        config.integrations.webhooks.enabled = webhook_data["enabled"]
                    if "port" in webhook_data:
                        config.integrations.webhooks.port = webhook_data["port"]
                    if "secret" in webhook_data:
                        config.integrations.webhooks.secret = webhook_data["secret"]

            if learning_data := data.get("learning"):
                if "enabled" in learning_data:
                    config.learning.enabled = learning_data["enabled"]
                if "auto_scan_enabled" in learning_data:
                    config.learning.auto_scan_enabled = learning_data["auto_scan_enabled"]
                if "auto_scan_interval_hours" in learning_data:
                    config.learning.auto_scan_interval_hours = learning_data[
                        "auto_scan_interval_hours"
                    ]
                if "min_pattern_occurrences" in learning_data:
                    config.learning.min_pattern_occurrences = learning_data[
                        "min_pattern_occurrences"
                    ]
                if "min_pattern_confidence" in learning_data:
                    config.learning.min_pattern_confidence = learning_data[
                        "min_pattern_confidence"
                    ]
                if "lookback_days" in learning_data:
                    config.learning.lookback_days = learning_data["lookback_days"]

            # Re-apply environment overrides
            config.__post_init__()
            config.model.__post_init__()
            config.memory.__post_init__()
            config.api.__post_init__()
            config.voice.__post_init__()
            config.integrations.google.__post_init__()
            config.integrations.todoist.__post_init__()
            config.integrations.filesystem.__post_init__()
            config.integrations.webhooks.__post_init__()
            config.learning.__post_init__()

        return config


def get_config() -> AnimusConfig:
    """Get the global configuration, loading from default location."""
    return AnimusConfig.load()
