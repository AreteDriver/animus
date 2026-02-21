"""Pydantic models for Animus configuration.

Nested section models use plain ``BaseModel`` to avoid pydantic-settings
reading environment variables for fields like ``path`` (which would
collide with ``$PATH``).  Only the top-level :class:`AnimusConfig` extends
``BaseSettings``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class AnimusSection(BaseModel):
    """Core Animus settings."""

    version: str = "0.1.0"
    first_run: bool = True
    data_dir: str = "~/.local/share/animus"


class ApiSection(BaseModel):
    """API key configuration."""

    anthropic_key: str = ""
    openai_key: str = ""


class ForgeSection(BaseModel):
    """Forge orchestration engine settings."""

    enabled: bool = False
    host: str = "localhost"
    port: int = 8000
    api_key: str = ""


class MemorySection(BaseModel):
    """Memory backend settings."""

    backend: str = "sqlite"
    path: str = "~/.local/share/animus/memory.db"
    max_context_tokens: int = 100_000


class IdentitySection(BaseModel):
    """User identity settings."""

    name: str = ""
    timezone: str = ""
    locale: str = ""
    identity_dir: str = "~/.config/animus/identity"


class OllamaSection(BaseModel):
    """Ollama configuration."""

    enabled: bool = True
    host: str = "localhost"
    port: int = 11434
    model: str = "llama3.2"
    autoinstall: bool = True


class SelfImprovementSection(BaseModel):
    """Self-improvement loop configuration."""

    reflection_enabled: bool = True
    reflection_interval_hours: int = 24
    reflection_min_interactions: int = 10
    approval_required: bool = True
    proposals_dir: str = "~/.config/animus/proposals"


class ServicesSection(BaseModel):
    """Background services settings."""

    autostart: bool = True
    port: int = 7700
    log_level: str = "info"
    update_check: bool = True


class GatewaySection(BaseModel):
    """Gateway core settings."""

    enabled: bool = True
    default_backend: str = "anthropic"
    system_prompt: str = ""
    max_response_tokens: int = 4096


class WebchatChannelConfig(BaseModel):
    """Built-in webchat channel configuration."""

    enabled: bool = True


class TelegramChannelConfig(BaseModel):
    """Telegram channel configuration."""

    enabled: bool = False
    bot_token: str = ""


class DiscordChannelConfig(BaseModel):
    """Discord channel configuration."""

    enabled: bool = False
    bot_token: str = ""
    allowed_guilds: list[str] = Field(default_factory=list)


class SlackChannelConfig(BaseModel):
    """Slack channel configuration."""

    enabled: bool = False
    bot_token: str = ""
    app_token: str = ""


class MatrixChannelConfig(BaseModel):
    """Matrix channel configuration."""

    enabled: bool = False
    homeserver: str = ""
    access_token: str = ""
    room_ids: list[str] = Field(default_factory=list)


class SignalChannelConfig(BaseModel):
    """Signal channel configuration."""

    enabled: bool = False
    phone_number: str = ""


class WhatsappChannelConfig(BaseModel):
    """WhatsApp channel configuration."""

    enabled: bool = False
    phone_number: str = ""


class EmailChannelConfig(BaseModel):
    """Email channel configuration."""

    enabled: bool = False
    imap_host: str = ""
    smtp_host: str = ""
    username: str = ""
    password: str = ""
    poll_interval: int = 60


class ChannelsSection(BaseModel):
    """Channel adapter configurations."""

    webchat: WebchatChannelConfig = Field(default_factory=WebchatChannelConfig)
    telegram: TelegramChannelConfig = Field(default_factory=TelegramChannelConfig)
    discord: DiscordChannelConfig = Field(default_factory=DiscordChannelConfig)
    slack: SlackChannelConfig = Field(default_factory=SlackChannelConfig)
    matrix: MatrixChannelConfig = Field(default_factory=MatrixChannelConfig)
    signal: SignalChannelConfig = Field(default_factory=SignalChannelConfig)
    whatsapp: WhatsappChannelConfig = Field(default_factory=WhatsappChannelConfig)
    email: EmailChannelConfig = Field(default_factory=EmailChannelConfig)


class MCPConfig(BaseModel):
    """MCP server configuration."""

    config_path: str = "~/.config/animus/mcp.json"
    auto_discover: bool = True


class IntelligenceSection(BaseModel):
    """Intelligence layer settings."""

    enabled: bool = True
    memory_backend: str = "sqlite"  # "sqlite" | "chromadb" | "animus"
    memory_db_path: str = "~/.local/share/animus/intelligence.db"
    tool_approval_default: str = "auto"  # "auto" | "approve" | "deny"
    max_tool_calls_per_turn: int = 5
    tool_timeout_seconds: int = 30
    mcp: MCPConfig = Field(default_factory=MCPConfig)


class ProactiveCheckConfig(BaseModel):
    """Configuration for a single proactive check."""

    enabled: bool = True
    schedule: str = ""
    channels: list[str] = Field(default_factory=list)


class ProactiveSection(BaseModel):
    """Proactive engine settings."""

    enabled: bool = True
    quiet_hours_start: str = "22:00"
    quiet_hours_end: str = "07:00"
    timezone: str = "UTC"
    checks: dict[str, ProactiveCheckConfig] = Field(default_factory=dict)


class PersonaVoiceConfig(BaseModel):
    """Voice configuration for a persona profile."""

    tone: str = "balanced"
    max_response_length: str = "medium"
    emoji_policy: str = "minimal"
    language: str = "en"
    custom_instructions: str = ""


class PersonaProfileConfig(BaseModel):
    """A persona profile in config."""

    name: str = ""
    description: str = ""
    system_prompt: str = ""
    tone: str = "balanced"
    knowledge_domains: list[str] = Field(default_factory=list)
    excluded_topics: list[str] = Field(default_factory=list)
    channel_bindings: dict[str, bool] = Field(default_factory=dict)


class PersonasSection(BaseModel):
    """Persona system settings."""

    enabled: bool = True
    default_name: str = "Animus"
    default_tone: str = "balanced"
    default_max_response_length: str = "medium"
    default_emoji_policy: str = "minimal"
    default_system_prompt: str = "You are Animus, a personal AI assistant."
    profiles: dict[str, PersonaProfileConfig] = Field(default_factory=dict)


class AnimusConfig(BaseSettings):
    """Top-level Animus configuration model.

    Maps to the TOML structure:
        [animus] / [api] / [forge] / [memory] / [identity] / [services]
        [gateway] / [channels] / [intelligence] / [proactive] / [personas]

    All fields are optional with sensible defaults. Config file lives at
    ``~/.config/animus/config.toml``.
    """

    animus: AnimusSection = Field(default_factory=AnimusSection)
    api: ApiSection = Field(default_factory=ApiSection)
    forge: ForgeSection = Field(default_factory=ForgeSection)
    memory: MemorySection = Field(default_factory=MemorySection)
    identity: IdentitySection = Field(default_factory=IdentitySection)
    services: ServicesSection = Field(default_factory=ServicesSection)
    gateway: GatewaySection = Field(default_factory=GatewaySection)
    channels: ChannelsSection = Field(default_factory=ChannelsSection)
    intelligence: IntelligenceSection = Field(default_factory=IntelligenceSection)
    proactive: ProactiveSection = Field(default_factory=ProactiveSection)
    personas: PersonasSection = Field(default_factory=PersonasSection)
    ollama: OllamaSection = Field(default_factory=OllamaSection)
    self_improvement: SelfImprovementSection = Field(default_factory=SelfImprovementSection)

    def get_data_path(self) -> Path:
        """Return the resolved data directory path."""
        return Path(self.animus.data_dir).expanduser()

    def get_memory_path(self) -> Path:
        """Return the resolved memory database path."""
        return Path(self.memory.path).expanduser()
