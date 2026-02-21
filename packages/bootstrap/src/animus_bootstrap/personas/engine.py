"""PersonaProfile model and PersonaEngine registry with message routing."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from animus_bootstrap.gateway.models import GatewayMessage
from animus_bootstrap.personas.voice import VoiceConfig

logger = logging.getLogger(__name__)


@dataclass
class PersonaProfile:
    """A complete persona definition."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Animus"
    description: str = "Default personal AI assistant"
    system_prompt: str = "You are Animus, a personal AI assistant."
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    knowledge_domains: list[str] = field(default_factory=list)
    excluded_topics: list[str] = field(default_factory=list)
    channel_bindings: dict[str, bool] = field(default_factory=dict)
    active: bool = True
    is_default: bool = False


class PersonaEngine:
    """Registry of persona profiles. Routes messages to the right persona."""

    def __init__(self) -> None:
        self._personas: dict[str, PersonaProfile] = {}
        self._default_id: str | None = None

    def register_persona(self, persona: PersonaProfile) -> None:
        """Register a persona. If is_default, set as default."""
        self._personas[persona.id] = persona
        if persona.is_default:
            self._default_id = persona.id

    def unregister_persona(self, persona_id: str) -> None:
        """Remove a persona."""
        self._personas.pop(persona_id, None)
        if self._default_id == persona_id:
            self._default_id = None

    def get_persona(self, persona_id: str) -> PersonaProfile | None:
        """Get a persona by ID."""
        return self._personas.get(persona_id)

    def list_personas(self) -> list[PersonaProfile]:
        """List all registered personas."""
        return list(self._personas.values())

    def set_default(self, persona_id: str) -> None:
        """Set a persona as the default. Raises ValueError if not found."""
        if persona_id not in self._personas:
            raise ValueError(f"Persona '{persona_id}' not found")
        # Clear old default
        if self._default_id and self._default_id in self._personas:
            self._personas[self._default_id].is_default = False
        self._default_id = persona_id
        self._personas[persona_id].is_default = True

    def get_default(self) -> PersonaProfile | None:
        """Get the default persona, or first registered if none set."""
        if self._default_id:
            return self._personas.get(self._default_id)
        # Return first persona if no default set
        if self._personas:
            return next(iter(self._personas.values()))
        return None

    def get_persona_for_message(self, message: GatewayMessage) -> PersonaProfile | None:
        """Select best persona for a message.

        Priority:
        1. Explicit /persona command in message text
        2. Channel-bound persona
        3. Default persona
        """
        # 1. Check for /persona command
        if message.text.startswith("/persona "):
            name = message.text[len("/persona ") :].strip()
            for persona in self._personas.values():
                if persona.name.lower() == name.lower() and persona.active:
                    return persona

        # 2. Check channel bindings
        for persona in self._personas.values():
            if not persona.active:
                continue
            if message.channel in persona.channel_bindings:
                if persona.channel_bindings[message.channel]:
                    return persona

        # 3. Fall back to default
        return self.get_default()

    @property
    def persona_count(self) -> int:
        """Number of registered personas."""
        return len(self._personas)
