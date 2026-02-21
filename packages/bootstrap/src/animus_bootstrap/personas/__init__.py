"""Persona & voice layer — identity, tone, and routing."""

from __future__ import annotations

from animus_bootstrap.personas.context import ContextAdapter
from animus_bootstrap.personas.engine import PersonaEngine, PersonaProfile
from animus_bootstrap.personas.knowledge import KnowledgeDomainRouter
from animus_bootstrap.personas.voice import VOICE_PRESETS, VoiceConfig

__all__ = [
    "ContextAdapter",
    "KnowledgeDomainRouter",
    "PersonaEngine",
    "PersonaProfile",
    "VOICE_PRESETS",
    "VoiceConfig",
]
