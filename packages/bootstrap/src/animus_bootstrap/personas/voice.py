"""Voice configuration and tone presets for persona system."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VoiceConfig:
    """Tone and style configuration for a persona."""

    tone: str = "balanced"  # formal, casual, technical, mentor, creative, balanced
    max_response_length: str = "medium"  # brief, medium, detailed
    emoji_policy: str = "minimal"  # none, minimal, expressive
    language: str = "en"
    custom_instructions: str = ""
    time_shifts: dict[str, str] = field(default_factory=dict)  # "morning": "energetic"


VOICE_PRESETS: dict[str, VoiceConfig] = {
    "formal": VoiceConfig(
        tone="formal",
        emoji_policy="none",
        custom_instructions=(
            "Use professional language. Avoid contractions. Be precise and structured."
        ),
    ),
    "casual": VoiceConfig(
        tone="casual",
        emoji_policy="minimal",
        custom_instructions="Be friendly and conversational. Use everyday language.",
    ),
    "technical": VoiceConfig(
        tone="technical",
        emoji_policy="none",
        max_response_length="detailed",
        custom_instructions=(
            "Be precise. Use technical terminology. Include code examples when relevant."
        ),
    ),
    "mentor": VoiceConfig(
        tone="mentor",
        emoji_policy="minimal",
        custom_instructions=(
            "Guide and teach. Explain reasoning. Ask Socratic questions. Encourage learning."
        ),
    ),
    "creative": VoiceConfig(
        tone="creative",
        emoji_policy="expressive",
        custom_instructions=(
            "Be imaginative and playful. Use metaphors and vivid language. Think outside the box."
        ),
    ),
    "balanced": VoiceConfig(),  # Default — no special instructions
}


def build_voice_prompt(voice: VoiceConfig) -> str:
    """Build a system prompt fragment from a VoiceConfig."""
    parts: list[str] = []

    # Tone instruction
    tone_instructions = {
        "formal": "Maintain a professional and formal tone.",
        "casual": "Use a casual, friendly tone.",
        "technical": "Be technically precise and detailed.",
        "mentor": "Adopt a mentoring tone — guide and teach.",
        "creative": "Be creative, imaginative, and expressive.",
        "balanced": "Use a balanced, neutral tone.",
    }
    if voice.tone in tone_instructions:
        parts.append(tone_instructions[voice.tone])

    # Response length
    length_instructions = {
        "brief": "Keep responses concise — 1-3 sentences when possible.",
        "medium": "Provide moderately detailed responses.",
        "detailed": "Give thorough, comprehensive responses with examples.",
    }
    if voice.max_response_length in length_instructions:
        parts.append(length_instructions[voice.max_response_length])

    # Emoji policy
    emoji_instructions = {
        "none": "Do not use emojis.",
        "minimal": "Use emojis sparingly, only when they add clarity.",
        "expressive": "Use emojis freely to express emotions and emphasis.",
    }
    if voice.emoji_policy in emoji_instructions:
        parts.append(emoji_instructions[voice.emoji_policy])

    # Custom instructions
    if voice.custom_instructions:
        parts.append(voice.custom_instructions)

    return "\n".join(parts)


def get_time_shift_tone(voice: VoiceConfig, hour: int) -> str | None:
    """Get a time-of-day tone shift if configured.

    Maps hour ranges to time periods:
    - morning: 6-11
    - afternoon: 12-16
    - evening: 17-21
    - night: 22-5
    """
    if not voice.time_shifts:
        return None

    if 6 <= hour <= 11:
        return voice.time_shifts.get("morning")
    if 12 <= hour <= 16:
        return voice.time_shifts.get("afternoon")
    if 17 <= hour <= 21:
        return voice.time_shifts.get("evening")
    return voice.time_shifts.get("night")
