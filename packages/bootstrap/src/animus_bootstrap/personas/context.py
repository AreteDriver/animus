"""Context-aware prompt adaptation for personas."""

from __future__ import annotations

from datetime import UTC, datetime

from animus_bootstrap.gateway.models import GatewayMessage
from animus_bootstrap.personas.engine import PersonaProfile
from animus_bootstrap.personas.voice import build_voice_prompt, get_time_shift_tone

# Channel-specific tone adjustments
_CHANNEL_NORMS: dict[str, str] = {
    "slack": (
        "This is a Slack conversation. Keep responses well-formatted "
        "with occasional emoji. Use threads-style concise replies."
    ),
    "discord": (
        "This is a Discord chat. Casual tone is appropriate. Shorter messages work better."
    ),
    "email": ("This is an email. Use proper greeting and sign-off. Be thorough and professional."),
    "telegram": "This is a Telegram chat. Keep it conversational and mobile-friendly.",
    "matrix": "This is a Matrix chat. Technical audience is likely.",
    "webchat": "",  # No special norms
    "whatsapp": "This is a WhatsApp message. Keep it brief and conversational.",
    "signal": "This is a Signal message. Keep it brief and conversational.",
}


class ContextAdapter:
    """Adapts persona behavior based on context signals."""

    def adapt_prompt(
        self,
        persona: PersonaProfile,
        message: GatewayMessage,
        session_history: list[dict] | None = None,
        now: datetime | None = None,
    ) -> str:
        """Build a fully context-adapted system prompt.

        Combines:
        1. Persona's base system_prompt
        2. Voice configuration instructions
        3. Time-of-day tone shift
        4. Channel-specific norms
        5. Conversation length adaptation
        6. Knowledge domain hints
        """
        now = now or datetime.now(UTC)
        parts: list[str] = []

        # 1. Base system prompt
        if persona.system_prompt:
            parts.append(persona.system_prompt)

        # 2. Voice instructions
        voice_prompt = build_voice_prompt(persona.voice)
        if voice_prompt:
            parts.append(voice_prompt)

        # 3. Time-of-day shift
        time_tone = get_time_shift_tone(persona.voice, now.hour)
        if time_tone:
            parts.append(f"Current time tone: {time_tone}")

        # 4. Channel norms
        channel_norm = _CHANNEL_NORMS.get(message.channel, "")
        if channel_norm:
            parts.append(channel_norm)

        # 5. Conversation length adaptation
        if session_history:
            turn_count = len(session_history)
            if turn_count > 20:
                parts.append("This is a long conversation. Be concise to stay focused.")
            elif turn_count > 10:
                parts.append("The conversation is progressing. Stay on topic.")

        # 6. Knowledge domain hints
        if persona.knowledge_domains:
            domains = ", ".join(persona.knowledge_domains)
            parts.append(f"Your areas of expertise: {domains}")

        if persona.excluded_topics:
            excluded = ", ".join(persona.excluded_topics)
            parts.append(f"Topics to decline or defer: {excluded}")

        return "\n\n".join(parts)
