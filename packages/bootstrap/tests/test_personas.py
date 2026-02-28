"""Comprehensive tests for the persona system."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from animus_bootstrap.gateway.models import GatewayMessage, create_message
from animus_bootstrap.personas.context import _CHANNEL_NORMS, ContextAdapter
from animus_bootstrap.personas.engine import PersonaEngine, PersonaProfile
from animus_bootstrap.personas.knowledge import _DOMAIN_KEYWORDS, KnowledgeDomainRouter
from animus_bootstrap.personas.storage import PersonaStorage
from animus_bootstrap.personas.voice import (
    VOICE_PRESETS,
    VoiceConfig,
    build_voice_prompt,
    get_time_shift_tone,
)

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_msg(
    text: str = "hello",
    channel: str = "webchat",
    sender: str = "user1",
) -> GatewayMessage:
    """Create a test GatewayMessage."""
    return create_message(
        channel=channel,
        sender_id=sender,
        sender_name=sender,
        text=text,
    )


def _make_persona(
    *,
    name: str = "TestBot",
    persona_id: str = "p-1",
    active: bool = True,
    is_default: bool = False,
    knowledge_domains: list[str] | None = None,
    excluded_topics: list[str] | None = None,
    channel_bindings: dict[str, bool] | None = None,
    voice: VoiceConfig | None = None,
    system_prompt: str = "You are TestBot.",
) -> PersonaProfile:
    return PersonaProfile(
        id=persona_id,
        name=name,
        active=active,
        is_default=is_default,
        knowledge_domains=knowledge_domains or [],
        excluded_topics=excluded_topics or [],
        channel_bindings=channel_bindings or {},
        voice=voice or VoiceConfig(),
        system_prompt=system_prompt,
    )


# ================================================================== #
# TestVoiceConfig
# ================================================================== #


class TestVoiceConfig:
    """VoiceConfig dataclass tests."""

    def test_defaults(self) -> None:
        v = VoiceConfig()
        assert v.tone == "balanced"
        assert v.max_response_length == "medium"
        assert v.emoji_policy == "minimal"
        assert v.language == "en"
        assert v.custom_instructions == ""
        assert v.time_shifts == {}

    def test_custom_values(self) -> None:
        v = VoiceConfig(
            tone="formal",
            max_response_length="brief",
            emoji_policy="none",
            language="fr",
            custom_instructions="Speak French.",
        )
        assert v.tone == "formal"
        assert v.max_response_length == "brief"
        assert v.emoji_policy == "none"
        assert v.language == "fr"
        assert v.custom_instructions == "Speak French."

    def test_time_shifts_dict(self) -> None:
        v = VoiceConfig(time_shifts={"morning": "energetic", "night": "calm"})
        assert v.time_shifts["morning"] == "energetic"
        assert v.time_shifts["night"] == "calm"

    def test_independent_instances(self) -> None:
        """Each VoiceConfig gets its own mutable fields."""
        a = VoiceConfig()
        b = VoiceConfig()
        a.time_shifts["morning"] = "loud"
        assert "morning" not in b.time_shifts


# ================================================================== #
# TestVoicePresets
# ================================================================== #


class TestVoicePresets:
    """VOICE_PRESETS dict tests."""

    def test_all_six_presets_exist(self) -> None:
        expected = {"formal", "casual", "technical", "mentor", "creative", "balanced"}
        assert set(VOICE_PRESETS.keys()) == expected

    def test_formal_tone(self) -> None:
        assert VOICE_PRESETS["formal"].tone == "formal"
        assert VOICE_PRESETS["formal"].emoji_policy == "none"

    def test_casual_tone(self) -> None:
        assert VOICE_PRESETS["casual"].tone == "casual"
        assert VOICE_PRESETS["casual"].emoji_policy == "minimal"

    def test_technical_tone(self) -> None:
        assert VOICE_PRESETS["technical"].tone == "technical"
        assert VOICE_PRESETS["technical"].max_response_length == "detailed"
        assert VOICE_PRESETS["technical"].emoji_policy == "none"

    def test_mentor_tone(self) -> None:
        assert VOICE_PRESETS["mentor"].tone == "mentor"
        assert "Socratic" in VOICE_PRESETS["mentor"].custom_instructions

    def test_creative_tone(self) -> None:
        assert VOICE_PRESETS["creative"].tone == "creative"
        assert VOICE_PRESETS["creative"].emoji_policy == "expressive"

    def test_balanced_is_default(self) -> None:
        v = VOICE_PRESETS["balanced"]
        assert v.tone == "balanced"
        assert v.custom_instructions == ""


# ================================================================== #
# TestBuildVoicePrompt
# ================================================================== #


class TestBuildVoicePrompt:
    """build_voice_prompt function tests."""

    def test_formal_tone_instruction(self) -> None:
        result = build_voice_prompt(VoiceConfig(tone="formal"))
        assert "professional and formal" in result

    def test_casual_tone_instruction(self) -> None:
        result = build_voice_prompt(VoiceConfig(tone="casual"))
        assert "casual, friendly" in result

    def test_technical_tone_instruction(self) -> None:
        result = build_voice_prompt(VoiceConfig(tone="technical"))
        assert "technically precise" in result

    def test_mentor_tone_instruction(self) -> None:
        result = build_voice_prompt(VoiceConfig(tone="mentor"))
        assert "mentoring" in result

    def test_creative_tone_instruction(self) -> None:
        result = build_voice_prompt(VoiceConfig(tone="creative"))
        assert "creative" in result.lower()

    def test_balanced_tone_instruction(self) -> None:
        result = build_voice_prompt(VoiceConfig(tone="balanced"))
        assert "balanced" in result.lower()

    def test_brief_length(self) -> None:
        result = build_voice_prompt(VoiceConfig(max_response_length="brief"))
        assert "concise" in result.lower()

    def test_medium_length(self) -> None:
        result = build_voice_prompt(VoiceConfig(max_response_length="medium"))
        assert "moderately" in result.lower()

    def test_detailed_length(self) -> None:
        result = build_voice_prompt(VoiceConfig(max_response_length="detailed"))
        assert "thorough" in result.lower()

    def test_emoji_none(self) -> None:
        result = build_voice_prompt(VoiceConfig(emoji_policy="none"))
        assert "Do not use emojis" in result

    def test_emoji_minimal(self) -> None:
        result = build_voice_prompt(VoiceConfig(emoji_policy="minimal"))
        assert "sparingly" in result

    def test_emoji_expressive(self) -> None:
        result = build_voice_prompt(VoiceConfig(emoji_policy="expressive"))
        assert "freely" in result

    def test_custom_instructions_included(self) -> None:
        result = build_voice_prompt(VoiceConfig(custom_instructions="Always cite sources."))
        assert "Always cite sources." in result

    def test_combined_prompt(self) -> None:
        v = VoiceConfig(
            tone="formal",
            max_response_length="brief",
            emoji_policy="none",
            custom_instructions="End with a summary.",
        )
        result = build_voice_prompt(v)
        assert "professional and formal" in result
        assert "concise" in result.lower()
        assert "Do not use emojis" in result
        assert "End with a summary." in result

    def test_unknown_tone_excluded(self) -> None:
        result = build_voice_prompt(VoiceConfig(tone="alien"))
        # Unknown tone should not crash, just omit tone instruction
        assert "alien" not in result

    def test_empty_custom_instructions_excluded(self) -> None:
        result = build_voice_prompt(VoiceConfig(custom_instructions=""))
        lines = [line for line in result.split("\n") if line.strip()]
        # Should have tone, length, emoji but NOT custom
        assert len(lines) == 3


# ================================================================== #
# TestGetTimeShiftTone
# ================================================================== #


class TestGetTimeShiftTone:
    """get_time_shift_tone function tests."""

    def test_no_shifts_returns_none(self) -> None:
        v = VoiceConfig()
        assert get_time_shift_tone(v, 10) is None

    def test_morning_range(self) -> None:
        v = VoiceConfig(time_shifts={"morning": "energetic"})
        assert get_time_shift_tone(v, 6) == "energetic"
        assert get_time_shift_tone(v, 9) == "energetic"
        assert get_time_shift_tone(v, 11) == "energetic"

    def test_afternoon_range(self) -> None:
        v = VoiceConfig(time_shifts={"afternoon": "focused"})
        assert get_time_shift_tone(v, 12) == "focused"
        assert get_time_shift_tone(v, 14) == "focused"
        assert get_time_shift_tone(v, 16) == "focused"

    def test_evening_range(self) -> None:
        v = VoiceConfig(time_shifts={"evening": "relaxed"})
        assert get_time_shift_tone(v, 17) == "relaxed"
        assert get_time_shift_tone(v, 19) == "relaxed"
        assert get_time_shift_tone(v, 21) == "relaxed"

    def test_night_range(self) -> None:
        v = VoiceConfig(time_shifts={"night": "calm"})
        assert get_time_shift_tone(v, 22) == "calm"
        assert get_time_shift_tone(v, 0) == "calm"
        assert get_time_shift_tone(v, 5) == "calm"

    def test_boundary_morning_to_afternoon(self) -> None:
        v = VoiceConfig(time_shifts={"morning": "m", "afternoon": "a"})
        assert get_time_shift_tone(v, 11) == "m"
        assert get_time_shift_tone(v, 12) == "a"

    def test_boundary_afternoon_to_evening(self) -> None:
        v = VoiceConfig(time_shifts={"afternoon": "a", "evening": "e"})
        assert get_time_shift_tone(v, 16) == "a"
        assert get_time_shift_tone(v, 17) == "e"

    def test_boundary_evening_to_night(self) -> None:
        v = VoiceConfig(time_shifts={"evening": "e", "night": "n"})
        assert get_time_shift_tone(v, 21) == "e"
        assert get_time_shift_tone(v, 22) == "n"

    def test_missing_period_returns_none(self) -> None:
        v = VoiceConfig(time_shifts={"morning": "energetic"})
        assert get_time_shift_tone(v, 14) is None  # afternoon not configured


# ================================================================== #
# TestPersonaProfile
# ================================================================== #


class TestPersonaProfile:
    """PersonaProfile dataclass tests."""

    def test_defaults(self) -> None:
        p = PersonaProfile()
        assert p.name == "Animus"
        assert p.description == "Default personal AI assistant"
        assert p.system_prompt == "You are Animus, a personal AI assistant."
        assert isinstance(p.voice, VoiceConfig)
        assert p.knowledge_domains == []
        assert p.excluded_topics == []
        assert p.channel_bindings == {}
        assert p.active is True
        assert p.is_default is False

    def test_generated_id(self) -> None:
        p1 = PersonaProfile()
        p2 = PersonaProfile()
        assert p1.id != p2.id
        assert len(p1.id) == 36  # UUID4 format

    def test_custom_values(self) -> None:
        v = VoiceConfig(tone="formal")
        p = PersonaProfile(
            id="custom-id",
            name="Sage",
            description="A wise assistant",
            system_prompt="You are Sage.",
            voice=v,
            knowledge_domains=["coding"],
            excluded_topics=["health"],
            channel_bindings={"slack": True},
            active=False,
            is_default=True,
        )
        assert p.id == "custom-id"
        assert p.name == "Sage"
        assert p.voice.tone == "formal"
        assert p.knowledge_domains == ["coding"]
        assert p.excluded_topics == ["health"]
        assert p.channel_bindings == {"slack": True}
        assert p.active is False
        assert p.is_default is True


# ================================================================== #
# TestPersonaEngine
# ================================================================== #


class TestPersonaEngine:
    """PersonaEngine registry and routing tests."""

    def test_register_and_get(self) -> None:
        engine = PersonaEngine()
        p = _make_persona()
        engine.register_persona(p)
        assert engine.get_persona("p-1") is p

    def test_register_default(self) -> None:
        engine = PersonaEngine()
        p = _make_persona(is_default=True)
        engine.register_persona(p)
        assert engine.get_default() is p

    def test_unregister(self) -> None:
        engine = PersonaEngine()
        p = _make_persona()
        engine.register_persona(p)
        engine.unregister_persona("p-1")
        assert engine.get_persona("p-1") is None

    def test_unregister_default_clears(self) -> None:
        engine = PersonaEngine()
        p = _make_persona(is_default=True)
        engine.register_persona(p)
        engine.unregister_persona("p-1")
        assert engine.get_default() is None

    def test_unregister_nonexistent_no_error(self) -> None:
        engine = PersonaEngine()
        engine.unregister_persona("nope")  # Should not raise

    def test_list_personas(self) -> None:
        engine = PersonaEngine()
        p1 = _make_persona(persona_id="p-1", name="A")
        p2 = _make_persona(persona_id="p-2", name="B")
        engine.register_persona(p1)
        engine.register_persona(p2)
        result = engine.list_personas()
        assert len(result) == 2

    def test_set_default(self) -> None:
        engine = PersonaEngine()
        p1 = _make_persona(persona_id="p-1", is_default=True)
        p2 = _make_persona(persona_id="p-2")
        engine.register_persona(p1)
        engine.register_persona(p2)
        engine.set_default("p-2")
        assert engine.get_default() is p2
        assert p1.is_default is False
        assert p2.is_default is True

    def test_set_default_not_found(self) -> None:
        engine = PersonaEngine()
        with pytest.raises(ValueError, match="not found"):
            engine.set_default("nope")

    def test_get_default_first_when_none_set(self) -> None:
        engine = PersonaEngine()
        p = _make_persona()
        engine.register_persona(p)
        assert engine.get_default() is p

    def test_get_default_empty_engine(self) -> None:
        engine = PersonaEngine()
        assert engine.get_default() is None

    def test_persona_count(self) -> None:
        engine = PersonaEngine()
        assert engine.persona_count == 0
        engine.register_persona(_make_persona(persona_id="p-1"))
        assert engine.persona_count == 1
        engine.register_persona(_make_persona(persona_id="p-2"))
        assert engine.persona_count == 2

    def test_get_persona_for_message_explicit_command(self) -> None:
        engine = PersonaEngine()
        p = _make_persona(name="Sage")
        engine.register_persona(p)
        msg = _make_msg(text="/persona Sage")
        result = engine.get_persona_for_message(msg)
        assert result is p

    def test_get_persona_for_message_command_case_insensitive(self) -> None:
        engine = PersonaEngine()
        p = _make_persona(name="Sage")
        engine.register_persona(p)
        msg = _make_msg(text="/persona sage")
        result = engine.get_persona_for_message(msg)
        assert result is p

    def test_get_persona_for_message_command_inactive_skipped(self) -> None:
        engine = PersonaEngine()
        p_inactive = _make_persona(persona_id="p-1", name="Sage", active=False)
        p_default = _make_persona(persona_id="p-2", name="Fallback", is_default=True)
        engine.register_persona(p_inactive)
        engine.register_persona(p_default)
        msg = _make_msg(text="/persona Sage")
        # Inactive persona should not be matched by command; falls back to default
        result = engine.get_persona_for_message(msg)
        assert result is p_default

    def test_get_persona_for_message_channel_binding(self) -> None:
        engine = PersonaEngine()
        p = _make_persona(channel_bindings={"slack": True})
        engine.register_persona(p)
        msg = _make_msg(channel="slack")
        result = engine.get_persona_for_message(msg)
        assert result is p

    def test_get_persona_for_message_channel_binding_false(self) -> None:
        engine = PersonaEngine()
        p = _make_persona(channel_bindings={"slack": False})
        engine.register_persona(p)
        msg = _make_msg(channel="slack")
        result = engine.get_persona_for_message(msg)
        # Binding is False, so should fallback to default
        assert result is not None  # Falls back to default (first persona)

    def test_get_persona_for_message_fallback_default(self) -> None:
        engine = PersonaEngine()
        p = _make_persona(is_default=True)
        engine.register_persona(p)
        msg = _make_msg(text="hi")
        result = engine.get_persona_for_message(msg)
        assert result is p

    def test_get_persona_for_message_inactive_skipped_in_channel(self) -> None:
        engine = PersonaEngine()
        p1 = _make_persona(
            persona_id="p-1",
            channel_bindings={"slack": True},
            active=False,
        )
        p2 = _make_persona(persona_id="p-2", is_default=True)
        engine.register_persona(p1)
        engine.register_persona(p2)
        msg = _make_msg(channel="slack")
        result = engine.get_persona_for_message(msg)
        assert result is p2  # Falls back to default

    def test_get_persona_for_message_empty_engine(self) -> None:
        engine = PersonaEngine()
        msg = _make_msg()
        assert engine.get_persona_for_message(msg) is None

    def test_set_default_clears_old_default(self) -> None:
        engine = PersonaEngine()
        p1 = _make_persona(persona_id="p-1", is_default=True)
        p2 = _make_persona(persona_id="p-2")
        engine.register_persona(p1)
        engine.register_persona(p2)
        assert p1.is_default is True
        engine.set_default("p-2")
        assert p1.is_default is False
        assert p2.is_default is True


# ================================================================== #
# TestKnowledgeDomainRouter
# ================================================================== #


class TestKnowledgeDomainRouter:
    """KnowledgeDomainRouter tests."""

    def test_classify_coding(self) -> None:
        router = KnowledgeDomainRouter()
        topics = router.classify_topic("Help me debug this Python function")
        assert "coding" in topics

    def test_classify_writing(self) -> None:
        router = KnowledgeDomainRouter()
        topics = router.classify_topic("Write me an essay about history")
        assert "writing" in topics

    def test_classify_general_fallback(self) -> None:
        router = KnowledgeDomainRouter()
        topics = router.classify_topic("What is the meaning of life?")
        assert topics == ["general"]

    def test_classify_multiple_domains(self) -> None:
        router = KnowledgeDomainRouter()
        topics = router.classify_topic("Deploy this Python code to Docker")
        assert "coding" in topics
        assert "devops" in topics

    def test_classify_case_insensitive(self) -> None:
        router = KnowledgeDomainRouter()
        topics = router.classify_topic("PYTHON CODE BUG")
        assert "coding" in topics

    def test_classify_architecture(self) -> None:
        router = KnowledgeDomainRouter()
        topics = router.classify_topic("Design a scalable microservice architecture")
        assert "architecture" in topics

    def test_classify_devops(self) -> None:
        router = KnowledgeDomainRouter()
        topics = router.classify_topic("Set up a CI/CD pipeline with Docker")
        assert "devops" in topics

    def test_classify_brainstorming(self) -> None:
        router = KnowledgeDomainRouter()
        topics = router.classify_topic("Brainstorm innovative strategies")
        assert "brainstorming" in topics

    def test_classify_health(self) -> None:
        router = KnowledgeDomainRouter()
        topics = router.classify_topic("What are the symptoms of sleep deprivation?")
        assert "health" in topics

    def test_classify_finance(self) -> None:
        router = KnowledgeDomainRouter()
        topics = router.classify_topic("How should I invest my savings?")
        assert "finance" in topics

    def test_classify_science(self) -> None:
        router = KnowledgeDomainRouter()
        topics = router.classify_topic("Design an experiment to test my hypothesis")
        assert "science" in topics

    def test_classify_art(self) -> None:
        router = KnowledgeDomainRouter()
        topics = router.classify_topic("Draw me a visual illustration")
        assert "art" in topics

    def test_domain_keywords_has_general(self) -> None:
        assert "general" in _DOMAIN_KEYWORDS
        assert _DOMAIN_KEYWORDS["general"] == []

    def test_find_best_persona_matching_domain(self) -> None:
        router = KnowledgeDomainRouter()
        p = _make_persona(knowledge_domains=["coding", "devops"])
        result = router.find_best_persona(["coding"], [p])
        assert result is p

    def test_find_best_persona_multiple_matches(self) -> None:
        router = KnowledgeDomainRouter()
        p1 = _make_persona(persona_id="p-1", knowledge_domains=["coding"])
        p2 = _make_persona(persona_id="p-2", knowledge_domains=["coding", "devops"])
        result = router.find_best_persona(["coding", "devops"], [p1, p2])
        assert result is p2  # Higher score

    def test_find_best_persona_excluded_topic(self) -> None:
        router = KnowledgeDomainRouter()
        p = _make_persona(
            knowledge_domains=["coding", "health"],
            excluded_topics=["health"],
        )
        result = router.find_best_persona(["health"], [p])
        # excluded_topics causes the for-else break
        assert result is None

    def test_find_best_persona_no_match_returns_none(self) -> None:
        router = KnowledgeDomainRouter()
        p = _make_persona(knowledge_domains=["writing"])
        result = router.find_best_persona(["coding"], [p])
        assert result is None

    def test_find_best_persona_empty_domains(self) -> None:
        router = KnowledgeDomainRouter()
        p = _make_persona(knowledge_domains=[])
        result = router.find_best_persona(["coding"], [p])
        assert result is None

    def test_find_best_persona_inactive_skipped(self) -> None:
        router = KnowledgeDomainRouter()
        p = _make_persona(knowledge_domains=["coding"], active=False)
        result = router.find_best_persona(["coding"], [p])
        assert result is None

    def test_find_best_persona_empty_list(self) -> None:
        router = KnowledgeDomainRouter()
        result = router.find_best_persona(["coding"], [])
        assert result is None

    def test_is_topic_excluded_true(self) -> None:
        router = KnowledgeDomainRouter()
        p = _make_persona(excluded_topics=["health"])
        assert router.is_topic_excluded("health", p) is True

    def test_is_topic_excluded_false(self) -> None:
        router = KnowledgeDomainRouter()
        p = _make_persona(excluded_topics=["health"])
        assert router.is_topic_excluded("coding", p) is False


# ================================================================== #
# TestContextAdapter
# ================================================================== #


class TestContextAdapter:
    """ContextAdapter tests."""

    def test_adapt_prompt_includes_base_prompt(self) -> None:
        adapter = ContextAdapter()
        p = _make_persona(system_prompt="You are TestBot.")
        msg = _make_msg()
        result = adapter.adapt_prompt(p, msg)
        assert "You are TestBot." in result

    def test_adapt_prompt_includes_voice(self) -> None:
        adapter = ContextAdapter()
        v = VoiceConfig(tone="formal")
        p = _make_persona(voice=v)
        msg = _make_msg()
        result = adapter.adapt_prompt(p, msg)
        assert "professional and formal" in result

    def test_adapt_prompt_includes_time_shift(self) -> None:
        adapter = ContextAdapter()
        v = VoiceConfig(time_shifts={"morning": "energetic"})
        p = _make_persona(voice=v)
        msg = _make_msg()
        now = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
        result = adapter.adapt_prompt(p, msg, now=now)
        assert "energetic" in result

    def test_adapt_prompt_no_time_shift_when_not_configured(self) -> None:
        adapter = ContextAdapter()
        p = _make_persona()
        msg = _make_msg()
        now = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
        result = adapter.adapt_prompt(p, msg, now=now)
        assert "Current time tone" not in result

    def test_adapt_prompt_slack_channel_norm(self) -> None:
        adapter = ContextAdapter()
        p = _make_persona()
        msg = _make_msg(channel="slack")
        result = adapter.adapt_prompt(p, msg)
        assert "Slack" in result

    def test_adapt_prompt_discord_channel_norm(self) -> None:
        adapter = ContextAdapter()
        p = _make_persona()
        msg = _make_msg(channel="discord")
        result = adapter.adapt_prompt(p, msg)
        assert "Discord" in result

    def test_adapt_prompt_email_channel_norm(self) -> None:
        adapter = ContextAdapter()
        p = _make_persona()
        msg = _make_msg(channel="email")
        result = adapter.adapt_prompt(p, msg)
        assert "email" in result.lower()

    def test_adapt_prompt_webchat_no_norm(self) -> None:
        adapter = ContextAdapter()
        p = _make_persona()
        msg = _make_msg(channel="webchat")
        result = adapter.adapt_prompt(p, msg)
        # webchat has empty norm
        assert "webchat" not in result.lower()

    def test_adapt_prompt_unknown_channel_no_norm(self) -> None:
        adapter = ContextAdapter()
        p = _make_persona()
        msg = _make_msg(channel="unknown_channel")
        result = adapter.adapt_prompt(p, msg)
        assert "unknown_channel" not in result

    def test_adapt_prompt_long_conversation_hint(self) -> None:
        adapter = ContextAdapter()
        p = _make_persona()
        msg = _make_msg()
        history = [{"role": "user", "text": f"msg {i}"} for i in range(25)]
        result = adapter.adapt_prompt(p, msg, session_history=history)
        assert "long conversation" in result.lower()

    def test_adapt_prompt_medium_conversation_hint(self) -> None:
        adapter = ContextAdapter()
        p = _make_persona()
        msg = _make_msg()
        history = [{"role": "user", "text": f"msg {i}"} for i in range(15)]
        result = adapter.adapt_prompt(p, msg, session_history=history)
        assert "progressing" in result.lower()

    def test_adapt_prompt_short_conversation_no_hint(self) -> None:
        adapter = ContextAdapter()
        p = _make_persona()
        msg = _make_msg()
        history = [{"role": "user", "text": f"msg {i}"} for i in range(5)]
        result = adapter.adapt_prompt(p, msg, session_history=history)
        assert "long conversation" not in result.lower()
        assert "progressing" not in result.lower()

    def test_adapt_prompt_no_history_no_hint(self) -> None:
        adapter = ContextAdapter()
        p = _make_persona()
        msg = _make_msg()
        result = adapter.adapt_prompt(p, msg, session_history=None)
        assert "conversation" not in result.lower()

    def test_adapt_prompt_knowledge_domains(self) -> None:
        adapter = ContextAdapter()
        p = _make_persona(knowledge_domains=["coding", "devops"])
        msg = _make_msg()
        result = adapter.adapt_prompt(p, msg)
        assert "coding, devops" in result

    def test_adapt_prompt_excluded_topics(self) -> None:
        adapter = ContextAdapter()
        p = _make_persona(excluded_topics=["health", "finance"])
        msg = _make_msg()
        result = adapter.adapt_prompt(p, msg)
        assert "health, finance" in result
        assert "decline or defer" in result.lower()

    def test_adapt_prompt_empty_system_prompt(self) -> None:
        adapter = ContextAdapter()
        p = _make_persona(system_prompt="")
        msg = _make_msg()
        result = adapter.adapt_prompt(p, msg)
        # Should still include voice etc.
        assert len(result) > 0

    def test_channel_norms_keys_complete(self) -> None:
        expected = {
            "slack",
            "discord",
            "email",
            "telegram",
            "matrix",
            "webchat",
            "whatsapp",
            "signal",
        }
        assert set(_CHANNEL_NORMS.keys()) == expected

    def test_adapt_prompt_telegram_norm(self) -> None:
        adapter = ContextAdapter()
        p = _make_persona()
        msg = _make_msg(channel="telegram")
        result = adapter.adapt_prompt(p, msg)
        assert "Telegram" in result

    def test_adapt_prompt_matrix_norm(self) -> None:
        adapter = ContextAdapter()
        p = _make_persona()
        msg = _make_msg(channel="matrix")
        result = adapter.adapt_prompt(p, msg)
        assert "Matrix" in result

    def test_adapt_prompt_default_now(self) -> None:
        """adapt_prompt uses current time if now is not provided."""
        adapter = ContextAdapter()
        v = VoiceConfig(
            time_shifts={
                "morning": "m",
                "afternoon": "a",
                "evening": "e",
                "night": "n",
            }
        )
        p = _make_persona(voice=v)
        msg = _make_msg()
        result = adapter.adapt_prompt(p, msg)
        # Should include some time tone since all periods are configured
        assert "Current time tone" in result


# ================================================================== #
# TestPersonaStorage
# ================================================================== #


class TestPersonaStorage:
    """PersonaStorage SQLite tests."""

    def test_save_and_load(self, tmp_path: pytest.TempPathFactory) -> None:
        db = tmp_path / "personas.db"
        store = PersonaStorage(db)
        p = _make_persona()
        store.save(p)
        loaded = store.load("p-1")
        assert loaded is not None
        assert loaded.id == "p-1"
        assert loaded.name == "TestBot"
        store.close()

    def test_load_nonexistent(self, tmp_path: pytest.TempPathFactory) -> None:
        db = tmp_path / "personas.db"
        store = PersonaStorage(db)
        assert store.load("nope") is None
        store.close()

    def test_load_all(self, tmp_path: pytest.TempPathFactory) -> None:
        db = tmp_path / "personas.db"
        store = PersonaStorage(db)
        store.save(_make_persona(persona_id="p-1", name="A"))
        store.save(_make_persona(persona_id="p-2", name="B"))
        loaded = store.load_all()
        assert len(loaded) == 2
        store.close()

    def test_load_all_empty(self, tmp_path: pytest.TempPathFactory) -> None:
        db = tmp_path / "personas.db"
        store = PersonaStorage(db)
        assert store.load_all() == []
        store.close()

    def test_delete(self, tmp_path: pytest.TempPathFactory) -> None:
        db = tmp_path / "personas.db"
        store = PersonaStorage(db)
        store.save(_make_persona())
        deleted = store.delete("p-1")
        assert deleted is True
        assert store.load("p-1") is None
        store.close()

    def test_delete_nonexistent(self, tmp_path: pytest.TempPathFactory) -> None:
        db = tmp_path / "personas.db"
        store = PersonaStorage(db)
        deleted = store.delete("nope")
        assert deleted is False
        store.close()

    def test_save_update_existing(self, tmp_path: pytest.TempPathFactory) -> None:
        db = tmp_path / "personas.db"
        store = PersonaStorage(db)
        p = _make_persona(name="Original")
        store.save(p)
        p.name = "Updated"
        store.save(p)
        loaded = store.load("p-1")
        assert loaded is not None
        assert loaded.name == "Updated"
        store.close()

    def test_wal_mode(self, tmp_path: pytest.TempPathFactory) -> None:
        db = tmp_path / "personas.db"
        store = PersonaStorage(db)
        cur = store._conn.execute("PRAGMA journal_mode")
        mode = cur.fetchone()[0]
        assert mode == "wal"
        store.close()

    def test_roundtrip_voice_config(self, tmp_path: pytest.TempPathFactory) -> None:
        db = tmp_path / "personas.db"
        store = PersonaStorage(db)
        v = VoiceConfig(
            tone="formal",
            max_response_length="detailed",
            emoji_policy="none",
            language="fr",
            custom_instructions="Be precise.",
            time_shifts={"morning": "energetic", "night": "calm"},
        )
        p = _make_persona(voice=v)
        store.save(p)
        loaded = store.load("p-1")
        assert loaded is not None
        assert loaded.voice.tone == "formal"
        assert loaded.voice.max_response_length == "detailed"
        assert loaded.voice.emoji_policy == "none"
        assert loaded.voice.language == "fr"
        assert loaded.voice.custom_instructions == "Be precise."
        assert loaded.voice.time_shifts == {"morning": "energetic", "night": "calm"}
        store.close()

    def test_roundtrip_knowledge_domains(self, tmp_path: pytest.TempPathFactory) -> None:
        db = tmp_path / "personas.db"
        store = PersonaStorage(db)
        p = _make_persona(knowledge_domains=["coding", "devops", "science"])
        store.save(p)
        loaded = store.load("p-1")
        assert loaded is not None
        assert loaded.knowledge_domains == ["coding", "devops", "science"]
        store.close()

    def test_roundtrip_excluded_topics(self, tmp_path: pytest.TempPathFactory) -> None:
        db = tmp_path / "personas.db"
        store = PersonaStorage(db)
        p = _make_persona(excluded_topics=["health", "finance"])
        store.save(p)
        loaded = store.load("p-1")
        assert loaded is not None
        assert loaded.excluded_topics == ["health", "finance"]
        store.close()

    def test_roundtrip_channel_bindings(self, tmp_path: pytest.TempPathFactory) -> None:
        db = tmp_path / "personas.db"
        store = PersonaStorage(db)
        p = _make_persona(channel_bindings={"slack": True, "discord": False})
        store.save(p)
        loaded = store.load("p-1")
        assert loaded is not None
        assert loaded.channel_bindings == {"slack": True, "discord": False}
        store.close()

    def test_roundtrip_active_false(self, tmp_path: pytest.TempPathFactory) -> None:
        db = tmp_path / "personas.db"
        store = PersonaStorage(db)
        p = _make_persona(active=False)
        store.save(p)
        loaded = store.load("p-1")
        assert loaded is not None
        assert loaded.active is False
        store.close()

    def test_roundtrip_is_default(self, tmp_path: pytest.TempPathFactory) -> None:
        db = tmp_path / "personas.db"
        store = PersonaStorage(db)
        p = _make_persona(is_default=True)
        store.save(p)
        loaded = store.load("p-1")
        assert loaded is not None
        assert loaded.is_default is True
        store.close()

    def test_load_all_ordered_by_name(self, tmp_path: pytest.TempPathFactory) -> None:
        db = tmp_path / "personas.db"
        store = PersonaStorage(db)
        store.save(_make_persona(persona_id="p-2", name="Zeta"))
        store.save(_make_persona(persona_id="p-1", name="Alpha"))
        loaded = store.load_all()
        assert loaded[0].name == "Alpha"
        assert loaded[1].name == "Zeta"
        store.close()

    def test_voice_to_dict_static(self) -> None:
        v = VoiceConfig(tone="casual", language="de")
        d = PersonaStorage._voice_to_dict(v)
        assert d["tone"] == "casual"
        assert d["language"] == "de"

    def test_dict_to_voice_static(self) -> None:
        d = {
            "tone": "mentor",
            "max_response_length": "brief",
            "emoji_policy": "expressive",
            "language": "en",
            "custom_instructions": "Teach.",
            "time_shifts": {},
        }
        v = PersonaStorage._dict_to_voice(d)
        assert v.tone == "mentor"
        assert v.max_response_length == "brief"
        assert v.custom_instructions == "Teach."
