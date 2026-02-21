"""Tests for the reflection proactive check."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from animus_bootstrap.identity.manager import IdentityFileManager
from animus_bootstrap.intelligence.feedback import FeedbackStore
from animus_bootstrap.intelligence.proactive.checks.reflection import (
    _parse_reflection_entries,
    _run_reflection,
    get_reflection_check,
    set_reflection_deps,
)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        # Restore a fresh event loop so downstream tests aren't affected
        asyncio.set_event_loop(asyncio.new_event_loop())


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    set_reflection_deps()


@pytest.fixture()
def identity_mgr(tmp_path):
    mgr = IdentityFileManager(tmp_path / "identity")
    mgr.generate_from_templates({"name": "TestUser", "timezone": "UTC"})
    return mgr


@pytest.fixture()
def feedback_store(tmp_path):
    s = FeedbackStore(tmp_path / "feedback.db")
    yield s
    s.close()


class TestReflectionCheck:
    def test_get_reflection_check(self):
        check = get_reflection_check()
        assert check.name == "reflection"
        assert check.enabled is True

    def test_skips_without_identity_manager(self):
        set_reflection_deps()
        result = _run(_run_reflection())
        assert result is None

    def test_skips_without_feedback_data(self, identity_mgr):
        feedback = MagicMock()
        feedback.get_stats.return_value = {"total": 0}
        set_reflection_deps(identity_manager=identity_mgr, feedback_store=feedback)
        result = _run(_run_reflection())
        assert result is None

    def test_fallback_writes_to_learned(self, identity_mgr, feedback_store):
        # Add some feedback
        feedback_store.record("hello", "hi", 1)
        feedback_store.record("bad response", "sorry", -1, comment="too verbose")
        feedback_store.record("another", "reply", 1)
        feedback_store.record("q4", "a4", 1)
        feedback_store.record("q5", "a5", -1)

        set_reflection_deps(
            identity_manager=identity_mgr,
            feedback_store=feedback_store,
        )

        result = _run(_run_reflection())
        assert result is not None
        assert "entries added" in result

        learned = identity_mgr.read("LEARNED.md")
        assert "Feedback" in learned or "Reflection" in learned

    def test_ai_reflection_writes_entries(self, identity_mgr, feedback_store):
        # Add feedback
        feedback_store.record("hello", "hi there", 1)
        feedback_store.record("bad q", "bad a", -1, comment="wrong")

        # Mock cognitive backend
        mock_backend = AsyncMock()
        mock_backend.generate_response.return_value = (
            "- User prefers concise answers\n"
            "- User dislikes verbose explanations\n"
            "- User responds well to bullet points\n"
        )

        mock_config = MagicMock()
        mock_config.identity.name = "TestUser"

        set_reflection_deps(
            identity_manager=identity_mgr,
            feedback_store=feedback_store,
            cognitive_backend=mock_backend,
            config=mock_config,
        )

        result = _run(_run_reflection())
        assert result is not None
        assert "3 new insights" in result

        learned = identity_mgr.read("LEARNED.md")
        assert "concise answers" in learned

    def test_ai_reflection_fallback_on_error(self, identity_mgr, feedback_store):
        feedback_store.record("q1", "a1", -1, comment="bad")
        feedback_store.record("q2", "a2", 1)
        feedback_store.record("q3", "a3", 1)
        feedback_store.record("q4", "a4", 1)
        feedback_store.record("q5", "a5", 1)

        mock_backend = AsyncMock()
        mock_backend.generate_response.side_effect = RuntimeError("API down")

        set_reflection_deps(
            identity_manager=identity_mgr,
            feedback_store=feedback_store,
            cognitive_backend=mock_backend,
        )

        # Should fall back to simple feedback-based entries
        result = _run(_run_reflection())
        assert result is not None


class TestParseReflectionEntries:
    def test_parses_bullet_points(self):
        text = "- First entry\n- Second entry\n* Third entry\n"
        entries = _parse_reflection_entries(text)
        assert len(entries) == 3
        assert entries[0] == "First entry"

    def test_ignores_non_bullets(self):
        text = "Some preamble\n- Real entry\nMore text\n"
        entries = _parse_reflection_entries(text)
        assert len(entries) == 1

    def test_caps_at_10(self):
        text = "\n".join(f"- Entry {i}" for i in range(20))
        entries = _parse_reflection_entries(text)
        assert len(entries) == 10

    def test_empty_input(self):
        assert _parse_reflection_entries("") == []

    def test_strips_whitespace(self):
        entries = _parse_reflection_entries("  -   padded entry  \n")
        assert entries == ["padded entry"]
