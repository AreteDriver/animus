"""Tests for RegisterTranslator.translate_response().

Covers: neutral passthrough, short text skip, code-like skip,
LLM-based translation, explicit target override, error handling.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from animus.register import Register, RegisterTranslator


class TestTranslateResponse:
    """RegisterTranslator.translate_response() behaviour."""

    def test_neutral_returns_unchanged(self) -> None:
        """NEUTRAL target returns text as-is without calling cognitive."""
        translator = RegisterTranslator(default_register=Register.NEUTRAL)
        cognitive = MagicMock()

        result = translator.translate_response(
            "This is a long enough sentence for testing purposes.",
            cognitive,
        )

        assert result == "This is a long enough sentence for testing purposes."
        cognitive.think.assert_not_called()

    def test_short_text_returns_unchanged(self) -> None:
        """Text shorter than 20 chars is returned unchanged."""
        translator = RegisterTranslator(default_register=Register.FORMAL)
        cognitive = MagicMock()

        result = translator.translate_response("Hello there.", cognitive)

        assert result == "Hello there."
        cognitive.think.assert_not_called()

    def test_code_like_text_returns_unchanged(self) -> None:
        """Code-like text (starting with code patterns) is skipped."""
        translator = RegisterTranslator(default_register=Register.FORMAL)
        cognitive = MagicMock()

        code_texts = [
            "```python\ndef foo():\n    pass\n```",
            "def my_function(arg1, arg2):",
            "class MyClass(Base):",
            "import os\nimport sys",
            "from pathlib import Path",
            '{"key": "value", "nested": true}',
            "[1, 2, 3, 4, 5, 6, 7, 8, 9]",
        ]

        for text in code_texts:
            result = translator.translate_response(text, cognitive)
            assert result == text, f"Code-like text should be unchanged: {text[:30]}"

        cognitive.think.assert_not_called()

    def test_calls_cognitive_for_formal(self) -> None:
        """Translates to FORMAL register via cognitive.think()."""
        translator = RegisterTranslator(default_register=Register.FORMAL)
        cognitive = MagicMock()
        cognitive.think.return_value = "I would be delighted to assist you with that matter."

        result = translator.translate_response(
            "Sure thing! I can help with that.",
            cognitive,
        )

        assert result == "I would be delighted to assist you with that matter."
        cognitive.think.assert_called_once()
        call_args = cognitive.think.call_args[0][0]
        assert "formal" in call_args.lower()

    def test_calls_cognitive_for_casual(self) -> None:
        """Translates to CASUAL register via cognitive.think()."""
        translator = RegisterTranslator(default_register=Register.CASUAL)
        cognitive = MagicMock()
        cognitive.think.return_value = "Yeah totally, I got you!"

        result = translator.translate_response(
            "I would be happy to assist you with that inquiry.",
            cognitive,
        )

        assert result == "Yeah totally, I got you!"

    def test_calls_cognitive_for_technical(self) -> None:
        """Translates to TECHNICAL register via cognitive.think()."""
        translator = RegisterTranslator(default_register=Register.TECHNICAL)
        cognitive = MagicMock()
        cognitive.think.return_value = "The API endpoint returns a 200 status code."

        result = translator.translate_response(
            "The thing works and sends back a good response.",
            cognitive,
        )

        assert result == "The API endpoint returns a 200 status code."

    def test_explicit_target_overrides_active(self) -> None:
        """Explicit target parameter overrides the active register."""
        translator = RegisterTranslator(default_register=Register.NEUTRAL)
        cognitive = MagicMock()
        cognitive.think.return_value = "Translated text here."

        result = translator.translate_response(
            "This is some text that needs translating.",
            cognitive,
            target=Register.FORMAL,
        )

        assert result == "Translated text here."
        call_args = cognitive.think.call_args[0][0]
        assert "formal" in call_args.lower()

    def test_cognitive_error_returns_original(self) -> None:
        """If cognitive.think() raises, return original text."""
        translator = RegisterTranslator(default_register=Register.FORMAL)
        cognitive = MagicMock()
        cognitive.think.side_effect = RuntimeError("LLM unavailable")

        result = translator.translate_response(
            "This is some text to translate now.",
            cognitive,
        )

        assert result == "This is some text to translate now."

    def test_cognitive_returns_empty_uses_original(self) -> None:
        """If cognitive.think() returns empty string, use original."""
        translator = RegisterTranslator(default_register=Register.FORMAL)
        cognitive = MagicMock()
        cognitive.think.return_value = ""

        result = translator.translate_response(
            "This is some text to translate now.",
            cognitive,
        )

        assert result == "This is some text to translate now."

    def test_strips_whitespace_from_result(self) -> None:
        """Result from cognitive.think() is stripped of whitespace."""
        translator = RegisterTranslator(default_register=Register.FORMAL)
        cognitive = MagicMock()
        cognitive.think.return_value = "  Translated with spaces.  \n"

        result = translator.translate_response(
            "This is some text to translate now.",
            cognitive,
        )

        assert result == "Translated with spaces."
