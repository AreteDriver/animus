"""Tests for the redaction module.

Covers:
- Universal pattern matching (API keys, tokens, SSN)
- Personal pattern matching via env vars
- Empty input handling
- include_personal toggle
- has_secrets() convenience function
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from animus_kernel.memory.redaction import has_secrets, redact, RedactionHit


class TestRedactUniversal:
    def test_openai_api_key(self):
        key = "sk-" + "a" * 48  # 32+ chars after sk-
        text = f"key is {key}"
        result, hits = redact(text)
        assert "[REDACTED:" in result
        assert len(hits) >= 1
        assert all(isinstance(h, RedactionHit) for h in hits)

    def test_anthropic_key(self):
        key = "sk-ant-api03-" + "x" * 20  # 20+ chars after sk-ant-
        text = key
        result, hits = redact(text)
        assert "[REDACTED:" in result
        assert len(hits) >= 1

    def test_github_token(self):
        key = "ghp_" + "x" * 36  # 20+ chars after ghp_
        text = key
        result, hits = redact(text)
        assert "[REDACTED:" in result
        assert len(hits) >= 1

    def test_ssn(self):
        text = "SSN: 123-45-6789"
        result, hits = redact(text)
        assert "[REDACTED:ssn]" in result
        assert any(h.type == "ssn" for h in hits)

    def test_multiple_secrets(self):
        key1 = "sk-" + "a" * 48
        key2 = "ghp_" + "b" * 36
        text = f"{key1} and {key2}"
        result, hits = redact(text)
        assert len(hits) == 2

    def test_no_secrets(self):
        text = "just some normal text without keys"
        result, hits = redact(text)
        assert result == text
        assert hits == []

    def test_empty_string(self):
        result, hits = redact("")
        assert result == ""
        assert hits == []

    def test_spans_in_original_positions(self):
        key = "sk-" + "a" * 48
        text = f"prefix {key} suffix"
        result, hits = redact(text)
        hit = hits[0]
        # Spans refer to original text positions
        assert text[hit.start:hit.end] == key

    def test_overlapping_patterns_merged(self):
        # sk-ant- prefix might match both openai (negative lookahead prevents) and anthropic
        key = "sk-ant-api03-" + "x" * 20
        text = key
        result, hits = redact(text)
        # Should not crash; overlapping spans are merged
        assert len(hits) >= 1
        # Verify the merged span covers the whole key
        first = hits[0]
        assert first.end - first.start >= len(key)


class TestRedactPersonal:
    def test_personal_email_via_env(self):
        with patch.dict(os.environ, {"ANIMUS_REDACT_EMAILS": "alice@example.com"}):
            text = "Contact alice@example.com for info"
            result, hits = redact(text)
            assert any(h.type == "personal_email" for h in hits)
            assert "alice@example.com" not in result

    def test_personal_phone_via_env(self):
        with patch.dict(os.environ, {"ANIMUS_REDACT_PHONES": "555-1234"}):
            text = "Call 555-1234"
            result, hits = redact(text)
            assert any(h.type == "personal_phone" for h in hits)

    def test_sensitive_path_via_env(self):
        with patch.dict(os.environ, {"ANIMUS_REDACT_PATHS": "/home/alice/.ssh"}):
            text = "Key at /home/alice/.ssh/id_rsa"
            result, hits = redact(text)
            assert any(h.type == "sensitive_path" for h in hits)
            assert "/home/alice/.ssh/id_rsa" not in result

    def test_include_personal_false(self):
        with patch.dict(os.environ, {"ANIMUS_REDACT_EMAILS": "alice@example.com"}):
            text = "Contact alice@example.com"
            result, hits = redact(text, include_personal=False)
            assert not any(h.type == "personal_email" for h in hits)


class TestHasSecrets:
    def test_true_when_secret_present(self):
        assert has_secrets("sk-" + "a" * 48) is True

    def test_false_when_clean(self):
        assert has_secrets("hello world") is False

    def test_empty_is_false(self):
        assert has_secrets("") is False

    def test_include_personal_respected(self):
        with patch.dict(os.environ, {"ANIMUS_REDACT_EMAILS": "a@b.com"}):
            assert has_secrets("a@b.com", include_personal=False) is False
            assert has_secrets("a@b.com", include_personal=True) is True
