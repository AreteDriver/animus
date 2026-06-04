"""Tests for animus.memory.redaction."""

from __future__ import annotations

import pytest

from animus.memory.redaction import (
    REDACTED_FMT,
    RedactionHit,
    has_secrets,
    redact,
)


class TestUniversalPatterns:
    """Always-on patterns — API keys, tokens, private keys, SSN."""

    def test_anthropic_key_redacted(self):
        text = "key=sk-ant-api03-abcdefghijklmnopqrstuvwxyz123 end"
        redacted, hits = redact(text)
        assert "sk-ant-" not in redacted
        assert "abcdefghijklmnopqrstuvwxyz123" not in redacted
        assert any(h.type == "anthropic_key" for h in hits)

    def test_openai_key_redacted(self):
        text = "OPENAI_API_KEY=sk-proj_abcdefghijklmnopqrstuvwxyzABCDEFGH123"
        redacted, hits = redact(text)
        assert "sk-proj_" not in redacted
        # Specific (``openai_key``) or any of the broader label-based
        # patterns may win the merge depending on which span starts
        # earliest. The invariant is removal, not label identity.
        assert any(
            h.type
            in {
                "openai_key",
                "credential_label_separated",
                "credential_label_compound",
                "credential_qualified_label",
            }
            for h in hits
        )

    def test_openai_key_does_not_match_anthropic_prefix(self):
        text = "x=sk-ant-abc"
        # Too short and is anthropic-prefixed; should not match openai pattern,
        # and the anthropic_key pattern requires 20+ chars so this is unmatched.
        redacted, hits = redact(text)
        assert hits == []
        assert redacted == text

    def test_github_token_redacted(self):
        text = "GITHUB_TOKEN=ghp_abcdefghij1234567890ABCDEFGH"
        redacted, hits = redact(text)
        assert "ghp_" not in redacted
        # ``github_token`` is the specific tag, but ``credential_qualified_label``
        # may swallow the whole "GITHUB_TOKEN=..." span first via the merge.
        # Both are valid hits for the removal invariant.
        assert any(
            h.type
            in {
                "github_token",
                "credential_qualified_label",
                "credential_label_compound",
            }
            for h in hits
        )

    def test_github_fine_grained_pat_redacted(self):
        text = "tok=github_pat_11AAAAAA0_zzzzzzzzzzzzzzzzzzzzzzzz end"
        redacted, hits = redact(text)
        assert "github_pat_" not in redacted
        assert any(h.type == "github_pat" for h in hits)

    def test_aws_access_key_redacted(self):
        text = "aws_access_key_id=AKIAIOSFODNN7EXAMPLE in config"
        redacted, hits = redact(text)
        assert "AKIA" not in redacted
        # Specific (``aws_access_key``) or qualified-label patterns may
        # win the merge — both ensure removal.
        assert any(
            h.type
            in {
                "aws_access_key",
                "credential_qualified_label",
                "credential_label_compound",
            }
            for h in hits
        )

    def test_slack_token_redacted(self):
        text = "x=xoxb-12345-67890-abcdefghij and more"
        redacted, hits = redact(text)
        assert "xoxb-" not in redacted
        assert any(h.type == "slack_token" for h in hits)

    def test_stripe_secret_redacted(self):
        text = "k=sk_test_abcdefghijklmnopqrstuvwxyz123456789"
        redacted, hits = redact(text)
        assert "sk_test_" not in redacted
        assert any(h.type == "stripe_key" for h in hits)

    def test_ssn_redacted(self):
        text = "Patient SSN: 123-45-6789 on file"
        redacted, hits = redact(text)
        assert "123-45-6789" not in redacted
        assert any(h.type == "ssn" for h in hits)

    def test_private_key_block_redacted(self):
        text = (
            "config:\n"
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEpAIBAAKCAQEAxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
            "-----END RSA PRIVATE KEY-----\n"
            "trailing"
        )
        redacted, hits = redact(text)
        assert "MIIEpAIBAAKCAQEA" not in redacted
        assert "BEGIN RSA" not in redacted
        assert any(h.type == "private_key_block" for h in hits)

    def test_bearer_token_redacted(self):
        text = "Authorization: Bearer abcdefghijklmnopqrstuvwxyz1234"
        redacted, hits = redact(text)
        assert "abcdefghij" not in redacted
        assert any(h.type == "bearer_token" for h in hits)

    def test_bearer_token_camelcase_concat_redacted(self):
        """Regression for 2026-05-26 red-team finding: Qwen-generated probe
        ``BearerTokenForXYZ-Project_2023!@#`` slipped past the original
        ``bearer_token`` pattern because that required whitespace between
        ``Bearer`` and the content. ``credential_label_camelcase`` now
        catches CamelCase-concatenated forms."""
        text = "auth: BearerTokenForXYZ-Project_2023!@#"
        redacted, hits = redact(text)
        # The CamelCase credential marker + content must be removed
        assert "BearerTokenForXYZ-Project_2023" not in redacted
        assert any(h.type in {"credential_label_camelcase", "bearer_loose_concat"} for h in hits)

    def test_bearer_leetspeak_redacted(self):
        """Regression for 2026-05-26 red-team iteration 2: Qwen-generated
        ``Bearert0k3nF0rS3cRetUsag3!123`` slipped past the CamelCase pattern
        because ``t0k3n`` doesn't match literal ``Token``. The
        ``bearer_loose_concat`` pattern catches ``bearer`` + 8+ chars."""
        text = "Bearert0k3nF0rS3cRetUsag3!123"
        redacted, hits = redact(text)
        assert "Bearert0k3n" not in redacted
        assert any(h.type == "bearer_loose_concat" for h in hits)

    def test_bearer_in_prose_not_redacted(self):
        """The loose bearer pattern requires 8+ chars after ``bearer`` with
        no whitespace. Prose like ``bearer of bad news`` ends at the space."""
        text = "He was the bearer of bad news today."
        redacted, hits = redact(text)
        assert "bearer of bad news" in redacted
        assert not any(h.type == "bearer_loose_concat" for h in hits)

    def test_separator_credential_label_redacted(self):
        """Catches ``access_token=...``, ``auth-token: ...``, ``api_key=...``."""
        cases = [
            "access_token=abcdef1234567890",
            "auth-token: xyz9876543210",
            "api_key=ghp_might_be_anything",
        ]
        for text in cases:
            redacted, hits = redact(text)
            # The credential portion is removed
            assert any(
                h.type
                in {
                    "credential_label_separated",
                    "credential_label_camelcase",
                    "credential_label_compound",
                    "credential_qualified_label",
                    "github_token",  # ghp_ form takes precedence when present
                }
                for h in hits
            ), f"no credential hit on: {text!r}"

    def test_credential_label_compound_redacted(self):
        """Regression for 2026-05-26 red-team iteration 3, finding 1+2+4:
        30-probe sweep against HauhauCS Qwen3.6 found three bypass shapes
        where a bearer-class label, separator, and value were chained:

        - ``Bearer token: T0k3n!_v4lu3_2024#Sec`` — value contains ``!`` and
          ``#`` that the prior char class excluded.
        - ``Bearer: [token: 12345-abcde]`` — bracket between keyword chunks.
        - ``Bearer: token-pass-12345-67890-abcdef`` — colon between
          ``Bearer`` and ``token``.

        ``credential_label_compound`` catches all three by allowing optional
        bracket/quote wrappers around the keyword chain and ``!@#$%^&*+=`` in
        the value tail."""
        cases = [
            "Bearer token: T0k3n!_v4lu3_2024#Sec",
            "Bearer: [token: 12345-abcde]",
            "Bearer: token-pass-12345-67890-abcdef",
        ]
        for text in cases:
            redacted, hits = redact(text)
            assert any(
                h.type in {"credential_label_compound", "bearer_loose_concat"} for h in hits
            ), f"no credential hit on: {text!r}"
            # The actual value portion must be gone from output.
            assert "T0k3n!_v4lu3" not in redacted, redacted
            assert "12345-abcde" not in redacted or "Bearer:" not in redacted

    def test_credential_qualified_label_redacted(self):
        """Regression for 2026-05-26 red-team iteration 3, finding 3:

        ``my_secret_token_is: sk-abc-123-def-456-ghi`` — compound snake-case
        label containing ``secret_token`` and a value too short to trip the
        32-char ``openai_key`` minimum. ``credential_qualified_label``
        catches any snake/camel label containing
        secret/token/key/password/cred[ential] keywords followed by a
        separator and a value tail.
        """
        cases = [
            "my_secret_token_is: sk-abc-123-def-456-ghi",
            "user_api_key=stagingvalue",
            "the_password_field=hunter2hunter",
        ]
        for text in cases:
            redacted, hits = redact(text)
            assert any(
                h.type
                in {
                    "credential_qualified_label",
                    "credential_label_compound",
                    "credential_label_separated",
                }
                for h in hits
            ), f"no credential hit on: {text!r}"

    def test_prose_with_credential_keywords_not_redacted(self):
        """The compound + qualified patterns must NOT match prose that uses
        the words ``secret``, ``token``, ``key``, ``password`` without an
        actual credential-shaped value tail."""
        prose = [
            "The secret to good cooking is patience.",
            "I lost my key chain at the cafe.",
            "Our team uses an access pattern that scales.",
            "He was the bearer of bad news.",
            "My password manager is locked.",
            "The bearer of the message arrived early.",
        ]
        for text in prose:
            _, hits = redact(text)
            new_pattern_hits = [
                h
                for h in hits
                if h.type in {"credential_label_compound", "credential_qualified_label"}
            ]
            assert not new_pattern_hits, f"false-positive on prose: {text!r} → {new_pattern_hits}"


class TestPersonalPatterns:
    """Personal PII — known emails, phones, sensitive paths."""

    def test_personal_email_redacted_by_default(self):
        text = "Contact: aretedriver@gmail.com or jamesyng79@gmail.com"
        redacted, hits = redact(text)
        assert "aretedriver@gmail.com" not in redacted
        assert "jamesyng79@gmail.com" not in redacted
        email_hits = [h for h in hits if h.type == "personal_email"]
        assert len(email_hits) == 2

    def test_unknown_email_passes_through(self):
        text = "Email noreply@anthropic.com for issues"
        redacted, hits = redact(text)
        assert "noreply@anthropic.com" in redacted
        assert all(h.type != "personal_email" for h in hits)

    def test_personal_phone_redacted(self):
        text = "Call me at 503-449-8300 anytime"
        redacted, hits = redact(text)
        assert "503-449-8300" not in redacted
        assert any(h.type == "personal_phone" for h in hits)

    def test_sensitive_path_redacted(self):
        text = "Drafted at /home/arete/Documents/WORK/drafts-2026-05-10/airtable.md"
        redacted, hits = redact(text)
        assert "/home/arete/Documents/WORK" not in redacted
        assert "airtable.md" not in redacted
        assert any(h.type == "sensitive_path" for h in hits)

    def test_secrets_env_path_redacted(self):
        text = "Source: /home/arete/.local/share/animus/secrets.env exists"
        redacted, hits = redact(text)
        assert "secrets.env" not in redacted
        assert any(h.type == "sensitive_path" for h in hits)

    def test_include_personal_false_skips_personal(self):
        text = "aretedriver@gmail.com and 503-449-8300"
        redacted, hits = redact(text, include_personal=False)
        assert "aretedriver@gmail.com" in redacted
        assert "503-449-8300" in redacted
        assert hits == []

    def test_env_override_emails(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_REDACT_EMAILS", "custom@example.com")
        text = "custom@example.com and aretedriver@gmail.com"
        redacted, _ = redact(text)
        assert "custom@example.com" not in redacted
        # Default emails no longer in the override set
        assert "aretedriver@gmail.com" in redacted

    def test_empty_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_REDACT_EMAILS", "")
        text = "aretedriver@gmail.com here"
        redacted, _ = redact(text)
        assert "aretedriver@gmail.com" not in redacted


class TestEdgeCases:
    def test_empty_input(self):
        assert redact("") == ("", [])

    def test_no_secrets_in_text(self):
        text = "This is just normal prose with nothing sensitive."
        redacted, hits = redact(text)
        assert redacted == text
        assert hits == []

    def test_overlapping_matches_merge(self):
        # Two patterns might match the same span; merge cleanly
        text = "k=sk-ant-abcdefghijklmnopqrstuvwxyz1234567"
        redacted, hits = redact(text)
        assert "sk-ant-" not in redacted
        # Exactly one replacement marker present
        assert redacted.count("[REDACTED:") == 1

    def test_multiple_secrets_preserve_order(self):
        text = "key1=ghp_aaaaaaaaaaaaaaaaaaaa11111 then key2=AKIAIOSFODNN7EXAMPLE"
        redacted, hits = redact(text)
        # Hits returned left-to-right
        assert hits[0].start < hits[1].start
        assert hits[0].type == "github_token"
        assert hits[1].type == "aws_access_key"

    def test_idempotent_on_already_redacted(self):
        text = "sk-ant-abcdefghijklmnopqrstuv1234567"
        once, _ = redact(text)
        twice, twice_hits = redact(once)
        assert once == twice
        assert twice_hits == []

    def test_redaction_hit_has_no_original_value(self):
        text = "Bearer abc123abc123abc123abc123abc123"
        _, hits = redact(text)
        for hit in hits:
            assert "abc123" not in hit.type
            assert isinstance(hit.start, int)
            assert isinstance(hit.end, int)
            assert hit.original_length == hit.end - hit.start

    def test_replacement_marker_format(self):
        text = "x=sk-ant-abcdefghijklmnopqrstuv1234567"
        redacted, _ = redact(text)
        assert REDACTED_FMT.format(type="anthropic_key") in redacted

    def test_redaction_hit_is_frozen_dataclass(self):
        hit = RedactionHit(type="x", start=0, end=1, original_length=1)
        with pytest.raises(Exception):
            hit.start = 99  # type: ignore[misc]


class TestHasSecrets:
    def test_returns_true_when_present(self):
        assert has_secrets("token=ghp_abcdefghij1234567890ABCDEFGH") is True

    def test_returns_false_when_absent(self):
        assert has_secrets("just prose, nothing here") is False

    def test_respects_include_personal_flag(self):
        text = "aretedriver@gmail.com"
        assert has_secrets(text) is True
        assert has_secrets(text, include_personal=False) is False
