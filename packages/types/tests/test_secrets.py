"""Tests for the canonical credential scanner (roadmap A4)."""

from __future__ import annotations

import pytest

from animus_types.secrets import CREDENTIAL_PATTERNS, scan_for_secrets


class TestScanForSecrets:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("key is sk-ant-abc1234567890ABCDEFGHIJ here", ["anthropic_key"]),
            ("token ghp_ABCDEFGHIJ1234567890xyz", ["github_token"]),
            ("AWS AKIAIOSFODNN7EXAMPLE creds", ["aws_access_key"]),
            ("authorization: Bearer abcdefghijklmnopqrstuvwxyz", ["bearer_token"]),
        ],
    )
    def test_detects_credentials(self, text, expected):
        assert scan_for_secrets(text) == expected

    def test_clean_prose_has_no_hits(self):
        assert scan_for_secrets("a normal sentence about token budgets and keys") == []
        assert scan_for_secrets("") == []

    def test_returns_names_not_values(self):
        # The scanner must never echo the secret — only pattern names.
        hits = scan_for_secrets("sk-ant-abc1234567890ABCDEFGHIJ")
        assert hits == ["anthropic_key"]
        assert "sk-ant" not in ",".join(hits)

    def test_leetspeak_bearer_caught(self):
        # Red-team-hardened pattern (carried from redaction).
        assert "bearer_loose_concat" in scan_for_secrets("Bearert0k3nF0rS3cRet!123")

    def test_pattern_set_is_credentials_only(self):
        # SSN/email/phone are redaction-only PII, NOT in the egress set.
        assert "ssn" not in CREDENTIAL_PATTERNS
        assert "anthropic_key" in CREDENTIAL_PATTERNS
