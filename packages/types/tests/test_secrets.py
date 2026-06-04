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
            # C13: Google API key (Vertex / Google provider surface). The key is
            # also long + high-entropy, so C1-6 flags it a second time.
            (
                "key AIzaSyD1234567890abcdefghijklmnopqrstuv x",
                ["google_api_key", "high_entropy_token"],
            ),
        ],
    )
    def test_detects_credentials(self, text, expected):
        assert scan_for_secrets(text) == expected

    def test_clean_prose_has_no_hits(self):
        assert scan_for_secrets("a normal sentence about token budgets and keys") == []
        assert scan_for_secrets("") == []


class TestHighEntropyHeuristic:
    """C1-6 — prefixless high-entropy secrets (no recognizable prefix).

    Fixtures are assembled at runtime so no secret-shaped literal sits in the
    source (the repo's secret scanner would flag it)."""

    @staticmethod
    def _mixed(n: int) -> str:
        # Deterministic high-entropy, 3-char-class token (upper+lower+digit).
        import string

        alpha = string.ascii_letters + string.digits  # 62 symbols
        return "".join(alpha[(i * 37 + 11) % len(alpha)] for i in range(n))

    def test_prefixless_secret_caught(self):
        tok = self._mixed(40)  # no known prefix, mixed classes, high entropy
        assert "high_entropy_token" in scan_for_secrets(f"token={tok}")

    def test_git_sha_not_flagged(self):
        # 40-char hex (lower+digit only = 2 classes) — benign, must not fire.
        sha = "".join("0123456789abcdef"[(i * 7) % 16] for i in range(40))
        assert scan_for_secrets(f"commit {sha}") == []

    def test_uuid_not_flagged(self):
        # Hyphen-split into sub-32 tokens → never reaches the length bar.
        u = "-".join(["".join("0123456789abcdef"[(i) % 16] for i in range(8)) for _ in range(4)])
        assert scan_for_secrets(f"id {u}") == []

    def test_normal_prose_not_flagged(self):
        assert scan_for_secrets("the quick brown fox jumps over the lazy dog repeatedly") == []

    def test_opt_out_disables_entropy(self):
        tok = self._mixed(40)
        assert scan_for_secrets(f"x={tok}", include_high_entropy=False) == []

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
