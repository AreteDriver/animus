"""Tests for the canonical egress policy in animus_types.

This module is the single source of truth that Core and Forge both re-export,
replacing two hand-synced copies. These tests pin the policy and the
dual-call-convention contract (Core's ``tier`` positional/keyword, Forge's
``sensitivity`` keyword).
"""

from __future__ import annotations

import pytest

from animus_types import EgressDeniedError, Sensitivity, is_egress_allowed


class TestLoopbackAlwaysAllowed:
    @pytest.mark.parametrize(
        "dest",
        [
            "localhost",
            "127.0.0.1",
            "::1",
            "0.0.0.0",
            "http://localhost:11434",
            "ollama.local:11434",
        ],
    )
    def test_loopback_allowed_even_for_secret(self, dest):
        assert is_egress_allowed(dest, Sensitivity.SECRET) is True


class TestTierRules:
    def test_public_allowed(self):
        assert is_egress_allowed("https://api.anthropic.com", Sensitivity.PUBLIC) is True

    @pytest.mark.parametrize("tier", [Sensitivity.CONFIDENTIAL, Sensitivity.SECRET])
    def test_confidential_secret_denied(self, tier):
        assert is_egress_allowed("https://api.anthropic.com", tier) is False

    def test_personal_allowed_by_default(self):
        assert is_egress_allowed("https://api.anthropic.com", Sensitivity.PERSONAL) is True

    def test_personal_denied_under_local_only(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_LOCAL_ONLY", "1")
        assert is_egress_allowed("https://api.anthropic.com", Sensitivity.PERSONAL) is False

    def test_offline_denies_all_non_loopback(self, monkeypatch):
        monkeypatch.setenv("ANIMUS_OFFLINE", "1")
        assert is_egress_allowed("https://api.anthropic.com", Sensitivity.PUBLIC) is False
        assert is_egress_allowed("http://localhost:11434", Sensitivity.SECRET) is True


class TestDualCallConvention:
    """Both historical call conventions must keep working off one impl."""

    def test_core_positional_tier(self):
        assert is_egress_allowed("https://api.openai.com", Sensitivity.SECRET) is False

    def test_core_tier_keyword(self):
        assert is_egress_allowed("https://api.openai.com", tier=Sensitivity.SECRET) is False

    def test_forge_sensitivity_keyword(self):
        assert is_egress_allowed("https://api.openai.com", sensitivity=Sensitivity.SECRET) is False

    def test_sensitivity_wins_when_both_given(self):
        # sensitivity (Forge) takes precedence over tier (Core) if both passed.
        assert (
            is_egress_allowed(
                "https://api.openai.com",
                tier=Sensitivity.PUBLIC,
                sensitivity=Sensitivity.SECRET,
            )
            is False
        )

    def test_no_tier_is_permissive(self):
        assert is_egress_allowed("https://api.anthropic.com") is True


class TestSharedErrorClass:
    def test_egress_denied_is_runtime_error(self):
        assert issubclass(EgressDeniedError, RuntimeError)

    def test_class_identity_across_packages(self):
        from animus.network import EgressDeniedError as CoreErr
        from animus_forge.network import EgressDeniedError as ForgeErr

        assert CoreErr is ForgeErr is EgressDeniedError


class TestContentAwareEgress:
    """A4: even when the tier permits egress, a credential-bearing payload is
    blocked — the gate stops trusting the caller's tier alone."""

    SECRET = "deploy with sk-ant-abc1234567890ABCDEFGHIJ now"

    def test_public_tier_with_secret_payload_is_blocked(self):
        assert is_egress_allowed("https://api.anthropic.com", Sensitivity.PUBLIC) is True
        assert (
            is_egress_allowed("https://api.anthropic.com", Sensitivity.PUBLIC, content=self.SECRET)
            is False
        )

    def test_clean_payload_still_allowed(self):
        assert (
            is_egress_allowed(
                "https://api.anthropic.com",
                Sensitivity.PUBLIC,
                content="summarize this quarterly budget report",
            )
            is True
        )

    def test_loopback_bypasses_content_scan(self):
        # Local Ollama is on-box; sending a secret locally is fine.
        assert (
            is_egress_allowed("http://localhost:11434", Sensitivity.PUBLIC, content=self.SECRET)
            is True
        )

    def test_content_none_is_backward_compatible(self):
        assert is_egress_allowed("https://api.openai.com", Sensitivity.PUBLIC) is True
