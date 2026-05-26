"""Tests for the Stage 3.C sibling-adopter egress gate in bootstrap's
AnthropicBackend (gateway/cognitive.py).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from animus_bootstrap.gateway.cognitive import AnthropicBackend


class TestAnthropicBackendEgressGate:
    """``AnthropicBackend._check_egress`` honors ``ANIMUS_OFFLINE=1`` via
    the soft-imported ``animus.network.is_egress_allowed`` helper."""

    def _patch_egress(self, allowed: bool):
        """Patch is_egress_allowed at the cognitive module's import site."""
        return patch(
            "animus_bootstrap.gateway.cognitive.is_egress_allowed",
            return_value=allowed,
        )

    def test_check_egress_passes_when_allowed(self):
        backend = AnthropicBackend(api_key="dummy", model="claude-x")
        with self._patch_egress(True):
            backend._check_egress()  # no raise

    def test_check_egress_raises_when_denied(self):
        from animus_bootstrap.gateway.cognitive import EgressDeniedError

        backend = AnthropicBackend(api_key="dummy", model="claude-x")
        with self._patch_egress(False):
            with pytest.raises(EgressDeniedError, match="ANIMUS_OFFLINE"):
                backend._check_egress()

    @pytest.mark.asyncio
    async def test_generate_response_refuses_when_offline(self):
        from animus_bootstrap.gateway.cognitive import EgressDeniedError

        backend = AnthropicBackend(api_key="dummy", model="claude-x")
        with self._patch_egress(False):
            with pytest.raises(EgressDeniedError):
                await backend.generate_response(messages=[{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_generate_structured_refuses_when_offline(self):
        from animus_bootstrap.gateway.cognitive import EgressDeniedError

        backend = AnthropicBackend(api_key="dummy", model="claude-x")
        with self._patch_egress(False):
            with pytest.raises(EgressDeniedError):
                await backend.generate_structured(messages=[{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_generate_response_proceeds_when_allowed(self):
        """When egress passes the gate, the httpx call site IS reached.
        We patch httpx itself so we don't actually hit api.anthropic.com."""
        backend = AnthropicBackend(api_key="dummy", model="claude-x")

        mock_response_payload = {"content": [{"type": "text", "text": "ok"}]}

        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return mock_response_payload

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, *_args, **_kwargs):
                return FakeResp()

        with self._patch_egress(True):
            with patch(
                "animus_bootstrap.gateway.cognitive.httpx.AsyncClient",
                return_value=FakeClient(),
            ):
                result = await backend.generate_response(
                    messages=[{"role": "user", "content": "hi"}]
                )
        assert result == "ok"


class TestSoftImportFallback:
    """When animus-core isn't installed, the gate is a no-op (fail-open)."""

    def test_module_exports_egress_helper_or_fallback(self):
        from animus_bootstrap.gateway import cognitive

        # Either the real helper from animus.network OR the inline fallback.
        assert hasattr(cognitive, "is_egress_allowed")
        assert hasattr(cognitive, "EgressDeniedError")
        # The module-level flag tells callers which path is active.
        assert isinstance(cognitive._EGRESS_GATE_AVAILABLE, bool)

    def test_fallback_returns_true(self, monkeypatch):
        """If we patch the helper to the fallback shape, it must return True
        for any input (preserving pre-hardening behavior)."""
        from animus_bootstrap.gateway import cognitive

        def fake_allow(destination, tier=None):
            return True

        monkeypatch.setattr(cognitive, "is_egress_allowed", fake_allow)
        assert cognitive.is_egress_allowed("https://api.anthropic.com") is True
        assert cognitive.is_egress_allowed("https://api.openai.com", "any") is True
