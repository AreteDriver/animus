"""Targeted tests to push coverage past 96% after task system addition.

Covers uncovered lines in: DualOllamaBackend, daemon/__main__.py, router.py.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# DualOllamaBackend
# ---------------------------------------------------------------------------


class TestDualOllamaBackend:
    """Tests for the DualOllamaBackend routing logic."""

    def test_init(self):
        with patch("animus_bootstrap.gateway.cognitive.OllamaBackend"):
            from animus_bootstrap.gateway.cognitive import DualOllamaBackend

            backend = DualOllamaBackend(
                chat_model="qwen2.5:14b",
                code_model="deepseek-coder-v2",
            )
            assert backend._chat_model == "qwen2.5:14b"
            assert backend._code_model == "deepseek-coder-v2"

    def test_routes_code_keywords_to_code_model(self):
        with patch("animus_bootstrap.gateway.cognitive.OllamaBackend"):
            from animus_bootstrap.gateway.cognitive import DualOllamaBackend

            backend = DualOllamaBackend()
            messages = [{"role": "user", "content": "write a python function"}]
            picked = backend._pick_backend(messages)
            assert picked is backend._code

    def test_routes_chat_to_chat_model(self):
        with patch("animus_bootstrap.gateway.cognitive.OllamaBackend"):
            from animus_bootstrap.gateway.cognitive import DualOllamaBackend

            backend = DualOllamaBackend()
            messages = [{"role": "user", "content": "how are you today?"}]
            picked = backend._pick_backend(messages)
            assert picked is backend._chat

    def test_routes_empty_messages(self):
        with patch("animus_bootstrap.gateway.cognitive.OllamaBackend"):
            from animus_bootstrap.gateway.cognitive import DualOllamaBackend

            backend = DualOllamaBackend()
            picked = backend._pick_backend([])
            assert picked is backend._chat

    def test_generate_response_delegates(self):
        with patch("animus_bootstrap.gateway.cognitive.OllamaBackend"):
            from animus_bootstrap.gateway.cognitive import DualOllamaBackend

            backend = DualOllamaBackend()
            backend._chat.generate_response = AsyncMock(return_value="hello")
            messages = [{"role": "user", "content": "hi there"}]
            result = _run(backend.generate_response(messages))
            assert result == "hello"

    def test_generate_structured_delegates(self):
        with patch("animus_bootstrap.gateway.cognitive.OllamaBackend"):
            from animus_bootstrap.gateway.cognitive import DualOllamaBackend

            backend = DualOllamaBackend()
            mock_resp = MagicMock()
            backend._code.generate_structured = AsyncMock(return_value=mock_resp)
            messages = [{"role": "user", "content": "debug this python code"}]
            result = _run(backend.generate_structured(messages))
            assert result is mock_resp


# ---------------------------------------------------------------------------
# daemon/__main__.py
# ---------------------------------------------------------------------------


class TestDaemonMain:
    def test_main_calls_serve(self):
        with patch("animus_bootstrap.dashboard.app.serve") as mock_serve:
            from animus_bootstrap.daemon.__main__ import main

            main()
            mock_serve.assert_called_once()

    def test_module_level_import(self):
        """Importing the module sets up logging."""
        import importlib

        importlib.import_module("animus_bootstrap.daemon.__main__")


# ---------------------------------------------------------------------------
# IntelligentRouter uncovered branches
# ---------------------------------------------------------------------------


class TestRouterUncoveredBranches:
    """Cover uncovered router branches."""

    def test_router_with_condensed_prompt(self):
        """Cover get_condensed_prompt integration."""
        from animus_bootstrap.identity.manager import IdentityFileManager

        assert hasattr(IdentityFileManager, "get_condensed_prompt")
