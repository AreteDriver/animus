"""E12 — Model supply-chain pin tests.

Verifies that OllamaProvider refuses inference when a model's digest
does not match the recorded pin, and that pinning/unpinning round-trips.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from animus_forge.providers.model_pin import ModelPinStore, fetch_ollama_digest
from animus_forge.providers.ollama_provider import OllamaProvider


class TestModelPinStore:
    def test_round_trip(self, tmp_path: Path) -> None:
        store = ModelPinStore(path=tmp_path / "pins.json")
        store.pin_model("qwen2.5:14b", "sha256:abc123")
        assert store.get_pin("qwen2.5:14b") == "sha256:abc123"

    def test_persistence(self, tmp_path: Path) -> None:
        path = tmp_path / "pins.json"
        store = ModelPinStore(path=path)
        store.pin_model("llama3.2", "sha256:def456")

        store2 = ModelPinStore(path=path)
        assert store2.get_pin("llama3.2") == "sha256:def456"

    def test_unpin(self, tmp_path: Path) -> None:
        store = ModelPinStore(path=tmp_path / "pins.json")
        store.pin_model("mistral", "sha256:aaa")
        store.unpin_model("mistral")
        assert store.get_pin("mistral") is None

    def test_verify_pin_no_pin(self, tmp_path: Path) -> None:
        store = ModelPinStore(path=tmp_path / "pins.json")
        assert store.verify_pin("any_model", "sha256:zzz") is True

    def test_verify_pin_matches(self, tmp_path: Path) -> None:
        store = ModelPinStore(path=tmp_path / "pins.json")
        store.pin_model("qwen2.5:14b", "sha256:abc")
        assert store.verify_pin("qwen2.5:14b", "sha256:abc") is True

    def test_verify_pin_mismatch(self, tmp_path: Path) -> None:
        store = ModelPinStore(path=tmp_path / "pins.json")
        store.pin_model("qwen2.5:14b", "sha256:abc")
        assert store.verify_pin("qwen2.5:14b", "sha256:def") is False


class TestFetchOllamaDigest:
    def test_fetch_returns_digest(self) -> None:
        mock_data = {"models": [{"model": "qwen2.5:14b", "digest": "sha256:abc123"}]}
        with patch("animus_forge.providers.model_pin.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(json=lambda: mock_data, raise_for_status=lambda: None)
            digest = fetch_ollama_digest("qwen2.5:14b")
            assert digest == "sha256:abc123"

    def test_fetch_returns_none_when_missing(self) -> None:
        mock_data = {"models": [{"model": "llama3.2", "digest": "sha256:other"}]}
        with patch("animus_forge.providers.model_pin.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(json=lambda: mock_data, raise_for_status=lambda: None)
            digest = fetch_ollama_digest("qwen2.5:14b")
            assert digest is None

    def test_fetch_returns_none_on_error(self) -> None:
        with patch("animus_forge.providers.model_pin.httpx.get") as mock_get:
            mock_get.side_effect = Exception("connection refused")
            digest = fetch_ollama_digest("qwen2.5:14b")
            assert digest is None


class TestE12OllamaProvider:
    @pytest.fixture
    def provider(self) -> OllamaProvider:
        return OllamaProvider(model="qwen2.5:14b")

    def test_complete_blocked_on_digest_mismatch(
        self, provider: OllamaProvider, tmp_path: Path
    ) -> None:
        store = ModelPinStore(path=tmp_path / "pins.json")
        store.pin_model("qwen2.5:14b", "sha256:expected")

        provider._client = MagicMock()
        provider._initialized = True

        with patch(
            "animus_forge.providers.ollama_provider.fetch_ollama_digest",
            return_value="sha256:attacker",
        ):
            with patch(
                "animus_forge.providers.ollama_provider.ModelPinStore",
                return_value=store,
            ):
                from animus_forge.providers.base import CompletionRequest, ProviderError

                with pytest.raises(ProviderError) as exc_info:
                    provider.complete(
                        CompletionRequest(prompt="hello", model="qwen2.5:14b")
                    )
                assert "digest mismatch" in str(exc_info.value).lower()

    def test_complete_allowed_when_no_pin(
        self, provider: OllamaProvider, tmp_path: Path
    ) -> None:
        store = ModelPinStore(path=tmp_path / "pins.json")
        provider._client = MagicMock()
        provider._client.post.return_value = MagicMock(
            json=lambda: {
                "message": {"content": "hi"},
                "model": "qwen2.5:14b",
                "done": True,
                "prompt_eval_count": 1,
                "eval_count": 1,
            },
            raise_for_status=lambda: None,
        )
        provider._initialized = True

        with patch(
            "animus_forge.providers.ollama_provider.ModelPinStore",
            return_value=store,
        ):
            from animus_forge.providers.base import CompletionRequest

            resp = provider.complete(
                CompletionRequest(prompt="hello", model="qwen2.5:14b")
            )
            assert resp.content == "hi"

    def test_complete_allowed_when_digest_matches(
        self, provider: OllamaProvider, tmp_path: Path
    ) -> None:
        store = ModelPinStore(path=tmp_path / "pins.json")
        store.pin_model("qwen2.5:14b", "sha256:good")

        provider._client = MagicMock()
        provider._client.post.return_value = MagicMock(
            json=lambda: {
                "message": {"content": "ok"},
                "model": "qwen2.5:14b",
                "done": True,
                "prompt_eval_count": 1,
                "eval_count": 1,
            },
            raise_for_status=lambda: None,
        )
        provider._initialized = True

        with patch(
            "animus_forge.providers.ollama_provider.fetch_ollama_digest",
            return_value="sha256:good",
        ):
            with patch(
                "animus_forge.providers.ollama_provider.ModelPinStore",
                return_value=store,
            ):
                from animus_forge.providers.base import CompletionRequest

                resp = provider.complete(
                    CompletionRequest(prompt="hello", model="qwen2.5:14b")
                )
                assert resp.content == "ok"

    def test_complete_blocked_when_fetch_fails(
        self, provider: OllamaProvider, tmp_path: Path
    ) -> None:
        store = ModelPinStore(path=tmp_path / "pins.json")
        store.pin_model("qwen2.5:14b", "sha256:expected")

        provider._client = MagicMock()
        provider._initialized = True

        with patch(
            "animus_forge.providers.ollama_provider.fetch_ollama_digest",
            return_value=None,
        ):
            with patch(
                "animus_forge.providers.ollama_provider.ModelPinStore",
                return_value=store,
            ):
                from animus_forge.providers.base import CompletionRequest, ProviderError

                with pytest.raises(ProviderError) as exc_info:
                    provider.complete(
                        CompletionRequest(prompt="hello", model="qwen2.5:14b")
                    )
                assert "could not be retrieved" in str(exc_info.value).lower()
