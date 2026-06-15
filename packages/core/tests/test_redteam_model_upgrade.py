"""E14 — Ollama upgrade: HauhauCS Qwen3.6-35B-A3B loads + reasoning fallback.

Verifies that the red-team driver:
1. Defaults to the aggressive HauhauCS model
2. Handles thinking-model responses where content is empty but reasoning has the payload
3. Extracts the actual payload from reasoning text with Output:/Payload:/Result: markers
4. Falls back gracefully when neither content nor reasoning is present
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from animus.redteam.driver import DEFAULT_RED_TEAM_MODEL, RedTeamDriver


class TestDefaultModel:
    """The default must be the aggressive HauhauCS model, not qwen2.5:14b."""

    def test_default_is_hauhaucs_35b(self) -> None:
        assert "HauhauCS" in DEFAULT_RED_TEAM_MODEL
        assert "35B" in DEFAULT_RED_TEAM_MODEL
        assert "A3B" in DEFAULT_RED_TEAM_MODEL

    def test_driver_uses_default(self) -> None:
        driver = RedTeamDriver()
        assert driver.model == DEFAULT_RED_TEAM_MODEL


class TestReasoningFallback:
    """E14: thinking models put everything in ``reasoning``; driver must extract it."""

    def _make_driver(self) -> RedTeamDriver:
        return RedTeamDriver(model="fake-model", timeout_seconds=5.0)

    def test_content_populated_no_fallback(self) -> None:
        """Normal path: content has text, reasoning is ignored."""
        driver = self._make_driver()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "  '<script>alert(1)</script>'  ",
                        "reasoning": "some thinking text",
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response):
            results = driver._generate("test prompt")
        # n_per_category=3, all three return the same stripped content
        assert len(results) == 3
        assert all(r == "<script>alert(1)</script>" for r in results)

    def test_empty_content_falls_back_to_reasoning_with_output_marker(self) -> None:
        """Thinking model: content empty, reasoning has 'Output: <payload>'."""
        driver = self._make_driver()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning": "some thinking\n\n   Output: `<img src=x onerror=alert(1)>`",
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response):
            results = driver._generate("test prompt")
        assert len(results) == 3
        assert all(r == "<img src=x onerror=alert(1)>" for r in results)

    def test_empty_content_falls_back_to_reasoning_with_payload_marker(self) -> None:
        driver = self._make_driver()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning": "analysis...\nPayload: OR 1=1--",
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response):
            results = driver._generate("test prompt")
        assert len(results) == 3
        assert all(r == "OR 1=1--" for r in results)

    def test_empty_content_falls_back_to_reasoning_with_result_marker(self) -> None:
        driver = self._make_driver()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning": "analysis...\nResult: fake-token-12345",
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response):
            results = driver._generate("test prompt")
        assert len(results) == 3
        assert all(r == "fake-token-12345" for r in results)

    def test_empty_content_and_reasoning_returns_empty(self) -> None:
        """Graceful degradation when the model gives nothing useful."""
        driver = self._make_driver()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "", "reasoning": ""}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response):
            results = driver._generate("test prompt")
        assert results == []

    def test_reasoning_without_marker_uses_full_text(self) -> None:
        """If no Output:/Payload:/Result: marker, use the whole reasoning text."""
        driver = self._make_driver()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning": "just a raw payload string",
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response):
            results = driver._generate("test prompt")
        assert len(results) == 3
        assert all(r == "just a raw payload string" for r in results)

    def test_inline_thinking_tags_stripped(self) -> None:
        """When the backend ignores enable_thinking, tags leak into content."""
        driver = self._make_driver()
        mock_response = MagicMock()
        # Both start and end tags must have \\n\\n prefix for the driver to strip
        thinking_block = "\n\n<antThinking>some analysis\n\n</antThinking>\n\npayload-after"
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": thinking_block,
                        "reasoning": "",
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response):
            results = driver._generate("test prompt")
        assert len(results) == 3
        assert all(r == "payload-after" for r in results)


class TestMaxTokens:
    """E14: max_tokens bumped to 1000 to account for thinking overhead."""

    def test_max_tokens_is_1000(self) -> None:
        driver = RedTeamDriver(model="fake-model", timeout_seconds=5.0)
        captured = {}

        def capture_post(url, **kwargs):
            captured["json"] = kwargs.get("json", {})
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch("httpx.post", side_effect=capture_post):
            driver._generate("test prompt")

        assert captured["json"].get("max_tokens") == 1000
