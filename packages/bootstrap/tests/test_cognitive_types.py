"""Tests for the CognitiveResponse and ToolCall structured types."""

from __future__ import annotations

from animus_bootstrap.gateway.cognitive_types import CognitiveResponse, ToolCall

# ======================================================================
# ToolCall
# ======================================================================


class TestToolCall:
    def test_fields(self) -> None:
        tc = ToolCall(id="toolu_01", name="web_search", arguments={"query": "test"})
        assert tc.id == "toolu_01"
        assert tc.name == "web_search"
        assert tc.arguments == {"query": "test"}

    def test_empty_arguments(self) -> None:
        tc = ToolCall(id="toolu_02", name="get_time", arguments={})
        assert tc.arguments == {}

    def test_nested_arguments(self) -> None:
        args = {"filters": {"category": "news", "limit": 5}, "sort": "date"}
        tc = ToolCall(id="toolu_03", name="search", arguments=args)
        assert tc.arguments["filters"]["category"] == "news"
        assert tc.arguments["filters"]["limit"] == 5


# ======================================================================
# CognitiveResponse
# ======================================================================


class TestCognitiveResponse:
    def test_text_only(self) -> None:
        resp = CognitiveResponse(text="Hello world")
        assert resp.text == "Hello world"
        assert resp.tool_calls == []
        assert resp.stop_reason == "end_turn"
        assert resp.usage == {}

    def test_has_tool_calls_false(self) -> None:
        resp = CognitiveResponse(text="No tools needed")
        assert resp.has_tool_calls is False

    def test_has_tool_calls_true(self) -> None:
        tc = ToolCall(id="toolu_01", name="search", arguments={"q": "test"})
        resp = CognitiveResponse(text="Let me check.", tool_calls=[tc])
        assert resp.has_tool_calls is True

    def test_stop_reason_tool_use(self) -> None:
        tc = ToolCall(id="toolu_01", name="search", arguments={})
        resp = CognitiveResponse(text="", tool_calls=[tc], stop_reason="tool_use")
        assert resp.stop_reason == "tool_use"

    def test_stop_reason_max_tokens(self) -> None:
        resp = CognitiveResponse(text="truncated...", stop_reason="max_tokens")
        assert resp.stop_reason == "max_tokens"

    def test_multiple_tool_calls(self) -> None:
        tc1 = ToolCall(id="toolu_01", name="search", arguments={"q": "a"})
        tc2 = ToolCall(id="toolu_02", name="read_file", arguments={"path": "/tmp/x"})
        resp = CognitiveResponse(text="", tool_calls=[tc1, tc2])
        assert len(resp.tool_calls) == 2
        assert resp.tool_calls[0].name == "search"
        assert resp.tool_calls[1].name == "read_file"

    def test_empty_text_with_tool_calls(self) -> None:
        tc = ToolCall(id="toolu_01", name="calc", arguments={"expr": "2+2"})
        resp = CognitiveResponse(text="", tool_calls=[tc], stop_reason="tool_use")
        assert resp.text == ""
        assert resp.has_tool_calls is True


class TestCognitiveResponseUsage:
    def test_usage_dict(self) -> None:
        resp = CognitiveResponse(
            text="result",
            usage={"input_tokens": 100, "output_tokens": 50},
        )
        assert resp.usage["input_tokens"] == 100
        assert resp.usage["output_tokens"] == 50

    def test_usage_default_empty(self) -> None:
        resp = CognitiveResponse(text="hi")
        assert resp.usage == {}

    def test_usage_with_cache_tokens(self) -> None:
        resp = CognitiveResponse(
            text="cached",
            usage={"input_tokens": 200, "output_tokens": 30, "cache_read_input_tokens": 150},
        )
        assert resp.usage["cache_read_input_tokens"] == 150
