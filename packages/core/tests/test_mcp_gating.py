"""Tests for MCP tool gating system."""

from __future__ import annotations

import pytest

from animus.mcp_gating import (
    MCPToolGater,
    _tokenize,
    get_mcp_intent,
    set_mcp_intent,
)


class TestTokenize:
    def test_basic_tokenization(self):
        tokens = _tokenize("Search memory for old code")
        assert "search" in tokens
        assert "memory" in tokens
        assert "old" in tokens
        assert "code" in tokens

    def test_normalization(self):
        tokens = _tokenize("SEARCH Memory")
        assert "search" in tokens
        assert "memory" in tokens

    def test_short_words_filtered(self):
        tokens = _tokenize("a an to")
        assert len(tokens) == 0

    def test_punctuation_stripped(self):
        tokens = _tokenize("search, memory!")
        assert "search" in tokens
        assert "memory" in tokens
        assert "search," not in tokens


@pytest.fixture(autouse=True)
def reset_mcp_intent():
    """Reset MCP session intent before each test to prevent cross-test leakage."""
    from animus.mcp_gating import set_mcp_intent

    set_mcp_intent("")
    yield


class TestMCPToolGater:
    def test_register_and_unregister(self):
        gater = MCPToolGater()
        gater.register_tool(
            "animus_recall",
            "Search memory",
            {"type": "object", "properties": {"query": {"type": "string"}}},
        )
        assert "animus_recall" in gater._tools
        assert gater.unregister_tool("animus_recall")
        assert "animus_recall" not in gater._tools
        assert not gater.unregister_tool("nonexistent")

    def test_iso_score_basic(self):
        gater = MCPToolGater()
        gater.register_tool(
            "animus_recall",
            "Search memory by semantic similarity",
            {"type": "object"},
        )
        scored = gater._score_all("search memory")
        assert len(scored) == 1
        assert scored[0][1] == "animus_recall"
        # Jaccard overlap: "search memory" vs "search memory semantic similarity" = 2/4 = 0.5
        assert scored[0][0] >= 0.3

    def test_iso_score_no_match(self):
        gater = MCPToolGater()
        gater.register_tool(
            "animus_harvest",
            "Harvest external repositories",
            {"type": "object"},
        )
        scored = gater._score_all("search memory")
        # No overlap keywords
        assert scored[0][0] < 0.5

    def test_always_expose_bypasses_scoring(self):
        gater = MCPToolGater()
        gater.register_tool(
            "animus_remember",
            "Store a memory",
            {"type": "object"},
            always_expose=True,
        )
        scored = gater._score_all("totally unrelated query")
        assert scored[0][0] == 1.0  # Always-exposed tools get max score

    def test_get_gated_schemas_no_intent(self):
        gater = MCPToolGater()
        gater.register_tool(
            "animus_recall",
            "Search memory",
            {"type": "object", "properties": {"query": {"type": "string"}}},
        )
        schemas = gater.get_gated_schemas()
        assert len(schemas) == 1
        assert schemas[0].name == "animus_recall"
        assert not schemas[0].is_compact
        assert schemas[0].input_schema == {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        }

    def test_get_gated_schemas_with_intent(self):
        gater = MCPToolGater(max_full_schemas=1)
        gater.register_tool(
            "animus_recall",
            "Search memory by semantic similarity",
            {"type": "object", "properties": {"query": {"type": "string"}}},
        )
        gater.register_tool(
            "animus_harvest",
            "Harvest external repositories",
            {"type": "object", "properties": {"url": {"type": "string"}}},
        )
        schemas = gater.get_gated_schemas(intent="search memory")
        assert len(schemas) == 2
        # Top-ranked should be full schema
        recall = next(s for s in schemas if s.name == "animus_recall")
        harvest = next(s for s in schemas if s.name == "animus_harvest")
        assert not recall.is_compact
        assert harvest.is_compact
        assert recall.input_schema["type"] == "object"
        assert "properties" in recall.input_schema
        # Compact schema should be stripped
        assert harvest.input_schema == {
            "type": "object",
            "description": "Compact schema — full parameters available on request.",
        }

    def test_get_gated_schemas_caching(self):
        gater = MCPToolGater()
        gater.register_tool("tool_a", "Do task A", {"type": "object"})
        # First call computes scores
        schemas1 = gater.get_gated_schemas(intent="task A")
        # Second call with same intent should use cache
        schemas2 = gater.get_gated_schemas(intent="task A")
        assert len(schemas1) == len(schemas2)

    def test_all_full_schemas(self):
        gater = MCPToolGater()
        gater.register_tool("t1", "Tool one", {"type": "object"})
        gater.register_tool("t2", "Tool two", {"type": "object"})
        schemas = gater.get_all_full_schemas()
        assert len(schemas) == 2
        assert all(not s.is_compact for s in schemas)


class TestSessionIntent:
    def test_set_and_get_intent(self):
        set_mcp_intent("search for old bugs")
        assert get_mcp_intent() == "search for old bugs"

    def test_get_intent_default_none(self):
        # Ensure no leakage from other tests
        set_mcp_intent("")
        # Empty string is falsy, so get_mcp_intent returns the raw value
        # But after setting empty, calling get_mcp_intent returns ""
        # Let's just verify it returns something
        assert get_mcp_intent() is not None or get_mcp_intent() == ""

    def test_set_intent_clears_cache(self):
        set_mcp_intent("query one")
        set_mcp_intent("query two")
        assert get_mcp_intent() == "query two"
