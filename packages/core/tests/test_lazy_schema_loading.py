"""Tests for lazy schema loading and Intent–Schema Overlap (ISO) scoring.

Validates P-20260706-001: Dynamic Tool Gating + Lazy Schema Loading.
"""

import pytest

from animus.tools import Tool, ToolRegistry, tools_to_anthropic_format


@pytest.fixture
def registry():
    """Registry with diverse tools for lazy loading tests."""
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="read_file",
            description="Read contents of a local file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "max_size": {"type": "integer", "description": "Max bytes"},
                },
                "required": ["path"],
            },
            handler=lambda p: None,
            category="filesystem",
        )
    )
    reg.register(
        Tool(
            name="web_search",
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
            handler=lambda p: None,
            category="web",
        )
    )
    reg.register(
        Tool(
            name="run_command",
            description="Execute a shell command",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command"},
                },
                "required": ["command"],
            },
            handler=lambda p: None,
            category="system",
        )
    )
    return reg


class TestISOScoring:
    """Intent–Schema Overlap (ISO) scoring."""

    def test_iso_score_matches_name(self, registry):
        """Keyword overlap with tool name raises score."""
        read_tool = registry.get("read_file")
        score = registry._iso_score("read the file contents", read_tool)
        assert score >= 0.25

    def test_iso_score_matches_param(self, registry):
        """Keyword overlap with parameter names raises score."""
        search_tool = registry.get("web_search")
        score = registry._iso_score("search for python documentation", search_tool)
        assert score > 0.2

    def test_iso_score_low_for_mismatch(self, registry):
        """Unrelated intent yields low score."""
        cmd_tool = registry.get("run_command")
        score = registry._iso_score("what is the weather today", cmd_tool)
        assert score < 0.2

    def test_iso_score_neutral_for_empty_intent(self, registry):
        """Empty intent returns neutral 0.5."""
        tool = registry.get("read_file")
        score = registry._iso_score("", tool)
        assert score == 0.5


class TestHistoryBoost:
    """History-aware routing boosts/penalties."""

    def test_success_boost(self, registry):
        """Recent successful uses boost score."""
        registry.record_tool_use("read_file", success=True)
        base = registry._iso_score("read file", registry.get("read_file"))
        boosted = registry._apply_history_boost(base, "read_file")
        assert boosted > base

    def test_failure_penalty(self, registry):
        """Recent failures penalize score."""
        registry.record_tool_use("run_command", success=False)
        base = registry._iso_score("run shell", registry.get("run_command"))
        penalized = registry._apply_history_boost(base, "run_command")
        assert penalized < base

    def test_history_bounded(self, registry):
        """History list is bounded to last 20 entries."""
        for i in range(25):
            registry.record_tool_use("read_file", success=True)
        assert len(registry._tool_history["read_file"]) == 20


class TestLazySchemaLoading:
    """Two-phase lazy schema loading."""

    def test_lazy_returns_compact_for_low_relevance(self, registry):
        """Low-relevance tools get compact schemas."""
        intent = "search for python docs"
        schemas = registry.get_schema(
            intent=intent, lazy=True, max_full_schemas=1
        )
        # Top 1 should be web_search (full), others compact
        full_count = sum(1 for s in schemas if "parameters" in s)
        compact_count = sum(1 for s in schemas if "params" in s)
        assert full_count == 1
        assert compact_count == 2

    def test_lazy_returns_full_for_top_n(self, registry):
        """Top-N tools get full schemas."""
        intent = "read a file from disk"
        schemas = registry.get_schema(
            intent=intent, lazy=True, max_full_schemas=2
        )
        full = [s for s in schemas if "parameters" in s]
        assert len(full) == 2

    def test_non_lazy_returns_all_full(self, registry):
        """lazy=False returns full schemas for all tools."""
        schemas = registry.get_schema(lazy=False)
        assert all("parameters" in s for s in schemas)

    def test_lazy_adds_iso_score(self, registry):
        """Lazy schemas include _iso_score metadata."""
        schemas = registry.get_schema(
            intent="read file", lazy=True, max_full_schemas=2
        )
        for s in schemas:
            assert "_iso_score" in s
            assert 0.0 <= s["_iso_score"] <= 1.0

    def test_lazy_text_format(self, registry):
        """get_schema_text supports lazy loading."""
        text = registry.get_schema_text(
            intent="read file", lazy=True, max_full_schemas=1
        )
        lines = text.split("\n")
        # Should include relevance scores
        assert any("relevance:" in line for line in lines)


class TestToolsToAnthropicFormat:
    """Conversion to Anthropic format with lazy loading."""

    def test_anthropic_format_includes_top_tools(self, registry):
        """Top tools get full input_schema."""
        intent = "read file contents"
        result = tools_to_anthropic_format(
            registry, intent=intent, lazy=True, max_full_schemas=2
        )
        # Verify no _iso_score leaks into Anthropic format
        for item in result:
            assert "_iso_score" not in item
            assert "name" in item
            assert "description" in item
            assert "input_schema" in item

    def test_anthropic_format_compact_params(self, registry):
        """Compact tools still have params in input_schema."""
        intent = "read file"
        result = tools_to_anthropic_format(
            registry, intent=intent, lazy=True, max_full_schemas=1
        )
        # The top tool should have full schema
        top = result[0]
        assert isinstance(top["input_schema"], dict)
        # Other tools should have list of param names
        compact = result[1]
        assert isinstance(compact["input_schema"], (dict, list))
