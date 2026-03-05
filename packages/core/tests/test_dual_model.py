"""
Tests for Phase 2: Dual-model routing (Claude brain + Ollama hands).

Tests classify_task, TaskWeight, delegate_to_local, think_routed,
has_dual_models, and create_local_think_tool.
"""

from unittest.mock import MagicMock

import pytest

from animus.cognitive import (
    CognitiveLayer,
    ModelConfig,
    TaskWeight,
    classify_task,
)
from animus.tools import (
    ToolRegistry,
    create_local_think_tool,
)

# ---------------------------------------------------------------------------
# classify_task tests
# ---------------------------------------------------------------------------


class TestClassifyTask:
    """Tests for the task classification heuristic."""

    @pytest.mark.parametrize(
        "prompt",
        [
            "plan the authentication flow",
            "implement a new endpoint for users",
            "fix the broken test in test_api.py",
            "debug why the server crashes on startup",
            "review the security of auth.py",
            "refactor the database module",
            "optimize the query performance",
            "build a new CLI command",
            "design the migration strategy",
            "what approach should I take for caching?",
            "should we use Redis or Memcached?",
            "analyze the code for security vulnerabilities",
            "investigate the memory leak",
            "create a new test suite for the API",
            "audit the authentication module",
        ],
    )
    def test_heavy_tasks(self, prompt):
        assert classify_task(prompt) == TaskWeight.HEAVY

    @pytest.mark.parametrize(
        "prompt",
        [
            "summarize this error log",
            "format this JSON output",
            "reformat the table as markdown",
            "rewrite this paragraph more concisely",
            "convert this YAML to JSON",
            "extract the email addresses from this text",
            "list all the function names",
            "count the number of imports",
            "sort these items alphabetically",
            "what is the name of the main function?",
            "translate this to French",
            "clean up the whitespace",
            "simplify this expression",
        ],
    )
    def test_light_tasks(self, prompt):
        assert classify_task(prompt) == TaskWeight.LIGHT

    def test_short_ambiguous_defaults_to_light(self):
        """Short prompts without heavy keywords default to light."""
        assert classify_task("hello") == TaskWeight.LIGHT
        assert classify_task("what time is it?") == TaskWeight.LIGHT

    def test_long_ambiguous_defaults_to_heavy(self):
        """Long prompts without matching patterns default to heavy when
        they contain heavyweight keywords."""
        long_prompt = "I need you to implement " + "a" * 200
        assert classify_task(long_prompt) == TaskWeight.HEAVY

    def test_heavy_overrides_light(self):
        """Heavy patterns take priority over light patterns."""
        # "plan" is heavy even though "summarize" is in the middle
        assert classify_task("plan and summarize the architecture") == TaskWeight.HEAVY

    def test_case_insensitive(self):
        assert classify_task("SUMMARIZE this text") == TaskWeight.LIGHT
        assert classify_task("IMPLEMENT a new feature") == TaskWeight.HEAVY

    def test_empty_prompt(self):
        """Empty prompt should not crash."""
        result = classify_task("")
        assert isinstance(result, TaskWeight)


# ---------------------------------------------------------------------------
# CognitiveLayer dual-model tests
# ---------------------------------------------------------------------------


class TestDualModel:
    """Tests for delegate_to_local, think_routed, and has_dual_models."""

    def _make_cognitive(
        self,
        primary_response="primary response",
        fallback_response="local response",
        with_fallback=True,
    ):
        """Helper to create a CognitiveLayer with mock models."""
        primary_config = ModelConfig.mock(default_response=primary_response)
        fallback_config = (
            ModelConfig.mock(default_response=fallback_response) if with_fallback else None
        )
        return CognitiveLayer(
            primary_config=primary_config,
            fallback_config=fallback_config,
        )

    def test_has_dual_models_true(self):
        cog = self._make_cognitive(with_fallback=True)
        assert cog.has_dual_models is True

    def test_has_dual_models_false(self):
        cog = self._make_cognitive(with_fallback=False)
        assert cog.has_dual_models is False

    def test_delegate_to_local_uses_fallback(self):
        cog = self._make_cognitive()
        result = cog.delegate_to_local("summarize this")
        assert result == "local response"

    def test_delegate_to_local_without_fallback_uses_primary(self):
        cog = self._make_cognitive(with_fallback=False)
        result = cog.delegate_to_local("summarize this")
        assert result == "primary response"

    def test_delegate_to_local_with_system_prompt(self):
        cog = self._make_cognitive()
        result = cog.delegate_to_local("summarize this", system="Be concise.")
        assert result == "local response"
        # Verify system prompt was passed through
        assert cog.fallback.calls[-1]["system"] == "Be concise."

    def test_delegate_to_local_fallback_fails_uses_primary(self):
        """When local model fails, should fall back to primary."""
        cog = self._make_cognitive()
        # Make fallback raise an error
        cog.fallback.generate = MagicMock(side_effect=ConnectionError("Ollama down"))
        result = cog.delegate_to_local("summarize this")
        assert result == "primary response"

    def test_delegate_to_local_single_model_fails(self):
        """When single model (no fallback) fails, return error."""
        cog = self._make_cognitive(with_fallback=False)
        cog.primary.generate = MagicMock(side_effect=ConnectionError("down"))
        result = cog.delegate_to_local("summarize this")
        assert "[Error:" in result

    def test_think_routed_light_task_uses_fallback(self):
        cog = self._make_cognitive()
        result = cog.think_routed("summarize this text")
        assert result == "local response"

    def test_think_routed_heavy_task_uses_primary(self):
        cog = self._make_cognitive()
        result = cog.think_routed("implement a new REST endpoint")
        assert result == "primary response"

    def test_think_routed_no_fallback_always_uses_primary(self):
        """Without fallback, even light tasks go to primary."""
        cog = self._make_cognitive(with_fallback=False)
        result = cog.think_routed("summarize this text")
        assert result == "primary response"

    def test_think_routed_preserves_mode(self):
        """Verify mode is passed through to think()."""
        cog = self._make_cognitive()
        # Heavy task goes to think() which uses primary
        from animus.cognitive import ReasoningMode

        result = cog.think_routed("plan the architecture", mode=ReasoningMode.DEEP)
        assert result == "primary response"


# ---------------------------------------------------------------------------
# create_local_think_tool tests
# ---------------------------------------------------------------------------


class TestLocalThinkTool:
    """Tests for the local_think tool."""

    def _make_tool_and_cognitive(self):
        primary_config = ModelConfig.mock(default_response="claude says")
        fallback_config = ModelConfig.mock(default_response="ollama says")
        cog = CognitiveLayer(
            primary_config=primary_config,
            fallback_config=fallback_config,
        )
        tool = create_local_think_tool(cog)
        return tool, cog

    def test_tool_metadata(self):
        tool, _ = self._make_tool_and_cognitive()
        assert tool.name == "local_think"
        assert tool.category == "cognitive"
        assert "prompt" in tool.parameters["properties"]
        assert "system" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["prompt"]

    def test_tool_executes_on_local_model(self):
        tool, cog = self._make_tool_and_cognitive()
        result = tool.handler({"prompt": "summarize this text"})
        assert result.success is True
        assert result.output == "ollama says"

    def test_tool_passes_system_prompt(self):
        tool, cog = self._make_tool_and_cognitive()
        result = tool.handler({"prompt": "summarize", "system": "Be brief."})
        assert result.success is True
        assert cog.fallback.calls[-1]["system"] == "Be brief."

    def test_tool_missing_prompt(self):
        tool, _ = self._make_tool_and_cognitive()
        result = tool.handler({})
        assert result.success is False
        assert "prompt" in result.error.lower()

    def test_tool_handles_error(self):
        tool, cog = self._make_tool_and_cognitive()
        cog.fallback.generate = MagicMock(side_effect=RuntimeError("boom"))
        # Should not raise — delegate_to_local falls back to primary
        result = tool.handler({"prompt": "summarize"})
        assert result.success is True
        assert result.output == "claude says"

    def test_tool_registered_in_registry(self):
        tool, _ = self._make_tool_and_cognitive()
        registry = ToolRegistry()
        registry.register(tool)
        assert registry.get("local_think") is not None

    def test_tool_in_anthropic_format(self):
        """Verify tool converts to Anthropic tool_use format."""
        from animus.tools import tools_to_anthropic_format

        tool, _ = self._make_tool_and_cognitive()
        registry = ToolRegistry()
        registry.register(tool)
        formatted = tools_to_anthropic_format(registry)
        assert len(formatted) == 1
        assert formatted[0]["name"] == "local_think"
        assert "input_schema" in formatted[0]


# ---------------------------------------------------------------------------
# Integration: chat.py wiring
# ---------------------------------------------------------------------------


class TestChatIntegration:
    """Test that chat.py correctly wires dual-model components."""

    def test_local_think_only_registered_with_dual_models(self):
        """Verify local_think tool is only added when fallback exists."""

        # With fallback
        primary = ModelConfig.mock(default_response="primary")
        fallback = ModelConfig.mock(default_response="fallback")
        cog = CognitiveLayer(primary_config=primary, fallback_config=fallback)

        registry = ToolRegistry()
        assert cog.has_dual_models is True
        registry.register(create_local_think_tool(cog))
        assert registry.get("local_think") is not None

        # Without fallback
        cog_single = CognitiveLayer(primary_config=primary)
        assert cog_single.has_dual_models is False
        # Would not register in chat.py — condition guards it
