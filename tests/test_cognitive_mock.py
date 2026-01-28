"""
Tests for MockModel and cognitive features using it.

Enables CI/CD testing of cognitive features without a live LLM backend.
"""

import asyncio
from unittest.mock import MagicMock

from animus.cognitive import (
    CognitiveLayer,
    MockModel,
    ModelConfig,
    ModelProvider,
    ReasoningMode,
    create_model,
)
from animus.decision import DecisionFramework
from animus.protocols.intelligence import IntelligenceProvider

# ---------------------------------------------------------------------------
# MockModel unit tests
# ---------------------------------------------------------------------------


class TestMockModel:
    """Tests for MockModel itself."""

    def test_default_response(self):
        config = ModelConfig.mock()
        model = MockModel(config)
        assert model.generate("anything") == "This is a mock response."

    def test_custom_default_response(self):
        config = ModelConfig.mock(default_response="custom reply")
        model = MockModel(config)
        assert model.generate("anything") == "custom reply"

    def test_response_map_matching(self):
        config = ModelConfig.mock(response_map={"weather": "It's sunny."})
        model = MockModel(config)
        assert model.generate("What's the weather?") == "It's sunny."
        assert model.generate("unrelated") == "This is a mock response."

    def test_response_map_first_match_wins(self):
        config = ModelConfig.mock(
            response_map={
                "alpha": "first",
                "alpha beta": "second",
            }
        )
        model = MockModel(config)
        assert model.generate("alpha beta gamma") == "first"

    def test_call_history_recording(self):
        config = ModelConfig.mock()
        model = MockModel(config)
        model.generate("prompt1", system="sys1")
        model.generate("prompt2")
        assert len(model.calls) == 2
        assert model.calls[0] == {"prompt": "prompt1", "system": "sys1"}
        assert model.calls[1] == {"prompt": "prompt2", "system": None}

    def test_reset_clears_history(self):
        config = ModelConfig.mock()
        model = MockModel(config)
        model.generate("a")
        model.generate("b")
        assert len(model.calls) == 2
        model.reset()
        assert len(model.calls) == 0

    def test_stream_output(self):
        config = ModelConfig.mock(default_response="abcdef")
        model = MockModel(config)
        chunks = asyncio.run(_collect_stream(model, "test"))
        joined = "".join(chunks)
        assert joined == "abcdef"
        assert len(chunks) >= 2  # split into multiple chunks

    def test_factory_creates_mock(self):
        config = ModelConfig.mock()
        model = create_model(config)
        assert isinstance(model, MockModel)

    def test_mock_provider_enum(self):
        assert ModelProvider.MOCK.value == "mock"


async def _collect_stream(model: IntelligenceProvider, prompt: str) -> list[str]:
    chunks = []
    async for chunk in model.generate_stream(prompt):
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# CognitiveLayer integration tests with MockModel
# ---------------------------------------------------------------------------


class TestCognitiveWithMock:
    """Test CognitiveLayer features using MockModel."""

    def _make_cognitive(self, default_response="Mock thinking.", response_map=None):
        config = ModelConfig.mock(
            default_response=default_response,
            response_map=response_map or {},
        )
        return CognitiveLayer(primary_config=config)

    def test_think_quick(self):
        cog = self._make_cognitive()
        result = cog.think("Hello", mode=ReasoningMode.QUICK)
        assert result == "Mock thinking."

    def test_think_deep(self):
        cog = self._make_cognitive()
        result = cog.think("Analyze this", mode=ReasoningMode.DEEP)
        assert result == "Mock thinking."

    def test_think_research(self):
        cog = self._make_cognitive()
        result = cog.think("Research topic", mode=ReasoningMode.RESEARCH)
        assert result == "Mock thinking."

    def test_think_with_context(self):
        cog = self._make_cognitive()
        result = cog.think("Hello", context="Some prior context")
        assert result == "Mock thinking."
        # Verify system prompt included the context
        model = cog.primary
        assert "Some prior context" in model.calls[0]["system"]

    def test_system_prompt_includes_deep_instruction(self):
        cog = self._make_cognitive()
        cog.think("test", mode=ReasoningMode.DEEP)
        model = cog.primary
        assert "step by step" in model.calls[0]["system"]

    def test_system_prompt_includes_research_instruction(self):
        cog = self._make_cognitive()
        cog.think("test", mode=ReasoningMode.RESEARCH)
        model = cog.primary
        assert "Research" in model.calls[0]["system"]

    def test_think_with_tools_no_tools(self):
        """Without tools, think_with_tools falls back to think."""
        cog = self._make_cognitive()
        result = cog.think_with_tools("Hello")
        assert result == "Mock thinking."

    def test_think_with_tools_tool_call(self):
        """Model returns a tool call JSON, verify agentic loop executes it."""
        tool_response = (
            'Let me check.\n```tool\n{"tool": "test_tool", "params": {"key": "val"}}\n```'
        )
        # First iteration matches "User: Hello" -> tool call
        # Second iteration matches "Tool results:" -> final answer (no tool block)
        cog = self._make_cognitive(
            default_response=tool_response,
            response_map={"Tool results:": "Final answer after tools."},
        )

        # Set up a mock tool registry
        mock_registry = MagicMock()
        mock_registry.list_tools.return_value = ["test_tool"]
        mock_registry.get_schema_text.return_value = "Available tools: test_tool"

        mock_tool = MagicMock()
        mock_tool.requires_approval = False
        mock_registry.get.return_value = mock_tool

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.to_context.return_value = "Tool result: done"
        mock_registry.execute.return_value = mock_result

        result = cog.think_with_tools("Hello", tools=mock_registry, max_iterations=3)
        assert result == "Final answer after tools."
        mock_registry.execute.assert_called_once_with("test_tool", {"key": "val"})

    def test_fallback_model(self):
        """Primary fails, fallback MockModel succeeds."""
        primary = ModelConfig.mock(default_response="primary")
        fallback = ModelConfig.mock(default_response="fallback")

        cog = CognitiveLayer(primary_config=primary, fallback_config=fallback)

        # Make primary raise
        cog.primary.generate = MagicMock(side_effect=RuntimeError("down"))

        result = cog.think("test")
        assert result == "fallback"

    def test_brief_no_memories(self):
        """brief() with empty memory returns no-memories message."""
        cog = self._make_cognitive()
        mock_memory = MagicMock()
        mock_memory.store.list_all.return_value = []

        result = cog.brief(mock_memory)
        assert result == "No memories available for briefing."

    def test_brief_with_memories(self):
        """brief() with memories calls think and returns result."""
        cog = self._make_cognitive(default_response="Here's your briefing.")
        mock_memory = MagicMock()

        mem = MagicMock()
        mem.created_at.strftime.return_value = "2026-01-15"
        mem.content = "Important event happened"
        mem.tags = ["work"]
        mock_memory.store.list_all.return_value = [mem]

        result = cog.brief(mock_memory)
        assert result == "Here's your briefing."


# ---------------------------------------------------------------------------
# DecisionFramework tests with MockModel
# ---------------------------------------------------------------------------


class TestDecisionFrameworkWithMock:
    """Test DecisionFramework using MockModel."""

    def test_analyze_with_provided_options_and_criteria(self):
        """Full analysis with explicit options and criteria."""
        response_map = {
            "Briefly assess": "Good option for this criterion.",
            "make a recommendation": "RECOMMENDATION: Option A\nREASONING: Best overall fit.",
        }
        config = ModelConfig.mock(
            default_response="Mock analysis.",
            response_map=response_map,
        )
        cog = CognitiveLayer(primary_config=config)
        framework = DecisionFramework(cog)

        decision = framework.analyze(
            question="Which DB?",
            options=["Postgres", "SQLite"],
            criteria=["Performance", "Simplicity"],
        )

        assert decision.question == "Which DB?"
        assert len(decision.options) == 2
        assert len(decision.criteria) == 2
        assert "Postgres" in decision.analysis
        assert decision.recommendation == "Option A"

    def test_analyze_auto_identify_options(self):
        """When no options provided, model identifies them."""
        response_map = {
            "identify 2-4 distinct options": "Option A\nOption B\nOption C",
            "suggest 3-5 important": "Cost\nSpeed\nReliability",
            "Briefly assess": "Adequate.",
            "make a recommendation": "RECOMMENDATION: Option B\nREASONING: Best value.",
        }
        config = ModelConfig.mock(
            default_response="Fallback.",
            response_map=response_map,
        )
        cog = CognitiveLayer(primary_config=config)
        framework = DecisionFramework(cog)

        decision = framework.analyze("Which cloud provider?")
        assert len(decision.options) > 0
        assert len(decision.criteria) > 0

    def test_quick_decide(self):
        config = ModelConfig.mock(default_response="Go with option B.")
        cog = CognitiveLayer(primary_config=config)
        framework = DecisionFramework(cog)

        result = framework.quick_decide("Should I use Redis or Memcached?")
        assert result == "Go with option B."
