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


# ---------------------------------------------------------------------------
# Provider flexibility tests
# ---------------------------------------------------------------------------


class TestModelProviders:
    """Test that all model providers are properly wired."""

    def test_create_model_ollama(self):
        from animus.cognitive import OllamaModel

        config = ModelConfig.ollama("llama3:8b")
        model = create_model(config)
        assert isinstance(model, OllamaModel)

    def test_create_model_anthropic(self):
        from animus.cognitive import AnthropicModel

        config = ModelConfig.anthropic("claude-3-haiku-20240307")
        model = create_model(config)
        assert isinstance(model, AnthropicModel)

    def test_create_model_openai(self):
        from animus.cognitive import OpenAIModel

        config = ModelConfig.openai("gpt-4o")
        model = create_model(config)
        assert isinstance(model, OpenAIModel)

    def test_create_model_mock(self):
        config = ModelConfig.mock()
        model = create_model(config)
        assert isinstance(model, MockModel)

    def test_create_model_unknown_provider_raises(self):
        import pytest

        config = ModelConfig(provider=ModelProvider.OLLAMA, model_name="test")
        # Manually set a bogus provider to trigger the else branch
        config.provider = "bogus"
        with pytest.raises(ValueError, match="Unsupported provider"):
            create_model(config)

    def test_openai_config_factory(self):
        config = ModelConfig.openai("gpt-4o-mini")
        assert config.provider == ModelProvider.OPENAI
        assert config.model_name == "gpt-4o-mini"

    def test_openai_config_base_url(self):
        """OpenAI-compatible endpoints use base_url for custom servers."""
        config = ModelConfig.openai("local-model")
        config.base_url = "http://localhost:8080/v1"
        assert config.base_url == "http://localhost:8080/v1"

    def test_all_providers_in_enum(self):
        assert ModelProvider.OLLAMA.value == "ollama"
        assert ModelProvider.ANTHROPIC.value == "anthropic"
        assert ModelProvider.OPENAI.value == "openai"
        assert ModelProvider.MOCK.value == "mock"


class TestConfigProviderWiring:
    """Test that AnimusConfig properly maps to cognitive ModelConfig."""

    def test_config_openai_base_url_field(self):
        from animus.config import ModelConfig as CfgModelConfig

        cfg = CfgModelConfig(
            provider="openai",
            name="gpt-4o",
            openai_base_url="http://localhost:1234/v1",
        )
        assert cfg.openai_base_url == "http://localhost:1234/v1"

    def test_config_serialization_includes_openai_base_url(self):
        from animus.config import AnimusConfig

        config = AnimusConfig()
        config.model.openai_base_url = "http://custom:8080/v1"
        d = config.to_dict()
        assert d["model"]["openai_base_url"] == "http://custom:8080/v1"

    def test_config_roundtrip_openai_base_url(self):
        import tempfile
        from pathlib import Path

        from animus.config import AnimusConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = AnimusConfig(data_dir=Path(tmpdir))
            config.model.provider = "openai"
            config.model.name = "gpt-4o"
            config.model.openai_base_url = "http://myserver:8080/v1"
            config.save()

            loaded = AnimusConfig.load(config.config_file)
            assert loaded.model.provider == "openai"
            assert loaded.model.name == "gpt-4o"
            assert loaded.model.openai_base_url == "http://myserver:8080/v1"

    def test_ollama_is_default_provider(self):
        from animus.config import ModelConfig as CfgModelConfig

        cfg = CfgModelConfig()
        assert cfg.provider == "ollama"
        assert cfg.name == "llama3:8b"


# ---------------------------------------------------------------------------
# Anthropic native tool_use tests
# ---------------------------------------------------------------------------


class TestToolsToAnthropicFormat:
    """Test tools_to_anthropic_format conversion."""

    def test_basic_conversion(self):
        from animus.tools import Tool, ToolRegistry, tools_to_anthropic_format

        registry = ToolRegistry()
        registry.register(
            Tool(
                name="my_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {"x": {"type": "string"}}},
                handler=lambda p: None,
            )
        )
        result = tools_to_anthropic_format(registry)
        assert len(result) == 1
        assert result[0]["name"] == "my_tool"
        assert result[0]["description"] == "A test tool"
        assert "input_schema" in result[0]
        assert "parameters" not in result[0]
        assert result[0]["input_schema"]["properties"]["x"]["type"] == "string"


class TestThinkWithToolsAnthropicPath:
    """Test _think_with_tools_anthropic using mocked AnthropicModel."""

    def _make_anthropic_cognitive(self):
        """Create a CognitiveLayer with a real AnthropicModel (mocked calls)."""
        from animus.cognitive import AnthropicModel

        config = ModelConfig.anthropic("claude-sonnet-4-20250514")
        cog = CognitiveLayer(primary_config=config)
        assert isinstance(cog.primary, AnthropicModel)
        return cog

    def _make_tool_registry(self):
        from animus.tools import Tool, ToolRegistry, ToolResult

        registry = ToolRegistry()
        registry.register(
            Tool(
                name="get_time",
                description="Get current time",
                parameters={
                    "type": "object",
                    "properties": {"tz": {"type": "string"}},
                    "required": [],
                },
                handler=lambda p: ToolResult(tool_name="get_time", success=True, output="12:00 PM"),
            )
        )
        registry.register(
            Tool(
                name="dangerous_tool",
                description="Needs approval",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=lambda p: ToolResult(
                    tool_name="dangerous_tool", success=True, output="executed"
                ),
                requires_approval=True,
            )
        )
        return registry

    def _mock_text_response(self, text):
        """Build a mock Anthropic Message with only text content."""
        block = MagicMock()
        block.type = "text"
        block.text = text
        msg = MagicMock()
        msg.content = [block]
        return msg

    def _mock_tool_use_response(self, tool_name, tool_input, tool_use_id="tu_123"):
        """Build a mock Anthropic Message with a tool_use block."""
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Let me check."

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = tool_name
        tool_block.input = tool_input
        tool_block.id = tool_use_id

        msg = MagicMock()
        msg.content = [text_block, tool_block]
        return msg

    def test_no_tool_call_returns_text(self):
        """Text-only response from Anthropic — no tool calls."""
        cog = self._make_anthropic_cognitive()
        tools = self._make_tool_registry()

        cog.primary.generate_with_tools = MagicMock(
            return_value=self._mock_text_response("Just a simple answer.")
        )

        result = cog.think_with_tools("What is 2+2?", tools=tools)
        assert result == "Just a simple answer."

    def test_with_tool_call(self):
        """Model calls a tool, then returns final text."""
        cog = self._make_anthropic_cognitive()
        tools = self._make_tool_registry()

        # First call: tool_use, second call: text-only
        cog.primary.generate_with_tools = MagicMock(
            side_effect=[
                self._mock_tool_use_response("get_time", {"tz": "UTC"}),
                self._mock_text_response("It's 12:00 PM UTC."),
            ]
        )

        result = cog.think_with_tools("What time is it?", tools=tools)
        assert result == "It's 12:00 PM UTC."
        assert cog.primary.generate_with_tools.call_count == 2

    def test_approval_denied(self):
        """Tool requiring approval is denied by callback."""
        cog = self._make_anthropic_cognitive()
        tools = self._make_tool_registry()

        cog.primary.generate_with_tools = MagicMock(
            side_effect=[
                self._mock_tool_use_response("dangerous_tool", {}, "tu_456"),
                self._mock_text_response("OK, I won't do that."),
            ]
        )

        result = cog.think_with_tools(
            "Do the dangerous thing",
            tools=tools,
            approval_callback=lambda name, params: False,
        )
        assert result == "OK, I won't do that."

    def test_multi_tool(self):
        """Multiple tool_use blocks in one response."""
        cog = self._make_anthropic_cognitive()
        tools = self._make_tool_registry()

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Checking both."

        tool1 = MagicMock()
        tool1.type = "tool_use"
        tool1.name = "get_time"
        tool1.input = {"tz": "UTC"}
        tool1.id = "tu_1"

        tool2 = MagicMock()
        tool2.type = "tool_use"
        tool2.name = "get_time"
        tool2.input = {"tz": "EST"}
        tool2.id = "tu_2"

        multi_msg = MagicMock()
        multi_msg.content = [text_block, tool1, tool2]

        cog.primary.generate_with_tools = MagicMock(
            side_effect=[
                multi_msg,
                self._mock_text_response("UTC is 12:00 PM, EST is 7:00 AM."),
            ]
        )

        result = cog.think_with_tools("Times in UTC and EST?", tools=tools)
        assert "12:00 PM" in result
        assert cog.primary.generate_with_tools.call_count == 2

    def test_unknown_tool(self):
        """Model calls a tool that doesn't exist in registry."""
        cog = self._make_anthropic_cognitive()
        tools = self._make_tool_registry()

        cog.primary.generate_with_tools = MagicMock(
            side_effect=[
                self._mock_tool_use_response("nonexistent", {}, "tu_bad"),
                self._mock_text_response("Sorry, that tool failed."),
            ]
        )

        result = cog.think_with_tools("Use the missing tool", tools=tools)
        assert result == "Sorry, that tool failed."

    def test_markdown_path_still_works(self):
        """Regression: Ollama/Mock still use the markdown path."""
        tool_response = (
            'Let me check.\n```tool\n{"tool": "test_tool", "params": {"key": "val"}}\n```'
        )
        config = ModelConfig.mock(
            default_response=tool_response,
            response_map={"Tool results:": "Final answer from markdown path."},
        )
        cog = CognitiveLayer(primary_config=config)

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
        assert result == "Final answer from markdown path."
        mock_registry.execute.assert_called_once_with("test_tool", {"key": "val"})
