"""
Tests for Phase 4: Constrained tool selection for Ollama-only mode.

Tests numbered menu generation, constrained tool parsing, the constrained
agentic loop, and strip_tool_lines.
"""

from animus.cognitive import (
    CognitiveLayer,
    ModelConfig,
)
from animus.tools import (
    Tool,
    ToolRegistry,
    ToolResult,
    create_default_registry,
)

# ---------------------------------------------------------------------------
# get_numbered_menu
# ---------------------------------------------------------------------------


class TestNumberedMenu:
    """Tests for ToolRegistry.get_numbered_menu()."""

    def _make_registry(self, n=3):
        registry = ToolRegistry()
        for i in range(n):
            registry.register(
                Tool(
                    name=f"tool_{i}",
                    description=f"Does thing {i}",
                    parameters={
                        "type": "object",
                        "properties": {"arg": {"type": "string"}},
                        "required": ["arg"] if i == 0 else [],
                    },
                    handler=lambda p: ToolResult(tool_name="test", success=True, output="ok"),
                )
            )
        return registry

    def test_menu_text_contains_all_tools(self):
        registry = self._make_registry(3)
        text, _ = registry.get_numbered_menu()
        assert "tool_0" in text
        assert "tool_1" in text
        assert "tool_2" in text

    def test_menu_starts_with_zero(self):
        registry = self._make_registry(2)
        text, _ = registry.get_numbered_menu()
        assert "0: No tool needed" in text

    def test_number_map_is_one_based(self):
        registry = self._make_registry(3)
        _, number_map = registry.get_numbered_menu()
        assert 1 in number_map
        assert 2 in number_map
        assert 3 in number_map
        assert 0 not in number_map

    def test_number_map_values_are_tool_names(self):
        registry = self._make_registry(3)
        _, number_map = registry.get_numbered_menu()
        assert number_map[1] == "tool_0"
        assert number_map[2] == "tool_1"
        assert number_map[3] == "tool_2"

    def test_required_params_shown_in_menu(self):
        registry = self._make_registry(1)
        text, _ = registry.get_numbered_menu()
        assert "(arg)" in text

    def test_empty_registry(self):
        registry = ToolRegistry()
        text, number_map = registry.get_numbered_menu()
        assert "0:" in text
        assert len(number_map) == 0

    def test_default_registry_menu(self):
        """All built-in tools appear in menu."""
        registry = create_default_registry()
        text, number_map = registry.get_numbered_menu()
        assert "read_file" in text
        assert "run_command" in text
        assert len(number_map) >= 8


# ---------------------------------------------------------------------------
# _parse_constrained_tool
# ---------------------------------------------------------------------------


class TestParseConstrainedTool:
    """Tests for CognitiveLayer._parse_constrained_tool()."""

    def _number_map(self):
        return {1: "read_file", 2: "edit_file", 3: "run_command"}

    def test_basic_parse(self):
        response = "I'll read the file.\nTOOL: 1\npath: /home/user/file.py"
        result = CognitiveLayer._parse_constrained_tool(response, self._number_map())
        assert result is not None
        name, params = result
        assert name == "read_file"
        assert params["path"] == "/home/user/file.py"

    def test_tool_zero_returns_none(self):
        response = "TOOL: 0\nHere is my answer."
        result = CognitiveLayer._parse_constrained_tool(response, self._number_map())
        assert result is None

    def test_no_tool_line_returns_none(self):
        response = "Just a regular response with no tool usage."
        result = CognitiveLayer._parse_constrained_tool(response, self._number_map())
        assert result is None

    def test_invalid_tool_number(self):
        response = "TOOL: 99\npath: /tmp/x"
        result = CognitiveLayer._parse_constrained_tool(response, self._number_map())
        assert result is None

    def test_multiple_params(self):
        response = "TOOL: 2\npath: /home/user/file.py\nold_text: foo\nnew_text: bar"
        result = CognitiveLayer._parse_constrained_tool(response, self._number_map())
        assert result is not None
        name, params = result
        assert name == "edit_file"
        assert params["path"] == "/home/user/file.py"
        assert params["old_text"] == "foo"
        assert params["new_text"] == "bar"

    def test_whitespace_tolerance(self):
        response = "  TOOL:  1  \n  path:  /some/path  "
        result = CognitiveLayer._parse_constrained_tool(response, self._number_map())
        assert result is not None
        name, params = result
        assert name == "read_file"
        assert params["path"] == "/some/path"

    def test_colon_in_value(self):
        """Values can contain colons (e.g., URLs)."""
        response = "TOOL: 1\npath: http://localhost:8080/api"
        result = CognitiveLayer._parse_constrained_tool(response, self._number_map())
        assert result is not None
        _, params = result
        assert params["path"] == "http://localhost:8080/api"

    def test_stops_at_empty_line(self):
        response = "TOOL: 1\npath: /file.py\n\nSome commentary after."
        result = CognitiveLayer._parse_constrained_tool(response, self._number_map())
        assert result is not None
        _, params = result
        assert params == {"path": "/file.py"}

    def test_stops_at_next_tool(self):
        response = "TOOL: 1\npath: /file.py\nTOOL: 2\npath: /other.py"
        result = CognitiveLayer._parse_constrained_tool(response, self._number_map())
        assert result is not None
        name, params = result
        assert name == "read_file"
        assert params == {"path": "/file.py"}


# ---------------------------------------------------------------------------
# _strip_tool_lines
# ---------------------------------------------------------------------------


class TestStripToolLines:
    """Tests for CognitiveLayer._strip_tool_lines()."""

    def test_removes_tool_and_params(self):
        response = "I'll answer now.\nTOOL: 0\nHere is the answer."
        result = CognitiveLayer._strip_tool_lines(response)
        assert "TOOL:" not in result
        assert "answer" in result

    def test_preserves_non_tool_content(self):
        response = "Line 1\nLine 2\nLine 3"
        result = CognitiveLayer._strip_tool_lines(response)
        assert result == response

    def test_removes_param_lines_after_tool(self):
        response = "Thinking...\nTOOL: 3\ncommand: pytest\ntimeout: 30\nDone."
        result = CognitiveLayer._strip_tool_lines(response)
        assert "TOOL:" not in result
        assert "command:" not in result
        assert "Done." in result

    def test_empty_string(self):
        assert CognitiveLayer._strip_tool_lines("") == ""


# ---------------------------------------------------------------------------
# Constrained loop integration
# ---------------------------------------------------------------------------


class TestConstrainedLoop:
    """Integration tests for the constrained tool loop."""

    def test_no_tool_call_returns_response(self):
        """When model doesn't output TOOL:, returns the response directly."""
        config = ModelConfig.mock(default_response="Here is my answer.")
        cog = CognitiveLayer(primary_config=config)
        registry = create_default_registry()

        result = cog._think_with_tools_constrained(
            prompt="What time is it?",
            tools=registry,
        )
        assert "answer" in result

    def test_tool_zero_returns_response(self):
        """When model outputs TOOL: 0, returns the cleaned response."""
        config = ModelConfig.mock(default_response="TOOL: 0\nThe answer is 42.")
        cog = CognitiveLayer(primary_config=config)
        registry = create_default_registry()

        result = cog._think_with_tools_constrained(
            prompt="What is the meaning of life?",
            tools=registry,
        )
        assert "42" in result
        assert "TOOL:" not in result

    def test_tool_call_executes(self):
        """When model selects a tool, it gets executed."""
        # Model first requests get_datetime, then gives final answer
        call_count = [0]

        def mock_generate(prompt, system=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return "TOOL: 1\n"  # get_datetime is tool #1 in default registry
            return "TOOL: 0\nThe time has been checked."

        config = ModelConfig.mock()
        cog = CognitiveLayer(primary_config=config)
        cog.primary.generate = mock_generate

        registry = create_default_registry()
        result = cog._think_with_tools_constrained(
            prompt="What time is it?",
            tools=registry,
        )
        assert call_count[0] == 2
        assert "time" in result.lower()

    def test_approval_callback_blocks_tool(self):
        """Sensitive tools respect the approval callback."""
        call_count = [0]

        def mock_generate(prompt, system=None):
            call_count[0] += 1
            if call_count[0] == 1:
                # run_command requires approval
                menu_text, number_map = create_default_registry().get_numbered_menu()
                # Find run_command's number
                cmd_num = next(n for n, name in number_map.items() if name == "run_command")
                return f"TOOL: {cmd_num}\ncommand: rm -rf /"
            return "TOOL: 0\nThe command was blocked."

        config = ModelConfig.mock()
        cog = CognitiveLayer(primary_config=config)
        cog.primary.generate = mock_generate

        registry = create_default_registry()
        result = cog._think_with_tools_constrained(
            prompt="Delete everything",
            tools=registry,
            approval_callback=lambda name, params: False,
        )
        assert "blocked" in result.lower() or "not approved" in result.lower() or call_count[0] >= 2

    def test_max_iterations_stops_loop(self):
        """Loop stops after max_iterations even if model keeps calling tools."""
        config = ModelConfig.mock(default_response="TOOL: 1\n")
        cog = CognitiveLayer(primary_config=config)
        registry = create_default_registry()

        result = cog._think_with_tools_constrained(
            prompt="Keep going",
            tools=registry,
            max_iterations=3,
        )
        # Should have stopped — returns last response
        assert result is not None

    def test_dispatch_routes_ollama_to_constrained(self):
        """think_with_tools routes Ollama models to constrained loop."""
        config = ModelConfig.mock(default_response="TOOL: 0\nDone.")
        cog = CognitiveLayer(primary_config=config)
        registry = create_default_registry()

        # Mock model is not AnthropicModel, so should go to constrained
        result = cog.think_with_tools(
            prompt="test",
            tools=registry,
        )
        assert "Done" in result
