"""Tests for ForgeToolRegistry and tool-equipped agent execution."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from animus_forge.tools.registry import (
    MAX_TOOL_OUTPUT_CHARS,
    ForgeToolRegistry,
    ToolDefinition,
)

# --- ForgeToolRegistry unit tests ---


class TestToolDefinition:
    """Tests for ToolDefinition data class."""

    def test_to_anthropic_format(self):
        tool = ToolDefinition(
            name="read_file",
            description="Read a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            handler=lambda args: "",
        )
        result = tool.to_anthropic()
        assert result["name"] == "read_file"
        assert result["description"] == "Read a file"
        assert result["input_schema"]["type"] == "object"

    def test_to_ollama_format(self):
        tool = ToolDefinition(
            name="search_code",
            description="Search code",
            parameters={"type": "object", "properties": {}},
            handler=lambda args: "",
        )
        result = tool.to_ollama()
        assert result["type"] == "function"
        assert result["function"]["name"] == "search_code"
        assert result["function"]["description"] == "Search code"


class TestForgeToolRegistry:
    """Tests for the tool registry."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create a test file
        test_file = Path(self.tmpdir) / "test.py"
        test_file.write_text("print('hello')\n")
        self.registry = ForgeToolRegistry(project_root=self.tmpdir)

    def test_registers_default_tools(self):
        names = {t.name for t in self.registry.tools}
        assert "read_file" in names
        assert "list_files" in names
        assert "search_code" in names
        assert "get_project_structure" in names
        assert "write_file" in names
        # shell not enabled by default
        assert "run_command" not in names

    def test_registers_shell_tool_when_enabled(self):
        reg = ForgeToolRegistry(project_root=self.tmpdir, enable_shell=True)
        names = {t.name for t in reg.tools}
        assert "run_command" in names

    def test_get_tool_by_name(self):
        tool = self.registry.get("read_file")
        assert tool is not None
        assert tool.name == "read_file"

    def test_get_unknown_tool(self):
        assert self.registry.get("nonexistent") is None

    def test_to_anthropic_tools(self):
        tools = self.registry.to_anthropic_tools()
        assert len(tools) >= 5
        assert all("name" in t and "input_schema" in t for t in tools)

    def test_to_ollama_tools(self):
        tools = self.registry.to_ollama_tools()
        assert len(tools) >= 5
        assert all(t["type"] == "function" for t in tools)

    def test_execute_read_file(self):
        result = self.registry.execute("read_file", {"path": "test.py"})
        assert "print('hello')" in result
        assert "test.py" in result

    def test_execute_list_files(self):
        result = self.registry.execute("list_files", {"path": "."})
        assert "test.py" in result

    def test_execute_search_code(self):
        result = self.registry.execute("search_code", {"pattern": "hello"})
        assert "hello" in result
        assert "test.py" in result

    def test_execute_get_structure(self):
        result = self.registry.execute("get_project_structure", {})
        assert "test.py" in result

    def test_execute_write_file(self):
        result = self.registry.execute("write_file", {
            "path": "new_file.txt",
            "content": "new content",
        })
        assert "Written" in result
        assert (Path(self.tmpdir) / "new_file.txt").read_text() == "new content"

    def test_execute_write_file_creates_subdirs(self):
        result = self.registry.execute("write_file", {
            "path": "sub/dir/file.txt",
            "content": "deep content",
        })
        assert "Written" in result
        assert (Path(self.tmpdir) / "sub/dir/file.txt").read_text() == "deep content"

    def test_execute_unknown_tool(self):
        result = self.registry.execute("nonexistent", {})
        assert "Unknown tool" in result

    def test_execute_truncates_long_output(self):
        # Create a large file
        big_file = Path(self.tmpdir) / "big.txt"
        big_file.write_text("x" * (MAX_TOOL_OUTPUT_CHARS + 1000))
        result = self.registry.execute("read_file", {"path": "big.txt"})
        assert "truncated" in result
        assert len(result) <= MAX_TOOL_OUTPUT_CHARS + 50  # +margin for "truncated" suffix

    def test_execute_handles_errors_gracefully(self):
        result = self.registry.execute("read_file", {"path": "nonexistent.py"})
        assert "Error" in result

    def test_register_custom_tool(self):
        custom = ToolDefinition(
            name="custom_tool",
            description="A custom tool",
            parameters={"type": "object", "properties": {}},
            handler=lambda args: "custom result",
        )
        self.registry.register(custom)
        assert self.registry.get("custom_tool") is not None
        assert self.registry.execute("custom_tool", {}) == "custom result"


class TestForgeToolRegistryShell:
    """Tests for shell command tool."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.registry = ForgeToolRegistry(
            project_root=self.tmpdir,
            enable_shell=True,
        )

    def test_run_allowed_command(self):
        result = self.registry.execute("run_command", {"command": "ls"})
        assert "Error" not in result or "not in allowed" not in result

    def test_run_disallowed_command(self):
        result = self.registry.execute("run_command", {"command": "rm -rf /"})
        assert "not in allowed" in result

    def test_run_custom_allowed_commands(self):
        reg = ForgeToolRegistry(
            project_root=self.tmpdir,
            enable_shell=True,
            allowed_commands=["echo"],
        )
        result = reg.execute("run_command", {"command": "echo hello"})
        assert "hello" in result

    def test_run_empty_command(self):
        result = self.registry.execute("run_command", {"command": ""})
        assert "Empty command" in result

    def test_run_command_timeout(self):
        result = self.registry.execute("run_command", {
            "command": "python3 -c \"import time; time.sleep(10)\"",
            "timeout": 1,
        })
        assert "timed out" in result


# --- AgentProvider tool loop tests ---


class TestAgentProviderToolLoop:
    """Tests for the iterative tool loop in AgentProvider."""

    def _make_provider(self, responses):
        """Create a mock AgentProvider with predefined responses."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        mock_provider = MagicMock()
        mock_provider._initialized = True
        mock_provider.name = "test"

        response_iter = iter(responses)

        async def mock_complete_async(request):
            return next(response_iter)

        mock_provider.complete_async = mock_complete_async
        return AgentProvider(mock_provider)

    def test_complete_with_tools_no_tool_calls(self):
        """Model responds with text only — no tool loop."""
        from animus_forge.providers.base import CompletionResponse

        agent = self._make_provider([
            CompletionResponse(content="Just text", model="test", provider="test"),
        ])
        registry = ForgeToolRegistry(project_root=tempfile.mkdtemp())

        result = asyncio.run(agent.complete_with_tools(
            messages=[{"role": "user", "content": "hello"}],
            tool_registry=registry,
        ))
        assert result == "Just text"

    def test_complete_with_tools_single_iteration(self):
        """Model calls a tool, then responds with text."""
        from animus_forge.providers.base import CompletionResponse, ToolCall

        tmpdir = tempfile.mkdtemp()
        Path(tmpdir, "main.py").write_text("x = 1\n")

        agent = self._make_provider([
            # First call: tool use
            CompletionResponse(
                content="Let me check",
                model="test",
                provider="test",
                tool_calls=[
                    ToolCall(id="tc1", name="read_file", arguments={"path": "main.py"}),
                ],
            ),
            # Second call: text response
            CompletionResponse(
                content="The file contains x = 1",
                model="test",
                provider="test",
            ),
        ])
        registry = ForgeToolRegistry(project_root=tmpdir)

        result = asyncio.run(agent.complete_with_tools(
            messages=[{"role": "user", "content": "read main.py"}],
            tool_registry=registry,
        ))
        assert "x = 1" in result

    def test_complete_with_tools_multiple_iterations(self):
        """Model calls tools across 2 iterations."""
        from animus_forge.providers.base import CompletionResponse, ToolCall

        tmpdir = tempfile.mkdtemp()
        Path(tmpdir, "a.py").write_text("a = 1\n")
        Path(tmpdir, "b.py").write_text("b = 2\n")

        agent = self._make_provider([
            CompletionResponse(
                content="", model="test", provider="test",
                tool_calls=[ToolCall(id="t1", name="read_file", arguments={"path": "a.py"})],
            ),
            CompletionResponse(
                content="", model="test", provider="test",
                tool_calls=[ToolCall(id="t2", name="read_file", arguments={"path": "b.py"})],
            ),
            CompletionResponse(
                content="a=1, b=2", model="test", provider="test",
            ),
        ])
        registry = ForgeToolRegistry(project_root=tmpdir)

        result = asyncio.run(agent.complete_with_tools(
            messages=[{"role": "user", "content": "read both files"}],
            tool_registry=registry,
        ))
        assert "a=1" in result

    def test_complete_with_tools_max_iterations(self):
        """Hits max iterations and forces a final text response."""
        from animus_forge.providers.base import CompletionResponse, ToolCall

        tmpdir = tempfile.mkdtemp()

        # 3 tool responses (one per iteration) + 1 forced final text response
        responses = [
            CompletionResponse(
                content="", model="test", provider="test",
                tool_calls=[ToolCall(id=f"t{i}", name="list_files", arguments={})],
            )
            for i in range(3)
        ]
        responses.append(
            CompletionResponse(content="forced final", model="test", provider="test")
        )

        agent = self._make_provider(responses)
        registry = ForgeToolRegistry(project_root=tmpdir)

        result = asyncio.run(agent.complete_with_tools(
            messages=[{"role": "user", "content": "loop forever"}],
            tool_registry=registry,
            max_iterations=3,
        ))
        assert "forced final" in result

    def test_complete_with_tools_progress_callback(self):
        """Progress callback is called during tool iterations."""
        from animus_forge.providers.base import CompletionResponse, ToolCall

        tmpdir = tempfile.mkdtemp()
        progress_calls = []

        def cb(stage, detail=""):
            progress_calls.append((stage, detail))

        agent = self._make_provider([
            CompletionResponse(
                content="", model="test", provider="test",
                tool_calls=[ToolCall(id="t1", name="list_files", arguments={})],
            ),
            CompletionResponse(content="done", model="test", provider="test"),
        ])
        registry = ForgeToolRegistry(project_root=tmpdir)

        asyncio.run(agent.complete_with_tools(
            messages=[{"role": "user", "content": "test"}],
            tool_registry=registry,
            progress_callback=cb,
        ))
        assert any("tools" in call[0] for call in progress_calls)

    def test_complete_with_tools_unknown_tool_handled(self):
        """Tool not in registry returns error string, doesn't crash."""
        from animus_forge.providers.base import CompletionResponse, ToolCall

        agent = self._make_provider([
            CompletionResponse(
                content="", model="test", provider="test",
                tool_calls=[ToolCall(id="t1", name="nonexistent_tool", arguments={})],
            ),
            CompletionResponse(content="handled gracefully", model="test", provider="test"),
        ])
        registry = ForgeToolRegistry(project_root=tempfile.mkdtemp())

        result = asyncio.run(agent.complete_with_tools(
            messages=[{"role": "user", "content": "test"}],
            tool_registry=registry,
        ))
        assert "handled gracefully" in result


# --- SupervisorAgent tool-equipped sub-agents tests ---


class TestSupervisorToolEquipped:
    """Tests for tool-equipped agent delegation in SupervisorAgent."""

    def _make_supervisor(self, responses, tool_registry=None):
        """Create a SupervisorAgent with mock provider."""
        from animus_forge.agents.provider_wrapper import AgentProvider
        from animus_forge.agents.supervisor import SupervisorAgent

        mock_provider = MagicMock()
        mock_provider._initialized = True
        mock_provider.name = "test"

        response_iter = iter(responses)

        async def mock_complete_async(request):
            return next(response_iter)

        mock_provider.complete_async = mock_complete_async
        agent_provider = AgentProvider(mock_provider)

        return SupervisorAgent(
            provider=agent_provider,
            tool_registry=tool_registry,
        )

    def test_builder_gets_tools(self):
        """Builder role should use complete_with_tools when registry is set."""
        from animus_forge.providers.base import CompletionResponse

        tmpdir = tempfile.mkdtemp()
        registry = ForgeToolRegistry(project_root=tmpdir)

        supervisor = self._make_supervisor(
            [CompletionResponse(content="built it", model="test", provider="test")],
            tool_registry=registry,
        )

        with patch.object(supervisor.provider, "complete_with_tools", new_callable=AsyncMock) as mock_tools:
            mock_tools.return_value = "built with tools"
            result = asyncio.run(supervisor._run_agent("builder", "build something", []))
            mock_tools.assert_called_once()
            assert result == "built with tools"

    def test_planner_does_not_get_tools(self):
        """Planner role should use plain complete, not tools."""
        from animus_forge.providers.base import CompletionResponse

        registry = ForgeToolRegistry(project_root=tempfile.mkdtemp())
        supervisor = self._make_supervisor(
            [CompletionResponse(content="planned it", model="test", provider="test")],
            tool_registry=registry,
        )

        with patch.object(supervisor.provider, "complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "planned"
            asyncio.run(supervisor._run_agent("planner", "plan something", []))
            mock_complete.assert_called_once()

    def test_tester_gets_tools(self):
        """Tester role should use complete_with_tools."""
        from animus_forge.providers.base import CompletionResponse

        registry = ForgeToolRegistry(project_root=tempfile.mkdtemp())
        supervisor = self._make_supervisor(
            [CompletionResponse(content="tested", model="test", provider="test")],
            tool_registry=registry,
        )

        with patch.object(supervisor.provider, "complete_with_tools", new_callable=AsyncMock) as mock_tools:
            mock_tools.return_value = "tested with tools"
            asyncio.run(supervisor._run_agent("tester", "test something", []))
            mock_tools.assert_called_once()

    def test_no_registry_falls_back_to_text(self):
        """Without a tool registry, all agents use text-only completion."""
        from animus_forge.providers.base import CompletionResponse

        supervisor = self._make_supervisor(
            [CompletionResponse(content="text only", model="test", provider="test")],
            tool_registry=None,
        )

        result = asyncio.run(supervisor._run_agent("builder", "build", []))
        assert result == "text only"

    def test_tool_equipped_roles_constant(self):
        """Verify which roles are tool-equipped."""
        from animus_forge.agents.supervisor import SupervisorAgent

        assert "builder" in SupervisorAgent.TOOL_EQUIPPED_ROLES
        assert "tester" in SupervisorAgent.TOOL_EQUIPPED_ROLES
        assert "reviewer" in SupervisorAgent.TOOL_EQUIPPED_ROLES
        assert "analyst" in SupervisorAgent.TOOL_EQUIPPED_ROLES
        assert "planner" not in SupervisorAgent.TOOL_EQUIPPED_ROLES
        assert "documenter" not in SupervisorAgent.TOOL_EQUIPPED_ROLES


# --- Provider tool_calls extraction tests ---


class TestProviderToolCalls:
    """Tests for tool_calls extraction in providers."""

    def test_completion_response_has_tool_calls_field(self):
        from animus_forge.providers.base import CompletionResponse

        resp = CompletionResponse(content="", model="test", provider="test")
        assert resp.tool_calls == []

    def test_tool_call_dataclass(self):
        from animus_forge.providers.base import ToolCall

        tc = ToolCall(id="tc_1", name="read_file", arguments={"path": "test.py"})
        assert tc.id == "tc_1"
        assert tc.name == "read_file"
        assert tc.arguments == {"path": "test.py"}

    def test_completion_request_tools_field(self):
        from animus_forge.providers.base import CompletionRequest

        req = CompletionRequest(prompt="test", tools=[{"name": "t1"}])
        assert req.tools == [{"name": "t1"}]

    def test_completion_request_tools_default_none(self):
        from animus_forge.providers.base import CompletionRequest

        req = CompletionRequest(prompt="test")
        assert req.tools is None

    def test_anthropic_extract_response_text_only(self):
        from animus_forge.providers.anthropic_provider import AnthropicProvider

        mock_response = MagicMock()
        block = MagicMock()
        block.text = "hello"
        block.type = "text"
        mock_response.content = [block]

        content, tool_calls = AnthropicProvider._extract_response(mock_response)
        assert content == "hello"
        assert tool_calls == []

    def test_anthropic_extract_response_with_tool_use(self):
        from animus_forge.providers.anthropic_provider import AnthropicProvider

        text_block = MagicMock()
        text_block.text = "I'll read the file"
        text_block.type = "text"

        tool_block = MagicMock(spec=[])
        tool_block.type = "tool_use"
        tool_block.id = "toolu_123"
        tool_block.name = "read_file"
        tool_block.input = {"path": "test.py"}
        # Remove 'text' attr so hasattr(block, "text") is False
        del tool_block.text

        mock_response = MagicMock()
        mock_response.content = [text_block, tool_block]

        content, tool_calls = AnthropicProvider._extract_response(mock_response)
        assert "I'll read the file" in content
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "read_file"
        assert tool_calls[0].id == "toolu_123"

    def test_ollama_extract_tool_calls_empty(self):
        from animus_forge.providers.ollama_provider import OllamaProvider

        calls = OllamaProvider._extract_tool_calls({"content": "hello"})
        assert calls == []

    def test_ollama_extract_tool_calls(self):
        from animus_forge.providers.ollama_provider import OllamaProvider

        message = {
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "read_file",
                        "arguments": {"path": "test.py"},
                    }
                },
                {
                    "function": {
                        "name": "list_files",
                        "arguments": {"path": "."},
                    }
                },
            ],
        }
        calls = OllamaProvider._extract_tool_calls(message)
        assert len(calls) == 2
        assert calls[0].name == "read_file"
        assert calls[1].name == "list_files"
        assert calls[0].id == "ollama_tool_0"


# --- AgentProvider message building tests ---


class TestAgentProviderMessageBuilding:
    """Tests for tool message construction helpers."""

    def test_split_system(self):
        from animus_forge.agents.provider_wrapper import AgentProvider

        messages = [
            {"role": "system", "content": "sys1"},
            {"role": "system", "content": "sys2"},
            {"role": "user", "content": "hello"},
        ]
        system, filtered = AgentProvider._split_system(messages)
        assert "sys1" in system
        assert "sys2" in system
        assert len(filtered) == 1
        assert filtered[0]["role"] == "user"

    def test_build_assistant_tool_message_anthropic(self):
        from animus_forge.agents.provider_wrapper import AgentProvider
        from animus_forge.providers.base import CompletionResponse, ToolCall

        response = CompletionResponse(
            content="thinking",
            model="test",
            provider="test",
            tool_calls=[ToolCall(id="tc1", name="read_file", arguments={"path": "x"})],
        )
        msg = AgentProvider._build_assistant_tool_message(response, "anthropic")
        assert msg["role"] == "assistant"
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][1]["type"] == "tool_use"

    def test_build_assistant_tool_message_ollama(self):
        from animus_forge.agents.provider_wrapper import AgentProvider
        from animus_forge.providers.base import CompletionResponse, ToolCall

        response = CompletionResponse(
            content="thinking",
            model="test",
            provider="test",
            tool_calls=[ToolCall(id="tc1", name="list_files", arguments={})],
        )
        msg = AgentProvider._build_assistant_tool_message(response, "ollama")
        assert msg["role"] == "assistant"
        assert "tool_calls" in msg
        assert msg["tool_calls"][0]["function"]["name"] == "list_files"

    def test_build_tool_result_message_anthropic(self):
        from animus_forge.agents.provider_wrapper import AgentProvider
        from animus_forge.providers.base import ToolCall

        tc = ToolCall(id="tc1", name="read_file", arguments={})
        msg = AgentProvider._build_tool_result_message(tc, "file content", "anthropic")
        assert msg["role"] == "user"
        assert msg["content"][0]["type"] == "tool_result"
        assert msg["content"][0]["tool_use_id"] == "tc1"

    def test_build_tool_result_message_ollama(self):
        from animus_forge.agents.provider_wrapper import AgentProvider
        from animus_forge.providers.base import ToolCall

        tc = ToolCall(id="tc1", name="read_file", arguments={})
        msg = AgentProvider._build_tool_result_message(tc, "file content", "ollama")
        assert msg["role"] == "tool"
        assert msg["content"] == "file content"


# --- Model routing and text-based fallback tests ---


class TestModelRouting:
    """Tests for dual-model routing (tool model vs reasoning model)."""

    def test_resolve_tool_model_anthropic_returns_none(self):
        """Anthropic always supports tools — no model switch needed."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        mock_provider = MagicMock()
        mock_provider._initialized = True
        mock_provider.name = "anthropic"

        agent = AgentProvider(mock_provider)
        assert agent._resolve_tool_model() is None

    def test_resolve_tool_model_ollama_with_tool_capable_model(self):
        """Ollama with qwen2.5:14b — already supports tools."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        mock_provider = MagicMock()
        mock_provider._initialized = True
        mock_provider.name = "ollama"
        mock_provider.default_model = "qwen2.5:14b"

        agent = AgentProvider(mock_provider)
        assert agent._resolve_tool_model() is None

    def test_resolve_tool_model_ollama_no_tool_support_returns_none(self):
        """Ollama with deepseek-coder-v2 and no explicit tool_model — returns None (text fallback)."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        mock_provider = MagicMock()
        mock_provider._initialized = True
        mock_provider.name = "ollama"
        mock_provider.default_model = "deepseek-coder-v2"

        agent = AgentProvider(mock_provider)
        assert agent._resolve_tool_model() is None
        assert not agent.supports_native_tools

    def test_resolve_tool_model_explicit_override(self):
        """Explicit tool_model overrides everything."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        mock_provider = MagicMock()
        mock_provider._initialized = True
        mock_provider.name = "ollama"
        mock_provider.default_model = "qwen2.5:14b"

        agent = AgentProvider(mock_provider, tool_model="llama3.1:8b")
        assert agent._resolve_tool_model() == "llama3.1:8b"

    def test_tool_model_passed_in_request(self):
        """Tool model override is passed in CompletionRequest.model."""
        from animus_forge.agents.provider_wrapper import AgentProvider
        from animus_forge.providers.base import CompletionResponse

        mock_provider = MagicMock()
        mock_provider._initialized = True
        mock_provider.name = "ollama"
        mock_provider.default_model = "deepseek-coder-v2"

        captured_requests = []

        async def capture_complete(request):
            captured_requests.append(request)
            return CompletionResponse(content="done", model="test", provider="test")

        mock_provider.complete_async = capture_complete

        agent = AgentProvider(mock_provider, tool_model="qwen2.5:14b")
        registry = ForgeToolRegistry(project_root=tempfile.mkdtemp())

        asyncio.run(agent.complete_with_tools(
            messages=[{"role": "user", "content": "test"}],
            tool_registry=registry,
        ))
        assert captured_requests[0].model == "qwen2.5:14b"

    def test_create_agent_provider_ollama_sets_tool_model(self):
        """create_agent_provider for ollama sets tool_model."""
        from animus_forge.agents.provider_wrapper import DEFAULT_TOOL_MODEL, create_agent_provider

        with patch("animus_forge.providers.ollama_provider.OllamaProvider") as MockOllama:
            mock_instance = MagicMock()
            mock_instance._initialized = True
            MockOllama.return_value = mock_instance

            provider = create_agent_provider("ollama")
            assert provider._tool_model == DEFAULT_TOOL_MODEL


class TestTextToolFallback:
    """Tests for text-based tool parsing fallback."""

    def test_build_text_tool_prompt(self):
        """Text prompt includes tool descriptions."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        registry = ForgeToolRegistry(project_root=tempfile.mkdtemp())
        prompt = AgentProvider._build_text_tool_prompt(registry)
        assert "read_file" in prompt
        assert "list_files" in prompt
        assert "tool_call" in prompt

    def test_parse_tool_call_block(self):
        """Parse ```tool_call JSON block."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        registry = ForgeToolRegistry(project_root=tempfile.mkdtemp())
        text = '''Let me read the file.
```tool_call
{"tool": "read_file", "arguments": {"path": "test.py"}}
```'''
        calls = AgentProvider._parse_text_tool_calls(text, registry)
        assert len(calls) == 1
        assert calls[0][0] == "read_file"
        assert calls[0][1] == {"path": "test.py"}

    def test_parse_json_block_with_tool(self):
        """Parse ```json block containing tool call."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        registry = ForgeToolRegistry(project_root=tempfile.mkdtemp())
        text = '''I'll search for it.
```json
{"tool": "search_code", "arguments": {"pattern": "def main"}}
```'''
        calls = AgentProvider._parse_text_tool_calls(text, registry)
        assert len(calls) == 1
        assert calls[0][0] == "search_code"

    def test_parse_bare_json_tool_call(self):
        """Parse bare JSON object with tool key."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        registry = ForgeToolRegistry(project_root=tempfile.mkdtemp())
        text = 'I will use {"tool": "list_files", "arguments": {"path": "."}} to check.'
        calls = AgentProvider._parse_text_tool_calls(text, registry)
        assert len(calls) == 1
        assert calls[0][0] == "list_files"

    def test_parse_no_tool_calls(self):
        """No tool calls in text — returns empty list."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        registry = ForgeToolRegistry(project_root=tempfile.mkdtemp())
        text = "Just a normal response without any tools."
        calls = AgentProvider._parse_text_tool_calls(text, registry)
        assert calls == []

    def test_parse_unknown_tool_ignored(self):
        """Tool not in registry is ignored."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        registry = ForgeToolRegistry(project_root=tempfile.mkdtemp())
        text = '```tool_call\n{"tool": "hack_mainframe", "arguments": {}}\n```'
        calls = AgentProvider._parse_text_tool_calls(text, registry)
        assert calls == []

    def test_parse_malformed_json_ignored(self):
        """Malformed JSON is silently skipped."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        registry = ForgeToolRegistry(project_root=tempfile.mkdtemp())
        text = '```tool_call\n{not valid json}\n```'
        calls = AgentProvider._parse_text_tool_calls(text, registry)
        assert calls == []

    def test_text_tool_loop_executes(self):
        """Full text-based tool loop: parse, execute, feed back."""
        from animus_forge.agents.provider_wrapper import AgentProvider
        from animus_forge.providers.base import CompletionResponse

        tmpdir = tempfile.mkdtemp()
        Path(tmpdir, "main.py").write_text("x = 42\n")

        mock_provider = MagicMock()
        mock_provider._initialized = True
        mock_provider.name = "ollama"
        mock_provider.default_model = "codellama"  # No tool support

        responses = iter([
            # First: model outputs a text tool call
            CompletionResponse(
                content='```tool_call\n{"tool": "read_file", "arguments": {"path": "main.py"}}\n```',
                model="codellama",
                provider="ollama",
            ),
            # Second: model responds with final answer
            CompletionResponse(
                content="The file contains x = 42",
                model="codellama",
                provider="ollama",
            ),
        ])

        async def mock_complete(request):
            return next(responses)

        mock_provider.complete_async = mock_complete
        agent = AgentProvider(mock_provider)

        registry = ForgeToolRegistry(project_root=tmpdir)

        result = asyncio.run(agent.complete_with_tools(
            messages=[{"role": "user", "content": "read main.py"}],
            tool_registry=registry,
        ))
        assert "x = 42" in result


# --- Tool audit logging tests ---


class TestToolAuditLogging:
    """Tests for structured audit log emission on tool execution."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        Path(self.tmpdir, "test.py").write_text("x = 1\n")
        self.registry = ForgeToolRegistry(project_root=self.tmpdir)

    def test_audit_emitted_on_success(self):
        """Successful tool execution emits an audit log entry."""
        with patch("animus_forge.tools.registry.audit_logger") as mock_audit:
            self.registry.execute("read_file", {"path": "test.py"})
            mock_audit.info.assert_called_once()
            entry = json.loads(mock_audit.info.call_args[0][0])
            assert entry["event"] == "tool_execution"
            assert entry["tool"] == "read_file"
            assert entry["success"] is True
            assert "duration_ms" in entry

    def test_audit_emitted_on_failure(self):
        """Failed tool execution emits audit with error."""
        with patch("animus_forge.tools.registry.audit_logger") as mock_audit:
            self.registry.execute("read_file", {"path": "nonexistent.py"})
            mock_audit.info.assert_called_once()
            entry = json.loads(mock_audit.info.call_args[0][0])
            assert entry["success"] is False
            assert "error" in entry

    def test_audit_emitted_on_unknown_tool(self):
        """Unknown tool emits audit with error."""
        with patch("animus_forge.tools.registry.audit_logger") as mock_audit:
            self.registry.execute("fake_tool", {})
            mock_audit.info.assert_called_once()
            entry = json.loads(mock_audit.info.call_args[0][0])
            assert entry["success"] is False
            assert entry["error"] == "unknown_tool"

    def test_audit_includes_agent_id(self):
        """Agent ID is recorded in audit entries."""
        with patch("animus_forge.tools.registry.audit_logger") as mock_audit:
            self.registry.execute("list_files", {}, agent_id="builder")
            entry = json.loads(mock_audit.info.call_args[0][0])
            assert entry["agent_id"] == "builder"

    def test_audit_sanitizes_content(self):
        """File content is not logged — only size."""
        with patch("animus_forge.tools.registry.audit_logger") as mock_audit:
            self.registry.execute("write_file", {
                "path": "out.txt",
                "content": "secret data here",
            })
            entry = json.loads(mock_audit.info.call_args[0][0])
            assert "secret data" not in json.dumps(entry)
            assert "(16 chars)" in json.dumps(entry)

    def test_audit_never_breaks_execution(self):
        """Audit logger error doesn't break tool execution."""
        with patch("animus_forge.tools.registry.audit_logger") as mock_audit:
            mock_audit.info.side_effect = RuntimeError("logger broken")
            result = self.registry.execute("list_files", {})
            assert "Error" not in result


# --- Write file approval gate tests ---


class TestWriteApprovalGate:
    """Tests for the write_file approval gate."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.registry = ForgeToolRegistry(
            project_root=self.tmpdir,
            require_write_approval=True,
        )

    def test_write_queued_not_applied(self):
        """With approval gate on, write_file queues instead of writing."""
        result = self.registry.execute("write_file", {
            "path": "test.txt",
            "content": "hello",
        })
        assert "queued for approval" in result
        assert not (Path(self.tmpdir) / "test.txt").exists()

    def test_pending_writes_tracked(self):
        """Pending writes are accessible via property."""
        self.registry.execute("write_file", {
            "path": "a.txt", "content": "aaa",
        })
        self.registry.execute("write_file", {
            "path": "b.txt", "content": "bbb",
        })
        assert len(self.registry.pending_writes) == 2
        assert self.registry.pending_writes[0]["path"] == "a.txt"

    def test_approve_single_write(self):
        """Approve a specific pending write by index."""
        self.registry.execute("write_file", {
            "path": "a.txt", "content": "aaa",
        })
        self.registry.execute("write_file", {
            "path": "b.txt", "content": "bbb",
        })
        result = self.registry.approve_write(0)
        assert "Approved" in result
        assert (Path(self.tmpdir) / "a.txt").read_text() == "aaa"
        assert not (Path(self.tmpdir) / "b.txt").exists()
        assert len(self.registry.pending_writes) == 1

    def test_approve_invalid_index(self):
        """Invalid index returns error."""
        result = self.registry.approve_write(99)
        assert "Invalid" in result

    def test_approve_all_writes(self):
        """Approve all pending writes at once."""
        self.registry.execute("write_file", {
            "path": "x.txt", "content": "xxx",
        })
        self.registry.execute("write_file", {
            "path": "y.txt", "content": "yyy",
        })
        results = self.registry.approve_all_writes()
        assert len(results) == 2
        assert (Path(self.tmpdir) / "x.txt").read_text() == "xxx"
        assert (Path(self.tmpdir) / "y.txt").read_text() == "yyy"
        assert len(self.registry.pending_writes) == 0

    def test_reject_all_writes(self):
        """Reject discards all pending writes."""
        self.registry.execute("write_file", {
            "path": "a.txt", "content": "aaa",
        })
        count = self.registry.reject_all_writes()
        assert count == 1
        assert len(self.registry.pending_writes) == 0
        assert not (Path(self.tmpdir) / "a.txt").exists()

    def test_read_not_affected_by_approval_gate(self):
        """Read operations work normally with approval gate on."""
        Path(self.tmpdir, "existing.py").write_text("code\n")
        result = self.registry.execute("read_file", {"path": "existing.py"})
        assert "code" in result

    def test_write_without_approval_gate(self):
        """Without the gate, writes apply immediately."""
        reg = ForgeToolRegistry(project_root=self.tmpdir)
        reg.execute("write_file", {"path": "direct.txt", "content": "direct"})
        assert (Path(self.tmpdir) / "direct.txt").read_text() == "direct"


# --- Budget gate for tool calls tests ---


class TestToolBudgetGate:
    """Tests for budget enforcement in tool execution."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        Path(self.tmpdir, "test.py").write_text("x = 1\n")

    def _make_budget(self, can_allocate=True, remaining=50000):
        """Create a mock BudgetManager."""
        bm = MagicMock()
        bm.can_allocate.return_value = can_allocate
        bm.remaining = remaining
        bm.record_usage = MagicMock()
        return bm

    def test_budget_allows_execution(self):
        """Tool executes when budget has capacity."""
        bm = self._make_budget(can_allocate=True)
        reg = ForgeToolRegistry(
            project_root=self.tmpdir, budget_manager=bm,
        )
        result = reg.execute("read_file", {"path": "test.py"}, agent_id="builder")
        assert "x = 1" in result
        bm.record_usage.assert_called_once()

    def test_budget_blocks_execution(self):
        """Tool blocked when budget exceeded."""
        bm = self._make_budget(can_allocate=False, remaining=0)
        reg = ForgeToolRegistry(
            project_root=self.tmpdir, budget_manager=bm,
        )
        result = reg.execute("read_file", {"path": "test.py"}, agent_id="builder")
        assert "Budget exceeded" in result
        bm.record_usage.assert_not_called()

    def test_budget_records_custom_tokens_per_call(self):
        """Custom tokens_per_call is recorded."""
        bm = self._make_budget()
        reg = ForgeToolRegistry(
            project_root=self.tmpdir,
            budget_manager=bm,
            budget_tokens_per_call=200,
        )
        reg.execute("list_files", {}, agent_id="analyst")
        bm.record_usage.assert_called_once_with(
            agent_id="analyst",
            tokens=200,
            operation="tool:list_files",
        )

    def test_budget_recording_failure_doesnt_break_execution(self):
        """Budget recording error doesn't break tool execution."""
        bm = self._make_budget()
        bm.record_usage.side_effect = RuntimeError("db error")
        reg = ForgeToolRegistry(
            project_root=self.tmpdir, budget_manager=bm,
        )
        result = reg.execute("read_file", {"path": "test.py"})
        assert "x = 1" in result

    def test_no_budget_manager_executes_normally(self):
        """Without budget manager, everything works normally."""
        reg = ForgeToolRegistry(project_root=self.tmpdir)
        result = reg.execute("read_file", {"path": "test.py"})
        assert "x = 1" in result

    def test_budget_gate_emits_audit_on_block(self):
        """Budget block emits an audit entry."""
        bm = self._make_budget(can_allocate=False, remaining=0)
        reg = ForgeToolRegistry(
            project_root=self.tmpdir, budget_manager=bm,
        )
        with patch("animus_forge.tools.registry.audit_logger") as mock_audit:
            reg.execute("read_file", {"path": "test.py"})
            entry = json.loads(mock_audit.info.call_args[0][0])
            assert entry["success"] is False
            assert entry["error"] == "budget_exceeded"


# --- Supervisor prompt improvement tests ---


class TestSupervisorPromptImprovement:
    """Tests for the improved supervisor system prompt."""

    def test_prompt_mentions_tool_equipped_agents(self):
        from animus_forge.agents.supervisor import SUPERVISOR_SYSTEM_PROMPT

        assert "Tool-equipped agents" in SUPERVISOR_SYSTEM_PROMPT
        assert "Builder" in SUPERVISOR_SYSTEM_PROMPT
        assert "Tester" in SUPERVISOR_SYSTEM_PROMPT
        assert "Reviewer" in SUPERVISOR_SYSTEM_PROMPT
        assert "Analyst" in SUPERVISOR_SYSTEM_PROMPT

    def test_prompt_mentions_text_only_agents(self):
        from animus_forge.agents.supervisor import SUPERVISOR_SYSTEM_PROMPT

        assert "Text-only agents" in SUPERVISOR_SYSTEM_PROMPT
        assert "Planner" in SUPERVISOR_SYSTEM_PROMPT
        assert "Architect" in SUPERVISOR_SYSTEM_PROMPT
        assert "Documenter" in SUPERVISOR_SYSTEM_PROMPT

    def test_prompt_instructs_file_access_via_tools(self):
        from animus_forge.agents.supervisor import SUPERVISOR_SYSTEM_PROMPT

        assert "MUST delegate to a tool-equipped agent" in SUPERVISOR_SYSTEM_PROMPT

    def test_prompt_names_animus_forge(self):
        from animus_forge.agents.supervisor import SUPERVISOR_SYSTEM_PROMPT

        assert "Animus Forge" in SUPERVISOR_SYSTEM_PROMPT


# --- edit_file tool tests ---


class TestEditFileTool:
    """Tests for the edit_file search-and-replace tool."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        Path(self.tmpdir, "example.py").write_text(
            "def hello():\n    return 'hello'\n\ndef goodbye():\n    return 'bye'\n"
        )
        self.registry = ForgeToolRegistry(project_root=self.tmpdir)

    def test_edit_file_registered(self):
        """edit_file tool is registered by default."""
        assert self.registry.get("edit_file") is not None

    def test_edit_replaces_unique_string(self):
        """Replaces a unique string in the file."""
        result = self.registry.execute("edit_file", {
            "path": "example.py",
            "old_string": "return 'hello'",
            "new_string": "return 'hi'",
        })
        assert "Edited" in result
        content = (Path(self.tmpdir) / "example.py").read_text()
        assert "return 'hi'" in content
        assert "return 'hello'" not in content

    def test_edit_not_found(self):
        """Error when old_string not in file."""
        result = self.registry.execute("edit_file", {
            "path": "example.py",
            "old_string": "nonexistent string",
            "new_string": "replacement",
        })
        assert "not found" in result

    def test_edit_multiple_matches_rejected(self):
        """Error when old_string appears more than once."""
        Path(self.tmpdir, "dup.py").write_text("x = 1\nx = 1\n")
        result = self.registry.execute("edit_file", {
            "path": "dup.py",
            "old_string": "x = 1",
            "new_string": "x = 2",
        })
        assert "found 2 times" in result

    def test_edit_file_not_found(self):
        """Error when file doesn't exist."""
        result = self.registry.execute("edit_file", {
            "path": "nonexistent.py",
            "old_string": "x",
            "new_string": "y",
        })
        assert "not found" in result

    def test_edit_empty_old_string(self):
        """Error when old_string is empty."""
        result = self.registry.execute("edit_file", {
            "path": "example.py",
            "old_string": "",
            "new_string": "something",
        })
        assert "must not be empty" in result

    def test_edit_with_approval_gate(self):
        """With approval gate, edit queues instead of applying."""
        reg = ForgeToolRegistry(
            project_root=self.tmpdir, require_write_approval=True,
        )
        result = reg.execute("edit_file", {
            "path": "example.py",
            "old_string": "return 'hello'",
            "new_string": "return 'hi'",
        })
        assert "queued for approval" in result
        # File unchanged
        content = (Path(self.tmpdir) / "example.py").read_text()
        assert "return 'hello'" in content

    def test_edit_multiline(self):
        """Can replace multiline strings."""
        result = self.registry.execute("edit_file", {
            "path": "example.py",
            "old_string": "def hello():\n    return 'hello'",
            "new_string": "def greet(name):\n    return f'hello {name}'",
        })
        assert "Edited" in result
        content = (Path(self.tmpdir) / "example.py").read_text()
        assert "def greet(name):" in content


# --- Multi-round supervisor loop tests ---


class TestMultiRoundSupervisor:
    """Tests for the multi-round delegation loop."""

    def _make_supervisor(self, responses, tool_registry=None):
        from animus_forge.agents.provider_wrapper import AgentProvider
        from animus_forge.agents.supervisor import SupervisorAgent

        mock_provider = MagicMock()
        mock_provider._initialized = True
        mock_provider.name = "test"

        response_iter = iter(responses)

        async def mock_complete_async(request):
            return next(response_iter)

        mock_provider.complete_async = mock_complete_async
        agent_provider = AgentProvider(mock_provider)
        return SupervisorAgent(provider=agent_provider, tool_registry=tool_registry)

    def test_single_round_default(self):
        """max_rounds=1 is single-pass (no verification)."""
        from animus_forge.providers.base import CompletionResponse

        supervisor = self._make_supervisor([
            # Supervisor analysis — direct response
            CompletionResponse(content="Direct answer", model="test", provider="test"),
        ])
        result = asyncio.run(supervisor.process_message("hello", max_rounds=1))
        assert result == "Direct answer"

    def test_multi_round_satisfied(self):
        """Supervisor says SATISFIED — no second delegation."""
        from animus_forge.providers.base import CompletionResponse

        delegation_json = json.dumps({
            "analysis": "Need to check code",
            "delegations": [{"agent": "reviewer", "task": "review the code"}],
            "synthesis_approach": "summarize findings",
        })
        supervisor = self._make_supervisor([
            # Round 1: supervisor delegates
            CompletionResponse(
                content=f"```json\n{delegation_json}\n```",
                model="test", provider="test",
            ),
            # Round 1: reviewer agent result
            CompletionResponse(content="Code looks good", model="test", provider="test"),
            # Follow-up check: satisfied
            CompletionResponse(content="SATISFIED", model="test", provider="test"),
            # Synthesis
            CompletionResponse(content="All good", model="test", provider="test"),
        ])

        result = asyncio.run(supervisor.process_message("review code", max_rounds=2))
        assert "good" in result.lower()

    def test_multi_round_follow_up(self):
        """Supervisor requests follow-up delegation."""
        from animus_forge.providers.base import CompletionResponse

        delegation1 = json.dumps({
            "analysis": "Build feature",
            "delegations": [{"agent": "builder", "task": "build it"}],
            "synthesis_approach": "deliver",
        })
        delegation2 = json.dumps({
            "analysis": "Fix issues found",
            "delegations": [{"agent": "tester", "task": "test it"}],
            "synthesis_approach": "verify",
        })
        supervisor = self._make_supervisor([
            # Round 1: delegate to builder
            CompletionResponse(
                content=f"```json\n{delegation1}\n```",
                model="test", provider="test",
            ),
            # Builder result
            CompletionResponse(content="Built it", model="test", provider="test"),
            # Follow-up: not satisfied, delegate to tester
            CompletionResponse(
                content=f"```json\n{delegation2}\n```",
                model="test", provider="test",
            ),
            # Tester result
            CompletionResponse(content="Tests pass", model="test", provider="test"),
            # Synthesis
            CompletionResponse(content="Built and tested", model="test", provider="test"),
        ])

        result = asyncio.run(supervisor.process_message("build and test", max_rounds=2))
        assert "tested" in result.lower()

    def test_process_message_has_max_rounds_param(self):
        """process_message accepts max_rounds parameter."""
        import inspect

        from animus_forge.agents.supervisor import SupervisorAgent

        sig = inspect.signature(SupervisorAgent.process_message)
        assert "max_rounds" in sig.parameters
