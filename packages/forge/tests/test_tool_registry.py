"""Tests for ForgeToolRegistry and tool-equipped agent execution."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
        from animus_forge.providers.base import CompletionResponse, ToolCall

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
        from animus_forge.providers.base import CompletionResponse

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
            result = asyncio.run(supervisor._run_agent("planner", "plan something", []))
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
            result = asyncio.run(supervisor._run_agent("tester", "test something", []))
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
