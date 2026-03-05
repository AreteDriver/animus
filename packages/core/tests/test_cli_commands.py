"""Tests for unified CLI slash commands and agent loop wiring.

Tests the /build, /model, /auto commands and streaming callback integration
without requiring an interactive terminal.
"""

from __future__ import annotations

import os

from animus.cognitive import CognitiveLayer, ModelConfig, ReasoningMode, detect_mode


class TestDetectMode:
    """Test detect_mode routing."""

    def test_quick_mode(self):
        assert detect_mode("hello") == ReasoningMode.QUICK

    def test_deep_mode(self):
        assert detect_mode("analyze the performance bottleneck") == ReasoningMode.DEEP

    def test_research_mode(self):
        assert detect_mode("research the latest Python PEPs") == ReasoningMode.RESEARCH


class TestDualModelRouting:
    """Test dual-model config creation."""

    def test_ollama_only(self):
        config = ModelConfig.ollama("deepseek-coder-v2")
        cognitive = CognitiveLayer(config)
        assert cognitive.primary_config.provider.value == "ollama"
        assert cognitive.fallback_config is None
        assert not cognitive.has_dual_models

    def test_dual_model(self):
        primary = ModelConfig.anthropic()
        fallback = ModelConfig.ollama("deepseek-coder-v2")
        cognitive = CognitiveLayer(primary_config=primary, fallback_config=fallback)
        assert cognitive.primary_config.provider.value == "anthropic"
        assert cognitive.fallback_config is not None
        assert cognitive.has_dual_models


class TestAutoApproveToggle:
    """Test /auto command behavior."""

    def test_toggle_on(self):
        os.environ.pop("ANIMUS_AUTO_APPROVE", None)
        # Simulate /auto: reads current (default false), flips to true
        current = os.environ.get("ANIMUS_AUTO_APPROVE", "false")
        new_val = "false" if current.lower() in ("1", "true", "yes") else "true"
        os.environ["ANIMUS_AUTO_APPROVE"] = new_val
        assert os.environ["ANIMUS_AUTO_APPROVE"] == "true"

    def test_toggle_off(self):
        os.environ["ANIMUS_AUTO_APPROVE"] = "true"
        current = os.environ.get("ANIMUS_AUTO_APPROVE", "false")
        new_val = "false" if current.lower() in ("1", "true", "yes") else "true"
        os.environ["ANIMUS_AUTO_APPROVE"] = new_val
        assert os.environ["ANIMUS_AUTO_APPROVE"] == "false"

    def teardown_method(self):
        os.environ.pop("ANIMUS_AUTO_APPROVE", None)


class TestBuildCommand:
    """Test /build command validation and workflow loading."""

    def test_build_no_description(self):
        """'/build' with no description should prompt usage."""
        user_input = "/build"
        parts = user_input.split(None, 1)
        task_desc = parts[1].strip() if len(parts) > 1 else ""
        assert task_desc == ""

    def test_build_extracts_description(self):
        """'/build fix the bug' extracts task description."""
        user_input = "/build fix the import error in memory.py"
        parts = user_input.split(None, 1)
        task_desc = parts[1].strip() if len(parts) > 1 else ""
        assert task_desc == "fix the import error in memory.py"

    def test_build_workflow_yaml_exists(self):
        """build_task.yaml should exist in configs/examples."""
        from pathlib import Path

        build_yaml = Path(__file__).parent.parent / "configs" / "examples" / "build_task.yaml"
        assert build_yaml.exists(), f"Missing: {build_yaml}"

    def test_build_workflow_loads(self):
        """build_task.yaml should parse without errors."""
        from pathlib import Path

        from animus.forge.loader import load_workflow

        build_yaml = Path(__file__).parent.parent / "configs" / "examples" / "build_task.yaml"
        config = load_workflow(build_yaml)
        assert config.name == "build_task"
        assert len(config.agents) == 4
        assert config.agents[0].name == "planner"
        assert config.agents[1].name == "coder"
        assert config.agents[2].name == "verifier"
        assert config.agents[3].name == "fixer"

    def test_build_injects_task_into_planner(self):
        """Task description is injected into the planner's system_prompt."""
        from pathlib import Path

        from animus.forge.loader import load_workflow

        build_yaml = Path(__file__).parent.parent / "configs" / "examples" / "build_task.yaml"
        config = load_workflow(build_yaml)
        existing = config.agents[0].system_prompt or ""
        config.agents[0].system_prompt = f"{existing}\n\n## Task\nadd a docstring"
        assert "## Task" in config.agents[0].system_prompt
        assert "add a docstring" in config.agents[0].system_prompt


class TestModelCommand:
    """Test /model command output."""

    def test_model_single(self):
        """Single model mode shows provider/model."""
        config = ModelConfig.ollama("deepseek-coder-v2")
        cognitive = CognitiveLayer(config)
        p = cognitive.primary_config
        info = f"{p.provider.value}/{p.model_name}"
        assert "ollama" in info
        assert "deepseek-coder-v2" in info

    def test_model_dual(self):
        """Dual model mode shows both."""
        primary = ModelConfig.anthropic("claude-sonnet-4-20250514")
        fallback = ModelConfig.ollama("deepseek-coder-v2")
        cognitive = CognitiveLayer(primary_config=primary, fallback_config=fallback)
        assert cognitive.primary_config.provider.value == "anthropic"
        assert cognitive.fallback_config.provider.value == "ollama"


class TestStreamingWiring:
    """Test that stream_callback flows through the agent loop."""

    def test_stream_callback_signature_check(self):
        """inspect.signature correctly identifies stream_callback support."""
        import inspect

        # MockModel.generate does NOT have stream_callback
        config = ModelConfig.mock()
        cognitive = CognitiveLayer(config)
        sig = inspect.signature(cognitive.primary.generate)
        assert "stream_callback" not in sig.parameters

        # OllamaModel.generate DOES have stream_callback
        from animus.cognitive import OllamaModel

        ollama_model = OllamaModel(ModelConfig.ollama())
        sig = inspect.signature(ollama_model.generate)
        assert "stream_callback" in sig.parameters

    def test_stream_callback_not_passed_to_mock(self):
        """MockModel should not receive stream_callback (no support)."""
        from animus.tools import Tool, ToolRegistry, ToolResult

        cognitive = CognitiveLayer(ModelConfig.mock(default_response="TOOL: 0\nDone."))
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="t",
                description="t",
                parameters={},
                handler=lambda p: ToolResult(tool_name="t", success=True, output="ok"),
            )
        )
        # Should not raise even though MockModel doesn't accept stream_callback
        result = cognitive.think_with_tools(
            "test",
            tools=registry,
            stream_callback=lambda t: None,
        )
        assert isinstance(result, str)


class TestApprovalCallback:
    """Test the approval callback pattern."""

    def test_callback_blocks_tool(self):
        """Approval callback returning False blocks tool execution."""
        from animus.tools import Tool, ToolRegistry, ToolResult

        executed = {"count": 0}

        def handler(params):
            executed["count"] += 1
            return ToolResult(tool_name="danger", success=True, output="boom")

        call_count = {"n": 0}

        def sequenced(prompt, system=None, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return "TOOL: 1\n"
            return "TOOL: 0\nBlocked."

        cognitive = CognitiveLayer(ModelConfig.mock())
        cognitive.primary.generate = sequenced

        registry = ToolRegistry()
        registry.register(
            Tool(
                name="danger",
                description="Dangerous",
                parameters={},
                handler=handler,
                requires_approval=True,
            )
        )
        result = cognitive.think_with_tools(
            "do it",
            tools=registry,
            approval_callback=lambda name, params: False,
        )
        assert executed["count"] == 0  # Tool was never executed
        assert isinstance(result, str)


class TestWriteRootsSandboxIntegration:
    """Test sandbox wiring for /build command."""

    def test_security_config_has_write_roots(self):
        from animus.config import ToolsSecurityConfig

        config = ToolsSecurityConfig()
        assert hasattr(config, "write_roots")
        assert config.write_roots == []

    def test_sandbox_restricts_writes(self, tmp_path):
        from animus.config import ToolsSecurityConfig
        from animus.tools import _set_security_config, _validate_write_path

        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()

        config = ToolsSecurityConfig(
            allowed_paths=[str(tmp_path)],
            write_roots=[str(sandbox)],
        )
        _set_security_config(config)
        try:
            # Inside sandbox — OK
            ok, err = _validate_write_path(str(sandbox / "file.py"))
            assert ok

            # Outside sandbox — blocked
            ok, err = _validate_write_path(str(tmp_path / "outside.py"))
            assert not ok
            assert "Write denied" in err
        finally:
            _set_security_config(None)


class TestCommandSandbox:
    """Test run_command sandbox hardening."""

    def test_command_cwd_set_to_write_root(self, tmp_path):
        """Commands run from within sandbox directory."""
        from animus.config import ToolsSecurityConfig
        from animus.tools import _set_security_config, _tool_run_command

        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()

        config = ToolsSecurityConfig(
            allowed_paths=[str(tmp_path)],
            write_roots=[str(sandbox)],
        )
        _set_security_config(config)
        try:
            result = _tool_run_command({"command": "pwd"})
            assert result.success
            assert str(sandbox) in result.output
        finally:
            _set_security_config(None)

    def test_rm_outside_sandbox_blocked(self, tmp_path):
        """rm targeting paths outside sandbox is blocked."""
        from animus.config import ToolsSecurityConfig
        from animus.tools import _set_security_config, _validate_command

        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()

        config = ToolsSecurityConfig(
            allowed_paths=[str(tmp_path)],
            write_roots=[str(sandbox)],
        )
        _set_security_config(config)
        try:
            # rm inside sandbox — OK
            ok, err = _validate_command(f"rm {sandbox}/file.txt")
            assert ok, f"Expected OK, got: {err}"

            # rm outside sandbox — blocked
            ok, err = _validate_command(f"rm {tmp_path}/important.py")
            assert not ok
            assert "outside sandbox" in err
        finally:
            _set_security_config(None)

    def test_mv_outside_sandbox_blocked(self, tmp_path):
        """mv targeting paths outside sandbox is blocked."""
        from animus.config import ToolsSecurityConfig
        from animus.tools import _set_security_config, _validate_command

        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()

        config = ToolsSecurityConfig(
            allowed_paths=[str(tmp_path)],
            write_roots=[str(sandbox)],
        )
        _set_security_config(config)
        try:
            ok, err = _validate_command(f"mv {tmp_path}/file.py {sandbox}/file.py")
            assert not ok
            assert "outside sandbox" in err
        finally:
            _set_security_config(None)

    def test_non_destructive_commands_allowed(self, tmp_path):
        """Non-destructive commands (ls, cat, etc.) are not blocked by sandbox."""
        from animus.config import ToolsSecurityConfig
        from animus.tools import _set_security_config, _validate_command

        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()

        config = ToolsSecurityConfig(
            allowed_paths=[str(tmp_path)],
            write_roots=[str(sandbox)],
        )
        _set_security_config(config)
        try:
            ok, err = _validate_command(f"ls {tmp_path}")
            assert ok

            ok, err = _validate_command("echo hello")
            assert ok

            ok, err = _validate_command("ruff check .")
            assert ok
        finally:
            _set_security_config(None)
