"""Tests for the hook gate system — spec §4.8 + R7 + R13 + R14 + A5/A16/A18."""

import logging
from pathlib import Path

import pytest

from animus_forge.agent.hooks import (
    HookOutcome,
    load_user_hooks,
    run_agent_end,
    run_agent_start,
    run_post_tool_use,
    run_pre_tool_use,
)
from animus_forge.agent.hooks.builtin import identity_guard, p1_p9_stub
from animus_forge.agent.types import (
    AgentContext,
    Allow,
    Deny,
    Receipt,
    ReceiptStatus,
    ToolResult,
)


@pytest.fixture
def ctx(tmp_path: Path) -> AgentContext:
    return AgentContext(
        task="test",
        project_root=tmp_path,
        provider="llamacpp",
        model="qwen3.6-research",
    )


@pytest.fixture
def receipt() -> Receipt:
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    return Receipt(
        task="t",
        status=ReceiptStatus.SUCCESS,
        turns_taken=1,
        tokens_in=1,
        tokens_out=1,
        wall_seconds=0.1,
        model="m",
        tool_calls=[],
        skill_load_warnings=[],
        hook_errors=[],
        final_message="",
        error=None,
        started_at=now,
        ended_at=now,
    )


def _write_hook(hook_dir: Path, name: str, body: str) -> None:
    hook_dir.mkdir(parents=True, exist_ok=True)
    (hook_dir / name).write_text(body, encoding="utf-8")


class TestIdentityGuard:
    def _setup_identity_paths(self, tmp_path: Path) -> None:
        from animus_forge.agent.identity_roots import IDENTITY_ROOTS

        for rel in IDENTITY_ROOTS:
            p = tmp_path / rel
            if rel.endswith(".md"):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
            else:
                p.mkdir(parents=True, exist_ok=True)

    def test_writefile_to_identity_denied(self, ctx: AgentContext):
        self._setup_identity_paths(ctx.project_root)
        d = identity_guard.pre_tool_use(
            ctx, "WriteFile", {"path": "packages/core/animus/CORE_VALUES.md"}
        )
        assert isinstance(d, Deny)
        assert "identity-file write blocked" in d.reason

    def test_editfile_to_identity_denied(self, ctx: AgentContext):
        self._setup_identity_paths(ctx.project_root)
        d = identity_guard.pre_tool_use(
            ctx, "EditFile", {"path": "packages/core/animus/identity/foo.yaml"}
        )
        assert isinstance(d, Deny)

    def test_writefile_to_safe_path_allowed(self, ctx: AgentContext):
        self._setup_identity_paths(ctx.project_root)
        d = identity_guard.pre_tool_use(ctx, "WriteFile", {"path": "README.md"})
        assert isinstance(d, Allow)

    def test_readfile_unguarded(self, ctx: AgentContext):
        """Identity guard only fires on WriteFile/EditFile, not Read."""
        self._setup_identity_paths(ctx.project_root)
        d = identity_guard.pre_tool_use(
            ctx, "ReadFile", {"path": "packages/core/animus/CORE_VALUES.md"}
        )
        assert isinstance(d, Allow)

    def test_bash_unguarded(self, ctx: AgentContext):
        d = identity_guard.pre_tool_use(ctx, "Bash", {"command": "echo hi"})
        assert isinstance(d, Allow)

    def test_missing_path_arg_allows(self, ctx: AgentContext):
        """Malformed args fall through to tool's own ToolError."""
        d = identity_guard.pre_tool_use(ctx, "WriteFile", {})
        assert isinstance(d, Allow)

    def test_non_string_path_allows(self, ctx: AgentContext):
        d = identity_guard.pre_tool_use(ctx, "WriteFile", {"path": 42})
        assert isinstance(d, Allow)


class TestP1P9Stub:
    def test_always_allows(self, ctx: AgentContext):
        d = p1_p9_stub.pre_tool_use(ctx, "Bash", {"command": "rm -rf /"})
        assert isinstance(d, Allow)

    def test_emits_stub_log(self, ctx: AgentContext, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.INFO, logger="animus_forge.agent.hooks.builtin.p1_p9_stub"):
            p1_p9_stub.pre_tool_use(ctx, "WriteFile", {"path": "x"})
        assert p1_p9_stub.STUB_LOG_MESSAGE in caplog.text


class TestLoader:
    def test_missing_dir_returns_empty(self, tmp_path: Path):
        assert load_user_hooks(tmp_path / "missing") == []

    def test_loads_python_files_alphabetically(self, tmp_path: Path):
        _write_hook(tmp_path, "z_last.py", "name = 'z'\n")
        _write_hook(tmp_path, "a_first.py", "name = 'a'\n")
        modules = load_user_hooks(tmp_path)
        names = [getattr(m, "name", None) for m in modules]
        assert names == ["a", "z"]

    def test_skips_underscore_files(self, tmp_path: Path):
        _write_hook(tmp_path, "_internal.py", "name = 'skip'\n")
        _write_hook(tmp_path, "real.py", "name = 'real'\n")
        modules = load_user_hooks(tmp_path)
        assert len(modules) == 1
        assert modules[0].name == "real"

    def test_syntax_error_annotates_path(self, tmp_path: Path):
        _write_hook(tmp_path, "broken.py", "def bad(:\n")
        with pytest.raises(SyntaxError, match="broken.py"):
            load_user_hooks(tmp_path)

    def test_non_py_files_ignored(self, tmp_path: Path):
        (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
        _write_hook(tmp_path, "ok.py", "x = 1\n")
        modules = load_user_hooks(tmp_path)
        assert len(modules) == 1


class TestPreToolUseRunner:
    def test_builtins_run_when_no_user_hooks(self, ctx: AgentContext):
        outcome = run_pre_tool_use(ctx, "Bash", {"command": "echo hi"}, [])
        assert outcome.allowed is True
        assert outcome.deny_reason == ""

    def test_identity_guard_short_circuits_a5_a16(self, ctx: AgentContext, tmp_path: Path):
        """A5/A16: built-in identity deny fires before user hooks.

        Even if a user hook tries to Allow, the built-in's Deny stops the chain.
        """
        _write_hook(
            tmp_path / "hooks.d",
            "force_allow.py",
            "from animus_forge.agent.types import Allow\n"
            "def pre_tool_use(ctx, tool, args):\n"
            "    return Allow('user override')\n",
        )
        from animus_forge.agent.identity_roots import IDENTITY_ROOTS

        for rel in IDENTITY_ROOTS:
            p = ctx.project_root / rel
            if rel.endswith(".md"):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
            else:
                p.mkdir(parents=True, exist_ok=True)
        user_hooks = load_user_hooks(tmp_path / "hooks.d")

        outcome = run_pre_tool_use(
            ctx,
            "WriteFile",
            {"path": "packages/core/animus/CORE_VALUES.md"},
            user_hooks,
        )
        assert outcome.allowed is False
        assert "identity-file write blocked" in outcome.deny_reason

    def test_user_deny_fires(self, ctx: AgentContext, tmp_path: Path):
        """A5: user hook Deny aborts the tool call."""
        _write_hook(
            tmp_path / "hooks.d",
            "no_bash.py",
            "from animus_forge.agent.types import Allow, Deny\n"
            "def pre_tool_use(ctx, tool, args):\n"
            "    return Deny('bash forbidden by user policy') if tool == 'Bash' else Allow()\n",
        )
        user_hooks = load_user_hooks(tmp_path / "hooks.d")
        outcome = run_pre_tool_use(ctx, "Bash", {"command": "echo x"}, user_hooks)
        assert outcome.allowed is False
        assert "bash forbidden" in outcome.deny_reason

    def test_user_allow_does_not_override_builtin_deny(self, ctx: AgentContext, tmp_path: Path):
        """Spec R26 — user Allow after built-in Deny does NOT override."""
        from animus_forge.agent.identity_roots import IDENTITY_ROOTS

        for rel in IDENTITY_ROOTS:
            p = ctx.project_root / rel
            if rel.endswith(".md"):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
            else:
                p.mkdir(parents=True, exist_ok=True)

        _write_hook(
            tmp_path / "hooks.d",
            "z_override.py",
            "from animus_forge.agent.types import Allow\n"
            "def pre_tool_use(ctx, tool, args):\n"
            "    return Allow('I say it is fine')\n",
        )
        user_hooks = load_user_hooks(tmp_path / "hooks.d")
        outcome = run_pre_tool_use(
            ctx,
            "WriteFile",
            {"path": "packages/core/animus/identity/x.yaml"},
            user_hooks,
        )
        assert outcome.allowed is False  # built-in identity Deny wins

    def test_hook_exception_denies_fail_closed(self, ctx: AgentContext, tmp_path: Path):
        _write_hook(
            tmp_path / "hooks.d",
            "broken.py",
            "def pre_tool_use(ctx, tool, args):\n    raise ValueError('boom')\n",
        )
        user_hooks = load_user_hooks(tmp_path / "hooks.d")
        outcome = run_pre_tool_use(ctx, "Bash", {"command": "x"}, user_hooks)
        assert outcome.allowed is False
        assert any("raised" in e for e in outcome.hook_errors)

    def test_non_decision_return_treated_as_deny(self, ctx: AgentContext, tmp_path: Path):
        _write_hook(
            tmp_path / "hooks.d",
            "wrong.py",
            "def pre_tool_use(ctx, tool, args):\n    return 'sure go ahead'\n",
        )
        user_hooks = load_user_hooks(tmp_path / "hooks.d")
        outcome = run_pre_tool_use(ctx, "Bash", {"command": "x"}, user_hooks)
        assert outcome.allowed is False

    def test_module_without_pre_tool_use_skipped(self, ctx: AgentContext, tmp_path: Path):
        """User module that doesn't define pre_tool_use is harmlessly skipped."""
        _write_hook(tmp_path / "hooks.d", "no_pre.py", "x = 1\n")
        user_hooks = load_user_hooks(tmp_path / "hooks.d")
        outcome = run_pre_tool_use(ctx, "Bash", {"command": "echo"}, user_hooks)
        assert outcome.allowed is True


class TestFireAndForgetRunners:
    def test_agent_start_invokes_user_hook(self, ctx: AgentContext, tmp_path: Path):
        _write_hook(
            tmp_path / "hooks.d",
            "marker.py",
            "from pathlib import Path\n"
            "def agent_start(ctx):\n"
            "    (Path(ctx.project_root) / '.agent_started').touch()\n",
        )
        user_hooks = load_user_hooks(tmp_path / "hooks.d")
        errors = run_agent_start(ctx, user_hooks)
        assert errors == []
        assert (ctx.project_root / ".agent_started").exists()

    def test_post_tool_use_invokes_user_hook(self, ctx: AgentContext, tmp_path: Path):
        _write_hook(
            tmp_path / "hooks.d",
            "logger.py",
            "from pathlib import Path\n"
            "def post_tool_use(ctx, tool, args, result):\n"
            "    (Path(ctx.project_root) / f'.post-{tool}').touch()\n",
        )
        user_hooks = load_user_hooks(tmp_path / "hooks.d")
        result = ToolResult(tool="Bash", status="ok", output="x", wall_ms=1)
        errors = run_post_tool_use(ctx, "Bash", {}, result, user_hooks)
        assert errors == []
        assert (ctx.project_root / ".post-Bash").exists()

    def test_agent_end_invokes_user_hook(self, ctx: AgentContext, tmp_path: Path, receipt: Receipt):
        _write_hook(
            tmp_path / "hooks.d",
            "marker.py",
            "from pathlib import Path\n"
            "def agent_end(ctx, receipt):\n"
            "    (Path(ctx.project_root) / '.agent_ended').touch()\n",
        )
        user_hooks = load_user_hooks(tmp_path / "hooks.d")
        errors = run_agent_end(ctx, receipt, user_hooks)
        assert errors == []
        assert (ctx.project_root / ".agent_ended").exists()

    def test_fire_and_forget_captures_exceptions(self, ctx: AgentContext, tmp_path: Path):
        _write_hook(
            tmp_path / "hooks.d",
            "broken.py",
            "def agent_start(ctx):\n    raise RuntimeError('boom')\n",
        )
        user_hooks = load_user_hooks(tmp_path / "hooks.d")
        errors = run_agent_start(ctx, user_hooks)
        assert len(errors) == 1
        assert "boom" in errors[0]


class TestHookOutcome:
    def test_default_allows_no_errors(self):
        outcome = HookOutcome(allowed=True)
        assert outcome.allowed is True
        assert outcome.deny_reason == ""
        assert outcome.hook_errors == []
