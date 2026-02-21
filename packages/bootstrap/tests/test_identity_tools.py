"""Tests for identity_tools — guardrailed identity file access."""

from __future__ import annotations

import asyncio
import json

import pytest

from animus_bootstrap.identity.manager import IdentityFileManager
from animus_bootstrap.intelligence.tools.builtin import identity_tools
from animus_bootstrap.intelligence.tools.builtin.identity_tools import (
    _identity_append_learned,
    _identity_list,
    _identity_read,
    _identity_write,
    get_identity_tools,
    set_identity_improvement_store,
    set_identity_manager,
)


def _run(coro):
    """Run an async coroutine synchronously without closing the global event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


@pytest.fixture(autouse=True)
def _wire_manager(tmp_path):
    """Set up and tear down the module-level identity manager."""
    mgr = IdentityFileManager(tmp_path / "identity")
    mgr.generate_from_templates({"name": "TestUser", "timezone": "UTC"})
    set_identity_manager(mgr)
    yield
    set_identity_manager(None)
    set_identity_improvement_store(None)


@pytest.fixture()
def manager():
    """Return the current wired manager."""
    return identity_tools._identity_manager


class TestIdentityRead:
    def test_read_existing_file(self):
        result = _run(_identity_read({"filename": "CORE_VALUES.md"}))
        assert "Sovereignty" in result

    def test_read_unknown_file(self):
        result = _run(_identity_read({"filename": "BAD.md"}))
        assert "Unknown identity file" in result

    def test_read_no_manager(self):
        set_identity_manager(None)
        result = _run(_identity_read({"filename": "IDENTITY.md"}))
        assert "not available" in result


class TestIdentityWrite:
    def test_write_small_change(self, manager):
        original = manager.read("PREFERENCES.md")
        small_change = original + "\n- Extra pref"
        result = _run(
            _identity_write(
                {
                    "filename": "PREFERENCES.md",
                    "content": small_change,
                    "reason": "add pref",
                }
            )
        )
        assert "Successfully updated" in result

    def test_write_core_values_blocked(self):
        result = _run(
            _identity_write(
                {
                    "filename": "CORE_VALUES.md",
                    "content": "hacked",
                }
            )
        )
        assert "immutable" in result
        assert "cannot be modified" in result

    def test_write_large_change_creates_proposal(self, manager):
        manager.write("CONTEXT.md", "short")
        result = _run(
            _identity_write(
                {
                    "filename": "CONTEXT.md",
                    "content": "a" * 1000,
                    "reason": "complete rewrite",
                }
            )
        )
        assert "exceeds 20% threshold" in result
        assert "Proposal #" in result

    def test_write_unknown_file(self):
        result = _run(
            _identity_write(
                {
                    "filename": "RANDOM.md",
                    "content": "bad",
                }
            )
        )
        assert "Unknown identity file" in result

    def test_write_no_manager(self):
        set_identity_manager(None)
        result = _run(
            _identity_write(
                {
                    "filename": "IDENTITY.md",
                    "content": "test",
                }
            )
        )
        assert "not available" in result

    def test_write_to_empty_file_succeeds(self, manager):
        path = manager.identity_dir / "GOALS.md"
        path.unlink(missing_ok=True)
        result = _run(
            _identity_write(
                {
                    "filename": "GOALS.md",
                    "content": "New goals",
                }
            )
        )
        assert "Successfully updated" in result


class TestIdentityAppendLearned:
    def test_append(self, manager):
        result = _run(
            _identity_append_learned(
                {
                    "section": "Patterns",
                    "entry": "User prefers bullet points",
                }
            )
        )
        assert "Added entry" in result
        content = manager.read("LEARNED.md")
        assert "User prefers bullet points" in content

    def test_append_empty_entry(self):
        result = _run(_identity_append_learned({"section": "Test", "entry": ""}))
        assert "No entry" in result

    def test_append_no_manager(self):
        set_identity_manager(None)
        result = _run(
            _identity_append_learned(
                {
                    "section": "Test",
                    "entry": "something",
                }
            )
        )
        assert "not available" in result


class TestIdentityList:
    def test_list_returns_json(self):
        result = _run(_identity_list({}))
        data = json.loads(result)
        assert len(data) == 6
        filenames = {f["filename"] for f in data}
        assert "CORE_VALUES.md" in filenames
        assert "IDENTITY.md" in filenames

    def test_list_shows_locked_status(self):
        result = _run(_identity_list({}))
        data = json.loads(result)
        core = next(f for f in data if f["filename"] == "CORE_VALUES.md")
        assert core["locked"] is True
        identity = next(f for f in data if f["filename"] == "IDENTITY.md")
        assert identity["locked"] is False

    def test_list_no_manager(self):
        set_identity_manager(None)
        result = _run(_identity_list({}))
        assert "not available" in result


class TestGetIdentityTools:
    def test_returns_four_tools(self):
        tools = get_identity_tools()
        assert len(tools) == 4

    def test_tool_names(self):
        tools = get_identity_tools()
        names = {t.name for t in tools}
        assert names == {
            "identity_read",
            "identity_write",
            "identity_append_learned",
            "identity_list",
        }

    def test_write_requires_approval(self):
        tools = get_identity_tools()
        write_tool = next(t for t in tools if t.name == "identity_write")
        assert write_tool.permission == "approve"

    def test_read_is_auto(self):
        tools = get_identity_tools()
        read_tool = next(t for t in tools if t.name == "identity_read")
        assert read_tool.permission == "auto"
