"""Tests for the self-heal proactive check and improvement sandbox."""

from __future__ import annotations

import asyncio
import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import tomli_w

from animus_bootstrap.intelligence.proactive.checks.self_heal import (
    _MIN_EXECUTIONS,
    _run_self_heal,
    clear_proposed_areas,
    get_self_heal_check,
    set_self_heal_deps,
)
from animus_bootstrap.intelligence.tools.builtin.improvement_store import (
    ImprovementStore,
)
from animus_bootstrap.intelligence.tools.builtin.sandbox import (
    ImprovementSandbox,
    _get_nested,
    _set_nested,
)


def _run(coro):
    return asyncio.run(coro)


# ======================================================================
# Self-heal check
# ======================================================================


class TestSelfHealCheck:
    def setup_method(self):
        clear_proposed_areas()
        set_self_heal_deps(None, None, None, None)

    def test_get_self_heal_check_returns_proactive_check(self) -> None:
        check = get_self_heal_check()
        assert check.name == "self_heal"
        assert check.enabled is True
        assert "6" in check.schedule  # every 6 hours

    def test_skips_without_executor(self) -> None:
        set_self_heal_deps(None, None, None, None)
        result = _run(_run_self_heal())
        assert result is None

    def test_skips_with_empty_history(self) -> None:
        executor = MagicMock()
        executor.get_history.return_value = []
        set_self_heal_deps(tool_executor=executor)
        result = _run(_run_self_heal())
        assert result is None

    def test_skips_with_insufficient_history(self) -> None:
        executor = MagicMock()
        results = [_make_result(success=True) for _ in range(3)]
        executor.get_history.return_value = results
        set_self_heal_deps(tool_executor=executor)
        result = _run(_run_self_heal())
        assert result is None

    def test_detects_high_failure_rate(self, tmp_path: Path) -> None:
        store = ImprovementStore(tmp_path / "improvements.db")
        executor = MagicMock()

        # 4 failures, 2 successes for same tool = 66% failure rate
        results = []
        for i in range(4):
            results.append(_make_result(success=False, tool_name="web_search", output=f"err{i}"))
        for _ in range(2):
            results.append(_make_result(success=True, tool_name="web_search"))

        executor.get_history.return_value = results
        set_self_heal_deps(tool_executor=executor, improvement_store=store)

        result = _run(_run_self_heal())
        assert result is not None
        assert "1 improvement proposal" in result

        proposals = store.list_all()
        assert len(proposals) == 1
        assert proposals[0]["area"] == "tool:web_search"
        assert "66%" in proposals[0]["description"] or "67%" in proposals[0]["description"]
        store.close()

    def test_detects_slow_tools(self, tmp_path: Path) -> None:
        store = ImprovementStore(tmp_path / "improvements.db")
        executor = MagicMock()

        results = [
            _make_result(success=True, tool_name="big_query", duration_ms=15000)
            for _ in range(_MIN_EXECUTIONS)
        ]
        executor.get_history.return_value = results
        set_self_heal_deps(tool_executor=executor, improvement_store=store)

        result = _run(_run_self_heal())
        assert result is not None

        proposals = store.list_all()
        assert any("perf:big_query" in p["area"] for p in proposals)
        store.close()

    def test_detects_repeated_errors(self, tmp_path: Path) -> None:
        store = ImprovementStore(tmp_path / "improvements.db")
        executor = MagicMock()

        # 3 identical errors + 3 successes
        results = [
            _make_result(success=False, tool_name="api_call", output="connection refused")
            for _ in range(3)
        ]
        results += [_make_result(success=True, tool_name="other_tool") for _ in range(3)]
        executor.get_history.return_value = results
        set_self_heal_deps(tool_executor=executor, improvement_store=store)

        result = _run(_run_self_heal())
        assert result is not None
        proposals = store.list_all()
        assert any("error_pattern" in p["area"] for p in proposals)
        store.close()

    def test_no_duplicate_proposals(self, tmp_path: Path) -> None:
        store = ImprovementStore(tmp_path / "improvements.db")
        executor = MagicMock()

        results = [
            _make_result(success=False, tool_name="web_search", output="timeout")
            for _ in range(_MIN_EXECUTIONS)
        ]
        executor.get_history.return_value = results
        set_self_heal_deps(tool_executor=executor, improvement_store=store)

        _run(_run_self_heal())
        first_count = len(store.list_all())

        # Run again — should not create duplicates
        _run(_run_self_heal())
        assert len(store.list_all()) == first_count
        store.close()

    def test_returns_none_when_all_healthy(self, tmp_path: Path) -> None:
        store = ImprovementStore(tmp_path / "improvements.db")
        executor = MagicMock()

        results = [
            _make_result(success=True, tool_name="tool_a", duration_ms=100) for _ in range(10)
        ]
        executor.get_history.return_value = results
        set_self_heal_deps(tool_executor=executor, improvement_store=store)

        result = _run(_run_self_heal())
        assert result is None
        assert len(store.list_all()) == 0
        store.close()


# ======================================================================
# Improvement Sandbox
# ======================================================================


def _write_toml(path: Path, data: dict) -> None:
    path.write_bytes(tomli_w.dumps(data).encode())


def _read_toml(path: Path) -> dict:
    return tomllib.loads(path.read_bytes().decode())


class TestImprovementSandbox:
    def test_apply_config_change(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        _write_toml(config_path, {"intelligence": {"tool_timeout_seconds": 30}})

        sandbox = ImprovementSandbox(tmp_path, config_path=config_path)
        result = sandbox.apply_config_change(1, "intelligence.tool_timeout_seconds", 60)

        assert result["status"] == "applied"
        assert result["old_value"] == 30
        assert result["new_value"] == 60

        updated = _read_toml(config_path)
        assert updated["intelligence"]["tool_timeout_seconds"] == 60

    def test_apply_config_creates_backup(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        _write_toml(config_path, {"key": "old"})

        sandbox = ImprovementSandbox(tmp_path, config_path=config_path)
        sandbox.apply_config_change(42, "key", "new")

        backups = sandbox.list_backups()
        assert len(backups) == 1
        assert backups[0]["proposal_id"] == 42

    def test_rollback_restores_original(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        _write_toml(config_path, {"key": "original_value"})

        sandbox = ImprovementSandbox(tmp_path, config_path=config_path)
        sandbox.apply_config_change(1, "key", "changed")

        assert _read_toml(config_path)["key"] == "changed"

        result = sandbox.rollback(1, config_path)
        assert result["status"] == "rolled_back"

        assert _read_toml(config_path)["key"] == "original_value"

    def test_rollback_missing_backup(self, tmp_path: Path) -> None:
        sandbox = ImprovementSandbox(tmp_path)
        result = sandbox.rollback(999, tmp_path / "nonexistent.toml")
        assert result["status"] == "error"

    def test_apply_identity_append(self, tmp_path: Path) -> None:
        identity_mgr = MagicMock()
        identity_mgr.read.return_value = "# LEARNED.md\n"

        sandbox = ImprovementSandbox(tmp_path)
        result = sandbox.apply_identity_append(
            1, "LEARNED.md", "Self-Heal", "Reduce timeout for web_search", identity_mgr
        )
        assert result["status"] == "applied"
        identity_mgr.append_to_learned.assert_called_once()

    def test_no_config_path_returns_error(self, tmp_path: Path) -> None:
        sandbox = ImprovementSandbox(tmp_path, config_path=None)
        result = sandbox.apply_config_change(1, "key", "val")
        assert result["status"] == "error"

    def test_nested_config_change(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        _write_toml(config_path, {"a": {"b": {"c": 1}}})

        sandbox = ImprovementSandbox(tmp_path, config_path=config_path)
        result = sandbox.apply_config_change(1, "a.b.c", 99)
        assert result["old_value"] == 1
        assert result["new_value"] == 99

        updated = _read_toml(config_path)
        assert updated["a"]["b"]["c"] == 99

    def test_on_config_changed_callback(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        _write_toml(config_path, {"key": "val"})

        callback = MagicMock()
        sandbox = ImprovementSandbox(tmp_path, config_path=config_path, on_config_changed=callback)
        sandbox.apply_config_change(1, "key", "new")

        callback.assert_called_once()

    def test_rollback_triggers_reload(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        _write_toml(config_path, {"key": "original"})

        callback = MagicMock()
        sandbox = ImprovementSandbox(tmp_path, config_path=config_path, on_config_changed=callback)
        sandbox.apply_config_change(1, "key", "changed")
        callback.reset_mock()

        sandbox.rollback(1, config_path)
        callback.assert_called_once()


# ======================================================================
# Impact measurement (improvement_store)
# ======================================================================


class TestImpactMeasurement:
    def test_baseline_and_post_metrics(self, tmp_path: Path) -> None:
        store = ImprovementStore(tmp_path / "improvements.db")
        pid = store.save(
            {
                "area": "tool:web_search",
                "description": "test",
                "status": "proposed",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        baseline = json.dumps({"failure_rate": 0.3, "avg_duration_ms": 5000})
        store.set_baseline_metrics(pid, baseline)

        proposal = store.get(pid)
        assert proposal["baseline_metrics"] == baseline

        post = json.dumps({"failure_rate": 0.1, "avg_duration_ms": 3000})
        store.set_post_metrics(pid, post, 45.0)

        proposal = store.get(pid)
        assert proposal["post_metrics"] == post
        assert proposal["impact_score"] == 45.0
        store.close()

    def test_row_to_dict_has_new_fields(self, tmp_path: Path) -> None:
        store = ImprovementStore(tmp_path / "improvements.db")
        pid = store.save(
            {
                "area": "test",
                "description": "test",
                "status": "proposed",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
        proposal = store.get(pid)
        assert "baseline_metrics" in proposal
        assert "post_metrics" in proposal
        assert "impact_score" in proposal
        store.close()


# ======================================================================
# Runtime config reload
# ======================================================================


class TestRuntimeReloadConfig:
    def test_reload_updates_tool_executor_settings(self) -> None:
        from unittest.mock import patch

        from animus_bootstrap.runtime import AnimusRuntime

        rt = AnimusRuntime.__new__(AnimusRuntime)
        rt._started = True
        rt._channels = {}
        rt.proactive_engine = None
        rt.router = None

        # Mock tool executor
        rt.tool_executor = MagicMock()
        rt.tool_executor._timeout = 30.0
        rt.tool_executor._max_calls = 10

        # Mock ConfigManager to return config with different values
        mock_config = MagicMock()
        mock_config.intelligence.tool_timeout_seconds = 120
        mock_config.intelligence.max_tool_calls_per_turn = 20

        with patch("animus_bootstrap.runtime.ConfigManager") as mock_cm:
            mock_cm.return_value.load.return_value = mock_config
            rt.reload_config()

        assert rt.tool_executor._timeout == 120.0
        assert rt.tool_executor._max_calls == 20
        assert rt._config is mock_config

    def test_reload_survives_load_failure(self) -> None:
        from unittest.mock import patch

        from animus_bootstrap.runtime import AnimusRuntime

        rt = AnimusRuntime.__new__(AnimusRuntime)
        rt._started = True
        rt._channels = {}
        rt.tool_executor = None
        rt.proactive_engine = None
        rt.router = None

        original_config = MagicMock()
        rt._config = original_config

        with patch("animus_bootstrap.runtime.ConfigManager") as mock_cm:
            mock_cm.return_value.load.side_effect = RuntimeError("bad config")
            rt.reload_config()

        # Config should remain unchanged
        assert rt._config is original_config


# ======================================================================
# Helpers
# ======================================================================


class TestHelpers:
    def test_get_nested(self) -> None:
        d = {"a": {"b": {"c": 42}}}
        assert _get_nested(d, ["a", "b", "c"]) == 42
        assert _get_nested(d, ["a", "x"]) is None

    def test_set_nested(self) -> None:
        d: dict = {}
        _set_nested(d, ["a", "b", "c"], 99)
        assert d["a"]["b"]["c"] == 99

    def test_set_nested_overwrites(self) -> None:
        d = {"a": {"b": 1}}
        _set_nested(d, ["a", "b"], 2)
        assert d["a"]["b"] == 2


# ======================================================================
# Fixtures
# ======================================================================


def _make_result(
    success: bool = True,
    tool_name: str = "test_tool",
    output: str = "ok",
    duration_ms: float = 100.0,
) -> MagicMock:
    r = MagicMock()
    r.success = success
    r.tool_name = tool_name
    r.output = output
    r.duration_ms = duration_ms
    r.timestamp = datetime.now(UTC)
    return r
