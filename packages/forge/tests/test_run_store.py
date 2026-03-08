"""Tests for AgentRunStore persistence layer."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from enum import StrEnum
from unittest.mock import MagicMock

import pytest

from animus_forge.agents.run_store import AgentRunStore


class MockStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MockConfig:
    timeout_seconds: int = 300
    max_output_chars: int = 50000
    model: str | None = None


@dataclass
class MockRun:
    run_id: str
    agent: str
    task: str
    status: MockStatus = MockStatus.COMPLETED
    result: str | None = None
    error: str | None = None
    started_at: float = 0.0
    completed_at: float = 0.0
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    config: MockConfig = field(default_factory=MockConfig)


class InMemoryBackend:
    """Minimal SQLite backend for testing."""

    def __init__(self):
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(
            """
            CREATE TABLE agent_runs (
                run_id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                task TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                result TEXT,
                error TEXT,
                started_at REAL NOT NULL DEFAULT 0.0,
                completed_at REAL NOT NULL DEFAULT 0.0,
                parent_id TEXT,
                children TEXT NOT NULL DEFAULT '[]',
                config_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX idx_agent_runs_status ON agent_runs(status);
            CREATE INDEX idx_agent_runs_agent ON agent_runs(agent);
            """
        )

    @property
    def placeholder(self):
        return "?"

    def adapt_query(self, query):
        return query

    def execute(self, query, params=()):
        cursor = self._conn.execute(query, params)
        self._conn.commit()
        return cursor

    def fetchone(self, query, params=()):
        cursor = self._conn.execute(query, params)
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def fetchall(self, query, params=()):
        cursor = self._conn.execute(query, params)
        return [dict(r) for r in cursor.fetchall()]


@pytest.fixture
def backend():
    return InMemoryBackend()


@pytest.fixture
def store(backend):
    return AgentRunStore(backend)


def _make_run(**kwargs) -> MockRun:
    defaults = {
        "run_id": "run-abc123",
        "agent": "builder",
        "task": "Build the feature",
        "status": MockStatus.COMPLETED,
        "result": "Feature built successfully",
        "started_at": time.time() - 10,
        "completed_at": time.time(),
    }
    defaults.update(kwargs)
    return MockRun(**defaults)


class TestSaveRun:
    def test_save_basic_run(self, store, backend):
        run = _make_run()
        store.save_run(run)

        row = backend.fetchone("SELECT * FROM agent_runs WHERE run_id = ?", ("run-abc123",))
        assert row is not None
        assert row["agent"] == "builder"
        assert row["status"] == "completed"
        assert row["result"] == "Feature built successfully"

    def test_save_run_with_children(self, store, backend):
        run = _make_run(children=["run-child1", "run-child2"])
        store.save_run(run)

        row = backend.fetchone("SELECT * FROM agent_runs WHERE run_id = ?", ("run-abc123",))
        children = json.loads(row["children"])
        assert children == ["run-child1", "run-child2"]

    def test_save_run_with_parent(self, store, backend):
        run = _make_run(parent_id="run-parent1")
        store.save_run(run)

        row = backend.fetchone("SELECT * FROM agent_runs WHERE run_id = ?", ("run-abc123",))
        assert row["parent_id"] == "run-parent1"

    def test_save_run_with_config(self, store, backend):
        config = MockConfig(timeout_seconds=600, model="gpt-4")
        run = _make_run(config=config)
        store.save_run(run)

        row = backend.fetchone("SELECT * FROM agent_runs WHERE run_id = ?", ("run-abc123",))
        config_data = json.loads(row["config_json"])
        assert config_data["timeout_seconds"] == 600
        assert config_data["model"] == "gpt-4"

    def test_save_run_upsert(self, store, backend):
        """Saving same run_id twice updates the row."""
        run = _make_run(result="v1")
        store.save_run(run)

        run.result = "v2"
        store.save_run(run)

        row = backend.fetchone("SELECT * FROM agent_runs WHERE run_id = ?", ("run-abc123",))
        assert row["result"] == "v2"

    def test_save_run_none_result(self, store, backend):
        run = _make_run(result=None)
        store.save_run(run)

        row = backend.fetchone("SELECT * FROM agent_runs WHERE run_id = ?", ("run-abc123",))
        assert row["result"] is None

    def test_save_run_with_error(self, store, backend):
        run = _make_run(status=MockStatus.FAILED, error="Timeout exceeded")
        store.save_run(run)

        row = backend.fetchone("SELECT * FROM agent_runs WHERE run_id = ?", ("run-abc123",))
        assert row["status"] == "failed"
        assert row["error"] == "Timeout exceeded"

    def test_save_truncates_long_task(self, store, backend):
        run = _make_run(task="x" * 5000)
        store.save_run(run)

        row = backend.fetchone("SELECT * FROM agent_runs WHERE run_id = ?", ("run-abc123",))
        assert len(row["task"]) == 2000

    def test_save_truncates_long_result(self, store, backend):
        run = _make_run(result="y" * 10000)
        store.save_run(run)

        row = backend.fetchone("SELECT * FROM agent_runs WHERE run_id = ?", ("run-abc123",))
        assert len(row["result"]) == 5000

    def test_save_run_with_bad_config(self, store, backend):
        """Config with unserializable values falls back to {}."""
        run = _make_run()
        bad_config = MagicMock()
        bad_config.timeout_seconds = float("inf")  # Not JSON serializable
        run.config = bad_config
        store.save_run(run)

        row = backend.fetchone("SELECT * FROM agent_runs WHERE run_id = ?", ("run-abc123",))
        assert row["config_json"] == "{}"

    def test_save_run_no_config(self, store, backend):
        run = _make_run()
        run.config = None
        store.save_run(run)

        row = backend.fetchone("SELECT * FROM agent_runs WHERE run_id = ?", ("run-abc123",))
        assert row["config_json"] == "{}"

    def test_save_run_string_status(self, store, backend):
        """Status without .value attribute still works."""
        run = _make_run()
        run.status = "completed"  # plain string
        store.save_run(run)

        row = backend.fetchone("SELECT * FROM agent_runs WHERE run_id = ?", ("run-abc123",))
        assert row["status"] == "completed"


class TestGetRun:
    def test_get_existing_run(self, store):
        store.save_run(_make_run())
        row = store.get_run("run-abc123")
        assert row is not None
        assert row["agent"] == "builder"

    def test_get_nonexistent_run(self, store):
        assert store.get_run("run-nope") is None

    def test_get_run_parses_children(self, store):
        store.save_run(_make_run(children=["c1", "c2"]))
        row = store.get_run("run-abc123")
        assert row["children"] == ["c1", "c2"]


class TestListRuns:
    def test_list_all_runs(self, store):
        store.save_run(_make_run(run_id="run-1", started_at=100))
        store.save_run(_make_run(run_id="run-2", started_at=200))
        runs = store.list_runs()
        assert len(runs) == 2
        assert runs[0]["run_id"] == "run-2"  # newest first

    def test_list_filter_by_status(self, store):
        store.save_run(_make_run(run_id="run-1", status=MockStatus.COMPLETED))
        store.save_run(_make_run(run_id="run-2", status=MockStatus.FAILED))
        runs = store.list_runs(status="completed")
        assert len(runs) == 1
        assert runs[0]["run_id"] == "run-1"

    def test_list_filter_by_agent(self, store):
        store.save_run(_make_run(run_id="run-1", agent="builder"))
        store.save_run(_make_run(run_id="run-2", agent="tester"))
        runs = store.list_runs(agent="tester")
        assert len(runs) == 1
        assert runs[0]["agent"] == "tester"

    def test_list_combined_filters(self, store):
        store.save_run(_make_run(run_id="run-1", agent="builder", status=MockStatus.COMPLETED))
        store.save_run(_make_run(run_id="run-2", agent="builder", status=MockStatus.FAILED))
        store.save_run(_make_run(run_id="run-3", agent="tester", status=MockStatus.COMPLETED))
        runs = store.list_runs(agent="builder", status="completed")
        assert len(runs) == 1
        assert runs[0]["run_id"] == "run-1"

    def test_list_respects_limit(self, store):
        for i in range(10):
            store.save_run(_make_run(run_id=f"run-{i}", started_at=float(i)))
        runs = store.list_runs(limit=3)
        assert len(runs) == 3

    def test_list_empty(self, store):
        assert store.list_runs() == []

    def test_list_parses_children(self, store):
        store.save_run(_make_run(children=["c1"]))
        runs = store.list_runs()
        assert runs[0]["children"] == ["c1"]


class TestDeleteOlderThan:
    def test_delete_old_runs(self, store):
        now = time.time()
        store.save_run(_make_run(run_id="run-old", completed_at=now - 7200))
        store.save_run(_make_run(run_id="run-new", completed_at=now))
        deleted = store.delete_older_than(now - 3600)
        assert deleted == 1
        assert store.get_run("run-old") is None
        assert store.get_run("run-new") is not None

    def test_delete_skips_incomplete_runs(self, store):
        """Runs with completed_at=0 are not deleted."""
        store.save_run(_make_run(run_id="run-active", completed_at=0))
        deleted = store.delete_older_than(time.time())
        assert deleted == 0

    def test_delete_none_to_delete(self, store):
        assert store.delete_older_than(time.time()) == 0


class TestCount:
    def test_count_all(self, store):
        store.save_run(_make_run(run_id="run-1"))
        store.save_run(_make_run(run_id="run-2"))
        assert store.count() == 2

    def test_count_by_status(self, store):
        store.save_run(_make_run(run_id="run-1", status=MockStatus.COMPLETED))
        store.save_run(_make_run(run_id="run-2", status=MockStatus.FAILED))
        assert store.count(status="completed") == 1
        assert store.count(status="failed") == 1

    def test_count_empty(self, store):
        assert store.count() == 0

    def test_count_nonexistent_status(self, store):
        store.save_run(_make_run())
        assert store.count(status="cancelled") == 0
