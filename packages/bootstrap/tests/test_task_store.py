"""Tests for TaskStore."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from animus_bootstrap.intelligence.tools.builtin.task_store import TaskStore


@pytest.fixture()
def store(tmp_path):
    """Return a TaskStore backed by tmp_path."""
    s = TaskStore(tmp_path / "tasks.db")
    yield s
    s.close()


class TestCreate:
    def test_create_returns_id(self, store):
        task_id = store.create("test task")
        assert isinstance(task_id, str)
        assert len(task_id) == 8

    def test_create_with_defaults(self, store):
        task_id = store.create("test task")
        task = store.get(task_id)
        assert task["name"] == "test task"
        assert task["description"] == ""
        assert task["status"] == "pending"
        assert task["priority"] == "normal"
        assert task["due_date"] is None

    def test_create_with_all_fields(self, store):
        task_id = store.create(
            name="urgent thing",
            description="do it now",
            priority="urgent",
            due_date="2025-03-15T09:00:00",
        )
        task = store.get(task_id)
        assert task["name"] == "urgent thing"
        assert task["description"] == "do it now"
        assert task["priority"] == "urgent"
        assert task["due_date"] == "2025-03-15T09:00:00"

    def test_create_multiple_unique_ids(self, store):
        ids = {store.create(f"task {i}") for i in range(10)}
        assert len(ids) == 10


class TestGet:
    def test_get_existing(self, store):
        task_id = store.create("hello")
        task = store.get(task_id)
        assert task is not None
        assert task["id"] == task_id

    def test_get_nonexistent(self, store):
        assert store.get("nope") is None


class TestListAll:
    def test_empty_store(self, store):
        assert store.list_all() == []

    def test_list_all_tasks(self, store):
        store.create("a")
        store.create("b")
        tasks = store.list_all()
        assert len(tasks) == 2

    def test_filter_by_status(self, store):
        id1 = store.create("pending task")
        store.create("another pending")
        store.complete(id1)
        pending = store.list_all(status="pending")
        assert len(pending) == 1
        assert pending[0]["name"] == "another pending"

    def test_filter_completed(self, store):
        id1 = store.create("task")
        store.complete(id1)
        completed = store.list_all(status="completed")
        assert len(completed) == 1
        assert completed[0]["status"] == "completed"

    def test_filter_no_results(self, store):
        store.create("task")
        assert store.list_all(status="completed") == []


class TestComplete:
    def test_complete_existing(self, store):
        task_id = store.create("task")
        assert store.complete(task_id) is True
        task = store.get(task_id)
        assert task["status"] == "completed"

    def test_complete_nonexistent(self, store):
        assert store.complete("nope") is False

    def test_complete_updates_timestamp(self, store):
        task_id = store.create("task")
        task_before = store.get(task_id)
        store.complete(task_id)
        task_after = store.get(task_id)
        assert task_after["updated"] >= task_before["updated"]


class TestDelete:
    def test_delete_existing(self, store):
        task_id = store.create("task")
        assert store.delete(task_id) is True
        assert store.get(task_id) is None

    def test_delete_nonexistent(self, store):
        assert store.delete("nope") is False

    def test_delete_removes_from_list(self, store):
        task_id = store.create("task")
        store.delete(task_id)
        assert store.list_all() == []


class TestOverdue:
    def test_no_overdue(self, store):
        store.create("no due date")
        assert store.get_overdue() == []

    def test_overdue_task(self, store):
        past = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        store.create("overdue", due_date=past)
        overdue = store.get_overdue()
        assert len(overdue) == 1
        assert overdue[0]["name"] == "overdue"

    def test_completed_not_overdue(self, store):
        past = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        task_id = store.create("done", due_date=past)
        store.complete(task_id)
        assert store.get_overdue() == []

    def test_future_not_overdue(self, store):
        future = (datetime.now(UTC) + timedelta(days=30)).isoformat()
        store.create("future", due_date=future)
        assert store.get_overdue() == []


class TestUpcoming:
    def test_no_upcoming(self, store):
        store.create("no due date")
        assert store.get_upcoming() == []

    def test_upcoming_within_window(self, store):
        soon = (datetime.now(UTC) + timedelta(hours=12)).isoformat()
        store.create("soon", due_date=soon)
        upcoming = store.get_upcoming(hours=24)
        assert len(upcoming) == 1

    def test_beyond_window(self, store):
        far = (datetime.now(UTC) + timedelta(hours=48)).isoformat()
        store.create("far", due_date=far)
        assert store.get_upcoming(hours=24) == []

    def test_completed_not_upcoming(self, store):
        soon = (datetime.now(UTC) + timedelta(hours=12)).isoformat()
        task_id = store.create("done", due_date=soon)
        store.complete(task_id)
        assert store.get_upcoming(hours=24) == []

    def test_overdue_not_upcoming(self, store):
        past = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        store.create("past", due_date=past)
        assert store.get_upcoming(hours=24) == []


class TestPersistence:
    def test_close_and_reopen(self, tmp_path):
        db_path = tmp_path / "tasks.db"
        s = TaskStore(db_path)
        s.create("persist me")
        s.close()

        s2 = TaskStore(db_path)
        tasks = s2.list_all()
        assert len(tasks) == 1
        assert tasks[0]["name"] == "persist me"
        s2.close()
