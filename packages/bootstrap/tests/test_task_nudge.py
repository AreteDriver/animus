"""Tests for task nudge proactive checker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from animus_bootstrap.intelligence.proactive.checks import tasks as task_checks
from animus_bootstrap.intelligence.tools.builtin.task_store import TaskStore


@pytest.fixture()
def store(tmp_path):
    """Return a TaskStore backed by tmp_path."""
    s = TaskStore(tmp_path / "tasks.db")
    yield s
    s.close()


@pytest.fixture(autouse=True)
def _wire_store(store):
    """Wire and clean up task store for each test."""
    task_checks.set_task_store(store)
    yield
    task_checks.set_task_store(None)


class TestTaskNudgeChecker:
    @pytest.mark.asyncio
    async def test_no_store_returns_none(self):
        task_checks.set_task_store(None)
        result = await task_checks.task_nudge_checker()
        assert result is None

    @pytest.mark.asyncio
    async def test_no_tasks_returns_none(self):
        result = await task_checks.task_nudge_checker()
        assert result is None

    @pytest.mark.asyncio
    async def test_no_due_dates_returns_none(self, store):
        store.create("no deadline")
        result = await task_checks.task_nudge_checker()
        assert result is None

    @pytest.mark.asyncio
    async def test_overdue_tasks(self, store):
        past = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        store.create("overdue thing", due_date=past, priority="high")
        result = await task_checks.task_nudge_checker()
        assert result is not None
        assert "Overdue tasks" in result
        assert "overdue thing" in result

    @pytest.mark.asyncio
    async def test_upcoming_tasks(self, store):
        soon = (datetime.now(UTC) + timedelta(hours=12)).isoformat()
        store.create("upcoming thing", due_date=soon)
        result = await task_checks.task_nudge_checker()
        assert result is not None
        assert "Upcoming tasks" in result
        assert "upcoming thing" in result

    @pytest.mark.asyncio
    async def test_both_overdue_and_upcoming(self, store):
        past = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        soon = (datetime.now(UTC) + timedelta(hours=12)).isoformat()
        store.create("late", due_date=past)
        store.create("soon", due_date=soon)
        result = await task_checks.task_nudge_checker()
        assert "Overdue" in result
        assert "Upcoming" in result

    @pytest.mark.asyncio
    async def test_completed_tasks_excluded(self, store):
        past = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        task_id = store.create("done", due_date=past)
        store.complete(task_id)
        result = await task_checks.task_nudge_checker()
        assert result is None

    @pytest.mark.asyncio
    async def test_far_future_excluded(self, store):
        far = (datetime.now(UTC) + timedelta(days=30)).isoformat()
        store.create("way later", due_date=far)
        result = await task_checks.task_nudge_checker()
        assert result is None


class TestGetTaskNudgeCheck:
    def test_returns_proactive_check(self):
        check = task_checks.get_task_nudge_check()
        assert check.name == "task_nudge"
        assert check.schedule == "0 */2 * * *"
        assert check.channels == ["webchat"]
        assert check.priority == "low"
