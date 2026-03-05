"""Tests for task management tools."""

from __future__ import annotations

import pytest

from animus_bootstrap.intelligence.tools.builtin import task_ctl
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
    task_ctl.set_task_store(store)
    yield
    task_ctl.set_task_store(None)


class TestTaskCreate:
    @pytest.mark.asyncio
    async def test_create_basic(self, store):
        result = await task_ctl._task_create("test task")
        assert "Task created" in result
        assert "test task" in result
        assert store.list_all()

    @pytest.mark.asyncio
    async def test_create_with_priority(self, store):
        result = await task_ctl._task_create("urgent", priority="urgent")
        assert "urgent" in result
        tasks = store.list_all()
        assert tasks[0]["priority"] == "urgent"

    @pytest.mark.asyncio
    async def test_create_invalid_priority(self):
        result = await task_ctl._task_create("bad", priority="mega")
        assert "Invalid priority" in result

    @pytest.mark.asyncio
    async def test_create_with_due_date(self, store):
        result = await task_ctl._task_create("timed", due_date="2025-12-31T23:59:59")
        assert "Task created" in result

    @pytest.mark.asyncio
    async def test_create_no_store(self):
        task_ctl.set_task_store(None)
        result = await task_ctl._task_create("no store")
        assert "not available" in result


class TestTaskList:
    @pytest.mark.asyncio
    async def test_list_empty(self):
        result = await task_ctl._task_list()
        assert "No tasks" in result

    @pytest.mark.asyncio
    async def test_list_with_tasks(self, store):
        store.create("task a")
        store.create("task b")
        result = await task_ctl._task_list()
        assert "Tasks (2)" in result
        assert "task a" in result
        assert "task b" in result

    @pytest.mark.asyncio
    async def test_list_filter_status(self, store):
        id1 = store.create("pending")
        store.create("also pending")
        store.complete(id1)
        result = await task_ctl._task_list(status="pending")
        assert "Tasks (1)" in result
        assert "also pending" in result

    @pytest.mark.asyncio
    async def test_list_invalid_status(self):
        result = await task_ctl._task_list(status="bogus")
        assert "Invalid status" in result

    @pytest.mark.asyncio
    async def test_list_no_store(self):
        task_ctl.set_task_store(None)
        result = await task_ctl._task_list()
        assert "not available" in result


class TestTaskComplete:
    @pytest.mark.asyncio
    async def test_complete_existing(self, store):
        task_id = store.create("doable")
        result = await task_ctl._task_complete(task_id)
        assert "completed" in result

    @pytest.mark.asyncio
    async def test_complete_nonexistent(self):
        result = await task_ctl._task_complete("nope")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_complete_no_store(self):
        task_ctl.set_task_store(None)
        result = await task_ctl._task_complete("any")
        assert "not available" in result


class TestTaskDelete:
    @pytest.mark.asyncio
    async def test_delete_existing(self, store):
        task_id = store.create("deletable")
        result = await task_ctl._task_delete(task_id)
        assert "deleted" in result
        assert store.get(task_id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        result = await task_ctl._task_delete("nope")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_delete_no_store(self):
        task_ctl.set_task_store(None)
        result = await task_ctl._task_delete("any")
        assert "not available" in result


class TestGetTaskTools:
    def test_returns_four_tools(self):
        tools = task_ctl.get_task_tools()
        assert len(tools) == 4
        names = {t.name for t in tools}
        assert names == {"task_create", "task_list", "task_complete", "task_delete"}

    def test_all_have_category(self):
        for tool in task_ctl.get_task_tools():
            assert tool.category == "task"

    def test_all_have_handlers(self):
        for tool in task_ctl.get_task_tools():
            assert callable(tool.handler)


class TestGetTaskStore:
    def test_get_store_returns_store(self, store):
        assert task_ctl.get_task_store() is store

    def test_get_store_returns_none(self):
        task_ctl.set_task_store(None)
        assert task_ctl.get_task_store() is None
