"""Tests for AsyncBackendWrapper — async wrapper over sync backends."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("pytest_asyncio")

from convergent.async_backend import AsyncBackendWrapper
from convergent.intent import (
    Evidence,
    Intent,
    InterfaceKind,
    InterfaceSpec,
)
from convergent.resolver import PythonGraphBackend
from convergent.sqlite_backend import SQLiteBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(name: str, tags: list[str] | None = None) -> InterfaceSpec:
    return InterfaceSpec(
        name=name,
        kind=InterfaceKind.FUNCTION,
        signature="(x: int) -> int",
        module_path="mod",
        tags=tags or [],
    )


def _make_intent(
    agent_id: str,
    intent: str,
    provides: list[InterfaceSpec] | None = None,
    evidence: list[Evidence] | None = None,
) -> Intent:
    return Intent(
        agent_id=agent_id,
        intent=intent,
        provides=provides or [],
        evidence=evidence or [],
    )


# ---------------------------------------------------------------------------
# SQLiteBackend wrapper
# ---------------------------------------------------------------------------


class TestAsyncSQLiteBackend:
    @pytest.fixture
    def wrapper(self):
        backend = SQLiteBackend(":memory:")
        w = AsyncBackendWrapper(backend)
        yield w
        backend.close()

    async def test_publish(self, wrapper):
        intent = _make_intent("a", "task", provides=[_make_spec("fn")])
        stability = await wrapper.publish(intent)
        assert isinstance(stability, float)
        assert 0.0 <= stability <= 1.0

    async def test_query_all(self, wrapper):
        await wrapper.publish(_make_intent("a", "task", provides=[_make_spec("fn")]))
        results = await wrapper.query_all()
        assert len(results) == 1
        assert results[0].intent == "task"

    async def test_query_all_min_stability(self, wrapper):
        await wrapper.publish(_make_intent("a", "task", provides=[_make_spec("fn")]))
        assert len(await wrapper.query_all(min_stability=0.9)) == 0
        assert len(await wrapper.query_all(min_stability=0.0)) == 1

    async def test_query_by_agent(self, wrapper):
        await wrapper.publish(_make_intent("a", "t1", provides=[_make_spec("fn1")]))
        await wrapper.publish(_make_intent("b", "t2", provides=[_make_spec("fn2")]))
        assert len(await wrapper.query_by_agent("a")) == 1
        assert len(await wrapper.query_by_agent("z")) == 0

    async def test_find_overlapping(self, wrapper):
        await wrapper.publish(_make_intent("a", "task", provides=[_make_spec("shared")]))
        specs = [_make_spec("shared")]
        overlaps = await wrapper.find_overlapping(specs, "b", 0.0)
        assert len(overlaps) >= 1

    async def test_count(self, wrapper):
        assert await wrapper.count() == 0
        await wrapper.publish(_make_intent("a", "t", provides=[_make_spec("fn")]))
        assert await wrapper.count() == 1

    async def test_close(self, wrapper):
        # Should not raise — SQLiteBackend has close()
        await wrapper.close()


# ---------------------------------------------------------------------------
# PythonGraphBackend wrapper
# ---------------------------------------------------------------------------


class TestAsyncPythonBackend:
    @pytest.fixture
    def wrapper(self):
        return AsyncBackendWrapper(PythonGraphBackend())

    async def test_publish_and_query(self, wrapper):
        intent = _make_intent("a", "task", provides=[_make_spec("fn")])
        stability = await wrapper.publish(intent)
        assert isinstance(stability, float)
        results = await wrapper.query_all()
        assert len(results) == 1

    async def test_close_safe_without_close_method(self, wrapper):
        # PythonGraphBackend has no close() — should not raise
        await wrapper.close()


# ---------------------------------------------------------------------------
# Close safety
# ---------------------------------------------------------------------------


class TestCloseSafety:
    async def test_close_with_closeable_backend(self):
        backend = SQLiteBackend(":memory:")
        wrapper = AsyncBackendWrapper(backend)
        await wrapper.close()  # Should call backend.close()

    async def test_close_with_non_closeable_backend(self):
        wrapper = AsyncBackendWrapper(PythonGraphBackend())
        await wrapper.close()  # No close() method — should not raise


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    async def test_concurrent_publishes_python_backend(self):
        """Multiple concurrent publishes to PythonGraphBackend should not deadlock."""
        wrapper = AsyncBackendWrapper(PythonGraphBackend())

        intents = [
            _make_intent(f"agent-{i}", f"task-{i}", provides=[_make_spec(f"fn_{i}")])
            for i in range(10)
        ]

        stabilities = await asyncio.gather(*[wrapper.publish(i) for i in intents])
        assert len(stabilities) == 10
        assert all(isinstance(s, float) for s in stabilities)
        assert await wrapper.count() == 10

    async def test_sequential_async_sqlite(self):
        """Sequential async operations on SQLiteBackend work correctly."""
        backend = SQLiteBackend(":memory:")
        wrapper = AsyncBackendWrapper(backend)

        for i in range(5):
            await wrapper.publish(
                _make_intent(f"a-{i}", f"t-{i}", provides=[_make_spec(f"fn_{i}")])
            )

        results = await wrapper.query_all()
        assert len(results) == 5
        assert await wrapper.count() == 5

        backend.close()
