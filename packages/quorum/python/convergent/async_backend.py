"""Async wrapper for any synchronous GraphBackend.

Uses ``asyncio.to_thread()`` to offload blocking backend calls — zero new
dependencies. Works with SQLiteBackend, PythonGraphBackend, or any
future backend implementing the GraphBackend protocol.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from convergent.intent import Intent, InterfaceSpec


class AsyncGraphBackend(Protocol):
    """Protocol for async intent graph backends."""

    async def publish(self, intent: Intent) -> float: ...
    async def query_all(self, min_stability: float | None = None) -> list[Intent]: ...
    async def query_by_agent(self, agent_id: str) -> list[Intent]: ...
    async def find_overlapping(
        self, specs: list[InterfaceSpec], exclude_agent: str, min_stability: float
    ) -> list[Intent]: ...
    async def count(self) -> int: ...


class AsyncBackendWrapper:
    """Wraps a synchronous ``GraphBackend`` with async methods via ``asyncio.to_thread()``.

    Example::

        from convergent.sqlite_backend import SQLiteBackend
        from convergent.async_backend import AsyncBackendWrapper

        backend = AsyncBackendWrapper(SQLiteBackend("intents.db"))
        stability = await backend.publish(intent)
        all_intents = await backend.query_all()
        await backend.close()

    Args:
        backend: Any synchronous GraphBackend instance.
    """

    def __init__(self, backend: object) -> None:
        self._backend = backend

    async def publish(self, intent: Intent) -> float:
        return await asyncio.to_thread(self._backend.publish, intent)  # type: ignore[union-attr]

    async def query_all(self, min_stability: float | None = None) -> list[Intent]:
        return await asyncio.to_thread(self._backend.query_all, min_stability)  # type: ignore[union-attr]

    async def query_by_agent(self, agent_id: str) -> list[Intent]:
        return await asyncio.to_thread(self._backend.query_by_agent, agent_id)  # type: ignore[union-attr]

    async def find_overlapping(
        self,
        specs: list[InterfaceSpec],
        exclude_agent: str,
        min_stability: float,
    ) -> list[Intent]:
        return await asyncio.to_thread(  # type: ignore[union-attr]
            self._backend.find_overlapping, specs, exclude_agent, min_stability
        )

    async def count(self) -> int:
        return await asyncio.to_thread(self._backend.count)  # type: ignore[union-attr]

    async def close(self) -> None:
        """Close the underlying backend if it has a close() method."""
        if hasattr(self._backend, "close"):
            await asyncio.to_thread(self._backend.close)  # type: ignore[union-attr]
