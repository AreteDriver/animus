"""Benchmark: SQLite FTS5 vs ChromaDB memory backend performance.

Compares store, search, delete, and get_stats operations across backends.
ChromaDB tests are skipped if the optional dependency is not installed.

Run with:
    pytest tests/test_memory_benchmarks.py -v --benchmark-only
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from animus_bootstrap.intelligence.memory_backends.sqlite_backend import (
    SQLiteMemoryBackend,
)

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

# ---------------------------------------------------------------------------
# ChromaDB conditional import
# ---------------------------------------------------------------------------

try:
    from animus_bootstrap.intelligence.memory_backends.chromadb_backend import (
        ChromaDBMemoryBackend,
    )

    HAS_CHROMADB = True
except (ImportError, RuntimeError):
    HAS_CHROMADB = False

skip_no_chromadb = pytest.mark.skipif(
    not HAS_CHROMADB,
    reason="chromadb not installed — skipping ChromaDB benchmarks",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_MEMORIES = 100
NUM_DELETES = 50

SAMPLE_MEMORIES: list[tuple[str, str, dict]] = [
    (
        "episodic",
        f"User discussed project architecture with focus on microservices pattern #{i}",
        {"source": "conversation", "session_id": f"sess-{i}", "importance": i % 5},
    )
    for i in range(NUM_MEMORIES)
]

SEARCH_QUERIES = [
    "microservices architecture",
    "project discussion",
    "user conversation",
    "pattern design",
    "session focus",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro) -> object:  # noqa: ANN001
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


def _seed_sqlite(db_path: Path) -> tuple[SQLiteMemoryBackend, list[str]]:
    """Create and seed a SQLite backend with NUM_MEMORIES entries."""
    backend = SQLiteMemoryBackend(db_path / "bench.db")
    ids: list[str] = []
    for mem_type, content, meta in SAMPLE_MEMORIES:
        mid = _run(backend.store(mem_type, content, meta))
        ids.append(mid)  # type: ignore[arg-type]
    return backend, ids


def _seed_chromadb() -> tuple[ChromaDBMemoryBackend, list[str]]:
    """Create and seed a ChromaDB backend (in-memory) with NUM_MEMORIES entries."""
    backend = ChromaDBMemoryBackend()  # in-memory, no persist_directory
    ids: list[str] = []
    for mem_type, content, meta in SAMPLE_MEMORIES:
        mid = _run(backend.store(mem_type, content, meta))
        ids.append(mid)
    return backend, ids


# ===================================================================
# SQLite FTS5 Benchmarks
# ===================================================================


class TestSQLiteBenchmarks:
    """Benchmark suite for the SQLite FTS5 memory backend."""

    def test_store(self, tmp_path: Path, benchmark: BenchmarkFixture) -> None:
        """Benchmark: insert 100 memories into SQLite."""
        backend = SQLiteMemoryBackend(tmp_path / "store_bench.db")

        counter = iter(range(NUM_MEMORIES))

        def store_one() -> str:
            idx = next(counter)
            mem_type, content, meta = SAMPLE_MEMORIES[idx]
            return _run(backend.store(mem_type, content, meta))  # type: ignore[return-value]

        benchmark.pedantic(store_one, iterations=NUM_MEMORIES, rounds=1)
        backend.close()

    def test_search(self, tmp_path: Path, benchmark: BenchmarkFixture) -> None:
        """Benchmark: search across 100 stored memories in SQLite."""
        backend, _ids = _seed_sqlite(tmp_path)

        query_iter = iter(SEARCH_QUERIES * 20)  # 100 queries

        def search_one() -> list[dict]:
            q = next(query_iter)
            return _run(backend.search(q, limit=10))  # type: ignore[return-value]

        benchmark.pedantic(search_one, iterations=NUM_MEMORIES, rounds=1)
        backend.close()

    def test_delete(self, tmp_path: Path, benchmark: BenchmarkFixture) -> None:
        """Benchmark: delete 50 memories from SQLite."""
        backend, ids = _seed_sqlite(tmp_path)
        delete_ids = iter(ids[:NUM_DELETES])

        def delete_one() -> bool:
            mid = next(delete_ids)
            return _run(backend.delete(mid))  # type: ignore[return-value]

        benchmark.pedantic(delete_one, iterations=NUM_DELETES, rounds=1)
        backend.close()

    def test_get_stats(self, tmp_path: Path, benchmark: BenchmarkFixture) -> None:
        """Benchmark: get_stats on populated SQLite backend."""
        backend, _ids = _seed_sqlite(tmp_path)

        def get_stats() -> dict:
            return _run(backend.get_stats())  # type: ignore[return-value]

        benchmark(get_stats)
        backend.close()


# ===================================================================
# ChromaDB Benchmarks
# ===================================================================


@skip_no_chromadb
class TestChromaDBBenchmarks:
    """Benchmark suite for the ChromaDB memory backend."""

    def test_store(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark: insert 100 memories into ChromaDB."""
        backend = ChromaDBMemoryBackend()

        counter = iter(range(NUM_MEMORIES))

        def store_one() -> str:
            idx = next(counter)
            mem_type, content, meta = SAMPLE_MEMORIES[idx]
            return _run(backend.store(mem_type, content, meta))  # type: ignore[return-value]

        benchmark.pedantic(store_one, iterations=NUM_MEMORIES, rounds=1)
        backend.close()

    def test_search(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark: search across 100 stored memories in ChromaDB."""
        backend, _ids = _seed_chromadb()

        query_iter = iter(SEARCH_QUERIES * 20)

        def search_one() -> list[dict]:
            q = next(query_iter)
            return _run(backend.search(q, limit=10))  # type: ignore[return-value]

        benchmark.pedantic(search_one, iterations=NUM_MEMORIES, rounds=1)
        backend.close()

    def test_delete(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark: delete 50 memories from ChromaDB."""
        backend, ids = _seed_chromadb()
        delete_ids = iter(ids[:NUM_DELETES])

        def delete_one() -> bool:
            mid = next(delete_ids)
            return _run(backend.delete(mid))  # type: ignore[return-value]

        benchmark.pedantic(delete_one, iterations=NUM_DELETES, rounds=1)
        backend.close()

    def test_get_stats(self, benchmark: BenchmarkFixture) -> None:
        """Benchmark: get_stats on populated ChromaDB backend."""
        backend, _ids = _seed_chromadb()

        def get_stats() -> dict:
            return _run(backend.get_stats())  # type: ignore[return-value]

        benchmark(get_stats)
        backend.close()
