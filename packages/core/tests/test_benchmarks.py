"""Performance benchmarks for Animus Core operations.

Run: pytest tests/test_benchmarks.py --benchmark-only
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("pytest_benchmark")

from animus.cognitive import detect_mode, should_delegate_to_gorgon
from animus.entities import Entity, EntityMemory, EntityType
from animus.memory import Conversation, LocalMemoryStore, Memory, MemoryType

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _make_memory(i: int) -> Memory:
    """Create a synthetic Memory for benchmarks."""
    return Memory(
        id=str(uuid.uuid4()),
        content=f"Memory entry {i}: the quick brown fox jumps over the lazy dog #{i}",
        memory_type=MemoryType.SEMANTIC if i % 3 else MemoryType.EPISODIC,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        metadata={"source": "benchmark", "index": i},
        tags=[f"tag_{i % 10}", "benchmark"],
        source="stated",
        confidence=0.8 + (i % 20) * 0.01,
    )


def _make_entity(i: int) -> Entity:
    """Create a synthetic Entity for benchmarks."""
    return Entity(
        id=str(uuid.uuid4()),
        name=f"Entity_{i}",
        entity_type=EntityType.PERSON if i % 3 == 0 else EntityType.PROJECT,
        aliases=[f"alias_{i}", f"ent{i}"],
        attributes={"role": f"role_{i % 5}", "team": f"team_{i % 8}"},
        created_at=datetime.now(),
        updated_at=datetime.now(),
        mention_count=i,
    )


# ═══════════════════════════════════════════════════════════════════
# Benchmarks
# ═══════════════════════════════════════════════════════════════════


class TestMemorySerdeBenchmark:
    """Memory serialization/deserialization throughput."""

    def test_memory_serde_500(self, benchmark):
        """Benchmark: Memory.to_dict/from_dict x500."""
        memories = [_make_memory(i) for i in range(500)]

        def serde_round_trip():
            for mem in memories:
                d = mem.to_dict()
                Memory.from_dict(d)

        benchmark(serde_round_trip)


class TestConversationSerdeBenchmark:
    """Conversation serialization with many messages."""

    def test_conversation_serde_50msg(self, benchmark):
        """Benchmark: 20 conversations x 50 messages serde."""
        convos = []
        for _ in range(20):
            c = Conversation.new()
            for j in range(50):
                role = "user" if j % 2 == 0 else "assistant"
                c.add_message(role, f"Message {j}: discussing topic in detail.")
            convos.append(c)

        def serde_round_trip():
            for c in convos:
                d = c.to_dict()
                Conversation.from_dict(d)

        benchmark(serde_round_trip)


class TestDetectModeBenchmark:
    """Mode detection regex throughput."""

    def test_detect_mode_1000(self, benchmark):
        """Benchmark: detect_mode x1000 mixed prompts."""
        prompts = [
            "think about the architecture",
            "analyze this code",
            "compare these approaches",
            "help me write a function",
            "what time is it",
            "explain how databases work",
            "create a plan for deployment",
            "debug this error message",
            "review the pull request",
            "quick question about python",
        ]

        def detect_all():
            for _ in range(100):
                for p in prompts:
                    detect_mode(p)

        benchmark(detect_all)


class TestDelegationDetectionBenchmark:
    """Delegation heuristic throughput."""

    def test_delegation_detection_500(self, benchmark):
        """Benchmark: should_delegate_to_gorgon x500 mixed prompts."""
        prompts = [
            "write tests for the authentication module in the codebase",
            "refactor the database layer to use async operations",
            "implement a new API endpoint for user management",
            "what is python",
            "hello world",
            "analyze the repository architecture and refactor",
            "deploy the application to production servers",
            "simple question about syntax",
            "build a CI pipeline for the project repository",
            "tell me a joke",
        ]

        def detect_all():
            for _ in range(50):
                for p in prompts:
                    should_delegate_to_gorgon(p)

        benchmark(detect_all)


class TestEntitySearchBenchmark:
    """EntityMemory search throughput."""

    def test_entity_search_200(self, benchmark, tmp_path: Path):
        """Benchmark: EntityMemory.search_entities over 200 entities."""
        em = EntityMemory(tmp_path / "entities")
        for i in range(200):
            ent = _make_entity(i)
            em._entities[ent.id] = ent

        queries = ["Entity_5", "alias_10", "ent42", "Entity_100", "unknown"]

        def search_all():
            for q in queries:
                em.search_entities(q, limit=10)

        benchmark(search_all)


class TestLocalSearchBenchmark:
    """LocalMemoryStore search throughput."""

    def test_local_search_500(self, benchmark, tmp_path: Path):
        """Benchmark: LocalMemoryStore.search over 500 memories."""
        store = LocalMemoryStore(tmp_path / "memories")
        for i in range(500):
            mem = _make_memory(i)
            store._memories[mem.id] = mem

        queries = ["quick brown", "Memory entry 42", "lazy dog", "benchmark", "nonexistent"]

        def search_all():
            for q in queries:
                store.search(q, limit=20)

        benchmark(search_all)
