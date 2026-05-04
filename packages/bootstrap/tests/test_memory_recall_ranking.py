"""Tests for MemoryManager hit-rate reranking, recall logging, and observers."""

from __future__ import annotations

import asyncio

import pytest

from animus_bootstrap.intelligence.memory import MemoryManager
from animus_bootstrap.intelligence.memory_scores import MemoryScoreStore
from animus_bootstrap.intelligence.recall_log import RecallLog


class FakeBackend:
    """In-memory backend that returns canned results per type."""

    def __init__(self, fixtures: dict[str, list[dict]] | None = None) -> None:
        self._fixtures = fixtures or {}
        self.search_calls: list[tuple[str, str, int]] = []

    async def search(self, query: str, memory_type: str = "all", limit: int = 5) -> list[dict]:
        self.search_calls.append((query, memory_type, limit))
        return [dict(r) for r in self._fixtures.get(memory_type, [])][:limit]

    async def store(self, memory_type: str, content: str, metadata: dict) -> str:
        return "stored"

    async def delete(self, memory_id: str) -> bool:
        return True

    async def get_stats(self) -> dict:
        return {"user_preferences": {"theme": "dark"}}

    def close(self) -> None:
        pass


@pytest.fixture
def stores(tmp_path):
    rl = RecallLog(tmp_path / "recall.db")
    ss = MemoryScoreStore(tmp_path / "scores.db")
    yield rl, ss
    rl.close()
    ss.close()


@pytest.fixture
def fixtures() -> dict[str, list[dict]]:
    return {
        "episodic": [
            {"id": "e1", "content": "first episodic", "type": "episodic"},
            {"id": "e2", "content": "second episodic", "type": "episodic"},
            {"id": "e3", "content": "third episodic", "type": "episodic"},
        ],
        "semantic": [
            {"id": "s1", "content": "first semantic", "type": "semantic"},
        ],
        "procedural": [],
    }


class TestRecallLogging:
    @pytest.mark.asyncio
    async def test_recall_logs_query_and_memory_ids(self, stores, fixtures) -> None:
        rl, ss = stores
        backend = FakeBackend(fixtures)
        mm = MemoryManager(backend, recall_log=rl, score_store=ss)

        ctx = await mm.recall(query="hello", limit=3)

        assert ctx.recall_id is not None
        record = rl.get_recall(ctx.recall_id)
        assert record is not None
        assert record.query == "hello"
        # All 3 episodic + 1 semantic memories logged
        assert set(record.memory_ids) == {"e1", "e2", "e3", "s1"}

    @pytest.mark.asyncio
    async def test_no_recall_log_omits_id(self, stores, fixtures) -> None:
        _, ss = stores
        backend = FakeBackend(fixtures)
        mm = MemoryManager(backend, recall_log=None, score_store=ss)
        ctx = await mm.recall(query="hello")
        assert ctx.recall_id is None

    @pytest.mark.asyncio
    async def test_recall_returns_user_prefs(self, stores, fixtures) -> None:
        rl, ss = stores
        backend = FakeBackend(fixtures)
        mm = MemoryManager(backend, recall_log=rl, score_store=ss)
        ctx = await mm.recall(query="hello")
        assert ctx.user_prefs == {"theme": "dark"}

    @pytest.mark.asyncio
    async def test_context_hint_propagated(self, stores, fixtures) -> None:
        rl, ss = stores
        backend = FakeBackend(fixtures)
        mm = MemoryManager(backend, recall_log=rl, score_store=ss)
        ctx = await mm.recall(query="q", context_hint="gateway:webchat")
        record = rl.get_recall(ctx.recall_id)
        assert record is not None
        assert record.context_hint == "gateway:webchat"


class TestReranking:
    @pytest.mark.asyncio
    async def test_no_score_store_passes_through(self, stores, fixtures) -> None:
        rl, _ = stores
        backend = FakeBackend(fixtures)
        mm = MemoryManager(backend, recall_log=rl, score_store=None)
        ctx = await mm.recall(query="hi", limit=3)
        # Order unchanged from backend
        assert ctx.episodic == ["first episodic", "second episodic", "third episodic"]

    @pytest.mark.asyncio
    async def test_alpha_zero_passes_through(self, stores, fixtures) -> None:
        rl, ss = stores
        ss.record_outcome("e3", 1.0)
        ss.record_outcome("e3", 1.0)
        backend = FakeBackend(fixtures)
        mm = MemoryManager(backend, recall_log=rl, score_store=ss, rerank_alpha=0.0)
        ctx = await mm.recall(query="hi", limit=3)
        # Order unchanged despite e3 having a positive score
        assert ctx.episodic[0] == "first episodic"

    @pytest.mark.asyncio
    async def test_high_score_memory_promoted(self, stores) -> None:
        rl, ss = stores
        # Two-result fixture; second position is 0.5 → boost of 2x flips it to 1.0 vs 1.0
        # Use a slightly higher score to clear the tie.
        for _ in range(3):
            ss.record_outcome("b", 1.0)
        backend = FakeBackend(
            {
                "episodic": [
                    {"id": "a", "content": "neutral first", "type": "episodic"},
                    {"id": "b", "content": "boosted second", "type": "episodic"},
                ]
            }
        )
        mm = MemoryManager(backend, recall_log=rl, score_store=ss, rerank_alpha=1.5)
        ctx = await mm.recall(query="hi", limit=2)
        assert ctx.episodic[0] == "boosted second"

    @pytest.mark.asyncio
    async def test_negative_score_demotes(self, stores) -> None:
        rl, ss = stores
        for _ in range(3):
            ss.record_outcome("a", -1.0)
        backend = FakeBackend(
            {
                "episodic": [
                    {"id": "a", "content": "penalized first", "type": "episodic"},
                    {"id": "b", "content": "neutral second", "type": "episodic"},
                ]
            }
        )
        mm = MemoryManager(backend, recall_log=rl, score_store=ss, rerank_alpha=1.5)
        ctx = await mm.recall(query="hi", limit=2)
        assert ctx.episodic[0] == "neutral second"

    @pytest.mark.asyncio
    async def test_rerank_handles_empty_results(self, stores) -> None:
        rl, ss = stores
        backend = FakeBackend({"episodic": [], "semantic": [], "procedural": []})
        mm = MemoryManager(backend, recall_log=rl, score_store=ss)
        ctx = await mm.recall(query="hi")
        assert ctx.episodic == []

    @pytest.mark.asyncio
    async def test_rerank_handles_single_result(self, stores) -> None:
        rl, ss = stores
        ss.record_outcome("only", 1.0)
        backend = FakeBackend({"episodic": [{"id": "only", "content": "x", "type": "episodic"}]})
        mm = MemoryManager(backend, recall_log=rl, score_store=ss, rerank_alpha=1.0)
        ctx = await mm.recall(query="hi")
        assert ctx.episodic == ["x"]


class TestRecallObserver:
    @pytest.mark.asyncio
    async def test_observer_propagates_positive_signal_to_all_memories(
        self, stores, fixtures
    ) -> None:
        rl, ss = stores
        backend = FakeBackend(fixtures)
        mm = MemoryManager(
            backend,
            recall_log=rl,
            score_store=ss,
            action_observers=[lambda since, until: 2],
            outcome_window_seconds=1,
        )
        try:
            await mm.recall(query="hello")
            await asyncio.sleep(1.2)
            for mid in ("e1", "e2", "e3", "s1"):
                score = ss.get(mid)
                assert score is not None
                assert score.sum_outcome == 1.0
        finally:
            await mm.shutdown()

    @pytest.mark.asyncio
    async def test_observer_records_zero_when_silent(self, stores, fixtures) -> None:
        rl, ss = stores
        backend = FakeBackend(fixtures)
        mm = MemoryManager(
            backend,
            recall_log=rl,
            score_store=ss,
            action_observers=[lambda since, until: 0],
            outcome_window_seconds=1,
        )
        try:
            await mm.recall(query="hello")
            await asyncio.sleep(1.2)
            score = ss.get("e1")
            assert score is not None
            assert score.n_recalls == 1
            assert score.sum_outcome == 0.0
        finally:
            await mm.shutdown()

    @pytest.mark.asyncio
    async def test_no_observer_means_no_score_change(self, stores, fixtures) -> None:
        rl, ss = stores
        backend = FakeBackend(fixtures)
        mm = MemoryManager(backend, recall_log=rl, score_store=ss, outcome_window_seconds=1)
        try:
            await mm.recall(query="hello")
            await asyncio.sleep(1.2)
            assert ss.get("e1") is None
        finally:
            await mm.shutdown()

    @pytest.mark.asyncio
    async def test_async_observer_supported(self, stores, fixtures) -> None:
        rl, ss = stores

        async def observer(since, until):
            return 3

        backend = FakeBackend(fixtures)
        mm = MemoryManager(
            backend,
            recall_log=rl,
            score_store=ss,
            action_observers=[observer],
            outcome_window_seconds=1,
        )
        try:
            await mm.recall(query="hello")
            await asyncio.sleep(1.2)
            assert ss.get("e1") is not None
            assert ss.get("e1").sum_outcome == 1.0
        finally:
            await mm.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_cancels_pending_observers(self, stores, fixtures) -> None:
        rl, ss = stores
        backend = FakeBackend(fixtures)
        mm = MemoryManager(
            backend,
            recall_log=rl,
            score_store=ss,
            action_observers=[lambda since, until: 1],
            outcome_window_seconds=300,
        )
        await mm.recall(query="hello")
        assert mm._observer_tasks
        await mm.shutdown()
        assert not mm._observer_tasks

    @pytest.mark.asyncio
    async def test_observer_failure_does_not_block_others(self, stores, fixtures) -> None:
        rl, ss = stores

        def bad(since, until):
            raise RuntimeError("nope")

        def good(since, until):
            return 4

        backend = FakeBackend(fixtures)
        mm = MemoryManager(
            backend,
            recall_log=rl,
            score_store=ss,
            action_observers=[bad, good],
            outcome_window_seconds=1,
        )
        try:
            await mm.recall(query="hello")
            await asyncio.sleep(1.2)
            assert ss.get("e1") is not None
            assert ss.get("e1").sum_outcome == 1.0
        finally:
            await mm.shutdown()


class TestExplicitFeedback:
    def test_explicit_feedback_propagates_to_all_memories(self, stores) -> None:
        rl, ss = stores
        rid = rl.log_recall(
            query="q",
            memory_entries=[("a", "episodic"), ("b", "semantic")],
        )
        backend = FakeBackend()
        mm = MemoryManager(backend, recall_log=rl, score_store=ss)
        mm.record_explicit_feedback(rid, 1.0)
        assert ss.get("a").sum_outcome == 1.0
        assert ss.get("b").sum_outcome == 1.0

    def test_explicit_feedback_no_op_without_stores(self, stores) -> None:
        rl, _ = stores
        rid = rl.log_recall(query="q", memory_entries=[("a", "episodic")])
        backend = FakeBackend()
        mm = MemoryManager(backend, recall_log=None, score_store=None)
        # Should not raise
        mm.record_explicit_feedback(rid, 1.0)

    def test_explicit_feedback_unknown_recall_no_op(self, stores) -> None:
        rl, ss = stores
        backend = FakeBackend()
        mm = MemoryManager(backend, recall_log=rl, score_store=ss)
        mm.record_explicit_feedback("unknown-rid", 1.0)
        assert ss.stats()["scored_memories"] == 0
