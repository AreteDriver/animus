"""Tests for D2 memory tiering (HOT/WARM/COLD + access tracking)."""

from datetime import datetime, timedelta
from unittest.mock import patch

from animus.memory import MemoryLayer, MemoryTier, TierManager
from animus.memory.stores.local import LocalMemoryStore
from animus.memory.types import Memory


class TestMemoryTierDefaults:
    def test_memory_defaults_to_warm(self):
        mem = Memory.create(content="test")
        assert mem.tier == MemoryTier.WARM
        assert mem.access_count == 0
        assert mem.last_accessed is None

    def test_to_dict_roundtrips_tier_fields(self):
        mem = Memory.create(
            content="test",
            tier=MemoryTier.HOT,
            access_count=5,
            last_accessed=datetime(2024, 1, 1, 12, 0),
        )
        d = mem.to_dict()
        assert d["tier"] == "hot"
        assert d["access_count"] == 5
        assert d["last_accessed"] == "2024-01-01T12:00:00"

        restored = Memory.from_dict(d)
        assert restored.tier == MemoryTier.HOT
        assert restored.access_count == 5
        assert restored.last_accessed == datetime(2024, 1, 1, 12, 0)

    def test_from_dict_defaults_for_missing_tier_fields(self):
        d = {
            "id": "abc",
            "content": "legacy",
            "memory_type": "semantic",
            "created_at": "2024-01-01T12:00:00",
            "updated_at": "2024-01-01T12:00:00",
            "metadata": {},
            "tags": [],
            "source": "stated",
            "confidence": 1.0,
            "sensitivity": "public",
        }
        mem = Memory.from_dict(d)
        assert mem.tier == MemoryTier.WARM
        assert mem.access_count == 0
        assert mem.last_accessed is None


class TestLocalStorePersist:
    def test_local_store_persists_tier(self, tmp_path):
        store = LocalMemoryStore(tmp_path)
        mem = Memory.create(
            content="hello",
            tier=MemoryTier.HOT,
            access_count=3,
            last_accessed=datetime(2024, 6, 1, 10, 0),
        )
        store.store(mem)

        # Re-instantiate to force reload from disk
        store2 = LocalMemoryStore(tmp_path)
        loaded = store2.retrieve(mem.id)
        assert loaded is not None
        assert loaded.tier == MemoryTier.HOT
        assert loaded.access_count == 3
        assert loaded.last_accessed == datetime(2024, 6, 1, 10, 0)


class TestRecallAccessTracking:
    def test_recall_increments_access_count(self, tmp_path):
        layer = MemoryLayer(tmp_path, backend="local")
        mem = layer.remember(content="python facts", tags=["python"])
        assert mem.access_count == 0

        results = layer.recall("python", tags=["python"])
        assert len(results) == 1
        assert results[0].access_count == 1

        # Second recall should bump again
        results = layer.recall("python", tags=["python"])
        assert results[0].access_count == 2

    def test_recall_updates_last_accessed(self, tmp_path):
        layer = MemoryLayer(tmp_path, backend="local")
        mem = layer.remember(content="time test")
        assert mem.last_accessed is None

        before = datetime.now()
        results = layer.recall("time")
        after = datetime.now()

        assert len(results) == 1
        assert results[0].last_accessed is not None
        assert before <= results[0].last_accessed <= after

    def test_recall_tier_filter(self, tmp_path):
        layer = MemoryLayer(tmp_path, backend="local")
        hot = layer.remember(content="hot topic", tier=MemoryTier.HOT)
        warm = layer.remember(content="warm topic", tier=MemoryTier.WARM)
        cold = layer.remember(content="cold topic", tier=MemoryTier.COLD)

        hot_results = layer.recall("topic", tier=MemoryTier.HOT)
        assert len(hot_results) == 1
        assert hot_results[0].id == hot.id

        warm_results = layer.recall("topic", tier=MemoryTier.WARM)
        assert len(warm_results) == 1
        assert warm_results[0].id == warm.id

        cold_results = layer.recall("topic", tier=MemoryTier.COLD)
        assert len(cold_results) == 1
        assert cold_results[0].id == cold.id


class TestPromotionTriggers:
    def test_warm_promotes_to_hot_on_threshold(self, tmp_path):
        layer = MemoryLayer(tmp_path, backend="local")
        mem = layer.remember(content="threshold test")
        assert mem.tier == MemoryTier.WARM
        assert mem.access_count == 0

        # 3 recalls should trigger promotion
        for _ in range(3):
            layer.recall("threshold")

        # Re-fetch from store directly to avoid triggering another access
        refreshed = layer.store.retrieve(mem.id)
        assert refreshed is not None
        assert refreshed.tier == MemoryTier.HOT
        assert refreshed.access_count == 3

    def test_cold_promotes_to_warm_on_any_access(self, tmp_path):
        layer = MemoryLayer(tmp_path, backend="local")
        mem = layer.remember(content="cold rescue", tier=MemoryTier.COLD)
        assert mem.tier == MemoryTier.COLD

        layer.recall("cold rescue")

        refreshed = layer.store.retrieve(mem.id)
        assert refreshed is not None
        assert refreshed.tier == MemoryTier.WARM
        assert refreshed.access_count == 1

    def test_hot_stays_hot_past_threshold(self, tmp_path):
        layer = MemoryLayer(tmp_path, backend="local")
        mem = layer.remember(content="already hot", tier=MemoryTier.HOT)

        for _ in range(5):
            layer.recall("already hot")

        refreshed = layer.store.retrieve(mem.id)
        assert refreshed is not None
        assert refreshed.tier == MemoryTier.HOT
        assert refreshed.access_count == 5


class TestExplicitPromotionDemotion:
    def test_promote_memory_warm_to_hot(self, tmp_path):
        layer = MemoryLayer(tmp_path, backend="local")
        mem = layer.remember(content="promote me", tier=MemoryTier.WARM)
        assert mem.tier == MemoryTier.WARM

        assert layer.promote_memory(mem.id) is True
        refreshed = layer.get_memory(mem.id)
        assert refreshed is not None
        assert refreshed.tier == MemoryTier.HOT

    def test_promote_memory_cold_to_warm(self, tmp_path):
        layer = MemoryLayer(tmp_path, backend="local")
        mem = layer.remember(content="promote me", tier=MemoryTier.COLD)
        assert layer.promote_memory(mem.id) is True
        refreshed = layer.store.retrieve(mem.id)
        assert refreshed is not None
        assert refreshed.tier == MemoryTier.WARM

    def test_promote_memory_hot_stays_hot(self, tmp_path):
        layer = MemoryLayer(tmp_path, backend="local")
        mem = layer.remember(content="already hot", tier=MemoryTier.HOT)
        assert layer.promote_memory(mem.id) is True  # No-op, returns True
        refreshed = layer.get_memory(mem.id)
        assert refreshed is not None
        assert refreshed.tier == MemoryTier.HOT

    def test_demote_memory_hot_to_warm(self, tmp_path):
        layer = MemoryLayer(tmp_path, backend="local")
        mem = layer.remember(content="demote me", tier=MemoryTier.HOT)
        assert layer.demote_memory(mem.id) is True
        refreshed = layer.store.retrieve(mem.id)
        assert refreshed is not None
        assert refreshed.tier == MemoryTier.WARM

    def test_demote_memory_warm_to_cold(self, tmp_path):
        layer = MemoryLayer(tmp_path, backend="local")
        mem = layer.remember(content="demote me", tier=MemoryTier.WARM)
        assert layer.demote_memory(mem.id) is True
        refreshed = layer.store.retrieve(mem.id)
        assert refreshed is not None
        assert refreshed.tier == MemoryTier.COLD

    def test_promote_bad_id(self, tmp_path):
        layer = MemoryLayer(tmp_path, backend="local")
        assert layer.promote_memory("nonexistent") is False

    def test_demote_bad_id(self, tmp_path):
        layer = MemoryLayer(tmp_path, backend="local")
        assert layer.demote_memory("nonexistent") is False


class TestTierReview:
    def test_review_demotes_stale_warm_to_cold(self, tmp_path):
        layer = MemoryLayer(tmp_path, backend="local")
        mem = layer.remember(content="stale memory")
        # Manually backdate last_accessed to 31 days ago
        mem.last_accessed = datetime.now() - timedelta(days=31)
        mem.access_count = 1
        layer.update_memory(mem)

        demoted, promoted = layer.run_tier_review()
        assert demoted == 1
        assert promoted == 0

        refreshed = layer.store.retrieve(mem.id)
        assert refreshed is not None
        assert refreshed.tier == MemoryTier.COLD

    def test_review_keeps_fresh_warm(self, tmp_path):
        layer = MemoryLayer(tmp_path, backend="local")
        mem = layer.remember(content="fresh memory")
        # last_accessed defaults to None, which review treats as "never accessed"
        # so we set it to recent
        mem.last_accessed = datetime.now() - timedelta(days=1)
        layer.update_memory(mem)

        demoted, promoted = layer.run_tier_review()
        assert demoted == 0
        assert promoted == 0

        refreshed = layer.store.retrieve(mem.id)
        assert refreshed is not None
        assert refreshed.tier == MemoryTier.WARM

    def test_review_enforces_hot_cap(self, tmp_path):
        layer = MemoryLayer(tmp_path, backend="local")
        # Create 3 HOT memories with different last_accessed times
        for i in range(3):
            mem = layer.remember(content=f"hot {i}", tier=MemoryTier.HOT)
            mem.last_accessed = datetime.now() - timedelta(days=i)
            layer.update_memory(mem)

        # Lower cap to 2 for testability
        with patch.object(TierManager, "HOT_LIMIT", 2):
            demoted, promoted = layer.run_tier_review()

        assert demoted == 1
        # The oldest (hot 2, 2 days ago) should be demoted
        oldest = layer.get_memory([m for m in layer.store.list_all() if "hot 2" in m.content][0].id)
        assert oldest is not None
        assert oldest.tier == MemoryTier.WARM


class TestRerank:
    def test_hot_boost_in_rerank(self, tmp_path):
        layer = MemoryLayer(tmp_path, backend="local")
        cold = layer.remember(content="cold content", tier=MemoryTier.COLD)
        warm = layer.remember(content="warm content", tier=MemoryTier.WARM)
        hot = layer.remember(content="hot content", tier=MemoryTier.HOT)

        # Give them different access counts
        cold.access_count = 10
        warm.access_count = 5
        hot.access_count = 1
        layer.update_memory(cold)
        layer.update_memory(warm)
        layer.update_memory(hot)

        # Use tier_manager.rerank_for_tier directly to avoid side effects
        # from on_access during recall.
        raw = [cold, warm, hot]
        results = layer.tier_manager.rerank_for_tier(raw)
        assert results[0].tier == MemoryTier.HOT
        assert results[1].tier == MemoryTier.WARM
        assert results[2].tier == MemoryTier.COLD

    def test_rerank_tiebreak_by_access_count(self, tmp_path):
        layer = MemoryLayer(tmp_path, backend="local")
        hot_a = layer.remember(content="hot A", tier=MemoryTier.HOT)
        hot_b = layer.remember(content="hot B", tier=MemoryTier.HOT)
        hot_a.access_count = 5
        hot_b.access_count = 10
        layer.update_memory(hot_a)
        layer.update_memory(hot_b)

        raw = [hot_a, hot_b]
        results = layer.tier_manager.rerank_for_tier(raw)
        assert results[0].id == hot_b.id  # higher access_count
        assert results[1].id == hot_a.id


class TestGetMemoryAccess:
    def test_get_memory_tracks_access(self, tmp_path):
        layer = MemoryLayer(tmp_path, backend="local")
        mem = layer.remember(content="get me")
        assert mem.access_count == 0

        found = layer.get_memory(mem.id)
        assert found is not None
        assert found.access_count == 1

        # Verify persistence
        store_mem = layer.store.retrieve(mem.id)
        assert store_mem is not None
        assert store_mem.access_count == 1


class TestStatistics:
    def test_statistics_include_tier_breakdown(self, tmp_path):
        layer = MemoryLayer(tmp_path, backend="local")
        layer.remember(content="hot", tier=MemoryTier.HOT)
        layer.remember(content="warm", tier=MemoryTier.WARM)
        layer.remember(content="cold", tier=MemoryTier.COLD)

        stats = layer.get_statistics()
        assert stats["by_tier"] == {"hot": 1, "warm": 1, "cold": 1}
