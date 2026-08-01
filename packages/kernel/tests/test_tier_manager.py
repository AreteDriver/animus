"""Tests for TierManager — promotion/demotion and HOT cap enforcement.

Covers:
- WARM → HOT auto-promotion on threshold access
- COLD → WARM on any access
- Explicit promote/demote via MemoryLayer
- Periodic review: stale WARM → COLD, HOT cap enforcement
- rerank_for_tier boosts HOT memories
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from animus_kernel.memory.layer import MemoryLayer
from animus_kernel.memory.types import Memory, MemoryTier, MemoryType


@pytest.fixture
def layer(tmp_path: Path) -> MemoryLayer:
    return MemoryLayer(data_dir=tmp_path, backend="local")


def _make_memory(
    content: str = "m", tier: MemoryTier = MemoryTier.WARM, access_count: int = 0
) -> Memory:
    return Memory.create(
        content=content,
        memory_type=MemoryType.SEMANTIC,
        tier=tier,
        access_count=access_count,
        last_accessed=datetime.now(),
    )


class TestOnAccess:
    def test_warm_promotes_to_hot_at_threshold(self, layer: MemoryLayer):
        mem = _make_memory(tier=MemoryTier.WARM, access_count=2)
        layer.store.store(mem)
        layer.tier_manager.on_access(mem)
        assert mem.tier == MemoryTier.HOT
        assert mem.access_count == 3

    def test_warm_stays_warm_below_threshold(self, layer: MemoryLayer):
        mem = _make_memory(tier=MemoryTier.WARM, access_count=1)
        layer.store.store(mem)
        layer.tier_manager.on_access(mem)
        assert mem.tier == MemoryTier.WARM
        assert mem.access_count == 2

    def test_cold_promotes_to_warm(self, layer: MemoryLayer):
        mem = _make_memory(tier=MemoryTier.COLD, access_count=0)
        layer.store.store(mem)
        layer.tier_manager.on_access(mem)
        assert mem.tier == MemoryTier.WARM

    def test_hot_stays_hot(self, layer: MemoryLayer):
        mem = _make_memory(tier=MemoryTier.HOT, access_count=10)
        layer.store.store(mem)
        layer.tier_manager.on_access(mem)
        assert mem.tier == MemoryTier.HOT

    def test_access_updates_last_accessed(self, layer: MemoryLayer):
        before = datetime.now()
        mem = _make_memory(tier=MemoryTier.WARM, access_count=2)
        layer.store.store(mem)
        layer.tier_manager.on_access(mem)
        assert mem.last_accessed is not None
        assert mem.last_accessed >= before


class TestReview:
    def test_stale_warm_demoted_to_cold(self, layer: MemoryLayer):
        stale = _make_memory(tier=MemoryTier.WARM)
        stale.last_accessed = datetime.now() - timedelta(days=40)
        layer.store.store(stale)

        demoted, promoted = layer.tier_manager.review()
        assert demoted >= 1
        assert stale.tier == MemoryTier.COLD

    def test_fresh_warm_not_demoted(self, layer: MemoryLayer):
        fresh = _make_memory(tier=MemoryTier.WARM)
        fresh.last_accessed = datetime.now()
        layer.store.store(fresh)

        demoted, promoted = layer.tier_manager.review()
        assert demoted == 0
        assert fresh.tier == MemoryTier.WARM

    def test_hot_cap_demotes_oldest(self, layer: MemoryLayer):
        # Create 52 HOT memories
        for i in range(52):
            mem = _make_memory(f"hot-{i}", tier=MemoryTier.HOT, access_count=10)
            mem.last_accessed = datetime.now() - timedelta(seconds=i)
            layer.store.store(mem)

        demoted, _ = layer.tier_manager.review()
        assert demoted == 2  # 52 - 50 cap

    def test_hot_cap_keeps_most_recent(self, layer: MemoryLayer):
        for i in range(52):
            mem = _make_memory(f"hot-{i}", tier=MemoryTier.HOT, access_count=10)
            mem.last_accessed = datetime.now() - timedelta(seconds=i)
            layer.store.store(mem)

        layer.tier_manager.review()
        hot_ids = {m.id for m in layer.store.list_all() if m.tier == MemoryTier.HOT}
        # The 50 most recent should be kept
        assert len(hot_ids) == 50


class TestRerank:
    def test_hot_ranked_above_warm(self, layer: MemoryLayer):
        hot = _make_memory("hot", tier=MemoryTier.HOT, access_count=1)
        warm = _make_memory("warm", tier=MemoryTier.WARM, access_count=5)
        ranked = layer.tier_manager.rerank_for_tier([warm, hot])
        assert ranked[0].tier == MemoryTier.HOT
        assert ranked[1].tier == MemoryTier.WARM

    def test_access_count_tiebreaker(self, layer: MemoryLayer):
        hot1 = _make_memory("a", tier=MemoryTier.HOT, access_count=3)
        hot2 = _make_memory("b", tier=MemoryTier.HOT, access_count=5)
        ranked = layer.tier_manager.rerank_for_tier([hot1, hot2])
        assert ranked[0].content == "b"
        assert ranked[1].content == "a"

    def test_cold_at_bottom(self, layer: MemoryLayer):
        cold = _make_memory("cold", tier=MemoryTier.COLD, access_count=100)
        warm = _make_memory("warm", tier=MemoryTier.WARM, access_count=1)
        ranked = layer.tier_manager.rerank_for_tier([cold, warm])
        assert ranked[0].tier == MemoryTier.WARM
        assert ranked[1].tier == MemoryTier.COLD


class TestExplicitPromoteDemote:
    def test_promote_cold_to_warm(self, layer: MemoryLayer):
        mem = _make_memory("c", tier=MemoryTier.COLD)
        layer.store.store(mem)
        ok = layer.promote_memory(mem.id)
        assert ok is True
        assert layer.get_memory(mem.id).tier == MemoryTier.WARM

    def test_promote_warm_to_hot(self, layer: MemoryLayer):
        mem = _make_memory("w", tier=MemoryTier.WARM)
        layer.store.store(mem)
        ok = layer.promote_memory(mem.id)
        assert ok is True
        assert layer.get_memory(mem.id).tier == MemoryTier.HOT

    def test_promote_hot_is_noop(self, layer: MemoryLayer):
        mem = _make_memory("h", tier=MemoryTier.HOT)
        layer.store.store(mem)
        ok = layer.promote_memory(mem.id)
        assert ok is True  # Returns True (already at max)
        assert layer.get_memory(mem.id).tier == MemoryTier.HOT

    def test_demote_hot_to_warm(self, layer: MemoryLayer):
        mem = _make_memory("h", tier=MemoryTier.HOT)
        layer.store.store(mem)
        ok = layer.demote_memory(mem.id)
        assert ok is True
        assert layer.get_memory(mem.id).tier == MemoryTier.WARM

    def test_demote_warm_to_cold(self, layer: MemoryLayer):
        mem = _make_memory("w", tier=MemoryTier.WARM)
        layer.store.store(mem)
        ok = layer.demote_memory(mem.id)
        assert ok is True
        # Use store.retrieve, not get_memory, to avoid triggering on_access promotion
        assert layer.store.retrieve(mem.id).tier == MemoryTier.COLD

    def test_demote_cold_is_noop(self, layer: MemoryLayer):
        mem = _make_memory("c", tier=MemoryTier.COLD)
        layer.store.store(mem)
        ok = layer.demote_memory(mem.id)
        assert ok is True
        # Use store.retrieve, not get_memory, to avoid triggering on_access promotion
        assert layer.store.retrieve(mem.id).tier == MemoryTier.COLD

    def test_promote_missing_returns_false(self, layer: MemoryLayer):
        assert layer.promote_memory("nonexistent") is False

    def test_demote_missing_returns_false(self, layer: MemoryLayer):
        assert layer.demote_memory("nonexistent") is False

    def test_partial_id_match_promote(self, layer: MemoryLayer):
        mem = _make_memory("partial", tier=MemoryTier.COLD)
        layer.store.store(mem)
        prefix = mem.id[:8]
        ok = layer.promote_memory(prefix)
        assert ok is True
        assert layer.get_memory(mem.id).tier == MemoryTier.WARM
