"""TierManager — promotion/demotion policy for HOT/WARM/COLD memory tiers.

D2 implementation.  A memory node's tier drifts based on access frequency,
recency, and explicit caller intent.  The manager is stateless (all state
lives on the Memory object and in the store); it can be reconstructed cheaply
from the layer at any time.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from animus_kernel.logger import get_logger
from animus_kernel.memory.types import Memory, MemoryTier

if TYPE_CHECKING:
    from animus_kernel.memory.layer import MemoryLayer

logger = get_logger("memory.tier")


class TierManager:
    """Temperature-based memory retention and retrieval policy.

    Rules (hardened in code, not config — policy changes require commit):

    - All new memories default to **WARM**.
    - **HOT** memories are boosted in search ranking.
    - A WARM memory promoted to HOT after ``PROMOTION_THRESHOLD`` accesses.
    - A WARM memory demoted to COLD after ``DEMOTION_DAYS`` without access.
    - A COLD memory retrieved explicitly promotes back to WARM.
    - HOT cap: ``HOT_LIMIT`` — oldest HOT (by last_accessed) demotes to WARM.
    """

    PROMOTION_THRESHOLD: int = 3
    DEMOTION_DAYS: int = 30
    HOT_LIMIT: int = 50

    def __init__(self, layer: MemoryLayer) -> None:
        self.layer = layer

    def on_access(self, memory: Memory) -> None:
        """Record an access and apply auto-promotion rules."""
        memory.access_count += 1
        memory.last_accessed = datetime.now()
        memory.updated_at = datetime.now()

        # Auto-promote WARM → HOT on threshold
        if memory.tier == MemoryTier.WARM and memory.access_count >= self.PROMOTION_THRESHOLD:
            memory.tier = MemoryTier.HOT
            logger.info(f"Promoted {memory.id[:8]} to HOT (access_count={memory.access_count})")

        # COLD → WARM on any access
        if memory.tier == MemoryTier.COLD:
            memory.tier = MemoryTier.WARM
            logger.info(f"Promoted {memory.id[:8]} COLD→WARM (retrieval)")

        # Persist the change
        self.layer.update_memory(memory)

    def review(self) -> tuple[int, int]:
        """Run periodic tier review.

        Returns:
            (demoted_count, promoted_count)
        """
        demoted = 0
        promoted = 0
        cutoff = datetime.now() - timedelta(days=self.DEMOTION_DAYS)

        all_memories = self.layer.store.list_all()

        # Demote stale WARM → COLD
        for mem in all_memories:
            if mem.tier == MemoryTier.WARM:
                if mem.last_accessed is None or mem.last_accessed < cutoff:
                    mem.tier = MemoryTier.COLD
                    mem.updated_at = datetime.now()
                    self.layer.update_memory(mem)
                    demoted += 1
                    logger.info(f"Demoted {mem.id[:8]} WARM→COLD (idle since {mem.last_accessed})")

        # Enforce HOT cap — demote oldest HOT to WARM
        hot_memories = [m for m in all_memories if m.tier == MemoryTier.HOT]
        if len(hot_memories) > self.HOT_LIMIT:
            # Sort by last_accessed descending (most recent first)
            hot_memories.sort(key=lambda m: m.last_accessed or datetime.min, reverse=True)
            to_demote = hot_memories[self.HOT_LIMIT :]
            for mem in to_demote:
                mem.tier = MemoryTier.WARM
                mem.updated_at = datetime.now()
                self.layer.update_memory(mem)
                demoted += 1
                logger.info(f"Demoted {mem.id[:8]} HOT→WARM (HOT cap {self.HOT_LIMIT})")

        return demoted, promoted

    def rerank_for_tier(self, results: list[Memory]) -> list[Memory]:
        """Boost HOT memories in result ordering.

        Sort key: tier priority (HOT=3, WARM=2, COLD=1) as the primary key,
        then access_count descending as tie-breaker.

        This is a lightweight re-ranking — it does not replace the semantic
        relevance score, it only re-orders within the already-retrieved set.
        """
        tier_priority = {MemoryTier.HOT: 3, MemoryTier.WARM: 2, MemoryTier.COLD: 1}

        def _key(mem: Memory) -> tuple[int, int]:
            return (tier_priority.get(mem.tier, 0), mem.access_count)

        return sorted(results, key=_key, reverse=True)
