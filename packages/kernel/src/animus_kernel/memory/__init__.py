"""
Animus Memory Layer — package facade.

Preserves the import surface of the previous single-file module so every
existing `from animus_kernel.memory import X` call site works unchanged. Internals
split into:

    types     — Memory / MemoryType / MemorySource / SemanticFact /
                Procedure / Message / Conversation
    stores/   — MemoryStore ABC + LocalMemoryStore + ChromaMemoryStore
    fusion    — _rrf_fuse (Reciprocal Rank Fusion)
    layer     — MemoryLayer façade

Phase 1: Structured memory with types, tags, confidence, and export/import.
"""

from __future__ import annotations

from animus_kernel.memory.layer import MemoryLayer
from animus_kernel.memory.stores import ChromaMemoryStore, DurableMemoryStore, LocalMemoryStore, MemoryStore
from animus_kernel.memory.tier import TierManager
from animus_kernel.memory.types import (
    Conversation,
    Memory,
    MemorySource,
    MemoryTier,
    MemoryType,
    Message,
    Procedure,
    SemanticFact,
)

__all__ = [
    "ChromaMemoryStore",
    "Conversation",
    "DurableMemoryStore",
    "LocalMemoryStore",
    "Memory",
    "MemoryLayer",
    "MemorySource",
    "MemoryStore",
    "MemoryTier",
    "MemoryType",
    "Message",
    "Procedure",
    "SemanticFact",
    "TierManager",
]
