"""
Animus Memory Layer — package facade.

Preserves the import surface of the previous single-file module so every
existing `from animus.memory import X` call site works unchanged. Internals
split into:

    types     — Memory / MemoryType / MemorySource / SemanticFact /
                Procedure / Message / Conversation
    stores/   — MemoryStore ABC + LocalMemoryStore + ChromaMemoryStore
    fusion    — _rrf_fuse (Reciprocal Rank Fusion)
    layer     — MemoryLayer façade

Phase 1: Structured memory with types, tags, confidence, and export/import.
"""

from __future__ import annotations

from animus.memory.layer import MemoryLayer
from animus.memory.stores import ChromaMemoryStore, LocalMemoryStore, MemoryStore
from animus.memory.types import (
    Conversation,
    Memory,
    MemorySource,
    MemoryType,
    Message,
    Procedure,
    SemanticFact,
)

__all__ = [
    "ChromaMemoryStore",
    "Conversation",
    "LocalMemoryStore",
    "Memory",
    "MemoryLayer",
    "MemorySource",
    "MemoryStore",
    "MemoryType",
    "Message",
    "Procedure",
    "SemanticFact",
]
