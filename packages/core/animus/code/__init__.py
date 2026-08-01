"""Animus Code Utilities — AST-aware chunking and codebase indexing.

Extracted from memboot (archived). Provides semantic chunking for Python,
Markdown, YAML, and JSON, plus credential redaction via
``animus.memory.redaction``.

Example:
    from animus.code import chunk_file, CodeChunk, ChunkType

    chunks = chunk_file(Path("src/main.py"))
    for c in chunks:
        print(c.chunk_type, c.metadata.get("name"))
"""

from __future__ import annotations

from animus.code.chunking import (
    ChunkingConfig,
    ChunkType,
    CodeChunk,
    chunk_codebase,
    chunk_file,
)

__all__ = [
    "ChunkType",
    "CodeChunk",
    "ChunkingConfig",
    "chunk_codebase",
    "chunk_file",
]
