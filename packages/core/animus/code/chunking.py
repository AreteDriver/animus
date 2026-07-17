"""AST-aware code chunking for Animus.

Ports memboot's semantic chunking into Animus conventions:
- Pure functions, no I/O except Path.read_text
- Integrates with ``animus.memory.redaction`` for credential scrubbing
- Returns dataclasses compatible with ``MemoryLayer.remember()``

Supports Python (AST-based), Markdown (header-based), YAML/JSON (key-based),
and generic sliding window fallback.
"""

from __future__ import annotations

import ast
import enum
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from animus.logging import get_logger
from animus.memory.redaction import redact

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

logger = get_logger("code.chunking")

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class ChunkType(enum.StrEnum):
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    MODULE = "module"
    MARKDOWN_SECTION = "markdown_section"
    YAML_KEY = "yaml_key"
    JSON_KEY = "json_key"
    WINDOW = "window"


@dataclass(frozen=True)
class CodeChunk:
    """A single semantic chunk extracted from source code."""

    content: str
    chunk_type: ChunkType
    start_line: int
    end_line: int
    source_path: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def to_memory_payload(self) -> dict:
        """Return a dict suitable for ``MemoryLayer.remember(metadata=...)``."""
        return {
            "chunk_type": self.chunk_type.value,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "source_path": self.source_path,
            **self.metadata,
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ChunkingConfig:
    """Tunable parameters for chunking."""

    max_chunk_tokens: int = 512
    overlap_tokens: int = 50

    @property
    def chars_per_chunk(self) -> int:
        """Rough character budget (~4 chars/token)."""
        return self.max_chunk_tokens * 4

    @property
    def overlap_chars(self) -> int:
        return self.overlap_tokens * 4


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_file(
    file_path: Path,
    *,
    config: ChunkingConfig | None = None,
    source_path: str = "",
    redact_credentials: bool = True,
) -> list[CodeChunk]:
    """Extract semantic chunks from *file_path*.

    Args:
        file_path: Filesystem path to read.
        config: Chunking parameters. Defaults to 512-token chunks.
        source_path: Logical source identifier (e.g. repo-relative path).
            If empty, falls back to ``str(file_path)``.
        redact_credentials: Whether to apply credential redaction.

    Returns:
        List of ``CodeChunk`` objects. Empty list on unreadable files.
    """
    config = config or ChunkingConfig()
    src = source_path or str(file_path)

    try:
        raw = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Cannot read %s: %s", file_path, exc)
        return []

    if not raw.strip():
        return []

    ext = file_path.suffix.lower()
    if ext == ".py":
        chunks = _chunk_python(raw, config)
    elif ext == ".md":
        chunks = _chunk_markdown(raw, config)
    elif ext in (".yaml", ".yml") and yaml is not None:
        chunks = _chunk_yaml(raw, config)
    elif ext == ".json":
        chunks = _chunk_json(raw, config)
    else:
        chunks = _chunk_window(raw, config)

    # Attach source path
    out: list[CodeChunk] = []
    for c in chunks:
        out.append(
            CodeChunk(
                content=c.content,
                chunk_type=c.chunk_type,
                start_line=c.start_line,
                end_line=c.end_line,
                source_path=src,
                metadata=c.metadata,
            )
        )

    if redact_credentials:
        out = [_redact_chunk(c) for c in out]

    return out


def chunk_codebase(
    root: Path,
    *,
    globs: list[str] | None = None,
    exclude: list[str] | None = None,
    config: ChunkingConfig | None = None,
    redact_credentials: bool = True,
) -> dict[str, list[CodeChunk]]:
    """Recursively chunk all matching files under *root*.

    Args:
        root: Base directory to scan.
        globs: Filename patterns to include (default: ``["*.py", "*.md"]``).
        exclude: Patterns to skip (default: ``["*/test*", "*/__pycache__/*",
            "*/node_modules/*", "*/.git/*"]``).
        config: Chunking parameters.
        redact_credentials: Apply credential redaction per-chunk.

    Returns:
        Mapping of relative-path string → list of chunks.
    """
    config = config or ChunkingConfig()
    globs = globs if globs is not None else ["*.py", "*.md"]
    exclude = exclude if exclude is not None else ["*/test*", "*/__pycache__/*", "*/node_modules/*", "*/.git/*"]

    results: dict[str, list[CodeChunk]] = {}
    for pattern in globs:
        for path in root.rglob(pattern):
            if any(path.match(p) for p in exclude):
                continue
            rel = str(path.relative_to(root))
            chunks = chunk_file(
                path,
                config=config,
                source_path=rel,
                redact_credentials=redact_credentials,
            )
            if chunks:
                results[rel] = chunks
    return results


# ---------------------------------------------------------------------------
# Chunkers by language
# ---------------------------------------------------------------------------


def _chunk_python(content: str, config: ChunkingConfig) -> list[CodeChunk]:
    """AST-based chunking for Python files."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _chunk_window(content, config)

    lines = content.splitlines(keepends=True)
    chunks: list[CodeChunk] = []
    covered: set[int] = set()

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = node.end_lineno or node.lineno
            chunks.append(
                CodeChunk(
                    content="".join(lines[start - 1 : end]).rstrip(),
                    chunk_type=ChunkType.FUNCTION,
                    start_line=start,
                    end_line=end,
                    metadata={"name": node.name},
                )
            )
            covered.update(range(start, end + 1))

        elif isinstance(node, ast.ClassDef):
            start = node.lineno
            end = node.end_lineno or node.lineno
            methods = [
                n
                for n in ast.iter_child_nodes(node)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            est_chars = (end - start + 1) * 25  # rough char estimate
            if methods and est_chars > config.chars_per_chunk:
                # Split large classes into method-level chunks
                for method in methods:
                    m_start = method.lineno
                    m_end = method.end_lineno or method.lineno
                    chunks.append(
                        CodeChunk(
                            content="".join(lines[m_start - 1 : m_end]).rstrip(),
                            chunk_type=ChunkType.METHOD,
                            start_line=m_start,
                            end_line=m_end,
                            metadata={"class": node.name, "name": method.name},
                        )
                    )
                    covered.update(range(m_start, m_end + 1))
                first_method = min(m.lineno for m in methods)
                if first_method > start + 1:
                    header = "".join(lines[start - 1 : first_method - 1])
                    if header.strip():
                        chunks.append(
                            CodeChunk(
                                content=header.rstrip(),
                                chunk_type=ChunkType.CLASS,
                                start_line=start,
                                end_line=first_method - 1,
                                metadata={"name": node.name},
                            )
                        )
                covered.update(range(start, end + 1))
            else:
                chunks.append(
                    CodeChunk(
                        content="".join(lines[start - 1 : end]).rstrip(),
                        chunk_type=ChunkType.CLASS,
                        start_line=start,
                        end_line=end,
                        metadata={"name": node.name},
                    )
                )
                covered.update(range(start, end + 1))

    # Module-level leftovers
    module_lines: list[str] = []
    module_start: int | None = None
    for i, line in enumerate(lines, 1):
        if i not in covered and line.strip() and not line.strip().startswith("#"):
            if module_start is None:
                module_start = i
            module_lines.append(line)

    if module_lines and module_start is not None:
        mod_text = "".join(module_lines).rstrip()
        if mod_text:
            chunks.append(
                CodeChunk(
                    content=mod_text,
                    chunk_type=ChunkType.MODULE,
                    start_line=module_start,
                    end_line=module_start + len(module_lines) - 1,
                )
            )

    return chunks if chunks else _chunk_window(content, config)


def _chunk_markdown(content: str, _config: ChunkingConfig) -> list[CodeChunk]:
    """Header-based chunking for Markdown."""
    header_pat = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    lines = content.split("\n")
    matches = list(header_pat.finditer(content))
    if not matches:
        return _chunk_window(content, _config)

    positions: list[tuple[int, str]] = []
    for m in matches:
        line_num = content[: m.start()].count("\n") + 1
        positions.append((line_num, m.group(0)))

    chunks: list[CodeChunk] = []
    for i, (line_num, header) in enumerate(positions):
        end_line = positions[i + 1][0] - 1 if i + 1 < len(positions) else len(lines)
        section = "\n".join(lines[line_num - 1 : end_line]).strip()
        if section:
            chunks.append(
                CodeChunk(
                    content=section,
                    chunk_type=ChunkType.MARKDOWN_SECTION,
                    start_line=line_num,
                    end_line=end_line,
                    metadata={"header": header.lstrip("#").strip()},
                )
            )

    if positions and positions[0][0] > 1:
        preamble = "\n".join(lines[: positions[0][0] - 1]).strip()
        if preamble:
            chunks.insert(
                0,
                CodeChunk(
                    content=preamble,
                    chunk_type=ChunkType.MARKDOWN_SECTION,
                    start_line=1,
                    end_line=positions[0][0] - 1,
                    metadata={"header": "preamble"},
                ),
            )

    return chunks


def _chunk_yaml(content: str, _config: ChunkingConfig) -> list[CodeChunk]:
    """Top-level key chunking for YAML."""
    if yaml is None:
        return _chunk_window(content, _config)
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError:
        return _chunk_window(content, _config)

    if not isinstance(data, dict):
        return _chunk_window(content, _config)

    lines = content.split("\n")
    key_positions: list[tuple[int, str]] = []
    for i, line in enumerate(lines, 1):
        match = re.match(r"^(\S+)\s*:", line)
        if match:
            key_positions.append((i, match.group(1)))

    chunks: list[CodeChunk] = []
    for i, (line_num, key) in enumerate(key_positions):
        end_line = key_positions[i + 1][0] - 1 if i + 1 < len(key_positions) else len(lines)
        section = "\n".join(lines[line_num - 1 : end_line]).strip()
        if section:
            chunks.append(
                CodeChunk(
                    content=section,
                    chunk_type=ChunkType.YAML_KEY,
                    start_line=line_num,
                    end_line=end_line,
                    metadata={"key": key},
                )
            )

    return chunks if chunks else _chunk_window(content, _config)


def _chunk_json(content: str, _config: ChunkingConfig) -> list[CodeChunk]:
    """Top-level key chunking for JSON."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return _chunk_window(content, _config)

    if isinstance(data, dict):
        chunks: list[CodeChunk] = []
        for key, value in data.items():
            serialized = json.dumps({key: value}, indent=2)
            chunks.append(
                CodeChunk(
                    content=serialized,
                    chunk_type=ChunkType.JSON_KEY,
                    start_line=1,
                    end_line=1,
                    metadata={"key": key},
                )
            )
        return chunks if chunks else _chunk_window(content, _config)

    return _chunk_window(content, _config)


def _chunk_window(content: str, config: ChunkingConfig) -> list[CodeChunk]:
    """Sliding window fallback for arbitrary text."""
    stride = max(1, config.chars_per_chunk - config.overlap_chars)
    chunks: list[CodeChunk] = []
    start = 0
    total = len(content)

    while start < total:
        end = min(start + config.chars_per_chunk, total)
        text = content[start:end]
        start_line = content[:start].count("\n") + 1
        end_line = content[:end].count("\n") + 1
        if text.strip():
            chunks.append(
                CodeChunk(
                    content=text.strip(),
                    chunk_type=ChunkType.WINDOW,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
        if end >= total:
            break
        start += stride

    return chunks


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def _redact_chunk(chunk: CodeChunk) -> CodeChunk:
    """Apply credential redaction to chunk content via ``animus.memory.redaction``."""
    redacted, hits = redact(chunk.content, include_personal=False)
    if not hits:
        return chunk

    meta = dict(chunk.metadata)
    meta["_redaction_count"] = str(len(hits))
    meta["_redaction_types"] = ",".join(sorted({h.type for h in hits}))

    return CodeChunk(
        content=redacted,
        chunk_type=chunk.chunk_type,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        source_path=chunk.source_path,
        metadata=meta,
    )
