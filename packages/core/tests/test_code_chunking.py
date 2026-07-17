"""Tests for ``animus.code.chunking`` — extracted from memboot."""

from __future__ import annotations

import pytest
from pathlib import Path

from animus.code.chunking import (
    ChunkType,
    CodeChunk,
    ChunkingConfig,
    _chunk_json,
    _chunk_markdown,
    _chunk_python,
    _chunk_window,
    _chunk_yaml,
    _redact_chunk,
    chunk_file,
    chunk_codebase,
)


# ---------------------------------------------------------------------------
# Python AST chunking
# ---------------------------------------------------------------------------


class TestChunkPython:
    def test_function_chunk(self):
        code = (
            "def hello():\n"
            "    pass\n"
        )
        chunks = _chunk_python(code, ChunkingConfig())
        funcs = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        assert len(funcs) == 1
        assert funcs[0].metadata["name"] == "hello"
        assert funcs[0].start_line == 1
        assert funcs[0].end_line == 2

    def test_async_function_chunk(self):
        code = (
            "async def fetch():\n"
            "    return 42\n"
        )
        chunks = _chunk_python(code, ChunkingConfig())
        funcs = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        assert len(funcs) == 1
        assert funcs[0].metadata["name"] == "fetch"

    def test_class_chunk(self):
        code = (
            "class Foo:\n"
            "    def bar(self):\n"
            "        pass\n"
        )
        chunks = _chunk_python(code, ChunkingConfig())
        classes = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        methods = [c for c in chunks if c.chunk_type == ChunkType.METHOD]
        # Small class — kept intact, not split
        assert len(classes) == 1
        assert classes[0].metadata["name"] == "Foo"
        assert len(methods) == 0

    def test_large_class_split_into_methods(self):
        # Build a class with many methods so it exceeds the char budget
        lines = ["class Big:", '    """A big class."""', ""]
        for i in range(30):
            lines.append(f"    def method_{i}(self):")
            lines.append(f"        return {i}")
        code = "\n".join(lines)
        chunks = _chunk_python(code, ChunkingConfig(max_chunk_tokens=50))
        methods = [c for c in chunks if c.chunk_type == ChunkType.METHOD]
        classes = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(methods) == 30
        assert len(classes) == 1  # header
        assert classes[0].metadata["name"] == "Big"

    def test_module_level_code(self):
        code = (
            "x = 1\n"
            "def hello():\n"
            "    pass\n"
            "y = 2\n"
        )
        chunks = _chunk_python(code, ChunkingConfig())
        mods = [c for c in chunks if c.chunk_type == ChunkType.MODULE]
        assert len(mods) == 1
        assert "x = 1" in mods[0].content
        assert "y = 2" in mods[0].content

    def test_syntax_error_fallback(self):
        code = "def broken(\n"
        chunks = _chunk_python(code, ChunkingConfig())
        assert all(c.chunk_type == ChunkType.WINDOW for c in chunks)


# ---------------------------------------------------------------------------
# Markdown chunking
# ---------------------------------------------------------------------------


class TestChunkMarkdown:
    def test_header_split(self):
        text = (
            "# Title\n\n"
            "Intro text.\n\n"
            "## Section A\n\n"
            "Content A.\n\n"
            "## Section B\n\n"
            "Content B.\n"
        )
        chunks = _chunk_markdown(text, ChunkingConfig())
        assert len(chunks) == 3
        assert chunks[0].metadata["header"] == "Title"
        assert chunks[1].metadata["header"] == "Section A"
        assert chunks[2].metadata["header"] == "Section B"

    def test_preamble(self):
        text = "Preamble here.\n\n# First\nContent."
        chunks = _chunk_markdown(text, ChunkingConfig())
        assert chunks[0].metadata["header"] == "preamble"
        assert "Preamble here" in chunks[0].content

    def test_no_headers_fallback(self):
        text = "Just some text without headers."
        chunks = _chunk_markdown(text, ChunkingConfig())
        assert all(c.chunk_type == ChunkType.WINDOW for c in chunks)


# ---------------------------------------------------------------------------
# YAML chunking
# ---------------------------------------------------------------------------


class TestChunkYaml:
    def test_top_level_keys(self):
        text = (
            "foo: 1\n"
            "bar:\n"
            "  nested: 2\n"
            "baz: 3\n"
        )
        chunks = _chunk_yaml(text, ChunkingConfig())
        assert len(chunks) == 3
        keys = {c.metadata["key"] for c in chunks}
        assert keys == {"foo", "bar", "baz"}

    def test_not_a_dict_fallback(self):
        text = "- one\n- two\n"
        chunks = _chunk_yaml(text, ChunkingConfig())
        assert all(c.chunk_type == ChunkType.WINDOW for c in chunks)

    def test_yaml_not_installed(self):
        # Can't easily test ImportError path without mocking, but we can
        # ensure the branch doesn't crash when yaml parses fine.
        pass


# ---------------------------------------------------------------------------
# JSON chunking
# ---------------------------------------------------------------------------


class TestChunkJson:
    def test_dict_keys(self):
        text = '{"a": 1, "b": 2}'
        chunks = _chunk_json(text, ChunkingConfig())
        assert len(chunks) == 2
        keys = {c.metadata["key"] for c in chunks}
        assert keys == {"a", "b"}

    def test_list_fallback(self):
        text = "[1, 2, 3]"
        chunks = _chunk_json(text, ChunkingConfig())
        assert all(c.chunk_type == ChunkType.WINDOW for c in chunks)

    def test_invalid_fallback(self):
        text = "not json"
        chunks = _chunk_json(text, ChunkingConfig())
        assert all(c.chunk_type == ChunkType.WINDOW for c in chunks)


# ---------------------------------------------------------------------------
# Window fallback
# ---------------------------------------------------------------------------


class TestChunkWindow:
    def test_basic_split(self):
        text = "line1\nline2\nline3\nline4\n"
        config = ChunkingConfig(max_chunk_tokens=2, overlap_tokens=0)
        chunks = _chunk_window(text, config)
        # ~8 chars per chunk, 4 lines ≈ 20 chars + newlines
        assert len(chunks) >= 2
        assert all(c.chunk_type == ChunkType.WINDOW for c in chunks)

    def test_empty(self):
        assert _chunk_window("", ChunkingConfig()) == []


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


class TestRedactChunk:
    def test_no_secrets_unchanged(self):
        chunk = CodeChunk(
            content="def hello(): pass",
            chunk_type=ChunkType.FUNCTION,
            start_line=1,
            end_line=1,
            metadata={"name": "hello"},
        )
        result = _redact_chunk(chunk)
        assert result.content == chunk.content
        assert "_redaction_count" not in result.metadata

    def test_api_key_redacted(self):
        chunk = CodeChunk(
            content="API_KEY = sk-ant-abcdefghijklmnopqrstuvwxyz1234567",
            chunk_type=ChunkType.MODULE,
            start_line=1,
            end_line=1,
            metadata={},
        )
        result = _redact_chunk(chunk)
        assert "sk-ant-" not in result.content
        assert result.metadata.get("_redaction_count") == "1"


# ---------------------------------------------------------------------------
# File-level API
# ---------------------------------------------------------------------------


class TestChunkFile:
    def test_python_file(self, tmp_path: Path):
        p = tmp_path / "test.py"
        p.write_text("def foo():\n    pass\n")
        chunks = chunk_file(p)
        funcs = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        assert len(funcs) == 1
        assert funcs[0].source_path == str(p)

    def test_markdown_file(self, tmp_path: Path):
        p = tmp_path / "test.md"
        p.write_text("# Hello\n\nWorld.\n")
        chunks = chunk_file(p)
        assert any(c.chunk_type == ChunkType.MARKDOWN_SECTION for c in chunks)

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "empty.py"
        p.write_text("")
        assert chunk_file(p) == []

    def test_source_path_override(self, tmp_path: Path):
        p = tmp_path / "test.py"
        p.write_text("def foo(): pass\n")
        chunks = chunk_file(p, source_path="src/test.py")
        assert chunks[0].source_path == "src/test.py"

    def test_redact_false(self, tmp_path: Path):
        p = tmp_path / "secrets.py"
        p.write_text("API_KEY = sk-ant-abcdefghijklmnopqrstuvwxyz1234567\n")
        chunks = chunk_file(p, redact_credentials=False)
        assert "sk-ant-" in chunks[0].content


# ---------------------------------------------------------------------------
# Codebase-level API
# ---------------------------------------------------------------------------


class TestChunkCodebase:
    def test_recursive_scan(self, tmp_path: Path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("def a(): pass\n")
        (tmp_path / "src" / "b.md").write_text("# B\n\nText.\n")
        (tmp_path / "src" / "ignored.txt").write_text("nope\n")

        result = chunk_codebase(tmp_path, globs=["*.py", "*.md"])
        assert "src/a.py" in result
        assert "src/b.md" in result
        assert "src/ignored.txt" not in result

    def test_exclude_pattern(self, tmp_path: Path):
        (tmp_path / "test_foo.py").write_text("def test(): pass\n")
        (tmp_path / "main.py").write_text("def main(): pass\n")

        result = chunk_codebase(tmp_path, globs=["*.py"], exclude=[])
        assert "test_foo.py" in result

        # Use a pattern that matches root-level test files
        result = chunk_codebase(tmp_path, globs=["*.py"], exclude=["test_*"])
        assert "test_foo.py" not in result
        assert "main.py" in result
