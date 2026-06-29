"""Tests for ``animus.workflows.ingest`` covering all 8 acceptance criteria."""

from __future__ import annotations

import io
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

from animus.lugh.sources.base import SourceItem
from animus.ogma.models import OgmaOutput
from animus.ogma.read import OgmaSynthesisError
from animus.workflows.ingest import (
    Cache,
    IngestResult,
    WorkflowError,
    ingest,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_source_item(**overrides) -> SourceItem:
    defaults = {
        "source_id": "test:item1",
        "item_id": "item1",
        "title": "Test Item",
        "url": "https://example.com/test",
        "published": None,
        "summary": "A test summary.",
        "author": None,
        "tags": [],
        "raw_text": "Full test text content.",
        "metadata": {},
    }
    defaults.update(overrides)
    return SourceItem(**defaults)


def _make_ogma_output(**overrides) -> OgmaOutput:
    defaults = {
        "title": "Test Synthesis",
        "source_id": "test:item1",
        "item_id": "item1",
        "date": date.today(),
        "concept": "Test concept.",
        "novelty": "None.",
        "animus_gap": "NONE",
        "animus_gap_notes": "Not implemented.",
        "weaknesses": "None.",
        "proposal": "Build it.",
        "roi_value": "High value.",
        "roi_effort": "trivial",
        "roi_priority": "Now",
        "risks": "None.",
        "confidence": 0.95,
        "confidence_justification": "Test.",
        "sources_cited": ["https://example.com/test"],
    }
    defaults.update(overrides)
    return OgmaOutput(**defaults)


# ---------------------------------------------------------------------------
# A1 — Full pipeline
# ---------------------------------------------------------------------------


def test_ingest_youtube_full_pipeline(tmp_path: Path):
    """A1: ingest with synthesize=True, tag=True produces a complete result."""
    item = _make_source_item(
        source_id="youtube:abc123",
        item_id="abc123",
        raw_text="transcript text here",
    )
    synthesis = _make_ogma_output()

    raw_base = tmp_path / "lugh_raw"
    real_cache = Cache(raw_base=raw_base, cache_path=tmp_path / "cache.sqlite")

    with patch("animus.workflows.ingest.Cache", return_value=real_cache):
        with patch("animus.workflows.ingest.resolve_source") as mock_resolve:
            fetcher = MagicMock()
            fetcher.fetch.return_value = item
            mock_resolve.return_value = fetcher

            with patch("animus.workflows.ingest.ogma_synthesize", return_value=synthesis):
                with patch(
                    "animus.workflows.ingest._tag_concepts", return_value=["mem-123"]
                ):
                    result = ingest(
                        "https://youtube.com/watch?v=abc123",
                        synthesize=True,
                        tag=True,
                    )

    assert result.item is not None
    assert result.item.raw_text == "transcript text here"
    assert result.success is True
    assert result.synthesis is not None
    assert result.memory_tags == ["mem-123"]
    # Cache file should exist
    cache_file = raw_base / "youtube" / "abc123.txt"
    assert cache_file.exists()


# ---------------------------------------------------------------------------
# A2 — Lugh only
# ---------------------------------------------------------------------------


def test_ingest_lugh_only():
    """A2: ingest with synthesize=False, tag=False skips downstream stages."""
    item = _make_source_item()

    with patch("animus.workflows.ingest.resolve_source") as mock_resolve:
        fetcher = MagicMock()
        fetcher.fetch.return_value = item
        mock_resolve.return_value = fetcher

        result = ingest("https://example.com/test", synthesize=False, tag=False)

    assert result.synthesis is None
    assert result.memory_tags is None
    assert result.success is True
    assert len(result.errors) == 0


# ---------------------------------------------------------------------------
# A3 — Ogma failure is partial
# ---------------------------------------------------------------------------


def test_ingest_ogma_failure_is_partial(tmp_path: Path):
    """A3: Ogma failure keeps Lugh data intact and records a partial error."""
    item = _make_source_item()
    raw_base = tmp_path / "lugh_raw"
    real_cache = Cache(raw_base=raw_base, cache_path=tmp_path / "cache.sqlite")

    with patch("animus.workflows.ingest.Cache", return_value=real_cache):
        with patch("animus.workflows.ingest.resolve_source") as mock_resolve:
            fetcher = MagicMock()
            fetcher.fetch.return_value = item
            mock_resolve.return_value = fetcher

            with patch(
                "animus.workflows.ingest.ogma_synthesize",
                side_effect=OgmaSynthesisError("synth failed"),
            ):
                result = ingest(
                    "https://example.com/test",
                    synthesize=True,
                    tag=False,
                )

    assert result.success is True
    assert len(result.errors) == 1
    assert result.errors[0].stage == "ogma"
    assert result.synthesis is None
    # Lugh cache file still exists
    cache_file = raw_base / "test" / "item1.txt"
    assert cache_file.exists()


# ---------------------------------------------------------------------------
# A4 — CLI invocation
# ---------------------------------------------------------------------------


def test_ingest_cli_invocation():
    """A4: ``animus ingest url --synthesize --tag`` returns exit code 0."""
    from animus.cli import main as cli_main

    with patch("animus.cli.ingest") as mock_ingest:
        mock_ingest.return_value = IngestResult(
            item=_make_source_item(),
            synthesis=_make_ogma_output(),
            memory_tags=["mem-1"],
            success=True,
        )
        code = cli_main(["ingest", "https://example.com/test", "--synthesize", "--tag"])

    assert code == 0


# ---------------------------------------------------------------------------
# A5 — Cache reuse
# ---------------------------------------------------------------------------


def test_ingest_reuses_cached_item():
    """A5: second ingest() on the same URL skips the network fetch."""
    item = _make_source_item()
    fetcher = MagicMock()
    fetcher.fetch.return_value = item

    with patch("animus.workflows.ingest.resolve_source") as mock_resolve:
        mock_resolve.return_value = fetcher

        with patch("animus.workflows.ingest.Cache") as mock_cache:
            cache_instance = MagicMock()
            # First call: cache miss
            cache_instance.find_by_url.return_value = None
            mock_cache.return_value = cache_instance

            result1 = ingest("https://example.com/test", synthesize=False, tag=False)
            assert result1.item is not None

            # Second call: cache hit
            cache_instance.find_by_url.return_value = item
            result2 = ingest("https://example.com/test", synthesize=False, tag=False)
            assert result2.item is not None

    # fetch should only be called once because second call hit cache
    assert fetcher.fetch.call_count == 1


# ---------------------------------------------------------------------------
# A6 — Invalid URL
# ---------------------------------------------------------------------------


def test_ingest_invalid_url():
    """A6: unresolvable URL yields fatal failure with stage == lugh."""
    with patch(
        "animus.workflows.ingest.resolve_source",
        side_effect=ValueError("unresolvable URL scheme: bad://url"),
    ):
        result = ingest("bad://url")

    assert result.success is False
    assert result.item is None
    assert len(result.errors) == 1
    assert result.errors[0].stage == "lugh"


# ---------------------------------------------------------------------------
# A7 — CLI partial exit code
# ---------------------------------------------------------------------------


def test_ingest_cli_partial_exit_code():
    """A7: CLI returns 0 with stderr warning when Ogma fails."""
    from animus.cli import main as cli_main

    with patch("animus.cli.ingest") as mock_ingest:
        mock_ingest.return_value = IngestResult(
            item=_make_source_item(),
            synthesis=None,
            memory_tags=None,
            errors=[WorkflowError("ogma", "OgmaSynthesisError", "synth failed")],
            success=True,
        )
        stderr_capture = io.StringIO()
        with patch.object(sys, "stderr", stderr_capture):
            code = cli_main(["ingest", "https://example.com/test", "--synthesize"])

    assert code == 0
    stderr_text = stderr_capture.getvalue()
    assert "ogma failed" in stderr_text.lower()


# ---------------------------------------------------------------------------
# A8 — Memory tag failure is partial
# ---------------------------------------------------------------------------


def test_ingest_memory_tag_failure_is_partial():
    """A8: memory tagging failure is non-fatal and recorded."""
    item = _make_source_item()
    synthesis = _make_ogma_output()

    with patch("animus.workflows.ingest.resolve_source") as mock_resolve:
        fetcher = MagicMock()
        fetcher.fetch.return_value = item
        mock_resolve.return_value = fetcher

        with patch("animus.workflows.ingest.ogma_synthesize", return_value=synthesis):
            with patch(
                "animus.workflows.ingest._tag_concepts",
                side_effect=RuntimeError("memory down"),
            ):
                result = ingest(
                    "https://example.com/test",
                    synthesize=True,
                    tag=True,
                )

    assert result.success is True
    assert result.memory_tags is None
    assert len(result.errors) == 1
    assert result.errors[0].stage == "memory"
