"""Tests for animus.ogma.cli — verb dispatch + cache lookup behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from animus.lugh.sources.base import SourceCache, SourceItem
from animus.ogma.cli import handle_ogma
from animus.ogma.grounding import GapResult


@pytest.fixture
def cache(tmp_path: Path) -> SourceCache:
    return SourceCache(path=tmp_path / "lugh.sqlite3")


@pytest.fixture
def seeded_cache(cache: SourceCache) -> SourceCache:
    cache.put(
        SourceItem(
            source_id="youtube:@AIDailyBrief",
            item_id="GWPpLdpTo90",
            title="Why Agents Still Need Humans",
            url="https://youtube.com/watch?v=GWPpLdpTo90",
            published=None,
            summary="The human sandwich.",
            raw_text="full transcript...",
            metadata={"video_id": "GWPpLdpTo90"},
        )
    )
    return cache


def _well_formed_response() -> str:
    return (
        "# Why Agents Still Need Humans\n"
        "\n"
        "**Source:** GWPpLdpTo90  •  **Date:** 2026-05-26\n"
        "**Cited from:** youtube:@AIDailyBrief\n"
        "\n"
        "## Concept\n"
        "Human-sandwich harness for agent collaboration.\n"
        "\n"
        "## Novelty\n"
        "Codifies anecdotal practice.\n"
        "\n"
        "## Animus gap\n"
        "**Status:** NONE\n"
        "no harness abstraction yet — greenfield.\n"
        "\n"
        "## Weaknesses in the source\n"
        "No measurement.\n"
        "\n"
        "## Proposal — how we build it better\n"
        "packages/core/animus/harness/ — frame/draft/judge.\n"
        "\n"
        "## ROI\n"
        "**Value:** Names the pattern\n"
        "**Effort:** moderate\n"
        "**Priority:** later\n"
        "\n"
        "## Risks\n"
        "Premature abstraction.\n"
        "\n"
        "## Confidence\n"
        "0.65 — strong source\n"
        "\n"
        "## Sources cited\n"
        "- https://youtube.com/watch?v=GWPpLdpTo90\n"
    )


class _StubModel:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str, system: str | None = None) -> str:
        return self.response


@pytest.fixture
def stubbed_pipeline(monkeypatch, tmp_path: Path):
    """Stub the model + grounding + redirect output dir; CLI calls become hermetic."""
    monkeypatch.setattr(
        "animus.ogma.read.create_model",
        lambda cfg: _StubModel(_well_formed_response()),
    )
    monkeypatch.setattr(
        "animus.ogma.read.verify_animus_gap",
        lambda concept, repo_root=None, **kw: GapResult(status="NONE", notes="stub"),
    )
    # Force the write target so we don't pollute the user's notes dir.
    monkeypatch.setattr(
        "animus.ogma.models.DEFAULT_OUTPUT_DIR",
        tmp_path / "ogma_out",
    )
    return tmp_path / "ogma_out"


class TestHandleOgmaDispatch:
    def test_no_args_prints_usage(self, cache, capsys):
        rc = handle_ogma([], cache=cache)
        assert rc == 2

    def test_unknown_verb_returns_nonzero(self, cache, capsys):
        rc = handle_ogma(["foobar"], cache=cache)
        assert rc == 2

    def test_brief_stub_returns_zero(self, cache):
        assert handle_ogma(["brief"], cache=cache) == 0

    def test_gap_stub_returns_zero(self, cache):
        assert handle_ogma(["gap", "test concept"], cache=cache) == 0

    def test_audit_stub_returns_zero(self, cache):
        assert handle_ogma(["audit", "animus"], cache=cache) == 0

    def test_sweep_stub_returns_zero(self, cache):
        assert handle_ogma(["sweep"], cache=cache) == 0


class TestReadDispatch:
    def test_missing_target_returns_two(self, cache):
        rc = handle_ogma(["read"], cache=cache)
        assert rc == 2

    def test_target_without_colon_returns_two(self, cache):
        rc = handle_ogma(["read", "no-colon-here"], cache=cache)
        assert rc == 2

    def test_cache_miss_returns_one(self, cache):
        rc = handle_ogma(["read", "youtube:@AIDailyBrief:does-not-exist"], cache=cache)
        assert rc == 1

    def test_cache_hit_invokes_pipeline(self, seeded_cache, stubbed_pipeline):
        rc = handle_ogma(["read", "youtube:@AIDailyBrief:GWPpLdpTo90"], cache=seeded_cache)
        assert rc == 0
        out_dir = stubbed_pipeline
        written = list(out_dir.glob("*.md"))
        assert len(written) == 1, f"expected exactly one output, got {written}"
        text = written[0].read_text(encoding="utf-8")
        assert "## Concept" in text
        assert "Why Agents Still Need Humans" in text
