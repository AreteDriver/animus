"""Tests for animus.ogma.read — end-to-end with a mocked CognitiveLayer."""

from __future__ import annotations

from pathlib import Path

import pytest

from animus.lugh.sources.base import SourceItem
from animus.ogma.grounding import GapResult
from animus.ogma.models import OgmaOutput
from animus.ogma.read import OgmaSynthesisError, synthesize


def _well_formed_response() -> str:
    """A SKILL.md-shaped response the parser must accept end-to-end."""
    return (
        "# Why Agents Still Need Humans\n"
        "\n"
        "**Source:** GWPpLdpTo90  •  **Date:** 2026-05-26\n"
        "**Cited from:** youtube:@AIDailyBrief\n"
        "\n"
        "## Concept\n"
        "Two operating modes for agent collaboration: employee-agents and the human-sandwich harness.\n"
        "\n"
        "## Novelty\n"
        "Codifies a pattern Lugh harvest had only described anecdotally.\n"
        "\n"
        "## Animus gap\n"
        "**Status:** NONE\n"
        "animus has no harness-shaped abstraction for human/agent co-work — greenfield.\n"
        "\n"
        "## Weaknesses in the source\n"
        "Anecdotal, no measurement.\n"
        "\n"
        "## Proposal — how we build it better\n"
        "Add packages/core/animus/harness/ with frame/draft/judge phases.\n"
        "\n"
        "## ROI\n"
        "**Value:** Names the pattern Ogma+Forge are already implementing implicitly\n"
        "**Effort:** moderate\n"
        "**Priority:** queue behind Ogma v0.1\n"
        "\n"
        "## Risks\n"
        "Premature abstraction if Ogma's flow doesn't generalize.\n"
        "\n"
        "## Confidence\n"
        "0.65 — strong source, weak measurement\n"
        "\n"
        "## Sources cited\n"
        "- https://youtube.com/watch?v=GWPpLdpTo90\n"
    )


def _item() -> SourceItem:
    return SourceItem(
        source_id="youtube:@AIDailyBrief",
        item_id="GWPpLdpTo90",
        title="Why Agents Still Need Humans",
        url="https://youtube.com/watch?v=GWPpLdpTo90",
        published=None,
        summary="The human sandwich — how agents and humans collaborate.",
        raw_text="Today on the AI Daily Brief, the next wave of human agent collaboration...",
        metadata={"video_id": "GWPpLdpTo90"},
    )


class _StubModel:
    """Stands in for ModelInterface — captures generate() calls."""

    def __init__(self, response: str = "") -> None:
        self.response = response
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        return self.response


@pytest.fixture
def stub_gap(monkeypatch):
    """Make verify_animus_gap deterministic + cheap for these tests."""

    def fake(concept, repo_root=None, **kw):
        return GapResult(status="NONE", paths_read=[], notes="(test stub) no animus repo scan")

    monkeypatch.setattr("animus.ogma.read.verify_animus_gap", fake)


class TestReadRelevanceGate:
    def test_score_below_threshold_returns_none(self, tmp_path: Path, stub_gap):
        stub = _StubModel("(should not be called)")
        result = synthesize(
            _item(),
            model=stub,
            min_relevance=0.5,
            relevance_score=0.2,
            output_dir=tmp_path,
        )
        assert result is None
        assert stub.calls == [], "model must not be invoked when item is gated"

    def test_score_above_threshold_invokes_model(self, tmp_path: Path, stub_gap):
        stub = _StubModel(_well_formed_response())
        result = synthesize(
            _item(),
            model=stub,
            min_relevance=0.5,
            relevance_score=0.9,
            output_dir=tmp_path,
        )
        assert result is not None
        assert stub.calls, "model must be invoked when item passes the gate"


class TestReadHappyPath:
    def test_returns_well_formed_ogma_output(self, tmp_path: Path, stub_gap):
        stub = _StubModel(_well_formed_response())
        result = synthesize(_item(), model=stub, output_dir=tmp_path)
        assert isinstance(result, OgmaOutput)
        assert result.title == "Why Agents Still Need Humans"
        assert result.animus_gap == "NONE"
        assert result.roi_effort == "moderate"
        assert 0.0 <= result.confidence <= 1.0

    def test_writes_file_at_dated_slug_path(self, tmp_path: Path, stub_gap):
        stub = _StubModel(_well_formed_response())
        result = synthesize(_item(), model=stub, output_dir=tmp_path)
        assert result is not None
        written = tmp_path / f"{result.date.isoformat()}-{result.slug}.md"
        assert written.exists()
        assert "## Concept" in written.read_text(encoding="utf-8")

    def test_persona_is_passed_as_system_prompt(self, tmp_path: Path, stub_gap):
        stub = _StubModel(_well_formed_response())
        synthesize(_item(), model=stub, output_dir=tmp_path)
        (prompt, system) = stub.calls[0]
        assert system is not None and "Ogma" in system
        assert "Why Agents Still Need Humans" in prompt
        assert "Status: NONE" in prompt  # grounding block landed in the user prompt


class TestReadProviderErrors:
    def test_provider_401_propagates_no_fallback(self, tmp_path: Path, stub_gap):
        class _Failing:
            def generate(self, prompt: str, system: str | None = None) -> str:
                raise PermissionError("401 Unauthorized")

        with pytest.raises(OgmaSynthesisError, match="provider.generate raised"):
            synthesize(_item(), model=_Failing(), output_dir=tmp_path)

    def test_unparseable_response_dumps_to_failures_dir(
        self, tmp_path: Path, stub_gap, monkeypatch
    ):
        monkeypatch.setattr("animus.ogma.read.DEFAULT_FAILURE_DIR", tmp_path / ".failures")
        stub = _StubModel("this is not a valid ogma markdown response")
        with pytest.raises(OgmaSynthesisError, match="could not parse"):
            synthesize(_item(), model=stub, output_dir=tmp_path)
        dumped = list((tmp_path / ".failures").glob("*.txt"))
        assert dumped, "expected a failure dump under .failures/"
        assert "this is not a valid" in dumped[0].read_text(encoding="utf-8")


class TestReadInputValidation:
    def test_empty_source_text_raises(self, tmp_path: Path, stub_gap):
        empty_item = SourceItem(
            source_id="youtube:@AIDailyBrief",
            item_id="empty",
            title="",
            url="",
            published=None,
            summary="",
            raw_text="",
        )
        stub = _StubModel("(should not be called)")
        with pytest.raises(ValueError, match="no text to synthesize"):
            synthesize(empty_item, model=stub, output_dir=tmp_path)
        assert stub.calls == []


class TestReadDefaultProvider:
    def test_default_resolves_to_ollama_without_anthropic_key(
        self, tmp_path: Path, stub_gap, monkeypatch
    ):
        """When model=None, the provider must come from ModelConfig.ollama() —
        no Anthropic API key is touched. We don't exercise the network; we
        just verify the construction path."""
        constructed: list[str] = []

        def fake_create_model(cfg):
            constructed.append(cfg.provider.value)
            return _StubModel(_well_formed_response())

        monkeypatch.setattr("animus.ogma.read.create_model", fake_create_model)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        synthesize(_item(), output_dir=tmp_path)
        assert constructed == ["ollama"]
