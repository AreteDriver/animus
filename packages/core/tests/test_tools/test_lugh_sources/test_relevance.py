"""Tests for animus.lugh.sources.relevance — keyword scorer."""

from __future__ import annotations

from datetime import datetime, timezone

from animus.lugh.sources.base import SourceItem
from animus.lugh.sources.relevance import (
    DEFAULT_KEYWORDS,
    KeywordScorer,
    default_scorer,
)


def _item(title: str = "", summary: str = "", raw_text: str = "") -> SourceItem:
    return SourceItem(
        source_id="arxiv:cs.LG",
        item_id="2604.00001",
        title=title,
        url="https://x",
        published=datetime(2026, 4, 18, tzinfo=timezone.utc),
        summary=summary,
        raw_text=raw_text,
    )


class TestKeywordScorer:
    def test_empty_item_scores_zero(self):
        scorer = KeywordScorer(keywords={"memory": 1.0})
        assert scorer.score(_item()) == 0.0

    def test_title_hit_counted_double(self):
        scorer = KeywordScorer(keywords={"memory": 1.0}, ceiling=10.0)
        title_score = scorer.score(_item(title="About memory"))
        summary_score = scorer.score(_item(summary="About memory"))
        # title match = weight*2 + weight (title also counted by summary branch? no — title/summary are separate)
        # title match: title_hit=True -> weight*2
        # summary match: body_hit=True -> weight*1
        # So title_score should be 2x summary_score.
        assert title_score == 2 * summary_score

    def test_negative_weights_penalize(self):
        scorer = KeywordScorer(keywords={"LLM agent": 2.0, "nft": -2.0}, ceiling=5.0)
        positive_only = scorer.score(_item(summary="a great LLM agent paper"))
        with_negative = scorer.score(_item(summary="a great LLM agent NFT scheme"))
        assert positive_only > 0
        assert with_negative < positive_only
        # If the raw goes net-negative or zero we floor at 0.
        noise_only = scorer.score(_item(summary="just an NFT airdrop"))
        assert noise_only == 0.0

    def test_normalization_caps_at_one(self):
        scorer = KeywordScorer(keywords={"memory": 10.0}, ceiling=1.0)  # small ceiling, huge weight
        assert scorer.score(_item(title="memory memory memory")) == 1.0

    def test_normalization_clamps_floor_at_zero(self):
        scorer = KeywordScorer(keywords={"spam": -5.0}, ceiling=5.0)
        assert scorer.score(_item(title="spam spam")) == 0.0

    def test_phrase_boundary(self):
        scorer = KeywordScorer(keywords={"rag": 1.0}, ceiling=5.0)
        # "rag" as whole word hits; "ragged" as substring should not.
        assert scorer.score(_item(summary="we used rag here")) > 0
        assert scorer.score(_item(summary="a ragged pattern")) == 0.0

    def test_case_insensitive(self):
        scorer = KeywordScorer(keywords={"claude": 1.0}, ceiling=5.0)
        assert scorer.score(_item(title="CLAUDE")) == scorer.score(_item(title="claude"))


class TestExplain:
    def test_explain_returns_hits(self):
        scorer = KeywordScorer(keywords={"memory": 2.0, "agent": 1.0, "nft": -2.0}, ceiling=10.0)
        item = _item(
            title="Agent memory architectures",
            summary="no NFT here though",
        )
        result = scorer.explain(item)
        assert result["raw_score"] > 0
        keywords_hit = {h["keyword"] for h in result["hits"]}
        # Patterns are escaped + wrapped with \b, so "memory" -> "\\bmemory\\b".
        assert any("memory" in k for k in keywords_hit)
        assert any("agent" in k for k in keywords_hit)
        assert any("nft" in k.lower() for k in keywords_hit)

    def test_explain_contribution_breakdown(self):
        scorer = KeywordScorer(keywords={"memory": 1.0}, ceiling=5.0)
        result = scorer.explain(_item(title="memory"))
        [hit] = [h for h in result["hits"] if "memory" in h["keyword"]]
        # title_hit=True -> contribution = 1.0 * 2 = 2.0
        # body_hit also not true because body is empty
        assert hit["contribution"] == 2.0
        assert hit["title_hit"] is True
        assert hit["body_hit"] is False


class TestDefaults:
    def test_default_scorer_has_portfolio_keywords(self):
        scorer = default_scorer()
        item = _item(title="Memory compression in learned systems")
        result = scorer.score(item)
        # Should hit "memory compression" and "learned memory"
        assert result > 0.3

    def test_default_scorer_penalizes_crypto_noise(self):
        scorer = default_scorer()
        clean = scorer.score(_item(title="Claude prompt caching analysis"))
        noise = scorer.score(_item(title="Meme coin airdrop NFT pump"))
        assert clean > noise
        assert noise == 0.0

    def test_default_keywords_include_memory_compression(self):
        # Explicit user call-out — don't let someone silently delete it.
        assert "memory compression" in DEFAULT_KEYWORDS
        assert DEFAULT_KEYWORDS["memory compression"] > 0
