"""Tests for animus.lugh.digest — categorization + digest + renderers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from animus.lugh.digest import (
    DailyDigest,
    ItemCategory,
    _matches,
    build_digest,
    categorize,
    to_discord_payload,
    to_markdown,
)
from animus.lugh.sources.base import SourceCache, SourceItem


def _item(**kw) -> SourceItem:
    d = dict(
        source_id="hn:front_page",
        item_id=str(kw.get("item_id", "42")),
        title=kw.get("title", "A story"),
        url="https://example.com/x",
        published=kw.get("published", datetime.now(timezone.utc)),
        summary=kw.get("summary", ""),
        raw_text=kw.get("raw_text", ""),
        tags=kw.get("tags", []),
    )
    d.update({k: v for k, v in kw.items() if k in SourceItem.__dataclass_fields__})
    return SourceItem(**d)


class TestMatches:
    def test_single_word_requires_boundary(self):
        # "forge" should not match "forged" or "forgery"
        assert _matches("he forged ahead", {"forge"}) == []
        assert _matches("lit the forge", {"forge"}) == ["forge"]

    def test_multiword_is_substring(self):
        # Multi-word matches via substring (spaces are natural boundaries)
        assert _matches("we use chromadb heavily", {"chromadb"}) == ["chromadb"]
        assert _matches("ran on fly.io last night", {"fly.io"}) == ["fly.io"]

    def test_case_insensitive(self):
        assert _matches("Animus is cool", {"animus"}) == ["animus"]


class TestCategorize:
    def test_portfolio_project_fires_project_category(self):
        ci = categorize(_item(title="Animus memory layer revamp"), 0.5)
        assert ci.category == ItemCategory.PROJECT
        assert "animus" in ci.portfolio_matches

    def test_single_tech_hit_is_not_enough_for_project(self):
        # "claude code" mentioned once in a Show HN should NOT be PROJECT —
        # too ambient. This was the Faceoff bug from 2026-04-20.
        ci = categorize(
            _item(
                title="Show HN: Terminal UI for NHL games",
                summary="Built with Python. Inspired by another TUI. Good with Claude Code.",
            ),
            0.17,
        )
        assert ci.category != ItemCategory.PROJECT

    def test_two_tech_hits_fires_project(self):
        ci = categorize(_item(summary="Uses chromadb and ollama for embeddings"), 0.5)
        assert ci.category == ItemCategory.PROJECT
        assert len(ci.tech_matches) >= 2

    def test_arxiv_source_fires_research(self):
        ci = categorize(_item(source_id="arxiv:cs.LG", item_id="2501.99999"), 0.3)
        assert ci.category == ItemCategory.RESEARCH

    def test_research_keyword_fires_research(self):
        ci = categorize(
            _item(title="We propose a novel benchmark", summary="ablation results"),
            0.2,
        )
        assert ci.category == ItemCategory.RESEARCH

    def test_industry_signal_fires_industry(self):
        ci = categorize(
            _item(title="Anthropic raised $3B series D", summary="valuation at $60B"),
            0.3,
        )
        assert ci.category == ItemCategory.INDUSTRY
        assert "Funding signal" in ci.action_hint

    def test_industry_mna_hint(self):
        ci = categorize(
            _item(title="OpenAI acquires X for $Y", summary="acquisition closed"),
            0.3,
        )
        assert ci.category == ItemCategory.INDUSTRY
        assert "M&A signal" in ci.action_hint

    def test_industry_distress_hint(self):
        ci = categorize(
            _item(title="Startup layoffs hit 30%", summary="down round feared"),
            0.2,
        )
        assert ci.category == ItemCategory.INDUSTRY
        assert "Distress" in ci.action_hint

    def test_show_hn_fires_idea(self):
        ci = categorize(_item(title="Show HN: Cool side project", summary="we built this"), 0.1)
        assert ci.category == ItemCategory.IDEA

    def test_low_score_below_threshold_is_noise(self):
        ci = categorize(_item(title="random thing"), 0.01)
        assert ci.category == ItemCategory.NOISE

    def test_uncategorized_scored_item_lands_in_idea(self):
        # Score above threshold but no keyword match anywhere
        ci = categorize(_item(title="just some thing"), 0.15)
        assert ci.category == ItemCategory.IDEA

    def test_project_hint_mentions_ogma(self):
        ci = categorize(_item(title="Forge improvements for Animus"), 0.5)
        assert "/ogma read" in ci.action_hint

    def test_recommended_for_ogma_threshold(self):
        # research + 0.15 → recommended
        r_hit = categorize(
            _item(title="we introduce a benchmark", source_id="arxiv:cs.LG", item_id="1"),
            0.20,
        )
        assert r_hit.recommended_for_ogma

        # idea at 0.20 → NOT recommended (needs 0.25 for non-research/project)
        i = categorize(_item(title="Show HN: thing", summary="we built"), 0.20)
        assert not i.recommended_for_ogma

        # idea at 0.30 → recommended
        i2 = categorize(_item(title="Show HN: thing", summary="we built"), 0.30)
        assert i2.recommended_for_ogma


class TestBuildDigest:
    def _cache_with(self, tmp_path, items_with_scores):
        cache = SourceCache(path=tmp_path / "t.sqlite")
        for item, score in items_with_scores:
            cache.put(item, relevance_score=score)
        return cache

    def test_dedup_same_item_across_source_kinds(self, tmp_path):
        # Same HN story posted to both front_page and show_hn — two fingerprints,
        # one logical item. Digest should collapse to one.
        now = datetime.now(timezone.utc)
        cache = self._cache_with(
            tmp_path,
            [
                (
                    _item(
                        source_id="hn:front_page",
                        item_id="42",
                        title="Show HN: thing",
                        summary="we built",
                    ),
                    0.17,
                ),
                (
                    _item(
                        source_id="hn:show_hn",
                        item_id="42",
                        title="Show HN: thing",
                        summary="we built",
                    ),
                    0.17,
                ),
            ],
        )
        digest = build_digest(cache, window_hours=24, min_score=0.05, now=now)
        all_items = [ci for bucket in digest.by_category.values() for ci in bucket]
        assert len(all_items) == 1

    def test_window_filters_by_published(self, tmp_path):
        now = datetime.now(timezone.utc)
        cache = self._cache_with(
            tmp_path,
            [
                (
                    _item(
                        item_id="recent",
                        published=now - timedelta(hours=2),
                        title="Show HN: recent",
                    ),
                    0.2,
                ),
                (
                    _item(item_id="old", published=now - timedelta(hours=48), title="Show HN: old"),
                    0.2,
                ),
            ],
        )
        digest = build_digest(cache, window_hours=24, min_score=0.05, now=now)
        all_items = [ci for bucket in digest.by_category.values() for ci in bucket]
        assert len(all_items) == 1
        assert all_items[0].item.item_id == "recent"

    def test_min_score_excludes_low_items(self, tmp_path):
        now = datetime.now(timezone.utc)
        cache = self._cache_with(
            tmp_path,
            [
                (_item(item_id="high", title="Show HN: high"), 0.5),
                (_item(item_id="low", title="Show HN: low"), 0.01),
            ],
        )
        digest = build_digest(cache, window_hours=24, min_score=0.05, now=now)
        all_items = [ci for bucket in digest.by_category.values() for ci in bucket]
        assert [ci.item.item_id for ci in all_items] == ["high"]

    def test_top_for_ogma_respects_thresholds(self, tmp_path):
        now = datetime.now(timezone.utc)
        cache = self._cache_with(
            tmp_path,
            [
                (_item(item_id="arxiv1", source_id="arxiv:cs.LG", title="Paper on X"), 0.20),
                (_item(item_id="idea1", title="Show HN: thing", summary="we built"), 0.18),
                (_item(item_id="idea2", title="Show HN: other", summary="we built"), 0.30),
            ],
        )
        digest = build_digest(cache, window_hours=24, min_score=0.05, now=now)
        top_ids = {ci.item.item_id for ci in digest.top_for_ogma}
        assert "arxiv1" in top_ids  # research at 0.20 clears bar
        assert "idea2" in top_ids  # idea at 0.30 clears 0.25 bar
        assert "idea1" not in top_ids  # idea at 0.18 does NOT clear 0.25


class TestRenderers:
    def _digest(self) -> DailyDigest:
        return DailyDigest(
            date="2026-04-20",
            window_hours=24,
            total_items=3,
            by_category={
                ItemCategory.PROJECT: [
                    categorize(_item(title="Animus memory work", url="https://x/1"), 0.6)
                ],
                ItemCategory.RESEARCH: [
                    categorize(
                        _item(
                            title="We introduce a benchmark",
                            url="https://arxiv.org/abs/1",
                            source_id="arxiv:cs.LG",
                            item_id="2501.99999",
                        ),
                        0.25,
                    )
                ],
                ItemCategory.INDUSTRY: [],
                ItemCategory.IDEA: [
                    categorize(
                        _item(
                            title="Show HN: Thing",
                            url="https://x/2",
                            summary="we built it",
                        ),
                        0.30,
                    )
                ],
                ItemCategory.NOISE: [],
            },
        )

    def test_markdown_includes_top_for_ogma(self):
        md = to_markdown(self._digest())
        assert "Top candidates for `/ogma read`" in md
        assert "Animus memory work" in md  # project item with 0.6 should surface
        assert "Project upgrades" in md
        assert "AI research" in md
        assert "Industry awareness" in md
        assert "Ideas / concepts" in md

    def test_markdown_shows_empty_industry_bucket(self):
        md = to_markdown(self._digest())
        # INDUSTRY has no items → should say "_None._"
        assert "## Industry awareness" in md
        # Check the immediately-following text
        idx = md.index("## Industry awareness")
        next_block = md[idx : idx + 400]
        assert "_None._" in next_block

    def test_markdown_has_portfolio_marker(self):
        md = to_markdown(self._digest())
        # Portfolio-matched items get 🎯 marker
        assert "🎯" in md

    def test_discord_payload_shape(self):
        payload = to_discord_payload(self._digest())
        assert "embeds" in payload
        embed = payload["embeds"][0]
        assert embed["title"].startswith("🌾 Lugh Daily Haul")
        assert embed["color"]
        # Must have at least top-for-ogma + one category field
        assert len(embed["fields"]) >= 2

    def test_discord_skips_empty_categories(self):
        payload = to_discord_payload(self._digest())
        field_names = [f["name"] for f in payload["embeds"][0]["fields"]]
        # INDUSTRY was empty → shouldn't appear as a field
        assert not any("Industry" in n for n in field_names)

    def test_discord_field_values_under_1024_chars(self):
        # Discord's hard limit. Long titles / many items must be capped.
        payload = to_discord_payload(self._digest())
        for f in payload["embeds"][0]["fields"]:
            assert len(f["value"]) <= 1024
