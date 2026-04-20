"""
Daily digest builder for harvested Lugh source items.

Takes the SourceCache, pulls items from the last N hours, applies
rule-based categorization against portfolio keywords + industry signals,
and emits two renderings:

    to_markdown(digest)        — full action-plan document
    to_discord_payload(digest) — scannable embed for the daily webhook

No LLM calls. Deterministic output. Portfolio keywords live in
PORTFOLIO_PROJECTS / PORTFOLIO_TECH — update them when the portfolio
shifts; the digest adapts without code changes.

Companion to ``animus.lugh.sources`` (harvest) and the
``/ogma`` skill (deep synthesis). The digest surfaces candidates for
the deep read; it does not replace it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

from animus.lugh.sources.base import SourceCache, SourceItem


class ItemCategory(Enum):
    """Where a digest item lands in the daily plan."""

    RESEARCH = "research"  # papers, benchmarks, method proposals
    INDUSTRY = "industry"  # funding, M&A, regulator, big-vendor moves
    PROJECT = "project"  # mentions portfolio projects or our tech stack
    IDEA = "idea"  # Show HN / Launch HN / novel OSS / concept pieces
    NOISE = "noise"  # scored but doesn't fit — diagnostic bucket


# --------------------------------------------------------------------------- #
# Keyword taxonomies — edit these when the portfolio shifts.
# --------------------------------------------------------------------------- #

PORTFOLIO_PROJECTS: set[str] = {
    # Active Arete projects (from MEMORY.md index, 2026-04-19)
    "animus",
    "forge",
    "lugh",
    "ogma",
    "quorum",
    "convergent",
    "monolith",
    "witness",
    "gatekeeper",
    "dossier",
    "benchgoblins",
    "argus",
    "anchormd",
    "overwatch",
    "memboot",
    "tiaid",
    "chainlog",
    "contribchain",
    "herald",
    "tidewise",
    "conduit",
    "azure-ops-mcp",
    "stellar audit agent",
    "frontier tribe os",
    "harvest",
    "aicards",
    "straykids",
    "crypto cards",
    "eve rebellion",
    "g13 linux",
    "likx",
    "marketing-engine",
    "promptctl",
    "context-hygiene",
    "agent-lint",
    "ai-spend",
    "mcp-manager",
    "bie",
    "bie-ontology",
    "drift-monitor",
}

PORTFOLIO_TECH: set[str] = {
    # Tech terms specific enough that mere mention IS signal.
    # Excluded (too ambient — people namedrop them constantly):
    #   "claude code", "anthropic api", "fastapi", "vercel", "stripe",
    #   "pytest", "ruff", "pydantic", "streamlit", "embedding", "rerank"
    "chromadb",
    "ollama",
    "llama.cpp",
    "mcp server",
    "model context protocol",
    "sui testnet",
    "sui mainnet",
    "base sepolia",
    "eve frontier",
    "eve online",
    "rank_bm25",
    "x402",
}

INDUSTRY_KEYWORDS: set[str] = {
    # Capital + corp events
    "raised",
    "series a",
    "series b",
    "series c",
    "series d",
    "seed round",
    "ipo",
    "valuation",
    "acquisition",
    "acquires",
    "acquired",
    "merger",
    "layoff",
    "layoffs",
    "shutting down",
    "shut down",
    "bankruptcy",
    "down round",
    "unicorn",
    "yc w25",
    "yc s25",
    "yc w26",
    "yc s26",
    "ycombinator",
    "a16z",
    "sequoia",
    "benchmark capital",
    "founders fund",
    # Big-vendor / regulator
    "anthropic",
    "openai",
    "deepmind",
    "google deepmind",
    "meta ai",
    "nvidia",
    "mistral",
    "hugging face",
    "eu ai act",
    "executive order",
    "export control",
    "doj ",
    "ftc ",
    "sec filing",
    "antitrust",
    # Investing / market-structure signals
    "earnings",
    "guidance",
    "capex",
    "datacenter",
    "compute cluster",
    "chip export",
    "gpu shortage",
    "lawsuit",
    "complaint",
    "subpoena",
}

RESEARCH_KEYWORDS: set[str] = {
    "arxiv.org",
    "we propose",
    "we introduce",
    "we present",
    "sota",
    "state-of-the-art",
    "benchmark",
    "ablation",
    "empirical",
    "pretrain",
    "pre-train",
    "fine-tune",
    "rlhf",
    "dpo",
    "distillation",
    "quantization",
    "inference-time",
    "test-time",
    "chain-of-thought",
    "speculative decoding",
}

IDEA_KEYWORDS: set[str] = {
    "show hn:",
    "launch hn:",
    "show hn -",
    "launch hn -",
    "we built",
    "i built",
    "launched",
    "open-sourced",
    "open sourced",
    "open source",
    "proof-of-concept",
    "prototype",
    "playground",
}


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class CategorizedItem:
    """A SourceItem plus its category assignment + actionable hints."""

    item: SourceItem
    relevance_score: float
    category: ItemCategory
    portfolio_matches: list[str] = field(default_factory=list)
    tech_matches: list[str] = field(default_factory=list)
    action_hint: str = ""

    @property
    def recommended_for_ogma(self) -> bool:
        """Flag items that warrant a deep /ogma read."""
        return (
            self.relevance_score >= 0.15
            and self.category in {ItemCategory.RESEARCH, ItemCategory.PROJECT}
        ) or (self.relevance_score >= 0.25)


@dataclass
class DailyDigest:
    """The day's haul, bucketed + ready to render."""

    date: str  # ISO YYYY-MM-DD
    window_hours: int
    total_items: int
    by_category: dict[ItemCategory, list[CategorizedItem]] = field(default_factory=dict)

    @property
    def top_for_ogma(self) -> list[CategorizedItem]:
        """Top candidates for deep /ogma read, sorted by score desc."""
        flat = [ci for bucket in self.by_category.values() for ci in bucket]
        flagged = [ci for ci in flat if ci.recommended_for_ogma]
        flagged.sort(key=lambda ci: -ci.relevance_score)
        return flagged[:5]


# --------------------------------------------------------------------------- #
# Categorization
# --------------------------------------------------------------------------- #


def _haystack(item: SourceItem) -> str:
    """Text surface used for keyword matching — lowercased concat."""
    return " ".join([item.title, item.summary, item.url, " ".join(item.tags)]).lower()


def _matches(haystack: str, vocab: set[str]) -> list[str]:
    """Return every vocab term that appears as a whole-word substring.

    Case-insensitive. For multi-word terms (or those containing punctuation
    like ``fly.io``), uses substring match; spaces/punctuation are natural
    boundaries. For single-word terms, requires word boundaries so ``forge``
    doesn't match ``forged``.
    """
    lower = haystack.lower()
    hits: list[str] = []
    for term in vocab:
        term_lower = term.lower()
        if " " in term_lower or "." in term_lower or "-" in term_lower:
            if term_lower in lower:
                hits.append(term)
        else:
            if re.search(rf"\b{re.escape(term_lower)}\b", lower):
                hits.append(term)
    return hits


def categorize(
    item: SourceItem,
    relevance_score: float,
    *,
    portfolio_projects: set[str] = PORTFOLIO_PROJECTS,
    portfolio_tech: set[str] = PORTFOLIO_TECH,
    industry_vocab: set[str] = INDUSTRY_KEYWORDS,
    research_vocab: set[str] = RESEARCH_KEYWORDS,
    idea_vocab: set[str] = IDEA_KEYWORDS,
    noise_threshold: float = 0.03,
) -> CategorizedItem:
    """Assign a single item to one category + attach action hints.

    Priority order (first match wins — higher buckets are more actionable):
        1. PROJECT  — touches our portfolio projects or core tech stack
        2. RESEARCH — arxiv source or research keywords
        3. INDUSTRY — funding / vendor / regulator signals
        4. IDEA     — Show HN / Launch HN / new OSS
        5. NOISE    — everything else (below threshold)
    """
    haystack = _haystack(item)
    portfolio_hits = _matches(haystack, portfolio_projects)
    tech_hits = _matches(haystack, portfolio_tech)
    is_arxiv = item.source_id.startswith("arxiv:")

    # Priority 1 — portfolio-relevant.
    # Require a portfolio-project name OR 2+ tech matches — single tech hit
    # is too noisy (people namedrop stacks in passing).
    strong_match = bool(portfolio_hits) or len(tech_hits) >= 2
    if strong_match:
        hint = _project_hint(portfolio_hits, tech_hits, item)
        return CategorizedItem(
            item=item,
            relevance_score=relevance_score,
            category=ItemCategory.PROJECT,
            portfolio_matches=sorted(set(portfolio_hits)),
            tech_matches=sorted(set(tech_hits)),
            action_hint=hint,
        )

    # Priority 2 — research
    if is_arxiv or _matches(haystack, research_vocab):
        return CategorizedItem(
            item=item,
            relevance_score=relevance_score,
            category=ItemCategory.RESEARCH,
            action_hint=_research_hint(item),
        )

    # Priority 3 — industry / investing / regulator signals
    industry_hits = _matches(haystack, industry_vocab)
    if industry_hits:
        return CategorizedItem(
            item=item,
            relevance_score=relevance_score,
            category=ItemCategory.INDUSTRY,
            action_hint=_industry_hint(industry_hits, item),
        )

    # Priority 4 — ideas / launches
    if _matches(haystack, idea_vocab):
        return CategorizedItem(
            item=item,
            relevance_score=relevance_score,
            category=ItemCategory.IDEA,
            action_hint="Concept worth noting; consider as reference when shaping similar tools",
        )

    # Priority 5 — noise (below threshold or uncategorized)
    if relevance_score < noise_threshold:
        return CategorizedItem(
            item=item, relevance_score=relevance_score, category=ItemCategory.NOISE
        )

    # Scored but uncategorized — treat as idea bucket by default
    return CategorizedItem(
        item=item,
        relevance_score=relevance_score,
        category=ItemCategory.IDEA,
        action_hint="Uncategorized but scored — review manually",
    )


def _project_hint(projects: list[str], tech: list[str], item: SourceItem) -> str:
    parts: list[str] = []
    if projects:
        parts.append(f"Touches portfolio: {', '.join(sorted(set(projects))[:3])}")
    if tech:
        parts.append(f"Tech stack overlap: {', '.join(sorted(set(tech))[:3])}")
    parts.append("Candidate for `/ogma read` → gap analysis + proposal")
    return " · ".join(parts)


def _research_hint(item: SourceItem) -> str:
    if item.source_id.startswith("arxiv:"):
        arxiv_id = item.metadata.get("arxiv_id", "")
        if arxiv_id:
            return f"arXiv {arxiv_id} · run `/ogma read {arxiv_id}` for full synthesis"
    return "Research signal · consider `/ogma read` if relevant to current work"


def _industry_hint(hits: list[str], item: SourceItem) -> str:
    hit_str = ", ".join(sorted(set(hits))[:3])
    text = (item.title + " " + item.summary).lower()
    if any(k in text for k in ("raised", "series ", "seed round", "valuation")):
        return f"Funding signal ({hit_str}) — market read"
    if any(k in text for k in ("acquisition", "acquire", "merger")):
        return f"M&A signal ({hit_str}) — competitive read"
    if any(k in text for k in ("layoff", "bankruptcy", "shutting down", "down round")):
        return f"Distress signal ({hit_str}) — watchlist"
    if any(k in text for k in ("eu ai act", "executive order", "antitrust", "sec filing")):
        return f"Regulatory signal ({hit_str}) — compliance watch"
    return f"Industry signal ({hit_str})"


# --------------------------------------------------------------------------- #
# Digest construction
# --------------------------------------------------------------------------- #


def build_digest(
    cache: SourceCache,
    window_hours: int = 24,
    min_score: float = 0.05,
    limit_per_category: int = 15,
    now: datetime | None = None,
) -> DailyDigest:
    """Pull recent items from the cache, categorize, bucket, and rank.

    Args:
        cache: initialised SourceCache.
        window_hours: how far back to look (published_at OR seen_at).
        min_score: floor — items below this are excluded entirely.
        limit_per_category: cap per category bucket.
        now: override for testing; defaults to UTC now.

    Returns:
        DailyDigest with categorized items, sorted by score desc per bucket.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=window_hours)

    # Pull generously — filter to window + score in Python since the cache
    # schema mixes published_at (feed) and seen_at (harvest) timestamps.
    candidates = cache.recent(limit=1000, min_score=min_score)

    # Dedup: same story can appear under multiple HN queries
    # (e.g., front_page AND show_hn) with distinct fingerprints.
    # Keep the highest-scoring instance per (source_kind, item_id).
    best_per_item: dict[str, tuple[SourceItem, float]] = {}
    for item in candidates:
        if item.published is not None and item.published < cutoff:
            continue
        score = _lookup_score(cache, item) or 0.0
        if score < min_score:
            continue
        # Group by source kind (hn / arxiv / podcast) + item_id.
        # Two different podcasts with the same episode guid would collide —
        # but podcast item_ids include the show_id in the GUID effectively.
        source_kind = item.source_id.split(":", 1)[0]
        key = f"{source_kind}:{item.item_id}"
        prior = best_per_item.get(key)
        if prior is None or score > prior[1]:
            best_per_item[key] = (item, score)

    categorized = [categorize(item, score) for item, score in best_per_item.values()]

    by_cat: dict[ItemCategory, list[CategorizedItem]] = {c: [] for c in ItemCategory}
    for ci in categorized:
        by_cat[ci.category].append(ci)

    # Sort each bucket by score desc, truncate
    for cat in by_cat:
        by_cat[cat].sort(key=lambda ci: -ci.relevance_score)
        by_cat[cat] = by_cat[cat][:limit_per_category]

    return DailyDigest(
        date=now.strftime("%Y-%m-%d"),
        window_hours=window_hours,
        total_items=len(categorized),
        by_category=by_cat,
    )


def _lookup_score(cache: SourceCache, item: SourceItem) -> float | None:
    """Fetch relevance_score from cache SQL directly (not exposed on recent())."""
    row = cache._conn.execute(
        "SELECT relevance_score FROM items WHERE fingerprint = ?",
        (item.fingerprint,),
    ).fetchone()
    if row is None:
        return None
    val = row["relevance_score"]
    return float(val) if val is not None else None


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #

_CATEGORY_ORDER = (
    ItemCategory.PROJECT,
    ItemCategory.RESEARCH,
    ItemCategory.INDUSTRY,
    ItemCategory.IDEA,
    ItemCategory.NOISE,
)

_CATEGORY_HEADERS = {
    ItemCategory.PROJECT: "Project upgrades — portfolio-relevant",
    ItemCategory.RESEARCH: "AI research",
    ItemCategory.INDUSTRY: "Industry awareness — funding / M&A / regulator / investing",
    ItemCategory.IDEA: "Ideas / concepts — stash for later",
    ItemCategory.NOISE: "Noise (diagnostic — review if categorizer looks off)",
}

DISCORD_COLOR_YELLOW = 0xF1C40F
DISCORD_COLOR_BLUE = 0x3498DB
DISCORD_COLOR_GREEN = 0x2ECC71


def to_markdown(digest: DailyDigest) -> str:
    """Render a DailyDigest as an action-plan markdown document."""
    lines: list[str] = []
    lines.append(f"# Lugh Daily Digest — {digest.date}")
    lines.append("")
    lines.append(
        f"Window: last {digest.window_hours}h · {digest.total_items} scored items categorized."
    )
    lines.append("")

    # Top for Ogma
    top = digest.top_for_ogma
    lines.append("## Top candidates for `/ogma read` (deep synthesis)")
    lines.append("")
    if not top:
        lines.append(
            "_No items cleared the 0.15-for-research-or-project / 0.25-anywhere threshold today._"
        )
    else:
        for ci in top:
            lines.append(
                f"- **[{ci.item.title}]({ci.item.url})** — "
                f"score {ci.relevance_score:.2f} · `{ci.category.value}`"
            )
            if ci.action_hint:
                lines.append(f"    - {ci.action_hint}")
    lines.append("")

    # Each category
    for cat in _CATEGORY_ORDER:
        items = digest.by_category.get(cat, [])
        if cat == ItemCategory.NOISE and not items:
            continue  # don't print empty noise bucket
        lines.append(f"## {_CATEGORY_HEADERS[cat]}")
        lines.append("")
        if not items:
            lines.append("_None._")
            lines.append("")
            continue
        for ci in items:
            marker = "🎯 " if ci.portfolio_matches else ""
            lines.append(
                f"- {marker}**[{ci.item.title}]({ci.item.url})** · score {ci.relevance_score:.2f}"
            )
            sub_parts: list[str] = []
            if ci.portfolio_matches:
                sub_parts.append(f"matches: `{', '.join(ci.portfolio_matches[:4])}`")
            if ci.tech_matches and not ci.portfolio_matches:
                sub_parts.append(f"tech: `{', '.join(ci.tech_matches[:4])}`")
            if ci.action_hint:
                sub_parts.append(ci.action_hint)
            if sub_parts:
                lines.append("    - " + " · ".join(sub_parts))
            # Inline one-line blurb from summary (truncated)
            blurb = ci.item.summary.replace("\n", " ").strip()
            if blurb:
                lines.append(f"    - _{blurb[:160]}_")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def to_discord_payload(digest: DailyDigest, max_per_field: int = 3) -> dict:
    """Render a DailyDigest as a Discord webhook embed payload."""
    fields: list[dict] = []

    # Top for Ogma
    top = digest.top_for_ogma
    if top:
        top_lines = [
            f"• [{ci.item.title[:70]}]({ci.item.url}) — {ci.relevance_score:.2f}"
            for ci in top[:max_per_field]
        ]
        fields.append(
            {
                "name": "🔭 Top for /ogma read",
                "value": "\n".join(top_lines)[:1024] or "_(none)_",
                "inline": False,
            }
        )

    # Per-category fields, skip NOISE
    category_field_names = {
        ItemCategory.PROJECT: "🎯 Project upgrades",
        ItemCategory.RESEARCH: "🧪 AI research",
        ItemCategory.INDUSTRY: "📈 Industry / investing",
        ItemCategory.IDEA: "💡 Ideas / concepts",
    }

    for cat, header in category_field_names.items():
        items = digest.by_category.get(cat, [])[:max_per_field]
        if not items:
            continue
        bullets = [f"• [{ci.item.title[:70]}]({ci.item.url})" for ci in items]
        fields.append({"name": header, "value": "\n".join(bullets)[:1024], "inline": False})

    return {
        "embeds": [
            {
                "title": f"🌾 Lugh Daily Haul — {digest.date}",
                "description": (
                    f"Harvested {digest.total_items} scored items in the last "
                    f"{digest.window_hours}h."
                ),
                "color": DISCORD_COLOR_BLUE,
                "fields": fields,
                "footer": {"text": "Categorizer: rule-based · deeper synthesis via /ogma read"},
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        ]
    }
