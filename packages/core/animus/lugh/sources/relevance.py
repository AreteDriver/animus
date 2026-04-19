"""
Keyword-based relevance scorer for SourceItems.

v1 is intentionally unlearned: a dict of {regex -> weight} applied to a
weighted concatenation of title + summary + raw_text snippet. This is
enough to triage hundreds of daily items into a manageable shortlist;
the synthesis persona (task #7) will do the deeper reading.

Scoring model:
    raw_score  = sum of matched weights (title counted 2x)
    normalized = clamp(raw_score / ceiling, 0.0, 1.0)

Negative weights penalize noise (crypto spam, listicles). The default
keyword set reflects the current Animus portfolio and explicit user
interests (memory compression, agent infra, LLM ops).

Override by passing your own dict — keywords and weights are data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from animus.lugh.sources.base import SourceItem

DEFAULT_KEYWORDS: dict[str, float] = {
    # Core portfolio — direct overlap with Animus/Forge/Quorum
    "exocortex": 4.0,
    "agent memory": 3.5,
    "memory compression": 4.0,
    "context compression": 3.5,
    "kv cache": 2.5,
    "multi-agent": 2.5,
    "agent orchestration": 2.5,
    "model context protocol": 3.0,
    "mcp server": 2.5,
    # Agent infrastructure (Forge, Claude Code, Quorum)
    "tool use": 1.5,
    "tool calling": 1.5,
    "autonomous agent": 2.0,
    "ai agent": 1.5,
    "llm agent": 2.0,
    "agent framework": 2.0,
    "retrieval augmented": 1.5,
    "rag": 1.0,
    # LLM ops
    "prompt caching": 2.5,
    "long context": 2.5,
    "context window": 1.5,
    "reasoning model": 2.0,
    "test-time": 2.0,
    "test time compute": 2.5,
    "inference cost": 2.0,
    "eval harness": 2.0,
    "evals": 1.0,
    "benchmark": 1.0,
    # Memory architectures (explicit user interest)
    "titans": 3.5,
    "learned memory": 3.0,
    "episodic memory": 2.5,
    "semantic memory": 2.5,
    "long-term memory": 2.5,
    "memory bank": 2.5,
    # Specific vendors we route to / compete with
    "anthropic": 1.0,
    "claude": 1.0,
    "ollama": 1.0,
    "chromadb": 1.5,
    "deepmind": 1.5,
    "google research": 1.5,
    # Adjacent fields the user builds in
    "cognitive architecture": 2.5,
    "self-improvement": 2.0,
    "self-improving": 2.0,
    # Negative weights — noise we want to deprioritize
    "nft": -2.0,
    "airdrop": -1.5,
    "meme coin": -2.0,
    "pump and dump": -2.5,
    "rug pull": -1.5,  # except for our rug-pull-museum project, but ranked low
    "celebrity": -1.0,
}

DEFAULT_CEILING = 6.0


@dataclass
class KeywordScorer:
    """Sum weights of matched keywords; normalize to 0–1.

    Attributes:
        keywords: phrase -> weight. Phrases match as case-insensitive whole-word
                  substrings (spaces allowed). Negatives penalize noise.
        ceiling:  raw score above this maps to 1.0.
    """

    keywords: dict[str, float]
    ceiling: float = DEFAULT_CEILING

    def __post_init__(self) -> None:
        self._compiled: list[tuple[re.Pattern[str], float]] = [
            (_compile(k), w) for k, w in self.keywords.items()
        ]

    def score(self, item: SourceItem) -> float:
        title = item.title or ""
        summary = item.summary or ""
        body_snippet = (item.raw_text or "")[:1500]

        raw = 0.0
        for pattern, weight in self._compiled:
            if pattern.search(title):
                raw += weight * 2.0  # title hits count double
            if pattern.search(summary) or pattern.search(body_snippet):
                raw += weight

        # Negatives can push the raw below zero; clamp to [0, 1].
        if raw <= 0:
            return 0.0
        return min(1.0, raw / self.ceiling)

    def explain(self, item: SourceItem) -> dict:
        """Diagnostic breakdown — which keywords hit and contributed what."""
        title = item.title or ""
        body = f"{item.summary or ''} {(item.raw_text or '')[:1500]}"
        hits: list[dict] = []
        raw = 0.0
        for pattern, weight in self._compiled:
            t = bool(pattern.search(title))
            b = bool(pattern.search(body))
            if t or b:
                contribution = (weight * 2.0 if t else 0.0) + (weight if b else 0.0)
                raw += contribution
                hits.append(
                    {
                        "keyword": pattern.pattern,
                        "weight": weight,
                        "title_hit": t,
                        "body_hit": b,
                        "contribution": round(contribution, 3),
                    }
                )
        return {
            "raw_score": round(raw, 3),
            "normalized": round(max(0.0, min(1.0, raw / self.ceiling)), 3),
            "ceiling": self.ceiling,
            "hits": sorted(hits, key=lambda h: -abs(h["contribution"])),
        }


def default_scorer() -> KeywordScorer:
    return KeywordScorer(keywords=dict(DEFAULT_KEYWORDS))


def _compile(phrase: str) -> re.Pattern[str]:
    """Compile a whole-word, case-insensitive pattern for a phrase."""
    escaped = re.escape(phrase.strip())
    return re.compile(rf"\b{escaped}\b", re.IGNORECASE)
