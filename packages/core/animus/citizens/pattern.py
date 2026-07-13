"""Citizen 009 — The Pattern Citizen.

The third stage of the Research Guild pipeline.

Responsibilities:
- Read MechanismCard objects from memory (produced by Abstraction Citizen)
- Cluster related mechanisms by category and shared tags
- Name emergent patterns that appear across ≥3 independent sources
- Produce PatternCard objects for the First-Principles Citizen (next stage)

Never:
- Reduce patterns to first principles (that's the First-Principles Citizen)
- Modify mechanism cards directly
- Act on findings without human approval

Instead:
    Read Mechanisms → Cluster → Name Pattern → Card → Human Approval → First-Principles
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from animus.citizens.proposal import (
    EvidenceItem,
    ImprovementProposal,
    ProposalConfidence,
    ProposalStatus,
    RiskAssessment,
)
from animus.logging import get_logger

if TYPE_CHECKING:
    from animus.memory import MemoryLayer

logger = get_logger("citizens.pattern")


# ═══════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════


@dataclass
class PatternCard:
    """A recurring pattern discovered across multiple mechanism cards.

    The Pattern Citizen identifies when related mechanisms appear
    across independent sources — an emergent architectural theme.
    """

    name: str  # e.g., "State externalization enables resilience"
    description: str  # e.g., "Separating state from computation..."
    constituent_mechanisms: list[str] = field(default_factory=list)  # Mechanism names
    occurrence_count: int = 0  # Number of independent sources
    confidence: float = 0.5  # 0.0–1.0
    category: str = ""  # Dominant category
    tags: list[str] = field(default_factory=list)
    source_provenance: list[str] = field(default_factory=list)  # Source IDs
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "constituent_mechanisms": self.constituent_mechanisms,
            "occurrence_count": self.occurrence_count,
            "confidence": self.confidence,
            "category": self.category,
            "tags": self.tags,
            "source_provenance": self.source_provenance,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PatternReport:
    """Report produced by the Pattern Citizen after a discovery run."""

    patterns: list[PatternCard] = field(default_factory=list)
    mechanisms_processed: int = 0
    mechanisms_with_no_pattern: int = 0
    errors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_discovered(self) -> int:
        return len(self.patterns)

    def summary(self) -> str:
        parts = [
            f"{self.total_discovered} pattern(s) discovered from {self.mechanisms_processed} mechanism(s)",
        ]
        if self.mechanisms_with_no_pattern:
            parts.append(
                f"{self.mechanisms_with_no_pattern} mechanism(s) with no recognizable pattern"
            )
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        return "; ".join(parts)


# ═══════════════════════════════════════════════════════════════════
# Pattern clustering heuristics
# ═══════════════════════════════════════════════════════════════════

# Keywords that bridge categories and suggest cross-cutting patterns
_CROSS_CUTTING_KEYWORDS: dict[str, str] = {
    "state": "State externalization",
    "resilien": "Resilience engineering",
    "fault tolerance": "Fault-tolerant design",
    "async": "Asynchronous architecture",
    "decoupl": "Loose coupling",
    "observ": "Observable systems",
    "monitor": "Observable systems",
    "test": "Testability as architecture",
    "secur": "Security by design",
    "scal": "Scalability patterns",
    "perform": "Performance optimization",
}


def _extract_cross_cutting_theme(mechanisms: list[dict[str, Any]]) -> str | None:
    """Look for shared keywords across mechanism descriptions that suggest a cross-cutting pattern."""
    text = " ".join(m.get("description", "") for m in mechanisms).lower()
    for keyword, theme in _CROSS_CUTTING_KEYWORDS.items():
        if keyword in text:
            return theme
    return None


def _generate_pattern_name(category: str, mechanisms: list[dict[str, Any]]) -> str:
    """Generate a human-readable pattern name from a cluster of mechanisms."""
    cross_theme = _extract_cross_cutting_theme(mechanisms)
    if cross_theme:
        return f"{cross_theme} via {category}"

    # Fallback: use category + dominant mechanism names
    names = [m.get("name", "") for m in mechanisms if m.get("name")]
    if len(names) >= 2:
        return f"Pattern in {category}: {', '.join(names[:2])}"
    elif names:
        return f"Pattern in {category}: {names[0]}"
    return f"Recurring {category} pattern"


def _generate_pattern_description(mechanisms: list[dict[str, Any]]) -> str:
    """Generate a description summarizing why these mechanisms form a pattern."""
    names = [m.get("name", "") for m in mechanisms if m.get("name")]
    if not names:
        return "A recurring architectural theme across multiple sources."

    desc = f"{len(names)} related mechanisms — {', '.join(names[:3])}"
    if len(names) > 3:
        desc += f" and {len(names) - 3} more"
    desc += " — suggest a transferable pattern."
    return desc


# ═══════════════════════════════════════════════════════════════════
# Pattern Citizen
# ═══════════════════════════════════════════════════════════════════


class PatternCitizen:
    """Citizen 009 — The Pattern Citizen.

    Reads mechanism cards, clusters related mechanisms, names
    emergent patterns, and produces PatternCards.
    """

    def __init__(
        self,
        memory_layer: MemoryLayer | None = None,
        codebase_path: Path | str = ".",
        evidence_dir: Path | str | None = None,
    ):
        self.memory = memory_layer
        self.codebase_path = Path(codebase_path).expanduser()
        self.evidence_dir = Path(evidence_dir).expanduser() if evidence_dir else None
        if self.evidence_dir:
            self.evidence_dir.mkdir(parents=True, exist_ok=True)

        self._patterns: list[PatternCard] = []

    # ------------------------------------------------------------------
    # Observation methods (for autonomous loop compatibility)
    # ------------------------------------------------------------------

    def observe_mechanisms(self) -> list[dict[str, Any]]:
        """Read mechanism cards from memory produced by the Abstraction Citizen.

        Returns:
            List of mechanism dicts compatible with autonomous loop.
        """
        findings: list[dict[str, Any]] = []
        if self.memory is None:
            logger.warning("Memory layer not available — observe_mechanisms skipped")
            return findings

        try:
            from animus.memory import MemoryType

            results = self.memory.search(
                query="abstraction mechanism research_guild",
                memory_type=MemoryType.SEMANTIC,
                limit=50,
            )
            for mem in results:
                content = mem.get("content", "") if hasattr(mem, "get") else getattr(mem, "content", "")
                meta = mem.get("metadata", {}) if hasattr(mem, "get") else getattr(mem, "metadata", {})
                if not isinstance(meta, dict):
                    meta = {}

                name = meta.get("name", "")
                category = meta.get("category", "")
                description = meta.get("description", "")
                source_provenance = meta.get("source_provenance", [])
                tags = meta.get("tags", [])

                if name or description:
                    findings.append({
                        "source": "memory",
                        "description": f"Mechanism: {name or description[:60]}",
                        "severity": "info",
                        "context": {
                            "name": name,
                            "category": category,
                            "description": description,
                            "source_provenance": source_provenance,
                            "tags": tags,
                            "pattern_type": "mechanism_card",
                        },
                    })

        except Exception as e:
            logger.warning("observe_mechanisms failed: %s", e)

        logger.info("Pattern observe_mechanisms: %d findings", len(findings))
        return findings

    # ------------------------------------------------------------------
    # Pattern discovery
    # ------------------------------------------------------------------

    def discover_patterns(self, mechanisms: list[dict[str, Any]] | None = None) -> list[PatternCard]:
        """Discover patterns by clustering related mechanisms.

        Uses two clustering strategies:
        1. Category clustering: ≥3 mechanisms in the same category → pattern
        2. Tag cross-clustering: ≥2 mechanisms sharing a non-category tag → pattern

        Args:
            mechanisms: List of mechanism dicts. If None, calls observe_mechanisms.

        Returns:
            List of PatternCard objects.
        """
        if mechanisms is None:
            mechanisms = [obs["context"] for obs in self.observe_mechanisms()]

        if not mechanisms:
            logger.info("No mechanisms observed — no patterns discovered")
            return []

        cards: list[PatternCard] = []
        used_mechanisms: set[int] = set()

        # Strategy 1: Category clustering (≥3 mechanisms)
        by_category: dict[str, list[dict[str, Any]]] = {}
        for i, m in enumerate(mechanisms):
            if not isinstance(m, dict):
                continue
            cat = m.get("category", "unknown")
            by_category.setdefault(cat, []).append((i, m))

        for category, items in by_category.items():
            if len(items) >= 3:
                indices, mechs = zip(*items)
                used_mechanisms.update(indices)

                names = [m.get("name", "") for m in mechs if m.get("name")]
                sources = []
                for m in mechs:
                    sp = m.get("source_provenance", [])
                    if isinstance(sp, list):
                        sources.extend(sp)

                all_tags: set[str] = set()
                for m in mechs:
                    t = m.get("tags", [])
                    if isinstance(t, list):
                        all_tags.update(t)

                cards.append(
                    PatternCard(
                        name=_generate_pattern_name(category, list(mechs)),
                        description=_generate_pattern_description(list(mechs)),
                        constituent_mechanisms=names,
                        occurrence_count=len(set(sources)) if sources else len(mechs),
                        confidence=min(0.5 + 0.1 * len(mechs), 0.9),
                        category=category,
                        tags=sorted(all_tags),
                        source_provenance=sorted(set(sources)),
                    )
                )

        # Strategy 2: Tag cross-clustering (≥2 mechanisms sharing a tag, not yet used)
        tag_to_indices: dict[str, set[int]] = {}
        for i, m in enumerate(mechanisms):
            if not isinstance(m, dict):
                continue
            if i in used_mechanisms:
                continue
            for tag in m.get("tags", []):
                if tag and tag != m.get("category", ""):
                    tag_to_indices.setdefault(tag, set()).add(i)

        for tag, indices in tag_to_indices.items():
            if len(indices) >= 2:
                mechs = [mechanisms[i] for i in indices]
                used_mechanisms.update(indices)

                names = [m.get("name", "") for m in mechs if m.get("name")]
                sources = []
                for m in mechs:
                    sp = m.get("source_provenance", [])
                    if isinstance(sp, list):
                        sources.extend(sp)

                cards.append(
                    PatternCard(
                        name=f"Cross-cutting pattern: {tag}",
                        description=_generate_pattern_description(mechs),
                        constituent_mechanisms=names,
                        occurrence_count=len(set(sources)) if sources else len(mechs),
                        confidence=min(0.4 + 0.15 * len(mechs), 0.8),
                        category="cross-cutting",
                        tags=[tag] + sorted({t for m in mechs for t in m.get("tags", []) if t}),
                        source_provenance=sorted(set(sources)),
                    )
                )

        logger.info("Pattern discover_patterns: %d pattern(s) from %d mechanism(s)", len(cards), len(mechanisms))
        return cards

    # ------------------------------------------------------------------
    # Proposal generation
    # ------------------------------------------------------------------

    def generate_proposal(self, patterns: list[PatternCard] | None = None) -> ImprovementProposal | None:
        """Generate an improvement proposal from discovered patterns.

        Args:
            patterns: List of PatternCards. If None, calls discover_patterns automatically.

        Returns:
            Improvement proposal, or None if no actionable findings.
        """
        if patterns is None:
            patterns = self.discover_patterns()

        if not patterns:
            logger.info("No patterns discovered — no proposal generated")
            return None

        # Categorize
        by_category: dict[str, list[PatternCard]] = {}
        for p in patterns:
            by_category.setdefault(p.category, []).append(p)

        top_category = max(by_category, key=lambda k: len(by_category[k]))
        top_count = len(by_category[top_category])
        total_mechanisms = sum(len(p.constituent_mechanisms) for p in patterns)

        evidence = [
            EvidenceItem(
                source="pattern",
                description=f"{p.name}: {p.description} ({len(p.constituent_mechanisms)} mechanisms)",
                data=p.to_dict(),
                timestamp=p.timestamp,
            )
            for p in patterns[:10]
        ]

        proposal = ImprovementProposal(
            id=f"PTN-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}",
            title=f"Pattern: {top_count} {top_category} pattern(s) discovered",
            problem=f"{len(patterns)} pattern(s) discovered from {total_mechanisms} mechanism(s) but not yet reduced to first principles",
            evidence=evidence,
            root_cause="Research Guild pipeline needs pattern synthesis before first-principles reasoning",
            recommendation=(
                "Feed discovered pattern cards into the First-Principles Citizen to reduce "
                "recurring structures to fundamental engineering truths. Prioritize patterns "
                f"with highest occurrence count in category '{top_category}'."
            ),
            alternatives_considered=[
                "Skip pattern synthesis and feed mechanisms to First-Principles (too granular)",
                "Manual pattern discovery (human-only, slower)",
            ],
            expected_benefits="Higher-level signal for downstream architectural reasoning",
            potential_risks=[
                RiskAssessment(
                    description="Heuristic clustering may group unrelated mechanisms",
                    severity="low",
                    mitigation="Human review of pattern cards; adjust clustering thresholds",
                    probability=0.3,
                ),
            ],
            confidence_score=0.6,
            estimated_effort_hours=1.0,
            affected_components=["ResearchGuild", "Memory"],
            evaluation_plan="Count patterns fed to First-Principles Citizen; measure principle yield",
            rollback_plan="Stop pattern synthesis; pipeline reverts to mechanism-level feeding",
            success_metrics=[
                f"{len(patterns)} pattern cards produced",
                "≥1 first principle per 3 pattern cards",
                "All patterns verified across ≥3 independent sources",
            ],
            status=ProposalStatus.DRAFT,
        )

        logger.info("Generated proposal %s: %s", proposal.id, proposal.title)
        return proposal

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def store_pattern(self, card: PatternCard) -> bool:
        """Store a pattern card in Animus memory.

        Args:
            card: PatternCard to store.

        Returns:
            True if stored successfully.
        """
        if self.memory is None:
            logger.warning("Memory layer not available — pattern not persisted")
            return False

        try:
            from animus.memory import MemoryType

            self.memory.remember(
                content=f"{card.name}: {card.description}",
                memory_type=MemoryType.SEMANTIC,
                tags=["pattern", "research_guild", card.category] + card.tags,
                metadata=card.to_dict(),
            )
            logger.info("Pattern '%s' stored in memory", card.name)
            return True
        except Exception as e:
            logger.error("Failed to store pattern: %s", e)
            return False

    def store_report(self, report: PatternReport) -> bool:
        """Store a pattern report in Animus memory.

        Args:
            report: Report to store.

        Returns:
            True if stored successfully.
        """
        if self.memory is None:
            logger.warning("Memory layer not available — report not persisted")
            return False

        try:
            from animus.memory import MemoryType

            self.memory.remember(
                content=report.summary(),
                memory_type=MemoryType.SEMANTIC,
                tags=["pattern", "research_guild", "report"],
                metadata={
                    "total_discovered": report.total_discovered,
                    "mechanisms_processed": report.mechanisms_processed,
                    "errors": report.errors,
                    "timestamp": report.timestamp.isoformat(),
                },
            )
            return True
        except Exception as e:
            logger.error("Failed to store report: %s", e)
            return False

    def store_proposal(self, proposal: ImprovementProposal) -> bool:
        """Store a proposal in Animus memory.

        Args:
            proposal: Proposal to store.

        Returns:
            True if stored successfully.
        """
        if self.memory is None:
            logger.warning("Memory layer not available — proposal not persisted")
            return False

        try:
            from animus.memory import MemoryType

            self.memory.remember(
                content=f"{proposal.title}\n\n{proposal.problem}\n\nRecommendation: {proposal.recommendation}",
                memory_type=MemoryType.PROCEDURAL,
                tags=["pattern", "proposal", proposal.status.value],
                metadata=proposal.to_dict(),
            )
            logger.info("Proposal %s stored in memory", proposal.id)
            return True
        except Exception as e:
            logger.error("Failed to store proposal: %s", e)
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def list_stored_patterns(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recently discovered patterns from memory.

        Args:
            limit: Maximum patterns to return.

        Returns:
            List of pattern dicts.
        """
        if self.memory is None:
            return []

        try:
            from animus.memory import MemoryType

            results = self.memory.search(
                query="pattern research_guild",
                memory_type=MemoryType.SEMANTIC,
                limit=limit,
            )
            return [
                {
                    "id": r.get("id", "") if hasattr(r, "get") else getattr(r, "id", ""),
                    "content": r.get("content", "") if hasattr(r, "get") else getattr(r, "content", ""),
                    "metadata": r.get("metadata", {}) if hasattr(r, "get") else getattr(r, "metadata", {}),
                }
                for r in results
            ]
        except Exception as e:
            logger.warning("list_stored_patterns failed: %s", e)
            return []

    def __repr__(self) -> str:
        return f"PatternCitizen(patterns={len(self._patterns)})"
