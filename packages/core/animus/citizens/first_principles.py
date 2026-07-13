"""Citizen 010 — The First-Principles Citizen.

The fourth stage of the Research Guild pipeline.

Responsibilities:
- Read PatternCard objects from memory (produced by Pattern Citizen)
- Reduce patterns to fundamental engineering truths
- Resolve contradictions between patterns (flag for human adjudication)
- Express principles that survive technology changes
- Produce PrincipleCard objects for the Architecture Citizen (next stage)

Never:
- Draft concrete RFCs or proposals (that's the Architecture Citizen)
- Modify pattern cards directly
- Act on findings without human approval

Instead:
    Read Patterns → Reduce to Principles → Flag Contradictions → Card → Human Approval → Architecture
"""

from __future__ import annotations

import re
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

logger = get_logger("citizens.first_principles")


# ═══════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════


@dataclass
class PrincipleCard:
    """A fundamental engineering truth distilled from one or more patterns.

    The First-Principles Citizen reduces recurring patterns to principles
    that survive technology changes — ideas that were true in 1990 and
    will be true in 2050.
    """

    principle_statement: str  # e.g., "Systems that separate concerns survive longer"
    supporting_patterns: list[str] = field(default_factory=list)  # Pattern names
    confidence: float = 0.5  # 0.0–1.0
    revision_history: list[str] = field(default_factory=list)  # Prior versions
    contradictions: list[str] = field(default_factory=list)  # Conflicting principle IDs
    category: str = ""  # e.g., "architecture", "reliability", "philosophy"
    tags: list[str] = field(default_factory=list)
    source_provenance: list[str] = field(default_factory=list)  # Source IDs
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "principle_statement": self.principle_statement,
            "supporting_patterns": self.supporting_patterns,
            "confidence": self.confidence,
            "revision_history": self.revision_history,
            "contradictions": self.contradictions,
            "category": self.category,
            "tags": self.tags,
            "source_provenance": self.source_provenance,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FirstPrinciplesReport:
    """Report produced by the First-Principles Citizen after a reduction run."""

    principles: list[PrincipleCard] = field(default_factory=list)
    patterns_processed: int = 0
    contradictions_found: int = 0
    errors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_reduced(self) -> int:
        return len(self.principles)

    def summary(self) -> str:
        parts = [
            f"{self.total_reduced} principle(s) reduced from {self.patterns_processed} pattern(s)",
        ]
        if self.contradictions_found:
            parts.append(f"{self.contradictions_found} contradiction(s) flagged for human adjudication")
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        return "; ".join(parts)


# ═══════════════════════════════════════════════════════════════════
# First-principles reduction rules
# ═══════════════════════════════════════════════════════════════════

# Mapping of pattern keywords → (principle_statement, category, tags)
_PRINCIPLE_RULES: list[tuple[re.Pattern, str, str, list[str]]] = [
    # State externalization
    (re.compile(r"\b(state|externaliz|checkpoint|immutable|source of truth)\b", re.IGNORECASE),
     "Systems that separate state from computation survive longer than systems that conflate them.",
     "architecture",
     ["state", "resilience", "portability"]),
    # Resilience / Fault tolerance
    (re.compile(r"\b(resilien|fault toleran|graceful|recovery|fail.*safe)\b", re.IGNORECASE),
     "Resilience is not a feature you add; it is a property you design for from the beginning.",
     "reliability",
     ["resilience", "design", "systems-thinking"]),
    # Async / Decoupling
    (re.compile(r"\b(async|decoupl|loose coupl|message|queue|event)\b", re.IGNORECASE),
     "Decoupling producers from consumers is the single most effective way to scale systems under uncertainty.",
     "architecture",
     ["decoupling", "scalability", "async"]),
    # Observability
    (re.compile(r"\b(observ|telemetry|trace|metric|monitor|dashboard)\b", re.IGNORECASE),
     "You cannot operate what you cannot observe. Observability is a prerequisite for reliability at scale.",
     "operations",
     ["observability", "telemetry", "operations"]),
    # Security / Identity
    (re.compile(r"\b(secur|identity|auth|encrypt|permission|trust)\b", re.IGNORECASE),
     "Security is a property of the system, not a layer you bolt on afterward.",
     "security",
     ["security", "trust", "design"]),
    # Testing / Quality
    (re.compile(r"\b(test|quality|verif|correct|contract|invariant)\b", re.IGNORECASE),
     "Testability is an architectural concern. If you cannot test it, you do not understand it.",
     "quality",
     ["testing", "quality", "understanding"]),
    # Abstraction / Inversion
    (re.compile(r"\b(abstract|inversion|coupling|interface|contract)\b", re.IGNORECASE),
     "Depend on abstractions, not concretions. This is the foundation of maintainable software.",
     "architecture",
     ["abstraction", "coupling", "maintainability"]),
    # Performance / Optimization
    (re.compile(r"\b(perform|optim|scal|throughput|latenc|bottleneck)\b", re.IGNORECASE),
     "Performance is a function of design, not tuning. Choose the right structure before optimizing the details.",
     "performance",
     ["performance", "design", "optimization"]),
    # Idempotency / Correctness
    (re.compile(r"\b(idempot|exactly once|correct|determin|repeatable)\b", re.IGNORECASE),
     "Correctness requires determinism. The same input must produce the same outcome, regardless of how many times the operation runs.",
     "reliability",
     ["correctness", "determinism", "idempotency"]),
    # Bounded context / Modularity
    (re.compile(r"\b(bounded|modular|domain|context|separation of concerns|cohesion)\b", re.IGNORECASE),
     "Complexity is managed by boundaries. Systems with clear boundaries outlast systems with vague ones.",
     "architecture",
     ["complexity", "boundaries", "modularity"]),
]

# Contradiction pairs: (keyword_a, keyword_b, description)
_CONTRADICTION_PAIRS: list[tuple[str, str, str]] = [
    ("stateless", "state externalization", "Stateless vs. state externalization tension"),
    ("strict typing", "dynamic", "Strict typing vs. dynamic flexibility tension"),
    ("centralized", "decentralized", "Centralized control vs. decentralized autonomy tension"),
    ("optimistic", "pessimistic", "Optimistic vs. pessimistic concurrency tension"),
    ("synchronous", "async", "Synchronous simplicity vs. async scalability tension"),
]


# ═══════════════════════════════════════════════════════════════════
# First-Principles Citizen
# ═══════════════════════════════════════════════════════════════════


class FirstPrinciplesCitizen:
    """Citizen 010 — The First-Principles Citizen.

    Reads pattern cards, reduces them to fundamental engineering truths,
    flags contradictions, and produces PrincipleCards.
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

        self._principles: list[PrincipleCard] = []

    # ------------------------------------------------------------------
    # Observation methods (for autonomous loop compatibility)
    # ------------------------------------------------------------------

    def observe_patterns(self) -> list[dict[str, Any]]:
        """Read pattern cards from memory produced by the Pattern Citizen.

        Returns:
            List of observation dicts compatible with autonomous loop.
        """
        findings: list[dict[str, Any]] = []
        if self.memory is None:
            logger.warning("Memory layer not available — observe_patterns skipped")
            return findings

        try:
            from animus.memory import MemoryType

            results = self.memory.search(
                query="pattern research_guild",
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
                constituent_mechanisms = meta.get("constituent_mechanisms", [])
                tags = meta.get("tags", [])
                source_provenance = meta.get("source_provenance", [])

                if name or description:
                    findings.append({
                        "source": "memory",
                        "description": f"Pattern: {name or description[:60]}",
                        "severity": "info",
                        "context": {
                            "name": name,
                            "category": category,
                            "description": description,
                            "constituent_mechanisms": constituent_mechanisms,
                            "tags": tags,
                            "source_provenance": source_provenance,
                            "pattern_type": "pattern_card",
                        },
                    })

        except Exception as e:
            logger.warning("observe_patterns failed: %s", e)

        logger.info("FirstPrinciples observe_patterns: %d findings", len(findings))
        return findings

    # ------------------------------------------------------------------
    # Principle reduction
    # ------------------------------------------------------------------

    def reduce_to_principles(self, patterns: list[dict[str, Any]] | None = None) -> list[PrincipleCard]:
        """Reduce patterns to fundamental engineering principles.

        Uses keyword matching against _PRINCIPLE_RULES to map patterns
        to timeless principles. Flags contradictions for human review.

        Args:
            patterns: List of pattern dicts. If None, calls observe_patterns.

        Returns:
            List of PrincipleCard objects.
        """
        if patterns is None:
            patterns = [obs["context"] for obs in self.observe_patterns()]

        if not patterns:
            logger.info("No patterns observed — no principles reduced")
            return []

        cards: list[PrincipleCard] = []

        for pattern in patterns:
            if not isinstance(pattern, dict):
                continue

            text = f"{pattern.get('name', '')} {pattern.get('description', '')}"
            if not text.strip():
                continue

            for pattern_re, statement, category, tags in _PRINCIPLE_RULES:
                if pattern_re.search(text):
                    cards.append(
                        PrincipleCard(
                            principle_statement=statement,
                            supporting_patterns=[pattern.get("name", "")],
                            confidence=0.6,
                            category=category,
                            tags=list(tags),
                            source_provenance=pattern.get("source_provenance", []),
                        )
                    )
                    break  # Only match the first applicable rule per pattern

        # Deduplicate: merge principles with identical statements
        merged: dict[str, PrincipleCard] = {}
        for card in cards:
            if card.principle_statement in merged:
                existing = merged[card.principle_statement]
                existing.supporting_patterns.extend(card.supporting_patterns)
                existing.source_provenance.extend(card.source_provenance)
                existing.tags = list(set(existing.tags) | set(card.tags))
                existing.confidence = min(existing.confidence + 0.1, 0.95)
            else:
                merged[card.principle_statement] = card

        # Find contradictions between principles
        principle_list = list(merged.values())
        for card in principle_list:
            for keyword_a, keyword_b, description in _CONTRADICTION_PAIRS:
                text = card.principle_statement.lower()
                if keyword_a in text and keyword_b in text:
                    card.contradictions.append(description)

        logger.info("FirstPrinciples reduce_to_principles: %d principle(s) from %d pattern(s)", len(principle_list), len(patterns))
        return principle_list

    # ------------------------------------------------------------------
    # Proposal generation
    # ------------------------------------------------------------------

    def generate_proposal(self, principles: list[PrincipleCard] | None = None) -> ImprovementProposal | None:
        """Generate an improvement proposal from reduced principles.

        Args:
            principles: List of PrincipleCards. If None, calls reduce_to_principles automatically.

        Returns:
            Improvement proposal, or None if no actionable findings.
        """
        if principles is None:
            principles = self.reduce_to_principles()

        if not principles:
            logger.info("No principles reduced — no proposal generated")
            return None

        # Categorize
        by_category: dict[str, list[PrincipleCard]] = {}
        for p in principles:
            by_category.setdefault(p.category, []).append(p)

        top_category = max(by_category, key=lambda k: len(by_category[k]))
        top_count = len(by_category[top_category])
        total_patterns = sum(len(p.supporting_patterns) for p in principles)
        total_contradictions = sum(len(p.contradictions) for p in principles)

        evidence = [
            EvidenceItem(
                source="first_principles",
                description=f"{p.principle_statement} (supports: {', '.join(p.supporting_patterns[:3])})",
                data=p.to_dict(),
                timestamp=p.timestamp,
            )
            for p in principles[:10]
        ]

        proposal = ImprovementProposal(
            id=f"FP-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}",
            title=f"First Principles: {top_count} {top_category} principle(s) reduced",
            problem=f"{len(principles)} principle(s) reduced from {total_patterns} pattern(s) but not yet drafted into architecture proposals",
            evidence=evidence,
            root_cause="Research Guild pipeline needs first-principles synthesis before architecture drafting",
            recommendation=(
                "Feed reduced principle cards into the Architecture Citizen to draft "
                "concrete RFCs and Improvement Proposals for Animus adoption. Prioritize "
                f"principles with highest confidence in category '{top_category}'."
            ),
            alternatives_considered=[
                "Skip first-principles and feed patterns to Architecture (too granular)",
                "Manual first-principles reasoning (human-only, slower)",
            ],
            expected_benefits="Timeless architectural guidance that survives technology churn",
            potential_risks=[
                RiskAssessment(
                    description="Heuristic reduction may oversimplify nuanced patterns",
                    severity="low",
                    mitigation="Human review of principle cards; refine rule set over time",
                    probability=0.3,
                ),
            ],
            confidence_score=0.6,
            estimated_effort_hours=1.0,
            affected_components=["ResearchGuild", "Memory"],
            evaluation_plan="Count principles fed to Architecture Citizen; measure RFC yield",
            rollback_plan="Stop first-principles synthesis; pipeline reverts to pattern-level feeding",
            success_metrics=[
                f"{len(principles)} principle cards produced",
                f"{total_contradictions} contradictions flagged for human review",
                "≥1 architecture proposal per 2 principle cards",
            ],
            status=ProposalStatus.DRAFT,
        )

        logger.info("Generated proposal %s: %s", proposal.id, proposal.title)
        return proposal

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def store_principle(self, card: PrincipleCard) -> bool:
        """Store a principle card in Animus memory.

        Args:
            card: PrincipleCard to store.

        Returns:
            True if stored successfully.
        """
        if self.memory is None:
            logger.warning("Memory layer not available — principle not persisted")
            return False

        try:
            from animus.memory import MemoryType

            self.memory.remember(
                content=card.principle_statement,
                memory_type=MemoryType.SEMANTIC,
                tags=["first_principles", "research_guild", "principle", card.category] + card.tags,
                metadata=card.to_dict(),
            )
            logger.info("Principle stored in memory: %s...", card.principle_statement[:60])
            return True
        except Exception as e:
            logger.error("Failed to store principle: %s", e)
            return False

    def store_report(self, report: FirstPrinciplesReport) -> bool:
        """Store a first-principles report in Animus memory.

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
                tags=["first_principles", "research_guild", "report"],
                metadata={
                    "total_reduced": report.total_reduced,
                    "patterns_processed": report.patterns_processed,
                    "contradictions_found": report.contradictions_found,
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
                tags=["first_principles", "proposal", proposal.status.value],
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

    def list_stored_principles(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recently reduced principles from memory.

        Args:
            limit: Maximum principles to return.

        Returns:
            List of principle dicts.
        """
        if self.memory is None:
            return []

        try:
            from animus.memory import MemoryType

            results = self.memory.search(
                query="first_principles principle research_guild",
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
            logger.warning("list_stored_principles failed: %s", e)
            return []

    def __repr__(self) -> str:
        return f"FirstPrinciplesCitizen(principles={len(self._principles)})"
