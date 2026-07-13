"""Citizen Council — Unified backlog from all Phase 0 citizens.

The Council synthesizes proposals from Architect, Conversation Designer,
Knowledge Curator, and Test Oracle into a single ranked backlog. It
handles deduplication, severity-weighted ranking, and produces a
consolidated view for human review.

Usage:
    council = CitizenCouncil(memory_layer=memory)
    council.collect_from_memory()
    ranked = council.rank_backlog()
    for item in ranked:
        print(f"{item.rank}. [{item.priority_score:.2f}] {item.proposal.title}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from animus.citizens.proposal import ImprovementProposal, ProposalConfidence, ProposalStatus
from animus.logging import get_logger

logger = get_logger("citizens.citizen_council")


@dataclass
class RankedProposal:
    """A proposal with council-computed ranking metadata."""

    proposal: ImprovementProposal
    rank: int = 0
    priority_score: float = 0.0
    severity_score: int = 0
    source_citizens: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)  # IDs of dupes

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal": self.proposal.to_dict(),
            "rank": self.rank,
            "priority_score": round(self.priority_score, 3),
            "severity_score": self.severity_score,
            "source_citizens": self.source_citizens,
            "duplicates": self.duplicates,
        }


class CitizenCouncil:
    """Collect, deduplicate, and rank proposals from all citizens.

    The Council is read-only — it never modifies proposals or executes
    them. Its output is a ranked list for human review and approval.
    """

    # Severity mapping for numeric scoring
    SEVERITY_MAP = {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "low": 1,
    }

    def __init__(self, memory_layer: Any = None):
        self.memory = memory_layer
        self._proposals: dict[str, RankedProposal] = {}

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def collect_from_citizens(self, citizens: dict[str, Any]) -> int:
        """Collect proposals directly from citizen instances.

        Args:
            citizens: Dict mapping citizen name -> citizen instance.
                Each instance must have a ``generate_proposal()`` method.

        Returns:
            Number of proposals collected.
        """
        count = 0
        for name, citizen in citizens.items():
            try:
                proposal = citizen.generate_proposal()
                if proposal:
                    self._add_proposal(proposal, source=name)
                    count += 1
            except Exception as e:
                logger.warning(f"CitizenCouncil: {name} failed to generate proposal: {e}")
        return count

    def collect_from_memory(self, citizen_names: list[str] | None = None) -> int:
        """Load proposals from Animus memory.

        Searches memory for procedural memories tagged with citizen names.

        Args:
            citizen_names: Optional list of citizen names to filter by.
                Defaults to all Phase 0 citizens.

        Returns:
            Number of proposals loaded.
        """
        if self.memory is None:
            logger.warning("CitizenCouncil: no memory layer available")
            return 0

        names = citizen_names or [
            "architect",
            "conversation_designer",
            "harvester",
            "intelligence",
            "knowledge_curator",
            "test_oracle",
        ]
        count = 0
        try:
            from animus.memory import MemoryType
            for name in names:
                try:
                    results = self.memory.search(
                        query=f"{name} proposal",
                        memory_type=MemoryType.PROCEDURAL,
                        limit=50,
                    )
                except Exception:
                    continue
                for mem in results:
                    meta = mem.get("metadata", {}) if hasattr(mem, "get") else getattr(mem, "metadata", {})
                    if not meta or not meta.get("id"):
                        continue
                    try:
                        proposal = ImprovementProposal.from_dict(meta)
                        self._add_proposal(proposal, source=name)
                        count += 1
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"CitizenCouncil: memory search failed: {e}")
        return count

    def _add_proposal(self, proposal: ImprovementProposal, source: str) -> None:
        """Internal: add or merge a proposal into the council index."""
        existing = self._proposals.get(proposal.id)
        if existing:
            if source not in existing.source_citizens:
                existing.source_citizens.append(source)
        else:
            self._proposals[proposal.id] = RankedProposal(
                proposal=proposal,
                source_citizens=[source],
            )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_priority_score(self, rp: RankedProposal) -> float:
        """Compute a composite priority score.

        Higher score = more urgent / higher value.

        Factors:
          - Severity (critical > high > medium > low)
          - Confidence (higher confidence = more reliable)
          - Inverse effort (smaller effort = easier win)
          - Component count (more components = broader impact)
        """
        p = rp.proposal

        # Severity from evidence items (take max)
        max_sev = 1
        for ev in p.evidence:
            if hasattr(ev, "severity"):
                max_sev = max(max_sev, self.SEVERITY_MAP.get(ev.severity, 1))

        # If no evidence severity, infer from confidence label
        if max_sev == 1:
            max_sev = {
                ProposalConfidence.VERY_HIGH: 3,
                ProposalConfidence.HIGH: 3,
                ProposalConfidence.MEDIUM: 2,
                ProposalConfidence.LOW: 1,
                ProposalConfidence.VERY_LOW: 1,
            }.get(p.confidence_label, 1)

        rp.severity_score = max_sev

        confidence = p.confidence_score
        effort = max(p.estimated_effort_hours, 0.5)
        component_bonus = min(len(p.affected_components), 5) * 0.1

        # Specificity bonus: file-level paths (e.g. "core/app.py") are more
        # actionable than vague top-level names (e.g. "Factory").
        specificity_bonus = 0.0
        for comp in p.affected_components:
            if "." in comp or ("/" in comp and len(comp.split("/")) > 1):
                specificity_bonus += 0.15
        specificity_bonus = min(specificity_bonus, 0.5)

        # Structural bonus: cross-module architectural issues rank higher
        # than surface-level code quality (long functions, missing docs).
        structural_patterns = {
            "circular_import": 0.6,
            "tight_coupling": 0.5,
            "god_class": 0.5,
            "singleton_abuse": 0.3,
            "interface_leakage": 0.3,
            "leaky_abstraction": 0.3,
            "eval_regression": 0.4,
            "eval_failure": 0.3,
        }
        structural_bonus = 0.0
        for ev in p.evidence:
            ptype = ev.data.get("pattern_type", "") if hasattr(ev, "data") else ""
            if ptype in structural_patterns:
                structural_bonus = max(structural_bonus, structural_patterns[ptype])

        # Score formula: severity * confidence / effort_factor
        score = (max_sev * confidence / (1 + effort / 8)) + component_bonus + specificity_bonus + structural_bonus
        return round(score, 3)

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_component(comp: str) -> str:
        """Normalize a component path for deduplication.

        Strips trailing slashes, collapses multiple slashes, and ensures
        consistent separators.
        """
        normalized = comp.strip().replace("//", "/")
        if normalized.endswith("/") and len(normalized) > 1:
            normalized = normalized[:-1]
        return normalized

    def _deduplicate(self, ranked: list[RankedProposal]) -> list[RankedProposal]:
        """Remove duplicate proposals by component overlap.

        If two proposals share any affected_component, the lower-scoring
        one is marked as a duplicate of the higher-scoring one and
        removed from the returned list.
        """
        sorted_by_score = sorted(ranked, key=lambda rp: -rp.priority_score)
        kept: list[RankedProposal] = []
        component_index: dict[str, RankedProposal] = {}

        for rp in sorted_by_score:
            components = {self._normalize_component(c) for c in rp.proposal.affected_components}
            duplicate_of = None
            for comp in components:
                if comp in component_index:
                    other = component_index[comp]
                    if other.proposal.id != rp.proposal.id:
                        duplicate_of = other
                        break

            if duplicate_of:
                duplicate_of.duplicates.append(rp.proposal.id)
                logger.debug(
                    f"CitizenCouncil: deduplicated {rp.proposal.id} as dupe of "
                    f"{duplicate_of.proposal.id} (shared component)"
                )
                continue

            kept.append(rp)
            for comp in components:
                component_index[comp] = rp

        return kept

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_backlog(self, deduplicate: bool = True) -> list[RankedProposal]:
        """Produce a unified, ranked backlog.

        Args:
            deduplicate: Whether to remove duplicates by component overlap.

        Returns:
            List of RankedProposal sorted by priority_score descending.
        """
        for rp in self._proposals.values():
            rp.priority_score = self._compute_priority_score(rp)

        ranked = sorted(self._proposals.values(), key=lambda rp: -rp.priority_score)

        if deduplicate:
            ranked = self._deduplicate(ranked)

        for i, rp in enumerate(ranked, start=1):
            rp.rank = i

        return ranked

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_component(self, component: str) -> list[RankedProposal]:
        """Return proposals affecting a specific component."""
        return [
            rp for rp in self._proposals.values()
            if component in rp.proposal.affected_components
        ]

    def filter_by_confidence(self, min_confidence: float = 0.5) -> list[RankedProposal]:
        """Return proposals with confidence_score >= threshold."""
        return [
            rp for rp in self._proposals.values()
            if rp.proposal.confidence_score >= min_confidence
        ]

    def filter_by_status(self, status: ProposalStatus) -> list[RankedProposal]:
        """Return proposals matching a specific status."""
        return [
            rp for rp in self._proposals.values()
            if rp.proposal.status == status
        ]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Get a summary of the council state."""
        sources: dict[str, int] = {}
        components: set[str] = set()
        total_effort = 0.0

        for rp in self._proposals.values():
            for src in rp.source_citizens:
                sources[src] = sources.get(src, 0) + 1
            components.update(rp.proposal.affected_components)
            total_effort += rp.proposal.estimated_effort_hours

        return {
            "total_proposals": len(self._proposals),
            "unique_components": len(components),
            "total_estimated_effort_hours": round(total_effort, 1),
            "sources": sources,
        }

    def clear(self) -> None:
        """Clear all collected proposals."""
        self._proposals.clear()
