"""Citizen 011 — The Architecture Citizen.

The fifth and final stage of the Research Guild pipeline before Forge.

Responsibilities:
- Read PrincipleCard objects from memory (produced by First-Principles Citizen)
- Compare each principle to the existing Animus codebase
- Identify gaps where the principle would improve Animus
- Draft concrete Improvement Proposals with specific recommendations
- Estimate integration effort and value

Never:
- Execute proposals directly (that's Forge)
- Modify code or memory autonomously
- Act on findings without human approval

Instead:
    Read Principles → Analyze Gaps → Draft Proposal → Human Approval → Forge
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
    ProposalStatus,
    RiskAssessment,
)
from animus.logging import get_logger

if TYPE_CHECKING:
    from animus.memory import MemoryLayer

logger = get_logger("citizens.architecture_citizen")


# ═══════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════


@dataclass
class GapAnalysis:
    """A gap identified between a principle and the current codebase."""

    principle_statement: str
    principle_category: str = ""
    gap_description: str = ""
    severity: str = "medium"  # low, medium, high, critical
    affected_files: list[str] = field(default_factory=list)
    keyword_matches: int = 0
    keyword_total: int = 0
    coverage_ratio: float = 0.0  # matches / total keywords
    confidence: float = 0.5
    recommendation: str = ""
    estimated_effort_hours: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "principle_statement": self.principle_statement,
            "principle_category": self.principle_category,
            "gap_description": self.gap_description,
            "severity": self.severity,
            "affected_files": self.affected_files,
            "keyword_matches": self.keyword_matches,
            "keyword_total": self.keyword_total,
            "coverage_ratio": self.coverage_ratio,
            "confidence": self.confidence,
            "recommendation": self.recommendation,
            "estimated_effort_hours": self.estimated_effort_hours,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ArchitectureReport:
    """Report produced by the Architecture Citizen after a gap analysis run."""

    gaps: list[GapAnalysis] = field(default_factory=list)
    principles_processed: int = 0
    errors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_gaps(self) -> int:
        return len(self.gaps)

    @property
    def critical_gaps(self) -> int:
        return sum(1 for g in self.gaps if g.severity == "critical")

    @property
    def high_gaps(self) -> int:
        return sum(1 for g in self.gaps if g.severity == "high")

    def summary(self) -> str:
        parts = [
            f"{self.total_gaps} gap(s) identified from {self.principles_processed} principle(s)",
        ]
        if self.critical_gaps:
            parts.append(f"{self.critical_gaps} critical")
        if self.high_gaps:
            parts.append(f"{self.high_gaps} high")
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        return "; ".join(parts)


# ═══════════════════════════════════════════════════════════════════
# Gap analysis heuristics
# ═══════════════════════════════════════════════════════════════════

# Category → (keywords to search, subsystem areas)
_GAP_KEYWORDS: dict[str, list[str]] = {
    "architecture": [
        "separation",
        "modular",
        "boundary",
        "layer",
        "interface",
        "abstraction",
        "coupling",
    ],
    "reliability": [
        "resilien",
        "fault",
        "retry",
        "circuit",
        "backoff",
        "graceful",
        "timeout",
        "recovery",
    ],
    "operations": [
        "observ",
        "metric",
        "trace",
        "log",
        "monitor",
        "telemetry",
        "alert",
        "dashboard",
    ],
    "security": ["auth", "encrypt", "tls", "permission", "identity", "rbac", "vault", "secret"],
    "quality": ["test", "mock", "fixture", "coverage", "contract", "validation", "assert"],
    "performance": [
        "cache",
        "memoiz",
        "paginat",
        "batch",
        "stream",
        "throttl",
        "limit",
        "optimize",
    ],
    "deployment": ["feature flag", "canary", "rollout", "toggle", "blue.green"],
}


def _extract_keywords(text: str) -> list[str]:
    """Extract searchable keywords from a principle statement."""
    words = re.findall(r"\b[a-z]{4,}\b", text.lower())
    # Filter to meaningful terms
    meaningful = {
        "separate",
        "separation",
        "state",
        "computation",
        "survive",
        "resilience",
        "fault",
        "tolerant",
        "graceful",
        "decouple",
        "async",
        "scale",
        "observ",
        "security",
        "design",
        "testability",
        "architectural",
        "abstraction",
        "performance",
        "correctness",
        "determinism",
        "complexity",
        "boundary",
        "modular",
        "maintainable",
        "reliable",
        "identity",
        "verify",
    }
    return [w for w in words if w in meaningful or len(w) > 5]


def _score_gap(keyword_matches: int, keyword_total: int) -> tuple[str, float]:
    """Score a gap based on keyword coverage ratio.

    Returns:
        (severity, coverage_ratio)
    """
    if keyword_total == 0:
        return ("medium", 0.0)
    ratio = keyword_matches / keyword_total
    if ratio >= 0.5:
        return ("low", ratio)
    elif ratio >= 0.25:
        return ("medium", ratio)
    elif ratio >= 0.1:
        return ("high", ratio)
    else:
        return ("critical", ratio)


# ═══════════════════════════════════════════════════════════════════
# Architecture Citizen
# ═══════════════════════════════════════════════════════════════════


class ArchitectureCitizen:
    """Citizen 011 — The Architecture Citizen.

    Reads principles, compares them to the codebase, identifies gaps,
    and drafts concrete Improvement Proposals for Forge execution.
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

        self._gaps: list[GapAnalysis] = []

    # ------------------------------------------------------------------
    # Observation methods (for autonomous loop compatibility)
    # ------------------------------------------------------------------

    def observe_principles(self) -> list[dict[str, Any]]:
        """Read principle cards from memory produced by the First-Principles Citizen.

        Returns:
            List of observation dicts compatible with autonomous loop.
        """
        findings: list[dict[str, Any]] = []
        if self.memory is None:
            logger.warning("Memory layer not available — observe_principles skipped")
            return findings

        try:
            from animus.memory import MemoryType

            results = self.memory.search(
                query="first_principles principle research_guild",
                memory_type=MemoryType.SEMANTIC,
                limit=50,
            )
            for mem in results:
                meta = (
                    mem.get("metadata", {}) if hasattr(mem, "get") else getattr(mem, "metadata", {})
                )
                if not isinstance(meta, dict):
                    meta = {}

                statement = meta.get("principle_statement", "")
                category = meta.get("category", "")
                supporting = meta.get("supporting_patterns", [])
                tags = meta.get("tags", [])
                source_provenance = meta.get("source_provenance", [])

                if statement:
                    findings.append(
                        {
                            "source": "memory",
                            "description": f"Principle: {statement[:80]}",
                            "severity": "info",
                            "context": {
                                "principle_statement": statement,
                                "category": category,
                                "supporting_patterns": supporting,
                                "tags": tags,
                                "source_provenance": source_provenance,
                                "pattern_type": "principle_card",
                            },
                        }
                    )

        except Exception as e:
            logger.warning("observe_principles failed: %s", e)

        logger.info("ArchitectureCitizen observe_principles: %d findings", len(findings))
        return findings

    # ------------------------------------------------------------------
    # Gap analysis
    # ------------------------------------------------------------------

    def analyze_gaps(self, principles: list[dict[str, Any]] | None = None) -> list[GapAnalysis]:
        """Compare principles to the codebase and identify gaps.

        For each principle, searches the codebase for relevant keywords.
        Low coverage indicates a gap where the principle is not yet
        reflected in the architecture.

        For media-derived principles (tagged "media"), uses a lightweight
        gap-proxy approach instead of keyword coverage: trusts Ogma's gap
        assessment and performs only a file-existence check for the
        specific modules mentioned.

        Args:
            principles: List of principle dicts. If None, calls observe_principles.

        Returns:
            List of GapAnalysis objects.
        """
        if principles is None:
            principles = [obs["context"] for obs in self.observe_principles()]

        if not principles:
            logger.info("No principles observed — no gaps analyzed")
            return []

        gaps: list[GapAnalysis] = []

        for principle in principles:
            if not isinstance(principle, dict):
                continue

            statement = principle.get("principle_statement", "")
            category = principle.get("category", "")
            tags = principle.get("tags", [])
            if not statement:
                continue

            is_media = "media" in tags

            if is_media:
                gap = self._analyze_media_gap(principle)
                if gap:
                    gaps.append(gap)
                continue

            # Standard keyword-based analysis for non-media principles
            keywords = _extract_keywords(statement)
            category_keywords = _GAP_KEYWORDS.get(category, [])
            if category_keywords:
                keywords = list(set(keywords) | set(category_keywords))

            if not keywords:
                continue

            # Search codebase
            matches: set[str] = set()
            affected_files: list[str] = []
            total_searched = 0

            if self.codebase_path.exists():
                for py_file in self.codebase_path.rglob("*.py"):
                    if any(
                        part.startswith(".")
                        or part in ("__pycache__", "node_modules", "venv", ".venv")
                        for part in py_file.parts
                    ):
                        continue
                    try:
                        text = py_file.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        continue

                    rel = str(py_file.relative_to(self.codebase_path))
                    file_matches = 0
                    for kw in keywords:
                        if kw.lower() in text.lower():
                            file_matches += 1
                            matches.add(kw)

                    if file_matches > 0:
                        affected_files.append(rel)

                    total_searched += 1
                    if total_searched > 500:
                        break  # Cap search to avoid slow analysis

            severity, coverage = _score_gap(len(matches), len(keywords))

            gap = GapAnalysis(
                principle_statement=statement,
                principle_category=category,
                gap_description=f"Principle '{statement[:60]}...' has {coverage:.0%} coverage in codebase ({len(matches)}/{len(keywords)} keywords found)",
                severity=severity,
                affected_files=affected_files[:10],
                keyword_matches=len(matches),
                keyword_total=len(keywords),
                coverage_ratio=coverage,
                confidence=0.5 + (coverage * 0.3),
                recommendation=self._draft_recommendation(
                    statement, category, severity, affected_files
                ),
                estimated_effort_hours=self._estimate_effort(severity, len(affected_files)),
            )
            gaps.append(gap)

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        gaps.sort(key=lambda g: severity_order.get(g.severity, 4))

        logger.info(
            "ArchitectureCitizen analyze_gaps: %d gap(s) from %d principle(s)",
            len(gaps),
            len(principles),
        )
        return gaps

    def _analyze_media_gap(self, principle: dict[str, Any]) -> GapAnalysis | None:
        """Analyze gap for a media-derived principle using Ogma's assessment.

        Uses Ogma's gap assessment + lightweight file existence check
        instead of keyword coverage ratio.
        """
        statement = principle.get("principle_statement", "")
        category = principle.get("category", "")
        tags = principle.get("tags", [])

        # Severity based on Ogma's gap status (if available in tags or metadata)
        gap_status = "NONE"
        for tag in tags:
            if tag.startswith("gap:"):
                gap_status = tag.split(":", 1)[1]
                break

        # Default severity mapping
        severity_map = {
            "NONE": "critical",
            "PARTIAL": "medium",
            "FULL": "low",
        }
        severity = severity_map.get(gap_status, "medium")

        # Lightweight check: look for module paths mentioned in the principle statement
        affected_files: list[str] = []
        if self.codebase_path.exists():
            # Extract potential module paths (e.g., packages/core/animus/...)
            path_pattern = re.compile(r"packages/core/animus/[a-z_/]+")
            for match in path_pattern.finditer(statement):
                candidate = match.group(0)
                full_path = self.codebase_path / candidate
                if full_path.exists():
                    affected_files.append(candidate)

        gap = GapAnalysis(
            principle_statement=statement,
            principle_category=category,
            gap_description=f"Media-derived principle '{statement[:60]}...' assessed via Ogma gap status ({gap_status})",
            severity=severity,
            affected_files=affected_files[:10],
            keyword_matches=len(affected_files),
            keyword_total=1,
            coverage_ratio=1.0 if affected_files else 0.0,
            confidence=0.6,
            recommendation=self._draft_recommendation(
                statement, category, severity, affected_files
            ),
            estimated_effort_hours=self._estimate_effort(severity, len(affected_files)),
        )
        return gap

    def _draft_recommendation(
        self, statement: str, category: str, severity: str, affected_files: list[str]
    ) -> str:
        """Draft a concrete recommendation for closing a gap."""
        rec = f"Consider applying the principle: '{statement[:100]}'"
        if affected_files:
            rec += f". Review files: {', '.join(affected_files[:3])}"
        if severity == "critical":
            rec += ". This is a significant architectural gap — prioritize for next sprint."
        elif severity == "high":
            rec += ". Moderate gap — include in upcoming refactoring plan."
        else:
            rec += ". Low-priority gap — address opportunistically."
        return rec

    def _estimate_effort(self, severity: str, affected_file_count: int) -> float:
        """Estimate integration effort in hours."""
        base = {"critical": 16.0, "high": 8.0, "medium": 4.0, "low": 2.0}.get(severity, 4.0)
        file_multiplier = min(affected_file_count * 0.5, 8.0)
        return round(base + file_multiplier, 1)

    # ------------------------------------------------------------------
    # Proposal generation
    # ------------------------------------------------------------------

    def generate_proposal(
        self, gaps: list[GapAnalysis] | None = None
    ) -> ImprovementProposal | None:
        """Generate an improvement proposal from identified gaps.

        Args:
            gaps: List of GapAnalysis objects. If None, calls analyze_gaps automatically.

        Returns:
            Improvement proposal, or None if no actionable findings.
        """
        if gaps is None:
            gaps = self.analyze_gaps()

        if not gaps:
            logger.info("No gaps identified — no proposal generated")
            return None

        # Categorize
        by_category: dict[str, list[GapAnalysis]] = {}
        for g in gaps:
            by_category.setdefault(g.principle_category, []).append(g)

        top_category = max(by_category, key=lambda k: len(by_category[k]))
        top_count = len(by_category[top_category])
        critical_count = sum(1 for g in gaps if g.severity == "critical")
        high_count = sum(1 for g in gaps if g.severity == "high")
        total_effort = sum(g.estimated_effort_hours for g in gaps)

        evidence = [
            EvidenceItem(
                source="architecture_citizen",
                description=f"[{g.severity.upper()}] {g.principle_statement[:60]}... — {g.coverage_ratio:.0%} coverage",
                data=g.to_dict(),
                timestamp=g.timestamp,
            )
            for g in gaps[:10]
        ]

        recommendation_parts = [
            f"Address {top_count} gap(s) in category '{top_category}'.",
            f"Total estimated effort: {total_effort:.1f}h.",
        ]
        if critical_count:
            recommendation_parts.append(f"Prioritize {critical_count} critical gap(s) immediately.")
        if high_count:
            recommendation_parts.append(
                f"Schedule {high_count} high-severity gap(s) for next sprint."
            )
        recommendation_parts.append(
            "Feed selected gaps into Forge for feasibility analysis and safe implementation."
        )

        proposal = ImprovementProposal(
            id=f"ARCH-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}",
            title=f"Architecture: {top_count} {top_category} gap(s) need addressing",
            problem=f"{len(gaps)} architectural gap(s) identified where first principles are not yet reflected in Animus. {critical_count} critical, {high_count} high.",
            evidence=evidence,
            root_cause="Research Guild pipeline has distilled principles but Animus codebase has not yet adopted them",
            recommendation=" ".join(recommendation_parts),
            alternatives_considered=[
                "Ignore gaps and continue with current architecture (technical debt accumulates)",
                "Manual architecture review (human-only, slower, less systematic)",
            ],
            expected_benefits="Animus architecture progressively aligned with timeless engineering principles",
            potential_risks=[
                RiskAssessment(
                    description="Heuristic gap analysis may misidentify coverage (false positives/negatives)",
                    severity="low",
                    mitigation="Human review of gap list before Forge execution; refine keyword mapping",
                    probability=0.3,
                ),
                RiskAssessment(
                    description="Large refactoring effort may destabilize existing functionality",
                    severity="medium",
                    mitigation="Use Forge's safe execution pipeline; implement incrementally",
                    probability=0.4,
                ),
            ],
            confidence_score=0.6,
            estimated_effort_hours=total_effort,
            affected_components=["ResearchGuild", "Memory", "Architecture"],
            evaluation_plan="Count gaps closed per sprint; measure principle coverage improvement",
            rollback_plan="Revert specific refactors via Forge rollback; architecture reverts to prior state",
            success_metrics=[
                f"{len(gaps)} gaps identified and triaged",
                "≥50% of critical/high gaps addressed within 2 sprints",
                "Principle coverage ratio improved by ≥20%",
            ],
            status=ProposalStatus.DRAFT,
        )

        logger.info("Generated proposal %s: %s", proposal.id, proposal.title)
        return proposal

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def store_gap(self, gap: GapAnalysis) -> bool:
        """Store a gap analysis in Animus memory.

        Args:
            gap: GapAnalysis to store.

        Returns:
            True if stored successfully.
        """
        if self.memory is None:
            logger.warning("Memory layer not available — gap not persisted")
            return False

        try:
            from animus.memory import MemoryType

            self.memory.remember(
                content=f"[{gap.severity.upper()}] {gap.principle_statement[:80]}: {gap.gap_description}",
                memory_type=MemoryType.SEMANTIC,
                tags=[
                    "architecture_citizen",
                    "research_guild",
                    "gap",
                    gap.principle_category,
                    gap.severity,
                ]
                + gap.affected_files,
                metadata=gap.to_dict(),
            )
            logger.info("Gap stored in memory: %s...", gap.principle_statement[:60])
            return True
        except Exception as e:
            logger.error("Failed to store gap: %s", e)
            return False

    def store_report(self, report: ArchitectureReport) -> bool:
        """Store an architecture report in Animus memory.

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
                tags=["architecture_citizen", "research_guild", "report"],
                metadata={
                    "total_gaps": report.total_gaps,
                    "critical_gaps": report.critical_gaps,
                    "high_gaps": report.high_gaps,
                    "principles_processed": report.principles_processed,
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
                tags=["architecture_citizen", "proposal", proposal.status.value],
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

    def list_stored_gaps(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recently identified gaps from memory.

        Args:
            limit: Maximum gaps to return.

        Returns:
            List of gap dicts.
        """
        if self.memory is None:
            return []

        try:
            from animus.memory import MemoryType

            results = self.memory.search(
                query="architecture_citizen gap research_guild",
                memory_type=MemoryType.SEMANTIC,
                limit=limit,
            )
            return [
                {
                    "id": r.get("id", "") if hasattr(r, "get") else getattr(r, "id", ""),
                    "content": r.get("content", "")
                    if hasattr(r, "get")
                    else getattr(r, "content", ""),
                    "metadata": r.get("metadata", {})
                    if hasattr(r, "get")
                    else getattr(r, "metadata", {}),
                }
                for r in results
            ]
        except Exception as e:
            logger.warning("list_stored_gaps failed: %s", e)
            return []

    def __repr__(self) -> str:
        return f"ArchitectureCitizen(gaps={len(self._gaps)})"
