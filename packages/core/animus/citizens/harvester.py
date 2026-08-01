"""Citizen 007 — The Harvester.

The first stage of the Research Guild pipeline.

Responsibilities:
- Collect raw external sources (repos, web pages, documents, memory corpus)
- Deduplicate against existing corpus
- Tag with source type, date, confidence, access method
- Feed raw source bundles to the Abstraction Citizen (next stage)

Never:
- Synthesize or summarize sources (that's the Abstraction Citizen)
- Modify code or memory directly
- Act on harvested findings without human approval

Instead:
    Observe → Collect → Tag → Deduplicate → Store → Human Approval → Abstraction
"""

from __future__ import annotations

import hashlib
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

logger = get_logger("citizens.harvester")


# ═══════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════


@dataclass
class HarvestedSource:
    """A single raw source collected by the Harvester.

    The Harvester does not synthesize — it only captures metadata
    and a content snippet sufficient for the Abstraction Citizen
    to decide whether to process the full source.
    """

    source_type: str  # "repo", "web_page", "document", "memory", "code_snippet"
    identifier: str  # URL, file path, memory ID, etc.
    title: str = ""
    content_snippet: str = ""  # Truncated raw content (first N chars)
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.5  # 0.0–1.0 reliability of source
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        """Return a stable hash for deduplication."""
        normalized = f"{self.source_type}|{self.identifier}|{self.title}".lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "identifier": self.identifier,
            "title": self.title,
            "content_snippet": self.content_snippet[:500],
            "tags": self.tags,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class HarvestReport:
    """Report produced by the Harvester after a collection run."""

    sources: list[HarvestedSource] = field(default_factory=list)
    duplicates_removed: int = 0
    errors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_collected(self) -> int:
        return len(self.sources)

    def summary(self) -> str:
        parts = [
            f"{self.total_collected} source(s) collected",
        ]
        if self.duplicates_removed:
            parts.append(f"{self.duplicates_removed} duplicate(s) removed")
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        return "; ".join(parts)


# ═══════════════════════════════════════════════════════════════════
# Harvester Citizen
# ═══════════════════════════════════════════════════════════════════


class HarvesterCitizen:
    """Citizen 007 — The Harvester.

    Collects raw sources from external repos, memory, and codebase
    for the Research Guild pipeline. NEVER synthesizes — only
    collects, tags, deduplicates, and stores.
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

        self._harvested: list[HarvestedSource] = []

    # ------------------------------------------------------------------
    # Observation methods (for autonomous loop compatibility)
    # ------------------------------------------------------------------

    def observe_codebase(self) -> list[dict[str, Any]]:
        """Scan codebase for TODO/FIXME/HACK comments as micro-sources.

        Returns:
            List of observation dicts compatible with autonomous loop.
        """
        findings: list[dict[str, Any]] = []
        if not self.codebase_path.exists():
            logger.warning("Codebase path does not exist: %s", self.codebase_path)
            return findings

        pattern = re.compile(
            r"#\s*(TODO|FIXME|HACK|XXX|NOTE|IDEA)\b.*?$", re.MULTILINE | re.IGNORECASE
        )
        for py_file in self.codebase_path.rglob("*.py"):
            if any(
                part.startswith(".") or part in ("__pycache__", "node_modules", "venv", ".venv")
                for part in py_file.parts
            ):
                continue
            try:
                text = py_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            matches = pattern.findall(text)
            if matches:
                rel = str(py_file.relative_to(self.codebase_path))
                counts: dict[str, int] = {}
                for m in matches:
                    counts[m.upper()] = counts.get(m.upper(), 0) + 1
                findings.append(
                    {
                        "source": "codebase",
                        "description": f"{len(matches)} tech-debt marker(s) in {rel}: {dict(counts)}",
                        "severity": "medium" if counts.get("HACK", 0) > 0 else "low",
                        "context": {
                            "file": rel,
                            "counts": counts,
                            "pattern_type": "tech_debt_comments",
                        },
                    }
                )

        # Also surface any Markdown documents as potential sources
        for md_file in self.codebase_path.rglob("*.md"):
            if any(part.startswith(".") for part in md_file.parts):
                continue
            try:
                text = md_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            rel = str(md_file.relative_to(self.codebase_path))
            if len(text) > 200:
                findings.append(
                    {
                        "source": "codebase",
                        "description": f"Document source: {rel} ({len(text)} chars)",
                        "severity": "info",
                        "context": {
                            "file": rel,
                            "word_count": len(text.split()),
                            "pattern_type": "document_source",
                        },
                    }
                )

        logger.info("Harvester observe_codebase: %d findings", len(findings))
        return findings

    def observe_memory(self) -> list[HarvestedSource]:
        """Search Animus memory for intelligence reports and proposals.

        Returns:
            List of HarvestedSource objects from memory corpus.
        """
        sources: list[HarvestedSource] = []
        if self.memory is None:
            logger.warning("Memory layer not available — observe_memory skipped")
            return sources

        try:
            from animus.memory import MemoryType

            # Search for intelligence reports
            results = self.memory.search(
                query="intelligence report proposal",
                memory_type=MemoryType.PROCEDURAL,
                limit=20,
            )
            for mem in results:
                content = (
                    mem.get("content", "") if hasattr(mem, "get") else getattr(mem, "content", "")
                )
                meta = (
                    mem.get("metadata", {}) if hasattr(mem, "get") else getattr(mem, "metadata", {})
                )
                if content:
                    sources.append(
                        HarvestedSource(
                            source_type="memory",
                            identifier=mem.get("id", "unknown")
                            if hasattr(mem, "get")
                            else getattr(mem, "id", "unknown"),
                            title="Memory: " + content[:60],
                            content_snippet=content[:500],
                            tags=["memory", "intelligence"],
                            confidence=0.6,
                            metadata=meta,
                        )
                    )

            # Search for semantic knowledge
            results = self.memory.search(
                query="architecture pattern mechanism",
                memory_type=MemoryType.SEMANTIC,
                limit=20,
            )
            for mem in results:
                content = (
                    mem.get("content", "") if hasattr(mem, "get") else getattr(mem, "content", "")
                )
                if content and not any(s.identifier == content[:80] for s in sources):
                    sources.append(
                        HarvestedSource(
                            source_type="memory",
                            identifier=mem.get("id", "unknown")
                            if hasattr(mem, "get")
                            else getattr(mem, "id", "unknown"),
                            title="Knowledge: " + content[:60],
                            content_snippet=content[:500],
                            tags=["memory", "knowledge"],
                            confidence=0.5,
                            metadata=mem.get("metadata", {})
                            if hasattr(mem, "get")
                            else getattr(mem, "metadata", {}),
                        )
                    )

        except Exception as e:
            logger.warning("observe_memory failed: %s", e)

        logger.info("Harvester observe_memory: %d sources", len(sources))
        return sources

    # ------------------------------------------------------------------
    # Harvest methods
    # ------------------------------------------------------------------

    def harvest_repository(self, target: str, depth: str = "quick") -> HarvestedSource | None:
        """Harvest an external GitHub repository using Lugh.

        Args:
            target: GitHub repo URL or username/repo.
            depth: "quick" (shallow clone) or "deep" (full clone).

        Returns:
            HarvestedSource or None if harvest failed.
        """
        try:
            from animus.lugh.repos import harvest_repo
        except ImportError:
            logger.warning("Lugh repos module not available — install with 'animus[lugh]'")
            return None

        try:
            result = harvest_repo(target=target, compare=True, depth=depth)
            return HarvestedSource(
                source_type="repo",
                identifier=target,
                title=f"Repo: {result.repo}",
                content_snippet=result.architecture[:500] if result.architecture else "",
                tags=["repo", "external", "architecture"],
                confidence=0.7 if result.score > 50 else 0.5,
                metadata={
                    "score": result.score,
                    "notable_patterns": result.notable_patterns,
                    "tools_worth_adopting": result.tools_worth_adopting,
                    "testing_approach": result.testing_approach,
                    "comparison": result.comparison,
                },
            )
        except Exception as e:
            logger.warning("harvest_repository failed for %s: %s", target, e)
            return None

    def harvest_watchlist(self, interval_hours: int = 0) -> HarvestReport:
        """Harvest all due repos from the competition watchlist.

        Args:
            interval_hours: Override scan interval (0 = use default 168h).

        Returns:
            HarvestReport with collected sources.
        """
        report = HarvestReport()
        try:
            from animus.lugh.watchlist import get_watchlist, run_watchlist_scan
        except ImportError:
            logger.warning("Lugh watchlist module not available")
            report.errors.append("Lugh watchlist not installed")
            return report

        try:
            entries = get_watchlist()
            if not entries:
                return report

            for entry in entries:
                target = entry.get("target", "")
                if not target:
                    continue
                source = self.harvest_repository(target, depth="quick")
                if source:
                    report.sources.append(source)

        except Exception as e:
            logger.warning("harvest_watchlist failed: %s", e)
            report.errors.append(str(e))

        return report

    def harvest_text(
        self, text: str, source_type: str = "text", identifier: str = ""
    ) -> HarvestedSource:
        """Harvest raw text as a source.

        Args:
            text: Raw text content.
            source_type: Type label (text, web_page, document, etc.).
            identifier: Optional identifier (URL, filename, etc.).

        Returns:
            HarvestedSource.
        """
        return HarvestedSource(
            source_type=source_type,
            identifier=identifier or f"text-{uuid.uuid4().hex[:8]}",
            title=text[:80].replace("\n", " "),
            content_snippet=text[:1000],
            tags=[source_type],
            confidence=0.5,
        )

    def harvest_file(self, file_path: Path | str) -> HarvestedSource | None:
        """Harvest a local file as a source.

        Args:
            file_path: Path to file.

        Returns:
            HarvestedSource or None if unreadable.
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning("File not found: %s", path)
            return None

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.warning("Failed to read %s: %s", path, e)
            return None

        return HarvestedSource(
            source_type="document",
            identifier=str(path),
            title=path.name,
            content_snippet=text[:1000],
            tags=["document", path.suffix.lstrip(".")],
            confidence=0.8,  # Local files are higher confidence
        )

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def deduplicate(self, sources: list[HarvestedSource]) -> list[HarvestedSource]:
        """Remove duplicate sources by content hash.

        Args:
            sources: List of sources to deduplicate.

        Returns:
            Deduplicated list (keeps first occurrence).
        """
        seen: set[str] = set()
        unique: list[HarvestedSource] = []
        for source in sources:
            h = source.content_hash()
            if h not in seen:
                seen.add(h)
                unique.append(source)
            else:
                logger.debug("Deduplicated source: %s", source.identifier)
        return unique

    # ------------------------------------------------------------------
    # Proposal generation
    # ------------------------------------------------------------------

    def generate_proposal(
        self, sources: list[HarvestedSource] | None = None
    ) -> ImprovementProposal | None:
        """Generate an improvement proposal from harvest findings.

        Args:
            sources: List of harvested sources. If None, runs
                observe_memory and observe_codebase automatically.

        Returns:
            Improvement proposal, or None if no actionable findings.
        """
        if sources is None:
            # Autonomous-loop path: gather from memory and codebase
            sources = []
            sources.extend(self.observe_memory())
            # Convert codebase observations to sources
            for obs in self.observe_codebase():
                sources.append(
                    HarvestedSource(
                        source_type="code_snippet",
                        identifier=obs["context"].get("file", "unknown"),
                        title=obs["description"][:80],
                        content_snippet=obs["description"],
                        tags=["codebase", obs["context"].get("pattern_type", "unknown")],
                        confidence=0.5,
                    )
                )
            sources = self.deduplicate(sources)

        if not sources:
            logger.info("No harvested sources — no proposal generated")
            return None

        # Categorize findings
        repo_sources = [s for s in sources if s.source_type == "repo"]
        doc_sources = [s for s in sources if s.source_type in ("document", "web_page")]
        memory_sources = [s for s in sources if s.source_type == "memory"]
        code_sources = [s for s in sources if s.source_type == "code_snippet"]

        # Build evidence
        evidence: list[EvidenceItem] = []
        for s in sources[:10]:  # Cap evidence for memory efficiency
            evidence.append(
                EvidenceItem(
                    source=s.source_type,
                    description=f"{s.title} ({s.identifier})",
                    data=s.to_dict(),
                    timestamp=s.timestamp,
                )
            )

        # Determine problem based on findings
        if repo_sources:
            problem = f"{len(repo_sources)} external repo(s) harvested with notable patterns"
            recommendation = (
                "Run the Abstraction Citizen on harvested repo data to extract "
                "transferable mechanisms. Consider adding high-value repos to the "
                "watchlist for continuous monitoring."
            )
            affected = ["ResearchGuild", "Lugh"]
        elif doc_sources:
            problem = f"{len(doc_sources)} document(s) harvested awaiting abstraction"
            recommendation = (
                "Feed harvested documents into the Abstraction Citizen pipeline "
                "to distill architectural principles."
            )
            affected = ["ResearchGuild"]
        elif memory_sources:
            problem = f"{len(memory_sources)} untapped memory source(s) identified"
            recommendation = (
                "Memory corpus contains intelligence reports and knowledge entries "
                "that have not been processed by the Research Guild. Run the "
                "Harvester periodically to feed the pipeline."
            )
            affected = ["ResearchGuild", "Memory"]
        elif code_sources:
            problem = f"{len(code_sources)} codebase marker(s) found as potential research inputs"
            recommendation = (
                "Codebase contains TODOs, FIXMEs, and design notes that could inform "
                "the Research Guild's architectural investigations."
            )
            affected = ["ResearchGuild", "Mind"]
        else:
            problem = f"{len(sources)} harvested source(s) require processing"
            recommendation = "Continue harvesting and queue for Abstraction Citizen processing."
            affected = ["ResearchGuild"]

        proposal = ImprovementProposal(
            id=f"HARV-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}",
            title=f"Harvester: {problem[:60]}",
            problem=problem,
            evidence=evidence,
            root_cause="Research Guild pipeline needs a continuous feed of raw external and internal sources",
            recommendation=recommendation,
            alternatives_considered=[
                "Manual research (human-only, slower)",
                "Skip Research Guild and feed Forge directly (noisy, inefficient)",
            ],
            expected_benefits="Richer architectural proposals through systematic external observation",
            potential_risks=[
                RiskAssessment(
                    description="Harvesting may produce too many low-value sources",
                    severity="low",
                    mitigation="Confidence threshold and deduplication filter noise",
                    probability=0.3,
                ),
                RiskAssessment(
                    description="External repos may change, making stored sources stale",
                    severity="low",
                    mitigation="Re-harvest watchlist on periodic cadence (weekly)",
                    probability=0.4,
                ),
            ],
            confidence_score=0.6,
            estimated_effort_hours=1.0,
            affected_components=affected,
            evaluation_plan="Count sources processed by Abstraction Citizen; measure proposal quality delta",
            rollback_plan="Stop harvesting; pipeline reverts to manual research only",
            success_metrics=[
                f"{len(sources)} sources fed to Abstraction Citizen",
                "≥1 pattern extracted per repo",
                "Zero duplicate sources in pipeline",
            ],
            status=ProposalStatus.DRAFT,
        )

        logger.info("Generated proposal %s: %s", proposal.id, proposal.title)
        return proposal

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def store_source(self, source: HarvestedSource) -> bool:
        """Store a harvested source in Animus memory.

        Args:
            source: Source to store.

        Returns:
            True if stored successfully.
        """
        if self.memory is None:
            logger.warning("Memory layer not available — source not persisted")
            return False

        try:
            from animus.memory import MemoryType

            self.memory.remember(
                content=source.content_snippet[:500],
                memory_type=MemoryType.SEMANTIC,
                tags=["harvester", "research_guild", source.source_type] + source.tags,
                metadata=source.to_dict(),
            )
            logger.info("Source %s stored in memory", source.identifier)
            return True
        except Exception as e:
            logger.error("Failed to store source: %s", e)
            return False

    def store_report(self, report: HarvestReport) -> bool:
        """Store a harvest report in Animus memory.

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
                tags=["harvester", "research_guild", "report"],
                metadata={
                    "total_collected": report.total_collected,
                    "duplicates_removed": report.duplicates_removed,
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
                tags=["harvester", "proposal", proposal.status.value],
                metadata=proposal.to_dict(),
            )
            logger.info("Proposal %s stored in memory", proposal.id)
            return True
        except Exception as e:
            logger.error("Failed to store proposal: %s", e)
            return False

    def _source_from_harvest_result(
        self,
        target: str,
        harvest_result: Any,
    ) -> HarvestedSource | None:
        """Convert a Lugh HarvestResult into a HarvestedSource.

        This bridge method is called by the Research Guild Orchestrator
        after ``harvest_repo()`` returns so the Harvester can store the
        result in Animus memory as a first-class source.
        """
        if harvest_result is None:
            return None
        try:
            return HarvestedSource(
                source_type="repo",
                identifier=target,
                title=f"Repo: {harvest_result.repo}",
                content_snippet=(
                    f"Architecture: {harvest_result.architecture}\n"
                    f"Testing: {harvest_result.testing_approach}\n"
                    f"Patterns: {', '.join(harvest_result.notable_patterns[:5])}\n"
                    f"Novel tools: {', '.join(harvest_result.tools_worth_adopting[:10])}"
                )[:500],
                tags=["repo", "external", "architecture"],
                confidence=0.7 if harvest_result.score > 50 else 0.5,
                metadata=harvest_result.to_dict(),
            )
        except Exception as e:
            logger.warning("Failed to convert harvest result for %s: %s", target, e)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def list_stored_sources(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recently harvested sources from memory.

        Args:
            limit: Maximum sources to return.

        Returns:
            List of source dicts.
        """
        if self.memory is None:
            return []

        try:
            from animus.memory import MemoryType

            results = self.memory.search(
                query="harvester research_guild",
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
            logger.warning("list_stored_sources failed: %s", e)
            return []

    def __repr__(self) -> str:
        return f"HarvesterCitizen(sources={len(self._harvested)})"
