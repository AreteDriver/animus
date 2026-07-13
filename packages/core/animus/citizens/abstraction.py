"""Citizen 008 — The Abstraction Citizen.

The second stage of the Research Guild pipeline.

Responsibilities:
- Read HarvestedSource objects from memory (produced by Harvester)
- Extract transferable mechanisms via keyword/pattern matching
- Strip implementation details (tech names, frameworks, languages)
- Produce MechanismCard objects for the Pattern Citizen (next stage)

Never:
- Synthesize across multiple sources (that's the Pattern Citizen)
- Modify harvested sources directly
- Act on findings without human approval

Instead:
    Read Source → Extract Mechanism → Strip Details → Card → Human Approval → Pattern
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

logger = get_logger("citizens.abstraction")


# ═══════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════


@dataclass
class MechanismCard:
    """A distilled mechanism extracted from a harvested source.

    The Abstraction Citizen strips implementation details and keeps
    only the transferable idea — the mechanism that could apply
    across technologies and contexts.
    """

    name: str  # e.g., "caching layer"
    description: str  # e.g., "Separate read-heavy data from computation"
    source_provenance: list[str] = field(default_factory=list)  # Source IDs
    confidence: float = 0.5  # 0.0–1.0
    implementation_stripped: str = ""  # Original text with tech names removed
    category: str = ""  # e.g., "performance", "reliability", "observability"
    tags: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "source_provenance": self.source_provenance,
            "confidence": self.confidence,
            "implementation_stripped": self.implementation_stripped,
            "category": self.category,
            "tags": self.tags,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AbstractionReport:
    """Report produced by the Abstraction Citizen after processing sources."""

    mechanisms: list[MechanismCard] = field(default_factory=list)
    sources_processed: int = 0
    sources_with_no_mechanism: int = 0
    errors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_extracted(self) -> int:
        return len(self.mechanisms)

    def summary(self) -> str:
        parts = [
            f"{self.total_extracted} mechanism(s) extracted from {self.sources_processed} source(s)",
        ]
        if self.sources_with_no_mechanism:
            parts.append(f"{self.sources_with_no_mechanism} source(s) with no recognizable mechanism")
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        return "; ".join(parts)


# ═══════════════════════════════════════════════════════════════════
# Mechanism extraction rules
# ═══════════════════════════════════════════════════════════════════

# Mapping of regex patterns → (mechanism_name, category, description_template)
_MECHANISM_RULES: list[tuple[re.Pattern, str, str, str]] = [
    # Caching
    (re.compile(r"\b(cach(?:e|ing)|memoiz(?:e|ation)|buffer|warm(?: up)?)\b", re.IGNORECASE),
     "caching layer", "performance",
     "Separate read-heavy or computed data from its source to reduce latency"),
    # Queue / Async
    (re.compile(r"\b(queue|message broker|event bus|pub[/-]?sub|async|message queue|stream processing)\b", re.IGNORECASE),
     "asynchronous communication", "reliability",
     "Decouple producers and consumers via an intermediary message channel"),
    # Retry / Resilience
    (re.compile(r"\b(retry|backoff|hedge request|circuit breaker|fail fast|graceful degrad|bulkhead|timeout)\b", re.IGNORECASE),
     "fault tolerance", "reliability",
     "Handle failures gracefully through retries, timeouts, or isolation boundaries"),
    # Observability
    (re.compile(r"\b(trace|metric|log|monitor|observ|telemetry|span|dashboard|alert)\b", re.IGNORECASE),
     "observability", "operations",
     "Expose internal system state through traces, metrics, and structured logs"),
    # State Separation
    (re.compile(r"\b(stateless|external state|state machine|separation of concerns|immutable|pure function)\b", re.IGNORECASE),
     "state externalization", "architecture",
     "Separate state from computation to enable portability and fault tolerance"),
    # Authentication / Identity
    (re.compile(r"\b(auth|oauth|jwt|token|session|identity|sso|mfa|rbac|permission)\b", re.IGNORECASE),
     "identity verification", "security",
     "Verify and authorize actors before granting access to resources"),
    # Encryption / Protection
    (re.compile(r"\b(encrypt|tls|ssl|cipher|hash|sign|certificate|vault|secret|key rotation)\b", re.IGNORECASE),
     "data protection", "security",
     "Protect data in transit and at rest through cryptographic controls"),
    # Testing
    (re.compile(r"\b(mock|stub|fixture|integration test|unit test|e2e test|test coverage|testability)\b", re.IGNORECASE),
     "testability", "quality",
     "Design systems so that components can be verified in isolation and in composition"),
    # Dependency Injection / Inversion
    (re.compile(r"\b(dependency injection|inversion of control|ioc|factory|builder|provider|composition root)\b", re.IGNORECASE),
     "dependency inversion", "architecture",
     "Depend on abstractions rather than concrete implementations to reduce coupling"),
    # Validation / Schema
    (re.compile(r"\b(validation|schema|contract|type safety|strict typing|assert|invariant)\b", re.IGNORECASE),
     "contract enforcement", "quality",
     "Enforce boundaries and assumptions through explicit contracts and validation"),
    # Pagination / Streaming
    (re.compile(r"\b(paginat|cursor|offset|limit|stream|chunk|batch|backpressure)\b", re.IGNORECASE),
     "bounded retrieval", "performance",
     "Process large datasets in bounded chunks to control memory and latency"),
    # Idempotency
    (re.compile(r"\b(idempoten|exactly.once|at.least.once|dedup|duplicate detection)\b", re.IGNORECASE),
     "idempotent processing", "reliability",
     "Ensure repeated operations produce the same outcome without side effects"),
    # Feature Flags
    (re.compile(r"\b(feature flag|toggle|launch darkly|canary|gradual rollout|a/b test)\b", re.IGNORECASE),
     "progressive rollout", "deployment",
     "Decouple release from deployment through runtime-configurable behavior switches"),
    # Rate Limiting
    (re.compile(r"\b(rate limit|throttl|quota|burst|token bucket|leaky bucket)\b", re.IGNORECASE),
     "flow control", "reliability",
     "Protect downstream services by constraining request volume and burstiness"),
]

# Technology names to strip from text (implementation details)
_TECH_STRIP_LIST = [
    "redis", "memcached", "nginx", "haproxy", "apache", "traefik",
    "kubernetes", "docker", "podman", "terraform", "pulumi", "ansible",
    "aws", "gcp", "azure", "heroku", "vercel", "netlify", "cloudflare",
    "postgres", "mysql", "mongodb", "sqlite", "dynamodb", "firebase",
    "react", "vue", "angular", "svelte", "next.js", "nuxt", "gatsby",
    "node.js", "deno", "bun", "python", "go", "rust", "java", "kotlin",
    "spring", "django", "flask", "fastapi", "express", "rails",
    "elasticsearch", "prometheus", "grafana", "jaeger", "zipkin",
    "kafka", "rabbitmq", "sqs", "pub/sub", "zeromq", "nats",
    "github actions", "gitlab ci", "jenkins", "circleci", "travis",
    "webpack", "vite", "rollup", "esbuild", "babel", "typescript",
]

_STRIP_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _TECH_STRIP_LIST) + r")\b",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════
# Abstraction Citizen
# ═══════════════════════════════════════════════════════════════════


class AbstractionCitizen:
    """Citizen 008 — The Abstraction Citizen.

    Reads harvested sources, extracts transferable mechanisms,
    strips implementation details, and produces MechanismCards.
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

        self._mechanisms: list[MechanismCard] = []

    # ------------------------------------------------------------------
    # Observation methods (for autonomous loop compatibility)
    # ------------------------------------------------------------------

    def observe_codebase(self) -> list[dict[str, Any]]:
        """Scan codebase for design-pattern-like comments as micro-sources.

        Returns:
            List of observation dicts compatible with autonomous loop.
        """
        findings: list[dict[str, Any]] = []
        if not self.codebase_path.exists():
            logger.warning("Codebase path does not exist: %s", self.codebase_path)
            return findings

        for py_file in self.codebase_path.rglob("*.py"):
            if any(part.startswith(".") or part in ("__pycache__", "node_modules", "venv", ".venv") for part in py_file.parts):
                continue
            try:
                text = py_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            rel = str(py_file.relative_to(self.codebase_path))
            stripped = self.strip_implementation(text)
            mechanisms = self._extract_from_text(stripped)
            if mechanisms:
                findings.append({
                    "source": "codebase",
                    "description": f"{len(mechanisms)} mechanism(s) in {rel}: {', '.join(m.name for m in mechanisms)}",
                    "severity": "low",
                    "context": {
                        "file": rel,
                        "mechanisms": [m.name for m in mechanisms],
                        "pattern_type": "mechanism_extraction",
                    },
                })

        logger.info("Abstraction observe_codebase: %d findings", len(findings))
        return findings

    def observe_harvested_sources(self) -> list[dict[str, Any]]:
        """Read harvested sources from memory and convert to dict observations.

        Returns:
            List of observation dicts compatible with autonomous loop.
        """
        findings: list[dict[str, Any]] = []
        if self.memory is None:
            logger.warning("Memory layer not available — observe_harvested_sources skipped")
            return findings

        try:
            from animus.memory import MemoryType

            results = self.memory.search(
                query="harvester research_guild",
                memory_type=MemoryType.SEMANTIC,
                limit=30,
            )
            for mem in results:
                content = mem.get("content", "") if hasattr(mem, "get") else getattr(mem, "content", "")
                meta = mem.get("metadata", {}) if hasattr(mem, "get") else getattr(mem, "metadata", {})
                identifier = meta.get("identifier", "") if isinstance(meta, dict) else ""
                if content or identifier:
                    findings.append({
                        "source": "memory",
                        "description": f"Harvested source: {identifier or content[:60]}",
                        "severity": "info",
                        "context": {
                            "identifier": identifier,
                            "source_type": meta.get("source_type", "unknown") if isinstance(meta, dict) else "unknown",
                            "content": content[:200],
                            "pattern_type": "harvested_source",
                        },
                    })

        except Exception as e:
            logger.warning("observe_harvested_sources failed: %s", e)

        logger.info("Abstraction observe_harvested_sources: %d findings", len(findings))
        return findings

    # ------------------------------------------------------------------
    # Mechanism extraction
    # ------------------------------------------------------------------

    def extract_mechanisms(self, source_text: str, source_id: str = "") -> list[MechanismCard]:
        """Extract mechanisms from source text.

        Args:
            source_text: Raw or harvested text to analyze.
            source_id: Identifier of the source (for provenance).

        Returns:
            List of MechanismCard objects.
        """
        stripped = self.strip_implementation(source_text)
        return self._extract_from_text(stripped, source_id)

    @staticmethod
    def strip_implementation(text: str) -> str:
        """Remove technology-specific implementation details from text.

        Args:
            text: Source text.

        Returns:
            Text with tech names replaced by placeholders.
        """
        if not text:
            return ""

        def _replacer(match: re.Match) -> str:
            return "[TECH]"

        return _STRIP_PATTERN.sub(_replacer, text)

    def _extract_from_text(self, text: str, source_id: str = "") -> list[MechanismCard]:
        """Internal: apply mechanism rules to text."""
        cards: list[MechanismCard] = []
        if not text:
            return cards

        seen: set[str] = set()
        for pattern, name, category, description in _MECHANISM_RULES:
            if pattern.search(text) and name not in seen:
                seen.add(name)
                cards.append(
                    MechanismCard(
                        name=name,
                        description=description,
                        source_provenance=[source_id] if source_id else [],
                        confidence=0.6,
                        implementation_stripped=text[:500],
                        category=category,
                        tags=[category, "mechanism"],
                    )
                )

        return cards

    # ------------------------------------------------------------------
    # Proposal generation
    # ------------------------------------------------------------------

    def generate_proposal(self, mechanisms: list[MechanismCard] | None = None) -> ImprovementProposal | None:
        """Generate an improvement proposal from extracted mechanisms.

        Args:
            mechanisms: List of MechanismCards. If None, runs
                observe_codebase and observe_harvested_sources automatically.

        Returns:
            Improvement proposal, or None if no actionable findings.
        """
        if mechanisms is None:
            # Autonomous-loop path: gather from codebase and memory
            mechanisms = []
            for obs in self.observe_codebase():
                mechs = obs["context"].get("mechanisms", [])
                for m_name in mechs:
                    mechanisms.append(
                        MechanismCard(
                            name=m_name,
                            description=f"Extracted from codebase: {m_name}",
                            source_provenance=[obs["context"].get("file", "unknown")],
                            confidence=0.5,
                            category="architecture",
                            tags=["codebase", "mechanism"],
                        )
                    )

            for obs in self.observe_harvested_sources():
                content = obs["context"].get("content", "")
                sid = obs["context"].get("identifier", "")
                if content:
                    mechs = self.extract_mechanisms(content, sid)
                    mechanisms.extend(mechs)

        if not mechanisms:
            logger.info("No mechanisms extracted — no proposal generated")
            return None

        # Categorize
        by_category: dict[str, list[MechanismCard]] = {}
        for m in mechanisms:
            by_category.setdefault(m.category, []).append(m)

        top_category = max(by_category, key=lambda k: len(by_category[k]))
        top_count = len(by_category[top_category])

        evidence = [
            EvidenceItem(
                source="abstraction",
                description=f"{m.name}: {m.description}",
                data=m.to_dict(),
                timestamp=m.timestamp,
            )
            for m in mechanisms[:10]
        ]

        proposal = ImprovementProposal(
            id=f"ABST-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}",
            title=f"Abstraction: {top_count} {top_category} mechanism(s) extracted",
            problem=f"{len(mechanisms)} mechanism(s) extracted from sources but not yet synthesized into patterns",
            evidence=evidence,
            root_cause="Research Guild pipeline needs continuous mechanism extraction before pattern discovery",
            recommendation=(
                "Feed extracted mechanism cards into the Pattern Citizen to find "
                "recurring structures across sources. Prioritize mechanisms with "
                f"highest confidence in category '{top_category}'."
            ),
            alternatives_considered=[
                "Skip abstraction and feed raw sources to Pattern (noisy)",
                "Manual mechanism extraction (human-only, slower)",
            ],
            expected_benefits="Cleaner signal-to-noise ratio for downstream pattern discovery",
            potential_risks=[
                RiskAssessment(
                    description="Heuristic extraction may miss nuanced mechanisms",
                    severity="low",
                    mitigation="Periodic human review of extracted cards; expand rule set over time",
                    probability=0.3,
                ),
            ],
            confidence_score=0.6,
            estimated_effort_hours=1.0,
            affected_components=["ResearchGuild", "Memory"],
            evaluation_plan="Count mechanisms fed to Pattern Citizen; measure pattern yield",
            rollback_plan="Stop abstraction; pipeline reverts to raw-source feeding",
            success_metrics=[
                f"{len(mechanisms)} mechanism cards produced",
                "Zero cards with implementation details remaining",
                "≥1 pattern per 5 mechanism cards",
            ],
            status=ProposalStatus.DRAFT,
        )

        logger.info("Generated proposal %s: %s", proposal.id, proposal.title)
        return proposal

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def store_mechanism(self, card: MechanismCard) -> bool:
        """Store a mechanism card in Animus memory.

        Args:
            card: MechanismCard to store.

        Returns:
            True if stored successfully.
        """
        if self.memory is None:
            logger.warning("Memory layer not available — mechanism not persisted")
            return False

        try:
            from animus.memory import MemoryType

            self.memory.remember(
                content=f"{card.name}: {card.description}",
                memory_type=MemoryType.SEMANTIC,
                tags=["abstraction", "research_guild", "mechanism", card.category] + card.tags,
                metadata=card.to_dict(),
            )
            logger.info("Mechanism '%s' stored in memory", card.name)
            return True
        except Exception as e:
            logger.error("Failed to store mechanism: %s", e)
            return False

    def store_report(self, report: AbstractionReport) -> bool:
        """Store an abstraction report in Animus memory.

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
                tags=["abstraction", "research_guild", "report"],
                metadata={
                    "total_extracted": report.total_extracted,
                    "sources_processed": report.sources_processed,
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
                tags=["abstraction", "proposal", proposal.status.value],
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

    def list_stored_mechanisms(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recently extracted mechanisms from memory.

        Args:
            limit: Maximum mechanisms to return.

        Returns:
            List of mechanism dicts.
        """
        if self.memory is None:
            return []

        try:
            from animus.memory import MemoryType

            results = self.memory.search(
                query="abstraction mechanism research_guild",
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
            logger.warning("list_stored_mechanisms failed: %s", e)
            return []

    def __repr__(self) -> str:
        return f"AbstractionCitizen(mechanisms={len(self._mechanisms)})"
