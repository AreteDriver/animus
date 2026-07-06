"""Citizen 001 — The Architect.

The permanent "chief architect" of Animus.

Responsibilities:
- Observe system behavior
- Analyze conversations
- Review Forge output
- Monitor evaluations
- Detect technical debt, architectural bottlenecks, user friction
- Research relevant engineering advances
- Produce evidence-backed improvement proposals

Never:
- Modify code directly
- Change memory autonomously
- Merge code
- Deploy changes

Instead:
    Observe → Analyze → Produce Proposal → Human Approval → Forge → Evidence → Merge
"""

from __future__ import annotations

import json
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
    from animus_forge.self_improve.analyzer import CodebaseAnalyzer, ImprovementSuggestion

logger = get_logger("citizens.architect")


@dataclass
class Observation:
    """A single observation from system monitoring."""

    source: str  # "codebase", "conversation", "evaluation", "forge_output"
    description: str
    severity: str = "info"  # "critical", "high", "medium", "low", "info"
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AnalysisReport:
    """Report produced by the Architect after observation and analysis."""

    observations: list[Observation] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    technical_debt_items: list[str] = field(default_factory=list)
    friction_points: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class ArchitectCitizen:
    """Citizen 001 — The Architect.

    Continuously evaluates Animus and proposes improvements.
    This citizen NEVER modifies code, memory, or systems directly.
    It only observes, analyzes, and produces proposals.
    """

    def __init__(
        self,
        codebase_path: Path | str = "~/projects/animus",
        memory_layer: MemoryLayer | None = None,
        conversation_log_dir: Path | str | None = None,
        evidence_dir: Path | str | None = None,
    ):
        self.codebase_path = Path(codebase_path).expanduser()
        self.memory = memory_layer
        self.conversation_log_dir = (
            Path(conversation_log_dir).expanduser() if conversation_log_dir else None
        )
        self.evidence_dir = Path(evidence_dir).expanduser() if evidence_dir else None

        if self.evidence_dir:
            self.evidence_dir.mkdir(parents=True, exist_ok=True)

        self._analyzer: Any = None
        self._observations: list[Observation] = []

    def _get_analyzer(self) -> Any:
        """Lazy-load the Forge CodebaseAnalyzer.

        Returns None if Forge is not installed.
        """
        if self._analyzer is not None:
            return self._analyzer

        try:
            from animus_forge.self_improve.analyzer import CodebaseAnalyzer

            self._analyzer = CodebaseAnalyzer(codebase_path=self.codebase_path)
            logger.info("Forge CodebaseAnalyzer loaded for Architect")
            return self._analyzer
        except ImportError:
            logger.warning(
                "Forge not installed. Architect will use heuristic analysis only. "
                "Install with: pip install -e packages/forge/"
            )
            return None

    # ------------------------------------------------------------------
    # Observation methods (read-only, never modify)
    # ------------------------------------------------------------------

    def observe_codebase(
        self,
        focus_paths: list[str] | None = None,
        categories: list[str] | None = None,
    ) -> list[Observation]:
        """Observe the codebase for improvement opportunities.

        Args:
            focus_paths: Specific paths to analyze.
            categories: Categories to focus on.

        Returns:
            List of observations.
        """
        observations: list[Observation] = []
        analyzer = self._get_analyzer()

        if analyzer is None:
            observations.append(
                Observation(
                    source="codebase",
                    description="Forge analyzer unavailable — cannot perform deep codebase analysis",
                    severity="high",
                )
            )
            return observations

        try:
            result = analyzer.analyze(focus_paths=focus_paths)
            for suggestion in result.suggestions:
                observations.append(
                    Observation(
                        source="codebase",
                        description=f"{suggestion.category}: {suggestion.title} — {suggestion.description}",
                        severity=self._map_priority_to_severity(suggestion.priority),
                        context={
                            "category": suggestion.category,
                            "affected_files": suggestion.affected_files,
                            "estimated_lines": suggestion.estimated_lines,
                            "reasoning": suggestion.reasoning,
                        },
                    )
                )
        except Exception as e:
            logger.error(f"Codebase analysis failed: {e}")
            observations.append(
                Observation(
                    source="codebase",
                    description=f"Analysis error: {e}",
                    severity="high",
                )
            )

        self._observations.extend(observations)
        return observations

    def observe_conversations(self, limit: int = 100) -> list[Observation]:
        """Observe conversation logs for repeated patterns and friction.

        Args:
            limit: Maximum number of recent conversations to analyze.

        Returns:
            List of observations.
        """
        observations: list[Observation] = []

        if not self.conversation_log_dir or not self.conversation_log_dir.exists():
            observations.append(
                Observation(
                    source="conversation",
                    description="Conversation log directory not configured or not found",
                    severity="medium",
                )
            )
            self._observations.extend(observations)
            return observations

        # Simple heuristic: look for repeated prompt patterns
        prompt_counts: dict[str, int] = {}
        for log_file in sorted(self.conversation_log_dir.glob("*.jsonl"))[-limit:]:
            try:
                for line in log_file.read_text().splitlines():
                    entry = json.loads(line)
                    prompt = entry.get("prompt", "").strip().lower()
                    if len(prompt) > 10:
                        prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1
            except Exception:
                continue

        # Report repeated prompts as potential friction
        for prompt, count in sorted(prompt_counts.items(), key=lambda x: -x[1]):
            if count >= 3:
                observations.append(
                    Observation(
                        source="conversation",
                        description=f"Repeated prompt detected ({count}×): {prompt[:80]}...",
                        severity="medium" if count >= 5 else "low",
                        context={"count": count, "prompt_prefix": prompt[:100]},
                    )
                )

        self._observations.extend(observations)
        return observations

    def observe_evaluations(self, eval_dir: Path | str | None = None) -> list[Observation]:
        """Observe evaluation results for trends and regressions.

        Args:
            eval_dir: Directory containing evaluation artifacts.

        Returns:
            List of observations.
        """
        observations: list[Observation] = []
        target_dir = Path(eval_dir) if eval_dir else self.codebase_path / "evidence"

        if not target_dir.exists():
            observations.append(
                Observation(
                    source="evaluation",
                    description=f"Evaluation directory not found: {target_dir}",
                    severity="medium",
                )
            )
            self._observations.extend(observations)
            return observations

        # Look for recent eval results
        for eval_file in sorted(target_dir.glob("eval_*.json"))[-10:]:
            try:
                data = json.loads(eval_file.read_text())
                score = data.get("score", 0)
                if score < 0.7:
                    observations.append(
                        Observation(
                            source="evaluation",
                            description=f"Low eval score in {eval_file.name}: {score:.2f}",
                            severity="high",
                            context={"score": score, "file": str(eval_file)},
                        )
                    )
            except Exception:
                continue

        self._observations.extend(observations)
        return observations

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def analyze(self) -> AnalysisReport:
        """Analyze all observations and produce a report.

        Returns:
            Analysis report with findings and recommendations.
        """
        if not self._observations:
            logger.info("No observations to analyze — running observation sweep")
            self.observe_codebase()
            self.observe_conversations()
            self.observe_evaluations()

        report = AnalysisReport()
        report.observations = list(self._observations)

        # Categorize observations
        for obs in self._observations:
            if obs.source == "codebase" and obs.severity in ("high", "critical"):
                report.technical_debt_items.append(obs.description)
            elif obs.source == "conversation":
                report.friction_points.append(obs.description)
            elif obs.severity in ("high", "critical"):
                report.findings.append(obs.description)

        # Generate high-level recommendations
        if report.technical_debt_items:
            report.recommendations.append(
                f"Address {len(report.technical_debt_items)} technical debt items identified in codebase"
            )
        if report.friction_points:
            report.recommendations.append(
                f"Reduce {len(report.friction_points)} conversation friction points via workflow improvements"
            )

        # Clear observations after analysis
        self._observations.clear()
        return report

    # ------------------------------------------------------------------
    # Proposal generation (the core output of the Architect)
    # ------------------------------------------------------------------

    def generate_proposal(self, report: AnalysisReport | None = None) -> ImprovementProposal | None:
        """Generate an improvement proposal from analysis.

        Args:
            report: Analysis report to base proposal on. If None, runs analysis first.

        Returns:
            Improvement proposal, or None if no actionable findings.
        """
        if report is None:
            report = self.analyze()

        if not report.findings and not report.technical_debt_items and not report.friction_points:
            logger.info("No actionable findings — no proposal generated")
            return None

        # Build evidence items
        evidence: list[EvidenceItem] = []
        for obs in report.observations:
            evidence.append(
                EvidenceItem(
                    source=obs.source,
                    description=obs.description,
                    data=obs.context,
                    timestamp=obs.timestamp,
                )
            )

        # Determine highest-severity problem
        if report.technical_debt_items:
            problem = f"Technical debt: {report.technical_debt_items[0]}"
            recommendation = "Address identified technical debt through structured refactoring"
            affected = ["Factory", "Kernel"]
        elif report.friction_points:
            problem = f"User friction: {report.friction_points[0]}"
            recommendation = "Reduce conversation friction via workflow shortcuts or context handling"
            affected = ["Mind", "Society"]
        else:
            problem = report.findings[0] if report.findings else "General improvement opportunity"
            recommendation = "Investigate and remediate identified issue"
            affected = ["Mind"]

        # Build risks
        risks = [
            RiskAssessment(
                description="Implementation may introduce regressions",
                severity="medium",
                mitigation="Full test suite + eval checkpoints before merge",
                probability=0.3,
            ),
            RiskAssessment(
                description="Effort estimate may be inaccurate",
                severity="low",
                mitigation="Time-box initial implementation to 4 hours",
                probability=0.5,
            ),
        ]

        proposal = ImprovementProposal(
            id=f"ADL-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}",
            title=f"Architect Proposal: {problem[:60]}",
            problem=problem,
            evidence=evidence,
            root_cause="Identified through systematic observation and analysis",
            recommendation=recommendation,
            alternatives_considered=["Status quo (no change)", "Manual remediation (human-only)"],
            expected_benefits="Reduced technical debt and/or improved user experience",
            potential_risks=risks,
            confidence_score=0.6,
            estimated_effort_hours=4.0,
            affected_components=affected,
            evaluation_plan="Run full test suite + benchmark comparison + manual verification",
            rollback_plan="Revert to previous commit via git revert",
            success_metrics=["Tests pass", "Benchmarks stable or improved", "No new regressions"],
            status=ProposalStatus.DRAFT,
        )

        logger.info(f"Generated proposal {proposal.id}: {proposal.title}")
        return proposal

    # ------------------------------------------------------------------
    # Persistence (proposals stored in memory, never modified directly)
    # ------------------------------------------------------------------

    def store_proposal(self, proposal: ImprovementProposal) -> bool:
        """Store a proposal in Animus memory for review.

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

            self.memory.store(
                content=f"{proposal.title}\n\n{proposal.problem}\n\nRecommendation: {proposal.recommendation}",
                memory_type=MemoryType.PROCEDURAL,
                tags=["architect", "proposal", proposal.status.value],
                metadata=proposal.to_dict(),
            )
            logger.info(f"Proposal {proposal.id} stored in memory")
            return True
        except Exception as e:
            logger.error(f"Failed to store proposal: {e}")
            return False

    def list_pending_proposals(self) -> list[ImprovementProposal]:
        """List all proposals awaiting human review.

        Returns:
            List of pending proposals.
        """
        if self.memory is None:
            return []

        try:
            from animus.memory import MemoryType

            results = self.memory.search(
                query="architect proposal submitted",
                memory_type=MemoryType.PROCEDURAL,
                limit=50,
            )
            proposals = []
            for mem in results:
                meta = mem.get("metadata", {})
                if meta.get("status") in ("draft", "submitted", "under_review"):
                    proposals.append(ImprovementProposal.from_dict(meta))
            return proposals
        except Exception as e:
            logger.error(f"Failed to list proposals: {e}")
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _map_priority_to_severity(priority: int) -> str:
        """Map analyzer priority (1=highest, 5=lowest) to severity."""
        mapping = {1: "critical", 2: "high", 3: "medium", 4: "low", 5: "info"}
        return mapping.get(priority, "medium")

    def __repr__(self) -> str:
        return f"ArchitectCitizen(codebase={self.codebase_path}, observations={len(self._observations)})"
