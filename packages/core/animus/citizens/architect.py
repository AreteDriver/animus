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

import ast
import json
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
from animus.memory.types import MemoryType

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
        focus_paths: list[str] | None = None,
    ):
        self.codebase_path = Path(codebase_path).expanduser()
        self.memory = memory_layer
        self.conversation_log_dir = (
            Path(conversation_log_dir).expanduser() if conversation_log_dir else None
        )
        self.evidence_dir = Path(evidence_dir).expanduser() if evidence_dir else None
        self.focus_paths = focus_paths or []

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
        # Use constructor focus_paths as default if none provided
        if focus_paths is None and self.focus_paths:
            focus_paths = self.focus_paths
        """Observe the codebase for improvement opportunities.

        Args:
            focus_paths: Specific paths to analyze.
            categories: Categories to focus on.

        Returns:
            List of observations.
        """
        observations: list[Observation] = []
        analyzer = self._get_analyzer()

        # Expand directory paths to glob patterns for Forge
        expanded_paths = self._expand_focus_paths(focus_paths)

        if analyzer is not None:
            try:
                result = analyzer.analyze(focus_paths=expanded_paths)
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
                logger.error(f"Forge analysis failed: {e}")
                # Fall through to heuristics
        else:
            logger.info("Forge analyzer unavailable — falling back to heuristics")

        # If Forge produced nothing (or is absent), run lightweight heuristics
        if not observations:
            observations.extend(self._observe_heuristics(focus_paths))

        # If memory layer is available, enrich with indexed code observations
        if self.memory is not None:
            indexed_obs = self._observe_indexed_code_memory(focus_paths)
            observations.extend(indexed_obs)

        # Deduplicate noisy Forge suggestions (e.g. "long function" flooding)
        observations = self._deduplicate_observations(observations, max_per_pattern=5)

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
            # No conversation logs configured yet — not an actionable finding.
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

        Queries both the Forge eval store and local evidence files.

        Args:
            eval_dir: Directory containing evaluation artifacts.

        Returns:
            List of observations.
        """
        observations: list[Observation] = []

        # Try Forge eval store first
        try:
            from animus.citizens.eval_evidence import query_eval_runs
            eval_runs = query_eval_runs(limit=20)
            for run in eval_runs:
                score = run.get("score", 0)
                suite = run.get("suite_name", "unknown")
                status = run.get("status", "unknown")
                failure_mode = run.get("failure_mode", "")
                rubric_band = run.get("rubric_band", "")

                if score < 0.7:
                    observations.append(
                        Observation(
                            source="evaluation",
                            description=f"Low eval score in '{suite}': {score:.2f} (status={status})",
                            severity="high",
                            context={
                                "score": score,
                                "suite": suite,
                                "status": status,
                                "failure_mode": failure_mode,
                                "rubric_band": rubric_band,
                                "pattern_type": "eval_regression",
                            },
                        )
                    )
                elif failure_mode:
                    observations.append(
                        Observation(
                            source="evaluation",
                            description=f"Eval failure in '{suite}': {failure_mode} (band={rubric_band})",
                            severity="medium",
                            context={
                                "suite": suite,
                                "failure_mode": failure_mode,
                                "rubric_band": rubric_band,
                                "score": score,
                                "pattern_type": "eval_failure",
                            },
                        )
                    )
        except Exception:
            pass

        # Fall back to local evidence files
        target_dir = Path(eval_dir) if eval_dir else (self.codebase_path / "evidence" if self.codebase_path else None)
        if target_dir and target_dir.exists():
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

        # Only record eval observations when there are actual findings.
        # "No regressions" is expected state, not actionable.

        self._observations.extend(observations)
        return observations

    def observe_adls(self, decisions_dir: Path | str | None = None) -> list[Observation]:
        """Read Architecture Decision Logs for constraints and standards.

        ADLs encode the project's architectural intent — what the code
        *should* be doing. Cross-referencing observations against ADL
        constraints produces grounded, specific proposals instead of
        generic "reduce coupling" boilerplate.

        Args:
            decisions_dir: Directory containing ADL markdown files.

        Returns:
            List of observations keyed by ADL constraints.
        """
        observations: list[Observation] = []
        target = Path(decisions_dir) if decisions_dir else self.codebase_path / "decisions"
        if not target.exists():
            target = self.codebase_path / "docs" / "architecture" / "decisions"

        if not target.exists():
            return observations

        adl_constraints: list[dict[str, Any]] = []
        for md_file in sorted(target.rglob("*.md")):
            try:
                text = md_file.read_text()
                # Extract ADL headers and decision blocks
                lines = text.splitlines()
                current_adl: str | None = None
                in_decision_block = False
                for line in lines:
                    if line.startswith("## ADL-"):
                        current_adl = line.strip("# ").strip()
                    elif "**Decision:**" in line:
                        in_decision_block = True
                    elif in_decision_block and line.startswith("**"):
                        in_decision_block = False
                    elif in_decision_block and current_adl and line.strip():
                        adl_constraints.append({
                            "adl": current_adl,
                            "constraint": line.strip(),
                            "source_file": str(md_file.relative_to(self.codebase_path)),
                        })
            except Exception:
                continue

        # Emit an observation summarizing active ADL constraints
        if adl_constraints:
            observations.append(
                Observation(
                    source="adl",
                    description=f"Active ADL constraints: {len(adl_constraints)} from {target.name}/",
                    severity="info",
                    context={
                        "adl_count": len(adl_constraints),
                        "constraints": adl_constraints[:10],  # cap for memory
                        "pattern_type": "adl_constraints",
                    },
                )
            )

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
        structural_patterns = {
            "circular_import", "tight_coupling", "god_class",
            "singleton_abuse", "interface_leakage", "leaky_abstraction",
        }
        for obs in self._observations:
            ptype = obs.context.get("pattern_type", "")
            is_structural = ptype in structural_patterns

            if is_structural:
                # Structural issues are always findings (cross-module impact)
                report.findings.append(obs.description)
                report.technical_debt_items.append(obs.description)
            elif obs.source == "codebase" and obs.severity in ("medium", "high", "critical"):
                # Codebase issues of medium+ severity are both technical debt and actionable findings
                report.technical_debt_items.append(obs.description)
                report.findings.append(obs.description)
            elif obs.source == "conversation":
                report.friction_points.append(obs.description)
            elif obs.severity in ("high", "critical"):
                report.findings.append(obs.description)
            elif obs.severity == "medium":
                # Medium-severity non-codebase observations are findings too
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

        Uses senior-level reasoning: cross-module dependency awareness,
        trend corroboration, impact estimation, and constraint validation.

        Args:
            report: Analysis report to base proposal on. If None, runs analysis first.

        Returns:
            Improvement proposal, or None if no actionable findings.
        """
        if report is None:
            report = self.analyze()
        elif isinstance(report, list):
            report = AnalysisReport(
                observations=report,
                findings=[o.description for o in report if o.severity in ("medium", "high", "critical")],
            )

        if not report.findings and not report.technical_debt_items and not report.friction_points:
            logger.info("No actionable findings — no proposal generated")
            return None

        # --- Senior: enrich with dependency and pattern observations ---
        dep_obs = self._analyze_dependencies(self.focus_paths or None)
        pattern_obs = self._detect_architectural_patterns(self.focus_paths or None)
        trend_obs = self._analyze_trends()
        all_obs = list(report.observations) + dep_obs + pattern_obs + trend_obs

        # Build evidence items
        evidence: list[EvidenceItem] = []
        for obs in all_obs:
            evidence.append(
                EvidenceItem(
                    source=obs.source,
                    description=obs.description,
                    data=obs.context,
                    timestamp=obs.timestamp,
                )
            )

        # Determine highest-severity problem, prioritizing focus-path observations
        def _is_focused(obs: Observation) -> bool:
            """Check if observation affects files in focus paths."""
            if not self.focus_paths:
                return False
            affected = obs.context.get("affected_files", [])
            for af in affected:
                for fp in self.focus_paths:
                    if fp in str(af):
                        return True
            return False

        # Sort observations: structural/architectural first, then focus-path, then severity
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
        structural_weight = {
            "circular_import": 3,
            "tight_coupling": 3,
            "god_class": 3,
            "singleton_abuse": 2,
            "interface_leakage": 2,
            "leaky_abstraction": 2,
            "high_complexity": 1,
            "missing_docstring": 0,
            "tech_debt_comments": 0,
            "unused_imports": 0,
            "missing_init": 0,
        }

        def _structural_priority(o: Observation) -> int:
            ptype = o.context.get("pattern_type", "")
            return structural_weight.get(ptype, 0)

        sorted_obs = sorted(
            all_obs,
            key=lambda o: (
                -_structural_priority(o),   # Structural first
                -int(_is_focused(o)),       # Focus path second
                -severity_order.get(o.severity, 0),  # Severity third
            ),
        )

        # Pick the most relevant observation
        top_obs = sorted_obs[0] if sorted_obs else None
        top_ptype = top_obs.context.get("pattern_type", "") if top_obs else ""
        is_structural = _structural_priority(top_obs) >= 2 if top_obs else False

        if report.technical_debt_items:
            if top_obs and top_obs.source in ("codebase", "trend"):
                problem = f"Technical debt: {top_obs.description}"
                affected = list(top_obs.context.get("affected_files", ["Factory", "Kernel"]))
            else:
                problem = f"Technical debt: {report.technical_debt_items[0]}"
                affected = ["Factory", "Kernel"]
            if is_structural:
                recommendation = (
                    "Refactor structural architecture to reduce coupling and improve cohesion. "
                    "Extract interfaces, split god classes, and introduce dependency boundaries."
                )
            else:
                recommendation = "Address identified technical debt through structured refactoring"
        elif report.friction_points:
            if top_obs and top_obs.source == "conversation":
                problem = f"User friction: {top_obs.description}"
            else:
                problem = f"User friction: {report.friction_points[0]}"
            recommendation = "Reduce conversation friction via workflow shortcuts or context handling"
            affected = ["Mind", "Society"]
        else:
            if top_obs:
                problem = top_obs.description
                affected = list(top_obs.context.get("affected_files", ["Mind"]))
            else:
                problem = report.findings[0] if report.findings else "General improvement opportunity"
                affected = ["Mind"]
            if is_structural:
                recommendation = (
                    "Refactor structural architecture to reduce coupling and improve cohesion. "
                    "Extract interfaces, split god classes, and introduce dependency boundaries."
                )
            else:
                recommendation = "Investigate and remediate identified issue"

        # --- Enrich with ADL constraints if available ---
        adl_contexts = [
            o.context.get("constraints", [])
            for o in all_obs
            if o.source == "adl" and o.context
        ]
        adl_constraints = []
        for ctx_list in adl_contexts:
            adl_constraints.extend(ctx_list)
        if adl_constraints:
            adl_refs = ", ".join({c["adl"] for c in adl_constraints[:5]})
            recommendation += (
                f"\n\nGrounded in ADL constraints ({adl_refs}): ensure changes align with "
                f"project architecture decisions and do not reintroduce patterns explicitly rejected."
            )

        # --- Senior: estimate impact and calibrate confidence ---
        impact = self._estimate_impact(affected)
        confidence = self._score_evidence_quality(evidence)
        effort = max(2.0, impact["impact_score"] * 8.0)  # Scale effort with blast radius

        # Build risks with impact awareness
        risks = [
            RiskAssessment(
                description="Implementation may introduce regressions",
                severity="medium" if impact["test_surface_estimate"] < 2 else "high",
                mitigation="Full test suite + eval checkpoints before merge",
                probability=min(0.5, impact["impact_score"]),
            ),
            RiskAssessment(
                description="Effort estimate may be inaccurate",
                severity="low",
                mitigation=f"Time-box initial implementation to {int(effort)} hours",
                probability=0.5,
            ),
        ]
        if impact["component_count"] > 3:
            risks.append(
                RiskAssessment(
                    description=f"Wide blast radius ({impact['component_count']} components)",
                    severity="high",
                    mitigation="Split into phased proposals, one component at a time",
                    probability=0.4,
                )
            )

        proposal = ImprovementProposal(
            id=f"ADL-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}",
            title=f"Architect Proposal: {problem[:60]}",
            problem=problem,
            evidence=evidence,
            root_cause="Identified through systematic observation, dependency analysis, and trend tracking",
            recommendation=recommendation,
            alternatives_considered=["Status quo (no change)", "Manual remediation (human-only)", "Defer to next maintenance window"],
            expected_benefits="Reduced technical debt and/or improved user experience; lower long-term maintenance cost",
            potential_risks=risks,
            confidence_score=round(confidence, 2),
            estimated_effort_hours=round(effort, 1),
            affected_components=affected,
            evaluation_plan=f"Run full test suite ({impact['test_surface_estimate']} test files affected) + benchmark comparison + manual verification + constraint re-check",
            rollback_plan="Revert to previous commit via git revert; re-run evaluation suite post-revert",
            success_metrics=["Tests pass", "Benchmarks stable or improved", "No new regressions", "Constraint check passes"],
            status=ProposalStatus.DRAFT,
        )

        # --- Senior: add trade-off analysis to recommendation ---
        trade_offs = self._build_trade_off_analysis(proposal)
        proposal.recommendation = f"{recommendation}\n\nTrade-off analysis:\n{trade_offs}"

        # --- Senior: validate against architectural constraints ---
        violations = self._check_architectural_constraints(proposal)
        if violations:
            for v in violations:
                logger.warning(v)
            # Downgrade confidence if constraints are violated
            proposal.confidence_score = max(0.25, proposal.confidence_score - 0.15 * len(violations))
            proposal.potential_risks.append(
                RiskAssessment(
                    description=f"Architectural constraint warning: {'; '.join(violations)}",
                    severity="medium",
                    mitigation="Revise proposal to comply with Citizen Contract",
                    probability=0.3,
                )
            )

        logger.info(f"Generated proposal {proposal.id}: {proposal.title} (confidence={proposal.confidence_score}, effort={proposal.estimated_effort_hours}h)")
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

            self.memory.remember(
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

    # ------------------------------------------------------------------
    # Heuristic analysis (fallback when Forge returns empty)
    # ------------------------------------------------------------------

    def _observe_heuristics(
        self, focus_paths: list[str] | None = None
    ) -> list[Observation]:
        """Run lightweight AST + text heuristics when Forge produces no suggestions.

        Checks:
        - Files >500 lines without module docstrings
        - Functions with cyclomatic complexity >10
        - TODO/FIXME/HACK comments
        - Directories with .py files but no __init__.py
        - Unused imports (imported but never referenced)
        """
        observations: list[Observation] = []
        if not self.codebase_path or not self.codebase_path.exists():
            return observations

        # Determine search roots
        roots = self._resolve_roots(focus_paths)
        py_files: list[Path] = []
        for root in roots:
            py_files.extend(root.rglob("*.py"))

        # Deduplicate and skip tests/venv
        seen = set()
        filtered: list[Path] = []
        for pf in py_files:
            rid = pf.resolve()
            if rid in seen:
                continue
            seen.add(rid)
            rel = str(pf.relative_to(self.codebase_path))
            if any(p in rel for p in ("tests/", "/tests/", "test_", ".venv/", "venv/", "node_modules/")):
                continue
            filtered.append(pf)

        self._check_file_sizes(filtered, observations)
        self._check_complexity(filtered, observations)
        self._check_todos(filtered, observations)
        self._check_missing_init(roots, observations)
        self._check_unused_imports(filtered, observations)

        logger.info(f"Heuristic analysis produced {len(observations)} observations")
        return observations

    def _resolve_roots(self, focus_paths: list[str] | None) -> list[Path]:
        """Resolve focus paths to directory roots on disk."""
        roots: list[Path] = []
        if focus_paths:
            for fp in focus_paths:
                p = self.codebase_path / fp
                if p.is_dir():
                    roots.append(p)
                elif p.parent.is_dir():
                    roots.append(p.parent)
        if not roots:
            roots.append(self.codebase_path)
        return roots

    def _check_file_sizes(self, files: list[Path], observations: list[Observation]) -> None:
        """Flag large files that lack a module docstring."""
        for pf in files:
            try:
                lines = pf.read_text().splitlines()
                line_count = len(lines)
                if line_count <= 300:
                    continue
                has_docstring = False
                if lines:
                    first = lines[0].strip()
                    has_docstring = first.startswith('"""') or first.startswith("'''")
                if not has_docstring:
                    rel = str(pf.relative_to(self.codebase_path))
                    observations.append(
                        Observation(
                            source="codebase",
                            description=f"Large file ({line_count} lines) lacks module docstring: {rel}",
                            severity="medium",
                            context={
                                "file": rel,
                                "line_count": line_count,
                                "pattern_type": "missing_docstring",
                            },
                        )
                    )
            except Exception:
                continue

    def _check_complexity(self, files: list[Path], observations: list[Observation]) -> None:
        """Flag functions/classes with high cyclomatic complexity."""
        for pf in files:
            try:
                source = pf.read_text()
                tree = ast.parse(source)
            except Exception:
                continue
            rel = str(pf.relative_to(self.codebase_path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    complexity = self._compute_complexity(node)
                    if complexity > 10:
                        observations.append(
                            Observation(
                                source="codebase",
                                description=f"High complexity function ({complexity} branches) in {rel}:{node.lineno}: {node.name}",
                                severity="high" if complexity > 20 else "medium",
                                context={
                                    "file": rel,
                                    "function": node.name,
                                    "line": node.lineno,
                                    "complexity": complexity,
                                    "pattern_type": "high_complexity",
                                },
                            )
                        )

    @staticmethod
    def _compute_complexity(node: ast.AST) -> int:
        """Simple cyclomatic complexity counter."""
        count = 1
        for child in ast.walk(node):
            if child is node:
                continue
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler,
                              ast.With, ast.Assert, ast.comprehension)):
                count += 1
            elif isinstance(child, ast.BoolOp):
                count += len(child.values) - 1
        return count

    def _check_todos(self, files: list[Path], observations: list[Observation]) -> None:
        """Surface TODO/FIXME/HACK comments as observations."""
        pattern = re.compile(r"#\s*(TODO|FIXME|HACK|XXX|BUG)\b.*?$", re.MULTILINE | re.IGNORECASE)
        for pf in files:
            try:
                text = pf.read_text()
            except Exception:
                continue
            matches = pattern.findall(text)
            if matches:
                rel = str(pf.relative_to(self.codebase_path))
                counts = {}
                for m in matches:
                    counts[m.upper()] = counts.get(m.upper(), 0) + 1
                severity = "high" if counts.get("HACK", 0) > 0 else "medium"
                observations.append(
                    Observation(
                        source="codebase",
                        description=f"{len(matches)} tech-debt comment(s) in {rel}: {dict(counts)}",
                        severity=severity,
                        context={
                            "file": rel,
                            "counts": counts,
                            "pattern_type": "tech_debt_comments",
                        },
                    )
                )

    def _check_missing_init(self, roots: list[Path], observations: list[Observation]) -> None:
        """Find directories with Python files but no __init__.py."""
        checked: set[Path] = set()
        for root in roots:
            for pf in root.rglob("*.py"):
                parent = pf.parent
                if parent in checked:
                    continue
                checked.add(parent)
                if (parent / "__init__.py").exists():
                    continue
                # Only flag if directory contains multiple .py files
                py_count = sum(1 for _ in parent.glob("*.py"))
                if py_count >= 2:
                    rel = str(parent.relative_to(self.codebase_path))
                    observations.append(
                        Observation(
                            source="codebase",
                            description=f"Package directory missing __init__.py: {rel}/ ({py_count} .py files)",
                            severity="low",
                            context={
                                "directory": rel,
                                "py_files": py_count,
                                "pattern_type": "missing_init",
                            },
                        )
                    )

    def _check_unused_imports(self, files: list[Path], observations: list[Observation]) -> None:
        """Detect top-level imports that appear unused in the same module."""
        for pf in files:
            try:
                source = pf.read_text()
                tree = ast.parse(source)
            except Exception:
                continue
            rel = str(pf.relative_to(self.codebase_path))
            imports: dict[str, str] = {}  # alias -> full_name
            used: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        imports[name] = alias.name
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        imports[name] = f"{node.module}.{alias.name}" if node.module else alias.name
                elif isinstance(node, ast.Name):
                    used.add(node.id)
                elif isinstance(node, ast.Attribute):
                    # Simple heuristic: collect first segment
                    value = node.value
                    while isinstance(value, ast.Attribute):
                        value = value.value
                    if isinstance(value, ast.Name):
                        used.add(value.id)
            unused = [name for name in imports if name not in used]
            if unused:
                observations.append(
                    Observation(
                        source="codebase",
                        description=f"Potentially unused imports in {rel}: {unused[:5]}",
                        severity="low",
                        context={
                            "file": rel,
                            "unused": unused,
                            "pattern_type": "unused_imports",
                        },
                    )
                )

    # ------------------------------------------------------------------
    # Observation deduplication (reduce noise from monorepo-scale analysis)
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate_observations(
        observations: list[Observation], max_per_pattern: int = 5
    ) -> list[Observation]:
        """Cap noisy Forge patterns so one issue type doesn't drown others.

        Groups by pattern_type (or description prefix) and keeps the top-N
        most severe per group, dropping the rest.
        """
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}

        # Bucket by pattern_type or fallback to first word of description
        buckets: dict[str, list[Observation]] = {}
        for obs in observations:
            ptype = obs.context.get("pattern_type", "")
            if not ptype:
                # Derive from description: e.g. "ImprovementCategory.REFACTORING: Long function:"
                words = obs.description.split()[:3]
                ptype = " ".join(words)
            buckets.setdefault(ptype, []).append(obs)

        kept: list[Observation] = []
        for ptype, bucket in buckets.items():
            # Sort by severity desc, then by line_count / estimated_lines desc
            def _sort_key(o: Observation):
                sev = severity_order.get(o.severity, 0)
                extra = 0
                if "line_count" in o.context:
                    extra = o.context["line_count"]
                elif "estimated_lines" in o.context:
                    extra = o.context["estimated_lines"]
                return (-sev, -extra)

            bucket_sorted = sorted(bucket, key=_sort_key)
            kept.extend(bucket_sorted[:max_per_pattern])
            dropped = len(bucket) - max_per_pattern
            if dropped > 0:
                logger.info(f"Dropped {dropped} '{ptype}' observations (kept top {max_per_pattern})")

        return kept

    # ------------------------------------------------------------------
    # Senior skillsets — cross-module analysis, trends, constraints
    # ------------------------------------------------------------------

    def _analyze_dependencies(
        self, focus_paths: list[str] | None = None
    ) -> list[Observation]:
        """Cross-module dependency analysis: circular imports and tight coupling.

        Builds a lightweight import graph from the codebase and flags:
        - Circular imports (Module A → B → A)
        - Tight coupling (one module importing too many others)
        - Interface leakage (internal modules imported by external ones)
        """
        observations: list[Observation] = []
        roots = self._resolve_roots(focus_paths)

        # Collect all Python files and their imports
        module_imports: dict[str, set[str]] = {}
        file_modules: dict[str, str] = {}  # rel_path -> dotted module path

        for root in roots:
            for pf in root.rglob("*.py"):
                rel = str(pf.relative_to(self.codebase_path))
                if any(p in rel for p in ("tests/", "/tests/", "test_", ".venv/", "venv/", "node_modules/")):
                    continue
                try:
                    source = pf.read_text()
                    tree = ast.parse(source)
                except Exception:
                    continue

                # Derive dotted module path from file path
                parts = pf.relative_to(self.codebase_path).with_suffix("").parts
                module_path = ".".join(parts)
                file_modules[rel] = module_path

                imported: set[str] = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imported.add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imported.add(node.module)
                        elif node.level > 0:
                            # Relative import: resolve against current module path
                            parts = module_path.split(".")
                            base = parts[: -node.level] if node.level <= len(parts) else []
                            for alias in node.names:
                                resolved = ".".join(base + [alias.name])
                                imported.add(resolved)
                module_imports[module_path] = imported

        # Detect tight coupling: modules importing >10 distinct modules
        for mod, imports in module_imports.items():
            if len(imports) > 10:
                observations.append(
                    Observation(
                        source="codebase",
                        description=f"Tight coupling: {mod} imports {len(imports)} distinct modules",
                        severity="medium",
                        context={
                            "module": mod,
                            "import_count": len(imports),
                            "pattern_type": "tight_coupling",
                        },
                    )
                )

        # Detect circular imports (simplified: 2-cycle detection)
        for mod_a, imports_a in module_imports.items():
            for imp in imports_a:
                if imp in module_imports:
                    if mod_a in module_imports.get(imp, set()):
                        observations.append(
                            Observation(
                                source="codebase",
                                description=f"Circular import detected: {mod_a} ↔ {imp}",
                                severity="high",
                                context={
                                    "module_a": mod_a,
                                    "module_b": imp,
                                    "pattern_type": "circular_import",
                                },
                            )
                        )

        # Detect interface leakage: internal modules exposed at package boundary
        for rel, mod in file_modules.items():
            if "/internal/" in rel or mod.endswith("_internal"):
                # Check if any non-internal module imports this
                for other_mod, other_imports in module_imports.items():
                    if "/internal/" not in other_mod and other_mod != mod:
                        prefix = mod.split(".")[0] if "." in mod else mod
                        if any(imp.startswith(prefix) for imp in other_imports):
                            # This is a weak signal, only add once per internal module
                            already = any(
                                o.context.get("module") == mod and o.context.get("pattern_type") == "interface_leakage"
                                for o in observations
                            )
                            if not already:
                                observations.append(
                                    Observation(
                                        source="codebase",
                                        description=f"Internal module {mod} may be leaking across package boundary",
                                        severity="low",
                                        context={
                                            "module": mod,
                                            "pattern_type": "interface_leakage",
                                        },
                                    )
                                )
                            break

        logger.info(f"Dependency analysis produced {len(observations)} observations")
        return observations

    def _detect_architectural_patterns(
        self, focus_paths: list[str] | None = None
    ) -> list[Observation]:
        """Detect architectural anti-patterns: god classes, singleton abuse, etc.

        Uses AST heuristics to flag patterns that senior engineers watch for.
        """
        observations: list[Observation] = []
        roots = self._resolve_roots(focus_paths)

        for root in roots:
            for pf in root.rglob("*.py"):
                rel = str(pf.relative_to(self.codebase_path))
                if any(p in rel for p in ("tests/", "/tests/", "test_", ".venv/", "venv/", "node_modules/")):
                    continue
                try:
                    source = pf.read_text()
                    tree = ast.parse(source)
                except Exception:
                    continue

                # God class: class with >15 methods or >500 lines
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                        if len(methods) > 15:
                            observations.append(
                                Observation(
                                    source="codebase",
                                    description=f"God class {node.name} has {len(methods)} methods in {rel}",
                                    severity="medium",
                                    context={
                                        "file": rel,
                                        "class": node.name,
                                        "method_count": len(methods),
                                        "pattern_type": "god_class",
                                    },
                                )
                            )

                # Singleton abuse: classes with __new__ or getInstance
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        has_singleton = False
                        for n in node.body:
                            if isinstance(n, ast.FunctionDef):
                                if n.name == "__new__" or "instance" in n.name.lower():
                                    has_singleton = True
                                    break
                        if has_singleton:
                            observations.append(
                                Observation(
                                    source="codebase",
                                    description=f"Singleton pattern detected in {node.name} ({rel}) — consider dependency injection",
                                    severity="low",
                                    context={
                                        "file": rel,
                                        "class": node.name,
                                        "pattern_type": "singleton_abuse",
                                    },
                                )
                            )

                # Leaky abstraction: class with public attributes (no property) and complex methods
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        public_attrs = [
                            n for n in node.body
                            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and not n.target.id.startswith("_")
                        ]
                        complex_methods = [
                            n for n in node.body
                            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and self._compute_complexity(n) > 10
                        ]
                        if len(public_attrs) >= 3 and len(complex_methods) >= 2:
                            observations.append(
                                Observation(
                                    source="codebase",
                                    description=f"Leaky abstraction: {node.name} in {rel} exposes {len(public_attrs)} public attrs with {len(complex_methods)} complex methods",
                                    severity="medium",
                                    context={
                                        "file": rel,
                                        "class": node.name,
                                        "public_attrs": len(public_attrs),
                                        "complex_methods": len(complex_methods),
                                        "pattern_type": "leaky_abstraction",
                                    },
                                )
                            )

        logger.info(f"Pattern analysis produced {len(observations)} observations")
        return observations

    def _analyze_trends(self) -> list[Observation]:
        """Query memory for past observations and detect degrading trends.

        A senior architect tracks whether the same problems keep appearing
        across cycles — recurrence is a stronger signal than first occurrence.
        """
        observations: list[Observation] = []
        if self.memory is None:
            return observations

        try:
            from animus.memory import MemoryType

            # Search for past architect observations
            results = self.memory.search(
                query="technical debt OR high complexity OR TODO OR FIXME",
                memory_type=MemoryType.PROCEDURAL,
                limit=50,
                tags=["architect", "proposal"],
            )

            # Bucket by pattern_type to detect recurrence
            pattern_counts: dict[str, int] = {}
            pattern_files: dict[str, set[str]] = {}
            for mem in results:
                meta = mem.get("metadata", {})
                for obs in meta.get("evidence", []):
                    ptype = obs.get("data", {}).get("pattern_type", "")
                    if ptype:
                        pattern_counts[ptype] = pattern_counts.get(ptype, 0) + 1
                        file_key = obs.get("data", {}).get("file", obs.get("data", {}).get("module", ""))
                        if file_key:
                            if ptype not in pattern_files:
                                pattern_files[ptype] = set()
                            pattern_files[ptype].add(file_key)

            # Flag patterns that appear in ≥3 past proposals
            for ptype, count in pattern_counts.items():
                if count >= 3:
                    files = pattern_files.get(ptype, set())
                    observations.append(
                        Observation(
                            source="trend",
                            description=f"Recurring pattern '{ptype}' observed in {count} past cycles across {len(files)} file(s)",
                            severity="high" if count >= 5 else "medium",
                            context={
                                "pattern_type": ptype,
                                "recurrence_count": count,
                                "affected_files": sorted(files)[:10],
                                "trend_direction": "worsening" if count >= 5 else "stable",
                            },
                        )
                    )

        except Exception as e:
            logger.debug(f"Trend analysis skipped: {e}")

        logger.info(f"Trend analysis produced {len(observations)} observations")
        return observations

    def _check_architectural_constraints(
        self, proposal: ImprovementProposal
    ) -> list[str]:
        """Validate that a proposal does not violate Animus architectural constraints.

        Constraints (hard rules):
        1. Citizens NEVER modify code directly (only observe → propose → approve → Forge)
        2. Citizens NEVER change memory autonomously
        3. Citizens NEVER merge or deploy
        4. A proposal must not recommend direct file writes by citizens
        5. A proposal must include evaluation plan and rollback plan
        """
        violations: list[str] = []
        rec = proposal.recommendation.lower()

        forbidden_verbs = ["modify directly", "edit directly", "write to file", "auto-merge", "auto-deploy", "bypass approval"]
        for verb in forbidden_verbs:
            if verb in rec:
                violations.append(f"Constraint violation: proposal suggests '{verb}' — citizens must not modify code directly")

        if not proposal.evaluation_plan:
            violations.append("Constraint violation: proposal lacks evaluation plan")

        if not proposal.rollback_plan:
            violations.append("Constraint violation: proposal lacks rollback plan")

        if len(proposal.affected_components) > 5:
            violations.append("Constraint warning: blast radius >5 components — consider splitting into smaller proposals")

        return violations

    def _estimate_impact(self, affected_files: list[str]) -> dict[str, Any]:
        """Estimate the blast radius of a proposed change.

        Returns:
            Dict with component count, test surface estimate,
            and a risk-weighted impact score.
        """
        if not affected_files:
            return {"component_count": 0, "test_surface_estimate": 0, "impact_score": 0.0}

        components = set()
        test_files_affected = 0

        for af in affected_files:
            # Infer component from path
            parts = str(af).split("/")
            if len(parts) >= 2:
                components.add(parts[0])
                if parts[0] == "packages" and len(parts) >= 3:
                    components.add(parts[1])

            # Rough test surface estimate: find corresponding test files
            rel = Path(af)
            test_globs = [
                self.codebase_path / f"tests/**/test_{rel.stem}.py",
                self.codebase_path / f"**/tests/**/test_{rel.stem}.py",
                self.codebase_path / f"**/*test_{rel.stem}.py",
            ]
            for glob in test_globs:
                if list(self.codebase_path.glob(str(glob.relative_to(self.codebase_path)))):
                    test_files_affected += 1
                    break

        component_count = len(components)
        impact_score = min(1.0, (component_count * 0.15) + (test_files_affected * 0.1))

        return {
            "component_count": component_count,
            "test_surface_estimate": test_files_affected,
            "impact_score": impact_score,
        }

    def _score_evidence_quality(self, evidence: list[EvidenceItem]) -> float:
        """Score the strength of evidence for a proposal.

        Higher scores when:
        - Multiple independent sources corroborate (codebase + eval + conversation)
        - Evidence includes quantitative data
        - Evidence is recent
        """
        if not evidence:
            return 0.3

        sources = set(e.source for e in evidence)
        source_diversity = len(sources) / 4.0  # max 4 sources

        quantitative = sum(1 for e in evidence if e.data)
        quantitative_ratio = quantitative / len(evidence) if evidence else 0

        # Recency bonus: evidence from last 7 days
        now = datetime.now()
        recent = sum(1 for e in evidence if (now - e.timestamp).days <= 7)
        recency_ratio = recent / len(evidence) if evidence else 0

        score = 0.4 + (source_diversity * 0.3) + (quantitative_ratio * 0.2) + (recency_ratio * 0.1)
        return min(1.0, score)

    def _build_trade_off_analysis(
        self, proposal: ImprovementProposal
    ) -> str:
        """Build a structured trade-off paragraph for the proposal.

        A senior architect doesn't just say "do X" — they explain why X over Y,
        what the costs are, and what the risks of inaction are.
        """
        parts: list[str] = []

        # Cost estimate
        effort = proposal.estimated_effort_hours or 4.0
        parts.append(f"Estimated effort: {effort} hours (including tests and validation).")

        # Blast radius
        affected = proposal.affected_components or ["Unknown"]
        parts.append(f"Blast radius: {len(affected)} component(s) — {', '.join(affected[:5])}")

        # Opportunity cost
        parts.append(
            f"Opportunity cost: {effort * 0.5:.1f} hours of other architect observations deferred."
        )

        # Risk of inaction
        risks = proposal.potential_risks or []
        if risks:
            highest = max(risks, key=lambda r: (r.probability * ({"critical": 4, "high": 3, "medium": 2, "low": 1}.get(r.severity, 2))))
            parts.append(
                f"Risk of inaction: If unaddressed, '{highest.description}' has {int(highest.probability * 100)}% probability of causing debt accumulation."
            )

        # Alternatives
        alts = proposal.alternatives_considered or []
        if alts:
            parts.append(f"Alternatives considered: {', '.join(alts)}")

        return "\n".join(parts)

    @staticmethod
    def _expand_focus_paths(focus_paths: list[str] | None) -> list[str]:
        """Expand directory paths to file glob patterns for Forge.

        Converts e.g. ['packages/core/animus'] → ['packages/core/animus/**/*.py']
        """
        if not focus_paths:
            return focus_paths or []
        expanded: list[str] = []
        for fp in focus_paths:
            if fp.endswith("/"):
                fp = fp[:-1]
            # If already a glob or file path, pass through
            if "*" in fp or fp.endswith(".py"):
                expanded.append(fp)
            else:
                expanded.append(f"{fp}/**/*.py")
        return expanded

    @staticmethod
    def _map_priority_to_severity(priority: int) -> str:
        """Map analyzer priority (1=highest, 5=lowest) to severity."""
        mapping = {1: "critical", 2: "high", 3: "medium", 4: "low", 5: "info"}
        return mapping.get(priority, "medium")

    # ------------------------------------------------------------------
    # Indexed code memory observation (codebase via semantic memory)
    # ------------------------------------------------------------------

    def _observe_indexed_code_memory(
        self, focus_paths: list[str] | None = None
    ) -> list[Observation]:
        """Query indexed code memory for coverage, recency, and hotspots.

        When the codebase has been ingested via ``ingest_codebase()``,
        semantic memory contains AST-level chunks with metadata. This
        method surfaces coverage gaps, recently indexed areas (proxy
        for active development), and high-complexity functions that
        were captured during chunking.

        Args:
            focus_paths: Optional paths to narrow the scope.

        Returns:
            Observations derived from indexed code memory.
        """
        observations: list[Observation] = []
        if self.memory is None or self.codebase_path is None:
            return observations

        try:
            indexed_chunks = self._get_indexed_code_chunks(focus_paths)
        except Exception as e:
            logger.debug(f"Indexed code memory query failed: {e}")
            return observations

        if not indexed_chunks:
            # Graceful degradation: no index yet, not an actionable finding
            return observations

        # --- Coverage analysis ------------------------------------------------
        indexed_files = {
            mem.metadata.get("file_path", "")
            for mem in indexed_chunks
            if mem.metadata.get("file_path")
        }
        # Count .py files on disk for comparison
        roots = self._resolve_roots(focus_paths)
        disk_files: set[str] = set()
        for root in roots:
            for pf in root.rglob("*.py"):
                rel = str(pf.relative_to(self.codebase_path))
                if any(p in rel for p in ("tests/", "/tests/", "test_", ".venv/", "venv/", "node_modules/")):
                    continue
                disk_files.add(rel)

        if disk_files:
            coverage = len(indexed_files) / len(disk_files)
            if coverage < 0.5:
                observations.append(
                    Observation(
                        source="codebase",
                        description=f"Indexed code memory covers only {coverage:.0%} of Python files ({len(indexed_files)}/{len(disk_files)}); consider re-running ingest_codebase()",
                        severity="medium",
                        context={
                            "indexed_files": len(indexed_files),
                            "total_files": len(disk_files),
                            "coverage_ratio": coverage,
                            "pattern_type": "indexed_memory_coverage",
                        },
                    )
                )
            else:
                observations.append(
                    Observation(
                        source="codebase",
                        description=f"Indexed code memory covers {len(indexed_files)} Python files ({coverage:.0%})",
                        severity="info",
                        context={
                            "indexed_files": len(indexed_files),
                            "total_files": len(disk_files),
                            "coverage_ratio": coverage,
                            "pattern_type": "indexed_memory_coverage",
                        },
                    )
                )

        # --- Recency / active-development hotspots ---------------------------
        now = datetime.now()
        recent_files: dict[str, int] = {}
        for mem in indexed_chunks:
            rel = mem.metadata.get("file_path", "")
            if not rel:
                continue
            # created_at is the ingest timestamp (last reindex of that file)
            age_hours = (now - mem.created_at).total_seconds() / 3600 if mem.created_at else 999
            if age_hours <= 24:
                recent_files[rel] = recent_files.get(rel, 0) + 1

        if recent_files:
            top_recent = sorted(recent_files.items(), key=lambda x: -x[1])[:5]
            observations.append(
                Observation(
                    source="codebase",
                    description=f"Recently indexed {len(recent_files)} file(s) with {sum(recent_files.values())} chunk(s) — active development detected in {', '.join(f for f, _ in top_recent)}",
                    severity="info",
                    context={
                        "recent_file_count": len(recent_files),
                        "recent_chunk_count": sum(recent_files.values()),
                        "top_recent": [f for f, _ in top_recent],
                        "pattern_type": "indexed_memory_recency",
                    },
                )
            )

        # --- Complexity hotspots from indexed chunks -------------------------
        complex_funcs: list[dict] = []
        for mem in indexed_chunks:
            meta = mem.metadata
            if meta.get("chunk_type") in ("function", "method"):
                # Try to extract complexity if it was computed during chunking
                content = mem.content
                func_name = meta.get("identifier", "unknown")
                file_path = meta.get("file_path", "")
                line_no = meta.get("line_no", 0)
                # Rough heuristic: count control-flow keywords in the chunk
                # (exact AST complexity isn't stored, but we can estimate)
                if content and isinstance(content, str):
                    flow_keywords = len(re.findall(r"\b(if|for|while|except|with|assert)\b", content))
                    if flow_keywords >= 8:
                        complex_funcs.append({
                            "file": file_path,
                            "function": func_name,
                            "line": line_no,
                            "flow_keywords": flow_keywords,
                        })

        if complex_funcs:
            # Deduplicate by file+function
            seen = set()
            unique = []
            for cf in complex_funcs:
                key = (cf["file"], cf["function"])
                if key not in seen:
                    seen.add(key)
                    unique.append(cf)
            top_complex = sorted(unique, key=lambda x: -x["flow_keywords"])[:5]
            observations.append(
                Observation(
                    source="codebase",
                    description=f"Indexed memory reveals {len(unique)} high-complexity function(s); top: {top_complex[0]['function']} in {top_complex[0]['file']}",
                    severity="medium",
                    context={
                        "complex_function_count": len(unique),
                        "top_functions": top_complex,
                        "pattern_type": "high_complexity",
                    },
                )
            )

        logger.info(f"Indexed code memory analysis produced {len(observations)} observation(s)")
        return observations

    def _get_indexed_code_chunks(
        self, focus_paths: list[str] | None = None
    ) -> list[Any]:
        """Query semantic memory for code chunks ingested via ``code_ingest``.

        Args:
            focus_paths: If provided, filters to chunks whose file_path
                metadata contains one of these path fragments.

        Returns:
            List of Memory objects with source="code_ingest".
        """
        if self.memory is None:
            return []

        results = self.memory.search(
            query="function class method codebase",
            memory_type=MemoryType.SEMANTIC,
            source="code_ingest",
            limit=500,
        )

        if not focus_paths:
            return results

        filtered = []
        for mem in results:
            fp = mem.metadata.get("file_path", "")
            for focus in focus_paths:
                focus = focus.rstrip("/")
                if focus in fp or fp.startswith(focus + "/"):
                    filtered.append(mem)
                    break
        return filtered

    def __repr__(self) -> str:
        return f"ArchitectCitizen(codebase={self.codebase_path}, observations={len(self._observations)})"
