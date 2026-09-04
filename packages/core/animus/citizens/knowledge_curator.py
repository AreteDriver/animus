"""Citizen 003 — The Knowledge Curator.

The permanent "librarian" of Animus.

Responsibilities:
- Observe memory for stale references (claims about files/functions that no longer exist)
- Detect contradictory memories on the same topic
- Identify orphan topic files not referenced from other docs
- Flag outdated time-sensitive claims without recent verification
- Propose knowledge maintenance: updates, archiving, cross-linking

Never:
- Modify memory or code directly
- Delete memories autonomously
- Change knowledge structure without human approval

Instead:
    Observe → Analyze → Curate Proposal → Human Approval → Forge → Evidence → Merge
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from animus.citizens.architect import Observation
from animus.citizens.proposal import (
    EvidenceItem,
    ImprovementProposal,
    ProposalStatus,
    RiskAssessment,
)
from animus.logging import get_logger

logger = get_logger("citizens.knowledge_curator")


@dataclass
class KnowledgeDrift:
    """A detected drift or inconsistency in the knowledge base."""

    drift_type: str  # "stale_reference", "contradiction", "orphan_topic", "outdated_claim"
    description: str
    severity: str = "low"  # "critical", "high", "medium", "low"
    affected_memory_id: str = ""
    suggested_action: str = ""
    context: dict[str, Any] = field(default_factory=dict)


class KnowledgeCuratorCitizen:
    """Citizen 003 — The Knowledge Curator.

    Continuously evaluates knowledge quality and proposes
    maintenance to keep the memory store accurate and useful.

    This citizen NEVER modifies code, memory, or systems directly.
    It only observes, analyzes, and produces proposals.
    """

    def __init__(
        self,
        codebase_path: Path | str | None = None,
        memory_layer: Any = None,
    ):
        self.codebase_path = Path(codebase_path).expanduser() if codebase_path else None
        self.memory = memory_layer
        self._drifts: list[KnowledgeDrift] = []

    # ------------------------------------------------------------------
    # Observation methods (read-only)
    # ------------------------------------------------------------------

    def observe_stale_references(self, limit: int = 100) -> list[Observation]:
        """Scan memories for references to files or functions that no longer exist.

        Args:
            limit: Maximum memories to scan.

        Returns:
            List of observations about stale references.
        """
        observations: list[Observation] = []

        if not self.memory:
            observations.append(
                Observation(
                    source="knowledge",
                    description="Memory layer not available for stale-reference scan",
                    severity="low",
                )
            )
            return observations

        if not self.codebase_path or not self.codebase_path.exists():
            observations.append(
                Observation(
                    source="knowledge",
                    description="Codebase path not configured or not found — cannot verify references",
                    severity="medium",
                )
            )
            return observations

        # Search for memories that mention files or functions
        try:
            memories = self._search_memories("file function method class", limit=limit)
        except Exception as e:
            logger.warning(f"Memory search failed: {e}")
            return observations

        # Regex patterns to detect file/function references
        file_patterns = [
            # "in file.py" or "file.py:123" or "`file.py`"
            r"[\s`\"']+([\w\-/]+\.(?:py|md|yaml|yml|json|sh|js|ts|go|rs|java))\b",
            # "module.ClassName" or "function_name"
            r"\b([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)\b",
        ]

        file_re = re.compile("|".join(file_patterns))

        for mem in memories:
            content = mem.get("content", "")
            mem_id = mem.get("id", "unknown")

            for match in file_re.finditer(content):
                ref = match.group(1)
                if not ref:
                    continue

                # Check if file reference exists
                if "." in ref and not ref.endswith(">"):
                    candidate = self.codebase_path / ref
                    if not candidate.exists():
                        # Also try relative to various common dirs
                        found = False
                        for subdir in ["packages/core", "src", "."]:
                            if (self.codebase_path / subdir / ref).exists():
                                found = True
                                break
                        if not found:
                            observations.append(
                                Observation(
                                    source="knowledge",
                                    description=f"Memory references non-existent file: '{ref}'",
                                    severity="medium",
                                    context={
                                        "memory_id": mem_id,
                                        "missing_ref": ref,
                                        "pattern_type": "stale_reference",
                                    },
                                )
                            )

        return observations

    def observe_contradictions(self, limit: int = 100) -> list[Observation]:
        """Find memories that make conflicting claims about the same topic.

        Uses simple heuristics: same keyword cluster, opposite sentiment.

        Args:
            limit: Maximum memories to compare.

        Returns:
            List of contradiction observations.
        """
        observations: list[Observation] = []

        if not self.memory:
            return observations

        try:
            memories = self._search_memories("", limit=limit)  # Get a broad sample
        except Exception as e:
            logger.warning(f"Memory search failed: {e}")
            return observations

        # Extract key claims from memories
        claims: list[tuple[str, str, dict]] = []  # (topic, claim_polarity, context)

        polarity_keywords = {
            "positive": [
                "is",
                "does",
                "supports",
                "enables",
                "improves",
                "correct",
                "fast",
                "safe",
            ],
            "negative": [
                "is not",
                "does not",
                "breaks",
                "disables",
                "degrades",
                "bug",
                "slow",
                "unsafe",
                "deprecated",
            ],
        }

        for mem in memories:
            raw_content = mem.get("content", "")
            content = raw_content.lower()
            mem_id = mem.get("id", "")

            # Look for module/file/function mentions as topic anchors
            topics = self._extract_topic_anchors(raw_content)
            if not topics:
                continue

            for topic in topics:
                topic_lc = topic.lower()
                pos_score = sum(
                    1
                    for kw in polarity_keywords["positive"]
                    if topic_lc in content and kw in content
                )
                neg_score = sum(
                    1
                    for kw in polarity_keywords["negative"]
                    if topic_lc in content and kw in content
                )

                if pos_score > 0 or neg_score > 0:
                    polarity = "positive" if pos_score >= neg_score else "negative"
                    claims.append(
                        (topic, polarity, {"memory_id": mem_id, "content_snippet": content[:100]})
                    )

        # Find contradictions: same topic, opposite polarity from different memories
        topic_polarities: dict[str, list[tuple[str, dict]]] = {}
        for topic, polarity, ctx in claims:
            topic_polarities.setdefault(topic, []).append((polarity, ctx))

        for topic, pols in topic_polarities.items():
            if len(pols) < 2:
                continue

            has_pos = any(p == "positive" for p, _ in pols)
            has_neg = any(p == "negative" for p, _ in pols)

            if has_pos and has_neg:
                mem_ids = [ctx["memory_id"] for p, ctx in pols]
                observations.append(
                    Observation(
                        source="knowledge",
                        description=f"Contradictory claims about '{topic}': some memories say it works, others say it breaks",
                        severity="high",
                        context={
                            "topic": topic,
                            "memory_ids": mem_ids,
                            "pattern_type": "contradiction",
                        },
                    )
                )

        return observations

    def observe_outdated_claims(self, limit: int = 100) -> list[Observation]:
        """Detect memories with time-sensitive claims that are old or undated.

        Args:
            limit: Maximum memories to scan.

        Returns:
            List of observations about outdated claims.
        """
        observations: list[Observation] = []

        if not self.memory:
            return observations

        try:
            memories = self._search_memories("", limit=limit)
        except Exception as e:
            logger.warning(f"Memory search failed: {e}")
            return observations

        now = datetime.now()
        stale_threshold_days = 30
        critical_threshold_days = 90

        # Time-sensitive keywords that degrade without dates
        time_sensitive_keywords = [
            "recently",
            "lately",
            "new",
            "upcoming",
            "soon",
            "last month",
            "last week",
            "yesterday",
            "just",
            "deprecated",
            "version",
            "v2",
            "v3",
            "latest",
            "now supports",
            "recently changed",
            "ccpgames",
        ]

        for mem in memories:
            content = mem.get("content", "")
            mem_id = mem.get("id", "")
            created_at = mem.get("created_at")

            # Parse created_at
            age_days = None
            if created_at:
                if isinstance(created_at, str):
                    try:
                        created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        age_days = (now - created_dt).days
                    except ValueError:
                        pass
                elif isinstance(created_at, datetime):
                    age_days = (now - created_at).days

            # Check for time-sensitive content without explicit date
            content_lower = content.lower()
            is_time_sensitive = any(kw in content_lower for kw in time_sensitive_keywords)

            if is_time_sensitive and age_days is not None:
                if age_days > critical_threshold_days:
                    observations.append(
                        Observation(
                            source="knowledge",
                            description=f"Memory contains time-sensitive claim ({age_days}d old) with no explicit verification date: '{content[:60]}...'",
                            severity="high",
                            context={
                                "memory_id": mem_id,
                                "age_days": age_days,
                                "pattern_type": "outdated_claim",
                            },
                        )
                    )
                elif age_days > stale_threshold_days:
                    observations.append(
                        Observation(
                            source="knowledge",
                            description=f"Memory may be stale ({age_days}d old): '{content[:60]}...'",
                            severity="medium",
                            context={
                                "memory_id": mem_id,
                                "age_days": age_days,
                                "pattern_type": "outdated_claim",
                            },
                        )
                    )

            # Also flag undated time-sensitive claims
            if is_time_sensitive and age_days is None:
                observations.append(
                    Observation(
                        source="knowledge",
                        description=f"Time-sensitive memory has no creation date — cannot verify freshness: '{content[:60]}...'",
                        severity="medium",
                        context={
                            "memory_id": mem_id,
                            "pattern_type": "outdated_claim",
                        },
                    )
                )

        return observations

    def observe_orphan_topics(self) -> list[Observation]:
        """Find markdown topic files not referenced from any other doc.

        Returns:
            List of orphan topic observations.
        """
        observations: list[Observation] = []

        if not self.codebase_path or not self.codebase_path.exists():
            observations.append(
                Observation(
                    source="knowledge",
                    description="Codebase path not configured — cannot detect orphan topics",
                    severity="low",
                )
            )
            return observations

        # Find all markdown files under topics/
        topics_dir = self.codebase_path / "topics"
        if not topics_dir.exists():
            # Also check docs/, notes/, or repo root
            for alt in [self.codebase_path / "docs", self.codebase_path]:
                if (alt / "topics").exists():
                    topics_dir = alt / "topics"
                    break

        if not topics_dir.exists():
            # No topics/ dir in this codebase — nothing to audit.
            return observations

        all_md_files = list(topics_dir.rglob("*.md"))
        if not all_md_files:
            return observations

        # Scan all markdown files for internal links
        referenced: set[Path] = set()
        for md_file in all_md_files:
            try:
                content = md_file.read_text()
                # Match [text](path) links
                for match in re.finditer(r"\[.*?\]\(([^)#]+)\)", content):
                    link = match.group(1)
                    # Resolve relative to md_file's directory
                    linked_path = (md_file.parent / link).resolve()
                    if linked_path in all_md_files:
                        referenced.add(linked_path)
            except Exception:
                continue

        orphans = [f for f in all_md_files if f not in referenced]
        for orphan in orphans:
            observations.append(
                Observation(
                    source="knowledge",
                    description=f"Orphan topic file not referenced from any other doc: '{orphan.relative_to(self.codebase_path)}'",
                    severity="low",
                    context={
                        "file": str(orphan),
                        "pattern_type": "orphan_topic",
                    },
                )
            )

        return observations

    def observe_eval_results(self) -> list[Observation]:
        """Observe Forge eval results for knowledge-quality signals.

        Eval results that fail with schema_drift or hallucination indicate
        that the system's understanding of its own interfaces has degraded.

        Returns:
            List of observations.
        """
        observations: list[Observation] = []

        try:
            from animus.citizens.eval_evidence import query_eval_runs

            eval_runs = query_eval_runs(limit=20)

            for run in eval_runs:
                suite = run.get("suite_name", "unknown")
                score = run.get("score", 0)
                failure_mode = run.get("failure_mode", "")
                rubric_band = run.get("rubric_band", "")

                # Knowledge-relevant failure modes
                knowledge_failures = {
                    "schema_drift",
                    "hallucination",
                    "wrong_answer",
                    "contract_violation",
                }
                if failure_mode in knowledge_failures:
                    observations.append(
                        Observation(
                            source="knowledge",
                            description=f"Eval '{suite}' failed with {failure_mode} (band={rubric_band}, score={score:.2f}) — may indicate stale knowledge",
                            severity="high",
                            context={
                                "suite": suite,
                                "failure_mode": failure_mode,
                                "rubric_band": rubric_band,
                                "score": score,
                                "pattern_type": "eval_knowledge_failure",
                            },
                        )
                    )
                elif score < 0.6:
                    observations.append(
                        Observation(
                            source="knowledge",
                            description=f"Eval '{suite}' has low score ({score:.2f}) — verify assumptions",
                            severity="medium",
                            context={
                                "suite": suite,
                                "score": score,
                                "pattern_type": "eval_low_score",
                            },
                        )
                    )
        except Exception:
            pass

        return observations

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def analyze(self) -> list[KnowledgeDrift]:
        """Run all observation methods and aggregate into KnowledgeDrift records.

        Returns:
            List of detected knowledge drifts.
        """
        observations: list[Observation] = []
        observations.extend(self.observe_stale_references())
        observations.extend(self.observe_contradictions())
        observations.extend(self.observe_outdated_claims())
        observations.extend(self.observe_orphan_topics())
        observations.extend(self.observe_eval_results())

        drifts: list[KnowledgeDrift] = []
        drift_groups: dict[str, list[Observation]] = {}

        for obs in observations:
            dt = obs.context.get("pattern_type", "unknown") if obs.context else "unknown"
            drift_groups.setdefault(dt, []).append(obs)

        for drift_type, group in drift_groups.items():
            for obs in group:
                drifts.append(
                    KnowledgeDrift(
                        drift_type=drift_type,
                        description=obs.description,
                        severity=obs.severity,
                        affected_memory_id=obs.context.get("memory_id", "") if obs.context else "",
                        suggested_action=self._suggest_for_drift(drift_type),
                        context=obs.context or {},
                    )
                )

        self._drifts = drifts
        return drifts

    # ------------------------------------------------------------------
    # Proposal generation
    # ------------------------------------------------------------------

    def generate_proposal(self) -> ImprovementProposal | None:
        """Generate an improvement proposal from knowledge drift analysis.

        Returns:
            Improvement proposal, or None if no actionable findings.
        """
        drifts = self.analyze()

        if not drifts:
            logger.info("No knowledge drift detected — no proposal generated")
            return None

        # Focus on highest-severity drift
        top = max(
            drifts,
            key=lambda d: {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(d.severity, 0),
        )

        # Build problem/recommendation based on drift type
        problem, recommendation = self._build_problem_recommendation(top)

        # Gather evidence from all drifts of the same type
        evidence = [
            EvidenceItem(
                source="knowledge_curator",
                description=f"{d.drift_type}: {d.description}",
                data={"severity": d.severity, "memory_id": d.affected_memory_id},
            )
            for d in drifts
            if d.drift_type == top.drift_type
        ]

        risks = [
            RiskAssessment(
                description="Updating memories may lose valid historical context",
                severity="low",
                mitigation="Archive old versions rather than overwrite; preserve parent chains",
                probability=0.3,
            ),
            RiskAssessment(
                description="Cross-linking may create circular references",
                severity="low",
                mitigation="Validate link graph before merging changes",
                probability=0.2,
            ),
        ]

        components = ["Mind"]
        if top.drift_type == "stale_reference":
            components = ["Mind", "Factory"]
        elif top.drift_type == "outdated_claim":
            components = ["Mind", "Society"]

        proposal = ImprovementProposal(
            id=f"ADL-{datetime.now().strftime('%Y%m%d')}-{__import__('uuid').uuid4().hex[:6]}",
            title=f"Knowledge Maintenance: {problem[:50]}",
            problem=problem,
            evidence=evidence[:5],  # Limit evidence to top 5
            root_cause="Knowledge accumulated without systematic verification or cross-referencing",
            recommendation=recommendation,
            alternatives_considered=[
                "Status quo (stale knowledge persists)",
                "Manual periodic audits",
            ],
            expected_benefits="Higher trust in Animus memory; fewer incorrect code suggestions",
            potential_risks=risks,
            confidence_score=0.6,
            estimated_effort_hours=4.0,
            affected_components=components,
            evaluation_plan="Re-run Knowledge Curator scan after maintenance; verify drift count reduced",
            rollback_plan="Restore memory from snapshot taken before maintenance",
            success_metrics=[
                "Stale reference count reduced",
                "Contradictions resolved",
                "Orphan topics linked",
            ],
            status=ProposalStatus.DRAFT,
        )

        logger.info(f"Knowledge Curator generated proposal {proposal.id}")
        return proposal

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

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
                tags=["knowledge_curator", "proposal", proposal.status.value],
                metadata=proposal.to_dict(),
            )
            logger.info(f"Proposal {proposal.id} stored in memory")
            return True
        except Exception as e:
            logger.error(f"Failed to store proposal: {e}")
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _search_memories(self, query: str, limit: int = 100) -> list[dict]:
        """Search memory and return raw dict results."""
        if not self.memory:
            return []

        try:
            results = self.memory.search(query=query, limit=limit)
            if results and hasattr(results[0], "to_dict"):
                return [r.to_dict() for r in results]
            if results and isinstance(results[0], dict):
                return results
            return []
        except Exception:
            # Fallback: try recall if search isn't available
            try:
                recalled = self.memory.recall(query=query, limit=limit)
                if recalled and hasattr(recalled[0], "to_dict"):
                    return [r.to_dict() for r in recalled]
                if recalled and isinstance(recalled[0], dict):
                    return recalled
            except Exception:
                pass
            return []

    @staticmethod
    def _extract_topic_anchors(content: str) -> list[str]:
        """Extract potential topic anchors from memory content."""
        anchors = []

        # Look for module/file references
        for match in re.finditer(
            r"\b([a-z_][a-z0-9_]*)\.(?:py|js|ts|go|rs)\b", content, re.IGNORECASE
        ):
            anchors.append(match.group(1))

        # Look for ClassName references (CamelCase)
        for match in re.finditer(r"\b([A-Z][a-zA-Z0-9_]+)\b", content):
            anchors.append(match.group(1))

        # Look for function_name references following "def " or "method "
        for match in re.finditer(
            r"\b(?:def|method|function)\s+([a-z_][a-z0-9_]*)\b", content, re.IGNORECASE
        ):
            anchors.append(match.group(1))

        return list(set(anchors))

    @staticmethod
    def _suggest_for_drift(drift_type: str) -> str:
        """Generate a suggestion for a given drift type."""
        suggestions = {
            "stale_reference": (
                "Update memory to reference current file paths or remove obsolete claims."
            ),
            "contradiction": (
                "Reconcile conflicting memories by verifying against current codebase "
                "and marking outdated claims as deprecated."
            ),
            "outdated_claim": (
                "Add verification date to time-sensitive memories or re-verify against current state."
            ),
            "orphan_topic": (
                "Add cross-references from related topic files or README to link orphan topics."
            ),
        }
        return suggestions.get(drift_type, "Review and update knowledge.")

    @staticmethod
    def _build_problem_recommendation(drift: KnowledgeDrift) -> tuple[str, str]:
        """Build problem/recommendation pair from drift."""
        if drift.drift_type == "stale_reference":
            return (
                f"Memory references code artifacts that no longer exist: {drift.description[:80]}",
                "Audit memories against current codebase. Update or archive stale references.",
            )
        elif drift.drift_type == "contradiction":
            return (
                f"Conflicting memories detected: {drift.description[:80]}",
                "Verify current behavior against codebase. Mark outdated claims as deprecated.",
            )
        elif drift.drift_type == "outdated_claim":
            return (
                f"Time-sensitive knowledge is stale or undated: {drift.description[:80]}",
                "Add verification dates to all time-sensitive claims. Re-verify monthly.",
            )
        elif drift.drift_type == "orphan_topic":
            return (
                f"Orphan topic file: {drift.description[:80]}",
                "Add cross-references from related docs. Ensure every topic is discoverable.",
            )
        else:
            return (
                f"Knowledge drift: {drift.description[:80]}",
                "Review and update knowledge base to reflect current reality.",
            )

    def __repr__(self) -> str:
        return f"KnowledgeCuratorCitizen(drifts={len(self._drifts)})"
