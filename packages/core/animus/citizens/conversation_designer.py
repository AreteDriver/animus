"""Citizen 002 — The Conversation Designer.

The permanent "interaction designer" of Animus.

Responsibilities:
- Observe conversation logs for repeated prompts and friction
- Analyze user intent patterns (vague requests, multi-step workflows)
- Detect cognitive overload (too many options, unclear next steps)
- Design better conversation flows, prompt templates, and shortcuts
- Propose NL interface improvements to reduce effort per turn

Never:
- Modify code directly
- Change conversation history
- Deploy new interfaces autonomously

Instead:
    Observe → Analyze → Design Proposal → Human Approval → Forge → Evidence → Merge
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from animus.citizens.architect import ArchitectCitizen, Observation
from animus.citizens.proposal import (
    EvidenceItem,
    ImprovementProposal,
    ProposalStatus,
    RiskAssessment,
)
from animus.logging import get_logger

logger = get_logger("citizens.conversation_designer")


@dataclass
class ConversationPattern:
    """A detected pattern in conversation logs."""

    pattern_type: str  # "repeated_prompt", "vague_request", "multi_step", "correction_loop"
    description: str
    frequency: int
    example: str
    suggestion: str
    severity: str = "low"  # "critical", "high", "medium", "low"


class ConversationDesignerCitizen:
    """Citizen 002 — The Conversation Designer.

    Continuously evaluates conversation quality and proposes
    improvements to the natural language interface.

    This citizen NEVER modifies code, memory, or systems directly.
    It only observes, analyzes, and produces proposals.
    """

    def __init__(
        self,
        conversation_log_dir: Path | str | None = None,
        memory_layer: Any = None,
    ):
        self.conversation_log_dir = (
            Path(conversation_log_dir).expanduser() if conversation_log_dir else None
        )
        self.memory = memory_layer
        self._patterns: list[ConversationPattern] = []

    # ------------------------------------------------------------------
    # Observation methods (read-only)
    # ------------------------------------------------------------------

    def observe_repeated_prompts(self, limit: int = 100) -> list[Observation]:
        """Observe conversation logs for repeated prompt patterns.

        Args:
            limit: Maximum number of recent conversations to analyze.

        Returns:
            List of observations.
        """
        observations: list[Observation] = []

        if not self.conversation_log_dir or not self.conversation_log_dir.exists():
            # No conversation logs configured yet — not an actionable finding.
            return observations

        prompt_counts: dict[str, int] = {}
        prompt_examples: dict[str, str] = {}

        for log_file in sorted(self.conversation_log_dir.glob("*.jsonl"))[-limit:]:
            try:
                for line in log_file.read_text().splitlines():
                    entry = json.loads(line)
                    prompt = entry.get("prompt", "").strip().lower()
                    # Normalize: strip punctuation, collapse whitespace
                    normalized = re.sub(r"[^\w\s]", "", prompt)
                    normalized = re.sub(r"\s+", " ", normalized).strip()
                    if len(normalized) > 10:
                        prompt_counts[normalized] = prompt_counts.get(normalized, 0) + 1
                        if normalized not in prompt_examples:
                            prompt_examples[normalized] = prompt[:100]
            except Exception:
                continue

        for prompt, count in sorted(prompt_counts.items(), key=lambda x: -x[1]):
            if count >= 3:
                severity = "high" if count >= 10 else "medium" if count >= 5 else "low"
                observations.append(
                    Observation(
                        source="conversation",
                        description=f"Repeated prompt detected ({count}×): {prompt[:60]}...",
                        severity=severity,
                        context={
                            "count": count,
                            "example": prompt_examples.get(prompt, ""),
                            "pattern_type": "repeated_prompt",
                        },
                    )
                )

        return observations

    def observe_vague_requests(self, limit: int = 100) -> list[Observation]:
        """Identify vague or underspecified user requests.

        These are prompts that lack specificity and require
        clarification, causing extra turns.

        Args:
            limit: Maximum conversations to analyze.

        Returns:
            List of observations.
        """
        observations: list[Observation] = []

        if not self.conversation_log_dir or not self.conversation_log_dir.exists():
            return observations

        vague_indicators = [
            r"\bhelp\b",
            r"\bfix\b",
            r"\bimprove\b",
            r"\bmake\s+(it|this)\b",
            r"\bdo\s+(something|it)\b",
            r"\bwhat\s+should\s+i\b",
            r"\bhow\s+do\s+i\b",
            r"\bcan\s+you\b",
            r"^\s*(?:try|attempt|maybe|perhaps)",
        ]
        vague_pattern = re.compile("|".join(vague_indicators), re.IGNORECASE)

        vague_counts: dict[str, int] = {}

        for log_file in sorted(self.conversation_log_dir.glob("*.jsonl"))[-limit:]:
            try:
                for line in log_file.read_text().splitlines():
                    entry = json.loads(line)
                    prompt = entry.get("prompt", "").strip()
                    if len(prompt) < 20 and vague_pattern.search(prompt):
                        normalized = prompt.lower().strip()
                        vague_counts[normalized] = vague_counts.get(normalized, 0) + 1
            except Exception:
                continue

        for prompt, count in sorted(vague_counts.items(), key=lambda x: -x[1]):
            if count >= 2:
                observations.append(
                    Observation(
                        source="conversation",
                        description=f"Vague request detected ({count}×): '{prompt[:80]}'",
                        severity="medium",
                        context={
                            "count": count,
                            "pattern_type": "vague_request",
                            "suggestion": "Add clarifying prompts or provide template options",
                        },
                    )
                )

        return observations

    def observe_correction_loops(self, limit: int = 100) -> list[Observation]:
        """Detect cycles where user corrects the AI repeatedly.

        These indicate the initial response missed intent.

        Args:
            limit: Maximum conversations to analyze.

        Returns:
            List of observations.
        """
        observations: list[Observation] = []

        if not self.conversation_log_dir or not self.conversation_log_dir.exists():
            return observations

        correction_keywords = [
            "no,", "not quite", "that's not", "wrong", "incorrect",
            "i meant", "actually,", "wait,", "no —", "nope",
        ]

        for log_file in sorted(self.conversation_log_dir.glob("*.jsonl"))[-limit:]:
            try:
                entries = []
                for line in log_file.read_text().splitlines():
                    entries.append(json.loads(line))

                correction_count = 0
                for entry in entries:
                    prompt = entry.get("prompt", "").lower()
                    if any(kw in prompt for kw in correction_keywords):
                        correction_count += 1

                if correction_count >= 2:
                    observations.append(
                        Observation(
                            source="conversation",
                            description=f"Correction loop detected ({correction_count} corrections in session)",
                            severity="high" if correction_count >= 4 else "medium",
                            context={
                                "file": log_file.name,
                                "correction_count": correction_count,
                                "pattern_type": "correction_loop",
                            },
                        )
                    )
            except Exception:
                continue

        return observations

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def analyze(self) -> list[ConversationPattern]:
        """Analyze all observations and produce patterns.

        Returns:
            List of detected conversation patterns.
        """
        observations: list[Observation] = []
        observations.extend(self.observe_repeated_prompts())
        observations.extend(self.observe_vague_requests())
        observations.extend(self.observe_correction_loops())

        patterns: list[ConversationPattern] = []
        pattern_groups: dict[str, list[Observation]] = {}

        for obs in observations:
            pt = obs.context.get("pattern_type", "unknown") if obs.context else "unknown"
            pattern_groups.setdefault(pt, []).append(obs)

        for pattern_type, group in pattern_groups.items():
            total_freq = sum(o.context.get("count", 1) for o in group if o.context)
            highest = max(group, key=lambda o: o.context.get("count", 1) if o.context else 0)

            suggestion = self._suggest_for_pattern(pattern_type, highest)
            severity = self._aggregate_severity(group)

            patterns.append(
                ConversationPattern(
                    pattern_type=pattern_type,
                    description=highest.description,
                    frequency=total_freq,
                    example=highest.context.get("example", "") if highest.context else "",
                    suggestion=suggestion,
                    severity=severity,
                )
            )

        self._patterns = patterns
        return patterns

    # ------------------------------------------------------------------
    # Proposal generation
    # ------------------------------------------------------------------

    def generate_proposal(self) -> ImprovementProposal | None:
        """Generate an improvement proposal from conversation analysis.

        Returns:
            Improvement proposal, or None if no actionable findings.
        """
        patterns = self.analyze()

        if not patterns:
            logger.info("No conversation patterns detected — no proposal generated")
            return None

        # Focus on highest-severity pattern
        top = max(patterns, key=lambda p: {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(p.severity, 0))

        evidence = [
            EvidenceItem(
                source="conversation_analysis",
                description=f"{top.pattern_type}: {top.description} (freq={top.frequency})",
                data={"example": top.example, "frequency": top.frequency},
            )
        ]

        # Build problem and recommendation based on pattern type
        problem, recommendation = self._build_problem_recommendation(top)

        risks = [
            RiskAssessment(
                description="New prompts may not cover all user intents",
                severity="low",
                mitigation="A/B test with subset of users before full rollout",
                probability=0.4,
            ),
            RiskAssessment(
                description="Template additions may clutter the interface",
                severity="low",
                mitigation="Limit to top 5 most frequent patterns",
                probability=0.3,
            ),
        ]

        components = ["Mind"]
        if top.pattern_type == "repeated_prompt":
            components = ["Mind", "Society"]
        elif top.pattern_type == "correction_loop":
            components = ["Mind", "Factory"]

        proposal = ImprovementProposal(
            id=f"ADL-{datetime.now().strftime('%Y%m%d')}-{__import__('uuid').uuid4().hex[:6]}",
            title=f"Conversation Design: {problem[:50]}",
            problem=problem,
            evidence=evidence,
            root_cause="Identified through systematic conversation pattern analysis",
            recommendation=recommendation,
            alternatives_considered=["Status quo (no change)", "Manual user coaching"],
            expected_benefits="Reduced cognitive effort and faster task completion for users",
            potential_risks=risks,
            confidence_score=0.65,
            estimated_effort_hours=3.0,
            affected_components=components,
            evaluation_plan="Measure repeat-prompt frequency before/after + user satisfaction",
            rollback_plan="Remove new prompts/templates via single config change",
            success_metrics=["Repeat-prompt frequency reduced", "Correction loops reduced", "User turn count stable or lower"],
            status=ProposalStatus.DRAFT,
        )

        logger.info(f"Conversation Designer generated proposal {proposal.id}")
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
                tags=["conversation_designer", "proposal", proposal.status.value],
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

    @staticmethod
    def _suggest_for_pattern(pattern_type: str, obs: Observation) -> str:
        """Generate a suggestion for a given pattern type."""
        suggestions = {
            "repeated_prompt": (
                "Create a shortcut command or template for this frequent request. "
                "Consider adding it to the natural language command parser."
            ),
            "vague_request": (
                "Add clarifying questions or provide 2-3 structured options "
                "when the intent is ambiguous. Pre-fill common parameters."
            ),
            "correction_loop": (
                "Add a confirmation step before execution. Improve intent parsing "
                "to catch the user's actual goal on the first turn."
            ),
            "multi_step": (
                "Bundle this multi-step workflow into a single command with "
                "sensible defaults. Reduce required parameters."
            ),
        }
        return suggestions.get(pattern_type, "Review and improve the interaction flow.")

    @staticmethod
    def _aggregate_severity(observations: list[Observation]) -> str:
        """Aggregate severity from a group of observations."""
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        max_score = max(
            (severity_order.get(o.severity, 0) for o in observations),
            default=0,
        )
        for sev, score in severity_order.items():
            if score == max_score:
                return sev
        return "low"

    @staticmethod
    def _build_problem_recommendation(pattern: ConversationPattern) -> tuple[str, str]:
        """Build problem/recommendation pair from pattern."""
        if pattern.pattern_type == "repeated_prompt":
            return (
                f"Users repeatedly ask: '{pattern.description[:80]}' — "
                f"indicating a missing shortcut or command.",
                f"Add a dedicated command/template for this workflow. "
                f"Current workaround requires {pattern.frequency} manual prompts per session.",
            )
        elif pattern.pattern_type == "vague_request":
            return (
                f"Vague requests detected: '{pattern.description[:80]}' — "
                f"users don't know how to specify what they need.",
                "Add clarifying prompts or structured option menus for ambiguous intents.",
            )
        elif pattern.pattern_type == "correction_loop":
            return (
                f"Correction loops detected ({pattern.frequency} instances) — "
                f"AI misses user intent on first response.",
                "Add a confirmation step before action. Improve intent disambiguation.",
            )
        else:
            return (
                f"Conversation friction: {pattern.description[:80]}",
                "Review and improve the natural language interaction flow.",
            )

    def __repr__(self) -> str:
        return f"ConversationDesignerCitizen(patterns={len(self._patterns)})"