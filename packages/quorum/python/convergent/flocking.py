"""Flocking — swarm coordination through local rules.

Three bio-inspired rules adapted for AI agents:

- **Alignment**: Agents working on related code should use consistent
  patterns and style. Reads stigmergy ``pattern_found`` markers.
- **Cohesion**: Agents should stay focused on the overall goal, not drift.
  Measures keyword overlap between task description and current work.
- **Separation**: Agents should not step on each other's work.
  Reads stigmergy ``file_modified`` markers to detect overlapping edits.

Together these produce emergent group coordination from simple local rules,
just like birds flocking or fish schooling.
"""

from __future__ import annotations

import logging
import re

from convergent.stigmergy import StigmergyField

logger = logging.getLogger(__name__)

# Common stop words excluded from keyword extraction
_STOP_WORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "can",
        "could",
        "of",
        "in",
        "to",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "and",
        "or",
        "but",
        "not",
        "no",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "than",
        "too",
        "very",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "they",
        "them",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "when",
        "where",
        "why",
        "if",
        "then",
        "else",
    ]
)

_WORD_RE = re.compile(r"[a-z][a-z0-9_]*", re.IGNORECASE)


class FlockingCoordinator:
    """Swarm coordination using alignment, cohesion, and separation rules.

    Args:
        stigmergy: StigmergyField for reading trail markers.
        separation_threshold: Minimum marker strength to consider a file
            as "actively being modified" by another agent.
    """

    def __init__(
        self,
        stigmergy: StigmergyField,
        separation_threshold: float = 0.3,
    ) -> None:
        self._stigmergy = stigmergy
        self._separation_threshold = separation_threshold

    def check_alignment(self, agent_id: str, file_paths: list[str]) -> list[str]:
        """Find style/pattern constraints from other agents' markers.

        Reads ``pattern_found`` markers on the given files and returns
        guidance strings the agent should follow for consistency.

        Args:
            agent_id: The agent requesting alignment info.
            file_paths: Files the agent is about to work on.

        Returns:
            List of pattern/style constraint strings.
        """
        constraints: list[str] = []
        seen: set[str] = set()

        for path in file_paths:
            markers = self._stigmergy.get_markers(path)
            for m in markers:
                if m.marker_type != "pattern_found":
                    continue
                if m.agent_id == agent_id:
                    continue  # Skip own markers
                if m.content in seen:
                    continue
                seen.add(m.content)
                constraints.append(m.content)

        return constraints

    def check_cohesion(self, task_description: str, current_work: str) -> float:
        """Measure how much the current work drifts from the task scope.

        Uses keyword overlap between the task description and current work
        description. A high drift score means the agent is wandering.

        Args:
            task_description: The original task the agent was assigned.
            current_work: Description of what the agent is currently doing.

        Returns:
            Drift score from 0.0 (perfectly on-task) to 1.0 (completely off-task).
        """
        task_keywords = _extract_keywords(task_description)
        work_keywords = _extract_keywords(current_work)

        if not task_keywords or not work_keywords:
            return 0.0  # Can't measure drift without keywords

        # Jaccard-like overlap: shared keywords / union of keywords
        task_set = set(task_keywords)
        work_set = set(work_keywords)
        overlap = len(task_set & work_set)
        union = len(task_set | work_set)

        if union == 0:
            return 0.0

        similarity = overlap / union
        # Drift is inverse of similarity
        return round(1.0 - similarity, 4)

    def check_separation(self, agent_id: str, file_paths: list[str]) -> list[str]:
        """Detect files that another agent is currently modifying.

        Reads ``file_modified`` markers to find potential conflicts.
        Only considers markers above the separation threshold strength
        (weak/old markers are ignored as the other agent likely moved on).

        Args:
            agent_id: The agent requesting separation info.
            file_paths: Files the agent wants to work on.

        Returns:
            List of file paths that have active modifications by other agents.
        """
        conflicts: list[str] = []

        for path in file_paths:
            markers = self._stigmergy.get_markers(path)
            for m in markers:
                if m.marker_type != "file_modified":
                    continue
                if m.agent_id == agent_id:
                    continue  # Own markers don't count
                if m.strength >= self._separation_threshold:
                    conflicts.append(path)
                    break  # One conflict per file is enough

        return conflicts

    def generate_constraints(
        self,
        agent_id: str,
        task_description: str,
        current_work: str,
        file_paths: list[str],
    ) -> str:
        """Combine all three flocking rules into a constraint block.

        This is the main integration point: call this before an agent
        starts work and inject the result into the agent's prompt.

        Args:
            agent_id: The agent to generate constraints for.
            task_description: The original task description.
            current_work: What the agent is currently doing.
            file_paths: Files the agent plans to work on.

        Returns:
            Multi-line constraint string, or empty string if no constraints.
        """
        sections: list[str] = []

        # Alignment
        patterns = self.check_alignment(agent_id, file_paths)
        if patterns:
            lines = ["### Alignment (follow these patterns)"]
            for p in patterns:
                lines.append(f"- {p}")
            sections.append("\n".join(lines))

        # Cohesion
        drift = self.check_cohesion(task_description, current_work)
        if drift > 0.5:
            sections.append(
                f"### Cohesion Warning (drift={drift:.2f})\n"
                f"Your current work may be drifting from the original task. "
                f"Re-focus on: {task_description}"
            )

        # Separation
        conflicts = self.check_separation(agent_id, file_paths)
        if conflicts:
            lines = ["### Separation (avoid these files)"]
            for f in conflicts:
                lines.append(f"- `{f}` is being modified by another agent")
            sections.append("\n".join(lines))

        if not sections:
            return ""

        return "## Flocking Constraints\n\n" + "\n\n".join(sections)


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text.

    Lowercases, tokenizes, removes stop words and short tokens.

    Args:
        text: Input text.

    Returns:
        List of keyword strings.
    """
    words = _WORD_RE.findall(text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 2]
