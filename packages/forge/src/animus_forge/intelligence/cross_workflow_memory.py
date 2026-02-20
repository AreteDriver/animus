"""Cross-workflow learning memory system.

Wraps AgentMemory to provide persistent, compounding knowledge across
workflow executions. Agents accumulate learnings, build profiles, and
receive contextually relevant memories when starting new tasks.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from animus_forge.state.memory import AgentMemory, MemoryEntry

logger = logging.getLogger(__name__)

# Prefix used for agent_id to namespace cross-workflow entries by role.
_ROLE_PREFIX = "role:"

# Stop words excluded from pattern detection.
_STOP_WORDS: set[str] = {
    "the",
    "a",
    "an",
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
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "need",
    "must",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "about",
    "than",
    "after",
    "before",
    "during",
    "without",
    "between",
    "through",
    "and",
    "but",
    "or",
    "nor",
    "not",
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
    "no",
    "only",
    "own",
    "same",
    "that",
    "this",
    "it",
    "its",
    "i",
    "we",
    "they",
    "he",
    "she",
    "you",
    "me",
    "us",
    "them",
    "my",
    "our",
    "their",
    "his",
    "her",
    "your",
}


@dataclass
class AgentProfile:
    """Accumulated profile for an agent role across all workflows.

    Attributes:
        agent_role: The role identifier (e.g. "reviewer", "planner").
        total_executions: Number of recorded learning entries.
        success_rate: Fraction of learnings tagged as successful outcomes.
        top_learnings: Most important learned memory contents.
        common_patterns: Frequently recurring phrases in learnings.
        last_active: Timestamp of the most recent learning entry.
    """

    agent_role: str
    total_executions: int = 0
    success_rate: float = 0.0
    top_learnings: list[str] = field(default_factory=list)
    common_patterns: list[str] = field(default_factory=list)
    last_active: datetime | None = None


@dataclass
class Pattern:
    """A recurring phrase detected across an agent's learnings.

    Attributes:
        phrase: The recurring phrase or keyword.
        occurrences: How many memory entries contain this phrase.
        avg_importance: Mean importance score of entries containing it.
        agent_role: The agent role these entries belong to.
    """

    phrase: str
    occurrences: int
    avg_importance: float
    agent_role: str


class CrossWorkflowMemory:
    """Cross-workflow learning memory built on top of AgentMemory.

    Stores agent learnings as MemoryEntry objects with memory_type="learned"
    and workflow_id=None, making them globally accessible across workflows.
    Uses agent_id="role:<role_name>" to namespace entries by agent role.

    Args:
        memory: An existing AgentMemory instance to use as the backing store.
    """

    def __init__(self, memory: AgentMemory) -> None:
        self._memory = memory

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _agent_id_for_role(agent_role: str) -> str:
        """Return the namespaced agent_id for a given role.

        Args:
            agent_role: The agent role name.

        Returns:
            A prefixed identifier string.
        """
        return f"{_ROLE_PREFIX}{agent_role}"

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def record_learning(
        self,
        agent_role: str,
        insight: str,
        source_workflow_id: str | None = None,
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> int:
        """Store a learned insight that persists across workflows.

        Args:
            agent_role: Role of the agent that produced the insight
                (e.g. "reviewer", "planner").
            insight: Free-text description of what was learned.
            source_workflow_id: The workflow where the learning originated.
                Stored in metadata rather than as the entry's workflow_id so
                the memory remains globally retrievable.
            importance: Importance score between 0.0 and 1.0.
            tags: Optional list of categorisation tags.

        Returns:
            The id of the newly created memory entry.
        """
        importance = max(0.0, min(1.0, importance))
        agent_id = self._agent_id_for_role(agent_role)

        metadata: dict[str, Any] = {}
        if source_workflow_id:
            metadata["source_workflow_id"] = source_workflow_id
        if tags:
            metadata["tags"] = tags

        memory_id = self._memory.store(
            agent_id=agent_id,
            content=insight,
            memory_type="learned",
            workflow_id=None,
            metadata=metadata,
            importance=importance,
        )

        logger.info(
            "Recorded learning for %s (id=%s, importance=%.2f): %s",
            agent_role,
            memory_id,
            importance,
            insight[:120],
        )
        return memory_id

    def get_agent_profile(
        self,
        agent_role: str,
        top_k: int = 10,
    ) -> AgentProfile:
        """Build an accumulated profile for an agent role.

        Aggregates all cross-workflow learnings to summarise what the agent
        role is good at, its failure modes, and preferred patterns.

        Args:
            agent_role: The agent role to profile.
            top_k: Number of top learnings to include in the profile.

        Returns:
            An AgentProfile dataclass populated from stored memories.
        """
        agent_id = self._agent_id_for_role(agent_role)

        # Fetch all learned memories (large limit to build full profile).
        all_memories = self._memory.recall(
            agent_id=agent_id,
            memory_type="learned",
            limit=1000,
        )

        if not all_memories:
            return AgentProfile(agent_role=agent_role)

        total = len(all_memories)

        # Success rate: proportion of entries tagged with "success".
        success_count = sum(1 for m in all_memories if "success" in (m.metadata.get("tags") or []))
        success_rate = success_count / total if total else 0.0

        # Top learnings by importance.
        sorted_by_importance = sorted(all_memories, key=lambda m: m.importance, reverse=True)
        top_learnings = [m.content for m in sorted_by_importance[:top_k]]

        # Common patterns (top 5 phrases).
        patterns = self.detect_patterns(agent_role, min_occurrences=2)
        common_patterns = [p.phrase for p in patterns[:5]]

        # Last active timestamp.
        last_active: datetime | None = None
        for m in all_memories:
            ts = m.created_at or m.accessed_at
            if ts and (last_active is None or ts > last_active):
                last_active = ts

        profile = AgentProfile(
            agent_role=agent_role,
            total_executions=total,
            success_rate=round(success_rate, 3),
            top_learnings=top_learnings,
            common_patterns=common_patterns,
            last_active=last_active,
        )

        logger.debug(
            "Built profile for %s: %d executions, %.1f%% success rate",
            agent_role,
            total,
            success_rate * 100,
        )
        return profile

    def build_context_for_agent(
        self,
        agent_role: str,
        task_description: str,
        max_tokens: int = 2000,
    ) -> str:
        """Assemble relevant cross-workflow context for an agent task.

        Retrieves learned memories for the agent role, ranks them by a
        combined score of importance and recency, then formats the most
        relevant ones into a context string that fits within a token budget.

        Args:
            agent_role: The agent role requesting context.
            task_description: Description of the current task, used for
                keyword-based relevance boosting.
            max_tokens: Approximate token budget (estimated at 4 chars/token).

        Returns:
            A formatted context string ready for injection into a prompt.
        """
        agent_id = self._agent_id_for_role(agent_role)

        memories = self._memory.recall(
            agent_id=agent_id,
            memory_type="learned",
            limit=200,
        )

        if not memories:
            return ""

        # Extract task keywords for relevance matching.
        task_keywords = self._extract_keywords(task_description.lower())

        now = datetime.now(UTC)

        scored: list[tuple[float, MemoryEntry]] = []
        for mem in memories:
            # Base score from importance.
            score = mem.importance

            # Recency bonus: exponential decay with 30-day half-life.
            if mem.created_at:
                age_days = max((now - mem.created_at).total_seconds() / 86400, 0)
                recency = math.exp(-0.693 * age_days / 30)  # ln(2) ~ 0.693
            else:
                recency = 0.1
            score += 0.3 * recency

            # Keyword relevance bonus.
            content_lower = mem.content.lower()
            if task_keywords:
                overlap = sum(1 for kw in task_keywords if kw in content_lower)
                score += 0.2 * min(overlap / max(len(task_keywords), 1), 1.0)

            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Build context string within token budget.
        max_chars = max_tokens * 4
        lines: list[str] = ["## Cross-Workflow Learnings", ""]
        current_chars = sum(len(line) + 1 for line in lines)

        for rank, (score, mem) in enumerate(scored, start=1):
            tags = mem.metadata.get("tags", [])
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            line = f"{rank}. (importance={mem.importance:.2f}){tag_str} {mem.content}"

            line_chars = len(line) + 1  # +1 for newline
            if current_chars + line_chars > max_chars:
                break

            lines.append(line)
            current_chars += line_chars

        if len(lines) <= 2:
            # Only the header was added; nothing fits.
            return ""

        context = "\n".join(lines)
        logger.debug(
            "Built context for %s: %d entries, ~%d tokens",
            agent_role,
            len(lines) - 2,
            current_chars // 4,
        )
        return context

    def detect_patterns(
        self,
        agent_role: str,
        min_occurrences: int = 3,
    ) -> list[Pattern]:
        """Find recurring phrases in an agent's learned memories.

        Performs simple keyword and bigram frequency analysis across all
        learned entries for the given role. Excludes common stop words.

        Args:
            agent_role: The agent role to analyse.
            min_occurrences: Minimum number of memories a phrase must appear
                in to be considered a pattern.

        Returns:
            List of Pattern objects sorted by occurrence count descending.
        """
        agent_id = self._agent_id_for_role(agent_role)

        memories = self._memory.recall(
            agent_id=agent_id,
            memory_type="learned",
            limit=1000,
        )

        if not memories:
            return []

        # Count phrases (unigrams + bigrams) across entries.
        # Track per-entry presence (not raw frequency within an entry).
        phrase_entries: dict[str, list[float]] = {}

        for mem in memories:
            words = self._tokenize(mem.content)
            seen_phrases: set[str] = set()

            # Unigrams.
            for w in words:
                if w not in _STOP_WORDS and len(w) > 2:
                    seen_phrases.add(w)

            # Bigrams.
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i + 1]}"
                if words[i] not in _STOP_WORDS or words[i + 1] not in _STOP_WORDS:
                    if len(bigram) > 5:
                        seen_phrases.add(bigram)

            for phrase in seen_phrases:
                phrase_entries.setdefault(phrase, []).append(mem.importance)

        patterns: list[Pattern] = []
        for phrase, importances in phrase_entries.items():
            count = len(importances)
            if count >= min_occurrences:
                patterns.append(
                    Pattern(
                        phrase=phrase,
                        occurrences=count,
                        avg_importance=round(sum(importances) / len(importances), 3),
                        agent_role=agent_role,
                    )
                )

        patterns.sort(key=lambda p: p.occurrences, reverse=True)

        logger.debug(
            "Detected %d patterns for %s (min_occurrences=%d)",
            len(patterns),
            agent_role,
            min_occurrences,
        )
        return patterns

    def promote_memory(
        self,
        memory_id: int,
        new_importance: float,
    ) -> bool:
        """Boost the importance of a memory that proved useful.

        Args:
            memory_id: The id of the memory entry to promote.
            new_importance: The new importance score (clamped to 0.0-1.0).

        Returns:
            True if the memory was updated, False if the id was not found.
        """
        new_importance = max(0.0, min(1.0, new_importance))
        updated = self._memory.update_importance(memory_id, new_importance)

        if updated:
            logger.info(
                "Promoted memory %d to importance %.2f",
                memory_id,
                new_importance,
            )
        else:
            logger.warning("Failed to promote memory %d: not found", memory_id)

        return updated

    def decay_memories(
        self,
        half_life_days: float = 90,
        min_importance: float = 0.05,
        agent_roles: list[str] | None = None,
    ) -> int:
        """Apply time-based decay to reduce importance of old memories.

        Iterates over specified agent roles (or all known roles if none
        provided) and decays their learned memories based on time since
        last access. Memories whose importance falls below
        ``min_importance`` after decay are deleted entirely.

        Args:
            half_life_days: Number of days for importance to halve if a
                memory is never re-accessed.
            min_importance: Memories decayed below this threshold are deleted.
            agent_roles: Roles to process. If None, attempts to decay all
                role-prefixed entries via a direct backend query.

        Returns:
            Number of memories that were updated or removed.
        """
        now = datetime.now(UTC)
        decay_constant = 0.693147 / max(half_life_days, 0.1)  # ln(2) / half_life
        affected = 0

        entries: list[MemoryEntry] = []

        if agent_roles is None:
            # Query all learned memories across roles via the backend.
            try:
                rows = self._memory.backend.fetchall(
                    "SELECT * FROM agent_memories "
                    "WHERE memory_type = 'learned' AND agent_id LIKE ?",
                    (f"{_ROLE_PREFIX}%",),
                )
                entries = [MemoryEntry.from_dict(r) for r in rows]
            except Exception:
                logger.warning(
                    "Could not query learned memories for decay. Provide agent_roles explicitly."
                )
                return 0
        else:
            for role in agent_roles:
                agent_id = self._agent_id_for_role(role)
                entries.extend(
                    self._memory.recall(
                        agent_id=agent_id,
                        memory_type="learned",
                        limit=1000,
                    )
                )

        for entry in entries:
            if entry.id is None:
                continue

            reference_time = entry.accessed_at or entry.created_at
            if reference_time is None:
                continue

            age_days = max((now - reference_time).total_seconds() / 86400, 0)
            decay_factor = math.exp(-decay_constant * age_days)
            new_importance = entry.importance * decay_factor

            if new_importance < min_importance:
                self._memory.forget(
                    agent_id=entry.agent_id,
                    memory_id=entry.id,
                )
                affected += 1
                logger.debug(
                    "Removed decayed memory %d (importance %.4f)",
                    entry.id,
                    new_importance,
                )
            elif abs(new_importance - entry.importance) > 0.001:
                self._memory.update_importance(entry.id, round(new_importance, 4))
                affected += 1

        logger.info(
            "Decay pass complete: %d memories affected (half_life=%.0f days)",
            affected,
            half_life_days,
        )
        return affected

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split text into lowercase word tokens.

        Args:
            text: Input text.

        Returns:
            List of lowercase word strings.
        """
        return re.findall(r"[a-z][a-z0-9_]+", text.lower())

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract meaningful keywords from text.

        Args:
            text: Input text (should already be lowercased).

        Returns:
            Deduplicated list of keywords excluding stop words.
        """
        words = re.findall(r"[a-z][a-z0-9_]+", text)
        seen: set[str] = set()
        keywords: list[str] = []
        for w in words:
            if w not in _STOP_WORDS and len(w) > 2 and w not in seen:
                seen.add(w)
                keywords.append(w)
        return keywords
