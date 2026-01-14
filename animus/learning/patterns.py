"""
Pattern Detection Engine

Identifies learnable behaviors from user interactions stored in memory.
"""

import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING

from animus.learning.categories import LearningCategory
from animus.logging import get_logger

if TYPE_CHECKING:
    from animus.memory import Memory, MemoryLayer

logger = get_logger("learning.patterns")


class PatternType(Enum):
    """Types of patterns that can be detected."""

    TEMPORAL = "temporal"  # Time-based patterns (e.g., morning routine)
    SEQUENTIAL = "sequential"  # A-then-B patterns
    FREQUENCY = "frequency"  # Repeated actions
    CONTEXTUAL = "contextual"  # Context-specific behaviors
    PREFERENCE = "preference"  # Expressed likes/dislikes
    CORRECTION = "correction"  # User corrections to AI behavior


@dataclass
class PatternSignal:
    """A signal that may indicate a pattern."""

    type: PatternType
    content: str
    memory_ids: list[str]
    timestamp: datetime
    strength: float = 0.0  # How strong this signal is (0.0-1.0)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "content": self.content,
            "memory_ids": self.memory_ids,
            "timestamp": self.timestamp.isoformat(),
            "strength": self.strength,
        }


@dataclass
class DetectedPattern:
    """A detected pattern ready for learning consideration."""

    id: str
    pattern_type: PatternType
    description: str
    occurrences: int
    confidence: float
    evidence: list[str]  # Memory IDs
    first_seen: datetime
    last_seen: datetime
    suggested_learning: str
    suggested_category: LearningCategory
    metadata: dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        pattern_type: PatternType,
        description: str,
        occurrences: int,
        confidence: float,
        evidence: list[str],
        first_seen: datetime,
        last_seen: datetime,
        suggested_learning: str,
        suggested_category: LearningCategory,
        metadata: dict | None = None,
    ) -> "DetectedPattern":
        """Create a new DetectedPattern with generated ID."""
        return cls(
            id=str(uuid.uuid4()),
            pattern_type=pattern_type,
            description=description,
            occurrences=occurrences,
            confidence=confidence,
            evidence=evidence,
            first_seen=first_seen,
            last_seen=last_seen,
            suggested_learning=suggested_learning,
            suggested_category=suggested_category,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "pattern_type": self.pattern_type.value,
            "description": self.description,
            "occurrences": self.occurrences,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "suggested_learning": self.suggested_learning,
            "suggested_category": self.suggested_category.value,
            "metadata": self.metadata,
        }


class PatternDetector:
    """
    Detects patterns in user behavior from memory.

    Runs in background mode, analyzing episodic memories to find:
    - Repeated actions/requests
    - Time-based patterns
    - Preference expressions
    - Workflow sequences
    """

    # Preference indicator phrases
    PREFERENCE_INDICATORS = {
        "positive": [
            r"i (?:like|love|prefer|enjoy|want)",
            r"(?:please|always) (?:use|do|include)",
            r"i'd (?:like|prefer|rather)",
            r"can you (?:always|usually)",
        ],
        "negative": [
            r"i (?:don't like|hate|dislike|avoid)",
            r"(?:please|never) (?:don't|avoid|skip)",
            r"i'd rather not",
            r"(?:stop|quit) (?:doing|using)",
        ],
    }

    # Correction indicators
    CORRECTION_INDICATORS = [
        r"(?:no|wrong|incorrect),?\s+(?:i meant|it should be|use)",
        r"that's not (?:right|correct|what i)",
        r"(?:actually|instead),?\s+(?:i want|use|do)",
        r"(?:please|can you) (?:fix|correct|change)",
    ]

    def __init__(
        self,
        memory: "MemoryLayer",
        min_occurrences: int = 3,
        min_confidence: float = 0.6,
        lookback_days: int = 30,
    ):
        self.memory = memory
        self.min_occurrences = min_occurrences
        self.min_confidence = min_confidence
        self.lookback_days = lookback_days
        self._detected_patterns: dict[str, DetectedPattern] = {}
        self._processed_memory_ids: set[str] = set()

    def scan_for_patterns(self) -> list[DetectedPattern]:
        """
        Scan memory for new patterns.

        Returns:
            List of newly detected patterns
        """
        logger.info("Starting pattern scan")

        # Get memories from lookback period
        cutoff = datetime.now() - timedelta(days=self.lookback_days)
        memories = self._get_recent_memories(cutoff)

        if not memories:
            logger.info("No memories to scan")
            return []

        # Collect signals from different detectors
        signals: list[PatternSignal] = []
        signals.extend(self._detect_frequency_patterns(memories))
        signals.extend(self._detect_preference_signals(memories))
        signals.extend(self._detect_corrections(memories))
        signals.extend(self._detect_temporal_patterns(memories))

        # Consolidate signals into patterns
        new_patterns = self._consolidate_signals(signals)

        # Store detected patterns
        for pattern in new_patterns:
            self._detected_patterns[pattern.id] = pattern

        logger.info(f"Detected {len(new_patterns)} new patterns")
        return new_patterns

    def _get_recent_memories(self, cutoff: datetime) -> list["Memory"]:
        """Get memories since cutoff date."""
        # Query for episodic memories (conversations, events)
        all_memories = []

        # Get episodic memories
        episodic = self.memory.recall(
            query="",
            memory_type=self.memory.MemoryType.EPISODIC,
            limit=1000,
        )
        all_memories.extend(episodic)

        # Get semantic memories (facts, preferences)
        semantic = self.memory.recall(
            query="",
            memory_type=self.memory.MemoryType.SEMANTIC,
            limit=500,
        )
        all_memories.extend(semantic)

        # Filter by date and exclude already processed
        filtered = [
            m
            for m in all_memories
            if m.created_at >= cutoff and m.id not in self._processed_memory_ids
        ]

        # Mark as processed
        for m in filtered:
            self._processed_memory_ids.add(m.id)

        return filtered

    def _detect_frequency_patterns(
        self, memories: list["Memory"]
    ) -> list[PatternSignal]:
        """Detect frequently repeated actions or requests."""
        signals: list[PatternSignal] = []

        # Extract action phrases from memory content
        action_counts: dict[str, list[str]] = defaultdict(list)

        for memory in memories:
            content = memory.content.lower()
            # Look for action patterns
            action_patterns = [
                r"(?:can you|please|could you)\s+(\w+(?:\s+\w+){0,3})",
                r"(?:i need|i want)\s+(?:to\s+)?(\w+(?:\s+\w+){0,3})",
                r"(?:help me|assist with)\s+(\w+(?:\s+\w+){0,3})",
            ]

            for pattern in action_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    normalized = match.strip()
                    if len(normalized) > 3:  # Filter out very short matches
                        action_counts[normalized].append(memory.id)

        # Create signals for frequent actions
        for action, memory_ids in action_counts.items():
            if len(memory_ids) >= self.min_occurrences:
                strength = min(1.0, len(memory_ids) / 10.0)  # Scale to 1.0
                signals.append(
                    PatternSignal(
                        type=PatternType.FREQUENCY,
                        content=f"Frequently requests: {action}",
                        memory_ids=memory_ids,
                        timestamp=datetime.now(),
                        strength=strength,
                    )
                )

        return signals

    def _detect_preference_signals(
        self, memories: list["Memory"]
    ) -> list[PatternSignal]:
        """Detect explicit preference expressions."""
        signals: list[PatternSignal] = []

        for memory in memories:
            content = memory.content.lower()

            # Check positive preferences
            for pattern in self.PREFERENCE_INDICATORS["positive"]:
                if re.search(pattern, content):
                    # Extract what they prefer
                    match = re.search(pattern + r"\s+(.+?)(?:\.|$)", content)
                    if match:
                        preference = match.group(1).strip()
                        signals.append(
                            PatternSignal(
                                type=PatternType.PREFERENCE,
                                content=f"Prefers: {preference}",
                                memory_ids=[memory.id],
                                timestamp=memory.created_at,
                                strength=0.8,
                            )
                        )

            # Check negative preferences
            for pattern in self.PREFERENCE_INDICATORS["negative"]:
                if re.search(pattern, content):
                    match = re.search(pattern + r"\s+(.+?)(?:\.|$)", content)
                    if match:
                        dislike = match.group(1).strip()
                        signals.append(
                            PatternSignal(
                                type=PatternType.PREFERENCE,
                                content=f"Dislikes: {dislike}",
                                memory_ids=[memory.id],
                                timestamp=memory.created_at,
                                strength=0.8,
                            )
                        )

        return signals

    def _detect_corrections(self, memories: list["Memory"]) -> list[PatternSignal]:
        """Detect user corrections to AI responses."""
        signals: list[PatternSignal] = []

        for memory in memories:
            content = memory.content.lower()

            for pattern in self.CORRECTION_INDICATORS:
                if re.search(pattern, content):
                    signals.append(
                        PatternSignal(
                            type=PatternType.CORRECTION,
                            content=f"Correction: {content[:100]}",
                            memory_ids=[memory.id],
                            timestamp=memory.created_at,
                            strength=0.9,  # High strength - user actively correcting
                        )
                    )
                    break  # One correction signal per memory

        return signals

    def _detect_temporal_patterns(
        self, memories: list["Memory"]
    ) -> list[PatternSignal]:
        """Detect time-of-day or day-of-week patterns."""
        signals: list[PatternSignal] = []

        # Group memories by hour of day
        hour_counts: dict[int, list[str]] = defaultdict(list)
        for memory in memories:
            hour = memory.created_at.hour
            hour_counts[hour].append(memory.id)

        # Detect peak activity hours
        for hour, memory_ids in hour_counts.items():
            if len(memory_ids) >= self.min_occurrences:
                # Determine time of day label
                if 5 <= hour < 12:
                    time_label = "morning"
                elif 12 <= hour < 17:
                    time_label = "afternoon"
                elif 17 <= hour < 21:
                    time_label = "evening"
                else:
                    time_label = "night"

                strength = min(1.0, len(memory_ids) / 15.0)
                signals.append(
                    PatternSignal(
                        type=PatternType.TEMPORAL,
                        content=f"Active during {time_label} ({hour}:00)",
                        memory_ids=memory_ids,
                        timestamp=datetime.now(),
                        strength=strength,
                    )
                )

        return signals

    def _consolidate_signals(
        self, signals: list[PatternSignal]
    ) -> list[DetectedPattern]:
        """Consolidate signals into patterns meeting threshold."""
        patterns: list[DetectedPattern] = []

        # Group signals by type and content similarity
        grouped: dict[str, list[PatternSignal]] = defaultdict(list)
        for signal in signals:
            # Create grouping key
            key = f"{signal.type.value}:{signal.content[:50]}"
            grouped[key].append(signal)

        for key, group in grouped.items():
            # Calculate aggregate metrics
            all_memory_ids = []
            total_strength = 0.0
            earliest = datetime.now()
            latest = datetime.min

            for signal in group:
                all_memory_ids.extend(signal.memory_ids)
                total_strength += signal.strength
                if signal.timestamp < earliest:
                    earliest = signal.timestamp
                if signal.timestamp > latest:
                    latest = signal.timestamp

            # Deduplicate memory IDs
            unique_memory_ids = list(set(all_memory_ids))
            occurrences = len(unique_memory_ids)
            avg_strength = total_strength / len(group)

            # Calculate confidence
            confidence = min(1.0, (occurrences / self.min_occurrences) * avg_strength)

            if confidence >= self.min_confidence:
                # Determine category and suggested learning
                pattern_type = group[0].type
                content = group[0].content
                category, learning = self._suggest_learning(pattern_type, content)

                pattern = DetectedPattern.create(
                    pattern_type=pattern_type,
                    description=content,
                    occurrences=occurrences,
                    confidence=confidence,
                    evidence=unique_memory_ids,
                    first_seen=earliest,
                    last_seen=latest,
                    suggested_learning=learning,
                    suggested_category=category,
                    metadata={"signal_count": len(group)},
                )
                patterns.append(pattern)

        return patterns

    def _suggest_learning(
        self, pattern_type: PatternType, content: str
    ) -> tuple[LearningCategory, str]:
        """Suggest what should be learned from a pattern."""
        if pattern_type == PatternType.PREFERENCE:
            if content.startswith("Prefers:"):
                return LearningCategory.PREFERENCE, content.replace("Prefers:", "User prefers")
            elif content.startswith("Dislikes:"):
                return LearningCategory.PREFERENCE, content.replace("Dislikes:", "User dislikes")

        elif pattern_type == PatternType.CORRECTION:
            return LearningCategory.STYLE, f"Adjust behavior based on: {content}"

        elif pattern_type == PatternType.FREQUENCY:
            return LearningCategory.WORKFLOW, f"Common request pattern: {content}"

        elif pattern_type == PatternType.TEMPORAL:
            return LearningCategory.PREFERENCE, f"Activity pattern: {content}"

        # Default
        return LearningCategory.FACT, f"Observed pattern: {content}"

    def get_detected_patterns(self) -> list[DetectedPattern]:
        """Get all detected patterns."""
        return list(self._detected_patterns.values())

    def clear_pattern(self, pattern_id: str) -> bool:
        """Remove a detected pattern."""
        if pattern_id in self._detected_patterns:
            del self._detected_patterns[pattern_id]
            return True
        return False
