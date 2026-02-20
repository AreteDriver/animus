"""
Preference Inference and Application System

Manages user preferences inferred from patterns.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from animus.logging import get_logger

if TYPE_CHECKING:
    from animus.learning.patterns import DetectedPattern

logger = get_logger("learning.preferences")


@dataclass
class Preference:
    """A user preference that affects AI behavior."""

    id: str
    domain: str  # communication, scheduling, tools, workflow, etc.
    key: str
    value: str
    confidence: float
    source_patterns: list[str]  # Pattern IDs
    created_at: datetime
    last_applied: datetime | None = None
    application_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        domain: str,
        key: str,
        value: str,
        confidence: float,
        source_patterns: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> "Preference":
        """Create a new Preference with generated ID and timestamp."""
        return cls(
            id=str(uuid.uuid4()),
            domain=domain,
            key=key,
            value=value,
            confidence=confidence,
            source_patterns=source_patterns,
            created_at=datetime.now(),
            metadata=metadata or {},
        )

    def apply(self) -> None:
        """Record that this preference was applied."""
        self.last_applied = datetime.now()
        self.application_count += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "domain": self.domain,
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "source_patterns": self.source_patterns,
            "created_at": self.created_at.isoformat(),
            "last_applied": (self.last_applied.isoformat() if self.last_applied else None),
            "application_count": self.application_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Preference":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            domain=data["domain"],
            key=data["key"],
            value=data["value"],
            confidence=data["confidence"],
            source_patterns=data["source_patterns"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_applied=(
                datetime.fromisoformat(data["last_applied"]) if data.get("last_applied") else None
            ),
            application_count=data.get("application_count", 0),
            metadata=data.get("metadata", {}),
        )


class PreferenceEngine:
    """
    Manages user preferences inferred from patterns.

    Preferences are organized by domain and applied contextually:
    - communication: tone, verbosity, formality
    - scheduling: preferred times, durations
    - tools: preferred tools, default parameters
    - workflow: default behaviors, shortcuts
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._preferences: dict[str, Preference] = {}
        self._load_preferences()

    def _load_preferences(self) -> None:
        """Load preferences from disk."""
        prefs_file = self.data_dir / "preferences.json"
        if prefs_file.exists():
            try:
                with open(prefs_file) as f:
                    data = json.load(f)
                for item in data:
                    pref = Preference.from_dict(item)
                    self._preferences[pref.id] = pref
                logger.info(f"Loaded {len(data)} preferences")
            except (json.JSONDecodeError, ValueError, OSError) as e:
                logger.error(f"Failed to load preferences: {e}")

    def _save_preferences(self) -> None:
        """Save preferences to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        prefs_file = self.data_dir / "preferences.json"
        data = [p.to_dict() for p in self._preferences.values()]
        with open(prefs_file, "w") as f:
            json.dump(data, f, indent=2)

    def infer_from_pattern(self, pattern: "DetectedPattern") -> Preference | None:
        """
        Infer a preference from a detected pattern.

        Args:
            pattern: The detected pattern

        Returns:
            A new Preference if one can be inferred, None otherwise
        """
        from animus.learning.patterns import PatternType

        # Map pattern types to domains
        domain_map = {
            PatternType.PREFERENCE: "communication",
            PatternType.TEMPORAL: "scheduling",
            PatternType.FREQUENCY: "workflow",
            PatternType.CORRECTION: "communication",
            PatternType.SEQUENTIAL: "workflow",
            PatternType.CONTEXTUAL: "general",
        }

        domain = domain_map.get(pattern.pattern_type, "general")

        # Refine domain based on content keywords
        desc_lower = pattern.description.lower()
        if any(w in desc_lower for w in ("time", "morning", "evening", "schedule", "hour")):
            domain = "scheduling"
        elif any(w in desc_lower for w in ("tone", "formal", "casual", "verbose", "brief")):
            domain = "communication"
        elif any(w in desc_lower for w in ("tool", "command", "file", "editor")):
            domain = "tools"

        # Extract key and value from pattern description
        description = pattern.description
        if ":" in description:
            parts = description.split(":", 1)
            key = parts[0].strip().lower().replace(" ", "_")
            value = parts[1].strip()
        else:
            key = pattern.pattern_type.value
            value = description

        # Check for existing preference with same key
        existing = self._find_by_key(domain, key)
        if existing:
            # Update confidence if this reinforces existing preference
            existing.confidence = min(1.0, existing.confidence + 0.1)
            existing.source_patterns.append(pattern.id)
            self._save_preferences()
            logger.info(f"Reinforced preference: {existing.id}")
            return existing

        # Create new preference
        preference = Preference.create(
            domain=domain,
            key=key,
            value=value,
            confidence=pattern.confidence,
            source_patterns=[pattern.id],
        )

        self._preferences[preference.id] = preference
        self._save_preferences()
        logger.info(f"Created preference: {preference.id} ({domain}/{key})")
        return preference

    def _find_by_key(self, domain: str, key: str) -> Preference | None:
        """Find existing preference by domain and key."""
        for pref in self._preferences.values():
            if pref.domain == domain and pref.key == key:
                return pref
        return None

    def get_preferences(self, domain: str | None = None) -> list[Preference]:
        """
        Get active preferences, optionally filtered by domain.

        Args:
            domain: Optional domain to filter by

        Returns:
            List of preferences
        """
        if domain is None:
            return list(self._preferences.values())
        return [p for p in self._preferences.values() if p.domain == domain]

    def get_preference(self, preference_id: str) -> Preference | None:
        """Get a specific preference by ID."""
        return self._preferences.get(preference_id)

    def apply_to_context(self, context: dict[str, Any], domain: str) -> dict[str, Any]:
        """
        Apply relevant preferences to a context dict.

        Args:
            context: The context to modify
            domain: The domain of preferences to apply

        Returns:
            Modified context with preferences applied
        """
        preferences = self.get_preferences(domain)
        modified = context.copy()

        for pref in preferences:
            if pref.confidence >= 0.6:  # Only apply confident preferences
                pref.apply()
                # Add preference hint to context
                if "preferences" not in modified:
                    modified["preferences"] = []
                modified["preferences"].append(
                    {"key": pref.key, "value": pref.value, "confidence": pref.confidence}
                )

        self._save_preferences()
        return modified

    def update_confidence(self, preference_id: str, delta: float) -> bool:
        """
        Update preference confidence based on feedback.

        Args:
            preference_id: ID of preference to update
            delta: Amount to adjust confidence (-1.0 to 1.0)

        Returns:
            True if updated, False if not found
        """
        pref = self._preferences.get(preference_id)
        if pref:
            pref.confidence = max(0.0, min(1.0, pref.confidence + delta))
            self._save_preferences()
            return True
        return False

    def remove_preference(self, preference_id: str) -> bool:
        """
        Remove a preference.

        Args:
            preference_id: ID of preference to remove

        Returns:
            True if removed, False if not found
        """
        if preference_id in self._preferences:
            del self._preferences[preference_id]
            self._save_preferences()
            logger.info(f"Removed preference: {preference_id}")
            return True
        return False

    def get_statistics(self) -> dict[str, Any]:
        """Get preference statistics."""
        by_domain: dict[str, int] = {}
        total_applications = 0
        avg_confidence = 0.0

        for pref in self._preferences.values():
            by_domain[pref.domain] = by_domain.get(pref.domain, 0) + 1
            total_applications += pref.application_count
            avg_confidence += pref.confidence

        count = len(self._preferences)
        if count > 0:
            avg_confidence /= count

        return {
            "total": count,
            "by_domain": by_domain,
            "total_applications": total_applications,
            "avg_confidence": avg_confidence,
        }
