"""Identity anchor — drift detection for Forge identity changes.

Ensures Forge cannot drift the system's identity beyond acceptable bounds.
Loads constraints from a YAML anchor definition and compares proposed changes
against immutable fields, core values, and a maximum change threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Defaults used when anchor file is missing or incomplete
_DEFAULT_IMMUTABLE_FIELDS = ["CORE_VALUES.md"]
_DEFAULT_MAX_CHANGE_THRESHOLD = 0.20
_DEFAULT_CORE_VALUES = ["sovereignty", "transparency", "safety"]


@dataclass
class DriftResult:
    """Result of a drift check against the identity anchor."""

    within_bounds: bool
    drift_score: float
    violations: list[str] = field(default_factory=list)


class IdentityAnchor:
    """Drift detector for identity changes.

    Loads an anchor YAML defining immutable fields, core values, and a
    maximum change threshold. Proposed changes are scored against these
    constraints — if drift_score exceeds the threshold, the change is
    rejected.

    Args:
        anchor_path: Path to the anchor YAML definition.
                     Defaults to forge/identity_anchor.yaml.
    """

    def __init__(self, anchor_path: str | Path = "forge/identity_anchor.yaml"):
        self._anchor_path = Path(anchor_path)
        self._immutable_fields: list[str] = list(_DEFAULT_IMMUTABLE_FIELDS)
        self._max_change_threshold: float = _DEFAULT_MAX_CHANGE_THRESHOLD
        self._core_values: list[str] = list(_DEFAULT_CORE_VALUES)
        self._load_anchor()

    @property
    def immutable_fields(self) -> list[str]:
        """Fields that cannot be modified."""
        return list(self._immutable_fields)

    @property
    def max_change_threshold(self) -> float:
        """Maximum allowed drift score (0.0-1.0)."""
        return self._max_change_threshold

    @property
    def core_values(self) -> list[str]:
        """Core values that must be preserved in all changes."""
        return list(self._core_values)

    def check_drift(self, proposed_changes: dict[str, Any]) -> DriftResult:
        """Compare proposed changes against anchor constraints.

        Scoring:
        - Each immutable field violation adds 0.5 to drift_score.
        - Each missing core value adds 0.15 to drift_score.
        - Base drift from change volume: len(changes) * 0.05, capped at 0.5.
        - drift_score is clamped to [0.0, 1.0].

        Args:
            proposed_changes: Dict of field_name -> new_value.

        Returns:
            DriftResult with within_bounds, drift_score, and violations.
        """
        violations: list[str] = []
        drift_score = 0.0

        if not proposed_changes:
            return DriftResult(within_bounds=True, drift_score=0.0, violations=[])

        # Check immutable fields
        for field_name in self._immutable_fields:
            if field_name in proposed_changes:
                violations.append(f"Immutable field modification attempted: {field_name}")
                drift_score += 0.5

        # Check core values preservation
        all_values_text = " ".join(
            str(v).lower() for v in proposed_changes.values()
        )
        for value in self._core_values:
            if value.lower() not in all_values_text:
                # Only flag if the change touches value-related fields
                value_fields = [
                    k for k in proposed_changes
                    if "value" in k.lower() or "principle" in k.lower() or "core" in k.lower()
                ]
                if value_fields:
                    violations.append(f"Core value may be lost: {value}")
                    drift_score += 0.15

        # Base drift from change volume
        volume_drift = min(len(proposed_changes) * 0.05, 0.5)
        drift_score += volume_drift

        # Clamp to [0.0, 1.0]
        drift_score = max(0.0, min(1.0, drift_score))

        within_bounds = drift_score <= self._max_change_threshold

        if not within_bounds:
            logger.warning(
                "Identity drift detected: score=%.2f threshold=%.2f violations=%d",
                drift_score,
                self._max_change_threshold,
                len(violations),
            )

        return DriftResult(
            within_bounds=within_bounds,
            drift_score=drift_score,
            violations=violations,
        )

    def _load_anchor(self) -> None:
        """Load anchor definition from YAML."""
        if not self._anchor_path.exists():
            logger.info(
                "Anchor file not found at %s, using defaults", self._anchor_path
            )
            return

        try:
            data = yaml.safe_load(self._anchor_path.read_text())
            if not isinstance(data, dict):
                logger.warning("Anchor YAML is not a mapping, using defaults")
                return

            if "immutable_fields" in data:
                self._immutable_fields = list(data["immutable_fields"])
            if "max_change_threshold" in data:
                self._max_change_threshold = float(data["max_change_threshold"])
            if "core_values" in data:
                self._core_values = list(data["core_values"])

            logger.info(
                "Loaded identity anchor: %d immutable fields, threshold=%.2f, %d core values",
                len(self._immutable_fields),
                self._max_change_threshold,
                len(self._core_values),
            )
        except (yaml.YAMLError, TypeError, ValueError):
            logger.warning(
                "Failed to parse anchor YAML at %s, using defaults",
                self._anchor_path,
            )
