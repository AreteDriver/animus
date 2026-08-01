"""Rubric data model: dimensions, criteria, and scored results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ScoreLevel(int, Enum):
    """Discrete score levels for rubric dimensions.

    Using 5-level scale (per ATLAS) instead of 0–10 continuous.
    Discrete levels reduce judge variance and improve calibration.
    Inherits from int to support comparison operators.
    """

    CRITICAL = 1  # Completely misses the mark
    POOR = 2  # Major issues
    ACCEPTABLE = 3  # Meets basic requirements
    GOOD = 4  # Solid, minor issues
    EXCELLENT = 5  # Exceeds expectations


@dataclass
class Dimension:
    """One scoring dimension in a rubric.

    Each dimension has criteria descriptions for each score level,
    enabling consistent judging across outputs.
    """

    name: str
    description: str
    criteria: dict[ScoreLevel, str] = field(default_factory=dict)
    weight: float = 1.0  # Relative weight in aggregate score
    required: bool = False  # If True, CRITICAL = overall failure

    def __post_init__(self):
        # Validate criteria cover all levels
        for level in ScoreLevel:
            if level not in self.criteria:
                self.criteria[level] = f"{level.name.lower().replace('_', ' ')}: no criteria set"

    def format_criteria(self) -> str:
        """Format criteria as judge-readable text."""
        lines = [f"### {self.name}: {self.description}"]
        for level in ScoreLevel:
            lines.append(f"  {level.value} — {self.criteria[level]}")
        return "\n".join(lines)


@dataclass
class Rubric:
    """A complete evaluation rubric with multiple dimensions.

    Rubrics are task-specific. A "code review" rubric weights correctness
    and security heavily; a "creative writing" rubric weights originality
    and style.
    """

    name: str
    description: str
    dimensions: list[Dimension] = field(default_factory=list)
    version: str = "1.0"
    task_type: str = "general"

    @property
    def total_weight(self) -> float:
        return sum(d.weight for d in self.dimensions)

    def add_dimension(self, dimension: Dimension) -> Rubric:
        """Add a dimension and return self for chaining."""
        self.dimensions.append(dimension)
        return self

    def format_for_judge(self) -> str:
        """Format the full rubric for a judge model prompt."""
        lines = [
            f"# {self.name}",
            f"{self.description}",
            "",
            "Rate the output on each dimension using the 1–5 scale.",
            "Provide a brief justification for each score.",
            "",
        ]
        for dim in self.dimensions:
            lines.append(dim.format_criteria())
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to dict for storage."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "task_type": self.task_type,
            "dimensions": [
                {
                    "name": d.name,
                    "description": d.description,
                    "weight": d.weight,
                    "required": d.required,
                    "criteria": {str(level.value): desc for level, desc in d.criteria.items()},
                }
                for d in self.dimensions
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Rubric:
        """Deserialize from dict."""
        rubric = cls(
            name=data["name"],
            description=data["description"],
            version=data.get("version", "1.0"),
            task_type=data.get("task_type", "general"),
        )
        for dim_data in data.get("dimensions", []):
            criteria = {
                ScoreLevel(int(level)): desc for level, desc in dim_data.get("criteria", {}).items()
            }
            rubric.add_dimension(
                Dimension(
                    name=dim_data["name"],
                    description=dim_data["description"],
                    weight=dim_data.get("weight", 1.0),
                    required=dim_data.get("required", True),
                    criteria=criteria,
                )
            )
        return rubric


@dataclass
class DimensionScore:
    """Score for a single dimension."""

    dimension_name: str
    level: ScoreLevel
    raw_score: float  # Normalized 0.0–1.0
    justification: str = ""
    confidence: float = 1.0  # Judge confidence 0.0–1.0


@dataclass
class Score:
    """Complete rubric score for one output evaluation."""

    rubric_name: str
    dimension_scores: list[DimensionScore] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def weighted_score(self) -> float:
        """Aggregate score weighted by dimension weights."""
        if not self.dimension_scores:
            return 0.0
        total = 0.0
        weight_sum = 0.0
        for ds in self.dimension_scores:
            # Find weight from rubric metadata if available
            weight = self.metadata.get("weights", {}).get(ds.dimension_name, 1.0)
            total += ds.raw_score * weight
            weight_sum += weight
        return total / weight_sum if weight_sum > 0 else 0.0

    @property
    def overall_level(self) -> ScoreLevel:
        """Map weighted score to nearest ScoreLevel."""
        score = self.weighted_score
        if score >= 0.9:
            return ScoreLevel.EXCELLENT
        elif score >= 0.7:
            return ScoreLevel.GOOD
        elif score >= 0.5:
            return ScoreLevel.ACCEPTABLE
        elif score >= 0.3:
            return ScoreLevel.POOR
        return ScoreLevel.CRITICAL

    @property
    def has_critical_failure(self) -> bool:
        """True if any required dimension scored CRITICAL."""
        for ds in self.dimension_scores:
            if ds.level == ScoreLevel.CRITICAL:
                is_required = self.metadata.get("required_dims", set()).get(ds.dimension_name, True)
                if is_required:
                    return True
        return False

    def failures(self) -> list[DimensionScore]:
        """Return dimensions that scored CRITICAL or POOR."""
        return [ds for ds in self.dimension_scores if ds.level <= ScoreLevel.POOR]

    def strengths(self) -> list[DimensionScore]:
        """Return dimensions that scored GOOD or EXCELLENT."""
        return [ds for ds in self.dimension_scores if ds.level >= ScoreLevel.GOOD]

    def summary(self) -> str:
        """Human-readable summary of the score."""
        lines = [
            f"Rubric: {self.rubric_name}",
            f"Weighted score: {self.weighted_score:.2f}",
            f"Overall level: {self.overall_level.name}",
        ]
        if self.has_critical_failure:
            lines.append("⚠️ Critical failure detected on required dimension(s)")
        lines.append("")
        lines.append("Dimension breakdown:")
        for ds in self.dimension_scores:
            marker = (
                "✓"
                if ds.level >= ScoreLevel.GOOD
                else "⚠️"
                if ds.level == ScoreLevel.ACCEPTABLE
                else "✗"
            )
            lines.append(
                f"  {marker} {ds.dimension_name}: {ds.level.name} "
                f"({ds.raw_score:.2f}) — {ds.justification[:60]}"
            )
        return "\n".join(lines)
