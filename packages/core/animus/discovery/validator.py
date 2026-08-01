"""SchemaValidator: validates discovered tool schemas for quality and correctness.

Uses the P2 rubric-based evaluation infrastructure to score discovered tools
before registration. Prevents low-quality or malformed schemas from polluting
the ToolRegistry.

Validation dimensions:
- Completeness: required fields present (name, description, parameters)
- Correctness: JSON Schema is valid and parseable
- Clarity: description is actionable and specific
- Safety: no obvious injection vectors or dangerous patterns
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from animus.logging import get_logger

logger = get_logger("discovery.validator")


@dataclass
class ValidationResult:
    """Result of schema validation for a discovered tool."""

    tool_name: str
    passed: bool
    score: float  # 0.0–1.0 aggregate
    dimension_scores: dict[str, float] = None  # type: ignore[assignment]
    errors: list[str] = None  # type: ignore[assignment]
    warnings: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.dimension_scores is None:
            self.dimension_scores = {}
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class SchemaValidator:
    """Validates discovered tool schemas before registration.

    Usage:
        validator = SchemaValidator(min_score=0.6)
        result = validator.validate_tool_schema(tool_schema)
        if result.passed:
            registry.register(tool)
    """

    def __init__(self, min_score: float = 0.6):
        self.min_score = min_score

    def validate_tool_schema(self, schema: dict[str, Any]) -> ValidationResult:
        """Validate a single tool schema.

        Args:
            schema: Dict with keys: name, description, parameters

        Returns:
            ValidationResult with scores and errors.
        """
        errors: list[str] = []
        warnings: list[str] = []
        dimension_scores: dict[str, float] = {}

        name = schema.get("name", "")
        description = schema.get("description", "")
        parameters = schema.get("parameters", {})

        # Completeness check
        completeness_score = self._score_completeness(name, description, parameters, errors)
        dimension_scores["completeness"] = completeness_score

        # Correctness check
        correctness_score = self._score_correctness(parameters, errors)
        dimension_scores["correctness"] = correctness_score

        # Clarity check
        clarity_score = self._score_clarity(description, warnings)
        dimension_scores["clarity"] = clarity_score

        # Safety check
        safety_score = self._score_safety(schema, errors, warnings)
        dimension_scores["safety"] = safety_score

        # Aggregate score (weighted)
        aggregate = (
            completeness_score * 0.35
            + correctness_score * 0.30
            + clarity_score * 0.20
            + safety_score * 0.15
        )

        passed = aggregate >= self.min_score and len(errors) == 0

        return ValidationResult(
            tool_name=name or "unknown",
            passed=passed,
            score=round(aggregate, 3),
            dimension_scores=dimension_scores,
            errors=errors,
            warnings=warnings,
        )

    def validate_batch(
        self,
        schemas: list[dict[str, Any]],
    ) -> tuple[list[ValidationResult], list[ValidationResult]]:
        """Validate multiple schemas and partition into passed/failed.

        Returns:
            (passed_results, failed_results)
        """
        passed: list[ValidationResult] = []
        failed: list[ValidationResult] = []

        for schema in schemas:
            result = self.validate_tool_schema(schema)
            if result.passed:
                passed.append(result)
            else:
                failed.append(result)

        return passed, failed

    def _score_completeness(
        self,
        name: str,
        description: str,
        parameters: dict,
        errors: list[str],
    ) -> float:
        """Score schema completeness (0.0–1.0)."""
        checks = 0
        total = 4

        if name and isinstance(name, str) and len(name) > 0:
            checks += 1
        else:
            errors.append("Missing or empty tool name")

        if description and isinstance(description, str) and len(description) > 10:
            checks += 1
        else:
            errors.append("Missing or insufficient description (min 10 chars)")

        if parameters and isinstance(parameters, dict):
            checks += 1
            if "properties" in parameters:
                checks += 1
            else:
                errors.append("Parameters missing 'properties' key")
        else:
            errors.append("Missing or invalid parameters schema")

        return checks / total

    def _score_correctness(self, parameters: dict, errors: list[str]) -> float:
        """Score JSON Schema correctness (0.0–1.0)."""
        if not isinstance(parameters, dict):
            errors.append("Parameters is not a dict")
            return 0.0

        try:
            # Validate it's serializable
            json.dumps(parameters)
        except (TypeError, ValueError) as e:
            errors.append(f"Parameters not JSON serializable: {e}")
            return 0.0

        # Check for valid JSON Schema structure
        props = parameters.get("properties", {})
        if not isinstance(props, dict):
            errors.append("Parameters.properties is not a dict")
            return 0.0

        # Check each property has at least a type
        invalid_props = []
        for pname, pschema in props.items():
            if not isinstance(pschema, dict):
                invalid_props.append(pname)
                continue
            if "type" not in pschema:
                invalid_props.append(pname)

        if invalid_props:
            errors.append(f"Properties missing type: {', '.join(invalid_props)}")
            return max(0.0, 1.0 - len(invalid_props) * 0.2)

        return 1.0

    def _score_clarity(self, description: str, warnings: list[str]) -> float:
        """Score description clarity (0.0–1.0)."""
        if not description:
            return 0.0

        score = 1.0

        # Penalize very short descriptions
        if len(description) < 20:
            score -= 0.3
            warnings.append(f"Description very short ({len(description)} chars)")

        # Penalize generic descriptions
        generic_phrases = ["this tool", "a function", "utility", "helper"]
        lower = description.lower()
        for phrase in generic_phrases:
            if phrase in lower:
                score -= 0.1
                warnings.append(f"Description contains generic phrase: '{phrase}'")

        # Penalize missing action verb
        action_verbs = [
            "get",
            "set",
            "create",
            "delete",
            "update",
            "send",
            "fetch",
            "generate",
            "convert",
            "parse",
            "validate",
            "check",
        ]
        has_action = any(verb in lower for verb in action_verbs)
        if not has_action:
            score -= 0.15
            warnings.append("Description lacks action verb")

        return max(0.0, score)

    def _score_safety(
        self,
        schema: dict[str, Any],
        errors: list[str],
        warnings: list[str],
    ) -> float:
        """Score schema safety (0.0–1.0)."""
        score = 1.0
        name = schema.get("name", "")
        description = schema.get("description", "")
        combined = f"{name} {description}".lower()

        # Flag dangerous patterns
        dangerous = [
            "exec",
            "eval",
            "shell",
            "subprocess",
            "os.system",
            "rm -rf",
            "delete all",
            "drop table",
            "truncate",
        ]
        for pattern in dangerous:
            if pattern in combined:
                warnings.append(f"Potential dangerous pattern detected: '{pattern}'")
                score -= 0.2

        # Flag tools that claim broad access
        access_keywords = [
            "all files",
            "any file",
            "full access",
            "unrestricted",
            "root",
            "admin",
            "sudo",
        ]
        for keyword in access_keywords:
            if keyword in combined:
                warnings.append(f"Broad access claim detected: '{keyword}'")
                score -= 0.15

        return max(0.0, score)
