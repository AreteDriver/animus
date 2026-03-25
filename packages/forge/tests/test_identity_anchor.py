"""Tests for the identity anchor drift detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from animus_forge.coordination.identity_anchor import (
    DriftResult,
    IdentityAnchor,
    _DEFAULT_CORE_VALUES,
    _DEFAULT_IMMUTABLE_FIELDS,
    _DEFAULT_MAX_CHANGE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def anchor_yaml(tmp_path: Path) -> Path:
    """Create a temporary anchor YAML."""
    p = tmp_path / "identity_anchor.yaml"
    p.write_text(
        "immutable_fields:\n"
        "  - CORE_VALUES.md\n"
        "  - mysoul.md\n"
        "max_change_threshold: 0.25\n"
        "core_values:\n"
        "  - sovereignty\n"
        "  - transparency\n"
        "  - safety\n"
    )
    return p


@pytest.fixture()
def anchor(anchor_yaml: Path) -> IdentityAnchor:
    return IdentityAnchor(anchor_path=anchor_yaml)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadAnchor:
    """Tests for anchor YAML loading."""

    def test_loads_from_yaml(self, anchor: IdentityAnchor) -> None:
        assert "CORE_VALUES.md" in anchor.immutable_fields
        assert "mysoul.md" in anchor.immutable_fields
        assert anchor.max_change_threshold == 0.25
        assert "sovereignty" in anchor.core_values

    def test_missing_file_uses_defaults(self, tmp_path: Path) -> None:
        anchor = IdentityAnchor(anchor_path=tmp_path / "nonexistent.yaml")
        assert anchor.immutable_fields == _DEFAULT_IMMUTABLE_FIELDS
        assert anchor.max_change_threshold == _DEFAULT_MAX_CHANGE_THRESHOLD
        assert anchor.core_values == _DEFAULT_CORE_VALUES

    def test_invalid_yaml_uses_defaults(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("!@#$%^&*(")
        anchor = IdentityAnchor(anchor_path=bad)
        assert anchor.immutable_fields == _DEFAULT_IMMUTABLE_FIELDS

    def test_non_mapping_yaml_uses_defaults(self, tmp_path: Path) -> None:
        bad = tmp_path / "list.yaml"
        bad.write_text("- item1\n- item2\n")
        anchor = IdentityAnchor(anchor_path=bad)
        assert anchor.immutable_fields == _DEFAULT_IMMUTABLE_FIELDS


class TestDriftDetection:
    """Tests for check_drift()."""

    def test_empty_changes_within_bounds(self, anchor: IdentityAnchor) -> None:
        result = anchor.check_drift({})
        assert result.within_bounds is True
        assert result.drift_score == 0.0
        assert result.violations == []

    def test_small_change_within_bounds(self, anchor: IdentityAnchor) -> None:
        result = anchor.check_drift({"LEARNED.md": "New learning"})
        assert result.within_bounds is True
        assert result.drift_score <= anchor.max_change_threshold

    def test_immutable_field_violation(self, anchor: IdentityAnchor) -> None:
        result = anchor.check_drift({"CORE_VALUES.md": "Modified values"})
        assert result.within_bounds is False
        assert result.drift_score >= 0.5
        assert any("Immutable field" in v for v in result.violations)

    def test_mysoul_immutable(self, anchor: IdentityAnchor) -> None:
        result = anchor.check_drift({"mysoul.md": "New soul"})
        assert result.within_bounds is False
        assert any("mysoul.md" in v for v in result.violations)

    def test_multiple_immutable_violations(self, anchor: IdentityAnchor) -> None:
        result = anchor.check_drift({
            "CORE_VALUES.md": "x",
            "mysoul.md": "y",
        })
        assert result.within_bounds is False
        assert result.drift_score >= 1.0
        assert len([v for v in result.violations if "Immutable" in v]) == 2


class TestCoreValues:
    """Tests for core value preservation."""

    def test_core_value_missing_from_value_field(
        self, anchor: IdentityAnchor
    ) -> None:
        # Changing a "values" field without mentioning core values
        result = anchor.check_drift({
            "core_values_config": "profit, speed, growth"
        })
        assert any("Core value may be lost" in v for v in result.violations)

    def test_core_values_preserved(self, anchor: IdentityAnchor) -> None:
        result = anchor.check_drift({
            "core_values_config": "sovereignty, transparency, safety, and more"
        })
        # All core values present — no violations from missing values
        value_violations = [v for v in result.violations if "Core value" in v]
        assert len(value_violations) == 0


class TestThreshold:
    """Tests for threshold enforcement."""

    def test_high_volume_changes_exceed_threshold(self, tmp_path: Path) -> None:
        """Many fields changed should push drift_score up."""
        p = tmp_path / "strict.yaml"
        p.write_text(
            "immutable_fields: []\n"
            "max_change_threshold: 0.10\n"
            "core_values: []\n"
        )
        anchor = IdentityAnchor(anchor_path=p)
        # 10 fields * 0.05 = 0.50 volume drift, well above 0.10
        changes = {f"field_{i}": f"value_{i}" for i in range(10)}
        result = anchor.check_drift(changes)
        assert result.within_bounds is False
        assert result.drift_score > 0.10

    def test_custom_threshold(self, tmp_path: Path) -> None:
        p = tmp_path / "lenient.yaml"
        p.write_text(
            "immutable_fields: []\n"
            "max_change_threshold: 0.90\n"
            "core_values: []\n"
        )
        anchor = IdentityAnchor(anchor_path=p)
        changes = {f"field_{i}": f"value_{i}" for i in range(5)}
        result = anchor.check_drift(changes)
        assert result.within_bounds is True


class TestDriftResult:
    """Tests for DriftResult dataclass."""

    def test_drift_result_fields(self) -> None:
        result = DriftResult(
            within_bounds=False,
            drift_score=0.75,
            violations=["test violation"],
        )
        assert result.within_bounds is False
        assert result.drift_score == 0.75
        assert result.violations == ["test violation"]

    def test_drift_result_default_violations(self) -> None:
        result = DriftResult(within_bounds=True, drift_score=0.0)
        assert result.violations == []

    def test_drift_score_clamped(self, anchor: IdentityAnchor) -> None:
        """Drift score should never exceed 1.0."""
        result = anchor.check_drift({
            "CORE_VALUES.md": "x",
            "mysoul.md": "y",
            "extra1": "a",
            "extra2": "b",
            "extra3": "c",
            "extra4": "d",
            "extra5": "e",
        })
        assert result.drift_score <= 1.0
