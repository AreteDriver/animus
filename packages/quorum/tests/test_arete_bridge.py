"""Tests for Arete Tools → Quorum bridge (autopsy → stability score + stigmergy)."""

from __future__ import annotations

from unittest.mock import MagicMock

from convergent.arete_bridge import (
    FAILURE_SEVERITY,
    leave_autopsy_marker,
    record_failure_outcome,
)

# ─── FAILURE_SEVERITY mapping ─────────────────────────────────────────


class TestFailureSeverity:
    """Tests for the failure_type → outcome mapping."""

    def test_goal_necrosis_maps_to_failed(self):
        assert FAILURE_SEVERITY["goal_necrosis"] == "failed"

    def test_goal_cancer_maps_to_failed(self):
        assert FAILURE_SEVERITY["goal_cancer"] == "failed"

    def test_goal_autoimmunity_maps_to_failed(self):
        assert FAILURE_SEVERITY["goal_autoimmunity"] == "failed"

    def test_tool_hallucination_maps_to_rejected(self):
        assert FAILURE_SEVERITY["tool_hallucination"] == "rejected"

    def test_tool_loop_maps_to_rejected(self):
        assert FAILURE_SEVERITY["tool_loop"] == "rejected"

    def test_overconfidence_maps_to_rejected(self):
        assert FAILURE_SEVERITY["overconfidence"] == "rejected"

    def test_unknown_maps_to_rejected(self):
        assert FAILURE_SEVERITY["unknown"] == "rejected"

    def test_all_values_are_valid_outcomes(self):
        valid = {"failed", "rejected", "approved"}
        for outcome in FAILURE_SEVERITY.values():
            assert outcome in valid


# ─── record_failure_outcome ───────────────────────────────────────────


class TestRecordFailureOutcome:
    """Tests for recording autopsy failures as PhiScorer outcomes."""

    def test_goal_failure_records_failed(self):
        scorer = MagicMock()
        scorer.record_outcome.return_value = 0.3
        result = record_failure_outcome(
            scorer, agent_id="agent-1", failure_type="goal_necrosis", domain="planning"
        )
        assert result == 0.3
        scorer.record_outcome.assert_called_once_with(
            agent_id="agent-1", skill_domain="planning", outcome="failed"
        )

    def test_tool_failure_records_rejected(self):
        scorer = MagicMock()
        scorer.record_outcome.return_value = 0.5
        result = record_failure_outcome(
            scorer, agent_id="agent-2", failure_type="tool_loop", domain="coding"
        )
        assert result == 0.5
        scorer.record_outcome.assert_called_once_with(
            agent_id="agent-2", skill_domain="coding", outcome="rejected"
        )

    def test_unknown_failure_type_defaults_to_rejected(self):
        scorer = MagicMock()
        scorer.record_outcome.return_value = 0.6
        record_failure_outcome(
            scorer,
            agent_id="agent-3",
            failure_type="never_seen_before",
            domain="research",
        )
        scorer.record_outcome.assert_called_once_with(
            agent_id="agent-3", skill_domain="research", outcome="rejected"
        )

    def test_returns_new_phi_score(self):
        scorer = MagicMock()
        scorer.record_outcome.return_value = 0.42
        result = record_failure_outcome(
            scorer, agent_id="a", failure_type="overconfidence", domain="d"
        )
        assert result == 0.42

    def test_all_known_failure_types(self):
        """Verify every FAILURE_SEVERITY key produces correct outcome."""
        for failure_type, expected_outcome in FAILURE_SEVERITY.items():
            scorer = MagicMock()
            scorer.record_outcome.return_value = 0.5
            record_failure_outcome(
                scorer, agent_id="test", failure_type=failure_type, domain="test"
            )
            actual_outcome = scorer.record_outcome.call_args.kwargs["outcome"]
            assert actual_outcome == expected_outcome, f"{failure_type} → {actual_outcome}"


# ─── leave_autopsy_marker ─────────────────────────────────────────────


class TestLeaveAutopsyMarker:
    """Tests for leaving stigmergy markers from autopsy results."""

    def test_creates_failure_pattern_marker(self):
        field = MagicMock()
        marker = MagicMock()
        field.leave_marker.return_value = marker
        result = leave_autopsy_marker(
            field,
            agent_id="agent-1",
            failure_type="tool_loop",
            target="wf-123/step-5",
            details="Retried 10 times without progress",
        )
        assert result is marker
        field.leave_marker.assert_called_once()
        call_kwargs = field.leave_marker.call_args.kwargs
        assert call_kwargs["marker_type"] == "failure_pattern"
        assert call_kwargs["target"] == "wf-123/step-5"
        assert "[tool_loop]" in call_kwargs["content"]
        assert "Retried 10 times" in call_kwargs["content"]

    def test_failed_severity_gets_strength_1(self):
        field = MagicMock()
        field.leave_marker.return_value = MagicMock()
        leave_autopsy_marker(
            field,
            agent_id="a",
            failure_type="goal_necrosis",
            target="t",
            details="d",
        )
        call_kwargs = field.leave_marker.call_args.kwargs
        assert call_kwargs["strength"] == 1.0

    def test_rejected_severity_gets_strength_07(self):
        field = MagicMock()
        field.leave_marker.return_value = MagicMock()
        leave_autopsy_marker(
            field,
            agent_id="a",
            failure_type="tool_hallucination",
            target="t",
            details="d",
        )
        call_kwargs = field.leave_marker.call_args.kwargs
        assert call_kwargs["strength"] == 0.7

    def test_unknown_failure_type_gets_rejected_strength(self):
        field = MagicMock()
        field.leave_marker.return_value = MagicMock()
        leave_autopsy_marker(
            field,
            agent_id="a",
            failure_type="exotic_failure",
            target="t",
            details="d",
        )
        call_kwargs = field.leave_marker.call_args.kwargs
        assert call_kwargs["strength"] == 0.7

    def test_agent_id_passed_through(self):
        field = MagicMock()
        field.leave_marker.return_value = MagicMock()
        leave_autopsy_marker(
            field,
            agent_id="special-agent",
            failure_type="overconfidence",
            target="t",
            details="d",
        )
        call_kwargs = field.leave_marker.call_args.kwargs
        assert call_kwargs["agent_id"] == "special-agent"

    def test_content_format(self):
        field = MagicMock()
        field.leave_marker.return_value = MagicMock()
        leave_autopsy_marker(
            field,
            agent_id="a",
            failure_type="goal_cancer",
            target="t",
            details="Scope expanded uncontrollably",
        )
        content = field.leave_marker.call_args.kwargs["content"]
        assert content == "[goal_cancer] Scope expanded uncontrollably"

    def test_returns_marker_object(self):
        field = MagicMock()
        expected_marker = MagicMock()
        field.leave_marker.return_value = expected_marker
        result = leave_autopsy_marker(
            field, agent_id="a", failure_type="unknown", target="t", details="d"
        )
        assert result is expected_marker
