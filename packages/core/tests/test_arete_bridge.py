"""Tests for Arete Tools → Core bridge (verdict→memory, calibrate→identity)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from animus.integrations.arete_bridge import (
    auto_sync_verdicts,
    sync_calibrate_to_identity,
    sync_verdict_to_memory,
)

# ─── sync_verdict_to_memory ───────────────────────────────────────────


class TestSyncVerdictToMemory:
    """Tests for syncing Verdict decisions to episodic memory."""

    def test_basic_decision_stored(self):
        memory_layer = MagicMock()
        memory_layer.remember.return_value = MagicMock(id="mem-1")
        decision = {
            "id": "dec-001",
            "title": "Use PostgreSQL",
            "reasoning": "Better for relational data",
            "alternatives": ["MongoDB", "DynamoDB"],
            "category": "architecture",
            "review_date": "2026-04-05",
        }
        result = sync_verdict_to_memory(memory_layer, decision)
        assert result is not None
        memory_layer.remember.assert_called_once()
        call_kwargs = memory_layer.remember.call_args.kwargs
        assert "Use PostgreSQL" in call_kwargs["content"]
        assert "Better for relational data" in call_kwargs["content"]
        assert "MongoDB" in call_kwargs["content"]

    def test_memory_type_is_episodic(self):
        from animus.memory import MemoryType

        memory_layer = MagicMock()
        decision = {"title": "Test", "category": "general"}
        sync_verdict_to_memory(memory_layer, decision)
        call_kwargs = memory_layer.remember.call_args.kwargs
        assert call_kwargs["memory_type"] == MemoryType.EPISODIC

    def test_tags_include_verdict_and_category(self):
        memory_layer = MagicMock()
        decision = {"title": "X", "category": "tooling"}
        sync_verdict_to_memory(memory_layer, decision)
        call_kwargs = memory_layer.remember.call_args.kwargs
        assert "verdict" in call_kwargs["tags"]
        assert "decision" in call_kwargs["tags"]
        assert "tooling" in call_kwargs["tags"]

    def test_source_is_learned(self):
        memory_layer = MagicMock()
        decision = {"title": "X"}
        sync_verdict_to_memory(memory_layer, decision)
        call_kwargs = memory_layer.remember.call_args.kwargs
        assert call_kwargs["source"] == "learned"

    def test_subtype_is_decision(self):
        memory_layer = MagicMock()
        decision = {"title": "X"}
        sync_verdict_to_memory(memory_layer, decision)
        call_kwargs = memory_layer.remember.call_args.kwargs
        assert call_kwargs["subtype"] == "decision"

    def test_confidence_is_high(self):
        memory_layer = MagicMock()
        decision = {"title": "X"}
        sync_verdict_to_memory(memory_layer, decision)
        call_kwargs = memory_layer.remember.call_args.kwargs
        assert call_kwargs["confidence"] == 0.95

    def test_metadata_includes_decision_id(self):
        memory_layer = MagicMock()
        decision = {"id": "dec-99", "title": "X", "category": "general"}
        sync_verdict_to_memory(memory_layer, decision)
        call_kwargs = memory_layer.remember.call_args.kwargs
        assert call_kwargs["metadata"]["decision_id"] == "dec-99"

    def test_no_alternatives_ok(self):
        memory_layer = MagicMock()
        decision = {"title": "Simple", "reasoning": "Because"}
        sync_verdict_to_memory(memory_layer, decision)
        content = memory_layer.remember.call_args.kwargs["content"]
        assert "Alternatives" not in content

    def test_no_review_date_ok(self):
        memory_layer = MagicMock()
        decision = {"title": "No Date"}
        sync_verdict_to_memory(memory_layer, decision)
        content = memory_layer.remember.call_args.kwargs["content"]
        assert "Review date" not in content

    def test_defaults_for_missing_keys(self):
        memory_layer = MagicMock()
        decision = {}
        sync_verdict_to_memory(memory_layer, decision)
        call_kwargs = memory_layer.remember.call_args.kwargs
        assert "Untitled Decision" in call_kwargs["content"]
        assert "general" in call_kwargs["tags"]


# ─── sync_calibrate_to_identity ───────────────────────────────────────


class TestSyncCalibrateToIdentity:
    """Tests for syncing Calibrate stats to identity model."""

    def test_basic_reflection_recorded(self):
        identity = MagicMock()
        stats = {
            "accuracy": 0.85,
            "total_predictions": 100,
            "overconfidence_rate": 0.1,
            "domains": {},
        }
        sync_calibrate_to_identity(identity, stats)
        identity.record_reflection.assert_called_once()
        call_kwargs = identity.record_reflection.call_args.kwargs
        assert "85.0%" in call_kwargs["summary"]
        assert "100" in call_kwargs["summary"]

    def test_high_overconfidence_generates_improvement(self):
        identity = MagicMock()
        stats = {
            "accuracy": 0.8,
            "total_predictions": 50,
            "overconfidence_rate": 0.3,
            "domains": {},
        }
        sync_calibrate_to_identity(identity, stats)
        improvements = identity.record_reflection.call_args.kwargs["improvements"]
        assert any("confidence" in i.lower() for i in improvements)

    def test_low_accuracy_generates_improvement(self):
        identity = MagicMock()
        stats = {
            "accuracy": 0.5,
            "total_predictions": 50,
            "overconfidence_rate": 0.1,
            "domains": {},
        }
        sync_calibrate_to_identity(identity, stats)
        improvements = identity.record_reflection.call_args.kwargs["improvements"]
        assert any("methodology" in i.lower() for i in improvements)

    def test_low_domain_accuracy_generates_domain_improvement(self):
        identity = MagicMock()
        stats = {
            "accuracy": 0.8,
            "total_predictions": 50,
            "overconfidence_rate": 0.1,
            "domains": {"weather": {"accuracy": 0.4}},
        }
        sync_calibrate_to_identity(identity, stats)
        improvements = identity.record_reflection.call_args.kwargs["improvements"]
        assert any("weather" in i for i in improvements)

    def test_no_issues_passes_none_improvements(self):
        identity = MagicMock()
        stats = {
            "accuracy": 0.95,
            "total_predictions": 200,
            "overconfidence_rate": 0.05,
            "domains": {"coding": {"accuracy": 0.9}},
        }
        sync_calibrate_to_identity(identity, stats)
        improvements = identity.record_reflection.call_args.kwargs["improvements"]
        assert improvements is None

    def test_empty_stats_defaults(self):
        identity = MagicMock()
        sync_calibrate_to_identity(identity, {})
        identity.record_reflection.assert_called_once()
        call_kwargs = identity.record_reflection.call_args.kwargs
        assert "0.0%" in call_kwargs["summary"]

    def test_multiple_low_domains(self):
        identity = MagicMock()
        stats = {
            "accuracy": 0.8,
            "total_predictions": 50,
            "overconfidence_rate": 0.1,
            "domains": {
                "sports": {"accuracy": 0.3},
                "politics": {"accuracy": 0.5},
            },
        }
        sync_calibrate_to_identity(identity, stats)
        improvements = identity.record_reflection.call_args.kwargs["improvements"]
        domain_improvements = [i for i in improvements if "sports" in i or "politics" in i]
        assert len(domain_improvements) == 2


# ─── auto_sync_verdicts ───────────────────────────────────────────────


class TestAutoSyncVerdicts:
    """Tests for batch syncing Verdict decisions."""

    @patch("animus.integrations.arete_bridge.HAS_VERDICT", False)
    def test_no_verdict_installed_returns_zero(self):
        memory_layer = MagicMock()
        count = auto_sync_verdicts(memory_layer)
        assert count == 0
        memory_layer.remember.assert_not_called()

    @patch("animus.integrations.arete_bridge.HAS_VERDICT", True)
    @patch("animus.integrations.arete_bridge.DecisionStore", create=True)
    def test_syncs_multiple_decisions(self, mock_store_cls):
        mock_store = MagicMock()
        mock_store.list_decisions.return_value = [
            {"id": "d1", "title": "A", "category": "arch"},
            {"id": "d2", "title": "B", "category": "tool"},
        ]
        mock_store_cls.return_value = mock_store
        memory_layer = MagicMock()
        count = auto_sync_verdicts(memory_layer)
        assert count == 2
        assert memory_layer.remember.call_count == 2

    @patch("animus.integrations.arete_bridge.HAS_VERDICT", True)
    @patch("animus.integrations.arete_bridge.DecisionStore", create=True)
    def test_since_filter_passed_to_store(self, mock_store_cls):
        mock_store = MagicMock()
        mock_store.list_decisions.return_value = []
        mock_store_cls.return_value = mock_store
        memory_layer = MagicMock()
        since = datetime(2026, 1, 1, tzinfo=timezone.utc)
        auto_sync_verdicts(memory_layer, since=since)
        mock_store.list_decisions.assert_called_once_with(since=since)

    @patch("animus.integrations.arete_bridge.HAS_VERDICT", True)
    @patch("animus.integrations.arete_bridge.DecisionStore", create=True)
    def test_db_path_passed_to_store(self, mock_store_cls):
        mock_store = MagicMock()
        mock_store.list_decisions.return_value = []
        mock_store_cls.return_value = mock_store
        memory_layer = MagicMock()
        db_path = Path("/tmp/verdict.db")
        auto_sync_verdicts(memory_layer, db_path=db_path)
        mock_store_cls.assert_called_once_with(db_path=db_path)

    @patch("animus.integrations.arete_bridge.HAS_VERDICT", True)
    @patch("animus.integrations.arete_bridge.DecisionStore", create=True)
    def test_individual_failure_continues(self, mock_store_cls):
        mock_store = MagicMock()
        mock_store.list_decisions.return_value = [
            {"id": "d1", "title": "A", "category": "arch"},
            {"id": "d2", "title": "B", "category": "tool"},
            {"id": "d3", "title": "C", "category": "general"},
        ]
        mock_store_cls.return_value = mock_store
        memory_layer = MagicMock()
        # Second call to remember raises
        memory_layer.remember.side_effect = [
            MagicMock(),
            Exception("DB error"),
            MagicMock(),
        ]
        count = auto_sync_verdicts(memory_layer)
        assert count == 2  # first and third succeeded

    @patch("animus.integrations.arete_bridge.HAS_VERDICT", True)
    @patch("animus.integrations.arete_bridge.DecisionStore", create=True)
    def test_empty_store_returns_zero(self, mock_store_cls):
        mock_store = MagicMock()
        mock_store.list_decisions.return_value = []
        mock_store_cls.return_value = mock_store
        memory_layer = MagicMock()
        count = auto_sync_verdicts(memory_layer)
        assert count == 0
