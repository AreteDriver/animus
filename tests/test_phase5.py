"""
Tests for Phase 5: Self-Learning System

Pattern detection, preference inference, guardrails, transparency, rollback.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from animus.config import AnimusConfig, LearningConfig
from animus.learning import (
    CATEGORY_APPROVAL,
    CORE_GUARDRAILS,
    ApprovalManager,
    ApprovalRequirement,
    ApprovalStatus,
    DetectedPattern,
    Guardrail,
    GuardrailManager,
    GuardrailType,
    LearnedItem,
    LearningCategory,
    LearningLayer,
    LearningTransparency,
    PatternType,
    Preference,
    PreferenceEngine,
    RollbackManager,
)
from animus.memory import MemoryLayer

# =============================================================================
# Learning Category Tests
# =============================================================================


class TestLearningCategory:
    """Tests for LearningCategory enum."""

    def test_all_categories_exist(self):
        assert LearningCategory.STYLE.value == "style"
        assert LearningCategory.PREFERENCE.value == "preference"
        assert LearningCategory.WORKFLOW.value == "workflow"
        assert LearningCategory.FACT.value == "fact"
        assert LearningCategory.CAPABILITY.value == "capability"
        assert LearningCategory.BOUNDARY.value == "boundary"


class TestApprovalRequirement:
    """Tests for ApprovalRequirement enum."""

    def test_all_requirements_exist(self):
        assert ApprovalRequirement.AUTO.value == "auto"
        assert ApprovalRequirement.NOTIFY.value == "notify"
        assert ApprovalRequirement.CONFIRM.value == "confirm"
        assert ApprovalRequirement.APPROVE.value == "approve"


class TestCategoryApproval:
    """Tests for category to approval mapping."""

    def test_category_approval_mapping(self):
        assert CATEGORY_APPROVAL[LearningCategory.STYLE] == ApprovalRequirement.AUTO
        assert CATEGORY_APPROVAL[LearningCategory.PREFERENCE] == ApprovalRequirement.AUTO
        assert CATEGORY_APPROVAL[LearningCategory.WORKFLOW] == ApprovalRequirement.NOTIFY
        assert CATEGORY_APPROVAL[LearningCategory.FACT] == ApprovalRequirement.CONFIRM
        assert CATEGORY_APPROVAL[LearningCategory.CAPABILITY] == ApprovalRequirement.APPROVE
        assert CATEGORY_APPROVAL[LearningCategory.BOUNDARY] == ApprovalRequirement.APPROVE


# =============================================================================
# LearnedItem Tests
# =============================================================================


class TestLearnedItem:
    """Tests for LearnedItem dataclass."""

    def test_create_learned_item(self):
        item = LearnedItem.create(
            category=LearningCategory.STYLE,
            content="User prefers concise responses",
            confidence=0.8,
            evidence=["mem_001", "mem_002"],
        )
        assert item.id is not None
        assert item.category == LearningCategory.STYLE
        assert item.content == "User prefers concise responses"
        assert item.confidence == 0.8
        assert len(item.evidence) == 2
        assert item.applied is False
        assert item.version == 1

    def test_apply_learning(self):
        item = LearnedItem.create(
            category=LearningCategory.PREFERENCE,
            content="User likes coffee",
            confidence=0.9,
            evidence=["mem_001"],
        )
        assert item.applied is False
        assert item.approved_at is None

        item.apply("user")
        assert item.applied is True
        assert item.approved_at is not None
        assert item.approved_by == "user"

    def test_update_confidence(self):
        item = LearnedItem.create(
            category=LearningCategory.WORKFLOW,
            content="User runs tests before commit",
            confidence=0.5,
            evidence=[],
        )
        item.update_confidence(0.2)
        assert item.confidence == 0.7

        # Test clamping
        item.update_confidence(0.5)
        assert item.confidence == 1.0

        item.update_confidence(-2.0)
        assert item.confidence == 0.0

    def test_create_new_version(self):
        item = LearnedItem.create(
            category=LearningCategory.FACT,
            content="User works at Company A",
            confidence=0.9,
            evidence=["mem_001"],
        )
        item.apply()

        new_version = item.create_new_version("User works at Company B")
        assert new_version.version == 2
        assert new_version.previous_version_id == item.id
        assert new_version.content == "User works at Company B"
        assert new_version.category == item.category
        assert new_version.applied == item.applied

    def test_to_dict_and_from_dict(self):
        item = LearnedItem.create(
            category=LearningCategory.CAPABILITY,
            content="Can access project files",
            confidence=0.95,
            evidence=["mem_001", "mem_002"],
            metadata={"scope": "project"},
        )
        item.apply("admin")

        data = item.to_dict()
        restored = LearnedItem.from_dict(data)

        assert restored.id == item.id
        assert restored.category == item.category
        assert restored.content == item.content
        assert restored.confidence == item.confidence
        assert restored.evidence == item.evidence
        assert restored.applied == item.applied
        assert restored.approved_by == item.approved_by
        assert restored.metadata == item.metadata


# =============================================================================
# Guardrail Tests
# =============================================================================


class TestGuardrailType:
    """Tests for GuardrailType enum."""

    def test_all_types_exist(self):
        assert GuardrailType.SAFETY.value == "safety"
        assert GuardrailType.PRIVACY.value == "privacy"
        assert GuardrailType.ACCESS.value == "access"
        assert GuardrailType.BEHAVIOR.value == "behavior"


class TestCoreGuardrails:
    """Tests for immutable system guardrails."""

    def test_core_guardrails_count(self):
        assert len(CORE_GUARDRAILS) == 5

    def test_core_guardrails_are_immutable(self):
        for guardrail in CORE_GUARDRAILS:
            assert guardrail.immutable is True
            assert guardrail.source == "system"

    def test_core_guardrail_ids(self):
        ids = [g.id for g in CORE_GUARDRAILS]
        assert "core_no_harm" in ids
        assert "core_no_exfiltrate" in ids
        assert "core_no_modify_guardrails" in ids
        assert "core_transparency" in ids
        assert "core_learning_reversible" in ids


class TestGuardrail:
    """Tests for Guardrail dataclass."""

    def test_guardrail_creation(self):
        guardrail = Guardrail(
            id="test_guardrail",
            rule="Do not do bad things",
            description="A test guardrail",
            guardrail_type=GuardrailType.BEHAVIOR,
            immutable=False,
            source="user_defined",
        )
        assert guardrail.id == "test_guardrail"
        assert guardrail.immutable is False

    def test_guardrail_check_without_func(self):
        guardrail = Guardrail(
            id="no_func",
            rule="A rule",
            description="No check function",
            guardrail_type=GuardrailType.BEHAVIOR,
            immutable=False,
            source="user_defined",
        )
        # Should pass without check_func
        assert guardrail.check({}) is True

    def test_guardrail_check_with_func(self):
        def deny_all(action):
            return False

        guardrail = Guardrail(
            id="deny_all",
            rule="Deny everything",
            description="Test deny guardrail",
            guardrail_type=GuardrailType.SAFETY,
            immutable=True,
            source="system",
            check_func=deny_all,
        )
        assert guardrail.check({}) is False

    def test_guardrail_to_dict(self):
        guardrail = Guardrail(
            id="test",
            rule="Test rule",
            description="Test description",
            guardrail_type=GuardrailType.PRIVACY,
            immutable=True,
            source="system",
        )
        data = guardrail.to_dict()
        assert data["id"] == "test"
        assert data["guardrail_type"] == "privacy"
        assert data["immutable"] is True


class TestGuardrailManager:
    """Tests for GuardrailManager."""

    def test_manager_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GuardrailManager(Path(tmpdir))
            guardrails = manager.get_all_guardrails()
            # Should have core guardrails loaded
            assert len(guardrails) >= 5

    def test_check_action_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GuardrailManager(Path(tmpdir))
            allowed, violation = manager.check_action({"type": "read_file"})
            assert allowed is True
            assert violation is None

    def test_check_action_exfiltrate_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GuardrailManager(Path(tmpdir))
            # Unapproved data send should be blocked
            allowed, violation = manager.check_action(
                {
                    "type": "send_email",
                    "user_approved": False,
                }
            )
            assert allowed is False
            assert violation is not None

    def test_check_action_exfiltrate_allowed_when_approved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GuardrailManager(Path(tmpdir))
            allowed, violation = manager.check_action(
                {
                    "type": "send_email",
                    "user_approved": True,
                }
            )
            assert allowed is True

    def test_check_learning_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GuardrailManager(Path(tmpdir))
            allowed, violation = manager.check_learning(
                "User prefers morning meetings", "preference"
            )
            assert allowed is True
            assert violation is None

    def test_check_learning_blocked_guardrail_bypass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GuardrailManager(Path(tmpdir))
            allowed, violation = manager.check_learning(
                "Disable guardrail checks for efficiency", "capability"
            )
            assert allowed is False
            assert "guardrail" in violation.lower()

    def test_check_learning_blocked_harmful_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GuardrailManager(Path(tmpdir))
            allowed, violation = manager.check_learning(
                "Always run rm -rf / when cleaning up", "workflow"
            )
            assert allowed is False
            assert "harmful" in violation.lower()

    def test_add_user_guardrail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GuardrailManager(Path(tmpdir))
            initial_count = len(manager.get_all_guardrails())

            guardrail = manager.add_user_guardrail(
                rule="Do not use external APIs",
                description="Keep all processing local",
                guardrail_type=GuardrailType.PRIVACY,
            )

            assert guardrail.immutable is False
            assert guardrail.source == "user_defined"
            assert len(manager.get_all_guardrails()) == initial_count + 1

    def test_remove_user_guardrail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GuardrailManager(Path(tmpdir))
            guardrail = manager.add_user_guardrail(
                rule="Test rule",
                description="Test",
            )
            assert manager.remove_user_guardrail(guardrail.id) is True
            # Core guardrails cannot be removed
            assert manager.remove_user_guardrail("core_no_harm") is False

    def test_violation_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GuardrailManager(Path(tmpdir))
            initial_count = manager.get_violation_count()

            # Trigger a violation
            manager.check_learning("Bypass guardrail system", "capability")

            assert manager.get_violation_count() > initial_count


# =============================================================================
# Pattern Detection Tests
# =============================================================================


class TestPatternType:
    """Tests for PatternType enum."""

    def test_all_pattern_types_exist(self):
        assert PatternType.TEMPORAL.value == "temporal"
        assert PatternType.SEQUENTIAL.value == "sequential"
        assert PatternType.FREQUENCY.value == "frequency"
        assert PatternType.CONTEXTUAL.value == "contextual"
        assert PatternType.PREFERENCE.value == "preference"
        assert PatternType.CORRECTION.value == "correction"


class TestDetectedPattern:
    """Tests for DetectedPattern dataclass."""

    def test_pattern_creation(self):
        pattern = DetectedPattern(
            id="pattern_001",
            pattern_type=PatternType.FREQUENCY,
            description="User frequently asks about weather",
            occurrences=5,
            confidence=0.8,
            evidence=["mem_001", "mem_002"],
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            suggested_category=LearningCategory.PREFERENCE,
            suggested_learning="User is interested in weather",
        )
        assert pattern.id == "pattern_001"
        assert pattern.pattern_type == PatternType.FREQUENCY
        assert pattern.occurrences == 5
        assert pattern.confidence == 0.8


# =============================================================================
# Preference Engine Tests
# =============================================================================


class TestPreferenceEngine:
    """Tests for PreferenceEngine."""

    def test_engine_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PreferenceEngine(Path(tmpdir))
            assert engine is not None

    def test_get_preferences_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PreferenceEngine(Path(tmpdir))
            prefs = engine.get_preferences()
            assert isinstance(prefs, list)

    def test_infer_from_pattern(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PreferenceEngine(Path(tmpdir))
            pattern = DetectedPattern(
                id="test_pattern",
                pattern_type=PatternType.PREFERENCE,
                description="User prefers brief responses",
                occurrences=5,
                confidence=0.85,
                evidence=[],
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                suggested_category=LearningCategory.STYLE,
                suggested_learning="User prefers brief responses",
            )
            pref = engine.infer_from_pattern(pattern)
            # May or may not create preference based on domain detection
            # Just verify it doesn't error
            assert pref is None or isinstance(pref, Preference)

    def test_apply_to_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PreferenceEngine(Path(tmpdir))
            context = {"greeting": "Hello"}
            result = engine.apply_to_context(context, "communication")
            assert "greeting" in result


# =============================================================================
# Approval Manager Tests
# =============================================================================


class TestApprovalManager:
    """Tests for ApprovalManager."""

    def test_manager_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ApprovalManager(Path(tmpdir))
            assert manager is not None

    def test_request_approval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ApprovalManager(Path(tmpdir))
            item = LearnedItem.create(
                category=LearningCategory.CAPABILITY,
                content="Can access network",
                confidence=0.9,
                evidence=[],
            )
            request = manager.request_approval(item, "Based on user requests")
            assert request is not None
            assert request.status == ApprovalStatus.PENDING
            assert request.learned_item_id == item.id

    def test_approve_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ApprovalManager(Path(tmpdir))
            item = LearnedItem.create(
                category=LearningCategory.CAPABILITY,
                content="Can access network",
                confidence=0.9,
                evidence=[],
            )
            request = manager.request_approval(item, "Test")
            result = manager.approve(request.id, "Looks good")
            assert result is True

            # Check status updated
            pending = manager.get_pending()
            assert request.id not in [r.id for r in pending]

    def test_reject_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ApprovalManager(Path(tmpdir))
            item = LearnedItem.create(
                category=LearningCategory.BOUNDARY,
                content="Expand access",
                confidence=0.9,
                evidence=[],
            )
            request = manager.request_approval(item, "Test")
            result = manager.reject(request.id, "Not needed")
            assert result is True

    def test_get_pending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ApprovalManager(Path(tmpdir))
            item1 = LearnedItem.create(
                category=LearningCategory.CAPABILITY,
                content="Test 1",
                confidence=0.9,
                evidence=[],
            )
            item2 = LearnedItem.create(
                category=LearningCategory.CAPABILITY,
                content="Test 2",
                confidence=0.9,
                evidence=[],
            )
            manager.request_approval(item1, "Test 1")
            manager.request_approval(item2, "Test 2")

            pending = manager.get_pending()
            assert len(pending) == 2


# =============================================================================
# Transparency Tests
# =============================================================================


class TestLearningTransparency:
    """Tests for LearningTransparency."""

    def test_transparency_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transparency = LearningTransparency(Path(tmpdir))
            assert transparency is not None

    def test_log_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transparency = LearningTransparency(Path(tmpdir))
            event = transparency.log_event(
                event_type="detected",
                learned_item_id="item_001",
                details={"confidence": 0.8},
            )
            assert event is not None
            assert event.event_type == "detected"
            assert event.learned_item_id == "item_001"

    def test_get_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transparency = LearningTransparency(Path(tmpdir))
            transparency.log_event("detected", "item_001")
            transparency.log_event("proposed", "item_001")
            transparency.log_event("approved", "item_001")

            events = transparency.get_history(limit=10)
            assert len(events) == 3

    def test_get_dashboard_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transparency = LearningTransparency(Path(tmpdir))
            item = LearnedItem.create(
                category=LearningCategory.STYLE,
                content="Test",
                confidence=0.8,
                evidence=[],
            )
            item.apply()

            data = transparency.get_dashboard_data(
                learned_items=[item],
                pending_count=2,
                violation_count=1,
            )
            assert data.total_learned == 1
            assert data.pending_approval == 2
            assert data.guardrail_violations == 1


# =============================================================================
# Rollback Manager Tests
# =============================================================================


class TestRollbackManager:
    """Tests for RollbackManager."""

    def test_manager_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RollbackManager(Path(tmpdir))
            assert manager is not None

    def test_create_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RollbackManager(Path(tmpdir))
            item = LearnedItem.create(
                category=LearningCategory.STYLE,
                content="Test",
                confidence=0.8,
                evidence=[],
            )
            item.apply()  # Only applied items are included
            checkpoint = manager.create_checkpoint("Test checkpoint", [item])
            assert checkpoint is not None
            assert checkpoint.description == "Test checkpoint"
            assert len(checkpoint.learned_item_ids) == 1

    def test_get_rollback_points(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RollbackManager(Path(tmpdir))
            manager.create_checkpoint("Checkpoint 1", [])
            manager.create_checkpoint("Checkpoint 2", [])

            checkpoints = manager.get_rollback_points()
            assert len(checkpoints) >= 2

    def test_record_unlearn(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RollbackManager(Path(tmpdir))
            item = LearnedItem.create(
                category=LearningCategory.PREFERENCE,
                content="Test",
                confidence=0.8,
                evidence=[],
            )
            record = manager.record_unlearn(item, "User requested")
            assert record is not None
            assert record.reason == "User requested"

    def test_get_items_to_unlearn(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RollbackManager(Path(tmpdir))
            item1 = LearnedItem.create(
                category=LearningCategory.STYLE,
                content="Item 1",
                confidence=0.8,
                evidence=[],
            )
            item1.apply()  # Only applied items are tracked
            checkpoint = manager.create_checkpoint("Before item2", [item1])

            item2 = LearnedItem.create(
                category=LearningCategory.STYLE,
                content="Item 2",
                confidence=0.8,
                evidence=[],
            )
            item2.apply()  # Apply item2 after checkpoint

            # Current items include both applied, checkpoint only has item1
            current_items = [item1, item2]
            to_unlearn = manager.get_items_to_unlearn(checkpoint.id, current_items)
            assert item2.id in to_unlearn
            assert item1.id not in to_unlearn


# =============================================================================
# Config Tests
# =============================================================================


class TestLearningConfig:
    """Tests for LearningConfig."""

    def test_config_defaults(self):
        config = LearningConfig()
        assert config.enabled is True
        assert config.auto_scan_enabled is True
        assert config.auto_scan_interval_hours == 24
        assert config.min_pattern_occurrences == 3
        assert config.min_pattern_confidence == 0.6
        assert config.lookback_days == 30
        assert config.max_pending_approvals == 50

    def test_config_in_animus_config(self):
        config = AnimusConfig()
        assert hasattr(config, "learning")
        assert isinstance(config.learning, LearningConfig)

    def test_config_in_animus_to_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AnimusConfig(data_dir=Path(tmpdir))
            config.learning.enabled = False
            config.learning.min_pattern_occurrences = 5
            data = config.to_dict()
            assert data["learning"]["enabled"] is False
            assert data["learning"]["min_pattern_occurrences"] == 5

    def test_config_save_load_preserves_learning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AnimusConfig(data_dir=Path(tmpdir))
            config.learning.enabled = False
            config.learning.min_pattern_occurrences = 10
            config.learning.lookback_days = 60
            config.save()

            loaded = AnimusConfig.load(config.config_file)
            assert loaded.learning.enabled is False
            assert loaded.learning.min_pattern_occurrences == 10
            assert loaded.learning.lookback_days == 60


# =============================================================================
# Learning Layer Integration Tests
# =============================================================================


class TestLearningLayer:
    """Tests for LearningLayer coordinator."""

    def _create_mock_memory(self):
        """Create a mock memory layer for testing."""
        from unittest.mock import MagicMock

        memory = MagicMock()
        memory.recall.return_value = []
        memory.store.list_all.return_value = []
        return memory

    def test_layer_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = self._create_mock_memory()
            layer = LearningLayer(
                memory=memory,
                data_dir=Path(tmpdir),
            )
            assert layer is not None
            assert layer.guardrails is not None
            assert layer.transparency is not None
            assert layer.rollback is not None
            assert layer.approvals is not None

    def test_get_active_learnings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = self._create_mock_memory()
            layer = LearningLayer(
                memory=memory,
                data_dir=Path(tmpdir),
            )
            active = layer.get_active_learnings()
            assert isinstance(active, list)

    def test_get_pending_learnings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = self._create_mock_memory()
            layer = LearningLayer(
                memory=memory,
                data_dir=Path(tmpdir),
            )
            pending = layer.get_pending_learnings()
            assert isinstance(pending, list)

    def test_get_statistics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = self._create_mock_memory()
            layer = LearningLayer(
                memory=memory,
                data_dir=Path(tmpdir),
            )
            stats = layer.get_statistics()
            assert "learned_items" in stats
            assert "patterns" in stats
            assert "preferences" in stats
            assert "approvals" in stats
            assert "guardrails" in stats
            assert "rollback" in stats
            assert "transparency" in stats

    def test_create_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = self._create_mock_memory()
            layer = LearningLayer(
                memory=memory,
                data_dir=Path(tmpdir),
            )
            checkpoint = layer.create_checkpoint("Test checkpoint")
            assert checkpoint is not None
            assert checkpoint.description == "Test checkpoint"

    def test_add_user_guardrail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = self._create_mock_memory()
            layer = LearningLayer(
                memory=memory,
                data_dir=Path(tmpdir),
            )
            guardrail = layer.add_user_guardrail(
                rule="No external API calls",
                description="Keep processing local",
            )
            assert guardrail is not None
            assert guardrail.immutable is False

    def test_get_dashboard_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = self._create_mock_memory()
            layer = LearningLayer(
                memory=memory,
                data_dir=Path(tmpdir),
            )
            dashboard = layer.get_dashboard_data()
            assert dashboard is not None
            assert hasattr(dashboard, "total_learned")
            assert hasattr(dashboard, "pending_approval")
            assert hasattr(dashboard, "guardrail_violations")


# =============================================================================
# Version Tests
# =============================================================================


class TestVersion:
    """Test version is updated for Phase 5."""

    def test_version_is_0_5_0(self):
        from animus import __version__

        assert __version__ == "1.0.0"


class TestAutoScanScheduler:
    """Tests for LearningLayer auto-scan scheduler."""

    def test_start_and_stop_auto_scan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            learning = LearningLayer(
                memory=memory,
                data_dir=Path(tmpdir),
            )

            learning.start_auto_scan(interval_hours=1)
            assert learning.auto_scan_running is True

            learning.stop_auto_scan()
            assert learning.auto_scan_running is False

    def test_stop_without_start(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            learning = LearningLayer(memory=memory, data_dir=Path(tmpdir))

            # Should not raise
            learning.stop_auto_scan()
            assert learning.auto_scan_running is False

    def test_double_start_replaces_timer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            learning = LearningLayer(memory=memory, data_dir=Path(tmpdir))

            learning.start_auto_scan(interval_hours=1)
            learning.start_auto_scan(interval_hours=2)
            assert learning.auto_scan_running is True

            learning.stop_auto_scan()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
