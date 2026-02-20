"""
Animus Learning Layer

Coordinates pattern detection, preference inference, and learning management
with guardrail enforcement. All learning is transparent, observable, and reversible.
"""

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from animus.learning.approval import ApprovalManager, ApprovalRequest, ApprovalStatus
from animus.learning.categories import (
    CATEGORY_APPROVAL,
    ApprovalRequirement,
    LearnedItem,
    LearningCategory,
)
from animus.learning.guardrails import (
    CORE_GUARDRAILS,
    Guardrail,
    GuardrailManager,
    GuardrailType,
    GuardrailViolation,
)
from animus.learning.patterns import DetectedPattern, PatternDetector, PatternType
from animus.learning.preferences import Preference, PreferenceEngine
from animus.learning.rollback import RollbackManager, RollbackPoint
from animus.learning.transparency import (
    LearningDashboardData,
    LearningEvent,
    LearningTransparency,
)
from animus.logging import get_logger
from animus.protocols.safety import SafetyGuard

if TYPE_CHECKING:
    from animus.memory import MemoryLayer

logger = get_logger("learning")


class LearningLayer:
    """
    Main learning layer coordinating all learning subsystems.

    Responsibilities:
    - Coordinate pattern detection
    - Manage learned items
    - Enforce guardrails
    - Handle approvals
    - Provide transparency
    - Support rollback
    """

    def __init__(
        self,
        memory: "MemoryLayer",
        data_dir: Path,
        min_pattern_occurrences: int = 3,
        min_pattern_confidence: float = 0.6,
        lookback_days: int = 30,
    ):
        self.memory = memory
        self.data_dir = data_dir / "learning"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize subsystems
        self.pattern_detector = PatternDetector(
            memory=memory,
            min_occurrences=min_pattern_occurrences,
            min_confidence=min_pattern_confidence,
            lookback_days=lookback_days,
        )
        self.preference_engine = PreferenceEngine(self.data_dir)
        self.guardrails: SafetyGuard = GuardrailManager(self.data_dir)
        self.transparency = LearningTransparency(self.data_dir)
        self.rollback = RollbackManager(self.data_dir)
        self.approvals = ApprovalManager(self.data_dir)

        # Learned items store
        self._learned_items: dict[str, LearnedItem] = {}
        self._load_learned_items()

        # Background scheduler
        self._scan_timer: threading.Timer | None = None
        self._scan_interval_seconds: float = 0  # 0 = disabled

        logger.info("LearningLayer initialized")

    def start_auto_scan(self, interval_hours: float = 24) -> None:
        """Start background auto-scanning at the given interval."""
        self.stop_auto_scan()
        self._scan_interval_seconds = interval_hours * 3600
        logger.info(f"Auto-scan started: interval={interval_hours}h")
        self._schedule_next_scan()

    def stop_auto_scan(self) -> None:
        """Stop background auto-scanning."""
        if self._scan_timer is not None:
            self._scan_timer.cancel()
            self._scan_timer = None
            logger.info("Auto-scan stopped")

    def _schedule_next_scan(self) -> None:
        """Schedule the next auto-scan."""
        if self._scan_interval_seconds <= 0:
            return
        self._scan_timer = threading.Timer(self._scan_interval_seconds, self._run_scheduled_scan)
        self._scan_timer.daemon = True
        self._scan_timer.start()

    def _run_scheduled_scan(self) -> None:
        """Execute a scheduled scan and reschedule."""
        try:
            logger.info("Running scheduled auto-scan")
            self.scan_and_learn()
        except (json.JSONDecodeError, ValueError, OSError, RuntimeError) as e:
            logger.error(f"Scheduled scan failed: {e}")
        finally:
            self._schedule_next_scan()

    @property
    def auto_scan_running(self) -> bool:
        """Check if auto-scan is active."""
        return self._scan_timer is not None and self._scan_timer.is_alive()

    def _load_learned_items(self) -> None:
        """Load learned items from disk."""
        items_file = self.data_dir / "learned_items.json"
        if items_file.exists():
            try:
                with open(items_file) as f:
                    data = json.load(f)
                for item_data in data:
                    item = LearnedItem.from_dict(item_data)
                    self._learned_items[item.id] = item
                logger.info(f"Loaded {len(self._learned_items)} learned items")
            except (json.JSONDecodeError, ValueError, OSError) as e:
                logger.error(f"Failed to load learned items: {e}")

    def _save_learned_items(self) -> None:
        """Save learned items to disk."""
        items_file = self.data_dir / "learned_items.json"
        data = [item.to_dict() for item in self._learned_items.values()]
        with open(items_file, "w") as f:
            json.dump(data, f, indent=2)

    def scan_and_learn(self) -> list[DetectedPattern]:
        """
        Run pattern detection and process new patterns.

        Returns:
            List of newly detected patterns
        """
        logger.info("Starting learning scan")
        patterns = self.pattern_detector.scan_for_patterns()

        for pattern in patterns:
            self._process_pattern(pattern)

        logger.info(f"Processed {len(patterns)} patterns")
        return patterns

    def _process_pattern(self, pattern: DetectedPattern) -> LearnedItem | None:
        """Process a detected pattern into a potential learning."""
        # Create learned item from pattern
        proposed = LearnedItem.create(
            category=pattern.suggested_category,
            content=pattern.suggested_learning,
            confidence=pattern.confidence,
            evidence=pattern.evidence,
            source_pattern_id=pattern.id,
        )

        # Check guardrails first
        allowed, violation = self.guardrails.check_learning(
            proposed.content, proposed.category.value
        )

        if not allowed:
            self.transparency.log_event(
                "blocked_by_guardrail",
                proposed.id,
                details={"violation": violation},
            )
            logger.info(f"Learning blocked by guardrail: {violation}")
            return None

        # Store the proposed item
        self._learned_items[proposed.id] = proposed
        self._save_learned_items()

        # Log detection
        self.transparency.log_event(
            "detected",
            proposed.id,
            details={
                "pattern_id": pattern.id,
                "category": proposed.category.value,
                "confidence": proposed.confidence,
            },
        )

        # Handle based on approval requirement
        approval_req = CATEGORY_APPROVAL[proposed.category]

        if approval_req == ApprovalRequirement.AUTO:
            self._apply_learning(proposed)
        elif approval_req == ApprovalRequirement.NOTIFY:
            self._apply_learning(proposed)
            self.approvals.notify_user(f"Learned: {proposed.content[:50]}...")
        elif approval_req in (ApprovalRequirement.CONFIRM, ApprovalRequirement.APPROVE):
            # Create approval request
            evidence_summary = f"Based on {len(pattern.evidence)} observations over {pattern.occurrences} occurrences"
            self.approvals.request_approval(proposed, evidence_summary)
            self.transparency.log_event("proposed", proposed.id)
            logger.info(f"Learning proposed, awaiting approval: {proposed.id}")

        return proposed

    def _apply_learning(self, item: LearnedItem) -> None:
        """Apply a learned item."""
        item.apply()
        self._save_learned_items()

        # Also infer preferences from this learning
        if item.source_pattern_id:
            pattern = self._get_pattern(item.source_pattern_id)
            if pattern:
                self.preference_engine.infer_from_pattern(pattern)

        self.transparency.log_event("applied", item.id)
        logger.info(f"Applied learning: {item.content[:50]}...")

    def _get_pattern(self, pattern_id: str) -> DetectedPattern | None:
        """Get a pattern by ID."""
        patterns = self.pattern_detector.get_detected_patterns()
        for p in patterns:
            if p.id == pattern_id:
                return p
        return None

    def approve_learning(self, learned_item_id: str) -> bool:
        """
        Approve a pending learning.

        Args:
            learned_item_id: ID of item to approve

        Returns:
            True if approved, False if not found or already applied
        """
        item = self._learned_items.get(learned_item_id)
        if not item or item.applied:
            return False

        self._apply_learning(item)
        self.transparency.log_event("approved", learned_item_id, user_action="approve")

        # Also update approval request if exists
        for req in self.approvals.get_pending():
            if req.learned_item_id == learned_item_id:
                self.approvals.approve(req.id)
                break

        return True

    def reject_learning(self, learned_item_id: str, reason: str = "") -> bool:
        """
        Reject a pending learning.

        Args:
            learned_item_id: ID of item to reject
            reason: Reason for rejection

        Returns:
            True if rejected, False if not found
        """
        if learned_item_id in self._learned_items:
            self.transparency.log_event(
                "rejected",
                learned_item_id,
                details={"reason": reason},
                user_action="reject",
            )

            # Update approval request if exists
            for req in self.approvals.get_pending():
                if req.learned_item_id == learned_item_id:
                    self.approvals.reject(req.id, reason)
                    break

            del self._learned_items[learned_item_id]
            self._save_learned_items()
            return True
        return False

    def unlearn(self, learned_item_id: str, reason: str = "") -> bool:
        """
        Unlearn a specific item.

        Args:
            learned_item_id: ID of item to unlearn
            reason: Reason for unlearning

        Returns:
            True if unlearned, False if not found
        """
        item = self._learned_items.get(learned_item_id)
        if not item:
            return False

        # Record the unlearn
        self.rollback.record_unlearn(item, reason)

        # Log the event
        self.transparency.log_event(
            "rolled_back",
            learned_item_id,
            details={"reason": reason},
            user_action="rollback",
        )

        # Remove the item
        del self._learned_items[learned_item_id]
        self._save_learned_items()

        logger.info(f"Unlearned: {learned_item_id}")
        return True

    def create_checkpoint(self, description: str) -> RollbackPoint:
        """
        Create a rollback checkpoint.

        Args:
            description: Description of this checkpoint

        Returns:
            The created checkpoint
        """
        items = list(self._learned_items.values())
        return self.rollback.create_checkpoint(description, items)

    def rollback_to(self, rollback_point_id: str) -> tuple[bool, list[str]]:
        """
        Rollback to a specific checkpoint.

        Args:
            rollback_point_id: ID of checkpoint to rollback to

        Returns:
            (success, list of unlearned item IDs)
        """
        items = list(self._learned_items.values())
        to_unlearn = self.rollback.get_items_to_unlearn(rollback_point_id, items)

        if not to_unlearn:
            return False, []

        unlearned = []
        for item_id in to_unlearn:
            if self.unlearn(item_id, f"Rollback to {rollback_point_id}"):
                unlearned.append(item_id)

        return True, unlearned

    def get_active_learnings(self) -> list[LearnedItem]:
        """Get all active (applied) learnings."""
        return [item for item in self._learned_items.values() if item.applied]

    def get_pending_learnings(self) -> list[LearnedItem]:
        """Get learnings pending approval."""
        return [item for item in self._learned_items.values() if not item.applied]

    def get_learning(self, item_id: str) -> LearnedItem | None:
        """Get a specific learned item."""
        return self._learned_items.get(item_id)

    def get_all_learnings(self) -> list[LearnedItem]:
        """Get all learned items."""
        return list(self._learned_items.values())

    def get_dashboard_data(self) -> LearningDashboardData:
        """Get data for transparency dashboard."""
        return self.transparency.get_dashboard_data(
            learned_items=list(self._learned_items.values()),
            pending_count=len(self.approvals.get_pending()),
            violation_count=self.guardrails.get_violation_count(),
        )

    def get_preferences(self, domain: str | None = None) -> list[Preference]:
        """Get active preferences."""
        return self.preference_engine.get_preferences(domain)

    def apply_preferences_to_context(self, context: dict[str, Any], domain: str) -> dict[str, Any]:
        """Apply learned preferences to a context."""
        return self.preference_engine.apply_to_context(context, domain)

    def add_user_guardrail(
        self,
        rule: str,
        description: str,
        guardrail_type: GuardrailType = GuardrailType.BEHAVIOR,
    ) -> Guardrail:
        """Add a user-defined guardrail."""
        return self.guardrails.add_user_guardrail(rule, description, guardrail_type)

    def get_statistics(self) -> dict[str, Any]:
        """Get learning system statistics."""
        return {
            "learned_items": {
                "total": len(self._learned_items),
                "applied": len(self.get_active_learnings()),
                "pending": len(self.get_pending_learnings()),
            },
            "patterns": {
                "detected": len(self.pattern_detector.get_detected_patterns()),
            },
            "preferences": self.preference_engine.get_statistics(),
            "approvals": self.approvals.get_statistics(),
            "guardrails": {
                "total": len(self.guardrails.get_all_guardrails()),
                "violations": self.guardrails.get_violation_count(),
            },
            "rollback": self.rollback.get_statistics(),
            "transparency": self.transparency.get_statistics(),
        }


# Package exports
__all__ = [
    # Main layer
    "LearningLayer",
    # Categories
    "LearningCategory",
    "LearnedItem",
    "ApprovalRequirement",
    "CATEGORY_APPROVAL",
    # Patterns
    "PatternDetector",
    "DetectedPattern",
    "PatternType",
    # Preferences
    "PreferenceEngine",
    "Preference",
    # Guardrails
    "GuardrailManager",
    "Guardrail",
    "GuardrailType",
    "GuardrailViolation",
    "CORE_GUARDRAILS",
    # Transparency
    "LearningTransparency",
    "LearningEvent",
    "LearningDashboardData",
    # Rollback
    "RollbackManager",
    "RollbackPoint",
    # Approval
    "ApprovalManager",
    "ApprovalRequest",
    "ApprovalStatus",
]
