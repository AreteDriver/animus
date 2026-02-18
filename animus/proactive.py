"""
Proactive Intelligence System

Scheduler-driven briefings, contextual nudges, and deadline awareness.
Transforms Animus from reactive to proactive.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from animus.logging import get_logger

if TYPE_CHECKING:
    from animus.cognitive import CognitiveLayer
    from animus.memory import MemoryLayer

logger = get_logger("proactive")


class NudgeType(Enum):
    """Types of proactive nudges."""

    MORNING_BRIEF = "morning_brief"
    DEADLINE_WARNING = "deadline_warning"
    MEETING_PREP = "meeting_prep"
    CONTEXT_RECALL = "context_recall"
    FOLLOW_UP = "follow_up"
    PATTERN_INSIGHT = "pattern_insight"


class NudgePriority(Enum):
    """Urgency levels for nudges."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Nudge:
    """A proactive nudge to present to the user."""

    id: str
    nudge_type: NudgeType
    priority: NudgePriority
    title: str
    content: str
    created_at: datetime
    expires_at: datetime | None = None
    dismissed: bool = False
    acted_on: bool = False
    source_memory_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if this nudge has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def is_active(self) -> bool:
        """Check if this nudge is still actionable."""
        return not self.dismissed and not self.acted_on and not self.is_expired()

    def dismiss(self) -> None:
        self.dismissed = True

    def mark_acted(self) -> None:
        self.acted_on = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "nudge_type": self.nudge_type.value,
            "priority": self.priority.value,
            "title": self.title,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "dismissed": self.dismissed,
            "acted_on": self.acted_on,
            "source_memory_ids": self.source_memory_ids,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Nudge:
        return cls(
            id=data["id"],
            nudge_type=NudgeType(data["nudge_type"]),
            priority=NudgePriority(data["priority"]),
            title=data["title"],
            content=data["content"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=(
                datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
            ),
            dismissed=data.get("dismissed", False),
            acted_on=data.get("acted_on", False),
            source_memory_ids=data.get("source_memory_ids", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ScheduledCheck:
    """A scheduled proactive check."""

    name: str
    interval_minutes: int
    last_run: datetime | None = None
    enabled: bool = True

    def is_due(self) -> bool:
        if not self.enabled:
            return False
        if self.last_run is None:
            return True
        return datetime.now() - self.last_run >= timedelta(minutes=self.interval_minutes)


class ProactiveEngine:
    """
    Proactive intelligence engine.

    Runs scheduled checks against memory and integrations to generate
    contextual nudges without being asked.

    Features:
    - Morning briefings synthesized from calendar, tasks, and recent context
    - Deadline awareness with escalating warnings
    - Meeting prep with context recall
    - Follow-up reminders based on conversation patterns
    - Pattern-based insights
    """

    def __init__(
        self,
        data_dir: Path,
        memory: MemoryLayer,
        cognitive: CognitiveLayer | None = None,
    ):
        self.data_dir = data_dir
        self.memory = memory
        self.cognitive = cognitive

        self._nudges: list[Nudge] = []
        self._callbacks: list[Callable[[Nudge], None]] = []
        self._running = False
        self._thread: threading.Thread | None = None

        # Scheduled checks
        self._checks: list[ScheduledCheck] = [
            ScheduledCheck(name="deadline_scan", interval_minutes=60),
            ScheduledCheck(name="follow_up_scan", interval_minutes=120),
            ScheduledCheck(name="context_refresh", interval_minutes=30),
        ]

        self._load_nudges()
        logger.info("ProactiveEngine initialized")

    def _load_nudges(self) -> None:
        """Load persisted nudges from disk."""
        nudges_file = self.data_dir / "nudges.json"
        if nudges_file.exists():
            try:
                data = json.loads(nudges_file.read_text())
                self._nudges = [Nudge.from_dict(n) for n in data]
                logger.debug(f"Loaded {len(self._nudges)} nudges")
            except Exception as e:
                logger.error(f"Failed to load nudges: {e}")

    def _save_nudges(self) -> None:
        """Persist nudges to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        nudges_file = self.data_dir / "nudges.json"
        # Keep only recent active nudges + last 50 dismissed
        active = [n for n in self._nudges if n.is_active()]
        inactive = [n for n in self._nudges if not n.is_active()]
        to_save = active + inactive[-50:]
        nudges_file.write_text(json.dumps([n.to_dict() for n in to_save], indent=2))

    def add_callback(self, callback: Callable[[Nudge], None]) -> None:
        """Register a callback for new nudges."""
        self._callbacks.append(callback)

    def _emit_nudge(self, nudge: Nudge) -> None:
        """Store and notify callbacks about a new nudge."""
        self._nudges.append(nudge)
        self._save_nudges()
        for cb in self._callbacks:
            try:
                cb(nudge)
            except Exception as e:
                logger.error(f"Nudge callback error: {e}")
        logger.info(f"Nudge emitted: [{nudge.priority.value}] {nudge.title}")

    # =========================================================================
    # Nudge Generators
    # =========================================================================

    def generate_morning_brief(self) -> Nudge:
        """
        Generate a morning briefing from recent memories, tasks, and patterns.

        Returns:
            A Nudge containing the briefing
        """
        import uuid

        sections = []

        # Recent memories (last 24h)
        yesterday = datetime.now() - timedelta(hours=24)
        recent = self.memory.store.list_all()
        recent_items = [m for m in recent if m.created_at >= yesterday]
        if recent_items:
            items_text = "\n".join(f"- {m.content[:120]}" for m in recent_items[:10])
            sections.append(f"Recent activity ({len(recent_items)} items):\n{items_text}")

        # Tasks due soon
        tasks_with_deadline = [
            m
            for m in self.memory.store.list_all()
            if m.subtype in ("task", "deadline") or "deadline" in m.tags or "due" in m.tags
        ]
        if tasks_with_deadline:
            tasks_text = "\n".join(f"- {m.content[:120]}" for m in tasks_with_deadline[:5])
            sections.append(f"Upcoming deadlines:\n{tasks_text}")

        # Follow-ups needed
        follow_ups = [m for m in recent if "follow-up" in m.tags or "follow_up" in m.tags]
        if follow_ups:
            fu_text = "\n".join(f"- {m.content[:120]}" for m in follow_ups[:5])
            sections.append(f"Follow-ups needed:\n{fu_text}")

        content = "\n\n".join(sections) if sections else "No notable items for today's briefing."

        # Use cognitive layer to synthesize if available
        if self.cognitive and sections:
            try:
                prompt = (
                    "Generate a concise morning briefing (3-5 bullet points) "
                    "from this raw data. Focus on what's actionable today:\n\n" + content
                )
                content = self.cognitive.think(prompt)
            except Exception as e:
                logger.warning(f"Failed to synthesize briefing: {e}")

        nudge = Nudge(
            id=str(uuid.uuid4()),
            nudge_type=NudgeType.MORNING_BRIEF,
            priority=NudgePriority.MEDIUM,
            title="Morning Briefing",
            content=content,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=12),
            source_memory_ids=[m.id for m in recent_items[:10]],
        )
        self._emit_nudge(nudge)
        return nudge

    def scan_deadlines(self) -> list[Nudge]:
        """
        Scan memories for upcoming deadlines and generate warnings.

        Returns:
            List of deadline warning nudges
        """
        import uuid

        nudges = []
        now = datetime.now()

        # Look for memories with deadline-related content
        deadline_memories = self.memory.recall("deadline due date", limit=20)

        for mem in deadline_memories:
            # Skip if we already nudged about this memory recently
            existing = [n for n in self._nudges if mem.id in n.source_memory_ids and n.is_active()]
            if existing:
                continue

            # Determine urgency from content and tags
            priority = NudgePriority.LOW
            if any(w in mem.content.lower() for w in ("urgent", "asap", "today", "tomorrow")):
                priority = NudgePriority.URGENT
            elif any(w in mem.content.lower() for w in ("this week", "soon", "upcoming")):
                priority = NudgePriority.HIGH
            elif "deadline" in mem.tags:
                priority = NudgePriority.MEDIUM

            nudge = Nudge(
                id=str(uuid.uuid4()),
                nudge_type=NudgeType.DEADLINE_WARNING,
                priority=priority,
                title=f"Deadline: {mem.content[:60]}",
                content=mem.content,
                created_at=now,
                expires_at=now + timedelta(days=1),
                source_memory_ids=[mem.id],
            )
            nudges.append(nudge)
            self._emit_nudge(nudge)

        return nudges

    def prepare_meeting_context(self, person_or_topic: str) -> Nudge:
        """
        Prepare context for an upcoming meeting by recalling
        relevant memories about the person or topic.

        Args:
            person_or_topic: Person name or meeting topic

        Returns:
            A nudge with relevant context
        """
        import uuid

        # Recall memories about this person/topic
        memories = self.memory.recall(person_or_topic, limit=15)

        if not memories:
            content = f"No prior context found for '{person_or_topic}'."
        else:
            context_items = []
            for mem in memories:
                date_str = mem.created_at.strftime("%Y-%m-%d")
                context_items.append(f"[{date_str}] {mem.content[:200]}")
            raw_context = "\n".join(context_items)

            if self.cognitive:
                try:
                    prompt = (
                        f"Prepare a brief meeting context summary about '{person_or_topic}'. "
                        f"What do I need to remember? What was last discussed?\n\n"
                        f"Past interactions:\n{raw_context}"
                    )
                    content = self.cognitive.think(prompt)
                except Exception:
                    content = f"Context for '{person_or_topic}':\n{raw_context}"
            else:
                content = f"Context for '{person_or_topic}':\n{raw_context}"

        nudge = Nudge(
            id=str(uuid.uuid4()),
            nudge_type=NudgeType.MEETING_PREP,
            priority=NudgePriority.MEDIUM,
            title=f"Meeting prep: {person_or_topic}",
            content=content,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=4),
            source_memory_ids=[m.id for m in memories],
        )
        self._emit_nudge(nudge)
        return nudge

    def scan_follow_ups(self) -> list[Nudge]:
        """
        Scan for conversations that may need follow-up.

        Looks for patterns like "I'll get back to you", "let me check",
        "remind me", etc. in recent conversations.

        Returns:
            List of follow-up nudges
        """
        import uuid

        follow_up_phrases = [
            "follow up",
            "get back to",
            "check on",
            "remind me",
            "let me know",
            "circle back",
            "i'll send",
            "need to",
            "don't forget",
            "todo",
            "action item",
        ]

        recent = self.memory.store.list_all()
        week_ago = datetime.now() - timedelta(days=7)
        recent_convos = [
            m
            for m in recent
            if m.created_at >= week_ago and m.memory_type.value in ("episodic", "active")
        ]

        nudges = []
        for mem in recent_convos:
            content_lower = mem.content.lower()
            matched = [p for p in follow_up_phrases if p in content_lower]
            if not matched:
                continue

            # Skip if already nudged
            existing = [
                n
                for n in self._nudges
                if mem.id in n.source_memory_ids
                and n.nudge_type == NudgeType.FOLLOW_UP
                and n.is_active()
            ]
            if existing:
                continue

            age_days = (datetime.now() - mem.created_at).days
            priority = NudgePriority.HIGH if age_days >= 3 else NudgePriority.MEDIUM

            nudge = Nudge(
                id=str(uuid.uuid4()),
                nudge_type=NudgeType.FOLLOW_UP,
                priority=priority,
                title=f"Follow up ({age_days}d ago)",
                content=f"You may need to follow up on: {mem.content[:200]}",
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=2),
                source_memory_ids=[mem.id],
                metadata={"matched_phrases": matched, "age_days": age_days},
            )
            nudges.append(nudge)
            self._emit_nudge(nudge)

        return nudges

    def generate_context_nudge(self, user_input: str) -> Nudge | None:
        """
        Generate a contextual nudge based on what the user is currently
        talking about. Called during conversation to proactively surface
        relevant past context.

        Args:
            user_input: Current user message

        Returns:
            A context recall nudge if relevant context exists, else None
        """
        import uuid

        # Search for related past memories
        related = self.memory.recall(user_input, limit=5, min_confidence=0.7)
        if not related:
            return None

        # Filter to memories that are at least 1 day old (not just current convo)
        day_ago = datetime.now() - timedelta(days=1)
        old_related = [m for m in related if m.created_at < day_ago]
        if not old_related:
            return None

        # Build context summary
        items = []
        for mem in old_related[:3]:
            date_str = mem.created_at.strftime("%b %d")
            items.append(f"[{date_str}] {mem.content[:150]}")

        content = "You previously discussed related topics:\n" + "\n".join(items)

        nudge = Nudge(
            id=str(uuid.uuid4()),
            nudge_type=NudgeType.CONTEXT_RECALL,
            priority=NudgePriority.LOW,
            title="Related context found",
            content=content,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
            source_memory_ids=[m.id for m in old_related[:3]],
        )
        # Don't auto-emit — let caller decide whether to show
        self._nudges.append(nudge)
        self._save_nudges()
        return nudge

    # =========================================================================
    # Scheduled Runner
    # =========================================================================

    def run_scheduled_checks(self) -> list[Nudge]:
        """Run all due scheduled checks and return generated nudges."""
        results = []
        for check in self._checks:
            if not check.is_due():
                continue

            logger.debug(f"Running scheduled check: {check.name}")
            try:
                if check.name == "deadline_scan":
                    results.extend(self.scan_deadlines())
                elif check.name == "follow_up_scan":
                    results.extend(self.scan_follow_ups())
                elif check.name == "context_refresh":
                    pass  # Context nudges are generated on-demand
                check.last_run = datetime.now()
            except Exception as e:
                logger.error(f"Scheduled check '{check.name}' failed: {e}")

        return results

    def start_background(self, interval_seconds: int = 300) -> None:
        """
        Start background proactive scanning.

        Args:
            interval_seconds: How often to run checks (default: 5 minutes)
        """
        if self._running:
            logger.warning("Proactive engine already running")
            return

        self._running = True

        def _loop():
            logger.info("Proactive background loop started")
            while self._running:
                try:
                    self.run_scheduled_checks()
                except Exception as e:
                    logger.error(f"Proactive loop error: {e}")

                # Sleep in small increments so we can stop quickly
                for _ in range(interval_seconds):
                    if not self._running:
                        break
                    import time

                    time.sleep(1)
            logger.info("Proactive background loop stopped")

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop_background(self) -> None:
        """Stop background proactive scanning."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    # =========================================================================
    # Nudge Management
    # =========================================================================

    def get_active_nudges(self) -> list[Nudge]:
        """Get all active (non-dismissed, non-expired) nudges."""
        return [n for n in self._nudges if n.is_active()]

    def get_nudges_by_type(self, nudge_type: NudgeType) -> list[Nudge]:
        """Get active nudges of a specific type."""
        return [n for n in self.get_active_nudges() if n.nudge_type == nudge_type]

    def get_nudges_by_priority(self, min_priority: NudgePriority) -> list[Nudge]:
        """Get active nudges at or above a priority level."""
        priority_order = [
            NudgePriority.LOW,
            NudgePriority.MEDIUM,
            NudgePriority.HIGH,
            NudgePriority.URGENT,
        ]
        min_idx = priority_order.index(min_priority)
        return [n for n in self.get_active_nudges() if priority_order.index(n.priority) >= min_idx]

    def dismiss_nudge(self, nudge_id: str) -> bool:
        """Dismiss a nudge by ID."""
        for n in self._nudges:
            if n.id == nudge_id:
                n.dismiss()
                self._save_nudges()
                return True
        return False

    def act_on_nudge(self, nudge_id: str) -> bool:
        """Mark a nudge as acted upon."""
        for n in self._nudges:
            if n.id == nudge_id:
                n.mark_acted()
                self._save_nudges()
                return True
        return False

    def dismiss_all(self) -> int:
        """Dismiss all active nudges. Returns count dismissed."""
        count = 0
        for n in self._nudges:
            if n.is_active():
                n.dismiss()
                count += 1
        if count > 0:
            self._save_nudges()
        return count

    def get_statistics(self) -> dict[str, Any]:
        """Get proactive engine statistics."""
        active = self.get_active_nudges()
        by_type: dict[str, int] = {}
        by_priority: dict[str, int] = {}
        for n in active:
            by_type[n.nudge_type.value] = by_type.get(n.nudge_type.value, 0) + 1
            by_priority[n.priority.value] = by_priority.get(n.priority.value, 0) + 1

        return {
            "total_nudges": len(self._nudges),
            "active_nudges": len(active),
            "by_type": by_type,
            "by_priority": by_priority,
            "background_running": self._running,
            "checks": [
                {
                    "name": c.name,
                    "interval_minutes": c.interval_minutes,
                    "last_run": c.last_run.isoformat() if c.last_run else None,
                    "enabled": c.enabled,
                }
                for c in self._checks
            ],
        }
