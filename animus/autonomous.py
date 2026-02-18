"""
Autonomous Action System

Enables Animus to independently take actions on the user's behalf.
Bridges the proactive engine (which detects what needs doing) with
the cognitive layer and tool registry (which can actually do things).

Safety model:
- Actions are classified by risk level (OBSERVE → NOTIFY → ACT → EXECUTE)
- Each level has a configurable policy (auto / approve / deny)
- All actions are logged to an append-only audit trail
- The user can review, pause, or revoke autonomy at any time
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from animus.logging import get_logger

if TYPE_CHECKING:
    from animus.cognitive import CognitiveLayer
    from animus.proactive import Nudge
    from animus.tools import ToolRegistry

logger = get_logger("autonomous")


# =============================================================================
# Action Classification
# =============================================================================


class ActionLevel(Enum):
    """Risk levels for autonomous actions, from safest to most powerful."""

    OBSERVE = "observe"  # Read-only: search memory, check status
    NOTIFY = "notify"  # Alert the user: send nudge, update dashboard
    ACT = "act"  # Modify Animus state: save memory, create entity
    EXECUTE = "execute"  # External effects: run commands, HTTP requests


class ActionPolicy(Enum):
    """How to handle actions at a given level."""

    AUTO = "auto"  # Execute without asking
    APPROVE = "approve"  # Queue for user approval before executing
    DENY = "deny"  # Never execute autonomously


class ActionStatus(Enum):
    """Lifecycle of an autonomous action."""

    PLANNED = "planned"  # Identified by the engine, not yet executed
    APPROVED = "approved"  # User approved (for APPROVE-policy actions)
    EXECUTING = "executing"  # Currently running
    COMPLETED = "completed"  # Finished successfully
    FAILED = "failed"  # Finished with error
    DENIED = "denied"  # User denied or policy blocked
    EXPIRED = "expired"  # Approval window passed


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class AutonomousAction:
    """A single autonomous action planned or executed by Animus."""

    id: str
    level: ActionLevel
    title: str
    description: str
    tool_name: str | None = None  # Tool to execute (None for cognitive-only)
    tool_params: dict[str, Any] = field(default_factory=dict)
    status: ActionStatus = ActionStatus.PLANNED
    result: str | None = None
    error: str | None = None
    source_nudge_id: str | None = None  # Nudge that triggered this action
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: datetime | None = None
    expires_at: datetime | None = None

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "level": self.level.value,
            "title": self.title,
            "description": self.description,
            "tool_name": self.tool_name,
            "tool_params": self.tool_params,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "source_nudge_id": self.source_nudge_id,
            "created_at": self.created_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AutonomousAction:
        return cls(
            id=data["id"],
            level=ActionLevel(data["level"]),
            title=data["title"],
            description=data["description"],
            tool_name=data.get("tool_name"),
            tool_params=data.get("tool_params", {}),
            status=ActionStatus(data["status"]),
            result=data.get("result"),
            error=data.get("error"),
            source_nudge_id=data.get("source_nudge_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            executed_at=(
                datetime.fromisoformat(data["executed_at"]) if data.get("executed_at") else None
            ),
            expires_at=(
                datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
            ),
        )


# =============================================================================
# Audit Log
# =============================================================================


class ActionLog:
    """Append-only audit trail for all autonomous actions."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._entries: list[AutonomousAction] = []
        self._load()

    def _log_path(self) -> Path:
        return self.data_dir / "action_log.json"

    def _load(self) -> None:
        path = self._log_path()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self._entries = [AutonomousAction.from_dict(e) for e in data]
            except Exception as e:
                logger.error(f"Failed to load action log: {e}")

    def _save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Keep last 500 entries
        to_save = self._entries[-500:]
        self._log_path().write_text(json.dumps([e.to_dict() for e in to_save], indent=2))

    def record(self, action: AutonomousAction) -> None:
        """Record an action to the log."""
        self._entries.append(action)
        self._save()
        logger.info(
            f"Action logged: [{action.level.value}] {action.title} -> {action.status.value}"
        )

    def update(self, action: AutonomousAction) -> None:
        """Update an existing action in the log."""
        for i, entry in enumerate(self._entries):
            if entry.id == action.id:
                self._entries[i] = action
                self._save()
                return
        # If not found, append
        self.record(action)

    def get_recent(self, limit: int = 20) -> list[AutonomousAction]:
        return list(reversed(self._entries[-limit:]))

    def get_pending_approval(self) -> list[AutonomousAction]:
        return [a for a in self._entries if a.status == ActionStatus.PLANNED and not a.is_expired()]

    def get_by_id(self, action_id: str) -> AutonomousAction | None:
        for a in self._entries:
            if a.id == action_id:
                return a
        return None

    def get_statistics(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        by_level: dict[str, int] = {}
        for a in self._entries:
            by_status[a.status.value] = by_status.get(a.status.value, 0) + 1
            by_level[a.level.value] = by_level.get(a.level.value, 0) + 1
        return {
            "total_actions": len(self._entries),
            "by_status": by_status,
            "by_level": by_level,
            "pending_approval": len(self.get_pending_approval()),
        }


# =============================================================================
# Autonomous Executor
# =============================================================================


class AutonomousExecutor:
    """
    Bridges the proactive engine with actual action execution.

    The executor:
    1. Receives nudges from the proactive engine
    2. Decides what action to take (using the cognitive layer)
    3. Checks the action against the configured policy
    4. Executes allowed actions or queues them for approval
    5. Logs everything to the audit trail
    """

    # Default policies: conservative out of the box
    DEFAULT_POLICIES: dict[ActionLevel, ActionPolicy] = {
        ActionLevel.OBSERVE: ActionPolicy.AUTO,
        ActionLevel.NOTIFY: ActionPolicy.AUTO,
        ActionLevel.ACT: ActionPolicy.APPROVE,
        ActionLevel.EXECUTE: ActionPolicy.DENY,
    }

    def __init__(
        self,
        data_dir: Path,
        cognitive: CognitiveLayer | None = None,
        tools: ToolRegistry | None = None,
        policies: dict[ActionLevel, ActionPolicy] | None = None,
        on_approval_needed: Callable[[AutonomousAction], None] | None = None,
        on_action_completed: Callable[[AutonomousAction], None] | None = None,
    ):
        self.cognitive = cognitive
        self.tools = tools
        self.policies = policies or dict(self.DEFAULT_POLICIES)
        self._on_approval_needed = on_approval_needed
        self._on_action_completed = on_action_completed
        self.log = ActionLog(data_dir)

        logger.info(
            "AutonomousExecutor initialized with policies: "
            + ", ".join(f"{k.value}={v.value}" for k, v in self.policies.items())
        )

    # -----------------------------------------------------------------
    # Policy Checking
    # -----------------------------------------------------------------

    def get_policy(self, level: ActionLevel) -> ActionPolicy:
        """Get the policy for a given action level."""
        return self.policies.get(level, ActionPolicy.DENY)

    def set_policy(self, level: ActionLevel, policy: ActionPolicy) -> None:
        """Update the policy for an action level."""
        old = self.policies.get(level)
        self.policies[level] = policy
        logger.info(f"Policy updated: {level.value} {old} -> {policy.value}")

    # -----------------------------------------------------------------
    # Action Planning
    # -----------------------------------------------------------------

    def plan_action_for_nudge(self, nudge: Nudge) -> AutonomousAction | None:
        """
        Use the cognitive layer to decide what action to take for a nudge.

        Returns an AutonomousAction if the LLM suggests one, or None if
        no action is warranted.
        """
        if not self.cognitive:
            return None

        # Build a prompt asking the LLM what to do
        available_tools = ""
        if self.tools:
            available_tools = "\n\nAvailable tools:\n" + "\n".join(
                f"- {t.name}: {t.description}" for t in self.tools.list_tools()
            )

        prompt = (
            "You are the autonomous reasoning engine for a personal AI assistant. "
            "A proactive check has generated the following nudge:\n\n"
            f"Type: {nudge.nudge_type.value}\n"
            f"Priority: {nudge.priority.value}\n"
            f"Title: {nudge.title}\n"
            f"Content: {nudge.content}\n"
            f"{available_tools}\n\n"
            "Decide whether to take an autonomous action. Respond with EXACTLY one of:\n"
            "1. NO_ACTION - if no action is needed (just a notification)\n"
            "2. A JSON object with this structure:\n"
            '{"action": "<short title>", "description": "<what and why>", '
            '"level": "<observe|notify|act|execute>", '
            '"tool": "<tool_name or null>", "params": {<tool params or empty>}}\n\n'
            "Be conservative. Only suggest actions that are clearly helpful. "
            "Prefer lower risk levels when possible."
        )

        try:
            response = self.cognitive.think(prompt)
        except Exception as e:
            logger.warning(f"Failed to plan action for nudge: {e}")
            return None

        # Parse the response
        if "NO_ACTION" in response:
            return None

        try:
            # Extract JSON from response (may have surrounding text)
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if not json_match:
                return None
            data = json.loads(json_match.group())

            level_str = data.get("level", "notify")
            try:
                level = ActionLevel(level_str)
            except ValueError:
                level = ActionLevel.NOTIFY

            return AutonomousAction(
                id=str(uuid.uuid4()),
                level=level,
                title=data.get("action", nudge.title),
                description=data.get("description", nudge.content),
                tool_name=data.get("tool"),
                tool_params=data.get("params", {}),
                source_nudge_id=nudge.id,
                expires_at=datetime.now() + timedelta(hours=4),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Could not parse action plan from LLM response: {e}")
            return None

    # -----------------------------------------------------------------
    # Action Execution
    # -----------------------------------------------------------------

    def execute_action(self, action: AutonomousAction) -> AutonomousAction:
        """
        Execute a single action, respecting policies.

        Returns the updated action with result/error populated.
        """
        policy = self.get_policy(action.level)

        if policy == ActionPolicy.DENY:
            action.status = ActionStatus.DENIED
            action.error = f"Policy denies autonomous {action.level.value} actions"
            self.log.update(action)
            return action

        if policy == ActionPolicy.APPROVE and action.status != ActionStatus.APPROVED:
            action.status = ActionStatus.PLANNED
            self.log.record(action)
            if self._on_approval_needed:
                self._on_approval_needed(action)
            logger.info(f"Action queued for approval: {action.title}")
            return action

        # AUTO policy or already APPROVED — execute
        action.status = ActionStatus.EXECUTING
        action.executed_at = datetime.now()
        self.log.update(action)

        try:
            if action.tool_name and self.tools:
                # Execute via tool registry
                result = self.tools.execute(action.tool_name, action.tool_params)
                if result.success:
                    action.status = ActionStatus.COMPLETED
                    action.result = str(result.output)[:2000]
                else:
                    action.status = ActionStatus.FAILED
                    action.error = result.error
            elif self.cognitive:
                # Cognitive-only action (e.g. synthesize a summary)
                result_text = self.cognitive.think(action.description)
                action.status = ActionStatus.COMPLETED
                action.result = result_text[:2000]
            else:
                action.status = ActionStatus.FAILED
                action.error = "No tools or cognitive layer available"
        except Exception as e:
            action.status = ActionStatus.FAILED
            action.error = str(e)
            logger.error(f"Action execution failed: {action.title}: {e}")

        self.log.update(action)

        if self._on_action_completed:
            try:
                self._on_action_completed(action)
            except Exception as e:
                logger.error(f"Action completion callback error: {e}")

        logger.info(
            f"Action {action.status.value}: {action.title}"
            + (f" -> {action.result[:80]}" if action.result else "")
        )
        return action

    # -----------------------------------------------------------------
    # Approval Management
    # -----------------------------------------------------------------

    def approve_action(self, action_id: str) -> AutonomousAction | None:
        """Approve a pending action and execute it."""
        action = self.log.get_by_id(action_id)
        if not action:
            return None
        if action.status != ActionStatus.PLANNED:
            return action
        if action.is_expired():
            action.status = ActionStatus.EXPIRED
            self.log.update(action)
            return action

        action.status = ActionStatus.APPROVED
        self.log.update(action)
        return self.execute_action(action)

    def deny_action(self, action_id: str) -> AutonomousAction | None:
        """Deny a pending action."""
        action = self.log.get_by_id(action_id)
        if not action:
            return None
        action.status = ActionStatus.DENIED
        action.error = "Denied by user"
        self.log.update(action)
        return action

    # -----------------------------------------------------------------
    # Proactive Integration
    # -----------------------------------------------------------------

    def handle_nudge(self, nudge: Nudge) -> AutonomousAction | None:
        """
        Process a nudge from the proactive engine.

        This is the main entry point called by the proactive engine's
        background loop. It plans an action, checks policy, and either
        executes or queues for approval.
        """
        action = self.plan_action_for_nudge(nudge)
        if action is None:
            return None

        self.log.record(action)
        return self.execute_action(action)

    def process_pending(self) -> list[AutonomousAction]:
        """Expire any timed-out pending actions."""
        expired = []
        for action in self.log.get_pending_approval():
            if action.is_expired():
                action.status = ActionStatus.EXPIRED
                self.log.update(action)
                expired.append(action)
        return expired

    # -----------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------

    def get_pending_actions(self) -> list[AutonomousAction]:
        """Get actions waiting for user approval."""
        return self.log.get_pending_approval()

    def get_recent_actions(self, limit: int = 20) -> list[AutonomousAction]:
        """Get recent action history."""
        return self.log.get_recent(limit)

    def get_statistics(self) -> dict[str, Any]:
        """Get executor statistics."""
        stats = self.log.get_statistics()
        stats["policies"] = {k.value: v.value for k, v in self.policies.items()}
        return stats
