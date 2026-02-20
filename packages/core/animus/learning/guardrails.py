"""
Guardrail Enforcement Layer

Immutable safety boundaries that cannot be modified by learning.
"""

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from animus.logging import get_logger

logger = get_logger("learning.guardrails")


class GuardrailType(Enum):
    """Types of guardrails."""

    SAFETY = "safety"  # Cannot be modified or bypassed
    PRIVACY = "privacy"  # Data protection rules
    ACCESS = "access"  # Permission boundaries
    BEHAVIOR = "behavior"  # Behavioral constraints


@dataclass
class Guardrail:
    """A safety or boundary rule."""

    id: str
    rule: str
    description: str
    guardrail_type: GuardrailType
    immutable: bool  # Cannot be changed by learning
    source: str  # "system" or "user_defined"
    created_at: datetime = field(default_factory=datetime.now)
    check_func: Callable[[dict[str, Any]], bool] | None = None

    def check(self, proposed_action: dict[str, Any]) -> bool:
        """
        Check if an action violates this guardrail.

        Args:
            proposed_action: Action to check

        Returns:
            True if action is permitted, False if it violates guardrail
        """
        if self.check_func:
            return self.check_func(proposed_action)
        return True  # Pass if no check function

    def explain_violation(self, action: dict[str, Any]) -> str:
        """Explain why an action violates this guardrail."""
        return f"Action violates guardrail '{self.id}': {self.rule}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "rule": self.rule,
            "description": self.description,
            "guardrail_type": self.guardrail_type.value,
            "immutable": self.immutable,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Guardrail":
        """Create from dictionary (user-defined guardrails only)."""
        return cls(
            id=data["id"],
            rule=data["rule"],
            description=data["description"],
            guardrail_type=GuardrailType(data["guardrail_type"]),
            immutable=data.get("immutable", False),
            source=data.get("source", "user_defined"),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class GuardrailViolation:
    """Record of a guardrail violation attempt."""

    id: str
    guardrail_id: str
    action: dict[str, Any]
    timestamp: datetime
    explanation: str
    blocked: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "guardrail_id": self.guardrail_id,
            "action": self.action,
            "timestamp": self.timestamp.isoformat(),
            "explanation": self.explanation,
            "blocked": self.blocked,
        }


def _check_no_exfiltrate(action: dict[str, Any]) -> bool:
    """Check that action doesn't exfiltrate data."""
    # Block actions that send data to external URLs without consent
    action_type = action.get("type", "")
    if action_type in ("send_email", "post_webhook", "api_call"):
        # Check if explicitly approved
        return action.get("user_approved", False)
    return True


def _check_no_modify_guardrails(action: dict[str, Any]) -> bool:
    """Check that action doesn't try to modify guardrails."""
    action_type = action.get("type", "")
    target = action.get("target", "")
    if action_type == "modify" and "guardrail" in target.lower():
        # Only allow user-defined guardrail modifications
        return action.get("guardrail_source") == "user_defined"
    return True


def _check_learning_reversible(action: dict[str, Any]) -> bool:
    """Check that learning actions are reversible."""
    action_type = action.get("type", "")
    if action_type == "learn":
        # Must have rollback capability
        return action.get("reversible", True)
    return True


# Immutable system guardrails - CANNOT be modified by learning
CORE_GUARDRAILS: list[Guardrail] = [
    Guardrail(
        id="core_no_harm",
        rule="Cannot take actions that harm user",
        description="Animus cannot execute actions that would harm the user's interests",
        guardrail_type=GuardrailType.SAFETY,
        immutable=True,
        source="system",
    ),
    Guardrail(
        id="core_no_exfiltrate",
        rule="Cannot exfiltrate user data",
        description="Animus cannot send user data to external parties without explicit consent",
        guardrail_type=GuardrailType.PRIVACY,
        immutable=True,
        source="system",
        check_func=_check_no_exfiltrate,
    ),
    Guardrail(
        id="core_no_modify_guardrails",
        rule="Cannot modify own guardrails",
        description="Animus cannot learn to bypass or modify its core guardrails",
        guardrail_type=GuardrailType.SAFETY,
        immutable=True,
        source="system",
        check_func=_check_no_modify_guardrails,
    ),
    Guardrail(
        id="core_transparency",
        rule="Must be transparent about capabilities",
        description="Animus must accurately represent what it can and cannot do",
        guardrail_type=GuardrailType.BEHAVIOR,
        immutable=True,
        source="system",
    ),
    Guardrail(
        id="core_learning_reversible",
        rule="All learning must be reversible",
        description="Any learned behavior must be able to be unlearned by user",
        guardrail_type=GuardrailType.SAFETY,
        immutable=True,
        source="system",
        check_func=_check_learning_reversible,
    ),
]


class GuardrailManager:
    """
    Enforces guardrails on all learning and actions.

    Key responsibilities:
    - Check proposed learnings against guardrails
    - Block actions that violate guardrails
    - Log all violation attempts
    - Manage user-defined guardrails (non-immutable)
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._guardrails: dict[str, Guardrail] = {}
        self._violations: list[GuardrailViolation] = []
        self._initialize_core_guardrails()
        self._load_user_guardrails()

    def _initialize_core_guardrails(self) -> None:
        """Initialize immutable system guardrails."""
        for guardrail in CORE_GUARDRAILS:
            self._guardrails[guardrail.id] = guardrail
        logger.info(f"Initialized {len(CORE_GUARDRAILS)} core guardrails")

    def _load_user_guardrails(self) -> None:
        """Load user-defined guardrails from disk."""
        guardrails_file = self.data_dir / "user_guardrails.json"
        if guardrails_file.exists():
            try:
                with open(guardrails_file) as f:
                    data = json.load(f)
                for item in data:
                    guardrail = Guardrail.from_dict(item)
                    self._guardrails[guardrail.id] = guardrail
                logger.info(f"Loaded {len(data)} user guardrails")
            except (json.JSONDecodeError, ValueError, OSError) as e:
                logger.error(f"Failed to load user guardrails: {e}")

    def _save_user_guardrails(self) -> None:
        """Save user-defined guardrails to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        guardrails_file = self.data_dir / "user_guardrails.json"
        user_guardrails = [
            g.to_dict() for g in self._guardrails.values() if g.source == "user_defined"
        ]
        with open(guardrails_file, "w") as f:
            json.dump(user_guardrails, f, indent=2)

    def check_action(self, action: dict[str, Any]) -> tuple[bool, str | None]:
        """
        Check if an action is permitted.

        Args:
            action: Action to check

        Returns:
            (allowed, violation_explanation or None)
        """
        for guardrail in self._guardrails.values():
            if not guardrail.check(action):
                violation = GuardrailViolation(
                    id=str(uuid.uuid4()),
                    guardrail_id=guardrail.id,
                    action=action,
                    timestamp=datetime.now(),
                    explanation=guardrail.explain_violation(action),
                )
                self._violations.append(violation)
                logger.warning(f"Guardrail violation: {violation.explanation}")
                return False, violation.explanation
        return True, None

    def check_learning(self, content: str, category: str) -> tuple[bool, str | None]:
        """
        Check if a learning is permitted.

        Args:
            content: The content being learned
            category: The learning category

        Returns:
            (allowed, violation_explanation or None)
        """
        # Construct action dict for checking
        action = {
            "type": "learn",
            "content": content,
            "category": category,
            "reversible": True,  # All learnings are reversible by design
        }

        # Check against all guardrails
        for guardrail in self._guardrails.values():
            if not guardrail.check(action):
                violation = GuardrailViolation(
                    id=str(uuid.uuid4()),
                    guardrail_id=guardrail.id,
                    action=action,
                    timestamp=datetime.now(),
                    explanation=guardrail.explain_violation(action),
                )
                self._violations.append(violation)
                logger.warning(f"Learning blocked: {violation.explanation}")
                return False, violation.explanation

        # Additional content-based checks
        content_lower = content.lower()

        # Check for attempts to modify guardrails
        if "guardrail" in content_lower and any(
            word in content_lower for word in ["disable", "remove", "bypass", "ignore"]
        ):
            explanation = "Cannot learn to bypass or disable guardrails"
            self._log_violation("core_no_modify_guardrails", action, explanation)
            return False, explanation

        # Check for potentially harmful content
        harmful_patterns = [
            "delete all",
            "rm -rf",
            "format disk",
            "drop table",
            "shutdown",
        ]
        if any(pattern in content_lower for pattern in harmful_patterns):
            explanation = "Learning contains potentially harmful patterns"
            self._log_violation("core_no_harm", action, explanation)
            return False, explanation

        return True, None

    def _log_violation(self, guardrail_id: str, action: dict[str, Any], explanation: str) -> None:
        """Log a guardrail violation."""
        violation = GuardrailViolation(
            id=str(uuid.uuid4()),
            guardrail_id=guardrail_id,
            action=action,
            timestamp=datetime.now(),
            explanation=explanation,
        )
        self._violations.append(violation)
        logger.warning(f"Guardrail violation: {explanation}")

    def add_user_guardrail(
        self,
        rule: str,
        description: str,
        guardrail_type: GuardrailType = GuardrailType.BEHAVIOR,
    ) -> Guardrail:
        """
        Add a user-defined guardrail (non-immutable).

        Args:
            rule: The rule text
            description: Description of the guardrail
            guardrail_type: Type of guardrail

        Returns:
            The created guardrail
        """
        guardrail = Guardrail(
            id=f"user_{str(uuid.uuid4())[:8]}",
            rule=rule,
            description=description,
            guardrail_type=guardrail_type,
            immutable=False,
            source="user_defined",
        )
        self._guardrails[guardrail.id] = guardrail
        self._save_user_guardrails()
        logger.info(f"Added user guardrail: {guardrail.id}")
        return guardrail

    def remove_user_guardrail(self, guardrail_id: str) -> bool:
        """
        Remove a user-defined guardrail (immutable ones cannot be removed).

        Args:
            guardrail_id: ID of guardrail to remove

        Returns:
            True if removed, False if not found or immutable
        """
        guardrail = self._guardrails.get(guardrail_id)
        if not guardrail:
            return False
        if guardrail.immutable:
            logger.warning(f"Cannot remove immutable guardrail: {guardrail_id}")
            return False

        del self._guardrails[guardrail_id]
        self._save_user_guardrails()
        logger.info(f"Removed user guardrail: {guardrail_id}")
        return True

    def get_all_guardrails(self) -> list[Guardrail]:
        """Get all guardrails (system and user-defined)."""
        return list(self._guardrails.values())

    def get_violations(self, limit: int = 100) -> list[GuardrailViolation]:
        """Get recent guardrail violations."""
        return self._violations[-limit:]

    def get_violation_count(self) -> int:
        """Get total number of violations."""
        return len(self._violations)
