"""Automation engine — evaluates rules against messages and events with SQLite persistence."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from animus_bootstrap.gateway.models import GatewayMessage
from animus_bootstrap.intelligence.automations.actions import execute_action
from animus_bootstrap.intelligence.automations.conditions import evaluate_conditions
from animus_bootstrap.intelligence.automations.models import (
    ActionConfig,
    AutomationResult,
    AutomationRule,
    Condition,
    TriggerConfig,
)
from animus_bootstrap.intelligence.automations.triggers import evaluate_trigger

logger = logging.getLogger(__name__)


class AutomationEngine:
    """Evaluates automation rules against messages and events."""

    def __init__(self, db_path: Path | str) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        self._rules: dict[str, AutomationRule] = {}
        self._history: list[AutomationResult] = []
        self._load_rules()

    def _init_db(self) -> None:
        """Create automations table and execution_log table."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS automations (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                enabled BOOLEAN NOT NULL DEFAULT 1,
                trigger_config TEXT NOT NULL,
                conditions TEXT NOT NULL,
                actions TEXT NOT NULL,
                cooldown_seconds INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                last_fired TEXT
            );
            CREATE TABLE IF NOT EXISTS automation_log (
                id TEXT PRIMARY KEY,
                rule_id TEXT NOT NULL,
                rule_name TEXT NOT NULL,
                triggered BOOLEAN NOT NULL,
                actions_executed TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                error TEXT
            );
        """)
        self._conn.commit()

    def _load_rules(self) -> None:
        """Load all rules from DB into memory."""
        cursor = self._conn.execute("SELECT * FROM automations")
        for row in cursor.fetchall():
            rule = self._row_to_rule(row)
            self._rules[rule.id] = rule

    @staticmethod
    def _row_to_rule(row: sqlite3.Row) -> AutomationRule:
        """Convert a database row to an AutomationRule."""
        trigger_data = json.loads(row["trigger_config"])
        conditions_data = json.loads(row["conditions"])
        actions_data = json.loads(row["actions"])

        return AutomationRule(
            id=row["id"],
            name=row["name"],
            enabled=bool(row["enabled"]),
            trigger=TriggerConfig(
                type=trigger_data["type"],
                params=trigger_data.get("params", {}),
            ),
            conditions=[
                Condition(type=c["type"], params=c.get("params", {})) for c in conditions_data
            ],
            actions=[
                ActionConfig(type=a["type"], params=a.get("params", {})) for a in actions_data
            ],
            cooldown_seconds=row["cooldown_seconds"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_fired=(datetime.fromisoformat(row["last_fired"]) if row["last_fired"] else None),
        )

    def _rule_to_row(self, rule: AutomationRule) -> dict[str, Any]:
        """Convert an AutomationRule to a dict suitable for DB insert."""
        return {
            "id": rule.id,
            "name": rule.name,
            "enabled": rule.enabled,
            "trigger_config": json.dumps(
                {"type": rule.trigger.type, "params": rule.trigger.params}
            ),
            "conditions": json.dumps(
                [{"type": c.type, "params": c.params} for c in rule.conditions]
            ),
            "actions": json.dumps([{"type": a.type, "params": a.params} for a in rule.actions]),
            "cooldown_seconds": rule.cooldown_seconds,
            "created_at": rule.created_at.isoformat(),
            "last_fired": rule.last_fired.isoformat() if rule.last_fired else None,
        }

    def add_rule(self, rule: AutomationRule) -> None:
        """Add/update a rule (persists to DB)."""
        row = self._rule_to_row(rule)
        self._conn.execute(
            """INSERT OR REPLACE INTO automations
               (id, name, enabled, trigger_config, conditions, actions,
                cooldown_seconds, created_at, last_fired)
               VALUES (:id, :name, :enabled, :trigger_config, :conditions,
                       :actions, :cooldown_seconds, :created_at, :last_fired)""",
            row,
        )
        self._conn.commit()
        self._rules[rule.id] = rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID. Returns True if found."""
        if rule_id not in self._rules:
            return False
        self._conn.execute("DELETE FROM automations WHERE id = ?", (rule_id,))
        self._conn.commit()
        del self._rules[rule_id]
        return True

    def get_rule(self, rule_id: str) -> AutomationRule | None:
        """Get a rule by ID."""
        return self._rules.get(rule_id)

    def list_rules(self) -> list[AutomationRule]:
        """Return all rules."""
        return list(self._rules.values())

    def enable_rule(self, rule_id: str) -> bool:
        """Enable a rule."""
        rule = self._rules.get(rule_id)
        if rule is None:
            return False
        rule.enabled = True
        self._conn.execute("UPDATE automations SET enabled = 1 WHERE id = ?", (rule_id,))
        self._conn.commit()
        return True

    def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule."""
        rule = self._rules.get(rule_id)
        if rule is None:
            return False
        rule.enabled = False
        self._conn.execute("UPDATE automations SET enabled = 0 WHERE id = ?", (rule_id,))
        self._conn.commit()
        return True

    async def evaluate_message(
        self,
        message: GatewayMessage,
        context: dict[str, Any] | None = None,
    ) -> list[AutomationResult]:
        """Evaluate all enabled message-triggered rules against a message.

        Returns list of results (only for rules that fired).
        Respects cooldown -- skip if last_fired + cooldown > now.
        """
        if context is None:
            context = {}

        results: list[AutomationResult] = []
        now = datetime.now(UTC)

        for rule in self._rules.values():
            if not rule.enabled:
                continue
            if rule.trigger.type != "message":
                continue

            # Cooldown check
            if (
                rule.cooldown_seconds > 0
                and rule.last_fired is not None
                and (now - rule.last_fired).total_seconds() < rule.cooldown_seconds
            ):
                continue

            # Trigger check
            if not evaluate_trigger(rule.trigger, message=message):
                continue

            # Conditions check
            if not evaluate_conditions(rule.conditions, message=message):
                continue

            # All checks passed — execute actions
            result = await self._execute_rule(rule, context, now)
            results.append(result)

        return results

    async def evaluate_event(
        self,
        event: dict,
        context: dict[str, Any] | None = None,
    ) -> list[AutomationResult]:
        """Evaluate all enabled event-triggered rules against an event."""
        if context is None:
            context = {}

        results: list[AutomationResult] = []
        now = datetime.now(UTC)

        for rule in self._rules.values():
            if not rule.enabled:
                continue
            if rule.trigger.type not in ("event", "webhook"):
                continue

            # Cooldown check
            if (
                rule.cooldown_seconds > 0
                and rule.last_fired is not None
                and (now - rule.last_fired).total_seconds() < rule.cooldown_seconds
            ):
                continue

            # Trigger check
            if not evaluate_trigger(rule.trigger, event=event):
                continue

            result = await self._execute_rule(rule, context, now)
            results.append(result)

        return results

    async def _execute_rule(
        self,
        rule: AutomationRule,
        context: dict[str, Any],
        now: datetime,
    ) -> AutomationResult:
        """Execute all actions for a rule and record the result."""
        actions_executed: list[str] = []
        error: str | None = None

        for action in rule.actions:
            try:
                desc = await execute_action(action, context)
                actions_executed.append(desc)
            except (OSError, ConnectionError, RuntimeError, ValueError, TimeoutError) as exc:
                error = f"Action {action.type} failed: {exc}"
                logger.warning("Automation %s action error: %s", rule.name, exc)
                break

        # Update last_fired
        rule.last_fired = now
        self._conn.execute(
            "UPDATE automations SET last_fired = ? WHERE id = ?",
            (now.isoformat(), rule.id),
        )
        self._conn.commit()

        result = AutomationResult(
            rule_id=rule.id,
            rule_name=rule.name,
            triggered=True,
            actions_executed=actions_executed,
            timestamp=now,
            error=error,
        )

        # Persist to log
        self._log_result(result)
        self._history.append(result)

        return result

    def _log_result(self, result: AutomationResult) -> None:
        """Persist an AutomationResult to the automation_log table."""
        self._conn.execute(
            """INSERT INTO automation_log
               (id, rule_id, rule_name, triggered, actions_executed, timestamp, error)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                result.rule_id,
                result.rule_name,
                result.triggered,
                json.dumps(result.actions_executed),
                result.timestamp.isoformat(),
                result.error,
            ),
        )
        self._conn.commit()

    def get_history(self, rule_id: str | None = None, limit: int = 50) -> list[AutomationResult]:
        """Return execution history, optionally filtered by rule."""
        if rule_id is not None:
            cursor = self._conn.execute(
                "SELECT * FROM automation_log WHERE rule_id = ? ORDER BY timestamp DESC LIMIT ?",
                (rule_id, limit),
            )
        else:
            cursor = self._conn.execute(
                "SELECT * FROM automation_log ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )

        results: list[AutomationResult] = []
        for row in cursor.fetchall():
            results.append(
                AutomationResult(
                    rule_id=row["rule_id"],
                    rule_name=row["rule_name"],
                    triggered=bool(row["triggered"]),
                    actions_executed=json.loads(row["actions_executed"]),
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    error=row["error"],
                )
            )
        return results

    def clear_history(self) -> None:
        """Clear all execution history."""
        self._conn.execute("DELETE FROM automation_log")
        self._conn.commit()
        self._history.clear()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
