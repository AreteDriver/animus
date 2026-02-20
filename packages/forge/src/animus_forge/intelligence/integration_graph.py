"""Cross-integration event routing for Gorgon.

Wires services together: a GitHub PR triggers a workflow that runs tests,
writes a Notion page, Slacks the team, and emails the stakeholder. The
wiring between services encodes business process — that's the lock-in.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TriggerRule:
    """A rule mapping a source event to a workflow execution."""

    id: str
    source: str
    event: str
    workflow_name: str
    transform: dict[str, str] | None = None
    conditions: list[dict[str, Any]] = field(default_factory=list)
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class DispatchResult:
    """Result of dispatching a single trigger."""

    trigger_id: str
    workflow_name: str
    inputs: dict[str, Any]
    dispatched: bool
    reason: str  # "ok", "condition_not_met", "disabled", "transform_error"


@dataclass
class IntegrationChain:
    """A named sequence of trigger rules forming a pipeline."""

    id: str
    name: str
    steps: list[TriggerRule]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def _resolve_dotted_path(obj: Any, path: str) -> Any:
    """Resolve a dotted path like 'pull_request.user.login' against a nested dict.

    Args:
        obj: Root object (usually a dict).
        path: Dot-separated key path.

    Returns:
        The resolved value, or None if any segment is missing.
    """
    current = obj
    for segment in path.split("."):
        if isinstance(current, dict):
            current = current.get(segment)
        else:
            return None
        if current is None:
            return None
    return current


def _evaluate_condition(condition: dict[str, Any], payload: dict) -> bool:
    """Evaluate a single condition against a payload.

    Supports operators: equals, not_equals, contains, exists, in.

    Args:
        condition: Dict with 'field' and one operator key.
        payload: Event payload to check against.

    Returns:
        True if condition is satisfied.
    """
    field_path = condition.get("field", "")
    value = _resolve_dotted_path(payload, field_path)

    if "equals" in condition:
        return value == condition["equals"]
    if "not_equals" in condition:
        return value != condition["not_equals"]
    if "contains" in condition:
        return condition["contains"] in value if isinstance(value, (str, list)) else False
    if "exists" in condition:
        return (value is not None) == condition["exists"]
    if "in" in condition:
        return value in condition["in"] if isinstance(condition["in"], (list, tuple)) else False

    return True  # No recognized operator — pass


def _apply_transform(transform: dict[str, str], payload: dict) -> dict[str, Any]:
    """Apply a field mapping transform to an event payload.

    Args:
        transform: Mapping of {workflow_input_name: payload_dotted_path}.
        payload: Source event payload.

    Returns:
        Dict of workflow inputs.
    """
    result: dict[str, Any] = {}
    for target_key, source_path in transform.items():
        result[target_key] = _resolve_dotted_path(payload, source_path)
    return result


class IntegrationGraph:
    """Event-driven routing between external services and Gorgon workflows.

    Manages trigger rules that map source events to workflow executions,
    with optional payload transformations and conditional filtering.

    Example:
        graph = IntegrationGraph()
        graph.register_trigger(
            source="github",
            event="pr_opened",
            workflow_name="review-and-test",
            transform={"pr_title": "pull_request.title"},
            conditions=[{"field": "action", "equals": "opened"}],
        )
        results = graph.dispatch_event("github", "pr_opened", payload)
    """

    def __init__(self) -> None:
        self._triggers: dict[str, TriggerRule] = {}
        self._chains: dict[str, IntegrationChain] = {}
        self._dispatch_callback: Callable[[str, dict], Any] | None = None

    def set_dispatch_callback(self, callback: Callable[[str, dict], Any]) -> None:
        """Set the callback used to actually execute workflows.

        Args:
            callback: Function(workflow_name, inputs) -> Any.
        """
        self._dispatch_callback = callback

    def register_trigger(
        self,
        source: str,
        event: str,
        workflow_name: str,
        transform: dict[str, str] | None = None,
        conditions: list[dict[str, Any]] | None = None,
        enabled: bool = True,
    ) -> str:
        """Register a trigger rule.

        Args:
            source: Event source (e.g., "github", "slack", "webhook").
            event: Event name (e.g., "pr_opened", "message_received").
            workflow_name: Workflow to execute when triggered.
            transform: Optional field mapping {input_name: payload_path}.
            conditions: Optional conditions that must all pass.
            enabled: Whether the trigger is active.

        Returns:
            The trigger ID.
        """
        trigger_id = f"trigger-{uuid.uuid4().hex[:8]}"
        rule = TriggerRule(
            id=trigger_id,
            source=source,
            event=event,
            workflow_name=workflow_name,
            transform=transform,
            conditions=conditions or [],
            enabled=enabled,
        )
        self._triggers[trigger_id] = rule
        logger.info(
            "Registered trigger %s: %s.%s -> %s",
            trigger_id,
            source,
            event,
            workflow_name,
        )
        return trigger_id

    def remove_trigger(self, trigger_id: str) -> bool:
        """Remove a trigger rule.

        Args:
            trigger_id: ID of the trigger to remove.

        Returns:
            True if removed, False if not found.
        """
        if trigger_id in self._triggers:
            del self._triggers[trigger_id]
            return True
        return False

    def enable_trigger(self, trigger_id: str, enabled: bool = True) -> bool:
        """Enable or disable a trigger.

        Args:
            trigger_id: Trigger to modify.
            enabled: New enabled state.

        Returns:
            True if found and updated.
        """
        if trigger_id in self._triggers:
            self._triggers[trigger_id].enabled = enabled
            return True
        return False

    def dispatch_event(
        self, source: str, event: str, payload: dict[str, Any]
    ) -> list[DispatchResult]:
        """Dispatch an event to all matching triggers.

        Args:
            source: Event source.
            event: Event name.
            payload: Event payload data.

        Returns:
            List of dispatch results (one per matching trigger).
        """
        results: list[DispatchResult] = []
        matching = [t for t in self._triggers.values() if t.source == source and t.event == event]

        for trigger in matching:
            result = self._dispatch_trigger(trigger, payload)
            results.append(result)

        if results:
            logger.info(
                "Dispatched %s.%s: %d triggers matched, %d dispatched",
                source,
                event,
                len(results),
                sum(1 for r in results if r.dispatched),
            )

        return results

    def _dispatch_trigger(self, trigger: TriggerRule, payload: dict[str, Any]) -> DispatchResult:
        """Evaluate and dispatch a single trigger.

        Args:
            trigger: The trigger rule.
            payload: Event payload.

        Returns:
            DispatchResult with status.
        """
        if not trigger.enabled:
            return DispatchResult(
                trigger_id=trigger.id,
                workflow_name=trigger.workflow_name,
                inputs={},
                dispatched=False,
                reason="disabled",
            )

        # Evaluate all conditions
        for condition in trigger.conditions:
            if not _evaluate_condition(condition, payload):
                return DispatchResult(
                    trigger_id=trigger.id,
                    workflow_name=trigger.workflow_name,
                    inputs={},
                    dispatched=False,
                    reason="condition_not_met",
                )

        # Apply transform
        if trigger.transform:
            try:
                inputs = _apply_transform(trigger.transform, payload)
            except Exception as e:
                logger.warning("Transform failed for trigger %s: %s", trigger.id, e)
                return DispatchResult(
                    trigger_id=trigger.id,
                    workflow_name=trigger.workflow_name,
                    inputs={},
                    dispatched=False,
                    reason="transform_error",
                )
        else:
            inputs = dict(payload)

        # Execute via callback
        if self._dispatch_callback:
            try:
                self._dispatch_callback(trigger.workflow_name, inputs)
            except Exception as e:
                logger.error("Dispatch callback failed for trigger %s: %s", trigger.id, e)

        return DispatchResult(
            trigger_id=trigger.id,
            workflow_name=trigger.workflow_name,
            inputs=inputs,
            dispatched=True,
            reason="ok",
        )

    def register_chain(
        self,
        name: str,
        steps: list[dict[str, Any]],
    ) -> str:
        """Register a named integration chain.

        Each step defines a trigger rule. The chain connects them conceptually
        for visualization and management.

        Args:
            name: Human-readable chain name.
            steps: List of trigger rule dicts with source, event, workflow_name, etc.

        Returns:
            Chain ID.
        """
        chain_id = f"chain-{uuid.uuid4().hex[:8]}"
        rules: list[TriggerRule] = []

        for i, step_data in enumerate(steps):
            trigger_id = self.register_trigger(
                source=step_data["source"],
                event=step_data["event"],
                workflow_name=step_data["workflow_name"],
                transform=step_data.get("transform"),
                conditions=step_data.get("conditions"),
            )
            rules.append(self._triggers[trigger_id])

        chain = IntegrationChain(id=chain_id, name=name, steps=rules)
        self._chains[chain_id] = chain
        logger.info("Registered chain %s (%s) with %d steps", chain_id, name, len(rules))
        return chain_id

    def get_graph(self) -> dict[str, Any]:
        """Return the full integration graph for visualization.

        Returns:
            Dict with nodes (services + workflows) and edges (triggers).
        """
        services: set[str] = set()
        workflows: set[str] = set()
        edges: list[dict[str, Any]] = []

        for trigger in self._triggers.values():
            services.add(trigger.source)
            workflows.add(trigger.workflow_name)
            edges.append(
                {
                    "id": trigger.id,
                    "from": trigger.source,
                    "to": trigger.workflow_name,
                    "event": trigger.event,
                    "enabled": trigger.enabled,
                    "conditions": len(trigger.conditions),
                    "has_transform": trigger.transform is not None,
                }
            )

        return {
            "nodes": {
                "services": sorted(services),
                "workflows": sorted(workflows),
            },
            "edges": edges,
            "chains": [
                {"id": c.id, "name": c.name, "steps": len(c.steps)} for c in self._chains.values()
            ],
            "total_triggers": len(self._triggers),
        }

    def validate_graph(self, known_workflows: list[str] | None = None) -> list[str]:
        """Validate the integration graph for issues.

        Args:
            known_workflows: List of workflow names that exist.
                If provided, checks for references to missing workflows.

        Returns:
            List of warning messages.
        """
        warnings: list[str] = []

        # Check for disabled triggers
        disabled = [t for t in self._triggers.values() if not t.enabled]
        if disabled:
            warnings.append(f"{len(disabled)} trigger(s) are disabled")

        # Check for missing workflows
        if known_workflows is not None:
            known = set(known_workflows)
            for trigger in self._triggers.values():
                if trigger.workflow_name not in known:
                    warnings.append(
                        f"Trigger {trigger.id} references unknown workflow "
                        f"'{trigger.workflow_name}'"
                    )

        # Check for duplicate source+event->workflow mappings
        seen: dict[tuple[str, str, str], str] = {}
        for trigger in self._triggers.values():
            key = (trigger.source, trigger.event, trigger.workflow_name)
            if key in seen:
                warnings.append(
                    f"Duplicate trigger: {trigger.id} and {seen[key]} both map "
                    f"{trigger.source}.{trigger.event} -> {trigger.workflow_name}"
                )
            else:
                seen[key] = trigger.id

        # Check for sources with no triggers (in chains)
        for chain in self._chains.values():
            if not chain.steps:
                warnings.append(f"Chain '{chain.name}' ({chain.id}) has no steps")

        return warnings

    def list_triggers(
        self, source: str | None = None, event: str | None = None
    ) -> list[TriggerRule]:
        """List triggers with optional filtering.

        Args:
            source: Filter by source.
            event: Filter by event.

        Returns:
            List of matching trigger rules.
        """
        triggers = list(self._triggers.values())
        if source:
            triggers = [t for t in triggers if t.source == source]
        if event:
            triggers = [t for t in triggers if t.event == event]
        return triggers
