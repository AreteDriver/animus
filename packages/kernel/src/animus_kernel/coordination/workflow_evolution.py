"""YAML workflow evolution fast path.

Forge can propose patches to its own YAML workflow definitions without
the full sandbox/PR pipeline. Patches are staged as .pending.yaml files
and require human approval before applying.

This is the safe fast path — YAML-only, no Python modifications.
For Python code changes, use self_improve/orchestrator.py.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Step types that invoke LLM calls
_LLM_STEP_TYPES = frozenset({"claude_code", "openai", "ollama"})

# Forbidden patterns in YAML content
_INJECTION_PATTERNS = ("```python", "exec(", "eval(", "__import__", "import ")


class WorkflowPatchInvalid(Exception):
    """Raised when a proposed workflow patch fails validation."""


class WorkflowPatch(BaseModel):
    """A proposed change to a YAML workflow."""

    workflow_id: str
    patch_type: str = "modify"  # add_step | remove_step | modify_step | update_metadata
    change_description: str
    new_yaml_content: str
    reasoning: str
    proposed_by: str = "human"
    performance_evidence: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WorkflowPatchResult(BaseModel):
    """Result of a patch proposal, approval, or rejection."""

    workflow_id: str
    approved: bool = False
    committed: bool = False
    rejection_reason: str | None = None


@dataclass
class EvolutionNote:
    """Entry in a workflow's evolution changelog."""

    version: str
    date: str
    change: str
    reasoning: str
    proposed_by: str


class WorkflowEvolution:
    """Manages the YAML workflow evolution fast path.

    Scope: forge/workflows/*.yaml ONLY.
    No .py modifications. No core/ or quorum/ changes.
    """

    def __init__(
        self,
        workflows_dir: Path,
        audit_log_path: Path | None = None,
    ):
        self._workflows_dir = workflows_dir
        self._audit_log_path = audit_log_path

    def propose_patch(self, patch: WorkflowPatch) -> WorkflowPatchResult:
        """Validate and stage a workflow patch.

        Writes to {workflow_id}.pending.yaml. Does NOT apply the change.
        """
        workflow_path = self._workflows_dir / f"{patch.workflow_id}.yaml"
        if not workflow_path.exists():
            raise WorkflowPatchInvalid(f"Workflow not found: {patch.workflow_id}.yaml")

        current = yaml.safe_load(workflow_path.read_text())
        self._validate_patch(current, patch.new_yaml_content)

        pending_path = self._workflows_dir / f"{patch.workflow_id}.pending.yaml"
        pending_path.write_text(patch.new_yaml_content)

        self._emit_audit(
            "workflow_patch_proposed",
            {
                "workflow_id": patch.workflow_id,
                "patch_type": patch.patch_type,
                "change": patch.change_description,
                "proposed_by": patch.proposed_by,
            },
        )

        return WorkflowPatchResult(workflow_id=patch.workflow_id)

    def approve(self, workflow_id: str) -> WorkflowPatchResult:
        """Apply a pending patch, append evolution notes, clean up staging."""
        pending_path = self._workflows_dir / f"{workflow_id}.pending.yaml"
        if not pending_path.exists():
            raise WorkflowPatchInvalid(f"No pending patch for: {workflow_id}")

        workflow_path = self._workflows_dir / f"{workflow_id}.yaml"
        new_content = pending_path.read_text()
        new_data = yaml.safe_load(new_content)

        # Load current for version comparison
        if workflow_path.exists():
            current = yaml.safe_load(workflow_path.read_text())
            old_version = current.get("version", "0")
        else:
            old_version = "0"

        # Append evolution note
        notes = new_data.get("evolution_notes", [])
        notes.append(
            {
                "version": str(new_data.get("version", "?")),
                "date": datetime.now(UTC).strftime("%Y-%m-%d"),
                "change": f"Evolved from v{old_version}",
                "reasoning": "Approved via CLI",
                "proposed_by": "human",
            }
        )
        new_data["evolution_notes"] = notes

        workflow_path.write_text(
            yaml.dump(new_data, default_flow_style=False, sort_keys=False),
        )
        pending_path.unlink()

        self._emit_audit(
            "workflow_patch_approved",
            {"workflow_id": workflow_id, "new_version": str(new_data.get("version"))},
        )

        return WorkflowPatchResult(
            workflow_id=workflow_id,
            approved=True,
            committed=True,
        )

    def reject(self, workflow_id: str, reason: str = "") -> WorkflowPatchResult:
        """Delete a pending patch and log the rejection."""
        pending_path = self._workflows_dir / f"{workflow_id}.pending.yaml"
        if pending_path.exists():
            pending_path.unlink()

        self._emit_audit(
            "workflow_patch_rejected",
            {"workflow_id": workflow_id, "reason": reason},
        )

        return WorkflowPatchResult(
            workflow_id=workflow_id,
            approved=False,
            rejection_reason=reason,
        )

    def list_pending(self) -> list[str]:
        """Return workflow IDs that have .pending.yaml files."""
        return [p.stem.replace(".pending", "") for p in self._workflows_dir.glob("*.pending.yaml")]

    def list_workflows(self) -> list[dict[str, Any]]:
        """List all workflows with version and last-evolved date."""
        results = []
        for path in sorted(self._workflows_dir.glob("*.yaml")):
            if ".pending." in path.name:
                continue
            try:
                data = yaml.safe_load(path.read_text())
                notes = data.get("evolution_notes", [])
                last_evolved = notes[-1]["date"] if notes else None
                results.append(
                    {
                        "id": path.stem,
                        "name": data.get("name", path.stem),
                        "version": str(data.get("version", "?")),
                        "last_evolved": last_evolved,
                        "steps": len(data.get("steps", [])),
                        "has_pending": (self._workflows_dir / f"{path.stem}.pending.yaml").exists(),
                    }
                )
            except Exception:
                results.append({"id": path.stem, "name": path.stem, "version": "?", "error": True})
        return results

    def history(self, workflow_id: str) -> list[dict[str, Any]]:
        """Return evolution_notes for a workflow."""
        path = self._workflows_dir / f"{workflow_id}.yaml"
        if not path.exists():
            return []
        data = yaml.safe_load(path.read_text())
        return data.get("evolution_notes", [])

    def _validate_patch(self, current: dict, proposed_yaml: str) -> None:
        """Validate a proposed YAML patch. Raises WorkflowPatchInvalid."""
        # Parse YAML
        try:
            parsed = yaml.safe_load(proposed_yaml)
        except yaml.YAMLError as e:
            raise WorkflowPatchInvalid(f"Invalid YAML: {e}") from e

        if not isinstance(parsed, dict):
            raise WorkflowPatchInvalid("YAML must be a mapping")

        # Required fields
        for field_name in ("name", "version", "steps"):
            if field_name not in parsed:
                raise WorkflowPatchInvalid(f"Missing required field: {field_name}")

        # Version must increment
        new_ver = str(parsed["version"])
        old_ver = str(current.get("version", "0"))
        if not _version_gt(new_ver, old_ver):
            raise WorkflowPatchInvalid(f"Version must increment: {old_ver} -> {new_ver}")

        # No code injection
        for pattern in _INJECTION_PATTERNS:
            if pattern in proposed_yaml:
                raise WorkflowPatchInvalid(f"Forbidden pattern in YAML: {pattern!r}")

        # Step type validation
        steps = parsed.get("steps", [])
        if not isinstance(steps, list):
            raise WorkflowPatchInvalid("Steps must be a list")

        for step in steps:
            step_type = step.get("type", step.get("id", ""))
            if step_type and not re.match(r"^[a-z_]+$", step_type):
                raise WorkflowPatchInvalid(f"Step type must be snake_case: {step_type!r}")

        # Budget estimate required if LLM steps present
        has_llm = any(s.get("type") in _LLM_STEP_TYPES for s in steps)
        if has_llm and "token_budget" not in parsed:
            raise WorkflowPatchInvalid("Workflows with LLM steps require token_budget")

    def _emit_audit(self, event_type: str, data: dict[str, Any]) -> None:
        """Append to audit log if configured."""
        if self._audit_log_path is None:
            return
        self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"event_type": event_type, "timestamp": datetime.now(UTC).isoformat(), **data}
        with open(self._audit_log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")


def _version_gt(new: str, old: str) -> bool:
    """Simple version comparison. Handles '1.0', '1.0.0', '2' etc."""

    def _parts(v: str) -> list[int]:
        try:
            return [int(x) for x in str(v).split(".")]
        except ValueError:
            return [0]

    new_parts = _parts(new)
    old_parts = _parts(old)
    # Pad to same length
    max_len = max(len(new_parts), len(old_parts))
    new_parts.extend([0] * (max_len - len(new_parts)))
    old_parts.extend([0] * (max_len - len(old_parts)))
    return new_parts > old_parts
