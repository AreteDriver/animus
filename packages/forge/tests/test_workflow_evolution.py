"""Tests for the YAML workflow evolution fast path."""

from __future__ import annotations

import json

import pytest
import yaml

from animus_forge.coordination.workflow_evolution import (
    WorkflowEvolution,
    WorkflowPatch,
    WorkflowPatchInvalid,
    WorkflowPatchResult,
    _version_gt,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASIC_WORKFLOW = {
    "name": "Test Workflow",
    "version": "1.0",
    "description": "A test workflow",
    "token_budget": 5000,
    "steps": [
        {"id": "step_1", "type": "shell", "params": {"command": "echo hi"}},
    ],
}

UPDATED_WORKFLOW = {
    "name": "Test Workflow",
    "version": "1.1",
    "description": "Updated test workflow",
    "token_budget": 5000,
    "steps": [
        {"id": "step_1", "type": "shell", "params": {"command": "echo hi"}},
        {"id": "step_2", "type": "shell", "params": {"command": "echo done"}},
    ],
}


@pytest.fixture()
def workflows_dir(tmp_path):
    d = tmp_path / "workflows"
    d.mkdir()
    (d / "test-workflow.yaml").write_text(yaml.dump(BASIC_WORKFLOW, default_flow_style=False))
    return d


@pytest.fixture()
def audit_log(tmp_path):
    return tmp_path / "audit.jsonl"


@pytest.fixture()
def evo(workflows_dir, audit_log):
    return WorkflowEvolution(
        workflows_dir=workflows_dir,
        audit_log_path=audit_log,
    )


def _make_patch(workflow_id="test-workflow", content=None, **kwargs):
    if content is None:
        content = yaml.dump(UPDATED_WORKFLOW, default_flow_style=False)
    defaults = {
        "workflow_id": workflow_id,
        "change_description": "Added step_2",
        "new_yaml_content": content,
        "reasoning": "Testing",
        "proposed_by": "test",
    }
    defaults.update(kwargs)
    return WorkflowPatch(**defaults)


# ---------------------------------------------------------------------------
# Version comparison
# ---------------------------------------------------------------------------


class TestVersionGt:
    def test_simple_increment(self):
        assert _version_gt("1.1", "1.0")

    def test_major_increment(self):
        assert _version_gt("2.0", "1.9")

    def test_same_version(self):
        assert not _version_gt("1.0", "1.0")

    def test_lower_version(self):
        assert not _version_gt("0.9", "1.0")

    def test_different_lengths(self):
        assert _version_gt("1.0.1", "1.0")

    def test_single_digit(self):
        assert _version_gt("2", "1")


# ---------------------------------------------------------------------------
# Propose patch
# ---------------------------------------------------------------------------


class TestProposePatch:
    def test_creates_pending_file(self, evo, workflows_dir):
        evo.propose_patch(_make_patch())
        assert (workflows_dir / "test-workflow.pending.yaml").exists()

    def test_returns_result(self, evo):
        result = evo.propose_patch(_make_patch())
        assert isinstance(result, WorkflowPatchResult)
        assert result.workflow_id == "test-workflow"
        assert not result.approved

    def test_emits_audit(self, evo, audit_log):
        evo.propose_patch(_make_patch())
        assert audit_log.exists()
        record = json.loads(audit_log.read_text().strip())
        assert record["event_type"] == "workflow_patch_proposed"

    def test_rejects_unknown_workflow(self, evo):
        with pytest.raises(WorkflowPatchInvalid, match="not found"):
            evo.propose_patch(_make_patch(workflow_id="nonexistent"))

    def test_rejects_same_version(self, evo):
        same_ver = {**UPDATED_WORKFLOW, "version": "1.0"}
        content = yaml.dump(same_ver, default_flow_style=False)
        with pytest.raises(WorkflowPatchInvalid, match="increment"):
            evo.propose_patch(_make_patch(new_yaml_content=content))

    def test_rejects_lower_version(self, evo):
        lower = {**UPDATED_WORKFLOW, "version": "0.5"}
        content = yaml.dump(lower, default_flow_style=False)
        with pytest.raises(WorkflowPatchInvalid, match="increment"):
            evo.propose_patch(_make_patch(new_yaml_content=content))

    def test_rejects_missing_name(self, evo):
        bad = {k: v for k, v in UPDATED_WORKFLOW.items() if k != "name"}
        content = yaml.dump(bad, default_flow_style=False)
        with pytest.raises(WorkflowPatchInvalid, match="name"):
            evo.propose_patch(_make_patch(new_yaml_content=content))

    def test_rejects_missing_steps(self, evo):
        bad = {k: v for k, v in UPDATED_WORKFLOW.items() if k != "steps"}
        content = yaml.dump(bad, default_flow_style=False)
        with pytest.raises(WorkflowPatchInvalid, match="steps"):
            evo.propose_patch(_make_patch(new_yaml_content=content))

    def test_rejects_exec_injection(self, evo):
        content = yaml.dump(UPDATED_WORKFLOW, default_flow_style=False)
        content += "\n# exec(malicious)"
        with pytest.raises(WorkflowPatchInvalid, match="exec"):
            evo.propose_patch(_make_patch(new_yaml_content=content))

    def test_rejects_eval_injection(self, evo):
        content = yaml.dump(UPDATED_WORKFLOW, default_flow_style=False)
        content += "\n# eval(something)"
        with pytest.raises(WorkflowPatchInvalid, match="eval"):
            evo.propose_patch(_make_patch(new_yaml_content=content))

    def test_rejects_import_injection(self, evo):
        content = yaml.dump(UPDATED_WORKFLOW, default_flow_style=False)
        content += "\n# import os"
        with pytest.raises(WorkflowPatchInvalid, match="import"):
            evo.propose_patch(_make_patch(new_yaml_content=content))

    def test_rejects_non_snake_case_step_type(self, evo):
        bad = {
            **UPDATED_WORKFLOW,
            "steps": [{"id": "s1", "type": "BadType"}],
        }
        content = yaml.dump(bad, default_flow_style=False)
        with pytest.raises(WorkflowPatchInvalid, match="snake_case"):
            evo.propose_patch(_make_patch(new_yaml_content=content))

    def test_rejects_llm_steps_without_budget(self, evo):
        no_budget = {
            "name": "Test",
            "version": "1.1",
            "steps": [{"id": "s1", "type": "claude_code"}],
        }
        content = yaml.dump(no_budget, default_flow_style=False)
        with pytest.raises(WorkflowPatchInvalid, match="token_budget"):
            evo.propose_patch(_make_patch(new_yaml_content=content))

    def test_rejects_invalid_yaml(self, evo):
        with pytest.raises(WorkflowPatchInvalid, match="Invalid YAML"):
            evo.propose_patch(_make_patch(new_yaml_content=": :\n  - [bad"))


# ---------------------------------------------------------------------------
# Approve
# ---------------------------------------------------------------------------


class TestApprove:
    def test_applies_pending(self, evo, workflows_dir):
        evo.propose_patch(_make_patch())
        result = evo.approve("test-workflow")
        assert result.approved
        assert result.committed
        assert not (workflows_dir / "test-workflow.pending.yaml").exists()

    def test_updates_workflow_file(self, evo, workflows_dir):
        evo.propose_patch(_make_patch())
        evo.approve("test-workflow")
        data = yaml.safe_load((workflows_dir / "test-workflow.yaml").read_text())
        assert str(data["version"]) == "1.1"

    def test_appends_evolution_notes(self, evo, workflows_dir):
        evo.propose_patch(_make_patch())
        evo.approve("test-workflow")
        data = yaml.safe_load((workflows_dir / "test-workflow.yaml").read_text())
        notes = data.get("evolution_notes", [])
        assert len(notes) == 1
        assert notes[0]["version"] == "1.1"

    def test_emits_audit(self, evo, audit_log):
        evo.propose_patch(_make_patch())
        evo.approve("test-workflow")
        lines = audit_log.read_text().strip().splitlines()
        events = [json.loads(line)["event_type"] for line in lines]
        assert "workflow_patch_approved" in events

    def test_raises_without_pending(self, evo):
        with pytest.raises(WorkflowPatchInvalid, match="No pending"):
            evo.approve("test-workflow")


# ---------------------------------------------------------------------------
# Reject
# ---------------------------------------------------------------------------


class TestReject:
    def test_removes_pending(self, evo, workflows_dir):
        evo.propose_patch(_make_patch())
        assert (workflows_dir / "test-workflow.pending.yaml").exists()
        evo.reject("test-workflow", "not needed")
        assert not (workflows_dir / "test-workflow.pending.yaml").exists()

    def test_returns_result(self, evo):
        evo.propose_patch(_make_patch())
        result = evo.reject("test-workflow", "bad idea")
        assert not result.approved
        assert result.rejection_reason == "bad idea"

    def test_emits_audit(self, evo, audit_log):
        evo.propose_patch(_make_patch())
        evo.reject("test-workflow", "reason")
        lines = audit_log.read_text().strip().splitlines()
        events = [json.loads(line)["event_type"] for line in lines]
        assert "workflow_patch_rejected" in events

    def test_reject_without_pending_is_safe(self, evo):
        result = evo.reject("test-workflow", "nothing to reject")
        assert not result.approved


# ---------------------------------------------------------------------------
# List and history
# ---------------------------------------------------------------------------


class TestListAndHistory:
    def test_list_pending_empty(self, evo):
        assert evo.list_pending() == []

    def test_list_pending_after_propose(self, evo):
        evo.propose_patch(_make_patch())
        assert "test-workflow" in evo.list_pending()

    def test_list_workflows(self, evo):
        workflows = evo.list_workflows()
        assert len(workflows) == 1
        assert workflows[0]["id"] == "test-workflow"
        assert workflows[0]["version"] == "1.0"

    def test_list_workflows_shows_pending(self, evo):
        evo.propose_patch(_make_patch())
        workflows = evo.list_workflows()
        assert workflows[0]["has_pending"] is True

    def test_history_empty(self, evo):
        assert evo.history("test-workflow") == []

    def test_history_after_approve(self, evo):
        evo.propose_patch(_make_patch())
        evo.approve("test-workflow")
        notes = evo.history("test-workflow")
        assert len(notes) == 1

    def test_history_nonexistent(self, evo):
        assert evo.history("doesnt-exist") == []
