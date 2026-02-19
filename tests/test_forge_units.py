"""Unit tests for the Animus Forge package.

Covers: models, loader, budget, gates, checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from animus.forge.budget import BudgetTracker
from animus.forge.checkpoint import CheckpointStore
from animus.forge.gates import evaluate_gate
from animus.forge.loader import load_workflow_str
from animus.forge.models import (
    AgentConfig,
    BudgetExhaustedError,
    ForgeError,
    GateConfig,
    GateFailedError,
    StepResult,
    WorkflowConfig,
    WorkflowState,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

VALID_YAML = dedent("""\
    name: test_pipeline
    description: Test workflow
    agents:
      - name: researcher
        archetype: researcher
        budget_tokens: 5000
        outputs: [brief]
      - name: writer
        archetype: writer
        budget_tokens: 8000
        inputs: [researcher.brief]
        outputs: [draft]
    gates:
      - name: quality
        after: writer
        type: automated
        pass_condition: "true"
""")


def _make_workflow(
    agents: list[AgentConfig] | None = None,
    max_cost: float = 1.0,
) -> WorkflowConfig:
    """Build a minimal WorkflowConfig for budget/tracker tests."""
    if agents is None:
        agents = [
            AgentConfig(name="a1", archetype="researcher", budget_tokens=1000),
            AgentConfig(name="a2", archetype="writer", budget_tokens=2000),
        ]
    return WorkflowConfig(name="test", agents=agents, max_cost_usd=max_cost)


# ===================================================================
# 1. Models
# ===================================================================


class TestModels:
    """Dataclass construction, defaults, and exception hierarchy."""

    def test_agent_config_defaults(self) -> None:
        ac = AgentConfig(name="r", archetype="researcher")
        assert ac.budget_tokens == 10_000
        assert ac.inputs == []
        assert ac.outputs == []
        assert ac.provider is None
        assert ac.model is None
        assert ac.system_prompt is None
        assert ac.tools == []

    def test_agent_config_custom_fields(self) -> None:
        ac = AgentConfig(
            name="w",
            archetype="writer",
            budget_tokens=500,
            inputs=["r.brief"],
            outputs=["draft"],
            provider="openai",
            model="gpt-4o",
            system_prompt="Write well.",
            tools=["search"],
        )
        assert ac.name == "w"
        assert ac.budget_tokens == 500
        assert ac.provider == "openai"
        assert ac.tools == ["search"]

    def test_gate_config_defaults(self) -> None:
        gc = GateConfig(name="g", after="w")
        assert gc.type == "automated"
        assert gc.pass_condition == ""
        assert gc.on_fail == "halt"
        assert gc.revise_target is None

    def test_workflow_config_defaults(self) -> None:
        wc = WorkflowConfig(name="pipeline")
        assert wc.description == ""
        assert wc.agents == []
        assert wc.gates == []
        assert wc.max_cost_usd == 1.0
        assert wc.provider == "ollama"
        assert wc.model == "llama3:8b"

    def test_step_result_defaults(self) -> None:
        sr = StepResult(agent_name="r")
        assert sr.outputs == {}
        assert sr.tokens_used == 0
        assert sr.cost_usd == 0.0
        assert sr.success is True
        assert sr.error is None

    def test_workflow_state_defaults(self) -> None:
        ws = WorkflowState(workflow_name="p")
        assert ws.status == "pending"
        assert ws.current_step == 0
        assert ws.results == []
        assert ws.total_tokens == 0
        assert ws.total_cost == 0.0

    def test_forge_error_hierarchy(self) -> None:
        assert issubclass(BudgetExhaustedError, ForgeError)
        assert issubclass(GateFailedError, ForgeError)

    def test_forge_error_message(self) -> None:
        err = ForgeError("something broke")
        assert str(err) == "something broke"


# ===================================================================
# 2. Loader
# ===================================================================


class TestLoader:
    """YAML parsing and validation via load_workflow_str."""

    def test_valid_workflow(self) -> None:
        wf = load_workflow_str(VALID_YAML)
        assert wf.name == "test_pipeline"
        assert wf.description == "Test workflow"
        assert len(wf.agents) == 2
        assert wf.agents[0].name == "researcher"
        assert wf.agents[0].budget_tokens == 5000
        assert wf.agents[1].inputs == ["researcher.brief"]
        assert len(wf.gates) == 1
        assert wf.gates[0].name == "quality"
        assert wf.gates[0].after == "writer"

    def test_missing_name_raises(self) -> None:
        bad = dedent("""\
            description: no name
            agents:
              - name: a
                archetype: x
        """)
        with pytest.raises(ForgeError, match="must have a 'name'"):
            load_workflow_str(bad)

    def test_empty_agents_raises(self) -> None:
        bad = dedent("""\
            name: empty
            agents: []
        """)
        with pytest.raises(ForgeError, match="at least one agent"):
            load_workflow_str(bad)

    def test_duplicate_agent_names_raises(self) -> None:
        bad = dedent("""\
            name: dup
            agents:
              - name: same
                archetype: x
              - name: same
                archetype: y
        """)
        with pytest.raises(ForgeError, match="Duplicate agent name"):
            load_workflow_str(bad)

    def test_bad_input_ref_format_raises(self) -> None:
        bad = dedent("""\
            name: bad_ref
            agents:
              - name: a
                archetype: x
                inputs: [no_dot_format]
        """)
        with pytest.raises(ForgeError, match="'agent.output' format"):
            load_workflow_str(bad)

    def test_bad_input_ref_undefined_agent_raises(self) -> None:
        bad = dedent("""\
            name: bad_ref2
            agents:
              - name: a
                archetype: x
                inputs: [ghost.brief]
        """)
        with pytest.raises(ForgeError, match="undefined agent 'ghost'"):
            load_workflow_str(bad)

    def test_gate_ref_undefined_agent_raises(self) -> None:
        bad = dedent("""\
            name: bad_gate
            agents:
              - name: a
                archetype: x
            gates:
              - name: g
                after: nonexistent
        """)
        with pytest.raises(ForgeError, match="undefined agent 'nonexistent'"):
            load_workflow_str(bad)

    def test_revise_without_target_raises(self) -> None:
        bad = dedent("""\
            name: bad_revise
            agents:
              - name: a
                archetype: x
            gates:
              - name: g
                after: a
                on_fail: revise
        """)
        with pytest.raises(ForgeError, match="no 'revise_target'"):
            load_workflow_str(bad)


# ===================================================================
# 3. Budget
# ===================================================================


class TestBudget:
    """BudgetTracker creation, recording, checking, and summaries."""

    def test_from_config(self) -> None:
        wf = _make_workflow()
        bt = BudgetTracker.from_config(wf)
        assert bt.agent_budgets == {"a1": 1000, "a2": 2000}
        assert bt.agent_usage == {"a1": 0, "a2": 0}
        assert bt.max_cost_usd == 1.0

    def test_record_and_remaining(self) -> None:
        bt = BudgetTracker.from_config(_make_workflow())
        bt.record("a1", 400, 0.01)
        assert bt.remaining("a1") == 600
        assert bt.check("a1") is True

    def test_check_false_when_at_budget(self) -> None:
        bt = BudgetTracker.from_config(_make_workflow())
        bt.record("a1", 1000, 0.0)
        assert bt.check("a1") is False
        assert bt.remaining("a1") == 0

    def test_record_over_budget_raises(self) -> None:
        bt = BudgetTracker.from_config(_make_workflow())
        bt.record("a1", 800, 0.01)
        with pytest.raises(BudgetExhaustedError, match="exceed budget"):
            bt.record("a1", 300, 0.01)

    def test_cost_ceiling_raises(self) -> None:
        wf = _make_workflow(max_cost=0.05)
        bt = BudgetTracker.from_config(wf)
        bt.record("a1", 100, 0.03)
        with pytest.raises(BudgetExhaustedError, match="exceed cost ceiling"):
            bt.record("a2", 100, 0.03)

    def test_summary(self) -> None:
        bt = BudgetTracker.from_config(_make_workflow())
        bt.record("a1", 250, 0.02)
        s = bt.summary()
        assert s["a1"]["budget"] == 1000
        assert s["a1"]["used"] == 250
        assert s["a1"]["remaining"] == 750
        assert s["a1"]["pct"] == 25.0
        assert s["a2"]["used"] == 0
        assert s["_total"]["cost_usd"] == 0.02
        assert s["_total"]["max_cost_usd"] == 1.0

    def test_remaining_unknown_agent(self) -> None:
        bt = BudgetTracker.from_config(_make_workflow())
        assert bt.remaining("ghost") == 0

    def test_multiple_records_accumulate(self) -> None:
        bt = BudgetTracker.from_config(_make_workflow())
        bt.record("a1", 200, 0.01)
        bt.record("a1", 300, 0.01)
        assert bt.remaining("a1") == 500
        assert bt.total_cost == pytest.approx(0.02)


# ===================================================================
# 4. Gates
# ===================================================================


class TestGates:
    """evaluate_gate with various condition types."""

    @staticmethod
    def _gate(condition: str, gate_type: str = "automated") -> GateConfig:
        return GateConfig(
            name="test_gate",
            after="agent",
            type=gate_type,
            pass_condition=condition,
        )

    def test_true_condition(self) -> None:
        passed, reason = evaluate_gate(self._gate("true"), {})
        assert passed is True
        assert reason == ""

    def test_false_condition(self) -> None:
        passed, reason = evaluate_gate(self._gate("false"), {})
        assert passed is False
        assert "always fails" in reason

    def test_empty_condition_passes(self) -> None:
        passed, _ = evaluate_gate(self._gate(""), {})
        assert passed is True

    def test_human_gate_always_passes(self) -> None:
        passed, reason = evaluate_gate(self._gate("false", gate_type="human"), {})
        assert passed is True
        assert reason == ""

    def test_numeric_gte(self) -> None:
        passed, _ = evaluate_gate(self._gate("score >= 0.8"), {"score": "0.9"})
        assert passed is True

    def test_numeric_gte_fails(self) -> None:
        passed, reason = evaluate_gate(self._gate("score >= 0.8"), {"score": "0.5"})
        assert passed is False
        assert "false" in reason

    def test_numeric_lte(self) -> None:
        passed, _ = evaluate_gate(self._gate("errors <= 3"), {"errors": "2"})
        assert passed is True

    def test_numeric_gt(self) -> None:
        passed, _ = evaluate_gate(self._gate("count > 0"), {"count": "1"})
        assert passed is True

    def test_numeric_lt(self) -> None:
        passed, _ = evaluate_gate(self._gate("risk < 0.5"), {"risk": "0.3"})
        assert passed is True

    def test_numeric_eq(self) -> None:
        passed, _ = evaluate_gate(self._gate("status == 1"), {"status": "1"})
        assert passed is True

    def test_numeric_neq(self) -> None:
        passed, _ = evaluate_gate(self._gate("status != 0"), {"status": "1"})
        assert passed is True

    def test_contains_check(self) -> None:
        passed, _ = evaluate_gate(
            self._gate('output contains "approved"'),
            {"output": "Result: approved by reviewer"},
        )
        assert passed is True

    def test_contains_check_fails(self) -> None:
        passed, reason = evaluate_gate(
            self._gate('output contains "approved"'),
            {"output": "rejected"},
        )
        assert passed is False
        assert "does not contain" in reason

    def test_length_check(self) -> None:
        passed, _ = evaluate_gate(
            self._gate("draft.length >= 10"),
            {"draft": "A sufficiently long string here"},
        )
        assert passed is True

    def test_json_field_access_via_dot(self) -> None:
        outputs = {"review": json.dumps({"score": 0.95, "notes": "good"})}
        passed, _ = evaluate_gate(self._gate("review.score >= 0.8"), outputs)
        assert passed is True

    def test_flat_key_dot_notation(self) -> None:
        outputs = {"reviewer.score": "0.9"}
        passed, _ = evaluate_gate(self._gate("reviewer.score >= 0.8"), outputs)
        assert passed is True

    def test_invalid_condition_syntax_raises(self) -> None:
        with pytest.raises(ForgeError, match="unsupported condition syntax"):
            evaluate_gate(self._gate("not a valid expression"), {})

    def test_missing_output_reference(self) -> None:
        passed, reason = evaluate_gate(self._gate("missing >= 1"), {})
        assert passed is False
        assert "not found" in reason


# ===================================================================
# 5. Checkpoint
# ===================================================================


class TestCheckpoint:
    """CheckpointStore save/load/delete/list with SQLite persistence."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        store = CheckpointStore(tmp_path / "cp.db")
        state = WorkflowState(
            workflow_name="wf1",
            status="running",
            current_step=2,
            total_tokens=500,
            total_cost=0.05,
            results=[
                StepResult(
                    agent_name="a1",
                    outputs={"brief": "hello"},
                    tokens_used=300,
                    cost_usd=0.03,
                ),
                StepResult(
                    agent_name="a2",
                    outputs={"draft": "world"},
                    tokens_used=200,
                    cost_usd=0.02,
                ),
            ],
        )
        store.save_state(state)
        loaded = store.load_state("wf1")

        assert loaded is not None
        assert loaded.workflow_name == "wf1"
        assert loaded.status == "running"
        assert loaded.current_step == 2
        assert loaded.total_tokens == 500
        assert loaded.total_cost == pytest.approx(0.05)
        assert len(loaded.results) == 2
        store.close()

    def test_load_nonexistent_returns_none(self, tmp_path: Path) -> None:
        store = CheckpointStore(tmp_path / "cp.db")
        assert store.load_state("nope") is None
        store.close()

    def test_delete_state(self, tmp_path: Path) -> None:
        store = CheckpointStore(tmp_path / "cp.db")
        state = WorkflowState(workflow_name="wf1", status="done")
        store.save_state(state)
        store.delete_state("wf1")
        assert store.load_state("wf1") is None
        store.close()

    def test_list_workflows(self, tmp_path: Path) -> None:
        store = CheckpointStore(tmp_path / "cp.db")
        store.save_state(WorkflowState(workflow_name="alpha", status="done", current_step=3))
        store.save_state(WorkflowState(workflow_name="beta", status="running", current_step=1))
        listing = store.list_workflows()
        assert len(listing) == 2
        names = [r[0] for r in listing]
        assert "alpha" in names
        assert "beta" in names
        store.close()

    def test_step_results_preserved(self, tmp_path: Path) -> None:
        store = CheckpointStore(tmp_path / "cp.db")
        sr = StepResult(
            agent_name="reviewer",
            outputs={"verdict": "pass", "notes": "looks good"},
            tokens_used=150,
            cost_usd=0.005,
            success=True,
            error=None,
        )
        state = WorkflowState(
            workflow_name="wf_detail",
            status="running",
            current_step=1,
            results=[sr],
        )
        store.save_state(state)
        loaded = store.load_state("wf_detail")
        assert loaded is not None
        r = loaded.results[0]
        assert r.agent_name == "reviewer"
        assert r.outputs == {"verdict": "pass", "notes": "looks good"}
        assert r.tokens_used == 150
        assert r.cost_usd == pytest.approx(0.005)
        assert r.success is True
        assert r.error is None
        store.close()

    def test_save_overwrites_previous(self, tmp_path: Path) -> None:
        store = CheckpointStore(tmp_path / "cp.db")
        store.save_state(WorkflowState(workflow_name="wf1", status="running", current_step=1))
        store.save_state(WorkflowState(workflow_name="wf1", status="done", current_step=3))
        loaded = store.load_state("wf1")
        assert loaded is not None
        assert loaded.status == "done"
        assert loaded.current_step == 3
        store.close()

    def test_auto_creates_parent_dirs(self, tmp_path: Path) -> None:
        deep_path = tmp_path / "a" / "b" / "c" / "cp.db"
        store = CheckpointStore(deep_path)
        assert deep_path.exists()
        store.close()

    def test_failed_step_result_roundtrip(self, tmp_path: Path) -> None:
        store = CheckpointStore(tmp_path / "cp.db")
        sr = StepResult(
            agent_name="failing",
            outputs={},
            tokens_used=50,
            cost_usd=0.001,
            success=False,
            error="LLM timeout",
        )
        state = WorkflowState(
            workflow_name="wf_fail",
            status="failed",
            current_step=0,
            results=[sr],
        )
        store.save_state(state)
        loaded = store.load_state("wf_fail")
        assert loaded is not None
        r = loaded.results[0]
        assert r.success is False
        assert r.error == "LLM timeout"
        store.close()
