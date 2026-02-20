"""Tests for Forge/Swarm revise gate loop-back.

Covers: ReviseRequestedError exception, revision loop-back, max revisions,
gate feedback injection, downstream clearing, checkpoint persistence,
loader ordering validation.
"""

from __future__ import annotations

from textwrap import dedent

import pytest

from animus.cognitive import CognitiveLayer, ModelConfig
from animus.forge.engine import ForgeEngine
from animus.forge.loader import load_workflow_str
from animus.forge.models import (
    AgentConfig,
    ForgeError,
    GateConfig,
    GateFailedError,
    ReviseRequestedError,
    WorkflowConfig,
    WorkflowState,
)
from animus.swarm.engine import SwarmEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Track how many times each agent has been called
_call_counts: dict[str, int] = {}


def _make_cognitive(
    default_response: str = "Mock default response.",
    response_map: dict[str, str] | None = None,
) -> CognitiveLayer:
    """Create a CognitiveLayer backed by a deterministic MockModel."""
    return CognitiveLayer(
        ModelConfig.mock(
            default_response=default_response,
            response_map=response_map or {},
        )
    )


def _make_revise_workflow(
    max_revisions: int = 3,
    pass_condition: str = "false",
    execution_mode: str = "sequential",
) -> WorkflowConfig:
    """Build a workflow with a revise gate: researcher -> writer -> gate(revise->writer)."""
    return WorkflowConfig(
        name="revise-test",
        execution_mode=execution_mode,
        agents=[
            AgentConfig(
                name="researcher",
                archetype="researcher",
                budget_tokens=50_000,
                outputs=["brief"],
            ),
            AgentConfig(
                name="writer",
                archetype="writer",
                budget_tokens=50_000,
                inputs=["researcher.brief"],
                outputs=["draft"],
            ),
        ],
        gates=[
            GateConfig(
                name="quality_check",
                after="writer",
                type="automated",
                pass_condition=pass_condition,
                on_fail="revise",
                revise_target="writer",
                max_revisions=max_revisions,
            ),
        ],
    )


def _make_three_agent_revise_workflow(
    pass_condition: str = "false",
    revise_target: str = "researcher",
) -> WorkflowConfig:
    """Three agents: researcher -> writer -> reviewer, gate after reviewer revises to target."""
    return WorkflowConfig(
        name="three-agent-revise",
        agents=[
            AgentConfig(
                name="researcher",
                archetype="researcher",
                budget_tokens=50_000,
                outputs=["brief"],
            ),
            AgentConfig(
                name="writer",
                archetype="writer",
                budget_tokens=50_000,
                inputs=["researcher.brief"],
                outputs=["draft"],
            ),
            AgentConfig(
                name="reviewer",
                archetype="reviewer",
                budget_tokens=50_000,
                inputs=["writer.draft"],
                outputs=["review"],
            ),
        ],
        gates=[
            GateConfig(
                name="final_check",
                after="reviewer",
                type="automated",
                pass_condition=pass_condition,
                on_fail="revise",
                revise_target=revise_target,
                max_revisions=2,
            ),
        ],
    )


# ===================================================================
# ReviseRequestedError model
# ===================================================================


class TestReviseRequestedError:
    """ReviseRequestedError exception carries the right data."""

    def test_attributes(self) -> None:
        exc = ReviseRequestedError(
            target="writer",
            gate_name="quality",
            reason="Score too low",
            max_revisions=3,
        )
        assert exc.target == "writer"
        assert exc.gate_name == "quality"
        assert exc.reason == "Score too low"
        assert exc.max_revisions == 3
        assert "quality" in str(exc)
        assert "writer" in str(exc)

    def test_is_forge_error(self) -> None:
        exc = ReviseRequestedError("w", "g", "r", 3)
        assert isinstance(exc, ForgeError)


# ===================================================================
# GateConfig defaults
# ===================================================================


class TestGateConfigDefaults:
    """GateConfig.max_revisions defaults."""

    def test_default_max_revisions(self) -> None:
        gc = GateConfig(name="g", after="a")
        assert gc.max_revisions == 3

    def test_custom_max_revisions(self) -> None:
        gc = GateConfig(name="g", after="a", max_revisions=5)
        assert gc.max_revisions == 5


# ===================================================================
# WorkflowState revision_counts
# ===================================================================


class TestWorkflowStateRevisionCounts:
    """WorkflowState.revision_counts defaults and usage."""

    def test_default_empty(self) -> None:
        ws = WorkflowState(workflow_name="test")
        assert ws.revision_counts == {}

    def test_tracks_counts(self) -> None:
        ws = WorkflowState(workflow_name="test")
        ws.revision_counts["gate1"] = 2
        assert ws.revision_counts["gate1"] == 2


# ===================================================================
# Forge Engine — Revise Gate
# ===================================================================


class TestForgeRevise:
    """ForgeEngine revise gate loop-back."""

    def test_revise_succeeds_on_retry(self) -> None:
        """Gate fails first time (condition=false), passes on second try.

        We use a pass_condition that checks for _gate_feedback — present only
        on revision runs, so first run fails, second succeeds.
        """
        # Gate condition: 'draft contains "revised"'
        # First writer run produces default response (no "revised")
        # After revision feedback, response_map kicks in
        config = WorkflowConfig(
            name="retry-test",
            agents=[
                AgentConfig(
                    name="researcher",
                    archetype="researcher",
                    budget_tokens=50_000,
                    outputs=["brief"],
                ),
                AgentConfig(
                    name="writer",
                    archetype="writer",
                    budget_tokens=50_000,
                    inputs=["researcher.brief"],
                    outputs=["draft"],
                ),
            ],
            gates=[
                GateConfig(
                    name="quality",
                    after="writer",
                    type="automated",
                    pass_condition="true",  # Will be overridden by mock
                    on_fail="revise",
                    revise_target="writer",
                    max_revisions=3,
                ),
            ],
        )

        # Track calls to determine when gate passes
        call_count = {"gate_evals": 0}
        original_evaluate = ForgeEngine._evaluate_gates

        def patched_evaluate(self, cfg, agent_name, all_outputs, state):
            for gate in cfg.gates:
                if gate.after != agent_name:
                    continue
                call_count["gate_evals"] += 1
                if call_count["gate_evals"] == 1:
                    # First time: fail with revise
                    raise ReviseRequestedError(
                        target=gate.revise_target,
                        gate_name=gate.name,
                        reason="Quality too low",
                        max_revisions=gate.max_revisions,
                    )
                # Second time: pass

        cognitive = _make_cognitive()
        engine = ForgeEngine(cognitive)

        ForgeEngine._evaluate_gates = patched_evaluate
        try:
            state = engine.run(config)
        finally:
            ForgeEngine._evaluate_gates = original_evaluate

        assert state.status == "completed"
        # researcher kept, writer's first result cleared then re-run
        assert len(state.results) == 2
        assert state.results[0].agent_name == "researcher"
        assert state.results[1].agent_name == "writer"
        assert state.revision_counts == {"quality": 1}

    def test_max_revisions_exceeded(self) -> None:
        """Gate always fails — should raise GateFailedError after max_revisions."""
        config = _make_revise_workflow(max_revisions=2, pass_condition="false")
        cognitive = _make_cognitive()
        engine = ForgeEngine(cognitive)

        with pytest.raises(GateFailedError, match="Max revisions"):
            engine.run(config)

    def test_revise_clears_downstream_results(self) -> None:
        """When revising from researcher, writer and reviewer results are cleared."""
        config = _make_three_agent_revise_workflow(
            pass_condition="false", revise_target="researcher"
        )

        call_count = {"gate_evals": 0}
        original_evaluate = ForgeEngine._evaluate_gates

        def patched_evaluate(self, cfg, agent_name, all_outputs, state):
            for gate in cfg.gates:
                if gate.after != agent_name:
                    continue
                call_count["gate_evals"] += 1
                if call_count["gate_evals"] == 1:
                    raise ReviseRequestedError(
                        target=gate.revise_target,
                        gate_name=gate.name,
                        reason="Needs revision",
                        max_revisions=gate.max_revisions,
                    )
                # Second time: pass

        cognitive = _make_cognitive()
        engine = ForgeEngine(cognitive)

        ForgeEngine._evaluate_gates = patched_evaluate
        try:
            state = engine.run(config)
        finally:
            ForgeEngine._evaluate_gates = original_evaluate

        assert state.status == "completed"
        # First run: researcher + writer + reviewer (3)
        # Revision clears all 3 (target=researcher), then re-runs all 3
        # Total results: 6 (but downstream clearing removes first 3)
        # Actually: after clearing, only results BEFORE target remain (none),
        # then 3 new results are appended
        assert len(state.results) == 3
        assert state.results[0].agent_name == "researcher"
        assert state.results[1].agent_name == "writer"
        assert state.results[2].agent_name == "reviewer"

    def test_revise_injects_gate_feedback(self) -> None:
        """Revised agent receives _gate_feedback in its inputs."""
        config = _make_revise_workflow(max_revisions=2, pass_condition="false")

        received_inputs: list[dict] = []
        call_count = {"gate_evals": 0, "agent_runs": 0}
        original_evaluate = ForgeEngine._evaluate_gates

        def patched_evaluate(self, cfg, agent_name, all_outputs, state):
            for gate in cfg.gates:
                if gate.after != agent_name:
                    continue
                call_count["gate_evals"] += 1
                if call_count["gate_evals"] == 1:
                    raise ReviseRequestedError(
                        target=gate.revise_target,
                        gate_name=gate.name,
                        reason="Draft is too short",
                        max_revisions=gate.max_revisions,
                    )

        from animus.forge.agent import ForgeAgent

        original_run = ForgeAgent.run

        def patched_run(self, inputs):
            if self.config.name == "writer":
                received_inputs.append(dict(inputs))
            return original_run(self, inputs)

        cognitive = _make_cognitive()
        engine = ForgeEngine(cognitive)

        ForgeEngine._evaluate_gates = patched_evaluate
        ForgeAgent.run = patched_run
        try:
            engine.run(config)
        finally:
            ForgeEngine._evaluate_gates = original_evaluate
            ForgeAgent.run = original_run

        # Writer called twice: first without feedback, second with
        assert len(received_inputs) == 2
        assert "_gate_feedback" not in received_inputs[0]
        assert "_gate_feedback" in received_inputs[1]
        assert "Draft is too short" in received_inputs[1]["_gate_feedback"]

    def test_revision_count_increments(self) -> None:
        """revision_counts tracks per-gate revision count."""
        config = _make_revise_workflow(max_revisions=3, pass_condition="false")

        call_count = {"gate_evals": 0}
        original_evaluate = ForgeEngine._evaluate_gates

        def patched_evaluate(self, cfg, agent_name, all_outputs, state):
            for gate in cfg.gates:
                if gate.after != agent_name:
                    continue
                call_count["gate_evals"] += 1
                if call_count["gate_evals"] <= 2:
                    raise ReviseRequestedError(
                        target=gate.revise_target,
                        gate_name=gate.name,
                        reason="Still not good",
                        max_revisions=gate.max_revisions,
                    )

        cognitive = _make_cognitive()
        engine = ForgeEngine(cognitive)

        ForgeEngine._evaluate_gates = patched_evaluate
        try:
            state = engine.run(config)
        finally:
            ForgeEngine._evaluate_gates = original_evaluate

        assert state.status == "completed"
        assert state.revision_counts == {"quality_check": 2}

    def test_checkpoint_preserves_revision_counts(self, tmp_path) -> None:
        """Revision counts survive checkpoint save/load cycle."""
        from animus.forge.checkpoint import CheckpointStore

        store = CheckpointStore(tmp_path / "test.db")
        state = WorkflowState(workflow_name="ckpt-test")
        state.revision_counts = {"gate_a": 2, "gate_b": 1}
        store.save_state(state)

        loaded = store.load_state("ckpt-test")
        assert loaded is not None
        assert loaded.revision_counts == {"gate_a": 2, "gate_b": 1}
        store.close()


# ===================================================================
# Swarm Engine — Revise Gate
# ===================================================================


class TestSwarmRevise:
    """SwarmEngine revise gate loop-back."""

    def test_revise_succeeds_on_retry(self) -> None:
        """Swarm revise loops back and succeeds on second attempt."""
        config = _make_revise_workflow(
            max_revisions=3, pass_condition="false", execution_mode="parallel"
        )

        call_count = {"gate_evals": 0}
        original_evaluate = SwarmEngine._evaluate_gates

        def patched_evaluate(self, cfg, agent_name, all_outputs, state):
            for gate in cfg.gates:
                if gate.after != agent_name:
                    continue
                call_count["gate_evals"] += 1
                if call_count["gate_evals"] == 1:
                    raise ReviseRequestedError(
                        target=gate.revise_target,
                        gate_name=gate.name,
                        reason="Quality too low",
                        max_revisions=gate.max_revisions,
                    )

        cognitive = _make_cognitive()
        engine = SwarmEngine(cognitive)

        SwarmEngine._evaluate_gates = patched_evaluate
        try:
            state = engine.run(config)
        finally:
            SwarmEngine._evaluate_gates = original_evaluate

        assert state.status == "completed"
        assert state.revision_counts == {"quality_check": 1}

    def test_max_revisions_exceeded(self) -> None:
        """Swarm raises GateFailedError after max revisions."""
        config = _make_revise_workflow(
            max_revisions=2, pass_condition="false", execution_mode="parallel"
        )
        cognitive = _make_cognitive()
        engine = SwarmEngine(cognitive)

        with pytest.raises(GateFailedError, match="Max revisions"):
            engine.run(config)

    def test_revise_clears_downstream_stages(self) -> None:
        """Swarm revision clears agents from target stage onward."""
        # researcher (stage 0) -> writer (stage 1) -> reviewer (stage 2)
        # gate after reviewer, revise_target=writer
        config = WorkflowConfig(
            name="swarm-downstream",
            execution_mode="parallel",
            agents=[
                AgentConfig(
                    name="researcher",
                    archetype="researcher",
                    budget_tokens=50_000,
                    outputs=["brief"],
                ),
                AgentConfig(
                    name="writer",
                    archetype="writer",
                    budget_tokens=50_000,
                    inputs=["researcher.brief"],
                    outputs=["draft"],
                ),
                AgentConfig(
                    name="reviewer",
                    archetype="reviewer",
                    budget_tokens=50_000,
                    inputs=["writer.draft"],
                    outputs=["review"],
                ),
            ],
            gates=[
                GateConfig(
                    name="final_gate",
                    after="reviewer",
                    type="automated",
                    pass_condition="false",
                    on_fail="revise",
                    revise_target="writer",
                    max_revisions=2,
                ),
            ],
        )

        call_count = {"gate_evals": 0}
        original_evaluate = SwarmEngine._evaluate_gates

        def patched_evaluate(self, cfg, agent_name, all_outputs, state):
            for gate in cfg.gates:
                if gate.after != agent_name:
                    continue
                call_count["gate_evals"] += 1
                if call_count["gate_evals"] == 1:
                    raise ReviseRequestedError(
                        target=gate.revise_target,
                        gate_name=gate.name,
                        reason="Needs revision",
                        max_revisions=gate.max_revisions,
                    )

        cognitive = _make_cognitive()
        engine = SwarmEngine(cognitive)

        SwarmEngine._evaluate_gates = patched_evaluate
        try:
            state = engine.run(config)
        finally:
            SwarmEngine._evaluate_gates = original_evaluate

        assert state.status == "completed"
        # After revision, researcher result kept, writer+reviewer re-run
        assert len(state.results) == 3
        assert state.results[0].agent_name == "researcher"


# ===================================================================
# Loader Validation
# ===================================================================


class TestLoaderReviseValidation:
    """Loader validates revise_target ordering in sequential mode."""

    def test_revise_target_before_after_valid(self) -> None:
        """revise_target before 'after' agent is valid in sequential mode."""
        yaml_str = dedent("""\
            name: test
            agents:
              - name: researcher
                archetype: researcher
                outputs: [brief]
              - name: writer
                archetype: writer
                inputs: [researcher.brief]
                outputs: [draft]
            gates:
              - name: quality
                after: writer
                on_fail: revise
                revise_target: researcher
        """)
        config = load_workflow_str(yaml_str)
        assert config.gates[0].revise_target == "researcher"

    def test_revise_target_same_as_after_valid(self) -> None:
        """revise_target == after agent is valid (re-run same agent)."""
        yaml_str = dedent("""\
            name: test
            agents:
              - name: researcher
                archetype: researcher
                outputs: [brief]
              - name: writer
                archetype: writer
                inputs: [researcher.brief]
                outputs: [draft]
            gates:
              - name: quality
                after: writer
                on_fail: revise
                revise_target: writer
        """)
        config = load_workflow_str(yaml_str)
        assert config.gates[0].revise_target == "writer"

    def test_revise_target_after_current_rejected_sequential(self) -> None:
        """revise_target after 'after' agent is rejected in sequential mode."""
        yaml_str = dedent("""\
            name: test
            agents:
              - name: researcher
                archetype: researcher
                outputs: [brief]
              - name: writer
                archetype: writer
                inputs: [researcher.brief]
                outputs: [draft]
              - name: reviewer
                archetype: reviewer
                inputs: [writer.draft]
                outputs: [review]
            gates:
              - name: quality
                after: writer
                on_fail: revise
                revise_target: reviewer
        """)
        with pytest.raises(ForgeError, match="must come before"):
            load_workflow_str(yaml_str)

    def test_revise_target_after_current_allowed_parallel(self) -> None:
        """revise_target after 'after' agent is allowed in parallel mode."""
        yaml_str = dedent("""\
            name: test
            execution_mode: parallel
            agents:
              - name: researcher
                archetype: researcher
                outputs: [brief]
              - name: writer
                archetype: writer
                inputs: [researcher.brief]
                outputs: [draft]
              - name: reviewer
                archetype: reviewer
                inputs: [writer.draft]
                outputs: [review]
            gates:
              - name: quality
                after: writer
                on_fail: revise
                revise_target: reviewer
        """)
        config = load_workflow_str(yaml_str)
        assert config.gates[0].revise_target == "reviewer"

    def test_max_revisions_parsed_from_yaml(self) -> None:
        """max_revisions is read from YAML gate config."""
        yaml_str = dedent("""\
            name: test
            agents:
              - name: writer
                archetype: writer
                outputs: [draft]
            gates:
              - name: quality
                after: writer
                on_fail: revise
                revise_target: writer
                max_revisions: 5
        """)
        config = load_workflow_str(yaml_str)
        assert config.gates[0].max_revisions == 5

    def test_max_revisions_defaults_to_three(self) -> None:
        """max_revisions defaults to 3 if not specified."""
        yaml_str = dedent("""\
            name: test
            agents:
              - name: writer
                archetype: writer
                outputs: [draft]
            gates:
              - name: quality
                after: writer
                on_fail: revise
                revise_target: writer
        """)
        config = load_workflow_str(yaml_str)
        assert config.gates[0].max_revisions == 3
