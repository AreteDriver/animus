"""Integration tests for Animus Forge agent and engine modules.

Tests ForgeAgent execution, output parsing, archetype prompts, and
ForgeEngine workflow orchestration including gates, budgets, checkpoints,
and failure modes.
"""

from __future__ import annotations

import pytest

from animus.cognitive import CognitiveLayer, ModelConfig
from animus.forge.agent import ARCHETYPE_PROMPTS, ForgeAgent
from animus.forge.engine import ForgeEngine
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
# Helpers
# ---------------------------------------------------------------------------


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


def _make_two_agent_workflow(
    name: str = "test-pipeline",
    budget_tokens: int = 10_000,
    gates: list[GateConfig] | None = None,
) -> WorkflowConfig:
    """Build a minimal two-agent workflow (researcher -> writer)."""
    return WorkflowConfig(
        name=name,
        agents=[
            AgentConfig(
                name="researcher",
                archetype="researcher",
                budget_tokens=budget_tokens,
                outputs=["brief"],
            ),
            AgentConfig(
                name="writer",
                archetype="writer",
                budget_tokens=budget_tokens,
                inputs=["brief"],
                outputs=["draft"],
            ),
        ],
        gates=gates or [],
    )


# ===========================================================================
# TestForgeAgent
# ===========================================================================


class TestForgeAgent:
    """ForgeAgent execution, output parsing, and archetype behaviour."""

    def test_agent_runs_with_mock_cognitive(self):
        """Agent produces a successful StepResult with mock backend."""
        cognitive = _make_cognitive(default_response="Research findings here.")
        config = AgentConfig(
            name="researcher",
            archetype="researcher",
            outputs=["brief"],
        )
        agent = ForgeAgent(config, cognitive)
        result = agent.run({})

        assert isinstance(result, StepResult)
        assert result.success is True
        assert result.agent_name == "researcher"
        assert result.tokens_used > 0
        assert result.error is None

    def test_inputs_passed_through_to_prompt(self):
        """Input dict values appear in the prompt sent to the model."""
        cognitive = _make_cognitive(default_response="Got it.")
        config = AgentConfig(name="summarizer", archetype="writer", outputs=["summary"])
        agent = ForgeAgent(config, cognitive)

        result = agent.run({"topic": "quantum computing", "notes": "key facts"})
        assert result.success is True

        # Verify the mock model received the input content in the prompt
        model = cognitive.primary
        prompt_sent = model.calls[0]["prompt"]
        assert "quantum computing" in prompt_sent
        assert "key facts" in prompt_sent

    def test_output_parsing_with_sections(self):
        """Response containing ## output_name sections gets parsed correctly."""
        response = (
            "## brief\n"
            "This is a research brief about the topic.\n\n"
            "## notes\n"
            "Additional notes here."
        )
        cognitive = _make_cognitive(default_response=response)
        config = AgentConfig(
            name="researcher",
            archetype="researcher",
            outputs=["brief", "notes"],
        )
        agent = ForgeAgent(config, cognitive)
        result = agent.run({})

        assert result.success is True
        assert "brief" in result.outputs
        assert "notes" in result.outputs
        assert "research brief" in result.outputs["brief"]
        assert "Additional notes" in result.outputs["notes"]

    def test_output_fallback_to_first_output(self):
        """Response without ## sections assigns entire text to first output."""
        cognitive = _make_cognitive(default_response="Plain text without sections.")
        config = AgentConfig(
            name="writer",
            archetype="writer",
            outputs=["draft"],
        )
        agent = ForgeAgent(config, cognitive)
        result = agent.run({})

        assert result.success is True
        assert result.outputs["draft"] == "Plain text without sections."

    def test_agent_no_declared_outputs_returns_response_key(self):
        """Agent with no outputs list returns {"response": response}."""
        cognitive = _make_cognitive(default_response="Free-form answer.")
        config = AgentConfig(name="helper", archetype="analyst", outputs=[])
        agent = ForgeAgent(config, cognitive)
        result = agent.run({})

        assert result.success is True
        assert result.outputs == {"response": "Free-form answer."}

    def test_agent_failure_returns_unsuccessful_result(self):
        """When cognitive.think raises, result.success=False and error is set."""
        cognitive = _make_cognitive()
        # Force the primary model to raise
        cognitive.primary.generate = lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("LLM backend unavailable")
        )

        config = AgentConfig(name="broken", archetype="researcher", outputs=["brief"])
        agent = ForgeAgent(config, cognitive)
        result = agent.run({})

        assert result.success is False
        assert result.tokens_used == 0
        assert result.outputs == {}
        assert "LLM backend unavailable" in result.error

    def test_known_archetype_gets_system_prompt(self):
        """Known archetypes receive their predefined system prompts."""
        for archetype_name, expected_prompt in ARCHETYPE_PROMPTS.items():
            cognitive = _make_cognitive()
            config = AgentConfig(
                name=f"test-{archetype_name}",
                archetype=archetype_name,
                outputs=["out"],
            )
            agent = ForgeAgent(config, cognitive)
            system_prompt = agent._build_system_prompt()
            assert expected_prompt in system_prompt, (
                f"Archetype {archetype_name!r} system prompt missing"
            )

    def test_unknown_archetype_gets_fallback_prompt(self):
        """Unknown archetype name produces a fallback system prompt."""
        cognitive = _make_cognitive()
        config = AgentConfig(name="custom", archetype="mythical_creature", outputs=["out"])
        agent = ForgeAgent(config, cognitive)
        system_prompt = agent._build_system_prompt()
        assert "mythical_creature" in system_prompt


# ===========================================================================
# TestForgeEngine
# ===========================================================================


class TestForgeEngine:
    """ForgeEngine workflow orchestration, gates, budgets, and checkpoints."""

    def test_full_workflow_two_agents(self):
        """Two-agent pipeline: writer uses researcher's output."""
        cognitive = _make_cognitive(
            default_response="## brief\nResearch findings.",
            response_map={
                "brief": "## draft\nHere is the written draft.",
            },
        )
        engine = ForgeEngine(cognitive)
        config = _make_two_agent_workflow()

        state = engine.run(config)

        assert state.status == "completed"
        assert len(state.results) == 2
        assert state.results[0].agent_name == "researcher"
        assert state.results[1].agent_name == "writer"
        assert state.total_tokens > 0

    def test_gate_pass(self):
        """Workflow completes when gate condition evaluates to true."""
        cognitive = _make_cognitive(
            default_response="## brief\nDone.",
            response_map={"brief": "## draft\nFinal draft."},
        )
        gate = GateConfig(
            name="always-pass",
            after="researcher",
            pass_condition="true",
            on_fail="halt",
        )
        config = _make_two_agent_workflow(gates=[gate])
        engine = ForgeEngine(cognitive)

        state = engine.run(config)
        assert state.status == "completed"
        assert len(state.results) == 2

    def test_gate_fail_halt(self):
        """Gate with condition='false' and on_fail='halt' raises GateFailedError."""
        cognitive = _make_cognitive(default_response="## brief\nSome research.")
        gate = GateConfig(
            name="quality-check",
            after="researcher",
            pass_condition="false",
            on_fail="halt",
        )
        config = _make_two_agent_workflow(gates=[gate])
        engine = ForgeEngine(cognitive)

        with pytest.raises(GateFailedError):
            engine.run(config)

    def test_gate_fail_skip(self):
        """Gate with condition='false' and on_fail='skip' lets workflow continue."""
        cognitive = _make_cognitive(
            default_response="## brief\nResearch.",
            response_map={"brief": "## draft\nDraft text."},
        )
        gate = GateConfig(
            name="soft-check",
            after="researcher",
            pass_condition="false",
            on_fail="skip",
        )
        config = _make_two_agent_workflow(gates=[gate])
        engine = ForgeEngine(cognitive)

        state = engine.run(config)
        assert state.status == "completed"
        assert len(state.results) == 2

    def test_budget_exhaustion_via_check(self):
        """Agent with budget_tokens=0 triggers BudgetExhaustedError on check()."""
        cognitive = _make_cognitive(default_response="Should never run.")
        config = WorkflowConfig(
            name="budget-test",
            agents=[
                AgentConfig(
                    name="expensive",
                    archetype="researcher",
                    budget_tokens=0,
                    outputs=["out"],
                ),
            ],
        )
        engine = ForgeEngine(cognitive)

        with pytest.raises(BudgetExhaustedError, match="no remaining budget"):
            engine.run(config)

    def test_checkpoint_resume(self, tmp_path):
        """Run first agent, save state, create new engine, resume from step 1."""
        # Phase 1: run a single-agent workflow to completion and checkpoint it
        cognitive = _make_cognitive(
            default_response="## brief\nResearch brief.",
            response_map={"brief": "## draft\nWritten draft."},
        )
        config = _make_two_agent_workflow(name="resumable")

        engine1 = ForgeEngine(cognitive, checkpoint_dir=tmp_path)
        state = engine1.run(config)
        assert state.status == "completed"
        assert len(state.results) == 2

        # Verify checkpoint was persisted
        saved = engine1.status("resumable")
        assert saved is not None
        assert saved.workflow_name == "resumable"
        assert saved.current_step == 2

        # Phase 2: tamper with checkpoint to simulate partial run (only step 0 done)
        saved.current_step = 1
        saved.status = "running"
        saved.results = saved.results[:1]  # keep only researcher result
        saved.total_tokens = saved.results[0].tokens_used
        engine1._checkpoint.save_state(saved)

        # Phase 3: new engine resumes from step 1
        engine2 = ForgeEngine(cognitive, checkpoint_dir=tmp_path)
        resumed = engine2.run(config, resume=True)

        assert resumed.status == "completed"
        assert len(resumed.results) == 2
        assert resumed.results[0].agent_name == "researcher"
        assert resumed.results[1].agent_name == "writer"

    def test_failed_agent_stops_workflow(self):
        """Agent failure (success=False) raises ForgeError and stops execution."""
        cognitive = _make_cognitive()
        # Force the model to raise so ForgeAgent returns success=False
        cognitive.primary.generate = lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("model crashed")
        )

        config = _make_two_agent_workflow()
        engine = ForgeEngine(cognitive)

        with pytest.raises(ForgeError, match="failed"):
            engine.run(config)

    def test_no_checkpoint_dir_works(self):
        """Engine works without a checkpoint store (no persistence)."""
        cognitive = _make_cognitive(
            default_response="## brief\nBrief.",
            response_map={"brief": "## draft\nDraft."},
        )
        engine = ForgeEngine(cognitive, checkpoint_dir=None)
        config = _make_two_agent_workflow()

        state = engine.run(config)
        assert state.status == "completed"

        # status/list_workflows return empty when no checkpoint store
        assert engine.status("test-pipeline") is None
        assert engine.list_workflows() == []

    def test_list_workflows(self, tmp_path):
        """list_workflows returns (name, status, step) tuples for all runs."""
        cognitive = _make_cognitive(default_response="## brief\nDone.")
        engine = ForgeEngine(cognitive, checkpoint_dir=tmp_path)

        config_a = WorkflowConfig(
            name="wf-alpha",
            agents=[
                AgentConfig(name="a1", archetype="researcher", outputs=["brief"]),
            ],
        )
        config_b = WorkflowConfig(
            name="wf-beta",
            agents=[
                AgentConfig(name="b1", archetype="writer", outputs=["brief"]),
            ],
        )

        engine.run(config_a)
        engine.run(config_b)

        workflows = engine.list_workflows()
        names = {w[0] for w in workflows}
        assert "wf-alpha" in names
        assert "wf-beta" in names
        assert all(w[1] == "completed" for w in workflows)

    def test_status_returns_workflow_state(self, tmp_path):
        """status() returns the correct WorkflowState after a run."""
        cognitive = _make_cognitive(default_response="## brief\nDone.")
        engine = ForgeEngine(cognitive, checkpoint_dir=tmp_path)

        config = WorkflowConfig(
            name="status-test",
            agents=[
                AgentConfig(name="s1", archetype="analyst", outputs=["brief"]),
            ],
        )
        engine.run(config)

        state = engine.status("status-test")
        assert isinstance(state, WorkflowState)
        assert state.workflow_name == "status-test"
        assert state.status == "completed"
        assert state.current_step == 1

    def test_pause_workflow(self, tmp_path):
        """pause() marks a running workflow as paused in checkpoint store."""
        cognitive = _make_cognitive(
            default_response="## brief\nResearch.",
            response_map={"brief": "## draft\nDraft."},
        )
        engine = ForgeEngine(cognitive, checkpoint_dir=tmp_path)
        config = _make_two_agent_workflow(name="pausable")

        engine.run(config)

        engine.pause("pausable")

        state = engine.status("pausable")
        assert state.status == "paused"

    def test_pause_without_checkpoint_raises(self):
        """pause() without a checkpoint store raises ForgeError."""
        cognitive = _make_cognitive()
        engine = ForgeEngine(cognitive, checkpoint_dir=None)

        with pytest.raises(ForgeError, match="No checkpoint store"):
            engine.pause("nonexistent")

    def test_pause_unknown_workflow_raises(self, tmp_path):
        """pause() on an unknown workflow name raises ForgeError."""
        cognitive = _make_cognitive()
        engine = ForgeEngine(cognitive, checkpoint_dir=tmp_path)

        with pytest.raises(ForgeError, match="No checkpoint found"):
            engine.pause("does-not-exist")
