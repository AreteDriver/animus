"""Tests for Animus Swarm parallel orchestration.

Covers: DAG analysis, stage derivation, intent graph, intent resolver,
SwarmEngine parallel execution, checkpoints, budget, gates, and loader changes.
"""

from __future__ import annotations

import threading
from pathlib import Path
from textwrap import dedent

import pytest

from animus.cognitive import CognitiveLayer, ModelConfig
from animus.forge.loader import load_workflow_str
from animus.forge.models import (
    AgentConfig,
    BudgetExhaustedError,
    ForgeError,
    GateConfig,
    GateFailedError,
    WorkflowConfig,
)
from animus.swarm.graph import build_dag, derive_stages, validate_dag
from animus.swarm.intent import IntentGraph, IntentResolver
from animus.swarm.models import CyclicDependencyError, IntentEntry, SwarmConfig

# ── Helpers ──────────────────────────────────────────────────────────


def _make_cognitive(
    default_response: str = "## response\nMock default response.",
    response_map: dict[str, str] | None = None,
) -> CognitiveLayer:
    """Create a CognitiveLayer backed by a deterministic MockModel."""
    return CognitiveLayer(
        ModelConfig.mock(
            default_response=default_response,
            response_map=response_map or {},
        )
    )


def _agent(
    name: str,
    archetype: str = "researcher",
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    budget_tokens: int = 10_000,
) -> AgentConfig:
    return AgentConfig(
        name=name,
        archetype=archetype,
        inputs=inputs or [],
        outputs=outputs or [],
        budget_tokens=budget_tokens,
    )


def _parallel_config(
    name: str = "test-parallel",
    agents: list[AgentConfig] | None = None,
    gates: list[GateConfig] | None = None,
    max_cost_usd: float = 10.0,
) -> WorkflowConfig:
    """Build a parallel WorkflowConfig with diamond shape by default."""
    if agents is None:
        agents = [
            _agent("researcher", outputs=["brief"]),
            _agent("fact_checker", archetype="analyst", outputs=["facts"]),
            _agent(
                "writer",
                archetype="writer",
                inputs=["researcher.brief", "fact_checker.facts"],
                outputs=["draft"],
            ),
        ]
    return WorkflowConfig(
        name=name,
        agents=agents,
        gates=gates or [],
        execution_mode="parallel",
        max_cost_usd=max_cost_usd,
    )


# ════════════════════════════════════════════════════════════════════
# TestBuildDag
# ════════════════════════════════════════════════════════════════════


class TestBuildDag:
    """DAG construction from agent input references."""

    def test_no_inputs_produces_empty_deps(self):
        agents = [_agent("a", outputs=["x"])]
        dag = build_dag(agents)
        assert dag == {"a": set()}

    def test_single_dependency(self):
        agents = [
            _agent("a", outputs=["x"]),
            _agent("b", inputs=["a.x"], outputs=["y"]),
        ]
        dag = build_dag(agents)
        assert dag["b"] == {"a"}
        assert dag["a"] == set()

    def test_multiple_dependencies(self):
        agents = [
            _agent("a", outputs=["x"]),
            _agent("b", outputs=["y"]),
            _agent("c", inputs=["a.x", "b.y"], outputs=["z"]),
        ]
        dag = build_dag(agents)
        assert dag["c"] == {"a", "b"}

    def test_all_agents_independent(self):
        agents = [_agent("a"), _agent("b"), _agent("c")]
        dag = build_dag(agents)
        assert all(deps == set() for deps in dag.values())

    def test_chain_dependency(self):
        agents = [
            _agent("a", outputs=["x"]),
            _agent("b", inputs=["a.x"], outputs=["y"]),
            _agent("c", inputs=["b.y"], outputs=["z"]),
        ]
        dag = build_dag(agents)
        assert dag == {"a": set(), "b": {"a"}, "c": {"b"}}

    def test_self_reference_ignored(self):
        """Self-references in inputs are excluded from dependencies."""
        agents = [_agent("a", inputs=["a.x"], outputs=["x"])]
        dag = build_dag(agents)
        assert dag["a"] == set()


# ════════════════════════════════════════════════════════════════════
# TestDeriveStages
# ════════════════════════════════════════════════════════════════════


class TestDeriveStages:
    """Stage derivation via topological sort."""

    def test_all_independent_single_stage(self):
        agents = [_agent("a"), _agent("b"), _agent("c")]
        dag = build_dag(agents)
        stages = derive_stages(agents, dag)
        assert len(stages) == 1
        assert sorted(stages[0].agent_names) == ["a", "b", "c"]

    def test_linear_chain_one_per_stage(self):
        agents = [
            _agent("a", outputs=["x"]),
            _agent("b", inputs=["a.x"], outputs=["y"]),
            _agent("c", inputs=["b.y"], outputs=["z"]),
        ]
        dag = build_dag(agents)
        stages = derive_stages(agents, dag)
        assert len(stages) == 3
        assert stages[0].agent_names == ["a"]
        assert stages[1].agent_names == ["b"]
        assert stages[2].agent_names == ["c"]

    def test_diamond_dependency(self):
        """A -> {B, C} -> D: 3 stages."""
        agents = [
            _agent("a", outputs=["x"]),
            _agent("b", inputs=["a.x"], outputs=["y"]),
            _agent("c", inputs=["a.x"], outputs=["z"]),
            _agent("d", inputs=["b.y", "c.z"], outputs=["w"]),
        ]
        dag = build_dag(agents)
        stages = derive_stages(agents, dag)
        assert len(stages) == 3
        assert stages[0].agent_names == ["a"]
        assert sorted(stages[1].agent_names) == ["b", "c"]
        assert stages[2].agent_names == ["d"]

    def test_mixed_parallel_and_sequential(self):
        """A, B (parallel) -> C -> D, E (parallel)."""
        agents = [
            _agent("a", outputs=["x"]),
            _agent("b", outputs=["y"]),
            _agent("c", inputs=["a.x", "b.y"], outputs=["z"]),
            _agent("d", inputs=["c.z"], outputs=["w1"]),
            _agent("e", inputs=["c.z"], outputs=["w2"]),
        ]
        dag = build_dag(agents)
        stages = derive_stages(agents, dag)
        assert len(stages) == 3
        assert sorted(stages[0].agent_names) == ["a", "b"]
        assert stages[1].agent_names == ["c"]
        assert sorted(stages[2].agent_names) == ["d", "e"]

    def test_cyclic_dependency_raises(self):
        dag = {"a": {"b"}, "b": {"a"}}
        agents = [_agent("a"), _agent("b")]
        with pytest.raises(CyclicDependencyError, match="Cycle detected"):
            derive_stages(agents, dag)

    def test_self_loop_raises(self):
        dag = {"a": {"a"}}
        agents = [_agent("a")]
        with pytest.raises(CyclicDependencyError, match="Cycle detected"):
            derive_stages(agents, dag)

    def test_stage_ordering_deterministic(self):
        """Agents within a stage are sorted by name."""
        agents = [_agent("z"), _agent("m"), _agent("a")]
        dag = build_dag(agents)
        stages = derive_stages(agents, dag)
        assert stages[0].agent_names == ["a", "m", "z"]

    def test_stage_index_sequential(self):
        agents = [
            _agent("a", outputs=["x"]),
            _agent("b", inputs=["a.x"]),
        ]
        dag = build_dag(agents)
        stages = derive_stages(agents, dag)
        assert stages[0].index == 0
        assert stages[1].index == 1


# ════════════════════════════════════════════════════════════════════
# TestValidateDag
# ════════════════════════════════════════════════════════════════════


class TestValidateDag:
    def test_clean_dag_no_warnings(self):
        agents = [_agent("a", outputs=["x"]), _agent("b", outputs=["y"])]
        dag = build_dag(agents)
        assert validate_dag(agents, dag) == []

    def test_self_loop_warning(self):
        agents = [_agent("a")]
        dag = {"a": {"a"}}
        warnings = validate_dag(agents, dag)
        assert any("self-loop" in w for w in warnings)

    def test_orphan_ref_warning(self):
        agents = [_agent("a")]
        dag = {"a": {"ghost"}}
        warnings = validate_dag(agents, dag)
        assert any("undefined agent" in w for w in warnings)


# ════════════════════════════════════════════════════════════════════
# TestIntentGraph
# ════════════════════════════════════════════════════════════════════


class TestIntentGraph:
    """In-memory intent graph operations."""

    def test_publish_and_read(self):
        graph = IntentGraph()
        entry = IntentEntry(agent="a", provides=["a.x"])
        graph.publish(entry)
        entries = graph.read_all()
        assert len(entries) == 1
        assert entries[0].agent == "a"

    def test_publish_updates_existing(self):
        graph = IntentGraph()
        graph.publish(IntentEntry(agent="a", stability=0.5))
        graph.publish(IntentEntry(agent="a", stability=0.9))
        entries = graph.read_all()
        assert len(entries) == 1
        assert entries[0].stability == 0.9

    def test_get_returns_specific_entry(self):
        graph = IntentGraph()
        graph.publish(IntentEntry(agent="a"))
        graph.publish(IntentEntry(agent="b"))
        assert graph.get("a") is not None
        assert graph.get("a").agent == "a"

    def test_get_missing_returns_none(self):
        graph = IntentGraph()
        assert graph.get("nonexistent") is None

    def test_remove_deletes_entry(self):
        graph = IntentGraph()
        graph.publish(IntentEntry(agent="a"))
        graph.remove("a")
        assert graph.get("a") is None
        assert len(graph.read_all()) == 0

    def test_remove_missing_no_error(self):
        graph = IntentGraph()
        graph.remove("nonexistent")  # Should not raise

    def test_find_conflicts_overlapping_provides(self):
        graph = IntentGraph()
        graph.publish(IntentEntry(agent="a", provides=["shared.output"]))
        entry_b = IntentEntry(agent="b", provides=["shared.output"])
        conflicts = graph.find_conflicts(entry_b)
        assert len(conflicts) == 1
        assert conflicts[0].agent == "a"

    def test_find_conflicts_no_overlap(self):
        graph = IntentGraph()
        graph.publish(IntentEntry(agent="a", provides=["a.x"]))
        entry_b = IntentEntry(agent="b", provides=["b.y"])
        assert graph.find_conflicts(entry_b) == []

    def test_find_conflicts_excludes_self(self):
        graph = IntentGraph()
        graph.publish(IntentEntry(agent="a", provides=["a.x"]))
        entry_a = IntentEntry(agent="a", provides=["a.x"])
        assert graph.find_conflicts(entry_a) == []

    def test_find_conflicts_empty_provides(self):
        graph = IntentGraph()
        graph.publish(IntentEntry(agent="a", provides=["a.x"]))
        entry_b = IntentEntry(agent="b", provides=[])
        assert graph.find_conflicts(entry_b) == []

    def test_clear_removes_all(self):
        graph = IntentGraph()
        graph.publish(IntentEntry(agent="a"))
        graph.publish(IntentEntry(agent="b"))
        graph.clear()
        assert len(graph.read_all()) == 0

    def test_history_tracks_all_publishes(self):
        graph = IntentGraph()
        graph.publish(IntentEntry(agent="a", stability=0.3))
        graph.publish(IntentEntry(agent="a", stability=0.9))
        graph.publish(IntentEntry(agent="b"))
        history = graph.history
        assert len(history) == 3
        assert history[0].stability == 0.3
        assert history[1].stability == 0.9

    def test_thread_safety(self):
        """Concurrent publishes from multiple threads don't corrupt state."""
        graph = IntentGraph()
        errors: list[Exception] = []

        def publish_many(prefix: str, count: int):
            try:
                for i in range(count):
                    graph.publish(IntentEntry(agent=f"{prefix}_{i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=publish_many, args=(f"t{t}", 50)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # 4 threads * 50 publishes, but entries are keyed by agent name
        entries = graph.read_all()
        assert len(entries) == 200
        assert len(graph.history) == 200


# ════════════════════════════════════════════════════════════════════
# TestIntentResolver
# ════════════════════════════════════════════════════════════════════


class TestIntentResolver:
    """Stability computation and conflict resolution."""

    def test_stability_no_inputs(self):
        ac = _agent("a")
        assert IntentResolver.compute_stability(ac, set()) == 1.0

    def test_stability_all_satisfied(self):
        ac = _agent("a", inputs=["b.x", "c.y"])
        available = {"b.x", "c.y", "other"}
        assert IntentResolver.compute_stability(ac, available) == 1.0

    def test_stability_partial(self):
        ac = _agent("a", inputs=["b.x", "c.y", "d.z", "e.w"])
        available = {"b.x", "c.y"}
        assert IntentResolver.compute_stability(ac, available) == 0.5

    def test_stability_none_satisfied(self):
        ac = _agent("a", inputs=["b.x", "c.y"])
        assert IntentResolver.compute_stability(ac, set()) == 0.0

    def test_resolve_no_conflicts(self):
        entry = IntentEntry(agent="a", stability=0.8)
        assert IntentResolver.resolve(entry, []) == "proceed"

    def test_resolve_higher_stability_wins(self):
        entry = IntentEntry(agent="a", stability=0.9)
        conflicts = [IntentEntry(agent="b", stability=0.5)]
        assert IntentResolver.resolve(entry, conflicts) == "proceed"

    def test_resolve_lower_stability_defers(self):
        entry = IntentEntry(agent="a", stability=0.3)
        conflicts = [IntentEntry(agent="b", stability=0.8)]
        assert IntentResolver.resolve(entry, conflicts) == "defer"

    def test_resolve_equal_stability_both_proceed(self):
        entry = IntentEntry(agent="a", stability=0.5)
        conflicts = [IntentEntry(agent="b", stability=0.5)]
        assert IntentResolver.resolve(entry, conflicts) == "proceed_both"


# ════════════════════════════════════════════════════════════════════
# TestSwarmEngine
# ════════════════════════════════════════════════════════════════════


class TestSwarmEngine:
    """SwarmEngine parallel workflow execution."""

    def test_two_independent_agents_parallel(self):
        from animus.swarm.engine import SwarmEngine

        cognitive = _make_cognitive()
        config = _parallel_config(
            agents=[
                _agent("a", outputs=["x"]),
                _agent("b", outputs=["y"]),
            ]
        )
        engine = SwarmEngine(cognitive)
        state = engine.run(config)

        assert state.status == "completed"
        assert len(state.results) == 2
        assert {r.agent_name for r in state.results} == {"a", "b"}

    def test_sequential_chain_produces_results(self):
        from animus.swarm.engine import SwarmEngine

        cognitive = _make_cognitive()
        config = _parallel_config(
            agents=[
                _agent("a", outputs=["x"]),
                _agent("b", inputs=["a.x"], outputs=["y"]),
                _agent("c", inputs=["b.y"], outputs=["z"]),
            ]
        )
        engine = SwarmEngine(cognitive)
        state = engine.run(config)

        assert state.status == "completed"
        assert len(state.results) == 3
        # Verify ordering: a first, then b, then c
        assert state.results[0].agent_name == "a"
        assert state.results[1].agent_name == "b"
        assert state.results[2].agent_name == "c"

    def test_diamond_workflow(self):
        from animus.swarm.engine import SwarmEngine

        cognitive = _make_cognitive()
        config = _parallel_config()  # Default diamond: researcher, fact_checker -> writer
        engine = SwarmEngine(cognitive)
        state = engine.run(config)

        assert state.status == "completed"
        assert len(state.results) == 3
        # Stage 0 agents should be first two results
        stage0_agents = {state.results[0].agent_name, state.results[1].agent_name}
        assert stage0_agents == {"fact_checker", "researcher"}
        assert state.results[2].agent_name == "writer"

    def test_single_agent_no_thread_pool(self):
        """Single-agent stage runs without ThreadPoolExecutor."""
        from animus.swarm.engine import SwarmEngine

        cognitive = _make_cognitive()
        config = _parallel_config(agents=[_agent("solo", outputs=["x"])])
        engine = SwarmEngine(cognitive)
        state = engine.run(config)

        assert state.status == "completed"
        assert len(state.results) == 1

    def test_budget_precheck_fails_fast(self):
        from animus.swarm.engine import SwarmEngine

        cognitive = _make_cognitive()
        config = _parallel_config(agents=[_agent("a", outputs=["x"], budget_tokens=0)])
        engine = SwarmEngine(cognitive)
        with pytest.raises(BudgetExhaustedError):
            engine.run(config)

    def test_budget_tracks_parallel_agents(self):
        from animus.swarm.engine import SwarmEngine

        cognitive = _make_cognitive()
        config = _parallel_config(
            agents=[
                _agent("a", outputs=["x"]),
                _agent("b", outputs=["y"]),
            ]
        )
        engine = SwarmEngine(cognitive)
        state = engine.run(config)

        assert state.total_tokens > 0
        assert all(r.tokens_used > 0 for r in state.results)

    def test_workflow_cost_ceiling(self):
        from animus.swarm.engine import SwarmEngine

        cognitive = _make_cognitive()
        # max_cost_usd = 0.0 means budget.check() fails (0.0 < 0.0 is False)
        config = _parallel_config(
            agents=[_agent("a", outputs=["x"])],
            max_cost_usd=0.0,
        )
        engine = SwarmEngine(cognitive)
        with pytest.raises(BudgetExhaustedError, match="no remaining budget"):
            engine.run(config)

    def test_gate_after_parallel_stage(self):
        from animus.swarm.engine import SwarmEngine

        cognitive = _make_cognitive()
        config = _parallel_config(
            agents=[
                _agent("a", outputs=["x"]),
                _agent("b", outputs=["y"]),
            ],
            gates=[
                GateConfig(name="check", after="b", pass_condition="true"),
            ],
        )
        engine = SwarmEngine(cognitive)
        state = engine.run(config)
        assert state.status == "completed"

    def test_gate_halt_stops_workflow(self):
        from animus.swarm.engine import SwarmEngine

        cognitive = _make_cognitive()
        config = _parallel_config(
            agents=[
                _agent("a", outputs=["x"]),
                _agent("b", inputs=["a.x"], outputs=["y"]),
            ],
            gates=[
                GateConfig(name="fail_gate", after="a", pass_condition="false", on_fail="halt"),
            ],
        )
        engine = SwarmEngine(cognitive)
        with pytest.raises(GateFailedError):
            engine.run(config)

    def test_gate_skip_continues(self):
        from animus.swarm.engine import SwarmEngine

        cognitive = _make_cognitive()
        config = _parallel_config(
            agents=[
                _agent("a", outputs=["x"]),
                _agent("b", inputs=["a.x"], outputs=["y"]),
            ],
            gates=[
                GateConfig(name="skip_gate", after="a", pass_condition="false", on_fail="skip"),
            ],
        )
        engine = SwarmEngine(cognitive)
        state = engine.run(config)
        assert state.status == "completed"
        assert len(state.results) == 2

    def test_checkpoint_saves_after_stage(self, tmp_path: Path):
        from animus.swarm.engine import SwarmEngine

        cognitive = _make_cognitive()
        config = _parallel_config()
        engine = SwarmEngine(cognitive, checkpoint_dir=tmp_path)
        state = engine.run(config)

        assert state.status == "completed"
        # Verify checkpoint store has state
        loaded = engine.status(config.name)
        assert loaded is not None
        assert loaded.status == "completed"
        assert len(loaded.results) == 3

    def test_resume_skips_completed_stages(self, tmp_path: Path):
        from animus.swarm.engine import SwarmEngine

        cognitive = _make_cognitive()
        config = _parallel_config()
        engine = SwarmEngine(cognitive, checkpoint_dir=tmp_path)

        # First run completes
        state1 = engine.run(config)
        assert state1.status == "completed"

        # Resume should detect all stages complete
        state2 = engine.run(config, resume=True)
        assert state2.status == "completed"

    def test_no_checkpoint_dir_works(self):
        from animus.swarm.engine import SwarmEngine

        cognitive = _make_cognitive()
        config = _parallel_config(agents=[_agent("a", outputs=["x"])])
        engine = SwarmEngine(cognitive)  # No checkpoint_dir
        state = engine.run(config)
        assert state.status == "completed"
        assert engine.status("test-parallel") is None  # No checkpoint store

    def test_agent_failure_fails_workflow(self):
        from unittest.mock import patch

        from animus.swarm.engine import SwarmEngine

        cognitive = _make_cognitive()
        config = _parallel_config(agents=[_agent("a", outputs=["x"])])
        engine = SwarmEngine(cognitive)

        # Make the agent fail
        with patch.object(
            cognitive.primary,
            "generate",
            side_effect=RuntimeError("LLM down"),
        ):
            with pytest.raises(ForgeError, match="failed"):
                engine.run(config)

    def test_cyclic_dependency_raises(self):
        from animus.swarm.engine import SwarmEngine

        cognitive = _make_cognitive()
        config = _parallel_config(
            agents=[
                _agent("a", inputs=["b.y"], outputs=["x"]),
                _agent("b", inputs=["a.x"], outputs=["y"]),
            ]
        )
        engine = SwarmEngine(cognitive)
        with pytest.raises(CyclicDependencyError):
            engine.run(config)

    def test_pause_workflow(self, tmp_path: Path):
        from animus.swarm.engine import SwarmEngine

        cognitive = _make_cognitive()
        config = _parallel_config(agents=[_agent("a", outputs=["x"])])
        engine = SwarmEngine(cognitive, checkpoint_dir=tmp_path)
        engine.run(config)
        engine.pause(config.name)

        loaded = engine.status(config.name)
        assert loaded is not None
        assert loaded.status == "paused"

    def test_pause_no_checkpoint_raises(self):
        from animus.swarm.engine import SwarmEngine

        engine = SwarmEngine(_make_cognitive())
        with pytest.raises(ForgeError, match="No checkpoint store"):
            engine.pause("nonexistent")

    def test_pause_missing_workflow_raises(self, tmp_path: Path):
        from animus.swarm.engine import SwarmEngine

        engine = SwarmEngine(_make_cognitive(), checkpoint_dir=tmp_path)
        with pytest.raises(ForgeError, match="No checkpoint found"):
            engine.pause("ghost")

    def test_list_workflows(self, tmp_path: Path):
        from animus.swarm.engine import SwarmEngine

        cognitive = _make_cognitive()
        config = _parallel_config(agents=[_agent("a", outputs=["x"])])
        engine = SwarmEngine(cognitive, checkpoint_dir=tmp_path)
        engine.run(config)

        workflows = engine.list_workflows()
        assert len(workflows) >= 1
        names = [w[0] for w in workflows]
        assert config.name in names

    def test_list_workflows_no_checkpoint(self):
        from animus.swarm.engine import SwarmEngine

        engine = SwarmEngine(_make_cognitive())
        assert engine.list_workflows() == []

    def test_swarm_config_max_workers(self):
        from animus.swarm.engine import SwarmEngine

        cognitive = _make_cognitive()
        sc = SwarmConfig(max_workers=2)
        config = _parallel_config(agents=[_agent("a", outputs=["x"]), _agent("b", outputs=["y"])])
        engine = SwarmEngine(cognitive, swarm_config=sc)
        state = engine.run(config)
        assert state.status == "completed"

    def test_intent_graph_populated(self):
        """After a run, the intent graph history shows all agents."""
        from animus.swarm.engine import SwarmEngine

        cognitive = _make_cognitive()
        config = _parallel_config()
        engine = SwarmEngine(cognitive)
        engine.run(config)

        history = engine._intent_graph.history
        agent_names = {e.agent for e in history}
        assert "researcher" in agent_names
        assert "fact_checker" in agent_names
        assert "writer" in agent_names


# ════════════════════════════════════════════════════════════════════
# TestLoaderParallel
# ════════════════════════════════════════════════════════════════════


class TestLoaderParallel:
    """YAML loader changes for parallel mode."""

    def test_execution_mode_parallel_parsed(self):
        yaml_str = dedent("""\
            name: par
            execution_mode: parallel
            agents:
              - name: a
                archetype: researcher
                outputs: [x]
              - name: b
                archetype: writer
                outputs: [y]
        """)
        config = load_workflow_str(yaml_str)
        assert config.execution_mode == "parallel"

    def test_execution_mode_default_sequential(self):
        yaml_str = dedent("""\
            name: seq
            agents:
              - name: a
                archetype: researcher
        """)
        config = load_workflow_str(yaml_str)
        assert config.execution_mode == "sequential"

    def test_execution_mode_invalid_raises(self):
        yaml_str = dedent("""\
            name: bad
            execution_mode: turbo
            agents:
              - name: a
                archetype: researcher
        """)
        with pytest.raises(ForgeError, match="Invalid execution_mode"):
            load_workflow_str(yaml_str)

    def test_parallel_allows_forward_refs(self):
        """In parallel mode, agents can reference agents defined later."""
        yaml_str = dedent("""\
            name: forward
            execution_mode: parallel
            agents:
              - name: writer
                archetype: writer
                inputs: [researcher.brief]
                outputs: [draft]
              - name: researcher
                archetype: researcher
                outputs: [brief]
        """)
        config = load_workflow_str(yaml_str)
        assert len(config.agents) == 2

    def test_sequential_rejects_forward_refs(self):
        """In sequential mode, forward references are rejected."""
        yaml_str = dedent("""\
            name: bad_forward
            execution_mode: sequential
            agents:
              - name: writer
                archetype: writer
                inputs: [researcher.brief]
              - name: researcher
                archetype: researcher
        """)
        with pytest.raises(ForgeError, match="not yet defined"):
            load_workflow_str(yaml_str)

    def test_undefined_agent_ref_raises_parallel(self):
        """Reference to nonexistent agent raises even in parallel mode."""
        yaml_str = dedent("""\
            name: ghost_ref
            execution_mode: parallel
            agents:
              - name: a
                archetype: researcher
                inputs: [ghost.brief]
        """)
        with pytest.raises(ForgeError, match="undefined agent"):
            load_workflow_str(yaml_str)


# ════════════════════════════════════════════════════════════════════
# TestSwarmInit
# ════════════════════════════════════════════════════════════════════


class TestSwarmInit:
    """Test swarm __init__.py lazy imports."""

    def test_lazy_import_swarm_engine(self):
        import animus.swarm as swarm

        engine_cls = swarm.SwarmEngine
        assert engine_cls.__name__ == "SwarmEngine"

    def test_lazy_import_intent_graph(self):
        import animus.swarm as swarm

        assert swarm.IntentGraph.__name__ == "IntentGraph"

    def test_lazy_import_intent_resolver(self):
        import animus.swarm as swarm

        assert swarm.IntentResolver.__name__ == "IntentResolver"

    def test_lazy_import_build_dag(self):
        import animus.swarm as swarm

        assert callable(swarm.build_dag)

    def test_lazy_import_derive_stages(self):
        import animus.swarm as swarm

        assert callable(swarm.derive_stages)

    def test_lazy_import_unknown_raises(self):
        import animus.swarm as swarm

        with pytest.raises(AttributeError, match="no attribute"):
            _ = swarm.NonExistentThing
