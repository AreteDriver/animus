"""Additional forge tests to cover remaining gaps.

Covers: __init__.py lazy import, loader file-based loading, loader edge cases,
gates edge cases, engine edge cases (revise gate, unexpected error).
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pytest

from animus.forge.gates import evaluate_gate
from animus.forge.loader import load_workflow, load_workflow_str
from animus.forge.models import (
    AgentConfig,
    ForgeError,
    GateConfig,
    GateFailedError,
    WorkflowConfig,
)


class TestForgeInit:
    """Test forge __init__.py lazy import."""

    def test_lazy_import_forge_engine(self):
        import animus.forge as forge

        engine_cls = forge.ForgeEngine
        from animus.forge.engine import ForgeEngine

        assert engine_cls is ForgeEngine

    def test_lazy_import_unknown_attr(self):
        import animus.forge as forge

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = forge.NonExistentThing


class TestLoaderFileBased:
    """Test load_workflow from file path."""

    def test_load_from_file(self, tmp_path: Path):
        yaml_content = dedent("""\
            name: file_test
            agents:
              - name: a1
                archetype: researcher
        """)
        yaml_file = tmp_path / "workflow.yaml"
        yaml_file.write_text(yaml_content)
        wf = load_workflow(yaml_file)
        assert wf.name == "file_test"

    def test_load_missing_file(self, tmp_path: Path):
        with pytest.raises(ForgeError, match="not found"):
            load_workflow(tmp_path / "missing.yaml")

    def test_invalid_yaml_raises(self):
        with pytest.raises(ForgeError, match="Invalid YAML"):
            load_workflow_str("{{invalid: yaml: [")

    def test_non_dict_yaml_raises(self):
        with pytest.raises(ForgeError, match="must be a mapping"):
            load_workflow_str("- just a list")

    def test_agent_not_dict_raises(self):
        bad = dedent("""\
            name: bad
            agents:
              - "just a string"
        """)
        with pytest.raises(ForgeError, match="must be a mapping"):
            load_workflow_str(bad)

    def test_agent_missing_archetype_raises(self):
        bad = dedent("""\
            name: bad
            agents:
              - name: a1
        """)
        with pytest.raises(ForgeError, match="must have an 'archetype'"):
            load_workflow_str(bad)

    def test_gate_not_dict_raises(self):
        bad = dedent("""\
            name: bad
            agents:
              - name: a1
                archetype: researcher
            gates:
              - "just a string"
        """)
        with pytest.raises(ForgeError, match="must be a mapping"):
            load_workflow_str(bad)

    def test_gate_missing_name_raises(self):
        bad = dedent("""\
            name: bad
            agents:
              - name: a1
                archetype: researcher
            gates:
              - after: a1
        """)
        with pytest.raises(ForgeError, match="must have a 'name'"):
            load_workflow_str(bad)

    def test_gate_missing_after_raises(self):
        bad = dedent("""\
            name: bad
            agents:
              - name: a1
                archetype: researcher
            gates:
              - name: g1
        """)
        with pytest.raises(ForgeError, match="must have an 'after'"):
            load_workflow_str(bad)

    def test_gate_revise_target_not_agent_raises(self):
        bad = dedent("""\
            name: bad
            agents:
              - name: a1
                archetype: researcher
            gates:
              - name: g1
                after: a1
                on_fail: revise
                revise_target: ghost
        """)
        with pytest.raises(ForgeError, match="not a defined agent"):
            load_workflow_str(bad)


class TestGatesEdgeCases:
    """Cover remaining gate edge cases."""

    @staticmethod
    def _gate(condition: str) -> GateConfig:
        return GateConfig(name="g", after="a", pass_condition=condition)

    def test_contains_missing_ref(self):
        passed, reason = evaluate_gate(self._gate('missing contains "x"'), {})
        assert passed is False
        assert "not found" in reason

    def test_numeric_non_numeric_value(self):
        passed, reason = evaluate_gate(self._gate("val >= 5"), {"val": "not_a_number"})
        assert passed is False
        assert "not numeric" in reason

    def test_json_field_not_found_returns_none(self):
        outputs = {"review": json.dumps({"notes": "good"})}
        passed, reason = evaluate_gate(self._gate("review.score >= 0.8"), outputs)
        assert passed is False
        assert "not found" in reason

    def test_json_parse_failure_returns_none(self):
        outputs = {"review": "not json at all"}
        passed, reason = evaluate_gate(self._gate("review.score >= 0.8"), outputs)
        assert passed is False
        assert "not found" in reason

    def test_length_missing_base(self):
        passed, reason = evaluate_gate(self._gate("missing.length >= 5"), {})
        assert passed is False
        assert "not found" in reason


class TestEngineEdgeCases:
    """Cover remaining engine edge cases."""

    def test_revise_gate_raises(self, tmp_path: Path):
        from animus.cognitive import CognitiveLayer, ModelConfig
        from animus.forge.engine import ForgeEngine

        cognitive = CognitiveLayer(ModelConfig.mock(default_response="## brief\nDone."))
        config = WorkflowConfig(
            name="revise-test",
            agents=[
                AgentConfig(name="a1", archetype="researcher", outputs=["brief"]),
            ],
            gates=[
                GateConfig(
                    name="g1",
                    after="a1",
                    pass_condition="false",
                    on_fail="revise",
                    revise_target="a1",
                ),
            ],
        )
        engine = ForgeEngine(cognitive, checkpoint_dir=tmp_path)
        with pytest.raises(GateFailedError, match="revise requested"):
            engine.run(config)

    def test_unexpected_error_wraps(self, tmp_path: Path):
        from animus.cognitive import CognitiveLayer, ModelConfig
        from animus.forge.agent import ForgeAgent
        from animus.forge.engine import ForgeEngine

        cognitive = CognitiveLayer(ModelConfig.mock(default_response="## brief\nDone."))
        config = WorkflowConfig(
            name="error-test",
            agents=[
                AgentConfig(name="a1", archetype="researcher", outputs=["brief"]),
            ],
        )
        engine = ForgeEngine(cognitive, checkpoint_dir=tmp_path)

        # Force an unexpected error by patching ForgeAgent.run to raise TypeError
        with patch.object(ForgeAgent, "run", side_effect=TypeError("unexpected")):
            with pytest.raises(ForgeError, match="Unexpected error"):
                engine.run(config)

    def test_input_not_found_warning(self):
        from animus.cognitive import CognitiveLayer, ModelConfig
        from animus.forge.engine import ForgeEngine

        cognitive = CognitiveLayer(ModelConfig.mock(default_response="## brief\nDone."))
        config = WorkflowConfig(
            name="input-warn",
            agents=[
                AgentConfig(
                    name="a1",
                    archetype="researcher",
                    inputs=["nonexistent.output"],
                    outputs=["brief"],
                ),
            ],
        )
        engine = ForgeEngine(cognitive)
        state = engine.run(config)
        assert state.status == "completed"


class TestForgeAgentEdgeCases:
    """Cover agent edge cases."""

    def test_agent_with_tools(self):
        from animus.cognitive import CognitiveLayer, ModelConfig
        from animus.forge.agent import ForgeAgent
        from animus.tools import ToolRegistry

        cognitive = CognitiveLayer(ModelConfig.mock(default_response="Tool result."))
        config = AgentConfig(
            name="tool-agent",
            archetype="researcher",
            tools=["search"],
            outputs=["brief"],
        )
        registry = ToolRegistry()
        agent = ForgeAgent(config, cognitive, tools=registry)
        result = agent.run({})
        assert result.success is True

    def test_custom_system_prompt_with_archetype(self):
        from animus.cognitive import CognitiveLayer, ModelConfig
        from animus.forge.agent import ForgeAgent

        cognitive = CognitiveLayer(ModelConfig.mock())
        config = AgentConfig(
            name="custom",
            archetype="researcher",
            system_prompt="Extra instructions.",
        )
        agent = ForgeAgent(config, cognitive)
        prompt = agent._build_system_prompt()
        assert "Extra instructions." in prompt
        assert "research analyst" in prompt

    def test_custom_system_prompt_no_archetype(self):
        from animus.cognitive import CognitiveLayer, ModelConfig
        from animus.forge.agent import ForgeAgent

        cognitive = CognitiveLayer(ModelConfig.mock())
        config = AgentConfig(
            name="custom",
            archetype="unknown_type",
            system_prompt="Only custom.",
        )
        agent = ForgeAgent(config, cognitive)
        prompt = agent._build_system_prompt()
        assert prompt == "Only custom."
