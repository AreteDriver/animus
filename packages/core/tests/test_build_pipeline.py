"""
Tests for Phase 3: Autonomous coding workflow (build pipeline).

Tests new archetypes, build_task.yaml loading, and _run_build_task wiring.
"""

from pathlib import Path

import pytest

from animus.cognitive import CognitiveLayer, ModelConfig
from animus.forge.agent import ARCHETYPE_PROMPTS, ForgeAgent
from animus.forge.loader import load_workflow
from animus.forge.models import AgentConfig

# ---------------------------------------------------------------------------
# New archetypes
# ---------------------------------------------------------------------------


class TestNewArchetypes:
    """Verify planner, coder, and verifier archetypes exist and work."""

    @pytest.mark.parametrize("archetype", ["planner", "coder", "verifier"])
    def test_archetype_registered(self, archetype):
        assert archetype in ARCHETYPE_PROMPTS
        assert len(ARCHETYPE_PROMPTS[archetype]) > 20

    def test_planner_prompt_mentions_read(self):
        assert "read" in ARCHETYPE_PROMPTS["planner"].lower()

    def test_coder_prompt_mentions_edit(self):
        assert "edit" in ARCHETYPE_PROMPTS["coder"].lower()

    def test_verifier_prompt_mentions_pass_fail(self):
        prompt = ARCHETYPE_PROMPTS["verifier"].lower()
        assert "pass" in prompt and "fail" in prompt

    @pytest.mark.parametrize("archetype", ["planner", "coder", "verifier"])
    def test_archetype_builds_system_prompt(self, archetype):
        config = AgentConfig(name=f"test_{archetype}", archetype=archetype)
        cognitive = CognitiveLayer(primary_config=ModelConfig.mock())
        agent = ForgeAgent(config, cognitive)
        prompt = agent._build_system_prompt()
        assert len(prompt) > 0

    def test_archetype_with_custom_system_prompt(self):
        config = AgentConfig(
            name="custom_planner",
            archetype="planner",
            system_prompt="Also consider security implications.",
        )
        cognitive = CognitiveLayer(primary_config=ModelConfig.mock())
        agent = ForgeAgent(config, cognitive)
        prompt = agent._build_system_prompt()
        assert "security" in prompt.lower()
        assert "read" in prompt.lower()  # Still has base archetype


# ---------------------------------------------------------------------------
# build_task.yaml loading
# ---------------------------------------------------------------------------


class TestBuildTaskWorkflow:
    """Tests for the build_task.yaml workflow definition."""

    @pytest.fixture
    def workflow_path(self):
        return Path(__file__).parent.parent / "configs" / "examples" / "build_task.yaml"

    def test_workflow_exists(self, workflow_path):
        assert workflow_path.exists()

    def test_workflow_loads(self, workflow_path):
        config = load_workflow(workflow_path)
        assert config.name == "build_task"

    def test_workflow_has_four_agents(self, workflow_path):
        config = load_workflow(workflow_path)
        assert len(config.agents) == 4

    def test_workflow_agent_order(self, workflow_path):
        config = load_workflow(workflow_path)
        names = [a.name for a in config.agents]
        assert names == ["planner", "coder", "verifier", "fixer"]

    def test_planner_has_read_tools(self, workflow_path):
        config = load_workflow(workflow_path)
        planner = config.agents[0]
        assert "read_file" in planner.tools
        assert "list_files" in planner.tools

    def test_coder_has_file_tools(self, workflow_path):
        config = load_workflow(workflow_path)
        coder = config.agents[1]
        assert "read_file" in coder.tools
        assert "write_file" in coder.tools
        assert "edit_file" in coder.tools

    def test_verifier_has_run_command(self, workflow_path):
        config = load_workflow(workflow_path)
        verifier = config.agents[2]
        assert "run_command" in verifier.tools

    def test_fixer_has_edit_and_run(self, workflow_path):
        config = load_workflow(workflow_path)
        fixer = config.agents[3]
        assert "edit_file" in fixer.tools
        assert "run_command" in fixer.tools

    def test_coder_depends_on_planner(self, workflow_path):
        config = load_workflow(workflow_path)
        coder = config.agents[1]
        assert "planner.plan" in coder.inputs

    def test_fixer_depends_on_verifier(self, workflow_path):
        config = load_workflow(workflow_path)
        fixer = config.agents[3]
        assert "verifier.verification_result" in fixer.inputs

    def test_workflow_has_gate(self, workflow_path):
        config = load_workflow(workflow_path)
        assert len(config.gates) == 1
        gate = config.gates[0]
        assert gate.after == "fixer"
        assert gate.on_fail == "halt"

    def test_workflow_budget(self, workflow_path):
        config = load_workflow(workflow_path)
        assert config.max_cost_usd == 2.0

    def test_planner_system_prompt_injectable(self, workflow_path):
        """Verify task description can be injected into planner's system prompt."""
        config = load_workflow(workflow_path)
        planner = config.agents[0]
        original = planner.system_prompt or ""
        planner.system_prompt = f"{original}\n\n## Task\nadd a health check endpoint"
        assert "health check" in planner.system_prompt


# ---------------------------------------------------------------------------
# ForgeAgent with tools
# ---------------------------------------------------------------------------


class TestForgeAgentWithTools:
    """Verify agents can be configured with tool names."""

    def test_agent_config_tools_field(self):
        config = AgentConfig(
            name="builder",
            archetype="coder",
            tools=["read_file", "edit_file", "run_command"],
        )
        assert len(config.tools) == 3

    def test_agent_runs_with_mock_cognitive(self):
        """Agent produces output using mock model."""
        config = AgentConfig(
            name="test_coder",
            archetype="coder",
            outputs=["changes_summary"],
        )
        mock_config = ModelConfig.mock(
            default_response="## changes_summary\nAdded health check endpoint."
        )
        cognitive = CognitiveLayer(primary_config=mock_config)
        agent = ForgeAgent(config, cognitive)
        result = agent.run({"plan": "Add /health endpoint"})
        assert result.success
        assert "changes_summary" in result.outputs

    def test_agent_fallback_output(self):
        """When no structured output found, whole response goes to first output."""
        config = AgentConfig(
            name="test_planner",
            archetype="planner",
            outputs=["plan"],
        )
        mock_config = ModelConfig.mock(default_response="Step 1: Read files. Step 2: Add endpoint.")
        cognitive = CognitiveLayer(primary_config=mock_config)
        agent = ForgeAgent(config, cognitive)
        result = agent.run({})
        assert result.success
        assert "plan" in result.outputs
        assert "Step 1" in result.outputs["plan"]


# ---------------------------------------------------------------------------
# Research workflow still works (regression)
# ---------------------------------------------------------------------------


class TestResearchWorkflowRegression:
    """Ensure research_report.yaml still loads correctly."""

    def test_research_workflow_loads(self):
        path = Path(__file__).parent.parent / "configs" / "examples" / "research_report.yaml"
        if not path.exists():
            pytest.skip("research_report.yaml not found")
        config = load_workflow(path)
        assert config.name == "research_report"
        assert len(config.agents) == 2
