"""Tests for workflow serialization compatibility.

These tests verify that workflows created by the visual builder
can be loaded by the backend workflow loader.
"""

import pytest

from animus_forge.workflow.loader import (
    load_workflow,
    validate_workflow,
)


class TestVisualBuilderYamlFormat:
    """Test that YAML format from visual builder is compatible with loader."""

    def test_basic_agent_workflow(self, tmp_path):
        """Test basic workflow with agent nodes."""
        yaml_content = """
name: Test Workflow
version: "1.0"
steps:
  - id: plan
    type: claude_code
    params:
      role: planner
      prompt: "Plan the feature"
    outputs:
      - plan
    on_failure: abort

  - id: build
    type: claude_code
    params:
      role: builder
      prompt: "Build the feature"
    depends_on: plan
    on_failure: retry
    max_retries: 2
"""
        workflow_file = tmp_path / "test.yaml"
        workflow_file.write_text(yaml_content)

        config = load_workflow(workflow_file, trusted_dir=tmp_path)

        assert config.name == "Test Workflow"
        assert config.version == "1.0"
        assert len(config.steps) == 2

        # Check first step
        assert config.steps[0].id == "plan"
        assert config.steps[0].type == "claude_code"
        assert config.steps[0].params["role"] == "planner"
        assert config.steps[0].on_failure == "abort"

        # Check second step
        assert config.steps[1].id == "build"
        assert config.steps[1].depends_on == ["plan"]
        assert config.steps[1].max_retries == 2

    def test_shell_node_workflow(self, tmp_path):
        """Test workflow with shell nodes."""
        yaml_content = """
name: Shell Test
version: "1.0"
steps:
  - id: run_tests
    type: shell
    params:
      command: "pytest tests/"
      allow_failure: false
    timeout_seconds: 300
"""
        workflow_file = tmp_path / "shell.yaml"
        workflow_file.write_text(yaml_content)

        config = load_workflow(workflow_file, trusted_dir=tmp_path)

        assert config.steps[0].type == "shell"
        assert config.steps[0].params["command"] == "pytest tests/"
        assert config.steps[0].timeout_seconds == 300

    def test_checkpoint_node_workflow(self, tmp_path):
        """Test workflow with checkpoint nodes."""
        yaml_content = """
name: Checkpoint Test
version: "1.0"
steps:
  - id: step1
    type: claude_code
    params:
      role: planner
      prompt: "Plan"

  - id: review_checkpoint
    type: checkpoint
    params:
      message: "Review the plan before proceeding"
    depends_on: step1

  - id: step2
    type: claude_code
    params:
      role: builder
      prompt: "Build"
    depends_on: review_checkpoint
"""
        workflow_file = tmp_path / "checkpoint.yaml"
        workflow_file.write_text(yaml_content)

        config = load_workflow(workflow_file, trusted_dir=tmp_path)

        assert len(config.steps) == 3
        assert config.steps[1].type == "checkpoint"
        assert config.steps[1].params["message"] == "Review the plan before proceeding"

    def test_multiple_dependencies(self, tmp_path):
        """Test workflow with multiple dependencies (fan-in pattern)."""
        yaml_content = """
name: Fan-In Test
version: "1.0"
steps:
  - id: analyze_code
    type: claude_code
    params:
      role: analyst
      prompt: "Analyze code"

  - id: analyze_tests
    type: claude_code
    params:
      role: analyst
      prompt: "Analyze tests"

  - id: summarize
    type: claude_code
    params:
      role: reporter
      prompt: "Summarize findings"
    depends_on:
      - analyze_code
      - analyze_tests
"""
        workflow_file = tmp_path / "fanin.yaml"
        workflow_file.write_text(yaml_content)

        config = load_workflow(workflow_file, trusted_dir=tmp_path)

        assert len(config.steps[2].depends_on) == 2
        assert "analyze_code" in config.steps[2].depends_on
        assert "analyze_tests" in config.steps[2].depends_on

    def test_single_dependency_string(self, tmp_path):
        """Test that single dependency can be a string (not array)."""
        yaml_content = """
name: Single Dep Test
version: "1.0"
steps:
  - id: step1
    type: claude_code
    params:
      role: planner
      prompt: "Plan"

  - id: step2
    type: claude_code
    params:
      role: builder
      prompt: "Build"
    depends_on: step1
"""
        workflow_file = tmp_path / "singledep.yaml"
        workflow_file.write_text(yaml_content)

        config = load_workflow(workflow_file, trusted_dir=tmp_path)

        # Should be normalized to list
        assert config.steps[1].depends_on == ["step1"]


class TestWorkflowValidation:
    """Test workflow validation for visual builder output."""

    def test_validates_required_fields(self):
        """Test validation catches missing required fields."""
        # Missing name
        errors = validate_workflow({"steps": [{"id": "s1", "type": "shell"}]})
        assert any("name" in e for e in errors)

        # Missing steps
        errors = validate_workflow({"name": "Test"})
        assert any("steps" in e for e in errors)

    def test_validates_step_types(self):
        """Test validation catches invalid step types."""
        errors = validate_workflow(
            {"name": "Test", "steps": [{"id": "s1", "type": "invalid_type"}]}
        )
        assert any("invalid type" in e.lower() for e in errors)

    def test_validates_on_failure_values(self):
        """Test validation catches invalid on_failure values."""
        errors = validate_workflow(
            {
                "name": "Test",
                "steps": [{"id": "s1", "type": "shell", "on_failure": "explode"}],
            }
        )
        assert any("on_failure" in e.lower() for e in errors)

    def test_accepts_valid_workflow(self):
        """Test validation passes for valid workflow."""
        errors = validate_workflow(
            {
                "name": "Valid Workflow",
                "version": "1.0",
                "steps": [
                    {
                        "id": "step1",
                        "type": "claude_code",
                        "params": {"role": "planner", "prompt": "Plan"},
                        "on_failure": "abort",
                    },
                    {
                        "id": "step2",
                        "type": "shell",
                        "params": {"command": "echo hello"},
                        "depends_on": "step1",
                    },
                ],
            }
        )
        assert errors == []


class TestRoundTrip:
    """Test that workflows can be serialized and deserialized."""

    def test_config_to_dict_roundtrip(self, tmp_path):
        """Test WorkflowConfig can be converted back to dict format."""
        yaml_content = """
name: Roundtrip Test
version: "2.0"
description: Test workflow for roundtrip
token_budget: 50000
timeout_seconds: 1800
steps:
  - id: analyze
    type: claude_code
    params:
      role: analyst
      prompt: "Analyze data"
    outputs:
      - analysis
    on_failure: retry
    max_retries: 3
    timeout_seconds: 600
"""
        workflow_file = tmp_path / "roundtrip.yaml"
        workflow_file.write_text(yaml_content)

        config = load_workflow(workflow_file, trusted_dir=tmp_path)

        # Verify loaded correctly
        assert config.name == "Roundtrip Test"
        assert config.version == "2.0"
        assert config.token_budget == 50000
        assert config.timeout_seconds == 1800
        assert config.steps[0].outputs == ["analysis"]
        assert config.steps[0].max_retries == 3


class TestParallelNodeTypes:
    """Test parallel execution node types from visual builder."""

    def test_parallel_node_workflow(self, tmp_path):
        """Test workflow with parallel node."""
        yaml_content = """
name: Parallel Test
version: "1.0"
steps:
  - id: parallel_analysis
    type: parallel
    params:
      strategy: threading
      max_workers: 4
      fail_fast: false
"""
        workflow_file = tmp_path / "parallel.yaml"
        workflow_file.write_text(yaml_content)

        config = load_workflow(workflow_file, trusted_dir=tmp_path)

        assert config.steps[0].type == "parallel"
        assert config.steps[0].params["strategy"] == "threading"
        assert config.steps[0].params["max_workers"] == 4
        assert config.steps[0].params["fail_fast"] is False

    def test_fan_out_node_workflow(self, tmp_path):
        """Test workflow with fan_out node."""
        yaml_content = """
name: Fan Out Test
version: "1.0"
steps:
  - id: scatter_reviews
    type: fan_out
    params:
      items: "${files}"
      max_concurrent: 5
      fail_fast: false
    outputs:
      - file_reviews
"""
        workflow_file = tmp_path / "fanout.yaml"
        workflow_file.write_text(yaml_content)

        config = load_workflow(workflow_file, trusted_dir=tmp_path)

        assert config.steps[0].type == "fan_out"
        assert config.steps[0].params["items"] == "${files}"
        assert config.steps[0].params["max_concurrent"] == 5
        assert config.steps[0].outputs == ["file_reviews"]

    def test_fan_in_node_workflow(self, tmp_path):
        """Test workflow with fan_in node."""
        yaml_content = """
name: Fan In Test
version: "1.0"
steps:
  - id: gather_results
    type: fan_in
    params:
      input: "${file_reviews}"
      aggregation: claude_code
      aggregate_prompt: "Summarize the reviews"
    outputs:
      - summary
"""
        workflow_file = tmp_path / "fanin.yaml"
        workflow_file.write_text(yaml_content)

        config = load_workflow(workflow_file, trusted_dir=tmp_path)

        assert config.steps[0].type == "fan_in"
        assert config.steps[0].params["input"] == "${file_reviews}"
        assert config.steps[0].params["aggregation"] == "claude_code"

    def test_map_reduce_node_workflow(self, tmp_path):
        """Test workflow with map_reduce node."""
        yaml_content = """
name: Map Reduce Test
version: "1.0"
steps:
  - id: analyze_logs
    type: map_reduce
    params:
      items: "${log_files}"
      max_concurrent: 3
      fail_fast: false
      map_prompt: "Analyze this log file"
      reduce_prompt: "Combine all analyses"
    outputs:
      - analysis_report
"""
        workflow_file = tmp_path / "mapreduce.yaml"
        workflow_file.write_text(yaml_content)

        config = load_workflow(workflow_file, trusted_dir=tmp_path)

        assert config.steps[0].type == "map_reduce"
        assert config.steps[0].params["items"] == "${log_files}"
        assert config.steps[0].params["max_concurrent"] == 3
        assert config.steps[0].outputs == ["analysis_report"]

    def test_fan_out_fan_in_pipeline(self, tmp_path):
        """Test complete fan-out/fan-in pipeline."""
        yaml_content = """
name: Scatter-Gather Pipeline
version: "1.0"
steps:
  - id: scatter
    type: fan_out
    params:
      items: "${items}"
      max_concurrent: 5

  - id: gather
    type: fan_in
    params:
      input: "${scatter_results}"
      aggregation: concat
    depends_on: scatter
"""
        workflow_file = tmp_path / "scattergather.yaml"
        workflow_file.write_text(yaml_content)

        config = load_workflow(workflow_file, trusted_dir=tmp_path)

        assert len(config.steps) == 2
        assert config.steps[0].type == "fan_out"
        assert config.steps[1].type == "fan_in"
        assert config.steps[1].depends_on == ["scatter"]


class TestAllAgentRoles:
    """Test all agent roles supported by visual builder."""

    @pytest.mark.parametrize(
        "role",
        [
            "planner",
            "builder",
            "tester",
            "reviewer",
            "architect",
            "documenter",
            "analyst",
            "visualizer",
            "reporter",
        ],
    )
    def test_agent_role(self, role, tmp_path):
        """Test each agent role can be used in workflow."""
        yaml_content = f"""
name: {role.title()} Test
version: "1.0"
steps:
  - id: {role}_step
    type: claude_code
    params:
      role: {role}
      prompt: "Do {role} work"
"""
        workflow_file = tmp_path / f"{role}.yaml"
        workflow_file.write_text(yaml_content)

        config = load_workflow(workflow_file, trusted_dir=tmp_path)

        assert config.steps[0].params["role"] == role


class TestConditionalBranching:
    """Test conditional branching from visual builder."""

    def test_agent_with_condition(self, tmp_path):
        """Test agent node with condition."""
        yaml_content = """
name: Conditional Agent Test
version: "1.0"
steps:
  - id: check_type
    type: claude_code
    params:
      role: planner
      prompt: "Determine file type"
    outputs:
      - file_type

  - id: handle_python
    type: claude_code
    params:
      role: builder
      prompt: "Handle Python file"
    condition:
      field: "file_type"
      operator: equals
      value: "python"
    depends_on: check_type
"""
        workflow_file = tmp_path / "conditional_agent.yaml"
        workflow_file.write_text(yaml_content)

        config = load_workflow(workflow_file, trusted_dir=tmp_path)

        assert len(config.steps) == 2
        assert config.steps[1].condition is not None
        assert config.steps[1].condition.field == "file_type"
        assert config.steps[1].condition.operator == "equals"
        assert config.steps[1].condition.value == "python"

    def test_shell_with_condition(self, tmp_path):
        """Test shell node with condition."""
        yaml_content = """
name: Conditional Shell Test
version: "1.0"
steps:
  - id: check_status
    type: claude_code
    params:
      role: analyst
      prompt: "Check status"
    outputs:
      - should_deploy

  - id: deploy
    type: shell
    params:
      command: "make deploy"
    condition:
      field: "should_deploy"
      operator: equals
      value: true
    depends_on: check_status
"""
        workflow_file = tmp_path / "conditional_shell.yaml"
        workflow_file.write_text(yaml_content)

        config = load_workflow(workflow_file, trusted_dir=tmp_path)

        assert config.steps[1].type == "shell"
        assert config.steps[1].condition is not None
        assert config.steps[1].condition.field == "should_deploy"
        assert config.steps[1].condition.value is True

    @pytest.mark.parametrize(
        "operator", ["equals", "not_equals", "contains", "greater_than", "less_than"]
    )
    def test_condition_operators(self, operator, tmp_path):
        """Test all condition operators."""
        yaml_content = f"""
name: Operator Test
version: "1.0"
steps:
  - id: source
    type: claude_code
    params:
      role: analyst
      prompt: "Analyze"
    outputs:
      - result

  - id: conditional
    type: claude_code
    params:
      role: builder
      prompt: "Build conditionally"
    condition:
      field: "result"
      operator: {operator}
      value: "test_value"
    depends_on: source
"""
        workflow_file = tmp_path / f"operator_{operator}.yaml"
        workflow_file.write_text(yaml_content)

        config = load_workflow(workflow_file, trusted_dir=tmp_path)

        assert config.steps[1].condition.operator == operator

    def test_condition_with_numeric_value(self, tmp_path):
        """Test condition with numeric comparison."""
        yaml_content = """
name: Numeric Condition Test
version: "1.0"
steps:
  - id: count
    type: claude_code
    params:
      role: analyst
      prompt: "Count items"
    outputs:
      - item_count

  - id: process_large
    type: claude_code
    params:
      role: builder
      prompt: "Process large dataset"
    condition:
      field: "item_count"
      operator: greater_than
      value: 100
    depends_on: count
"""
        workflow_file = tmp_path / "numeric_condition.yaml"
        workflow_file.write_text(yaml_content)

        config = load_workflow(workflow_file, trusted_dir=tmp_path)

        assert config.steps[1].condition.value == 100
