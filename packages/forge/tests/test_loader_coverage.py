"""Additional coverage tests for workflow loader."""

import sys

sys.path.insert(0, "src")

from animus_forge.workflow.loader import (
    ConditionConfig,
    FallbackConfig,
    StepConfig,
    WorkflowConfig,
    WorkflowSettings,
)


class TestStepConfigFromDict:
    def test_basic(self):
        step = StepConfig.from_dict(
            {
                "id": "s1",
                "type": "shell",
                "params": {"command": "echo hi"},
            }
        )
        assert step.id == "s1"
        assert step.type == "shell"

    def test_with_outputs(self):
        step = StepConfig.from_dict(
            {
                "id": "s1",
                "type": "shell",
                "outputs": ["stdout"],
            }
        )
        assert step.outputs == ["stdout"]

    def test_with_depends_on(self):
        step = StepConfig.from_dict(
            {
                "id": "s2",
                "type": "shell",
                "depends_on": ["s1"],
            }
        )
        assert step.depends_on == ["s1"]

    def test_with_condition(self):
        step = StepConfig.from_dict(
            {
                "id": "s1",
                "type": "shell",
                "condition": {"field": "ready", "operator": "equals", "value": "true"},
            }
        )
        assert step.condition is not None

    def test_with_fallback(self):
        step = StepConfig.from_dict(
            {
                "id": "s1",
                "type": "shell",
                "on_failure": "fallback",
                "fallback": {"type": "default_value", "value": "fallback_val"},
            }
        )
        assert step.fallback is not None
        assert step.fallback.type == "default_value"

    def test_with_all_options(self):
        step = StepConfig.from_dict(
            {
                "id": "s1",
                "type": "claude_code",
                "params": {"prompt": "test"},
                "on_failure": "skip",
                "max_retries": 5,
                "timeout_seconds": 600,
                "default_output": {"key": "val"},
                "circuit_breaker_key": "claude",
            }
        )
        assert step.on_failure == "skip"
        assert step.max_retries == 5
        assert step.timeout_seconds == 600


class TestStepConfigTopLevelPromotion:
    """Issue #37: top-level fields used to be silently dropped.
    Now they're promoted into params unless they're recognized
    StepConfig fields."""

    def test_top_level_model_and_prompt_promoted(self):
        # Exact pattern from fleet-degraded.yaml + siblings
        step = StepConfig.from_dict(
            {
                "id": "assess",
                "type": "ollama",
                "description": "Assess severity and recommend action",
                "model": "qwen2.5:14b",
                "prompt": "A fleet service is degraded...",
            }
        )
        assert step.params["model"] == "qwen2.5:14b"
        assert step.params["prompt"] == "A fleet service is degraded..."
        assert step.id == "assess"
        assert step.type == "ollama"

    def test_explicit_params_take_precedence_over_top_level(self):
        # If a user writes both, explicit nested wins
        step = StepConfig.from_dict(
            {
                "id": "s1",
                "type": "ollama",
                "model": "TOP-LEVEL-LOSES",
                "params": {"model": "PARAMS-WIN"},
            }
        )
        assert step.params["model"] == "PARAMS-WIN"

    def test_recognized_top_level_keys_not_promoted(self):
        # depends_on, on_failure, etc. must NOT end up in params
        step = StepConfig.from_dict(
            {
                "id": "s1",
                "type": "shell",
                "depends_on": ["other"],
                "on_failure": "skip",
                "outputs": ["x"],
                "max_retries": 7,
                "command": "echo hi",  # Promoted
            }
        )
        assert "depends_on" not in step.params
        assert "on_failure" not in step.params
        assert "outputs" not in step.params
        assert "max_retries" not in step.params
        assert step.params["command"] == "echo hi"

    def test_description_at_top_level_not_promoted(self):
        # description is a documentation field, ignored by executor.
        # Should not pollute params.
        step = StepConfig.from_dict(
            {
                "id": "s1",
                "type": "ollama",
                "description": "Just a comment",
                "prompt": "hi",
            }
        )
        assert "description" not in step.params
        assert step.params["prompt"] == "hi"

    def test_promotion_preserves_other_top_level_unrecognized(self):
        # Multiple unknown keys all merge into params
        step = StepConfig.from_dict(
            {
                "id": "s1",
                "type": "custom",
                "foo": 1,
                "bar": "two",
                "baz": [3],
            }
        )
        assert step.params == {"foo": 1, "bar": "two", "baz": [3]}

    def test_no_promotion_when_only_recognized_keys(self):
        # Backward-compat: legacy workflows that only use known
        # top-level keys + nested params behave exactly as before
        step = StepConfig.from_dict(
            {
                "id": "s1",
                "type": "shell",
                "params": {"command": "ls"},
                "outputs": ["files"],
            }
        )
        assert step.params == {"command": "ls"}
        assert step.outputs == ["files"]


class TestConditionConfig:
    def test_equals(self):
        cond = ConditionConfig(field="status", operator="equals", value="success")
        assert cond.evaluate({"status": "success"}) is True
        assert cond.evaluate({"status": "failed"}) is False

    def test_not_equals(self):
        cond = ConditionConfig(field="status", operator="not_equals", value="failed")
        assert cond.evaluate({"status": "success"}) is True
        assert cond.evaluate({"status": "failed"}) is False

    def test_contains(self):
        cond = ConditionConfig(field="items", operator="contains", value="error")
        assert cond.evaluate({"items": "has error here"}) is True
        assert cond.evaluate({"items": "all good"}) is False

    def test_not_empty(self):
        cond = ConditionConfig(field="data", operator="not_empty")
        assert cond.evaluate({"data": "something"}) is True
        assert cond.evaluate({"data": ""}) is False
        assert cond.evaluate({}) is False

    def test_in_operator(self):
        cond = ConditionConfig(field="status", operator="in", value=["success", "partial"])
        assert cond.evaluate({"status": "success"}) is True
        assert cond.evaluate({"status": "failed"}) is False

    def test_greater_than(self):
        cond = ConditionConfig(field="count", operator="greater_than", value=5)
        assert cond.evaluate({"count": 10}) is True
        assert cond.evaluate({"count": 3}) is False

    def test_less_than(self):
        cond = ConditionConfig(field="count", operator="less_than", value=5)
        assert cond.evaluate({"count": 3}) is True
        assert cond.evaluate({"count": 10}) is False


class TestFallbackConfig:
    def test_default_value(self):
        fb = FallbackConfig(type="default_value", value="fallback")
        assert fb.type == "default_value"
        assert fb.value == "fallback"

    def test_alternate_step(self):
        fb = FallbackConfig(
            type="alternate_step",
            step={"id": "alt", "type": "shell", "params": {"command": "echo alt"}},
        )
        assert fb.step is not None

    def test_callback(self):
        fb = FallbackConfig(type="callback", callback="my_callback")
        assert fb.callback == "my_callback"


class TestWorkflowSettings:
    def test_defaults(self):
        settings = WorkflowSettings()
        assert settings.auto_parallel is False
        assert settings.auto_parallel_max_workers == 4

    def test_custom(self):
        settings = WorkflowSettings(
            auto_parallel=True,
            auto_parallel_max_workers=8,
        )
        assert settings.auto_parallel is True
        assert settings.auto_parallel_max_workers == 8


class TestWorkflowConfigFromDict:
    def test_basic(self):
        wf = WorkflowConfig.from_dict(
            {
                "name": "test-wf",
                "version": "1.0",
                "description": "A test",
                "steps": [
                    {"id": "s1", "type": "shell", "params": {"command": "echo hi"}},
                ],
            }
        )
        assert wf.name == "test-wf"
        assert len(wf.steps) == 1

    def test_with_settings(self):
        wf = WorkflowConfig.from_dict(
            {
                "name": "wf",
                "version": "1.0",
                "description": "test",
                "steps": [],
                "settings": {
                    "auto_parallel": True,
                    "auto_parallel_max_workers": 8,
                },
            }
        )
        assert wf.settings.auto_parallel is True

    def test_with_inputs_outputs(self):
        wf = WorkflowConfig.from_dict(
            {
                "name": "wf",
                "version": "1.0",
                "description": "test",
                "steps": [],
                "inputs": {"name": {"required": True}},
                "outputs": ["result"],
            }
        )
        assert "name" in wf.inputs
        assert "result" in wf.outputs
