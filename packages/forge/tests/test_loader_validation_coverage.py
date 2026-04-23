"""Coverage for loader.py validation-branch gaps on 2c8cde2.

Targets the error-branch arms that no existing test exercises:
- 54: ConditionConfig.evaluate unknown-operator fallthrough → False
- 393: inputs must be object
- 399: output names must be strings
- 427: name must be non-empty
- 484: invalid condition operator
- 509: depends_on must be string or list
- 511: all depends_on values must be strings
- 529: step id must be non-empty
- 581-583: list_workflows swallows YAML parse errors
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "src")

from animus_forge.workflow.loader import (
    ConditionConfig,
    list_workflows,
    validate_workflow,
)


class TestConditionEvaluateUnknownOperator:
    def test_unknown_operator_returns_false(self):
        """Line 54: final fallthrough when operator isn't recognized."""
        cond = ConditionConfig(field="x", operator="mystery_op", value=1)
        assert cond.evaluate({"x": 1}) is False


class TestValidateWorkflow:
    def test_inputs_must_be_object(self):
        errors = validate_workflow(
            {"name": "w", "steps": [{"id": "s1", "type": "shell"}], "inputs": []}
        )
        assert any("inputs" in e and "object" in e for e in errors)

    def test_output_names_must_be_strings(self):
        errors = validate_workflow(
            {
                "name": "w",
                "steps": [{"id": "s1", "type": "shell"}],
                "outputs": ["ok", 42],
            }
        )
        assert any("output names must be strings" in e for e in errors)

    def test_name_must_be_non_empty(self):
        errors = validate_workflow({"name": "   ", "steps": []})
        assert any("non-empty" in e and "name" in e for e in errors)

    def test_invalid_condition_operator(self):
        errors = validate_workflow(
            {
                "name": "w",
                "steps": [
                    {
                        "id": "s1",
                        "type": "shell",
                        "condition": {
                            "field": "x",
                            "operator": "wat",
                            "value": 1,
                        },
                    }
                ],
            }
        )
        assert any("invalid condition operator" in e for e in errors)

    def test_depends_on_must_be_string_or_list(self):
        errors = validate_workflow(
            {
                "name": "w",
                "steps": [
                    {"id": "s1", "type": "shell", "depends_on": 42},
                ],
            }
        )
        assert any("depends_on must be a string or list" in e for e in errors)

    def test_depends_on_values_must_be_strings(self):
        errors = validate_workflow(
            {
                "name": "w",
                "steps": [
                    {"id": "s1", "type": "shell", "depends_on": ["ok", 42]},
                ],
            }
        )
        assert any("all depends_on values must be strings" in e for e in errors)

    def test_step_id_must_be_non_empty(self):
        errors = validate_workflow(
            {"name": "w", "steps": [{"id": "   ", "type": "shell"}]}
        )
        assert any("non-empty string" in e for e in errors)


class TestListWorkflowsSwallowsBadYaml:
    def test_unreadable_yaml_is_skipped(self):
        """Lines 581-583: unparseable YAML file is silently skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            (tmpdir / "good.yaml").write_text("name: good-wf\nversion: '1.0'\n")
            (tmpdir / "bad.yaml").write_text(":::not: valid yaml: ]\n[\n")

            result = list_workflows(tmpdir)

            names = {w["name"] for w in result}
            assert "good-wf" in names
            # Bad file simply doesn't contribute; no exception raised.
            assert "bad" not in names
