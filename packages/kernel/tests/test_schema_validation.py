"""Tests for tool-call schema validation."""

from __future__ import annotations

import pytest

from animus_kernel.contracts.base import ContractViolation
from animus_kernel.tools.registry import ForgeToolRegistry, ToolDefinition
from animus_kernel.tools.schema_validator import (
    ValidatingToolRegistry,
    _cached_model,
    _sanitize,
    _schema_to_fields,
    parse_hermes_tool_call,
    parse_json_tool_call,
    validate_tool_call,
)


class TestSanitize:
    def test_dotted_path(self):
        assert _sanitize("foo.bar-baz") == "Dyn_foo_bar_baz"

    def test_leading_digit(self):
        assert _sanitize("1test") == "Dyn_1test"

    def test_strips_underscores(self):
        assert _sanitize("_foo_") == "Dyn_foo"


class TestSchemaToFields:
    def test_required_and_optional(self):
        schema = {
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
            },
            "required": ["path"],
        }
        fields = _schema_to_fields(schema)
        assert fields["path"] == (str, ...)
        assert fields["start_line"] == (int | None, None)

    def test_object_nested(self):
        schema = {
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {"host": {"type": "string"}},
                    "required": ["host"],
                }
            },
        }
        fields = _schema_to_fields(schema)
        # Optional nested objects get None default; type is a generated BaseModel subclass
        assert fields["config"][1] is None
        assert fields["config"][0] is not type(None)

    def test_enum_literal(self):
        schema = {
            "properties": {
                "action": {"type": "string", "enum": ["read", "write"]},
            }
        }
        fields = _schema_to_fields(schema)
        # Optional enums are wrapped in Optional[Literal[...]]
        assert fields["action"][1] is None
        assert hasattr(fields["action"][0], "__origin__")  # Optional / Union type


class TestCachedModel:
    def test_caches_same_schema(self):
        schema = {"properties": {"x": {"type": "string"}}, "required": ["x"]}
        m1 = _cached_model(schema)
        m2 = _cached_model(schema)
        assert m1 is m2

    def test_forbids_extra_when_set(self):
        schema = {"properties": {}, "additionalProperties": False}
        model = _cached_model(schema)
        with pytest.raises(Exception):
            model(x="oops")


class TestValidateToolCall:
    def test_unknown_tool_raises(self):
        registry = ForgeToolRegistry(project_root="/tmp", enable_shell=False)
        with pytest.raises(ContractViolation, match="Unknown tool"):
            validate_tool_call("does_not_exist", {}, registry)

    def test_valid_params_pass(self):
        registry = ForgeToolRegistry(project_root="/tmp", enable_shell=False)
        registry.register(
            ToolDefinition(
                name="test_echo",
                description="echo",
                parameters={"properties": {"msg": {"type": "string"}}, "required": ["msg"]},
                handler=lambda args: args["msg"],
            )
        )
        validate_tool_call("test_echo", {"msg": "hello"}, registry)

    def test_missing_required_raises(self):
        registry = ForgeToolRegistry(project_root="/tmp", enable_shell=False)
        registry.register(
            ToolDefinition(
                name="test_echo",
                description="echo",
                parameters={"properties": {"msg": {"type": "string"}}, "required": ["msg"]},
                handler=lambda args: args["msg"],
            )
        )
        with pytest.raises(ContractViolation, match="Field required"):
            validate_tool_call("test_echo", {}, registry)

    def test_bad_type_raises(self):
        registry = ForgeToolRegistry(project_root="/tmp", enable_shell=False)
        registry.register(
            ToolDefinition(
                name="test_add",
                description="add",
                parameters={"properties": {"n": {"type": "integer"}}, "required": ["n"]},
                handler=lambda **kw: kw["n"],
            )
        )
        with pytest.raises(ContractViolation, match="Input should be a valid integer"):
            validate_tool_call("test_add", {"n": "not_a_number"}, registry)


class TestParseJsonToolCall:
    def test_anthropic_input_shape(self):
        raw = {"name": "read_file", "input": {"path": "a.py"}}
        name, args = parse_json_tool_call(raw)
        assert name == "read_file"
        assert args == {"path": "a.py"}

    def test_openai_function_shape(self):
        raw = {"function": {"name": "edit_file", "arguments": '{"path": "a.py"}'}}
        name, args = parse_json_tool_call(raw)
        assert name == "edit_file"
        assert args == {"path": "a.py"}

    def test_openai_function_args_already_dict(self):
        raw = {"function": {"name": "edit_file", "arguments": {"path": "a.py"}}}
        name, args = parse_json_tool_call(raw)
        assert args == {"path": "a.py"}

    def test_generic_shape(self):
        raw = {"name": "search", "arguments": {"pattern": "def"}}
        name, args = parse_json_tool_call(raw)
        assert name == "search"
        assert args == {"pattern": "def"}

    def test_bad_json_string_falls_back(self):
        raw = {"function": {"name": "x", "arguments": "not json"}}
        name, args = parse_json_tool_call(raw)
        assert args == {}


class TestParseHermesToolCall:
    def test_standard_shape(self):
        xml = '<tool_call><name>read_file</name><arguments>{"path": "a.py"}</arguments></tool_call>'
        name, args = parse_hermes_tool_call(xml)
        assert name == "read_file"
        assert args == {"path": "a.py"}

    def test_tool_name_alias(self):
        xml = (
            "<tool_call><tool_name>read_file</tool_name>"
            '<arguments>{"path": "a.py"}</arguments></tool_call>'
        )
        name, args = parse_hermes_tool_call(xml)
        assert name == "read_file"

    def test_parameters_alias(self):
        xml = (
            '<tool_call><name>read_file</name><parameters>{"path": "a.py"}</parameters></tool_call>'
        )
        name, args = parse_hermes_tool_call(xml)
        assert args == {"path": "a.py"}

    def test_params_alias(self):
        xml = '<tool_call><name>read_file</name><params>{"path": "a.py"}</params></tool_call>'
        name, args = parse_hermes_tool_call(xml)
        assert args == {"path": "a.py"}

    def test_malformed_xml_raises(self):
        with pytest.raises(ContractViolation, match="Invalid Hermes XML"):
            parse_hermes_tool_call("not xml")

    def test_missing_args_defaults_to_empty(self):
        xml = "<tool_call><name>noop</name></tool_call>"
        name, args = parse_hermes_tool_call(xml)
        assert name == "noop"
        assert args == {}


class TestValidatingToolRegistry:
    def test_delegates_getattr(self):
        inner = ForgeToolRegistry(project_root="/tmp", enable_shell=False)
        inner.register(
            ToolDefinition(
                name="echo",
                description="echo",
                parameters={"properties": {"msg": {"type": "string"}}, "required": ["msg"]},
                handler=lambda args: args["msg"],
            )
        )
        wrapper = ValidatingToolRegistry(inner)
        assert wrapper.get("echo") is not None

    def test_execute_validates_then_delegates(self):
        inner = ForgeToolRegistry(project_root="/tmp", enable_shell=False)
        inner.register(
            ToolDefinition(
                name="echo",
                description="echo",
                parameters={"properties": {"msg": {"type": "string"}}, "required": ["msg"]},
                handler=lambda args: args["msg"],
            )
        )
        wrapper = ValidatingToolRegistry(inner)
        result = wrapper.execute("echo", {"msg": "hi"}, agent_id="test")
        assert result == "hi"

    def test_execute_invalid_raises(self):
        inner = ForgeToolRegistry(project_root="/tmp", enable_shell=False)
        inner.register(
            ToolDefinition(
                name="echo",
                description="echo",
                parameters={"properties": {"msg": {"type": "string"}}, "required": ["msg"]},
                handler=lambda args: args["msg"],
            )
        )
        wrapper = ValidatingToolRegistry(inner)
        with pytest.raises(ContractViolation):
            wrapper.execute("echo", {}, agent_id="test")
