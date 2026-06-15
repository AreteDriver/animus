"""Tool-call schema validation using Pydantic.

Enforces JSON Schema compliance on every tool call before it reaches
the underlying handler. Supports both XML (Hermes) and JSON
(OpenAI/Anthropic) tool call formats.
"""

from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from animus_kernel.contracts.base import ContractViolation
from animus_kernel.tools.registry import ForgeToolRegistry

logger = logging.getLogger(__name__)

# Cache of compiled Pydantic models keyed by canonical JSON Schema hash.
_MODEL_CACHE: dict[str, type[BaseModel]] = {}


def _sanitize(name: str) -> str:
    """Turn a dotted path into a valid Python identifier."""
    cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in name.replace("-", "_"))
    cleaned = cleaned.strip("_")
    return f"Dyn_{cleaned}"


def _prop_to_type(prop: dict[str, Any], path: str) -> type[Any]:
    """Map a single JSON Schema property to a Python type annotation.

    Objects become inline ``BaseModel`` subclasses; arrays become
    ``list[T]``. Primitives map to the obvious Python builtins.
    """
    js_type = prop.get("type")
    enum_vals = prop.get("enum")

    if js_type == "object":
        fields = _schema_to_fields(prop, path)
        extra = "forbid" if prop.get("additionalProperties") is False else "ignore"
        return create_model(
            _sanitize(path),
            __config__=ConfigDict(extra=extra),
            **fields,
        )
    if js_type == "array":
        item_type = _prop_to_type(prop.get("items", {}), f"{path}_item")
        return list[item_type]
    if js_type == "string":
        if enum_vals and all(
            isinstance(v, (str, int, bool, type(None))) for v in enum_vals
        ):
            return Literal[*enum_vals]  # type: ignore[return-value]
        return str
    if js_type == "integer":
        return int
    if js_type == "boolean":
        return bool
    if js_type == "number":
        return float
    if js_type == "null":
        return type(None)
    return Any


def _schema_to_fields(
    schema: dict[str, Any], prefix: str = ""
) -> dict[str, tuple[type[Any], Any]]:
    """Convert JSON Schema ``properties`` into ``create_model`` kwargs."""
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, tuple[type[Any], Any]] = {}
    for key, prop in props.items():
        py_type = _prop_to_type(prop, f"{prefix}_{key}" if prefix else key)
        if key not in required:
            py_type = py_type | None
            default = None
        else:
            default = ...
        fields[key] = (py_type, default)
    return fields


def _cached_model(schema: dict[str, Any]) -> type[BaseModel]:
    """Return a cached Pydantic model for a JSON Schema dict."""
    key = json.dumps(schema, sort_keys=True)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    extra = "forbid" if schema.get("additionalProperties") is False else "ignore"
    fields = _schema_to_fields(schema)
    model = create_model(
        "ToolParams",
        __config__=ConfigDict(extra=extra),
        **fields,
    )
    _MODEL_CACHE[key] = model
    return model


def validate_tool_call(tool_name: str, params: dict, registry: ForgeToolRegistry) -> None:
    """Validate a tool call against its JSON Schema definition.

    Args:
        tool_name: Name of the tool to validate.
        params: Parsed parameters dict (normalised from either JSON or XML).
        registry: Tool registry containing the schema.

    Raises:
        ContractViolation: If the parameters are invalid.
    """
    tool = registry.get(tool_name)
    if tool is None:
        raise ContractViolation(
            message=f"Unknown tool '{tool_name}'",
            field_path="tool_name",
        )

    try:
        model_cls = _cached_model(tool.parameters)
        model_cls(**params)
    except ValidationError as exc:
        err = exc.errors()[0]
        loc = err.get("loc", ())
        field_path = "params" + "".join(f".{part}" for part in loc) if loc else "params"
        raise ContractViolation(
            message=err.get("msg", "Validation error"),
            field_path=field_path,
        )


def parse_json_tool_call(raw: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Extract *(tool_name, params)* from an OpenAI / Anthropic JSON tool call.

    Handles multiple common payload shapes:

    * Anthropic ``tool_use``: ``{"name": "...", "input": {...}}``
    * OpenAI ``function``:
      ``{"function": {"name": "...", "arguments": "json_str"}}``
    * Generic: ``{"name": "...", "arguments": {...}}``

    Args:
        raw: Raw tool call dict from the provider response.

    Returns:
        Tuple of *(tool_name, params_dict)*.
    """
    if "input" in raw:
        return raw.get("name", ""), raw.get("input", {})

    if "function" in raw:
        fn = raw["function"]
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}
        return name, args

    return raw.get("name", ""), raw.get("arguments", {})


def parse_hermes_tool_call(xml_str: str) -> tuple[str, dict[str, Any]]:
    """Extract *(tool_name, params)* from a Hermes XML tool call.

    Expected shape::

        <tool_call>
            <name>read_file</name>
            <arguments>{"path": "src/main.py"}</arguments>
        </tool_call>

    Args:
        xml_str: XML string from the model response.

    Returns:
        Tuple of *(tool_name, params_dict)*.

    Raises:
        ContractViolation: If the XML is malformed.
    """
    try:
        root = ET.fromstring(xml_str.strip())
    except ET.ParseError as exc:
        raise ContractViolation(
            message=f"Invalid Hermes XML: {exc}",
            field_path="tool_call",
        )

    name_el = root.find("name")
    if name_el is None:
        name_el = root.find("tool_name")
    args_el = root.find("arguments")
    if args_el is None:
        args_el = root.find("parameters")
    if args_el is None:
        args_el = root.find("params")

    name = name_el.text if name_el is not None else ""
    raw_args = args_el.text if args_el is not None else "{}"
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except (json.JSONDecodeError, TypeError):
            args = {}
    else:
        args = {}
    return name, args


class ValidatingToolRegistry:
    """Transparent wrapper that runs schema validation before execution.

    Delegates all other attribute access to the underlying registry so it
    remains compatible with :class:`~animus_kernel.agents.provider_wrapper.AgentProvider`
    without modifying the provider code.
    """

    def __init__(self, registry: ForgeToolRegistry) -> None:
        self._registry = registry

    def __getattr__(self, name: str) -> Any:
        return getattr(self._registry, name)

    def execute(self, tool_name: str, arguments: dict[str, Any], agent_id: str = "") -> str:
        validate_tool_call(tool_name, arguments, self._registry)
        return self._registry.execute(tool_name, arguments, agent_id)


if __name__ == "__main__":

    from animus_kernel.tools.registry import ForgeToolRegistry

    registry = ForgeToolRegistry(project_root="/tmp", enable_shell=True)

    # Warm cache so the benchmark measures steady-state throughput.
    validate_tool_call("read_file", {"path": "a.py"}, registry)

    start = time.perf_counter()
    for _ in range(1000):
        validate_tool_call("read_file", {"path": "a.py"}, registry)
    elapsed = time.perf_counter() - start
    print(f"1000 validations in {elapsed:.3f}s ({elapsed * 1000:.1f}ms total)")
    assert elapsed < 1.0, f"Benchmark failed: {elapsed:.3f}s"

    # Missing required field
    try:
        validate_tool_call("edit_file", {"path": "a.py"}, registry)
    except ContractViolation as e:
        assert e.field == "params.old_string", f"Bad field: {e.field}"
        print(f"Missing field caught: {e.field} -> {e}")

    # Bad type
    try:
        validate_tool_call("read_file", {"path": "a.py", "start_line": "oops"}, registry)
    except ContractViolation as e:
        assert e.field == "params.start_line"
        print(f"Bad type caught: {e.field} -> {e}")

    # JSON parser
    jname, jargs = parse_json_tool_call(
        {"name": "search_code", "input": {"pattern": "def"}}
    )
    validate_tool_call(jname, jargs, registry)

    # Hermes XML parser
    xname, xargs = parse_hermes_tool_call(
        '<tool_call><name>run_command</name>'
        '<arguments>{"command": "echo hi"}</arguments></tool_call>'
    )
    validate_tool_call(xname, xargs, registry)

    print("All schema_validator checks passed.")
