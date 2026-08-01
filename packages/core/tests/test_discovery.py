"""Tests for P5 Tool Auto-Discovery.

Covers MCP scanner, OpenAPI discovery, script discovery, schema validator,
and the orchestrator pipeline.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from animus.discovery.mcp_scanner import MCPScanner, MCPToolSpec
from animus.discovery.openapi_discovery import OpenAPIDiscovery, OpenAPIEndpoint
from animus.discovery.orchestrator import DiscoveryConfig, DiscoveryOrchestrator, DiscoveryRun
from animus.discovery.script_discovery import ScriptDiscovery
from animus.discovery.validator import SchemaValidator
from animus.tools import ToolRegistry

# ── MCP Scanner Tests ─────────────────────────────────────────────


class TestMCPScanner:
    def test_init(self):
        scanner = MCPScanner(timeout=5.0)
        assert scanner.timeout == 5.0

    def test_scan_server_no_requests(self):
        scanner = MCPScanner()
        # Without requests installed, should return empty list
        specs = scanner.scan_server("http://localhost:3000/sse")
        assert isinstance(specs, list)

    def test_mcp_tool_spec_to_animus_schema(self):
        spec = MCPToolSpec(
            name="test-tool",
            description="A test tool",
            input_schema={
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            server_name="test-server",
        )
        schema = spec.to_animus_schema()
        assert schema["type"] == "object"
        assert "x" in schema["properties"]
        assert "required" in schema

    def test_scan_stdio_timeout(self):
        scanner = MCPScanner(timeout=0.1)
        specs = scanner.scan_stdio_server(["sleep", "10"], "slow-server")
        assert specs == []  # Should timeout

    def test_scan_local_servers_empty(self):
        scanner = MCPScanner(timeout=0.1)
        specs = scanner.scan_local_servers(ports=[99999])
        assert specs == []


# ── OpenAPI Discovery Tests ───────────────────────────────────────


class TestOpenAPIDiscovery:
    def test_parse_spec_basic(self):
        discovery = OpenAPIDiscovery()
        spec = {
            "openapi": "3.0.0",
            "paths": {
                "/users": {
                    "get": {
                        "operationId": "listUsers",
                        "summary": "List users",
                        "description": "Get all users",
                    }
                }
            },
        }
        endpoints = discovery._parse_spec(spec, base_url="http://test")
        assert len(endpoints) == 1
        assert endpoints[0].tool_name == "listUsers"
        assert endpoints[0].method == "GET"
        assert endpoints[0].path == "/users"

    def test_parse_spec_with_parameters(self):
        discovery = OpenAPIDiscovery()
        spec = {
            "openapi": "3.0.0",
            "paths": {
                "/users/{id}": {
                    "get": {
                        "operationId": "getUser",
                        "parameters": [
                            {
                                "name": "id",
                                "in": "path",
                                "required": True,
                                "schema": {"type": "integer"},
                            }
                        ],
                    }
                }
            },
        }
        endpoints = discovery._parse_spec(spec, base_url="http://test")
        assert len(endpoints) == 1
        assert "id" in endpoints[0].parameters["properties"]
        assert "id" in endpoints[0].parameters["required"]

    def test_parse_spec_with_request_body(self):
        discovery = OpenAPIDiscovery()
        spec = {
            "openapi": "3.0.0",
            "paths": {
                "/users": {
                    "post": {
                        "operationId": "createUser",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "email": {"type": "string"},
                                        },
                                        "required": ["name", "email"],
                                    }
                                }
                            }
                        },
                    }
                }
            },
        }
        endpoints = discovery._parse_spec(spec, base_url="http://test")
        assert len(endpoints) == 1
        props = endpoints[0].parameters["properties"]
        assert "name" in props
        assert "email" in props
        assert "name" in endpoints[0].parameters["required"]

    def test_tool_name_from_path(self):
        ep = OpenAPIEndpoint(
            method="GET",
            path="/users/list",
            operation_id="",
            summary="",
            description="",
            parameters={},
        )
        assert ep.tool_name == "get-users-list"

    def test_scan_directory(self):
        with tempfile.TemporaryDirectory() as td:
            spec_file = Path(td) / "test-spec.json"
            spec_file.write_text(
                json.dumps(
                    {
                        "openapi": "3.0.0",
                        "paths": {
                            "/items": {
                                "get": {
                                    "operationId": "listItems",
                                    "summary": "List items",
                                }
                            }
                        },
                    }
                )
            )
            discovery = OpenAPIDiscovery()
            endpoints = discovery.scan_directory(td)
            assert len(endpoints) == 1
            assert endpoints[0].tool_name == "listItems"

    def test_scan_directory_no_specs(self):
        with tempfile.TemporaryDirectory() as td:
            discovery = OpenAPIDiscovery()
            endpoints = discovery.scan_directory(td)
            assert endpoints == []

    def test_convert_schema_nested(self):
        discovery = OpenAPIDiscovery()
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                }
            },
        }
        result = discovery._convert_schema(schema)
        assert result["type"] == "object"
        assert "properties" in result["properties"]["user"]


# ── Script Discovery Tests ────────────────────────────────────────


class TestScriptDiscovery:
    def test_scan_directory_no_scripts(self):
        with tempfile.TemporaryDirectory() as td:
            discovery = ScriptDiscovery()
            specs = discovery.scan_directory(td)
            assert specs == []

    def test_parse_annotated_python_script(self):
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "test_tool.py"
            script.write_text('''
"""Animus Tool: my-test-tool

This tool does something useful.

Args:
    input_file: Path to input file (type: str, required)
    verbose: Enable verbose mode (type: bool)
"""

def main():
    pass
''')
            discovery = ScriptDiscovery()
            spec = discovery._parse_script(script)
            assert spec is not None
            assert spec.name == "my-test-tool"
            assert "useful" in spec.description
            assert "input_file" in spec.parameters["properties"]
            assert "input_file" in spec.parameters["required"]

    def test_parse_script_no_annotation(self):
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "plain.py"
            script.write_text("print('hello')")
            discovery = ScriptDiscovery()
            spec = discovery._parse_script(script)
            assert spec is None

    def test_parse_bash_script(self):
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "test.sh"
            script.write_text("""#!/bin/bash
# Animus Tool: bash-test
# Description of the bash tool

echo "hello"
""")
            discovery = ScriptDiscovery()
            spec = discovery._parse_script(script)
            assert spec is not None
            assert spec.name == "bash-test"
            assert spec.interpreter == "bash"

    def test_map_types(self):
        assert ScriptDiscovery._map_type("str") == "string"
        assert ScriptDiscovery._map_type("int") == "integer"
        assert ScriptDiscovery._map_type("float") == "number"
        assert ScriptDiscovery._map_type("bool") == "boolean"
        assert ScriptDiscovery._map_type("list") == "array"
        assert ScriptDiscovery._map_type("unknown") == "string"

    def test_parse_args_block(self):
        discovery = ScriptDiscovery()
        block = """
            input_file: Path to the file (type: str, required)
            verbose: Enable verbose output (type: bool)
            count: Number of items (type: int)
        """
        params = discovery._parse_args_block(block)
        assert "input_file" in params["properties"]
        assert params["properties"]["input_file"]["type"] == "string"
        assert "input_file" in params["required"]
        assert "verbose" in params["properties"]
        assert "count" in params["properties"]
        assert params["properties"]["count"]["type"] == "integer"


# ── Schema Validator Tests ──────────────────────────────────────


class TestSchemaValidator:
    def test_valid_schema_passes(self):
        validator = SchemaValidator(min_score=0.6)
        schema = {
            "name": "test-tool",
            "description": "Fetches user data from the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "The user ID"},
                },
                "required": ["user_id"],
            },
        }
        result = validator.validate_tool_schema(schema)
        assert result.passed is True
        assert result.score >= 0.6
        assert len(result.errors) == 0

    def test_missing_name_fails(self):
        validator = SchemaValidator()
        schema = {
            "description": "A tool",
            "parameters": {"type": "object", "properties": {}},
        }
        result = validator.validate_tool_schema(schema)
        assert result.passed is False
        assert any("name" in e.lower() for e in result.errors)

    def test_missing_description_fails(self):
        validator = SchemaValidator()
        schema = {
            "name": "test-tool",
            "parameters": {"type": "object", "properties": {}},
        }
        result = validator.validate_tool_schema(schema)
        assert result.passed is False
        assert any("description" in e.lower() for e in result.errors)

    def test_short_description_warning(self):
        validator = SchemaValidator()
        schema = {
            "name": "test",
            "description": "Get data",
            "parameters": {"type": "object", "properties": {}},
        }
        result = validator.validate_tool_schema(schema)
        assert any("short" in w.lower() for w in result.warnings)

    def test_dangerous_pattern_warning(self):
        validator = SchemaValidator()
        schema = {
            "name": "exec-tool",
            "description": "Executes shell commands via eval",
            "parameters": {"type": "object", "properties": {}},
        }
        result = validator.validate_tool_schema(schema)
        assert any("eval" in w.lower() for w in result.warnings)

    def test_invalid_parameters_fails(self):
        validator = SchemaValidator()
        schema = {
            "name": "test",
            "description": "A test tool that does something useful",
            "parameters": "not a dict",
        }
        result = validator.validate_tool_schema(schema)
        assert result.passed is False

    def test_batch_validation(self):
        validator = SchemaValidator()
        schemas = [
            {
                "name": "good",
                "description": "A valid tool",
                "parameters": {"type": "object", "properties": {}},
            },
            {"name": "bad", "description": "", "parameters": {"type": "object", "properties": {}}},
        ]
        passed, failed = validator.validate_batch(schemas)
        assert len(passed) == 1
        assert len(failed) == 1


# ── Orchestrator Tests ──────────────────────────────────────────


class TestDiscoveryConfig:
    def test_defaults(self):
        config = DiscoveryConfig()
        assert config.min_validation_score == 0.6
        assert config.max_new_tools_per_run == 20
        assert config.deduplicate_by_hash is True

    def test_has_sources_with_servers(self):
        config = DiscoveryConfig(mcp_servers=["http://test"])
        assert config.has_sources is True

    def test_has_sources_empty(self):
        config = DiscoveryConfig(scan_localhost_ports=[])
        assert config.has_sources is False


class TestDiscoveryOrchestrator:
    def test_init_defaults(self):
        orch = DiscoveryOrchestrator()
        assert orch.config.enabled is True
        assert orch.registry is not None
        assert orch.validator is not None

    def test_run_discovery_disabled(self):
        config = DiscoveryConfig(enabled=False)
        orch = DiscoveryOrchestrator(config=config)
        run = orch.run_discovery()
        assert run.tools_discovered == 0
        assert run.completed_at is not None

    def test_run_discovery_no_sources(self):
        config = DiscoveryConfig()
        orch = DiscoveryOrchestrator(config=config)
        run = orch.run_discovery()
        assert run.tools_discovered == 0

    def test_run_discovery_with_scripts(self):
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "test_tool.py"
            script.write_text('''
"""Animus Tool: script-test

A useful script tool.

Args:
    param1: First parameter (type: str, required)
"""

def main():
    pass
''')
            config = DiscoveryConfig(
                script_dirs=[td],
                persistence_dir=td,
                min_validation_score=0.4,
                scan_localhost_ports=[],
            )
            orch = DiscoveryOrchestrator(config=config)
            run = orch.run_discovery()

            assert run.tools_discovered == 1
            assert run.tools_validated == 1
            assert run.tools_registered == 1
            assert len(run.errors) == 0

    def test_deduplication(self):
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "test_tool.py"
            script.write_text('''
"""Animus Tool: duplicate-test

A useful tool for testing deduplication logic.

Args:
    x: Param (type: str)
"""

def main():
    pass
''')
            config = DiscoveryConfig(
                script_dirs=[td],
                persistence_dir=td,
                min_validation_score=0.4,
                scan_localhost_ports=[],
            )
            orch = DiscoveryOrchestrator(config=config)

            # First run
            run1 = orch.run_discovery()
            assert run1.tools_registered == 1

            # Second run should find no new tools (deduplicated)
            run2 = orch.run_discovery()
            assert run2.tools_registered == 0

    def test_max_tools_per_run(self):
        with tempfile.TemporaryDirectory() as td:
            for i in range(5):
                script = Path(td) / f"tool_{i}.py"
                script.write_text(f'''
"""Animus Tool: tool-{i}

Tool {i} description.

Args:
    x: Param (type: str)
"""
def main(): pass
''')
            config = DiscoveryConfig(
                script_dirs=[td],
                persistence_dir=td,
                max_new_tools_per_run=2,
                min_validation_score=0.4,
                scan_localhost_ports=[],
            )
            orch = DiscoveryOrchestrator(config=config)
            run = orch.run_discovery()

            assert run.tools_discovered == 5
            assert run.tools_registered == 2  # Capped

    def test_placeholder_handler(self):
        orch = DiscoveryOrchestrator()
        handler = orch._make_placeholder_handler("test-tool")
        result = handler({})
        assert result.success is False
        assert "not yet wired" in result.error

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            config = DiscoveryConfig(
                script_dirs=[td],
                persistence_dir=td,
                min_validation_score=0.4,
                scan_localhost_ports=[],
            )
            script = Path(td) / "test.py"
            script.write_text('''
"""Animus Tool: persist-test

A useful tool for persistence testing.

Args:
    x: Param (type: str)
"""
def main(): pass
''')
            orch = DiscoveryOrchestrator(config=config)
            run = orch.run_discovery()

            # Check hashes persisted
            hash_file = Path(td) / "discovered_hashes.json"
            assert hash_file.exists()

            # Check history persisted
            history_file = Path(td) / "discovery_history.jsonl"
            assert history_file.exists()

    def test_get_status(self):
        orch = DiscoveryOrchestrator()
        status = orch.get_status()
        assert "config_sources" in status
        assert "registry_size" in status
        assert status["enabled"] is True

    def test_get_history(self):
        with tempfile.TemporaryDirectory() as td:
            config = DiscoveryConfig(
                script_dirs=[td],
                persistence_dir=td,
                min_validation_score=0.4,
            )
            script = Path(td) / "test.py"
            script.write_text('''
"""Animus Tool: hist-test

A useful tool for history testing.

Args:
    x: Param (type: str)
"""
def main(): pass
''')
            config = DiscoveryConfig(
                script_dirs=[td],
                persistence_dir=td,
                min_validation_score=0.4,
                scan_localhost_ports=[],
            )
            orch = DiscoveryOrchestrator(config=config)
            orch.run_discovery()

            history = orch.get_history(limit=5)
            assert len(history) == 1
            assert history[0]["tools_discovered"] == 1


# ── Integration Tests ─────────────────────────────────────────────


class TestDiscoveryPipeline:
    def test_full_pipeline(self):
        with tempfile.TemporaryDirectory() as td:
            # Create a script
            script = Path(td) / "pipeline_test.py"
            script.write_text('''
"""Animus Tool: pipeline-tool

A pipeline test tool.

Args:
    input_path: Path to input (type: str, required)
    output_path: Path to output (type: str)
"""

def main():
    pass
''')
            # Create an OpenAPI spec
            spec = Path(td) / "api.json"
            spec.write_text(
                json.dumps(
                    {
                        "openapi": "3.0.0",
                        "paths": {
                            "/items": {
                                "get": {
                                    "operationId": "listItems",
                                    "summary": "List items",
                                }
                            }
                        },
                    }
                )
            )

            config = DiscoveryConfig(
                script_dirs=[td],
                openapi_dirs=[td],
                persistence_dir=td,
                min_validation_score=0.4,
                max_new_tools_per_run=10,
                scan_localhost_ports=[],
            )
            orch = DiscoveryOrchestrator(config=config)
            run = orch.run_discovery()

            assert run.tools_discovered == 2
            assert run.sources_scanned == 2  # scripts + openapi
            assert run.tools_registered >= 1
            assert run.completed_at is not None
            assert run.duration_seconds >= 0

    def test_orchestrator_with_registry(self):
        registry = ToolRegistry()
        config = DiscoveryConfig(enabled=False)
        orch = DiscoveryOrchestrator(config=config, registry=registry)
        assert orch.registry is registry


class TestDiscoveryRun:
    def test_run_duration(self):
        run = DiscoveryRun(run_id="test", started_at=__import__("datetime").datetime.now())
        assert run.duration_seconds == 0.0
        run.completed_at = __import__("datetime").datetime.now()
        assert run.duration_seconds >= 0
