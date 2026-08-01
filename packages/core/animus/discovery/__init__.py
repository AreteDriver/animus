"""Animus Tool Auto-Discovery: dynamic tool registration from MCP servers, OpenAPI specs, and local scripts.

Implements P-20260706-006: Background tool discovery with daemon integration.
Inspired by MCP ecosystem auto-discovery and OpenAPI-to-function conversion patterns.

Key design:
- Pluggable scanners: MCP, OpenAPI, local scripts
- Schema validation: rubric-based quality gates (reuses P2 eval infrastructure)
- Lazy registration: discovered tools use compact schemas until first invocation
- Deduplication: hash-based identity prevents duplicate registrations
- Persistence: discovered tools survive daemon restarts
"""

from animus.discovery.mcp_scanner import MCPScanner, MCPToolSpec
from animus.discovery.openapi_discovery import OpenAPIDiscovery, OpenAPIEndpoint
from animus.discovery.orchestrator import DiscoveryConfig, DiscoveryOrchestrator
from animus.discovery.script_discovery import ScriptDiscovery, ScriptSpec
from animus.discovery.validator import SchemaValidator, ValidationResult

__all__ = [
    "MCPScanner",
    "MCPToolSpec",
    "OpenAPIDiscovery",
    "OpenAPIEndpoint",
    "ScriptDiscovery",
    "ScriptSpec",
    "SchemaValidator",
    "ValidationResult",
    "DiscoveryOrchestrator",
    "DiscoveryConfig",
]
