"""OpenAPIDiscovery: converts OpenAPI specs into Animus Tool schemas.

Reads OpenAPI 3.0/3.1 JSON/YAML specs and generates Tool definitions
with JSON Schema parameters. Supports local files and remote URLs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from animus.logging import get_logger

logger = get_logger("discovery.openapi")


@dataclass
class OpenAPIEndpoint:
    """A single OpenAPI endpoint converted to tool form."""

    method: str  # GET, POST, PUT, DELETE
    path: str
    operation_id: str
    summary: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for parameters
    base_url: str = ""

    @property
    def tool_name(self) -> str:
        """Generate a clean tool name from operationId or path."""
        if self.operation_id:
            return self.operation_id.replace("_", "-")
        return f"{self.method.lower()}-{self.path.replace('/', '-').strip('-')}"


class OpenAPIDiscovery:
    """Discovers tools from OpenAPI specifications.

    Usage:
        discovery = OpenAPIDiscovery()
        endpoints = discovery.load_from_file("api-spec.yaml")
        for ep in endpoints:
            registry.register(Tool(name=ep.tool_name, ...))
    """

    def __init__(self):
        self._endpoints: list[OpenAPIEndpoint] = []

    def load_from_file(self, path: str | Path) -> list[OpenAPIEndpoint]:
        """Load and parse a local OpenAPI spec file.

        Args:
            path: Path to .json or .yaml/.yml OpenAPI spec.

        Returns:
            List of discovered endpoints.
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"OpenAPI spec not found: {path}")
            return []

        try:
            if path.suffix in (".yaml", ".yml"):
                import yaml
                spec = yaml.safe_load(path.read_text())
            else:
                spec = json.loads(path.read_text())
        except Exception as e:
            logger.error(f"Failed to parse OpenAPI spec {path}: {e}")
            return []

        return self._parse_spec(spec, base_url=str(path))

    def load_from_url(self, url: str) -> list[OpenAPIEndpoint]:
        """Fetch and parse a remote OpenAPI spec.

        Args:
            url: URL to OpenAPI JSON or YAML spec.

        Returns:
            List of discovered endpoints.
        """
        try:
            import requests
        except ImportError:
            logger.warning("requests not installed; cannot fetch remote OpenAPI specs")
            return []

        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            if url.endswith((".yaml", ".yml")):
                import yaml
                spec = yaml.safe_load(resp.text)
            else:
                spec = resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch OpenAPI spec from {url}: {e}")
            return []

        return self._parse_spec(spec, base_url=url)

    def _parse_spec(self, spec: dict[str, Any], base_url: str) -> list[OpenAPIEndpoint]:
        """Parse an OpenAPI spec dict into endpoints."""
        endpoints: list[OpenAPIEndpoint] = []
        servers = spec.get("servers", [{}])
        base = servers[0].get("url", base_url) if servers else base_url

        paths = spec.get("paths", {})
        for path, methods in paths.items():
            for method, details in methods.items():
                if method in ("parameters", "summary", "description"):
                    continue
                if not isinstance(details, dict):
                    continue

                op_id = details.get("operationId", "")
                summary = details.get("summary", "")
                desc = details.get("description", summary)

                # Build parameter schema
                params: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

                # Path/query parameters
                for param in details.get("parameters", []):
                    if isinstance(param, dict):
                        pname = param.get("name", "")
                        if pname:
                            params["properties"][pname] = self._convert_schema(
                                param.get("schema", {"type": "string"})
                            )
                            if param.get("required", False):
                                params["required"].append(pname)

                # Request body parameters
                body = details.get("requestBody", {})
                if body:
                    content = body.get("content", {})
                    json_content = content.get("application/json", {})
                    body_schema = json_content.get("schema", {})
                    if body_schema.get("type") == "object":
                        for pname, pschema in body_schema.get("properties", {}).items():
                            params["properties"][pname] = self._convert_schema(pschema)
                        if "required" in body_schema:
                            params["required"].extend(body_schema["required"])

                endpoint = OpenAPIEndpoint(
                    method=method.upper(),
                    path=path,
                    operation_id=op_id,
                    summary=summary,
                    description=desc,
                    parameters=params,
                    base_url=base,
                )
                endpoints.append(endpoint)
                logger.debug(f"Discovered OpenAPI endpoint: {endpoint.tool_name}")

        self._endpoints.extend(endpoints)
        logger.info(f"OpenAPI parse complete: {len(endpoints)} endpoints from {base_url}")
        return endpoints

    def _convert_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAPI schema to JSON Schema compatible with model tools."""
        if not isinstance(schema, dict):
            return {"type": "string"}

        result: dict[str, Any] = {"type": schema.get("type", "string")}
        if "description" in schema:
            result["description"] = schema["description"]
        if "enum" in schema:
            result["enum"] = schema["enum"]
        if "default" in schema:
            result["default"] = schema["default"]

        # Nested objects
        if schema.get("type") == "object" and "properties" in schema:
            result["properties"] = {
                k: self._convert_schema(v) for k, v in schema["properties"].items()
            }

        return result

    def scan_directory(self, directory: str | Path) -> list[OpenAPIEndpoint]:
        """Recursively scan a directory for OpenAPI spec files.

        Args:
            directory: Directory to scan.

        Returns:
            Combined list of all discovered endpoints.
        """
        directory = Path(directory)
        all_endpoints: list[OpenAPIEndpoint] = []

        for ext in ("*.json", "*.yaml", "*.yml"):
            for path in directory.rglob(ext):
                try:
                    endpoints = self.load_from_file(path)
                    all_endpoints.extend(endpoints)
                except Exception as e:
                    logger.warning(f"Failed to parse {path}: {e}")

        logger.info(f"Directory scan complete: {len(all_endpoints)} endpoints from {directory}")
        return all_endpoints
