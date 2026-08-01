"""DiscoveryOrchestrator: ties all discovery scanners into a unified pipeline.

Runs as a background task via the daemon scheduler. Manages the full lifecycle:
scan → validate → deduplicate → register → persist.

Integration points:
- Daemon TaskScheduler: recurring discovery runs
- ToolRegistry: registration of validated tools
- SchemaValidator: quality gates before registration
- Session Steward: reports on discovery efficiency
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from animus.logging import get_logger
from animus.tools import Tool, ToolRegistry

from .mcp_scanner import MCPScanner
from .openapi_discovery import OpenAPIDiscovery
from .script_discovery import ScriptDiscovery
from .validator import SchemaValidator, ValidationResult

logger = get_logger("discovery.orchestrator")


@dataclass
class DiscoveryConfig:
    """Configuration for the discovery pipeline."""

    # Scanning
    mcp_servers: list[str] = field(default_factory=list)  # URLs to scan
    mcp_stdio_commands: list[tuple[list[str], str]] = field(default_factory=list)
    openapi_dirs: list[str] = field(default_factory=list)
    openapi_urls: list[str] = field(default_factory=list)
    script_dirs: list[str] = field(default_factory=list)
    scan_localhost_ports: list[int] = field(default_factory=lambda: [3000, 8080, 9000])

    # Validation
    min_validation_score: float = 0.6
    max_new_tools_per_run: int = 20

    # Persistence
    persistence_dir: str = "~/.animus/discovery"
    deduplicate_by_hash: bool = True

    # Scheduling
    run_interval_seconds: int = 3600  # Hourly
    enabled: bool = True

    @property
    def has_sources(self) -> bool:
        """Check if any discovery sources are configured."""
        return bool(
            self.mcp_servers
            or self.mcp_stdio_commands
            or self.openapi_dirs
            or self.openapi_urls
            or self.script_dirs
            or self.scan_localhost_ports
        )


@dataclass
class DiscoveryRun:
    """Result of a single discovery run."""

    run_id: str
    started_at: datetime
    completed_at: datetime | None = None
    sources_scanned: int = 0
    tools_discovered: int = 0
    tools_validated: int = 0
    tools_registered: int = 0
    tools_failed: int = 0
    errors: list[str] = field(default_factory=list)
    validation_results: list[ValidationResult] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        if self.completed_at is None:
            return 0.0
        return (self.completed_at - self.started_at).total_seconds()


class DiscoveryOrchestrator:
    """Main orchestrator for tool auto-discovery.

    Usage:
        config = DiscoveryConfig(
            mcp_servers=["http://localhost:3000/sse"],
            script_dirs=["~/scripts"],
        )
        orchestrator = DiscoveryOrchestrator(config, registry)
        run = orchestrator.run_discovery()

    Daemon integration:
        task = orchestrator.create_daemon_task(daemon_scheduler)
    """

    def __init__(
        self,
        config: DiscoveryConfig | None = None,
        registry: ToolRegistry | None = None,
    ):
        self.config = config or DiscoveryConfig()
        self.registry = registry or ToolRegistry()
        self.validator = SchemaValidator(min_score=self.config.min_validation_score)
        self.persistence_dir = Path(self.config.persistence_dir).expanduser()
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        self._discovered_hashes: set[str] = set()
        self._load_existing_hashes()

        # Scanners
        self.mcp_scanner = MCPScanner()
        self.openapi_discovery = OpenAPIDiscovery()
        self.script_discovery = ScriptDiscovery()

    # ── Persistence ───────────────────────────────────────────────────

    def _hash_tool(self, name: str, description: str, parameters: dict) -> str:
        """Generate a hash for deduplication."""
        content = f"{name}:{description}:{json.dumps(parameters, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _load_existing_hashes(self) -> None:
        """Load hashes of already-discovered tools."""
        hash_file = self.persistence_dir / "discovered_hashes.json"
        if hash_file.exists():
            try:
                data = json.loads(hash_file.read_text())
                self._discovered_hashes = set(data.get("hashes", []))
            except Exception:
                pass

    def _save_hashes(self) -> None:
        """Save discovered tool hashes."""
        hash_file = self.persistence_dir / "discovered_hashes.json"
        hash_file.write_text(json.dumps({"hashes": list(self._discovered_hashes)}))

    def _persist_run(self, run: DiscoveryRun) -> None:
        """Append run result to history."""
        history_file = self.persistence_dir / "discovery_history.jsonl"
        entry = {
            "run_id": run.run_id,
            "started_at": run.started_at.isoformat(),
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "sources_scanned": run.sources_scanned,
            "tools_discovered": run.tools_discovered,
            "tools_validated": run.tools_validated,
            "tools_registered": run.tools_registered,
            "tools_failed": run.tools_failed,
            "duration_seconds": run.duration_seconds,
            "errors": run.errors,
        }
        with history_file.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    # ── Discovery Pipeline ────────────────────────────────────────────

    def run_discovery(self) -> DiscoveryRun:
        """Execute a full discovery run.

        Returns:
            DiscoveryRun with results.
        """
        run = DiscoveryRun(
            run_id=f"discovery-{int(time.time())}",
            started_at=datetime.now(),
        )

        if not self.config.enabled:
            logger.info("Discovery is disabled in config")
            run.completed_at = datetime.now()
            return run

        if not self.config.has_sources:
            logger.info("No discovery sources configured")
            run.completed_at = datetime.now()
            return run

        logger.info(f"Starting discovery run: {run.run_id}")
        all_raw_tools: list[dict[str, Any]] = []

        # 1. MCP servers
        for url in self.config.mcp_servers:
            try:
                specs = self.mcp_scanner.scan_server(url)
                for spec in specs:
                    all_raw_tools.append(
                        {
                            "name": spec.name,
                            "description": spec.description,
                            "parameters": spec.to_animus_schema(),
                            "source": f"mcp:{spec.server_name}",
                        }
                    )
                run.sources_scanned += 1
            except Exception as e:
                run.errors.append(f"MCP scan failed for {url}: {e}")

        # MCP stdio servers
        for command, name in self.config.mcp_stdio_commands:
            try:
                specs = self.mcp_scanner.scan_stdio_server(command, name)
                for spec in specs:
                    all_raw_tools.append(
                        {
                            "name": spec.name,
                            "description": spec.description,
                            "parameters": spec.to_animus_schema(),
                            "source": f"mcp-stdio:{name}",
                        }
                    )
                run.sources_scanned += 1
            except Exception as e:
                run.errors.append(f"MCP stdio scan failed for {name}: {e}")

        # Localhost MCP sweep
        if self.config.scan_localhost_ports:
            specs = self.mcp_scanner.scan_local_servers(self.config.scan_localhost_ports)
            for spec in specs:
                all_raw_tools.append(
                    {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.to_animus_schema(),
                        "source": f"mcp-local:{spec.server_name}",
                    }
                )
            run.sources_scanned += 1

        # 2. OpenAPI specs
        for url in self.config.openapi_urls:
            try:
                endpoints = self.openapi_discovery.load_from_url(url)
                for ep in endpoints:
                    all_raw_tools.append(
                        {
                            "name": ep.tool_name,
                            "description": ep.description or ep.summary,
                            "parameters": ep.parameters,
                            "source": f"openapi:{url}",
                        }
                    )
                run.sources_scanned += 1
            except Exception as e:
                run.errors.append(f"OpenAPI load failed for {url}: {e}")

        for directory in self.config.openapi_dirs:
            try:
                endpoints = self.openapi_discovery.scan_directory(directory)
                for ep in endpoints:
                    all_raw_tools.append(
                        {
                            "name": ep.tool_name,
                            "description": ep.description or ep.summary,
                            "parameters": ep.parameters,
                            "source": f"openapi-dir:{directory}",
                        }
                    )
                run.sources_scanned += 1
            except Exception as e:
                run.errors.append(f"OpenAPI scan failed for {directory}: {e}")

        # 3. Local scripts
        for directory in self.config.script_dirs:
            try:
                specs = self.script_discovery.scan_directory(directory)
                for spec in specs:
                    all_raw_tools.append(
                        {
                            "name": spec.name,
                            "description": spec.description,
                            "parameters": spec.parameters,
                            "source": f"script:{spec.script_path}",
                        }
                    )
                run.sources_scanned += 1
            except Exception as e:
                run.errors.append(f"Script scan failed for {directory}: {e}")

        run.tools_discovered = len(all_raw_tools)
        logger.info(f"Discovery scan complete: {run.tools_discovered} raw tools")

        # 4. Validate and register
        registered = 0
        failed = 0

        for raw_tool in all_raw_tools[: self.config.max_new_tools_per_run]:
            # Deduplication
            if self.config.deduplicate_by_hash:
                tool_hash = self._hash_tool(
                    raw_tool["name"],
                    raw_tool["description"],
                    raw_tool["parameters"],
                )
                if tool_hash in self._discovered_hashes:
                    continue
                self._discovered_hashes.add(tool_hash)

            # Validate
            result = self.validator.validate_tool_schema(raw_tool)
            run.validation_results.append(result)

            if result.passed:
                run.tools_validated += 1
                # Register with placeholder handler
                tool = Tool(
                    name=raw_tool["name"],
                    description=raw_tool["description"],
                    parameters=raw_tool["parameters"],
                    handler=self._make_placeholder_handler(raw_tool["name"]),
                    category="discovered",
                )
                self.registry.register(tool)
                registered += 1
                logger.info(f"Registered discovered tool: {tool.name}")
            else:
                failed += 1
                logger.warning(
                    f"Tool validation failed: {result.tool_name} "
                    f"(score: {result.score}, errors: {result.errors})"
                )

        run.tools_registered = registered
        run.tools_failed = failed
        run.completed_at = datetime.now()

        # Persist
        self._save_hashes()
        self._persist_run(run)

        logger.info(
            f"Discovery run complete: {registered} registered, "
            f"{failed} failed, {run.duration_seconds:.1f}s"
        )
        return run

    def _make_placeholder_handler(self, tool_name: str):
        """Create a placeholder handler for discovered tools.

        Discovered tools need a real handler to be executable. This placeholder
        logs a warning and returns an error indicating the tool needs wiring.
        """

        def handler(params: dict) -> Any:
            from animus.tools import ToolResult

            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Tool '{tool_name}' is discovered but not yet wired to a real implementation. "
                f"Register a proper handler via registry.get('{tool_name}').handler = your_function",
            )

        return handler

    # ── Daemon Integration ──────────────────────────────────────────

    def create_daemon_task(self, daemon_scheduler: Any) -> Any:
        """Register a recurring discovery task with the daemon scheduler.

        Args:
            daemon_scheduler: A TaskScheduler instance.

        Returns:
            The scheduled task object.
        """

        def discovery_callback() -> None:
            logger.info("Running scheduled tool discovery")
            self.run_discovery()

        task = daemon_scheduler.schedule_interval(
            description="Auto-discover tools from MCP servers, OpenAPI specs, and local scripts",
            seconds=self.config.run_interval_seconds,
            priority="normal",
            metadata={"citizen": "discovery", "callback": discovery_callback},
        )
        return task

    # ── Status ────────────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Get current discovery status."""
        return {
            "config_sources": {
                "mcp_servers": len(self.config.mcp_servers),
                "openapi_dirs": len(self.config.openapi_dirs),
                "script_dirs": len(self.config.script_dirs),
            },
            "discovered_hashes": len(self._discovered_hashes),
            "registry_size": len(self.registry.list_tools()),
            "enabled": self.config.enabled,
        }

    def get_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent discovery run history."""
        history_file = self.persistence_dir / "discovery_history.jsonl"
        if not history_file.exists():
            return []

        lines = history_file.read_text().strip().splitlines()
        results = []
        for line in lines[-limit:]:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return results
