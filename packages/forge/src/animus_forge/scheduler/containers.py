"""ContainerManager — sandbox citizen execution via Docker or Podman.

Provides an alternative to ``ProcessPoolExecutor`` for stronger isolation.
Mounts the workspace into a throwaway container, runs the citizen, and
returns the structured output.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

from animus_types.secrets import mask_env_command_args, redact, redact_exception

logger = logging.getLogger(__name__)


@dataclass
class ContainerConfig:
    """Tuning for container-based execution."""

    image: str = "python:3.12-slim"
    runtime: str = "auto"  # "docker", "podman", or "auto"
    workspace_mount: str | None = None  # host path; defaults to cwd
    extra_volumes: list[str] = None  # ["/host:/container", ...]
    env: dict[str, str] = None
    timeout_seconds: int = 300
    remove: bool = True
    network: str = "none"  # "none", "host", or bridge name

    def __post_init__(self):
        if self.extra_volumes is None:
            self.extra_volumes = []
        if self.env is None:
            self.env = {}


class ContainerManager:
    """Run citizen tasks inside ephemeral containers.

    Detects the container runtime (docker → podman → None) and builds the
    appropriate CLI invocation.  If no runtime is available, ``is_available``
    returns ``False`` and the caller should fall back to process isolation.
    """

    def __init__(self, config: ContainerConfig | None = None):
        self.config = config or ContainerConfig()
        self._runtime_cmd = self._detect_runtime()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return self._runtime_cmd is not None

    def run_task(
        self,
        task_id: str,
        mission_id: str,
        citizen_role: str,
        description: str,
        context_json: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a citizen task inside a container.

        Returns:
            CitizenOutput-shaped dict.
        """
        if not self._runtime_cmd:
            return {
                "status": "failed",
                "summary": "No container runtime available",
                "changed_files": [],
                "evidence": [],
                "risks": [{"severity": "critical", "description": "docker/podman not found"}],
                "confidence": 0.0,
            }

        # Write task payload to a temp file that will be mounted
        payload = {
            "task_id": task_id,
            "mission_id": mission_id,
            "citizen_role": citizen_role,
            "description": description,
            "context": context_json,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(payload, f)
            payload_path = f.name

        try:
            return self._run_container(payload_path)
        finally:
            try:
                os.unlink(payload_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _detect_runtime(self) -> str | None:
        """Return ``'docker'`` or ``'podman'`` if available, else ``None``."""
        for cmd in ["docker", "podman"]:
            if shutil.which(cmd):
                logger.info("Container runtime detected: %s", cmd)
                return cmd
        logger.warning("No container runtime found (docker or podman)")
        return None

    def _build_command(self, payload_path: str) -> list[str]:
        """Build the container run command."""
        cmd = [
            self._runtime_cmd,
            "run",
            "--rm" if self.config.remove else "",
            "--network", self.config.network,
            "-v", f"{payload_path}:/tmp/task_payload.json:ro",
        ]

        # Workspace mount
        ws = self.config.workspace_mount or os.getcwd()
        cmd += ["-v", f"{ws}:/workspace"]

        # Extra volumes
        for vol in self.config.extra_volumes:
            cmd += ["-v", vol]

        # Environment
        for k, v in self.config.env.items():
            cmd += ["-e", f"{k}={v}"]

        # Image and inline runner
        cmd += [
            self.config.image,
            "python",
            "-c",
            self._INLINE_RUNNER,
        ]

        # Filter empty strings
        return [c for c in cmd if c]

    _INLINE_RUNNER = '''
import json, sys, os, uuid
sys.path.insert(0, "/workspace/src")

from animus_forge.citizens.base import Citizen
from animus_forge.citizens.builder import BuilderCitizen
from animus_forge.citizens.planner import PlannerCitizen
from animus_forge.citizens.reviewer import ReviewerCitizen
from animus_forge.missions.domain import Task, TaskContext

_REGISTRY = {
    "planner": PlannerCitizen,
    "builder": BuilderCitizen,
    "reviewer": ReviewerCitizen,
}

with open("/tmp/task_payload.json") as f:
    payload = json.load(f)

task = Task(
    task_id=uuid.UUID(payload["task_id"]),
    mission_id=uuid.UUID(payload["mission_id"]),
    citizen_role=payload["citizen_role"],
    description=payload["description"],
)
ctx = TaskContext(**payload["context"])

cls = _REGISTRY.get(payload["citizen_role"])
if cls is None:
    print(json.dumps({
        "status": "failed",
        "summary": f"Unknown role: {payload['citizen_role']}",
        "confidence": 0.0,
    }))
    sys.exit(0)

try:
    output = cls().run(task=task, context=ctx)
    print(json.dumps(output.model_dump(mode="json")))
except Exception as exc:
    print(json.dumps({
        "status": "failed",
        "summary": str(exc),
        "confidence": 0.0,
    }))
'''

    def _run_container(self, payload_path: str) -> dict[str, Any]:
        cmd = self._build_command(payload_path)
        # SEC-06: log the command with environment values masked; the real ``cmd``
        # is still passed to subprocess unchanged.
        safe_cmd = mask_env_command_args(cmd)
        logger.info("Container task: %s", " ".join(safe_cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "summary": f"Container timeout after {self.config.timeout_seconds}s",
                "changed_files": [],
                "evidence": [],
                "risks": [{"severity": "critical", "description": "Container timeout"}],
                "confidence": 0.0,
            }

        if result.returncode != 0:
            safe_stderr = redact(result.stderr)
            logger.error("Container stderr: %s", safe_stderr)
            return {
                "status": "failed",
                "summary": redact(
                    f"Container exit {result.returncode}: {result.stderr[:200]}"
                ),
                "changed_files": [],
                "evidence": [],
                "risks": [{"severity": "critical", "description": redact(result.stderr[:500])}],
                "confidence": 0.0,
            }

        # Last non-empty line of stdout should be JSON
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return {
                "status": "failed",
                "summary": "Empty container output",
                "changed_files": [],
                "evidence": [],
                "risks": [{"severity": "critical", "description": "No JSON output from container"}],
                "confidence": 0.0,
            }

        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            return {
                "status": "failed",
                "summary": f"Invalid JSON from container: {redact_exception(exc)}",
                "changed_files": [],
                "evidence": [{"type": "raw_output", "detail": redact(result.stdout[:500])}],
                "risks": [{"severity": "critical", "description": redact_exception(exc)}],
                "confidence": 0.0,
            }
