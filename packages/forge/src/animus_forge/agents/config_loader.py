"""Load agent configurations from YAML files.

Allows agent roles, tools, models, and limits to be configured via
an ``agents.yaml`` file instead of hardcoded DEFAULT_AGENT_CONFIGS.
Hot-reloadable: call ``load_agent_configs()`` again to pick up changes.

Example agents.yaml::

    agents:
      builder:
        enable_shell: true
        max_tool_iterations: 12
        timeout_seconds: 600
        model: deepseek-coder-v2
      tester:
        enable_shell: true
        allowed_tools:
          - read_file
          - list_files
          - search_code
          - run_command
        max_tool_iterations: 10
      reviewer:
        enable_shell: false
        denied_tools:
          - write_file
          - edit_file
          - run_command
      planner:
        allowed_tools: []
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from animus_forge.agents.agent_config import DEFAULT_AGENT_CONFIGS, AgentConfig

logger = logging.getLogger(__name__)

# Default search paths for agents.yaml
_DEFAULT_PATHS = [
    Path("agents.yaml"),
    Path("config/agents.yaml"),
    Path(".gorgon/agents.yaml"),
]


def load_agent_configs(
    path: Path | None = None,
    base_dir: Path | None = None,
) -> dict[str, AgentConfig]:
    """Load agent configs from a YAML file, merged with defaults.

    Args:
        path: Explicit path to agents.yaml. If None, searches default paths.
        base_dir: Base directory for relative path resolution.

    Returns:
        Dict mapping role names to AgentConfig instances.
        Falls back to DEFAULT_AGENT_CONFIGS if no YAML found.
    """
    yaml_path = _find_config(path, base_dir)
    if yaml_path is None:
        return dict(DEFAULT_AGENT_CONFIGS)

    try:
        import yaml
    except ImportError:
        logger.debug("PyYAML not installed, using default agent configs")
        return dict(DEFAULT_AGENT_CONFIGS)

    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load %s: %s", yaml_path, e)
        return dict(DEFAULT_AGENT_CONFIGS)

    if not isinstance(raw, dict):
        logger.warning("Invalid agents.yaml format (expected dict), using defaults")
        return dict(DEFAULT_AGENT_CONFIGS)

    agents_section = raw.get("agents", raw)
    if not isinstance(agents_section, dict):
        logger.warning("Invalid 'agents' section format, using defaults")
        return dict(DEFAULT_AGENT_CONFIGS)

    result = dict(DEFAULT_AGENT_CONFIGS)

    for role, overrides in agents_section.items():
        if not isinstance(overrides, dict):
            continue
        base = DEFAULT_AGENT_CONFIGS.get(role)
        result[role] = _merge_config(role, overrides, base)

    logger.info("Loaded agent configs from %s (%d agents)", yaml_path, len(result))
    return result


def _find_config(
    path: Path | None,
    base_dir: Path | None,
) -> Path | None:
    """Find the agents.yaml config file."""
    if path is not None:
        resolved = Path(path)
        if base_dir and not resolved.is_absolute():
            resolved = base_dir / resolved
        return resolved if resolved.exists() else None

    search_base = base_dir or Path.cwd()
    for candidate in _DEFAULT_PATHS:
        full = search_base / candidate
        if full.exists():
            return full

    return None


def _merge_config(
    role: str,
    overrides: dict[str, Any],
    base: AgentConfig | None,
) -> AgentConfig:
    """Merge YAML overrides onto a base AgentConfig."""
    if base is None:
        base = AgentConfig(role=role)

    workspace = overrides.get("workspace")
    workspace_path = Path(workspace) if workspace else base.workspace

    return AgentConfig(
        role=role,
        workspace=workspace_path,
        allowed_tools=overrides.get("allowed_tools", base.allowed_tools),
        denied_tools=overrides.get("denied_tools", base.denied_tools),
        model=overrides.get("model", base.model),
        provider_type=overrides.get("provider_type", base.provider_type),
        max_tool_iterations=overrides.get("max_tool_iterations", base.max_tool_iterations),
        timeout_seconds=overrides.get("timeout_seconds", base.timeout_seconds),
        max_output_chars=overrides.get("max_output_chars", base.max_output_chars),
        enable_shell=overrides.get("enable_shell", base.enable_shell),
        budget_pool=overrides.get("budget_pool", base.budget_pool),
        metadata=overrides.get("metadata", base.metadata),
    )
