"""YAML workflow loader with validation."""

from pathlib import Path

import yaml

from animus.forge.models import AgentConfig, ForgeError, GateConfig, WorkflowConfig
from animus.logging import get_logger

logger = get_logger("forge.loader")


def load_workflow(path: Path) -> WorkflowConfig:
    """Load a workflow configuration from a YAML file.

    Args:
        path: Path to the YAML workflow file.

    Returns:
        Validated WorkflowConfig.

    Raises:
        ForgeError: If the file is missing or the config is invalid.
    """
    if not path.exists():
        raise ForgeError(f"Workflow file not found: {path}")
    text = path.read_text()
    return load_workflow_str(text)


def load_workflow_str(yaml_str: str) -> WorkflowConfig:
    """Load a workflow configuration from a YAML string.

    Args:
        yaml_str: YAML content.

    Returns:
        Validated WorkflowConfig.

    Raises:
        ForgeError: If the config is invalid.
    """
    try:
        data = yaml.safe_load(yaml_str)
    except yaml.YAMLError as exc:
        raise ForgeError(f"Invalid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise ForgeError("Workflow YAML must be a mapping")

    name = data.get("name")
    if not name:
        raise ForgeError("Workflow must have a 'name' field")

    raw_agents = data.get("agents", [])
    if not raw_agents:
        raise ForgeError("Workflow must have at least one agent")

    agents = _parse_agents(raw_agents)
    agent_names = {a.name for a in agents}

    gates = _parse_gates(data.get("gates", []), agent_names)

    return WorkflowConfig(
        name=name,
        description=data.get("description", ""),
        agents=agents,
        gates=gates,
        max_cost_usd=float(data.get("max_cost_usd", 1.0)),
        provider=data.get("provider", "ollama"),
        model=data.get("model", "llama3:8b"),
    )


def _parse_agents(raw: list[dict]) -> list[AgentConfig]:
    """Parse and validate agent configs."""
    agents: list[AgentConfig] = []
    seen_names: set[str] = set()
    defined_outputs: set[str] = set()

    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ForgeError(f"Agent at index {i} must be a mapping")

        name = entry.get("name")
        if not name:
            raise ForgeError(f"Agent at index {i} must have a 'name'")
        if name in seen_names:
            raise ForgeError(f"Duplicate agent name: {name!r}")
        seen_names.add(name)

        archetype = entry.get("archetype")
        if not archetype:
            raise ForgeError(f"Agent {name!r} must have an 'archetype'")

        # Validate inputs reference previously defined agent outputs
        inputs = entry.get("inputs", [])
        for inp in inputs:
            if "." not in inp:
                raise ForgeError(f"Agent {name!r} input {inp!r} must use 'agent.output' format")
            ref_agent = inp.split(".")[0]
            if ref_agent not in seen_names:
                raise ForgeError(
                    f"Agent {name!r} input {inp!r} references undefined agent {ref_agent!r}"
                )

        outputs = entry.get("outputs", [])
        for out in outputs:
            defined_outputs.add(f"{name}.{out}")

        agents.append(
            AgentConfig(
                name=name,
                archetype=archetype,
                budget_tokens=int(entry.get("budget_tokens", 10_000)),
                inputs=inputs,
                outputs=outputs,
                provider=entry.get("provider"),
                model=entry.get("model"),
                system_prompt=entry.get("system_prompt"),
                tools=entry.get("tools", []),
            )
        )

    return agents


def _parse_gates(raw: list[dict], agent_names: set[str]) -> list[GateConfig]:
    """Parse and validate gate configs."""
    gates: list[GateConfig] = []

    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ForgeError(f"Gate at index {i} must be a mapping")

        name = entry.get("name")
        if not name:
            raise ForgeError(f"Gate at index {i} must have a 'name'")

        after = entry.get("after")
        if not after:
            raise ForgeError(f"Gate {name!r} must have an 'after' field")
        if after not in agent_names:
            raise ForgeError(f"Gate {name!r} references undefined agent {after!r}")

        on_fail = entry.get("on_fail", "halt")
        revise_target = entry.get("revise_target")
        if on_fail == "revise" and not revise_target:
            raise ForgeError(f"Gate {name!r} has on_fail='revise' but no 'revise_target'")
        if revise_target and revise_target not in agent_names:
            raise ForgeError(
                f"Gate {name!r} revise_target {revise_target!r} is not a defined agent"
            )

        gates.append(
            GateConfig(
                name=name,
                after=after,
                type=entry.get("type", "automated"),
                pass_condition=entry.get("pass_condition", ""),
                on_fail=on_fail,
                revise_target=revise_target,
            )
        )

    return gates
