"""DAG analysis and stage derivation for parallel agent execution."""

from animus.forge.models import AgentConfig
from animus.logging import get_logger
from animus.swarm.models import CyclicDependencyError, SwarmStage

logger = get_logger("swarm.graph")


def build_dag(agents: list[AgentConfig]) -> dict[str, set[str]]:
    """Build a dependency graph from agent input references.

    Parses each agent's ``inputs`` list (format: ``"agent.output"``) to
    extract dependency agent names.

    Args:
        agents: List of agent configurations.

    Returns:
        Dict mapping agent_name -> set of agent names it depends on.
        An agent with no inputs has an empty dependency set.
    """
    dag: dict[str, set[str]] = {}
    for agent in agents:
        deps: set[str] = set()
        for inp in agent.inputs:
            if "." in inp:
                ref_agent = inp.split(".")[0]
                if ref_agent != agent.name:
                    deps.add(ref_agent)
        dag[agent.name] = deps
    return dag


def derive_stages(
    agents: list[AgentConfig],
    dag: dict[str, set[str]],
) -> list[SwarmStage]:
    """Derive parallel execution stages via Kahn's topological sort.

    Agents with no unsatisfied dependencies go into the earliest possible
    stage.  Within each stage, agents are sorted by name for determinism.

    Args:
        agents: List of agent configurations.
        dag: Dependency graph from :func:`build_dag`.

    Returns:
        Ordered list of :class:`SwarmStage`.

    Raises:
        CyclicDependencyError: If the dependency graph contains a cycle.
    """
    # Build in-degree map
    in_degree: dict[str, int] = {name: len(deps) for name, deps in dag.items()}

    # Build reverse graph (name -> set of agents that depend on name)
    reverse: dict[str, set[str]] = {name: set() for name in dag}
    for name, deps in dag.items():
        for dep in deps:
            if dep in reverse:
                reverse[dep].add(name)

    # Seed with zero-in-degree agents
    queue = sorted(name for name, deg in in_degree.items() if deg == 0)
    stages: list[SwarmStage] = []
    placed = 0

    while queue:
        stage = SwarmStage(index=len(stages), agent_names=list(queue))
        stages.append(stage)
        next_queue: list[str] = []

        for name in queue:
            placed += 1
            for dependent in reverse.get(name, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    next_queue.append(dependent)

        queue = sorted(next_queue)

    if placed != len(dag):
        unplaced = {n for n, d in in_degree.items() if d > 0}
        raise CyclicDependencyError(f"Cycle detected among: {unplaced}")

    logger.debug(
        f"Derived {len(stages)} stages from {len(agents)} agents: "
        + ", ".join(f"[{', '.join(s.agent_names)}]" for s in stages)
    )
    return stages


def validate_dag(
    agents: list[AgentConfig],
    dag: dict[str, set[str]],
) -> list[str]:
    """Validate the DAG and return any warnings.

    Checks for self-loops, orphan references, and terminal nodes.

    Returns:
        List of warning messages (empty if clean).
    """
    warnings: list[str] = []
    agent_names = {a.name for a in agents}

    for name, deps in dag.items():
        if name in deps:
            warnings.append(f"Agent {name!r} has a self-loop dependency")
        for dep in deps:
            if dep not in agent_names:
                warnings.append(f"Agent {name!r} depends on undefined agent {dep!r}")

    # Terminal nodes (agents that no one depends on)
    depended_on = set()
    for deps in dag.values():
        depended_on.update(deps)
    terminals = agent_names - depended_on
    if terminals and len(agents) > 1:
        for t in sorted(terminals):
            if dag.get(t):  # Only warn about terminals that have deps (leaf nodes)
                warnings.append(f"Agent {t!r} is a terminal node (no dependents)")

    return warnings
