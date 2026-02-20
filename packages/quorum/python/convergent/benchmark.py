"""
Benchmark suite — measures coordination overhead and convergence quality.

Proves milestone #4: coordination overhead grows sublinearly as agents increase.

Scenarios:
  INDEPENDENT: N agents, no interface overlap. Baseline for pure overhead.
  SHARED_INTERFACE: N agents, all share one interface. Tests convergence speed.
  HIGH_CONTENTION: N agents, all provide the same interface. Stress test.
  REALISTIC: Mixed — some shared, some independent, some constrained.

Metrics collected:
  - conflict_rate: conflicts / total_resolutions
  - rework_rate: ConsumeInstead adjustments / total_provisions
  - convergence_rounds: rounds until all agents stabilize
  - coordination_overhead: (time_with_coordination - time_baseline) / time_baseline
  - adjustments_per_agent: average adjustments each agent made
  - constraint_violations: how many intents failed constraint gates
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from convergent.agent import AgentAction, SimulatedAgent, SimulationRunner
from convergent.intent import (
    Constraint,
    ConstraintSeverity,
    Evidence,
    Intent,
    InterfaceKind,
    InterfaceSpec,
)
from convergent.resolver import IntentResolver

# ---------------------------------------------------------------------------
# Scenario types
# ---------------------------------------------------------------------------


class ScenarioType(str, Enum):
    INDEPENDENT = "independent"
    SHARED_INTERFACE = "shared_interface"
    HIGH_CONTENTION = "high_contention"
    REALISTIC = "realistic"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkMetrics:
    """Collected metrics from a single benchmark run."""

    scenario: str
    agent_count: int
    total_intents: int = 0
    total_resolutions: int = 0
    total_conflicts: int = 0
    total_adjustments: int = 0
    consume_instead_count: int = 0
    adopt_constraint_count: int = 0
    convergence_rounds: int = 0
    all_converged: bool = False
    wall_clock_seconds: float = 0.0
    constraint_violations: int = 0

    # Economic metrics
    total_cost: float = 0.0
    escalations: int = 0
    auto_resolved: int = 0

    @property
    def conflict_rate(self) -> float:
        """Conflicts per resolution. Lower is better."""
        if self.total_resolutions == 0:
            return 0.0
        return self.total_conflicts / self.total_resolutions

    @property
    def rework_rate(self) -> float:
        """ConsumeInstead adjustments as fraction of total intents."""
        if self.total_intents == 0:
            return 0.0
        return self.consume_instead_count / self.total_intents

    @property
    def adjustments_per_agent(self) -> float:
        if self.agent_count == 0:
            return 0.0
        return self.total_adjustments / self.agent_count

    @property
    def cost_per_agent(self) -> float:
        if self.agent_count == 0:
            return 0.0
        return self.total_cost / self.agent_count

    def summary_line(self) -> str:
        return (
            f"{self.scenario:20s} | "
            f"agents={self.agent_count:3d} | "
            f"conflicts={self.conflict_rate:.3f} | "
            f"rework={self.rework_rate:.3f} | "
            f"rounds={self.convergence_rounds:3d} | "
            f"adj/agent={self.adjustments_per_agent:.1f} | "
            f"converged={'Y' if self.all_converged else 'N'} | "
            f"{self.wall_clock_seconds:.3f}s"
        )


@dataclass
class BenchmarkSuite:
    """Results from running multiple benchmark scenarios."""

    results: list[BenchmarkMetrics] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 100,
            "CONVERGENT BENCHMARK RESULTS",
            "=" * 100,
            f"{'Scenario':20s} | {'Agents':>6s} | {'Conflict':>8s} | "
            f"{'Rework':>6s} | {'Rounds':>6s} | {'Adj/Agt':>7s} | "
            f"{'Conv':>4s} | {'Time':>7s}",
            "-" * 100,
        ]
        for m in self.results:
            lines.append(m.summary_line())
        lines.append("=" * 100)

        # Scaling analysis
        if len(self.results) >= 2:
            lines.append("")
            lines.append("SCALING ANALYSIS:")
            by_scenario: dict[str, list[BenchmarkMetrics]] = {}
            for m in self.results:
                by_scenario.setdefault(m.scenario, []).append(m)

            for scenario, metrics in by_scenario.items():
                if len(metrics) < 2:
                    continue
                metrics.sort(key=lambda m: m.agent_count)
                first, last = metrics[0], metrics[-1]
                agent_ratio = last.agent_count / first.agent_count if first.agent_count else 1
                conflict_ratio = (
                    (last.conflict_rate / first.conflict_rate) if first.conflict_rate > 0 else 0
                )
                round_ratio = (
                    (last.convergence_rounds / first.convergence_rounds)
                    if first.convergence_rounds > 0
                    else 0
                )
                lines.append(
                    f"  {scenario}: {first.agent_count}→{last.agent_count} agents "
                    f"({agent_ratio:.1f}x), "
                    f"conflict_rate {conflict_ratio:.2f}x, "
                    f"rounds {round_ratio:.2f}x"
                )
                if agent_ratio > 1 and round_ratio < agent_ratio:
                    lines.append("    → SUBLINEAR scaling (rounds grow slower than agents)")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------


def _build_independent_agents(n: int, resolver: IntentResolver) -> list[SimulatedAgent]:
    """N agents with completely independent scopes."""
    agents = []
    for i in range(n):
        agent = SimulatedAgent(f"agent-{i}", resolver)
        agent.plan(
            [
                AgentAction(
                    intent=Intent(
                        agent_id=f"agent-{i}",
                        intent=f"Module {i}",
                        provides=[
                            InterfaceSpec(
                                name=f"Svc{i}Alpha",
                                kind=InterfaceKind.CLASS,
                                signature="run(x: int) -> str",
                                tags=[f"scope{i}", f"module{i}"],
                            ),
                        ],
                    ),
                    post_evidence=[
                        Evidence.code_committed(f"module{i}.py"),
                        Evidence.test_pass(f"test_module{i}"),
                    ],
                ),
                AgentAction(
                    intent=Intent(
                        agent_id=f"agent-{i}",
                        intent=f"Module {i} complete",
                        provides=[
                            InterfaceSpec(
                                name=f"Svc{i}Beta",
                                kind=InterfaceKind.FUNCTION,
                                signature="helper(y: str) -> bool",
                                tags=[f"scope{i}", f"helper{i}"],
                            ),
                        ],
                    ),
                    post_evidence=[
                        Evidence.test_pass(f"test_helper{i}"),
                    ],
                ),
            ]
        )
        agents.append(agent)
    return agents


def _build_shared_interface_agents(n: int, resolver: IntentResolver) -> list[SimulatedAgent]:
    """N agents that all depend on a shared User interface.
    Agent 0 provides User at high stability; others consume it."""
    agents = []
    for i in range(n):
        if i == 0:
            # Provider agent — high stability
            agent = SimulatedAgent(f"agent-{i}", resolver)
            agent.plan(
                [
                    AgentAction(
                        intent=Intent(
                            agent_id=f"agent-{i}",
                            intent="User model provider",
                            provides=[
                                InterfaceSpec(
                                    name="User",
                                    kind=InterfaceKind.MODEL,
                                    signature="id: UUID, email: str",
                                    tags=["user", "model", "shared"],
                                ),
                            ],
                            constraints=[
                                Constraint(
                                    target="User model",
                                    requirement="must have id: UUID",
                                    affects_tags=["user", "model"],
                                ),
                            ],
                        ),
                        post_evidence=[
                            Evidence.code_committed("models.py"),
                            Evidence.test_pass("test_user"),
                            Evidence.test_pass("test_user_validation"),
                        ],
                    ),
                ]
            )
        else:
            # Consumer agent — requires User, provides own scope
            agent = SimulatedAgent(f"agent-{i}", resolver)
            agent.plan(
                [
                    AgentAction(
                        intent=Intent(
                            agent_id=f"agent-{i}",
                            intent=f"Feature {i} using User",
                            provides=[
                                InterfaceSpec(
                                    name=f"Feature{i}Ctrl",
                                    kind=InterfaceKind.CLASS,
                                    signature="handle(user_id: UUID) -> dict",
                                    tags=[f"feature{i}", "api"],
                                ),
                            ],
                            requires=[
                                InterfaceSpec(
                                    name="User",
                                    kind=InterfaceKind.MODEL,
                                    signature="id: UUID",
                                    tags=["user", "model", "shared"],
                                ),
                            ],
                        ),
                        post_evidence=[
                            Evidence.code_committed(f"feature{i}.py"),
                        ],
                    ),
                ]
            )
        agents.append(agent)
    return agents


def _build_high_contention_agents(n: int, resolver: IntentResolver) -> list[SimulatedAgent]:
    """N agents all trying to provide the same Config interface.
    Only the highest-stability one should win; others yield."""
    agents = []
    for i in range(n):
        evidence = []
        if i == 0:
            # Agent 0 has highest stability
            evidence = [
                Evidence.code_committed("config.py"),
                Evidence.test_pass("test_config"),
                Evidence.test_pass("test_config_validation"),
            ]
        elif i == 1:
            evidence = [Evidence.code_committed("config_alt.py")]

        agent = SimulatedAgent(f"agent-{i}", resolver)
        agent.plan(
            [
                AgentAction(
                    intent=Intent(
                        agent_id=f"agent-{i}",
                        intent=f"Config provider (agent {i})",
                        provides=[
                            InterfaceSpec(
                                name="AppConfig",
                                kind=InterfaceKind.CONFIG,
                                signature="get(key: str) -> str, set(key: str, val: str) -> None",
                                tags=["config", "settings", "app"],
                            ),
                        ],
                    ),
                    post_evidence=evidence,
                ),
            ]
        )
        agents.append(agent)
    return agents


def _build_realistic_agents(n: int, resolver: IntentResolver) -> list[SimulatedAgent]:
    """Mixed scenario: some shared interfaces, some independent, constraints."""
    agents = []

    # Agent 0: Auth provider (high stability)
    agent0 = SimulatedAgent("agent-0", resolver)
    agent0.plan(
        [
            AgentAction(
                intent=Intent(
                    agent_id="agent-0",
                    intent="Auth module",
                    provides=[
                        InterfaceSpec(
                            name="User",
                            kind=InterfaceKind.MODEL,
                            signature="id: UUID, email: str",
                            tags=["user", "model", "auth"],
                        ),
                        InterfaceSpec(
                            name="AuthSvc",
                            kind=InterfaceKind.CLASS,
                            signature="login(email: str, pw: str) -> Token",
                            tags=["auth", "login"],
                        ),
                    ],
                    constraints=[
                        Constraint(
                            target="User",
                            requirement="id must be UUID",
                            severity=ConstraintSeverity.REQUIRED,
                            affects_tags=["user", "model"],
                        ),
                    ],
                ),
                post_evidence=[
                    Evidence.code_committed("auth.py"),
                    Evidence.test_pass("test_auth"),
                    Evidence.test_pass("test_login"),
                ],
            ),
        ]
    )
    agents.append(agent0)

    # Remaining agents: mix of consumers and independents
    for i in range(1, n):
        agent = SimulatedAgent(f"agent-{i}", resolver)
        if i % 3 == 0:
            # Contender: also provides User (should yield to agent-0)
            agent.plan(
                [
                    AgentAction(
                        intent=Intent(
                            agent_id=f"agent-{i}",
                            intent=f"Feature {i} with User",
                            provides=[
                                InterfaceSpec(
                                    name="User",
                                    kind=InterfaceKind.MODEL,
                                    signature="id: UUID, name: str",
                                    tags=["user", "model", f"feat{i}"],
                                ),
                            ],
                        ),
                    ),
                ]
            )
        elif i % 3 == 1:
            # Consumer: requires User
            agent.plan(
                [
                    AgentAction(
                        intent=Intent(
                            agent_id=f"agent-{i}",
                            intent=f"Feature {i} consumer",
                            provides=[
                                InterfaceSpec(
                                    name=f"Feat{i}Ctrl",
                                    kind=InterfaceKind.CLASS,
                                    signature="run(uid: UUID) -> dict",
                                    tags=[f"feat{i}", "api"],
                                ),
                            ],
                            requires=[
                                InterfaceSpec(
                                    name="User",
                                    kind=InterfaceKind.MODEL,
                                    signature="id: UUID",
                                    tags=["user", "model"],
                                ),
                            ],
                        ),
                        post_evidence=[Evidence.code_committed(f"feat{i}.py")],
                    ),
                ]
            )
        else:
            # Independent
            agent.plan(
                [
                    AgentAction(
                        intent=Intent(
                            agent_id=f"agent-{i}",
                            intent=f"Independent module {i}",
                            provides=[
                                InterfaceSpec(
                                    name=f"Indep{i}Svc",
                                    kind=InterfaceKind.CLASS,
                                    signature="process() -> bool",
                                    tags=[f"indep{i}", f"isolated{i}"],
                                ),
                            ],
                        ),
                        post_evidence=[Evidence.test_pass(f"test_indep{i}")],
                    ),
                ]
            )
        agents.append(agent)

    return agents


# ---------------------------------------------------------------------------
# Scenario map
# ---------------------------------------------------------------------------


_SCENARIO_BUILDERS = {
    ScenarioType.INDEPENDENT: _build_independent_agents,
    ScenarioType.SHARED_INTERFACE: _build_shared_interface_agents,
    ScenarioType.HIGH_CONTENTION: _build_high_contention_agents,
    ScenarioType.REALISTIC: _build_realistic_agents,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_benchmark(
    scenario: ScenarioType,
    agent_count: int,
    min_stability: float = 0.0,
) -> BenchmarkMetrics:
    """Run a single benchmark scenario and collect metrics."""
    resolver = IntentResolver(min_stability=min_stability)
    builder = _SCENARIO_BUILDERS[scenario]
    agents = builder(agent_count, resolver)

    runner = SimulationRunner(resolver)
    for agent in agents:
        runner.add_agent(agent)

    start = time.monotonic()
    result = runner.run()
    elapsed = time.monotonic() - start

    # Count adjustment types
    consume_count = 0
    adopt_count = 0
    for log in result.agent_logs.values():
        for adj in log.adjustments_applied:
            if adj.kind == "ConsumeInstead":
                consume_count += 1
            elif adj.kind == "AdoptConstraint":
                adopt_count += 1

    return BenchmarkMetrics(
        scenario=scenario.value,
        agent_count=agent_count,
        total_intents=result.total_intents,
        total_resolutions=sum(len(log.resolutions) for log in result.agent_logs.values()),
        total_conflicts=result.total_conflicts,
        total_adjustments=result.total_adjustments,
        consume_instead_count=consume_count,
        adopt_constraint_count=adopt_count,
        convergence_rounds=len(result.rounds),
        all_converged=result.all_converged,
        wall_clock_seconds=elapsed,
    )


def run_scaling_suite(
    agent_counts: list[int] | None = None,
    scenarios: list[ScenarioType] | None = None,
) -> BenchmarkSuite:
    """Run the full scaling benchmark suite.

    Default: 2, 5, 10, 25 agents across all scenario types.
    """
    agent_counts = agent_counts or [2, 5, 10, 25]
    scenarios = scenarios or [
        ScenarioType.INDEPENDENT,
        ScenarioType.SHARED_INTERFACE,
        ScenarioType.HIGH_CONTENTION,
        ScenarioType.REALISTIC,
    ]

    suite = BenchmarkSuite()
    for scenario in scenarios:
        for count in agent_counts:
            metrics = run_benchmark(scenario, count)
            suite.results.append(metrics)

    return suite
