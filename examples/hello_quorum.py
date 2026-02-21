#!/usr/bin/env python3
"""
hello_quorum.py — Two agents coordinate through a shared intent graph.

No supervisor. No message passing. Each agent reads the graph, adjusts
its plan for compatibility, and publishes. O(n) reads instead of O(n^2) messages.

Run:
    pip install -e packages/quorum
    python examples/hello_quorum.py
"""

from convergent.agent import AgentAction, SimulatedAgent, SimulationRunner
from convergent.intent import (
    Constraint,
    ConstraintSeverity,
    Evidence,
    Intent,
    InterfaceKind,
    InterfaceSpec,
)
from convergent.resolver import IntentResolver, PythonGraphBackend
from convergent.visualization import text_table

# --- Setup: shared graph, no supervisor ---

resolver = IntentResolver(backend=PythonGraphBackend(), min_stability=0.3)

# --- Agent A: builds the User model ---

agent_a = SimulatedAgent("auth-agent", resolver)
agent_a.plan([
    AgentAction(
        intent=Intent(
            agent_id="auth-agent",
            intent="Build authentication module",
            provides=[
                InterfaceSpec(
                    name="User",
                    kind=InterfaceKind.MODEL,
                    signature="id: UUID, email: str, hashed_password: str",
                    module_path="auth/models.py",
                    tags=["user", "auth", "model"],
                ),
                InterfaceSpec(
                    name="authenticate",
                    kind=InterfaceKind.FUNCTION,
                    signature="(email: str, password: str) -> User | None",
                    module_path="auth/service.py",
                    tags=["auth", "login"],
                ),
            ],
            constraints=[
                Constraint(
                    target="User model",
                    requirement="email must be unique, passwords must be hashed",
                    severity=ConstraintSeverity.REQUIRED,
                    affects_tags=["user", "account"],
                ),
            ],
        ),
        post_evidence=[
            Evidence.code_committed("auth/models.py — User model with email unique constraint"),
            Evidence.test_pass("test_user_creation — 4 cases passing"),
            Evidence.test_pass("test_authenticate — valid and invalid credentials"),
        ],
    ),
])

# --- Agent B: builds the API layer, depends on auth ---

agent_b = SimulatedAgent("api-agent", resolver)
agent_b.plan([
    AgentAction(
        intent=Intent(
            agent_id="api-agent",
            intent="Build REST API endpoints",
            requires=[
                InterfaceSpec(
                    name="User",
                    kind=InterfaceKind.MODEL,
                    signature="id: UUID, email: str",
                    module_path="auth/models.py",
                    tags=["user", "auth", "model"],
                ),
            ],
            provides=[
                InterfaceSpec(
                    name="/api/users",
                    kind=InterfaceKind.ENDPOINT,
                    signature="GET -> list[User], POST -> User",
                    module_path="api/routes.py",
                    tags=["api", "user", "rest"],
                ),
            ],
        ),
        post_evidence=[
            Evidence.code_committed("api/routes.py — user CRUD endpoints"),
            Evidence.test_pass("test_api_users — list, create, get"),
        ],
    ),
])

# --- Run simulation: round-robin execution ---

runner = SimulationRunner(resolver)
runner.add_agent(agent_a)
runner.add_agent(agent_b)
result = runner.run()

# --- Results ---

print("=" * 60)
print("QUORUM COORDINATION RESULT")
print("=" * 60)
print()
print(result.summary())
print()

# Show the shared intent graph
print("INTENT GRAPH STATE:")
print(text_table(resolver, show_evidence=True))

# Per-agent breakdown
for agent_id, log in result.agent_logs.items():
    print(f"\n--- {agent_id} ---")
    print(f"  Published: {len(log.published_intents)} intents")
    print(f"  Adjustments: {log.total_adjustments}")
    print(f"  Conflicts: {log.total_conflicts}")
    print(f"  Converged: {log.converged}")
