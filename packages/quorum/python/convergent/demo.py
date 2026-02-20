"""
Convergent Demo — Three Agents Building a Recipe App

Demonstrates how three agents independently converge on compatible
designs by observing a shared intent graph.

Agent A: Authentication module (User model, AuthService)
Agent B: Recipe module (Recipe model, RecipeService)
Agent C: Meal planning module (MealPlan model, MealPlanService)

Without Convergent: incompatible User models, conflicting schemas,
mismatched interfaces.

With Convergent: all three converge on compatible designs through
ambient awareness.
"""

import logging
import sys

from convergent.agent import AgentAction, SimulatedAgent, SimulationRunner
from convergent.intent import (
    Constraint,
    Evidence,
    Intent,
    InterfaceKind,
    InterfaceSpec,
)
from convergent.resolver import IntentResolver


def build_agent_a(resolver: IntentResolver) -> SimulatedAgent:
    """Agent A: Authentication module.

    Publishes: User model, AuthService, JWT token handling.
    This agent starts first and moves fast — its decisions become
    the attractors that other agents converge toward.
    """
    agent = SimulatedAgent("agent-a", resolver)

    # Step 1: Declare intent to build auth module (exploring)
    step1 = AgentAction(
        intent=Intent(
            agent_id="agent-a",
            intent="Build authentication module with JWT tokens",
            provides=[
                InterfaceSpec(
                    name="User",
                    kind=InterfaceKind.MODEL,
                    signature="id: UUID, email: str, hashed_password: str, created_at: datetime",
                    module_path="auth/models.py",
                    tags=["user", "auth", "model", "account"],
                ),
                InterfaceSpec(
                    name="AuthService.authenticate",
                    kind=InterfaceKind.FUNCTION,
                    signature="(email: str, password: str) -> Optional[User]",
                    module_path="auth/service.py",
                    tags=["auth", "login", "user"],
                ),
                InterfaceSpec(
                    name="AuthService.current_user",
                    kind=InterfaceKind.FUNCTION,
                    signature="(token: str) -> User",
                    module_path="auth/service.py",
                    tags=["auth", "user", "token"],
                ),
            ],
            constraints=[
                Constraint(
                    target="User model",
                    requirement="must have email: str as unique field",
                    affects_tags=["user", "account"],
                ),
                Constraint(
                    target="User model",
                    requirement="must have id: UUID as primary key",
                    affects_tags=["user", "model"],
                ),
            ],
        ),
        post_evidence=[
            Evidence.code_committed("auth/models.py — User model"),
            Evidence.test_pass("test_user_creation"),
        ],
    )

    # Step 2: Commit auth service (high stability)
    step2 = AgentAction(
        intent=Intent(
            agent_id="agent-a",
            intent="AuthService committed with JWT token validation",
            provides=[
                InterfaceSpec(
                    name="AuthService.create_token",
                    kind=InterfaceKind.FUNCTION,
                    signature="(user: User) -> str",
                    module_path="auth/service.py",
                    tags=["auth", "token", "jwt"],
                ),
                InterfaceSpec(
                    name="AuthService.validate_token",
                    kind=InterfaceKind.FUNCTION,
                    signature="(token: str) -> Optional[User]",
                    module_path="auth/service.py",
                    tags=["auth", "token", "jwt", "user"],
                ),
            ],
            constraints=[
                Constraint(
                    target="authentication",
                    requirement="all authenticated endpoints must accept Bearer token",
                    affects_tags=["auth", "endpoint", "api"],
                ),
            ],
        ),
        post_evidence=[
            Evidence.code_committed("auth/service.py — complete auth service"),
            Evidence.test_pass("test_jwt_creation"),
            Evidence.test_pass("test_jwt_validation"),
            Evidence.test_pass("test_auth_flow_integration"),
        ],
    )

    agent.plan([step1, step2])
    return agent


def build_agent_b(resolver: IntentResolver) -> SimulatedAgent:
    """Agent B: Recipe module.

    Publishes: Recipe model, RecipeService.
    Requires: User model (from Agent A).

    This agent will observe Agent A's User model and adopt it rather
    than creating its own. It will also adopt the UUID primary key constraint.
    """
    agent = SimulatedAgent("agent-b", resolver)

    # Step 1: Declare recipe module with its own User reference
    step1 = AgentAction(
        intent=Intent(
            agent_id="agent-b",
            intent="Build recipe module with CRUD operations",
            provides=[
                InterfaceSpec(
                    name="Recipe",
                    kind=InterfaceKind.MODEL,
                    signature=(
                        "id: UUID, title: str, author_id: UUID,"
                        " ingredients: list[str], steps: list[str]"
                    ),
                    module_path="recipes/models.py",
                    tags=["recipe", "model", "food"],
                ),
                InterfaceSpec(
                    name="RecipeService.create",
                    kind=InterfaceKind.FUNCTION,
                    signature="(recipe: Recipe, user: User) -> Recipe",
                    module_path="recipes/service.py",
                    tags=["recipe", "crud", "create", "user"],
                ),
                InterfaceSpec(
                    name="RecipeService.list_by_author",
                    kind=InterfaceKind.FUNCTION,
                    signature="(author_id: UUID) -> list[Recipe]",
                    module_path="recipes/service.py",
                    tags=["recipe", "crud", "list", "user"],
                ),
            ],
            requires=[
                InterfaceSpec(
                    name="User",
                    kind=InterfaceKind.MODEL,
                    signature="id: UUID, email: str",
                    module_path="auth/models.py",
                    tags=["user", "auth", "model"],
                ),
            ],
            constraints=[
                Constraint(
                    target="Recipe model",
                    requirement="author_id must reference User.id as foreign key",
                    affects_tags=["recipe", "user", "model"],
                ),
            ],
        ),
        post_evidence=[
            Evidence.code_committed("recipes/models.py — Recipe model"),
            Evidence.test_pass("test_recipe_creation"),
        ],
    )

    # Step 2: Commit RecipeService with batch operations
    step2 = AgentAction(
        intent=Intent(
            agent_id="agent-b",
            intent="RecipeService committed with batch operations",
            provides=[
                InterfaceSpec(
                    name="RecipeService.batch_get",
                    kind=InterfaceKind.FUNCTION,
                    signature="(ids: list[UUID]) -> list[Recipe]",
                    module_path="recipes/service.py",
                    tags=["recipe", "crud", "batch", "read"],
                ),
                InterfaceSpec(
                    name="RecipeService.search",
                    kind=InterfaceKind.FUNCTION,
                    signature="(query: str, limit: int = 20) -> list[Recipe]",
                    module_path="recipes/service.py",
                    tags=["recipe", "search", "read"],
                ),
            ],
        ),
        post_evidence=[
            Evidence.code_committed("recipes/service.py — batch operations"),
            Evidence.test_pass("test_batch_get"),
            Evidence.test_pass("test_search"),
        ],
    )

    agent.plan([step1, step2])
    return agent


def build_agent_c(resolver: IntentResolver) -> SimulatedAgent:
    """Agent C: Meal planning module.

    Publishes: MealPlan model, MealPlanService.
    Requires: User model (from Agent A), Recipe/RecipeService (from Agent B).

    This agent starts with its own User concept but should converge
    to use Agent A's User model. It should also adopt Agent B's Recipe
    interface rather than creating its own recipe abstraction.
    """
    agent = SimulatedAgent("agent-c", resolver)

    # Step 1: Declare meal planning — initially with its own User concept
    # The resolver should tell it to consume Agent A's User instead
    step1 = AgentAction(
        intent=Intent(
            agent_id="agent-c",
            intent="Build meal planning module",
            provides=[
                InterfaceSpec(
                    name="User",
                    kind=InterfaceKind.MODEL,
                    signature="id: UUID, name: str, preferences: dict",
                    module_path="meals/models.py",
                    tags=["user", "meal", "model"],
                ),
                InterfaceSpec(
                    name="MealPlan",
                    kind=InterfaceKind.MODEL,
                    signature="id: UUID, user_id: UUID, recipes: list[UUID], week_start: date",
                    module_path="meals/models.py",
                    tags=["meal", "plan", "model"],
                ),
                InterfaceSpec(
                    name="MealPlanService.create",
                    kind=InterfaceKind.FUNCTION,
                    signature=(
                        "(user_id: UUID, recipe_ids: list[UUID], week_start: date) -> MealPlan"
                    ),
                    module_path="meals/service.py",
                    tags=["meal", "plan", "crud", "create"],
                ),
            ],
            requires=[
                InterfaceSpec(
                    name="RecipeService.batch_get",
                    kind=InterfaceKind.FUNCTION,
                    signature="(ids: list[UUID]) -> list[Recipe]",
                    module_path="recipes/service.py",
                    tags=["recipe", "crud", "batch", "read"],
                ),
            ],
        ),
        post_evidence=[
            Evidence.code_committed("meals/models.py — MealPlan model"),
        ],
    )

    # Step 2: Commit MealPlanService
    step2 = AgentAction(
        intent=Intent(
            agent_id="agent-c",
            intent="MealPlanService committed with weekly planning",
            provides=[
                InterfaceSpec(
                    name="MealPlanService.get_week",
                    kind=InterfaceKind.FUNCTION,
                    signature="(user_id: UUID, week_start: date) -> Optional[MealPlan]",
                    module_path="meals/service.py",
                    tags=["meal", "plan", "read", "week"],
                ),
            ],
            requires=[
                InterfaceSpec(
                    name="AuthService.current_user",
                    kind=InterfaceKind.FUNCTION,
                    signature="(token: str) -> User",
                    module_path="auth/service.py",
                    tags=["auth", "user", "token"],
                ),
            ],
        ),
        post_evidence=[
            Evidence.code_committed("meals/service.py — weekly planning"),
            Evidence.test_pass("test_create_meal_plan"),
            Evidence.test_pass("test_get_week"),
        ],
    )

    agent.plan([step1, step2])
    return agent


def run_demo() -> None:
    """Run the convergence demo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    print()
    print("=" * 60)
    print("  CONVERGENT DEMO")
    print("  Three agents building a recipe app")
    print("=" * 60)
    print()

    # Create shared resolver (all agents observe the same graph)
    resolver = IntentResolver(min_stability=0.2)

    # Build agents
    agent_a = build_agent_a(resolver)
    agent_b = build_agent_b(resolver)
    agent_c = build_agent_c(resolver)

    # Run simulation (round-robin interleaving)
    runner = SimulationRunner(resolver)
    runner.add_agent(agent_a)
    runner.add_agent(agent_b)
    runner.add_agent(agent_c)

    result = runner.run()

    # Print results
    print()
    print(result.summary())

    # Highlight the key convergence moment
    print()
    print("KEY CONVERGENCE MOMENT:")
    print("-" * 40)

    for agent_id, log in result.agent_logs.items():
        for adj in log.adjustments_applied:
            if adj.kind == "ConsumeInstead":
                print(
                    f"  {agent_id} dropped its own provision and consumed another agent's instead:"
                )
                print(f"    {adj.description}")
                print()

    if result.all_converged:
        print("All agents converged on compatible designs.")
        print("No direct communication occurred between agents.")
        print("Coherence emerged from ambient intent awareness.")
    else:
        print(f"WARNING: {result.total_conflicts} unresolved conflicts remain.")
        print("This indicates the scope split was too ambiguous for")
        print("automatic convergence. Manual intervention needed.")


if __name__ == "__main__":
    run_demo()
