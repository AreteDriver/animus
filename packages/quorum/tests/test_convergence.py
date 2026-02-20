"""
Tests for Convergent — proving convergence properties.

These tests verify that:
1. Agents independently arrive at compatible outputs
2. Higher-stability intents become attractors
3. Duplicate provisions are detected and resolved
4. Constraints propagate correctly
5. Conflicts are reported when they can't be auto-resolved
"""

import pytest
from convergent.agent import AgentAction, SimulatedAgent, SimulationRunner
from convergent.intent import (
    Constraint,
    Evidence,
    Intent,
    InterfaceKind,
    InterfaceSpec,
)
from convergent.matching import (
    names_overlap,
    normalize_constraint_target,
    normalize_name,
    signatures_compatible,
)
from convergent.resolver import IntentResolver


@pytest.fixture
def resolver():
    return IntentResolver(min_stability=0.0)


class TestInterfaceOverlap:
    """Test structural overlap detection between interface specs."""

    def test_same_name_overlaps(self):
        a = InterfaceSpec(name="User", kind=InterfaceKind.MODEL, signature="id: UUID")
        b = InterfaceSpec(name="User", kind=InterfaceKind.MODEL, signature="id: int")
        assert a.structurally_overlaps(b)

    def test_different_name_no_overlap(self):
        a = InterfaceSpec(name="User", kind=InterfaceKind.MODEL, signature="id: UUID")
        b = InterfaceSpec(name="Recipe", kind=InterfaceKind.MODEL, signature="id: UUID")
        assert not a.structurally_overlaps(b)

    def test_tag_overlap_two_shared(self):
        a = InterfaceSpec(
            name="UserModel",
            kind=InterfaceKind.MODEL,
            signature="id: UUID",
            tags=["user", "auth", "model"],
        )
        b = InterfaceSpec(
            name="AccountModel",
            kind=InterfaceKind.MODEL,
            signature="id: UUID",
            tags=["user", "account", "model"],
        )
        # 2 shared tags: "user", "model"
        assert a.structurally_overlaps(b)

    def test_tag_overlap_one_shared_insufficient(self):
        a = InterfaceSpec(
            name="UserModel",
            kind=InterfaceKind.MODEL,
            signature="id: UUID",
            tags=["user", "auth"],
        )
        b = InterfaceSpec(
            name="RecipeModel",
            kind=InterfaceKind.MODEL,
            signature="id: UUID",
            tags=["user", "recipe"],
        )
        # Only 1 shared tag: "user"
        assert not a.structurally_overlaps(b)

    def test_signature_compatibility(self):
        a = InterfaceSpec(
            name="User",
            kind=InterfaceKind.MODEL,
            signature="id: UUID, email: str",
        )
        b = InterfaceSpec(
            name="User",
            kind=InterfaceKind.MODEL,
            signature="id: UUID, email: str",
        )
        assert a.signature_compatible(b)

    def test_signature_incompatibility(self):
        a = InterfaceSpec(
            name="User",
            kind=InterfaceKind.MODEL,
            signature="id: UUID, email: str",
        )
        b = InterfaceSpec(
            name="User",
            kind=InterfaceKind.MODEL,
            signature="id: int, name: str",
        )
        assert not a.signature_compatible(b)


class TestStabilityComputation:
    """Test stability scoring from evidence."""

    def test_base_stability(self):
        intent = Intent(agent_id="a", intent="test")
        assert abs(intent.compute_stability() - 0.3) < 0.01

    def test_code_committed_increases(self):
        intent = Intent(
            agent_id="a",
            intent="test",
            evidence=[Evidence.code_committed("initial")],
        )
        assert abs(intent.compute_stability() - 0.5) < 0.01

    def test_test_pass_increases(self):
        intent = Intent(
            agent_id="a",
            intent="test",
            evidence=[
                Evidence.test_pass("t1"),
                Evidence.test_pass("t2"),
            ],
        )
        # 0.3 + 2*0.05 = 0.4
        assert abs(intent.compute_stability() - 0.4) < 0.01

    def test_conflict_decreases(self):
        intent = Intent(
            agent_id="a",
            intent="test",
            evidence=[
                Evidence.code_committed("commit"),
                Evidence.conflict("schema clash"),
            ],
        )
        # 0.3 + 0.2 - 0.15 = 0.35
        assert abs(intent.compute_stability() - 0.35) < 0.01

    def test_high_stability_scenario(self):
        intent = Intent(
            agent_id="a",
            intent="test",
            evidence=[
                Evidence.code_committed("commit"),
                Evidence.test_pass("t1"),
                Evidence.test_pass("t2"),
                Evidence.consumed_by("agent-b"),
                Evidence.consumed_by("agent-c"),
            ],
        )
        # 0.3 + 0.2 + 2*0.05 + 2*0.1 = 0.8
        assert abs(intent.compute_stability() - 0.8) < 0.01


class TestConstraintPropagation:
    """Test that constraints from high-stability intents propagate to others."""

    def test_constraint_applies_by_tag(self):
        constraint = Constraint(
            target="User model",
            requirement="must have email: str",
            affects_tags=["user", "model"],
        )
        intent = Intent(
            agent_id="b",
            intent="recipe module",
            provides=[
                InterfaceSpec(
                    name="Recipe",
                    kind=InterfaceKind.MODEL,
                    signature="id: UUID",
                    tags=["recipe", "user", "model"],
                ),
            ],
        )
        assert constraint.applies_to(intent)

    def test_constraint_does_not_apply_without_tags(self):
        constraint = Constraint(
            target="User model",
            requirement="must have email: str",
            affects_tags=["user", "auth"],
        )
        intent = Intent(
            agent_id="b",
            intent="recipe module",
            provides=[
                InterfaceSpec(
                    name="Recipe",
                    kind=InterfaceKind.MODEL,
                    signature="id: UUID",
                    tags=["recipe", "food"],
                ),
            ],
        )
        assert not constraint.applies_to(intent)

    def test_constraint_conflict_detection(self):
        c1 = Constraint(target="User.id", requirement="must be UUID")
        c2 = Constraint(target="User.id", requirement="must be int")
        assert c1.conflicts_with(c2)

    def test_constraint_no_conflict_different_target(self):
        c1 = Constraint(target="User.id", requirement="must be UUID")
        c2 = Constraint(target="Recipe.id", requirement="must be int")
        assert not c1.conflicts_with(c2)


class TestIntentResolution:
    """Test the resolver's ability to detect overlaps and recommend adjustments."""

    def test_consume_instead_when_other_has_higher_stability(self, resolver):
        # Agent A publishes User model with high stability
        intent_a = Intent(
            agent_id="agent-a",
            intent="Auth module",
            provides=[
                InterfaceSpec(
                    name="User",
                    kind=InterfaceKind.MODEL,
                    signature="id: UUID, email: str",
                    tags=["user", "auth", "model"],
                ),
            ],
            evidence=[
                Evidence.code_committed("auth/models.py"),
                Evidence.test_pass("test_user"),
            ],
        )
        resolver.publish(intent_a)

        # Agent C tries to provide its own User model (low stability)
        intent_c = Intent(
            agent_id="agent-c",
            intent="Meal planning",
            provides=[
                InterfaceSpec(
                    name="User",
                    kind=InterfaceKind.MODEL,
                    signature="id: UUID, name: str",
                    tags=["user", "meal", "model"],
                ),
            ],
        )

        result = resolver.resolve(intent_c)
        assert result.has_adjustments
        assert any(a.kind == "ConsumeInstead" for a in result.adjustments)

    def test_no_adjustment_when_no_overlap(self, resolver):
        intent_a = Intent(
            agent_id="agent-a",
            intent="Auth module",
            provides=[
                InterfaceSpec(
                    name="AuthService",
                    kind=InterfaceKind.CLASS,
                    signature="authenticate, create_token",
                    tags=["auth", "service"],
                ),
            ],
        )
        resolver.publish(intent_a)

        intent_b = Intent(
            agent_id="agent-b",
            intent="Recipe module",
            provides=[
                InterfaceSpec(
                    name="RecipeService",
                    kind=InterfaceKind.CLASS,
                    signature="create, list, search",
                    tags=["recipe", "service"],
                ),
            ],
        )

        result = resolver.resolve(intent_b)
        assert result.is_clean
        assert not result.has_adjustments

    def test_constraint_adoption(self, resolver):
        # Agent A publishes constraint about User.id type
        intent_a = Intent(
            agent_id="agent-a",
            intent="Auth module",
            provides=[
                InterfaceSpec(
                    name="User",
                    kind=InterfaceKind.MODEL,
                    signature="id: UUID",
                    tags=["user", "model"],
                ),
            ],
            constraints=[
                Constraint(
                    target="User.id type",
                    requirement="must be UUID, not int",
                    affects_tags=["user", "model"],
                ),
            ],
            evidence=[Evidence.code_committed("committed")],
        )
        resolver.publish(intent_a)

        # Agent B has user-related interface
        intent_b = Intent(
            agent_id="agent-b",
            intent="Recipe module",
            provides=[
                InterfaceSpec(
                    name="Recipe",
                    kind=InterfaceKind.MODEL,
                    signature="id: UUID, author_id: UUID",
                    tags=["recipe", "user", "model"],
                ),
            ],
        )

        result = resolver.resolve(intent_b)
        assert len(result.adopted_constraints) > 0
        assert any("UUID" in c.requirement for c in result.adopted_constraints)

    def test_self_exclusion(self, resolver):
        """Agent should not detect overlap with its own intents."""
        intent = Intent(
            agent_id="agent-a",
            intent="Auth module",
            provides=[
                InterfaceSpec(
                    name="User",
                    kind=InterfaceKind.MODEL,
                    signature="id: UUID",
                    tags=["user", "model"],
                ),
            ],
        )
        resolver.publish(intent)

        # Same agent, same interface — should not conflict with self
        intent2 = Intent(
            agent_id="agent-a",
            intent="Auth v2",
            provides=[
                InterfaceSpec(
                    name="User",
                    kind=InterfaceKind.MODEL,
                    signature="id: UUID, email: str",
                    tags=["user", "model"],
                ),
            ],
        )

        result = resolver.resolve(intent2)
        assert result.is_clean


class TestSimulatedConvergence:
    """End-to-end tests proving that agents converge."""

    def test_three_agents_converge(self):
        """The full recipe app demo should converge with no conflicts."""
        resolver = IntentResolver(min_stability=0.0)

        # Import demo agents
        from convergent.demo import build_agent_a, build_agent_b, build_agent_c

        agent_a = build_agent_a(resolver)
        agent_b = build_agent_b(resolver)
        agent_c = build_agent_c(resolver)

        runner = SimulationRunner(resolver)
        runner.add_agent(agent_a)
        runner.add_agent(agent_b)
        runner.add_agent(agent_c)

        result = runner.run()

        # Core assertion: all agents converged
        assert result.all_converged, f"Agents did not converge. Conflicts: {result.total_conflicts}"

        # Agent C should have adjusted (consumed Agent A's User model)
        agent_c_log = result.agent_logs["agent-c"]
        consume_adjustments = [
            a for a in agent_c_log.adjustments_applied if a.kind == "ConsumeInstead"
        ]
        assert len(consume_adjustments) > 0, "Agent C should have consumed Agent A's User model"

    def test_two_agents_no_overlap(self):
        """Agents with completely independent scopes should have no adjustments."""
        resolver = IntentResolver(min_stability=0.0)

        agent_a = SimulatedAgent("agent-a", resolver)
        agent_a.plan(
            [
                AgentAction(
                    intent=Intent(
                        agent_id="agent-a",
                        intent="Logging module",
                        provides=[
                            InterfaceSpec(
                                name="Logger",
                                kind=InterfaceKind.CLASS,
                                signature="log(level, msg)",
                                tags=["logging", "infra"],
                            ),
                        ],
                    ),
                ),
            ]
        )

        agent_b = SimulatedAgent("agent-b", resolver)
        agent_b.plan(
            [
                AgentAction(
                    intent=Intent(
                        agent_id="agent-b",
                        intent="Email module",
                        provides=[
                            InterfaceSpec(
                                name="EmailService",
                                kind=InterfaceKind.CLASS,
                                signature="send(to, subject, body)",
                                tags=["email", "notifications"],
                            ),
                        ],
                    ),
                ),
            ]
        )

        runner = SimulationRunner(resolver)
        runner.add_agent(agent_a)
        runner.add_agent(agent_b)
        result = runner.run()

        assert result.all_converged
        assert result.total_adjustments == 0

    def test_stability_ordering_determines_winner(self):
        """When two agents provide the same thing, higher stability wins."""
        resolver = IntentResolver(min_stability=0.0)

        # Agent A publishes first with high evidence
        agent_a = SimulatedAgent("agent-a", resolver)
        agent_a.plan(
            [
                AgentAction(
                    intent=Intent(
                        agent_id="agent-a",
                        intent="Config service",
                        provides=[
                            InterfaceSpec(
                                name="Config",
                                kind=InterfaceKind.CLASS,
                                signature="get(key) -> str",
                                tags=["config", "settings"],
                            ),
                        ],
                        evidence=[
                            Evidence.code_committed("config.py"),
                            Evidence.test_pass("test_config"),
                        ],
                    ),
                ),
            ]
        )

        # Agent B also tries to provide Config (lower stability)
        agent_b = SimulatedAgent("agent-b", resolver)
        agent_b.plan(
            [
                AgentAction(
                    intent=Intent(
                        agent_id="agent-b",
                        intent="Settings handler",
                        provides=[
                            InterfaceSpec(
                                name="Config",
                                kind=InterfaceKind.CLASS,
                                signature="get(key) -> Optional[str]",
                                tags=["config", "settings"],
                            ),
                        ],
                    ),
                ),
            ]
        )

        runner = SimulationRunner(resolver)
        runner.add_agent(agent_a)
        runner.add_agent(agent_b)
        result = runner.run()

        # Agent B should yield to Agent A
        b_log = result.agent_logs["agent-b"]
        assert any(a.kind == "ConsumeInstead" for a in b_log.adjustments_applied), (
            "Agent B should consume Agent A's Config"
        )

    def test_constraint_propagation_across_agents(self):
        """Constraints from committed agents should propagate to new agents."""
        resolver = IntentResolver(min_stability=0.0)

        # Agent A commits with a constraint
        agent_a = SimulatedAgent("agent-a", resolver)
        agent_a.plan(
            [
                AgentAction(
                    intent=Intent(
                        agent_id="agent-a",
                        intent="Database layer",
                        provides=[
                            InterfaceSpec(
                                name="DBConnection",
                                kind=InterfaceKind.CLASS,
                                signature="query(sql) -> Result",
                                tags=["database", "infra"],
                            ),
                        ],
                        constraints=[
                            Constraint(
                                target="database access",
                                requirement="all queries must use parameterized SQL",
                                affects_tags=["database", "query"],
                            ),
                        ],
                        evidence=[Evidence.code_committed("db.py")],
                    ),
                ),
            ]
        )

        # Agent B works on something database-related
        agent_b = SimulatedAgent("agent-b", resolver)
        agent_b.plan(
            [
                AgentAction(
                    intent=Intent(
                        agent_id="agent-b",
                        intent="User repository",
                        provides=[
                            InterfaceSpec(
                                name="UserRepo",
                                kind=InterfaceKind.CLASS,
                                signature="find(id) -> User",
                                tags=["user", "database", "query"],
                            ),
                        ],
                    ),
                ),
            ]
        )

        runner = SimulationRunner(resolver)
        runner.add_agent(agent_a)
        runner.add_agent(agent_b)
        result = runner.run()

        # Agent B should adopt the SQL parameterization constraint
        b_log = result.agent_logs["agent-b"]
        adopt_adjustments = [a for a in b_log.adjustments_applied if a.kind == "AdoptConstraint"]
        assert len(adopt_adjustments) > 0, "Agent B should adopt the parameterized SQL constraint"


class TestSemanticMatching:
    """Test structural semantic matching — deterministic, no LLM."""

    # ── Name normalization ──────────────────────────────────────────

    def test_normalize_name_strips_model_suffix(self):
        assert normalize_name("UserModel") == "user"

    def test_normalize_name_strips_service_suffix(self):
        assert normalize_name("AuthService") == "auth"

    def test_normalize_name_camel_case_splitting(self):
        assert normalize_name("MealPlanService") == "meal plan"
        assert normalize_name("UserProfile") == "user profile"

    def test_normalize_name_simple(self):
        assert normalize_name("User") == "user"
        assert normalize_name("user") == "user"

    def test_normalize_name_empty(self):
        assert normalize_name("") == ""

    # ── Names overlap ───────────────────────────────────────────────

    def test_names_overlap_after_suffix_stripping(self):
        assert names_overlap("UserModel", "User")
        assert names_overlap("AuthService", "Auth")

    def test_names_overlap_prefix_match(self):
        assert names_overlap("User", "UserProfile")

    def test_names_overlap_containment(self):
        assert names_overlap("auth", "AuthService")

    def test_names_overlap_no_match(self):
        assert not names_overlap("User", "Recipe")
        assert not names_overlap("AuthService", "RecipeService")

    def test_names_overlap_empty(self):
        assert not names_overlap("", "User")
        assert not names_overlap("User", "")

    # ── Signature compatibility ─────────────────────────────────────

    def test_signatures_compatible_exact(self):
        assert signatures_compatible("id: UUID, email: str", "id: UUID, email: str")

    def test_signatures_compatible_superset(self):
        assert signatures_compatible("id: UUID, email: str", "id: UUID, email: str, name: str")

    def test_signatures_compatible_type_aliases(self):
        assert signatures_compatible("id: UUID, name: String", "id: uuid, name: str")
        assert signatures_compatible("count: i64", "count: int")
        assert signatures_compatible("score: f64", "score: float")

    def test_signatures_compatible_optional(self):
        assert signatures_compatible("name: str", "name: Optional[str]")
        # Optional[str] normalizes to "str", which matches "str"

    def test_signatures_compatible_containers(self):
        assert signatures_compatible("items: Vec<String>", "items: list[str]")
        assert signatures_compatible("items: List[str]", "items: list[str]")

    def test_signatures_incompatible_missing_field(self):
        assert not signatures_compatible("id: UUID, email: str", "id: UUID")

    def test_signatures_incompatible_type_mismatch(self):
        assert not signatures_compatible("id: UUID", "id: int")

    def test_signatures_compatible_empty(self):
        assert signatures_compatible("", "id: UUID")
        assert signatures_compatible("", "")

    def test_signatures_single_field(self):
        assert signatures_compatible("id: UUID", "id: UUID")
        assert not signatures_compatible("id: UUID", "id: int")

    # ── Constraint target normalization ─────────────────────────────

    def test_constraint_target_case_insensitive(self):
        assert normalize_constraint_target("User Model") == normalize_constraint_target(
            "user model"
        )

    def test_constraint_target_strips_model_suffix(self):
        assert normalize_constraint_target("User Model") == "user"
        assert normalize_constraint_target("user_model") == "user"

    def test_constraint_target_strips_service_suffix(self):
        assert normalize_constraint_target("user-service") == "user"

    def test_constraint_target_whitespace_normalization(self):
        assert normalize_constraint_target("User  Model") == normalize_constraint_target(
            "User Model"
        )
        assert normalize_constraint_target("user_model") == normalize_constraint_target(
            "user-model"
        )

    def test_constraint_target_empty(self):
        assert normalize_constraint_target("") == ""

    # ── Backward compatibility ──────────────────────────────────────

    def test_exact_name_match_still_overlaps(self):
        """Exact name matches that worked before still work."""
        a = InterfaceSpec(name="User", kind=InterfaceKind.MODEL, signature="id: UUID")
        b = InterfaceSpec(name="User", kind=InterfaceKind.MODEL, signature="id: int")
        assert a.structurally_overlaps(b)

    def test_exact_signature_still_compatible(self):
        """Exact signature matches that worked before still work."""
        a = InterfaceSpec(
            name="User",
            kind=InterfaceKind.MODEL,
            signature="id: UUID, email: str",
        )
        b = InterfaceSpec(
            name="User",
            kind=InterfaceKind.MODEL,
            signature="id: UUID, email: str",
        )
        assert a.signature_compatible(b)

    def test_exact_constraint_conflict_still_detected(self):
        """Exact constraint conflicts that worked before still work."""
        c1 = Constraint(target="User.id", requirement="must be UUID")
        c2 = Constraint(target="User.id", requirement="must be int")
        assert c1.conflicts_with(c2)

    # ── Integration: structural overlap with semantic matching ──────

    def test_model_suffix_overlap_detected(self):
        """UserModel and User should be detected as overlapping."""
        a = InterfaceSpec(
            name="UserModel",
            kind=InterfaceKind.MODEL,
            signature="id: UUID",
        )
        b = InterfaceSpec(
            name="User",
            kind=InterfaceKind.MODEL,
            signature="id: UUID",
        )
        assert a.structurally_overlaps(b)

    def test_service_suffix_overlap_detected(self):
        """AuthService and Auth should be detected as overlapping."""
        a = InterfaceSpec(
            name="AuthService",
            kind=InterfaceKind.CLASS,
            signature="authenticate, create_token",
        )
        b = InterfaceSpec(
            name="Auth",
            kind=InterfaceKind.CLASS,
            signature="authenticate",
        )
        assert a.structurally_overlaps(b)

    def test_constraint_conflict_across_formats(self):
        """Constraints with different formatting of same target should conflict."""
        c1 = Constraint(target="User Model", requirement="must have email: str")
        c2 = Constraint(target="user_model", requirement="must have username: str")
        assert c1.conflicts_with(c2)
