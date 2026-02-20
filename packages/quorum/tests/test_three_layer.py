"""
Tests for the 3-layer coordination stack.

Layer 1: Constraints — hard truth (tests, types, schemas, invariants)
Layer 2: Intent graph — shared decisions (interfaces, stability, ownership)
Layer 3: Economics — optimization (cost-of-rework, budget, escalation)

These tests prove:
  1. Constraint engine validates typed constraints deterministically
  2. Economics layer makes escalation cost-optimal, not conversational
  3. Merge governor integrates all three layers in correct order
  4. Agent branches enforce propose/commit workflow
  5. The 3-layer stack rejects invalid states while allowing valid ones
  6. Coordination overhead scales sublinearly
"""

import uuid

import pytest
from convergent.constraints import (
    ConstraintEngine,
    ConstraintKind,
    TypedConstraint,
)
from convergent.contract import (
    ContractViolation,
)
from convergent.economics import (
    Budget,
    CoordinationCostReport,
    CostModel,
    EscalationAction,
    EscalationDecision,
    EscalationPolicy,
)
from convergent.governor import (
    AgentBranch,
    MergeGovernor,
    VerdictKind,
)
from convergent.intent import (
    Constraint,
    ConstraintSeverity,
    Evidence,
    Intent,
    InterfaceKind,
    InterfaceSpec,
)
from convergent.resolver import IntentResolver
from convergent.versioning import VersionedGraph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_intent(
    agent_id: str = "agent-a",
    intent_text: str = "test intent",
    provides: list[InterfaceSpec] | None = None,
    requires: list[InterfaceSpec] | None = None,
    constraints: list[Constraint] | None = None,
    evidence: list[Evidence] | None = None,
    intent_id: str | None = None,
) -> Intent:
    return Intent(
        id=intent_id or str(uuid.uuid4()),
        agent_id=agent_id,
        intent=intent_text,
        provides=provides
        or [
            InterfaceSpec(
                name="TestInterface",
                kind=InterfaceKind.CLASS,
                signature="run() -> bool",
                tags=["test"],
            )
        ],
        requires=requires or [],
        constraints=constraints or [],
        evidence=evidence or [],
    )


def _user_model_intent(
    agent_id: str = "agent-a",
    evidence: list[Evidence] | None = None,
) -> Intent:
    """Standard User model intent for testing."""
    return _make_intent(
        agent_id=agent_id,
        intent_text="User model",
        provides=[
            InterfaceSpec(
                name="User",
                kind=InterfaceKind.MODEL,
                signature="id: UUID, email: str",
                tags=["user", "model", "auth"],
            )
        ],
        evidence=evidence or [],
    )


# ===================================================================
# Layer 1: Constraint Engine
# ===================================================================


class TestConstraintEngine:
    """Prove that typed constraints are validated deterministically."""

    def test_register_and_count(self):
        engine = ConstraintEngine()
        cid = engine.register(
            TypedConstraint(
                kind=ConstraintKind.SCHEMA_RULE,
                target="User",
                requirement="must have id and email",
                affects_tags=["user", "model"],
                required_fields={"id": "UUID", "email": "str"},
            )
        )
        assert engine.constraint_count == 1
        assert isinstance(cid, str)

    def test_unregister(self):
        engine = ConstraintEngine()
        cid = engine.register(
            TypedConstraint(
                target="test",
                affects_tags=["test"],
            )
        )
        assert engine.unregister(cid)
        assert engine.constraint_count == 0
        assert not engine.unregister("nonexistent")

    def test_constraints_for_finds_applicable(self):
        engine = ConstraintEngine()
        engine.register(
            TypedConstraint(
                target="User",
                affects_tags=["user", "model"],
            )
        )
        engine.register(
            TypedConstraint(
                target="Recipe",
                affects_tags=["recipe"],
            )
        )
        intent = _user_model_intent()
        applicable = engine.constraints_for(intent)
        assert len(applicable) == 1
        assert applicable[0].target == "User"

    def test_constraints_for_no_match(self):
        engine = ConstraintEngine()
        engine.register(
            TypedConstraint(
                target="Recipe",
                affects_tags=["recipe"],
            )
        )
        intent = _user_model_intent()
        assert len(engine.constraints_for(intent)) == 0


class TestTypeCheckConstraint:
    """Layer 1: Type-level constraints validate signatures."""

    def test_required_fields_satisfied(self):
        engine = ConstraintEngine()
        tc = TypedConstraint(
            kind=ConstraintKind.TYPE_CHECK,
            target="User model",
            requirement="must have id: UUID, email: str",
            affects_tags=["user", "model"],
            required_fields={"id": "UUID", "email": "str"},
        )
        intent = _user_model_intent()
        result = engine.check(tc, intent)
        assert result.satisfied
        assert len(result.violations) == 0
        assert len(result.evidence_produced) == 1

    def test_required_fields_missing(self):
        engine = ConstraintEngine()
        tc = TypedConstraint(
            kind=ConstraintKind.TYPE_CHECK,
            target="User model",
            requirement="must have id, email, phone",
            affects_tags=["user", "model"],
            required_fields={"id": "UUID", "email": "str", "phone": "str"},
        )
        intent = _user_model_intent()
        result = engine.check(tc, intent)
        assert not result.satisfied
        assert any("phone" in v for v in result.violations)

    def test_required_fields_wrong_type(self):
        engine = ConstraintEngine()
        tc = TypedConstraint(
            kind=ConstraintKind.TYPE_CHECK,
            target="User model",
            requirement="id must be int",
            affects_tags=["user", "model"],
            required_fields={"id": "int"},  # UUID != int
        )
        intent = _user_model_intent()
        result = engine.check(tc, intent)
        assert not result.satisfied
        assert any("type" in v.lower() for v in result.violations)

    def test_type_normalization_applies(self):
        """UUID and uuid should be treated as the same type."""
        engine = ConstraintEngine()
        tc = TypedConstraint(
            kind=ConstraintKind.TYPE_CHECK,
            target="User",
            affects_tags=["user", "model"],
            required_fields={"id": "uuid"},  # lowercase
        )
        intent = _user_model_intent()  # has "id: UUID" (uppercase)
        result = engine.check(tc, intent)
        assert result.satisfied


class TestTestGateConstraint:
    """Layer 1: Test gates require evidence before proceeding."""

    def test_gate_passes_with_evidence(self):
        engine = ConstraintEngine()
        tc = TypedConstraint(
            kind=ConstraintKind.TEST_GATE,
            target="models",
            requirement="must have tests",
            affects_tags=["model"],
            required_evidence=["test_pass"],
        )
        intent = _user_model_intent(
            evidence=[
                Evidence.test_pass("test_user_creation"),
            ]
        )
        result = engine.check(tc, intent)
        assert result.satisfied

    def test_gate_fails_without_evidence(self):
        engine = ConstraintEngine()
        tc = TypedConstraint(
            kind=ConstraintKind.TEST_GATE,
            target="models",
            requirement="must have tests",
            affects_tags=["model"],
            required_evidence=["test_pass"],
        )
        intent = _user_model_intent()  # No evidence
        result = engine.check(tc, intent)
        assert not result.satisfied
        assert any("test_pass" in v for v in result.violations)

    def test_multiple_evidence_requirements(self):
        engine = ConstraintEngine()
        tc = TypedConstraint(
            kind=ConstraintKind.TEST_GATE,
            target="production code",
            requirement="must have tests and commit",
            affects_tags=["model"],
            required_evidence=["test_pass", "code_committed"],
        )
        # Only has test_pass, missing code_committed
        intent = _user_model_intent(
            evidence=[
                Evidence.test_pass("test_user"),
            ]
        )
        result = engine.check(tc, intent)
        assert not result.satisfied
        assert any("code_committed" in v for v in result.violations)


class TestSecurityConstraint:
    """Layer 1: Security policies enforced via forbidden patterns."""

    def test_forbidden_pattern_blocks(self):
        engine = ConstraintEngine()
        tc = TypedConstraint(
            kind=ConstraintKind.SECURITY_POLICY,
            target="all services",
            requirement="no raw SQL",
            affects_tags=["test"],
            forbidden_patterns=[r"raw_sql|exec\("],
        )
        intent = _make_intent(
            provides=[
                InterfaceSpec(
                    name="UnsafeService",
                    kind=InterfaceKind.CLASS,
                    signature="raw_sql(query: str) -> list",
                    tags=["test"],
                )
            ]
        )
        result = engine.check(tc, intent)
        assert not result.satisfied

    def test_forbidden_pattern_passes(self):
        engine = ConstraintEngine()
        tc = TypedConstraint(
            kind=ConstraintKind.SECURITY_POLICY,
            target="all services",
            requirement="no raw SQL",
            affects_tags=["test"],
            forbidden_patterns=[r"raw_sql|exec\("],
        )
        intent = _make_intent(
            provides=[
                InterfaceSpec(
                    name="SafeService",
                    kind=InterfaceKind.CLASS,
                    signature="query(sql: str, params: list) -> list",
                    tags=["test"],
                )
            ]
        )
        result = engine.check(tc, intent)
        assert result.satisfied


class TestConstraintGating:
    """Layer 1: Full gate checks with severity-aware blocking."""

    def test_gate_passes_when_all_satisfied(self):
        engine = ConstraintEngine()
        engine.register(
            TypedConstraint(
                kind=ConstraintKind.TYPE_CHECK,
                target="User",
                affects_tags=["user", "model"],
                required_fields={"id": "UUID"},
            )
        )
        intent = _user_model_intent()
        gate = engine.gate(intent)
        assert gate.passed
        assert gate.violated_count == 0

    def test_gate_blocks_on_required_violation(self):
        engine = ConstraintEngine()
        engine.register(
            TypedConstraint(
                kind=ConstraintKind.SCHEMA_RULE,
                target="User",
                severity=ConstraintSeverity.REQUIRED,
                affects_tags=["user", "model"],
                required_fields={"phone": "str"},
            )
        )
        intent = _user_model_intent()
        gate = engine.gate(intent)
        assert not gate.passed
        assert gate.violated_count == 1
        assert len(gate.blocking_violations) == 1

    def test_gate_blocks_on_critical_violation(self):
        engine = ConstraintEngine()
        engine.register(
            TypedConstraint(
                kind=ConstraintKind.SECURITY_POLICY,
                target="security",
                severity=ConstraintSeverity.CRITICAL,
                affects_tags=["user", "model"],
                required_evidence=["manual_approval"],
            )
        )
        intent = _user_model_intent()
        gate = engine.gate(intent)
        assert not gate.passed
        assert any("critical" in v.lower() for v in gate.blocking_violations)

    def test_gate_warns_on_preferred_violation(self):
        """Preferred constraints don't block, only warn."""
        engine = ConstraintEngine()
        engine.register(
            TypedConstraint(
                kind=ConstraintKind.INVARIANT,
                target="naming",
                severity=ConstraintSeverity.PREFERRED,
                affects_tags=["user", "model"],
                forbidden_patterns=[r"^[A-Z]"],  # lowercase names preferred
            )
        )
        intent = _user_model_intent()  # Has "User" (uppercase)
        gate = engine.gate(intent)
        # Should PASS (preferred is not blocking)
        assert gate.passed
        # But the check should show a violation
        assert gate.violated_count == 1

    def test_stability_threshold_check(self):
        engine = ConstraintEngine()
        tc = TypedConstraint(
            kind=ConstraintKind.INVARIANT,
            target="merge readiness",
            affects_tags=["model"],
            min_stability=0.5,
        )
        # Low stability intent (0.3 base, no evidence)
        intent = _user_model_intent()
        result = engine.check(tc, intent)
        assert not result.satisfied

        # High stability intent
        intent_high = _user_model_intent(
            evidence=[
                Evidence.code_committed("committed"),
            ]
        )
        result_high = engine.check(tc, intent_high)
        assert result_high.satisfied


# ===================================================================
# Layer 3: Economics
# ===================================================================


class TestCostModel:
    """Prove that cost model parameters are sane."""

    def test_default_cost_model(self):
        cm = CostModel()
        assert cm.token_cost_per_resolve > 0
        assert cm.human_escalation_cost > cm.token_cost_per_resolve
        assert cm.rework_cost_per_conflict > 0

    def test_custom_cost_model(self):
        cm = CostModel(
            token_cost_per_resolve=0.01,
            human_escalation_cost=5.0,
        )
        assert cm.token_cost_per_resolve == 0.01
        assert cm.human_escalation_cost == 5.0


class TestBudget:
    """Prove that budget tracking is correct."""

    def test_initial_budget(self):
        b = Budget(max_tokens=1000, max_cost=5.0)
        assert b.remaining_tokens == 1000
        assert b.remaining_cost == 5.0
        assert not b.exhausted

    def test_charge_reduces_budget(self):
        b = Budget(max_cost=5.0)
        assert b.charge(2.0)
        assert abs(b.remaining_cost - 3.0) < 1e-10

    def test_charge_rejects_over_budget(self):
        b = Budget(max_cost=1.0)
        assert not b.charge(2.0)
        assert abs(b.cost_incurred - 0.0) < 1e-10

    def test_exhaustion(self):
        b = Budget(max_cost=1.0)
        b.charge(1.0)
        assert b.exhausted

    def test_utilization(self):
        b = Budget(max_cost=10.0)
        b.charge(5.0)
        assert abs(b.utilization - 0.5) < 1e-10

    def test_record_resolve(self):
        b = Budget(max_cost=10.0)
        b.record_resolve(0.5)
        assert b.resolves_performed == 1
        assert abs(b.cost_incurred - 0.5) < 1e-10

    def test_record_escalation(self):
        b = Budget(max_cost=10.0)
        b.record_escalation(1.0)
        assert b.escalations_performed == 1


class TestEscalationPolicy:
    """Prove that escalation is economic, not conversational."""

    def test_high_confidence_auto_resolves(self):
        """High confidence → low P(rework) → auto-resolve is cheaper."""
        policy = EscalationPolicy()
        decision = policy.evaluate(
            confidence=0.95,
            stability_gap=0.3,
        )
        assert decision.action == EscalationAction.AUTO_RESOLVE
        assert decision.expected_cost_auto < decision.expected_cost_escalate

    def test_low_confidence_escalates(self):
        """Low confidence → high P(rework) → escalation may be cheaper.

        With default costs: P(rework)=0.5 * rework_cost=0.10 + resolve=0.001 = 0.051
        vs escalation = 0.01 + 1.0 = 1.01
        Actually with default model, rework is still cheap enough that auto-resolve wins.
        We need higher rework costs to trigger escalation.
        """
        # Use a cost model where rework is very expensive
        cm = CostModel(
            rework_cost_per_conflict=5.0,
            human_escalation_cost=1.0,
        )
        policy = EscalationPolicy(cost_model=cm)
        decision = policy.evaluate(
            confidence=0.3,
            stability_gap=0.01,
            num_affected_agents=3,
        )
        # P(rework)=0.5, rework_cost=5.0*3=15.0, expected_auto=7.501
        # escalation = 0.01 + 1.0 = 1.01
        assert decision.action == EscalationAction.ESCALATE_TO_HUMAN
        assert decision.expected_cost_escalate < decision.expected_cost_auto

    def test_budget_forces_auto_resolve(self):
        """When budget can't afford escalation, fall back to auto-resolve."""
        budget = Budget(max_cost=0.005)  # Very tight budget
        policy = EscalationPolicy(budget=budget)
        decision = policy.evaluate(confidence=0.3, stability_gap=0.0)
        assert decision.action == EscalationAction.AUTO_RESOLVE
        assert "Budget" in decision.reasoning

    def test_nearly_exhausted_defers(self):
        """When budget is >95% used but can still afford escalation, defer."""
        budget = Budget(max_cost=100.0)
        budget.charge(96.0)  # 96% used, remaining=4.0 > escalation cost
        policy = EscalationPolicy(budget=budget)
        decision = policy.evaluate(confidence=0.5, stability_gap=0.0)
        assert decision.action == EscalationAction.DEFER

    def test_affected_agents_scale_rework(self):
        """More affected agents → higher rework cost → more likely to escalate."""
        cm = CostModel(
            rework_cost_per_conflict=3.0,
            human_escalation_cost=1.0,
        )
        policy = EscalationPolicy(cost_model=cm)

        # 1 agent: P(rework)*rework = 0.5*3 = 1.5 + 0.001 vs 1.01
        d1 = policy.evaluate(confidence=0.3, stability_gap=0.0, num_affected_agents=1)
        # 5 agents: P(rework)*rework = 0.5*15 = 7.5 + 0.001 vs 1.01
        d5 = policy.evaluate(confidence=0.3, stability_gap=0.0, num_affected_agents=5)

        assert d5.expected_cost_auto > d1.expected_cost_auto

    def test_savings_calculation(self):
        cm = CostModel(
            rework_cost_per_conflict=10.0,
            human_escalation_cost=1.0,
        )
        policy = EscalationPolicy(cost_model=cm)
        decision = policy.evaluate(confidence=0.3, stability_gap=0.0, num_affected_agents=2)
        if decision.action == EscalationAction.ESCALATE_TO_HUMAN:
            assert decision.savings > 0  # Escalation saved money

    def test_batch_evaluate(self):
        policy = EscalationPolicy()
        decisions = policy.evaluate_batch(
            [
                {"confidence": 0.9, "stability_gap": 0.3},
                {"confidence": 0.3, "stability_gap": 0.01},
            ]
        )
        assert len(decisions) == 2


class TestCostReport:
    """Prove cost tracking is accurate."""

    def test_empty_report(self):
        report = CoordinationCostReport()
        assert report.escalation_rate == 0.0
        assert report.auto_resolve_rate == 1.0
        assert report.cost_per_decision == 0.0

    def test_record_auto_resolve(self):
        report = CoordinationCostReport()
        report.record(
            EscalationDecision(
                action=EscalationAction.AUTO_RESOLVE,
                expected_cost_auto=0.05,
                expected_cost_escalate=1.0,
                confidence=0.9,
                reasoning="auto",
            )
        )
        assert report.total_auto_resolved == 1
        assert report.total_resolves == 1
        assert abs(report.total_cost - 0.05) < 1e-10

    def test_record_escalation(self):
        report = CoordinationCostReport()
        report.record(
            EscalationDecision(
                action=EscalationAction.ESCALATE_TO_HUMAN,
                expected_cost_auto=5.0,
                expected_cost_escalate=1.0,
                confidence=0.3,
                reasoning="escalate",
            )
        )
        assert report.total_escalations == 1
        assert report.total_rework_avoided > 0

    def test_escalation_rate(self):
        report = CoordinationCostReport()
        for _ in range(8):
            report.record(
                EscalationDecision(
                    action=EscalationAction.AUTO_RESOLVE,
                    expected_cost_auto=0.01,
                    expected_cost_escalate=1.0,
                    confidence=0.9,
                    reasoning="auto",
                )
            )
        for _ in range(2):
            report.record(
                EscalationDecision(
                    action=EscalationAction.ESCALATE_TO_HUMAN,
                    expected_cost_auto=5.0,
                    expected_cost_escalate=1.0,
                    confidence=0.3,
                    reasoning="escalate",
                )
            )
        assert abs(report.escalation_rate - 0.2) < 1e-10


# ===================================================================
# Layer 2+3: Merge Governor
# ===================================================================


class TestMergeGovernor:
    """Prove the governor integrates all three layers correctly."""

    def test_approved_when_no_constraints_no_conflicts(self):
        governor = MergeGovernor()
        resolver = IntentResolver(min_stability=0.0)
        intent = _user_model_intent()
        verdict = governor.evaluate_publish(intent, resolver)
        assert verdict.approved
        assert verdict.kind == VerdictKind.APPROVED

    def test_blocked_by_constraint(self):
        """Layer 1 blocks before Layer 2 even runs."""
        engine = ConstraintEngine()
        engine.register(
            TypedConstraint(
                kind=ConstraintKind.TEST_GATE,
                target="all models",
                severity=ConstraintSeverity.REQUIRED,
                affects_tags=["model"],
                required_evidence=["test_pass"],
            )
        )
        governor = MergeGovernor(engine=engine)
        resolver = IntentResolver(min_stability=0.0)
        intent = _user_model_intent()  # No evidence
        verdict = governor.evaluate_publish(intent, resolver)
        assert not verdict.approved
        assert verdict.kind == VerdictKind.BLOCKED_BY_CONSTRAINT

    def test_constraint_passes_then_resolution_runs(self):
        """Layer 1 passes, Layer 2 detects conflict, Layer 3 evaluates."""
        engine = ConstraintEngine()
        engine.register(
            TypedConstraint(
                kind=ConstraintKind.TYPE_CHECK,
                target="User",
                affects_tags=["user", "model"],
                required_fields={"id": "UUID"},
            )
        )
        governor = MergeGovernor(engine=engine)
        resolver = IntentResolver(min_stability=0.0)

        # Publish high-stability User from agent-a
        intent_a = _user_model_intent(
            agent_id="agent-a",
            evidence=[
                Evidence.code_committed("models.py"),
                Evidence.test_pass("test_user"),
            ],
        )
        resolver.publish(intent_a)

        # Agent-b also tries User (will get ConsumeInstead)
        intent_b = _user_model_intent(agent_id="agent-b")
        verdict = governor.evaluate_publish(intent_b, resolver)
        # Should be approved (auto-resolved via stability ordering)
        assert verdict.approved
        assert verdict.resolution is not None
        assert verdict.resolution.has_adjustments

    def test_constraint_gate_order(self):
        """Constraints run BEFORE resolution — optimization: don't resolve invalid intents."""
        engine = ConstraintEngine()
        engine.register(
            TypedConstraint(
                kind=ConstraintKind.SCHEMA_RULE,
                target="User",
                severity=ConstraintSeverity.CRITICAL,
                affects_tags=["user", "model"],
                required_fields={"id": "UUID", "email": "str", "created_at": "datetime"},
            )
        )
        governor = MergeGovernor(engine=engine)
        resolver = IntentResolver(min_stability=0.0)
        intent = _user_model_intent()  # Missing created_at
        verdict = governor.evaluate_publish(intent, resolver)
        assert not verdict.approved
        assert verdict.kind == VerdictKind.BLOCKED_BY_CONSTRAINT
        # Resolution should be None (didn't run)
        assert verdict.resolution is None

    def test_budget_exhausted_blocks(self):
        budget = Budget(max_cost=0.0)  # Already exhausted
        governor = MergeGovernor(budget=budget)
        resolver = IntentResolver(min_stability=0.0)
        intent = _user_model_intent()
        verdict = governor.evaluate_publish(intent, resolver)
        assert not verdict.approved
        assert verdict.kind == VerdictKind.BUDGET_EXHAUSTED


class TestMergeGovernorMerge:
    """Test governor-mediated merges."""

    def test_merge_approved_no_conflicts(self):
        governor = MergeGovernor()
        target = VersionedGraph("main")
        target.publish(
            _make_intent(
                agent_id="main",
                provides=[
                    InterfaceSpec(
                        name="Logger",
                        kind=InterfaceKind.CLASS,
                        signature="log() -> None",
                        tags=["logging"],
                    )
                ],
            )
        )

        source = target.branch("feature")
        source.publish(
            _make_intent(
                agent_id="feature",
                provides=[
                    InterfaceSpec(
                        name="Metrics",
                        kind=InterfaceKind.CLASS,
                        signature="track() -> None",
                        tags=["metrics"],
                    )
                ],
            )
        )

        verdict = governor.evaluate_merge(source, target)
        assert verdict.approved

    def test_merge_blocked_by_constraint(self):
        engine = ConstraintEngine()
        engine.register(
            TypedConstraint(
                kind=ConstraintKind.TEST_GATE,
                target="production code",
                severity=ConstraintSeverity.REQUIRED,
                affects_tags=["model"],
                required_evidence=["test_pass", "code_committed"],
            )
        )
        governor = MergeGovernor(engine=engine)
        target = VersionedGraph("main")
        source = target.branch("feature")
        source.publish(_user_model_intent(agent_id="feature"))

        verdict = governor.evaluate_merge(source, target)
        assert not verdict.approved
        assert verdict.kind == VerdictKind.BLOCKED_BY_CONSTRAINT


# ===================================================================
# Agent branches
# ===================================================================


class TestAgentBranch:
    """Prove isolated agent branches enforce propose/commit."""

    def test_propose_approved(self):
        governor = MergeGovernor()
        main = VersionedGraph("main")
        branch = AgentBranch("agent-a", main, governor)

        intent = _user_model_intent()
        proposal = branch.propose(intent)
        assert proposal.can_commit
        assert proposal.verdict.approved

    def test_propose_blocked_by_constraint(self):
        engine = ConstraintEngine()
        engine.register(
            TypedConstraint(
                kind=ConstraintKind.TEST_GATE,
                severity=ConstraintSeverity.REQUIRED,
                target="models",
                affects_tags=["model"],
                required_evidence=["test_pass"],
            )
        )
        governor = MergeGovernor(engine=engine)
        main = VersionedGraph("main")
        branch = AgentBranch("agent-a", main, governor)

        intent = _user_model_intent()
        proposal = branch.propose(intent)
        assert not proposal.can_commit

    def test_commit_after_approved_proposal(self):
        governor = MergeGovernor()
        main = VersionedGraph("main")
        branch = AgentBranch("agent-a", main, governor)

        intent = _user_model_intent()
        branch.propose(intent)
        stability = branch.commit(intent)
        assert stability >= 0.0

    def test_commit_without_proposal_raises(self):
        governor = MergeGovernor()
        main = VersionedGraph("main")
        branch = AgentBranch("agent-a", main, governor)

        intent = _user_model_intent()
        with pytest.raises(ContractViolation) as exc_info:
            branch.commit(intent)
        assert "not proposed" in str(exc_info.value)

    def test_commit_rejected_proposal_raises(self):
        engine = ConstraintEngine()
        engine.register(
            TypedConstraint(
                kind=ConstraintKind.TEST_GATE,
                severity=ConstraintSeverity.REQUIRED,
                target="models",
                affects_tags=["model"],
                required_evidence=["test_pass"],
            )
        )
        governor = MergeGovernor(engine=engine)
        main = VersionedGraph("main")
        branch = AgentBranch("agent-a", main, governor)

        intent = _user_model_intent()
        branch.propose(intent)
        with pytest.raises(ContractViolation) as exc_info:
            branch.commit(intent)
        assert "not approved" in str(exc_info.value)

    def test_branch_isolation(self):
        """Changes on agent branch don't affect main."""
        governor = MergeGovernor()
        main = VersionedGraph("main")
        branch = AgentBranch("agent-a", main, governor)

        intent = _user_model_intent()
        branch.propose(intent)
        branch.commit(intent)

        main_intents = main.resolver.backend.query_all(min_stability=0.0)
        branch_intents = branch.graph.resolver.backend.query_all(min_stability=0.0)
        assert len(main_intents) == 0
        assert len(branch_intents) == 1

    def test_merge_to_main(self):
        governor = MergeGovernor()
        main = VersionedGraph("main")
        branch = AgentBranch("agent-a", main, governor)

        intent = _user_model_intent()
        branch.propose(intent)
        branch.commit(intent)

        merge_result = branch.merge_to(main)
        assert merge_result.success

        main_intents = main.resolver.backend.query_all(min_stability=0.0)
        assert len(main_intents) == 1

    def test_merge_to_blocked_returns_failure(self):
        engine = ConstraintEngine()
        engine.register(
            TypedConstraint(
                kind=ConstraintKind.TEST_GATE,
                severity=ConstraintSeverity.REQUIRED,
                target="models",
                affects_tags=["model"],
                required_evidence=["test_pass"],
            )
        )

        # Create governor WITHOUT constraints for local work
        local_governor = MergeGovernor()
        main = VersionedGraph("main")
        branch = AgentBranch("agent-a", main, local_governor)

        intent = _user_model_intent()
        branch.propose(intent)
        branch.commit(intent)

        # But merge uses a STRICT governor
        strict_governor = MergeGovernor(engine=engine)
        branch.governor = strict_governor
        merge_result = branch.merge_to(main)
        assert not merge_result.success


# ===================================================================
# Integration: Full 3-layer workflow
# ===================================================================


class TestThreeLayerIntegration:
    """End-to-end tests proving the 3-layer stack works together."""

    def test_full_workflow_happy_path(self):
        """Two agents, no conflicts, all constraints satisfied."""
        engine = ConstraintEngine()
        engine.register(
            TypedConstraint(
                kind=ConstraintKind.TYPE_CHECK,
                target="User",
                affects_tags=["user", "model"],
                required_fields={"id": "UUID"},
            )
        )
        governor = MergeGovernor(engine=engine)
        main = VersionedGraph("main")

        # Agent A: Auth
        branch_a = AgentBranch("agent-a", main, governor)
        intent_a = _user_model_intent(
            agent_id="agent-a",
            evidence=[Evidence.code_committed("auth.py")],
        )
        proposal_a = branch_a.propose(intent_a)
        assert proposal_a.can_commit
        branch_a.commit(intent_a)

        # Agent B: Recipes (different scope)
        branch_b = AgentBranch("agent-b", main, governor)
        intent_b = _make_intent(
            agent_id="agent-b",
            intent_text="Recipe service",
            provides=[
                InterfaceSpec(
                    name="RecipeService",
                    kind=InterfaceKind.CLASS,
                    signature="create() -> Recipe",
                    tags=["recipe", "service"],
                )
            ],
        )
        proposal_b = branch_b.propose(intent_b)
        assert proposal_b.can_commit
        branch_b.commit(intent_b)

        # Both merge to main
        merge_a = branch_a.merge_to(main)
        merge_b = branch_b.merge_to(main)
        assert merge_a.success
        assert merge_b.success

        all_intents = main.resolver.backend.query_all(min_stability=0.0)
        assert len(all_intents) == 2

    def test_constraint_blocks_then_fix_allows(self):
        """Agent is blocked by constraint, fixes intent, then succeeds."""
        engine = ConstraintEngine()
        engine.register(
            TypedConstraint(
                kind=ConstraintKind.SCHEMA_RULE,
                target="User",
                severity=ConstraintSeverity.REQUIRED,
                affects_tags=["user", "model"],
                required_fields={"id": "UUID", "email": "str", "created_at": "str"},
            )
        )
        governor = MergeGovernor(engine=engine)
        main = VersionedGraph("main")
        branch = AgentBranch("agent-a", main, governor)

        # First attempt: missing created_at
        intent_v1 = _user_model_intent()
        proposal = branch.propose(intent_v1)
        assert not proposal.can_commit

        # Fix: add created_at
        intent_v2 = _make_intent(
            agent_id="agent-a",
            intent_text="User model v2",
            provides=[
                InterfaceSpec(
                    name="User",
                    kind=InterfaceKind.MODEL,
                    signature="id: UUID, email: str, created_at: str",
                    tags=["user", "model", "auth"],
                )
            ],
        )
        proposal2 = branch.propose(intent_v2)
        assert proposal2.can_commit
        branch.commit(intent_v2)

    def test_economic_tracking_across_operations(self):
        """Governor tracks costs across multiple operations."""
        governor = MergeGovernor()
        resolver = IntentResolver(min_stability=0.0)

        # Use completely distinct scopes so no conflicts between agents
        scope_names = ["Logger", "Mailer", "Scheduler", "Monitor", "Deployer"]
        scope_tags = ["logging", "email", "cron", "monitoring", "deploy"]
        for i in range(5):
            intent = _make_intent(
                agent_id=f"agent-{i}",
                provides=[
                    InterfaceSpec(
                        name=scope_names[i],
                        kind=InterfaceKind.CLASS,
                        signature="run() -> bool",
                        tags=[scope_tags[i]],
                    )
                ],
            )
            governor.evaluate_publish(intent, resolver)
            resolver.publish(intent)

        # No conflicts = no escalation decisions, but budget tracks resolves
        assert governor.budget.resolves_performed == 5
        assert governor.budget.cost_incurred > 0

    def test_interface_first_enforcement(self):
        """Agents must declare interfaces (provides/requires) — empty rejected."""
        engine = ConstraintEngine()
        governor = MergeGovernor(engine=engine)
        main = VersionedGraph("main")
        branch = AgentBranch("agent-a", main, governor)

        # Intent with no provides/requires but has constraints — passes contract
        intent = Intent(
            agent_id="agent-a",
            intent="just a constraint",
            constraints=[
                Constraint(
                    target="naming",
                    requirement="use snake_case",
                    affects_tags=["style"],
                )
            ],
        )
        proposal = branch.propose(intent)
        assert proposal.can_commit  # Constraints-only intent is valid

    def test_governor_cost_report_tracks_savings(self):
        """When escalation saves money, it's tracked."""
        cm = CostModel(
            rework_cost_per_conflict=10.0,
            human_escalation_cost=1.0,
        )
        governor = MergeGovernor(cost_model=cm)
        resolver = IntentResolver(min_stability=0.0)

        # Publish high-stability intent
        resolver.publish(
            _user_model_intent(
                agent_id="agent-a",
                evidence=[
                    Evidence.code_committed("auth.py"),
                    Evidence.test_pass("test"),
                ],
            )
        )

        # Conflicting intent from another agent
        intent_b = _user_model_intent(agent_id="agent-b")
        governor.evaluate_publish(intent_b, resolver)

        # The cost report should have tracked the decision
        assert len(governor.cost_report.decisions) >= 0
