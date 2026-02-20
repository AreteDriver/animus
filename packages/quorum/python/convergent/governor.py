"""
Merge Governor — Policy layer unifying constraints, intent graph, and economics.

This is the integration point for the 3-layer stack:
  Layer 1 (Constraints): Hard truth — tests, types, schemas, security
  Layer 2 (Intent Graph): Shared decisions — interfaces, ownership, stability
  Layer 3 (Economics): Optimization — cost-of-rework, budget, escalation

The governor enforces a strict ordering:
  1. Constraint gate FIRST (reject invalid states deterministically)
  2. Intent resolution SECOND (align decisions via stability)
  3. Economic escalation THIRD (only escalate when cost-justified)

This prevents the graph from becoming untrustworthy (failure risk #1)
and ensures coordination overhead doesn't eat the gains (failure risk #5).

The governor also provides isolated agent branches with a propose/commit
workflow, enforcing "no code commit without declared interfaces/assumptions"
(mitigating failure risk #4: incentive misalignment).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from convergent.constraints import (
    ConstraintEngine,
    GateResult,
)
from convergent.contract import (
    ConflictClass,
    ContractViolation,
    GraphInvariant,
    ResolutionPolicy,
)
from convergent.economics import (
    Budget,
    CoordinationCostReport,
    CostModel,
    EscalationAction,
    EscalationDecision,
    EscalationPolicy,
)
from convergent.intent import (
    Intent,
    ResolutionResult,
)
from convergent.resolver import IntentResolver
from convergent.versioning import MergeResult, VersionedGraph

# ---------------------------------------------------------------------------
# Governor verdict
# ---------------------------------------------------------------------------


class VerdictKind:
    """Possible merge governor verdicts."""

    APPROVED = "approved"
    BLOCKED_BY_CONSTRAINT = "blocked_by_constraint"
    BLOCKED_BY_CONFLICT = "blocked_by_conflict"
    NEEDS_ESCALATION = "needs_escalation"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass
class GovernorVerdict:
    """The governor's decision on whether an operation can proceed.

    Includes constraint check results, resolution results, and
    economic analysis — full traceability for debugging.
    """

    kind: str
    approved: bool
    gate_result: GateResult | None = None
    resolution: ResolutionResult | None = None
    escalation_decisions: list[EscalationDecision] = field(default_factory=list)
    blocking_reasons: list[str] = field(default_factory=list)

    @property
    def needs_human(self) -> bool:
        return any(
            d.action == EscalationAction.ESCALATE_TO_HUMAN for d in self.escalation_decisions
        )


# ---------------------------------------------------------------------------
# Proposal result (for agent branches)
# ---------------------------------------------------------------------------


@dataclass
class ProposalResult:
    """Result of proposing an intent through the governor.

    An agent must propose before committing. The proposal runs all
    three layers and returns whether the intent can proceed.
    """

    intent: Intent
    verdict: GovernorVerdict
    can_commit: bool = False

    @property
    def blocking_reasons(self) -> list[str]:
        return self.verdict.blocking_reasons


# ---------------------------------------------------------------------------
# Merge governor
# ---------------------------------------------------------------------------


class MergeGovernor:
    """Policy layer governing publish and merge through the 3-layer stack.

    The governor is the single entry point for all graph mutations.
    It enforces:
      1. Constraint satisfaction (Layer 1 — hard truth)
      2. Intent alignment (Layer 2 — decisions)
      3. Economic escalation (Layer 3 — optimization)

    Usage:
        engine = ConstraintEngine()
        engine.register(TypedConstraint(...))

        governor = MergeGovernor(engine=engine)

        # Evaluate before publishing
        verdict = governor.evaluate_publish(intent, resolver)
        if verdict.approved:
            resolver.publish(intent)
        elif verdict.needs_human:
            # Surface to human with full context
            ...

        # Evaluate a merge
        verdict = governor.evaluate_merge(branch, target)
    """

    def __init__(
        self,
        engine: ConstraintEngine | None = None,
        policy: ResolutionPolicy | None = None,
        cost_model: CostModel | None = None,
        budget: Budget | None = None,
    ) -> None:
        self.engine = engine or ConstraintEngine()
        self.policy = policy or ResolutionPolicy()
        self.cost_model = cost_model or CostModel()
        self.budget = budget or Budget()
        self.escalation_policy = EscalationPolicy(self.cost_model, self.budget)
        self.cost_report = CoordinationCostReport()

    def evaluate_publish(
        self,
        intent: Intent,
        resolver: IntentResolver,
    ) -> GovernorVerdict:
        """Evaluate whether an intent can be published.

        Runs the 3-layer stack in order:
        1. Constraint gate (hard fail if violated)
        2. Resolution (detect conflicts)
        3. Economic evaluation (escalate or auto-resolve)
        """
        blocking: list[str] = []
        escalation_decisions: list[EscalationDecision] = []

        # --- Layer 1: Constraint gate ---
        gate_result = self.engine.gate(intent)
        if not gate_result.passed:
            return GovernorVerdict(
                kind=VerdictKind.BLOCKED_BY_CONSTRAINT,
                approved=False,
                gate_result=gate_result,
                blocking_reasons=gate_result.blocking_violations,
            )

        # --- Layer 2: Intent resolution ---
        resolution = resolver.resolve(intent)

        if resolution.conflicts:
            my_stability = intent.compute_stability()

            for conflict in resolution.conflicts:
                # Classify the conflict
                stability_gap = abs(conflict.their_stability - my_stability)
                conflict_class = self.policy.classify_provision_conflict(
                    my_stability, conflict.their_stability
                )

                if conflict_class == ConflictClass.HARD_FAIL:
                    blocking.append(f"Hard fail: {conflict.description}")
                    continue

                # --- Layer 3: Economic escalation ---
                decision = self.escalation_policy.evaluate(
                    confidence=conflict.confidence,
                    stability_gap=stability_gap,
                    num_affected_agents=1,
                )
                escalation_decisions.append(decision)
                self.cost_report.record(decision)

                if decision.action == EscalationAction.ESCALATE_TO_HUMAN:
                    blocking.append(
                        f"Escalation needed: {conflict.description} ({decision.reasoning})"
                    )
                elif decision.action == EscalationAction.BLOCK:
                    blocking.append(f"Blocked: {conflict.description}")

        if blocking:
            kind = (
                VerdictKind.NEEDS_ESCALATION
                if any(d.action == EscalationAction.ESCALATE_TO_HUMAN for d in escalation_decisions)
                else VerdictKind.BLOCKED_BY_CONFLICT
            )
            return GovernorVerdict(
                kind=kind,
                approved=False,
                gate_result=gate_result,
                resolution=resolution,
                escalation_decisions=escalation_decisions,
                blocking_reasons=blocking,
            )

        # Budget check
        if self.budget.exhausted:
            return GovernorVerdict(
                kind=VerdictKind.BUDGET_EXHAUSTED,
                approved=False,
                gate_result=gate_result,
                resolution=resolution,
                blocking_reasons=["Coordination budget exhausted"],
            )

        self.budget.record_resolve(self.cost_model.token_cost_per_resolve)

        return GovernorVerdict(
            kind=VerdictKind.APPROVED,
            approved=True,
            gate_result=gate_result,
            resolution=resolution,
            escalation_decisions=escalation_decisions,
        )

    def evaluate_merge(
        self,
        source: VersionedGraph,
        target: VersionedGraph,
    ) -> GovernorVerdict:
        """Evaluate whether a branch can be merged.

        Checks each new intent from source against target's constraints,
        then runs resolution for conflicts, then applies economics.
        """
        my_ids = {i.id for i in target.resolver.backend.query_all(min_stability=0.0)}
        their_intents = source.resolver.backend.query_all(min_stability=0.0)
        new_intents = [i for i in their_intents if i.id not in my_ids]
        new_intents.sort(key=lambda i: i.timestamp)

        blocking: list[str] = []
        escalation_decisions: list[EscalationDecision] = []
        all_gate_results: list[GateResult] = []

        for intent in new_intents:
            # Layer 1: Constraint gate each new intent
            gate = self.engine.gate(intent)
            all_gate_results.append(gate)
            if not gate.passed:
                for v in gate.blocking_violations:
                    blocking.append(f"Merge blocked: {v}")

        if blocking:
            return GovernorVerdict(
                kind=VerdictKind.BLOCKED_BY_CONSTRAINT,
                approved=False,
                gate_result=all_gate_results[0] if all_gate_results else None,
                blocking_reasons=blocking,
            )

        # Layer 2 + 3: Run the merge (which does resolution internally)
        # We don't actually merge yet — just evaluate
        for intent in new_intents:
            resolution = target.resolver.resolve(intent)

            if resolution.conflicts:
                my_stability = intent.compute_stability()
                for conflict in resolution.conflicts:
                    stability_gap = abs(conflict.their_stability - my_stability)

                    decision = self.escalation_policy.evaluate(
                        confidence=conflict.confidence,
                        stability_gap=stability_gap,
                    )
                    escalation_decisions.append(decision)
                    self.cost_report.record(decision)

                    if decision.action == EscalationAction.ESCALATE_TO_HUMAN:
                        blocking.append(f"Merge escalation: {conflict.description}")
                    elif decision.action == EscalationAction.BLOCK:
                        blocking.append(f"Merge blocked: {conflict.description}")

        if blocking:
            return GovernorVerdict(
                kind=VerdictKind.NEEDS_ESCALATION,
                approved=False,
                gate_result=all_gate_results[0] if all_gate_results else None,
                escalation_decisions=escalation_decisions,
                blocking_reasons=blocking,
            )

        return GovernorVerdict(
            kind=VerdictKind.APPROVED,
            approved=True,
            gate_result=all_gate_results[0] if all_gate_results else None,
            escalation_decisions=escalation_decisions,
        )


# ---------------------------------------------------------------------------
# Agent branch (isolated workspace with propose/commit)
# ---------------------------------------------------------------------------


class AgentBranch:
    """Isolated branch for a single agent with propose/commit workflow.

    Each agent works in isolation (Git-native concurrency). Before any
    intent is published, it must pass through the governor's 3-layer
    gate. This enforces "no code commit without declared interfaces"
    and makes intent publication mandatory via gating.

    Usage:
        governor = MergeGovernor(engine=engine)
        main = VersionedGraph("main")

        # Each agent gets an isolated branch
        branch = AgentBranch("agent-a", main, governor)

        # Propose (checks constraints + resolves + economics)
        proposal = branch.propose(intent)
        if proposal.can_commit:
            branch.commit(intent)

        # Merge back to main when done
        merge_result = branch.merge_to(main)
    """

    def __init__(
        self,
        agent_id: str,
        source: VersionedGraph,
        governor: MergeGovernor,
    ) -> None:
        self.agent_id = agent_id
        self.governor = governor
        self.graph = source.branch(f"agent/{agent_id}")
        self._proposals: dict[str, ProposalResult] = {}

    @property
    def branch_name(self) -> str:
        return self.graph.branch_name

    def propose(self, intent: Intent) -> ProposalResult:
        """Propose an intent through the 3-layer governor.

        The intent is NOT published yet — just evaluated.
        Call commit() to actually publish after approval.
        """
        verdict = self.governor.evaluate_publish(intent, self.graph.resolver)

        proposal = ProposalResult(
            intent=intent,
            verdict=verdict,
            can_commit=verdict.approved,
        )
        self._proposals[intent.id] = proposal
        return proposal

    def commit(self, intent: Intent) -> float:
        """Commit a previously approved intent.

        Raises ContractViolation if the intent wasn't proposed or
        wasn't approved.
        """
        proposal = self._proposals.get(intent.id)
        if proposal is None:
            raise ContractViolation(
                invariant=GraphInvariant.APPEND_ONLY,
                message=f"Intent '{intent.id}' was not proposed through the governor",
            )
        if not proposal.can_commit:
            raise ContractViolation(
                invariant=GraphInvariant.APPEND_ONLY,
                message=(
                    f"Intent '{intent.id}' was not approved. Reasons: {proposal.blocking_reasons}"
                ),
            )
        return self.graph.publish(intent)

    def merge_to(self, target: VersionedGraph) -> MergeResult:
        """Merge this agent's branch into the target.

        Runs the governor's merge evaluation first.
        """
        verdict = self.governor.evaluate_merge(self.graph, target)
        if not verdict.approved:
            # Return a failed MergeResult with the reasons
            return MergeResult(
                success=False,
                hard_failures=[
                    (Intent(agent_id=self.agent_id, intent="merge"), reason)
                    for reason in verdict.blocking_reasons
                ],
            )
        return target.merge(self.graph)
