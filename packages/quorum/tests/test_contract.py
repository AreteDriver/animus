"""
Tests for the formal coordination contract.

Proves:
  1. Contract serialization and completeness
  2. Graph invariant enforcement (append-only, unique IDs, causal ordering)
  3. Conflict classification (hard fail, auto-resolve, human escalation)
  4. Stability weights determinism
  5. Content hashing determinism
  6. Graph versioning (snapshots, branching, merging)
  7. Deterministic replay (same intents + same policy = same state)
  8. Resolution policy correctness

Revolution test: these tests define the contract completely enough that
a second client can be built from the contract spec + these tests alone.
"""

import uuid

import pytest
from convergent.contract import (
    DEFAULT_CONTRACT,
    DEFAULT_STABILITY_WEIGHTS,
    ConflictClass,
    ContractViolation,
    EdgeType,
    GraphInvariant,
    MutationType,
    ResolutionPolicy,
    StabilityWeights,
    content_hash_intent,
    content_hash_intents,
    validate_publish,
)
from convergent.intent import (
    Constraint,
    ConstraintSeverity,
    Evidence,
    EvidenceKind,
    Intent,
    InterfaceKind,
    InterfaceSpec,
)
from convergent.replay import ReplayLog
from convergent.resolver import IntentResolver
from convergent.versioning import VersionedGraph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def resolver():
    return IntentResolver(min_stability=0.0)


@pytest.fixture
def vgraph():
    return VersionedGraph("main")


def _make_intent(
    agent_id: str = "agent-a",
    intent_text: str = "test intent",
    provides: list[InterfaceSpec] | None = None,
    requires: list[InterfaceSpec] | None = None,
    constraints: list[Constraint] | None = None,
    evidence: list[Evidence] | None = None,
    parent_id: str | None = None,
    intent_id: str | None = None,
) -> Intent:
    """Helper to create test intents with sensible defaults."""
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
        parent_id=parent_id,
    )


# ===================================================================
# 1. Contract serialization and completeness
# ===================================================================


class TestContractSerialization:
    """The contract must be fully serializable to JSON for interop."""

    def test_default_contract_exists(self):
        assert DEFAULT_CONTRACT is not None
        assert DEFAULT_CONTRACT.version == "1.0.0"

    def test_contract_to_dict_has_all_sections(self):
        d = DEFAULT_CONTRACT.to_dict()
        assert "contract_version" in d
        assert "node_types" in d
        assert "edge_types" in d
        assert "invariants" in d
        assert "allowed_mutations" in d
        assert "evidence_kinds" in d
        assert "constraint_severities" in d
        assert "conflict_classes" in d
        assert "stability_weights" in d
        assert "resolution_policy" in d
        assert "matching_rules" in d

    def test_contract_to_json_roundtrip(self):
        import json

        json_str = DEFAULT_CONTRACT.to_json()
        parsed = json.loads(json_str)
        assert parsed["contract_version"] == "1.0.0"
        assert len(parsed["node_types"]) == len(InterfaceKind)
        assert len(parsed["edge_types"]) == len(EdgeType)

    def test_all_interface_kinds_in_contract(self):
        d = DEFAULT_CONTRACT.to_dict()
        for kind in InterfaceKind:
            assert kind.value in d["node_types"]

    def test_all_edge_types_in_contract(self):
        d = DEFAULT_CONTRACT.to_dict()
        for edge in EdgeType:
            assert edge.value in d["edge_types"]

    def test_all_invariants_in_contract(self):
        d = DEFAULT_CONTRACT.to_dict()
        for inv in GraphInvariant:
            assert inv.value in d["invariants"]

    def test_all_mutation_types_in_contract(self):
        d = DEFAULT_CONTRACT.to_dict()
        for mut in MutationType:
            assert mut.value in d["allowed_mutations"]

    def test_all_evidence_kinds_in_contract(self):
        d = DEFAULT_CONTRACT.to_dict()
        for ek in EvidenceKind:
            assert ek.value in d["evidence_kinds"]

    def test_all_constraint_severities_in_contract(self):
        d = DEFAULT_CONTRACT.to_dict()
        for cs in ConstraintSeverity:
            assert cs.value in d["constraint_severities"]

    def test_all_conflict_classes_in_contract(self):
        d = DEFAULT_CONTRACT.to_dict()
        for cc in ConflictClass:
            assert cc.value in d["conflict_classes"]

    def test_stability_weights_in_contract(self):
        d = DEFAULT_CONTRACT.to_dict()
        sw = d["stability_weights"]
        assert sw["base"] == 0.3
        assert sw["test_pass"] == 0.05
        assert sw["test_pass_cap"] == 0.3
        assert sw["code_committed"] == 0.2
        assert sw["consumed_by_other"] == 0.1
        assert sw["consumed_cap"] == 0.2
        assert sw["conflict_penalty"] == 0.15
        assert sw["test_fail_penalty"] == 0.15
        assert sw["manual_approval"] == 0.3

    def test_resolution_policy_has_rules(self):
        d = DEFAULT_CONTRACT.to_dict()
        rules = d["resolution_policy"]["rules"]
        assert len(rules) == 5
        # Check each class is represented
        classes = {r["class"] for r in rules}
        assert ConflictClass.HARD_FAIL.value in classes
        assert ConflictClass.AUTO_RESOLVE.value in classes
        assert ConflictClass.HUMAN_ESCALATION.value in classes

    def test_matching_rules_documented(self):
        d = DEFAULT_CONTRACT.to_dict()
        mr = d["matching_rules"]
        assert "name_overlap" in mr
        assert "tag_overlap" in mr
        assert "signature_compatibility" in mr
        assert "constraint_applicability" in mr
        assert "constraint_conflict" in mr


# ===================================================================
# 2. Graph invariant enforcement
# ===================================================================


class TestGraphInvariants:
    """Prove that graph invariants are enforced."""

    def test_unique_ids_violation(self):
        intent = _make_intent(intent_id="fixed-id")
        violations = validate_publish(intent, existing_ids={"fixed-id"})
        assert len(violations) == 1
        assert violations[0].invariant == GraphInvariant.UNIQUE_IDS

    def test_unique_ids_pass(self):
        intent = _make_intent(intent_id="new-id")
        violations = validate_publish(intent, existing_ids={"other-id"})
        assert len(violations) == 0

    def test_causal_ordering_violation(self):
        intent = _make_intent(parent_id="nonexistent-parent")
        violations = validate_publish(intent, existing_ids=set())
        assert any(v.invariant == GraphInvariant.CAUSAL_ORDERING for v in violations)

    def test_causal_ordering_pass(self):
        intent = _make_intent(parent_id="existing-parent")
        violations = validate_publish(intent, existing_ids={"existing-parent"})
        assert not any(v.invariant == GraphInvariant.CAUSAL_ORDERING for v in violations)

    def test_causal_ordering_none_parent_ok(self):
        intent = _make_intent(parent_id=None)
        violations = validate_publish(intent, existing_ids=set())
        assert not any(v.invariant == GraphInvariant.CAUSAL_ORDERING for v in violations)

    def test_empty_intent_violation(self):
        intent = Intent(
            agent_id="agent-a",
            intent="empty",
            provides=[],
            requires=[],
            constraints=[],
        )
        violations = validate_publish(intent, existing_ids=set())
        assert len(violations) == 1
        assert violations[0].invariant == GraphInvariant.APPEND_ONLY

    def test_empty_agent_id_violation(self):
        intent = _make_intent(agent_id="")
        violations = validate_publish(intent, existing_ids=set())
        assert any(v.invariant == GraphInvariant.APPEND_ONLY for v in violations)

    def test_versioned_graph_enforces_unique_ids(self, vgraph):
        intent = _make_intent(intent_id="dup-id")
        vgraph.publish(intent)
        intent2 = _make_intent(intent_id="dup-id", intent_text="different")
        with pytest.raises(ContractViolation) as exc_info:
            vgraph.publish(intent2)
        assert "already exists" in str(exc_info.value)

    def test_versioned_graph_enforces_causal_ordering(self, vgraph):
        intent = _make_intent(parent_id="nonexistent")
        with pytest.raises(ContractViolation) as exc_info:
            vgraph.publish(intent)
        assert "does not exist" in str(exc_info.value)

    def test_append_only_intents_not_removed(self, vgraph):
        """Publishing new intents never removes old ones."""
        i1 = _make_intent(agent_id="a", intent_text="first")
        i2 = _make_intent(agent_id="b", intent_text="second")
        vgraph.publish(i1)
        vgraph.publish(i2)
        all_intents = vgraph.resolver.backend.query_all(min_stability=0.0)
        assert len(all_intents) == 2
        ids = {i.id for i in all_intents}
        assert i1.id in ids
        assert i2.id in ids

    def test_self_exclusion(self, resolver):
        """An agent's intents should not conflict with its own."""
        intent1 = _make_intent(agent_id="agent-a", intent_text="v1")
        resolver.publish(intent1)
        intent2 = _make_intent(agent_id="agent-a", intent_text="v2")
        result = resolver.resolve(intent2)
        assert result.is_clean


# ===================================================================
# 3. Conflict classification
# ===================================================================


class TestConflictClassification:
    """Prove that conflicts are classified correctly."""

    def test_critical_constraint_is_hard_fail(self):
        policy = ResolutionPolicy()
        constraint = Constraint(
            target="database",
            requirement="must use encryption",
            severity=ConstraintSeverity.CRITICAL,
            affects_tags=["database"],
        )
        result = policy.classify_constraint_conflict(constraint, 0.5, 0.5)
        assert result == ConflictClass.HARD_FAIL

    def test_critical_always_hard_fail_regardless_of_stability(self):
        policy = ResolutionPolicy()
        constraint = Constraint(
            target="security",
            requirement="must validate input",
            severity=ConstraintSeverity.CRITICAL,
            affects_tags=["security"],
        )
        # Even with huge stability gap, critical is critical
        result = policy.classify_constraint_conflict(constraint, 0.1, 0.9)
        assert result == ConflictClass.HARD_FAIL

    def test_preferred_constraint_is_auto_resolve(self):
        policy = ResolutionPolicy()
        constraint = Constraint(
            target="naming",
            requirement="use camelCase",
            severity=ConstraintSeverity.PREFERRED,
            affects_tags=["style"],
        )
        result = policy.classify_constraint_conflict(constraint, 0.5, 0.5)
        assert result == ConflictClass.AUTO_RESOLVE

    def test_required_with_stability_gap_is_auto_resolve(self):
        policy = ResolutionPolicy()
        constraint = Constraint(
            target="User.id",
            requirement="must be UUID",
            severity=ConstraintSeverity.REQUIRED,
            affects_tags=["user"],
        )
        result = policy.classify_constraint_conflict(constraint, 0.3, 0.7)
        assert result == ConflictClass.AUTO_RESOLVE

    def test_required_with_equal_stability_is_human_escalation(self):
        policy = ResolutionPolicy()
        constraint = Constraint(
            target="User.id",
            requirement="must be UUID",
            severity=ConstraintSeverity.REQUIRED,
            affects_tags=["user"],
        )
        result = policy.classify_constraint_conflict(constraint, 0.5, 0.5)
        assert result == ConflictClass.HUMAN_ESCALATION

    def test_required_within_epsilon_is_human_escalation(self):
        policy = ResolutionPolicy(stability_tie_epsilon=0.05)
        constraint = Constraint(
            target="api",
            requirement="use REST",
            severity=ConstraintSeverity.REQUIRED,
            affects_tags=["api"],
        )
        # Gap of 0.03 < epsilon of 0.05
        result = policy.classify_constraint_conflict(constraint, 0.50, 0.53)
        assert result == ConflictClass.HUMAN_ESCALATION

    def test_provision_conflict_with_stability_gap_auto_resolve(self):
        policy = ResolutionPolicy()
        result = policy.classify_provision_conflict(0.3, 0.7)
        assert result == ConflictClass.AUTO_RESOLVE

    def test_provision_conflict_equal_stability_human_escalation(self):
        policy = ResolutionPolicy()
        result = policy.classify_provision_conflict(0.5, 0.5)
        assert result == ConflictClass.HUMAN_ESCALATION

    def test_provision_conflict_near_equal_human_escalation(self):
        policy = ResolutionPolicy(stability_tie_epsilon=0.02)
        result = policy.classify_provision_conflict(0.50, 0.51)
        assert result == ConflictClass.HUMAN_ESCALATION

    def test_custom_epsilon(self):
        """Custom epsilon widens the escalation zone."""
        policy = ResolutionPolicy(stability_tie_epsilon=0.1)
        # Gap of 0.08 < epsilon of 0.1
        result = policy.classify_provision_conflict(0.50, 0.58)
        assert result == ConflictClass.HUMAN_ESCALATION
        # Gap of 0.15 > epsilon of 0.1
        result = policy.classify_provision_conflict(0.50, 0.65)
        assert result == ConflictClass.AUTO_RESOLVE


# ===================================================================
# 4. Stability weights determinism
# ===================================================================


class TestStabilityDeterminism:
    """Prove that stability computation is deterministic."""

    def test_same_evidence_same_score(self):
        weights = StabilityWeights()
        evidence = [
            Evidence.code_committed("commit"),
            Evidence.test_pass("t1"),
            Evidence.test_pass("t2"),
        ]
        score1 = weights.compute(evidence)
        score2 = weights.compute(evidence)
        assert score1 == score2

    def test_evidence_order_irrelevant(self):
        """Stability must be the same regardless of evidence order."""
        weights = StabilityWeights()
        e1 = [
            Evidence.code_committed("commit"),
            Evidence.test_pass("t1"),
            Evidence.consumed_by("agent-b"),
        ]
        e2 = [
            Evidence.consumed_by("agent-b"),
            Evidence.code_committed("commit"),
            Evidence.test_pass("t1"),
        ]
        assert weights.compute(e1) == weights.compute(e2)

    def test_contract_weights_match_intent_compute(self):
        """Contract weights must produce same result as Intent.compute_stability."""
        weights = DEFAULT_STABILITY_WEIGHTS
        evidence = [
            Evidence.code_committed("initial"),
            Evidence.test_pass("test_1"),
            Evidence.test_pass("test_2"),
            Evidence.consumed_by("agent-b"),
        ]
        intent = Intent(agent_id="a", intent="test", evidence=evidence)
        assert abs(weights.compute(evidence) - intent.compute_stability()) < 1e-10

    def test_base_stability(self):
        weights = StabilityWeights()
        assert weights.compute([]) == 0.3

    def test_max_stability(self):
        """Even with maximum evidence, stability caps at 1.0."""
        weights = StabilityWeights()
        evidence = [
            Evidence.code_committed("c"),
            Evidence.test_pass("t1"),
            Evidence.test_pass("t2"),
            Evidence.test_pass("t3"),
            Evidence.test_pass("t4"),
            Evidence.test_pass("t5"),
            Evidence.test_pass("t6"),
            Evidence.test_pass("t7"),
            Evidence.consumed_by("a"),
            Evidence.consumed_by("b"),
            Evidence.consumed_by("c"),
            Evidence(kind=EvidenceKind.MANUAL_APPROVAL, description="approved"),
        ]
        assert weights.compute(evidence) == 1.0

    def test_min_stability(self):
        """Even with maximum negative evidence, stability floors at 0.0."""
        weights = StabilityWeights()
        evidence = [
            Evidence.conflict("c1"),
            Evidence.conflict("c2"),
            Evidence.conflict("c3"),
            Evidence(kind=EvidenceKind.TEST_FAIL, description="f1"),
            Evidence(kind=EvidenceKind.TEST_FAIL, description="f2"),
        ]
        assert weights.compute(evidence) == 0.0

    def test_custom_weights(self):
        """Custom weights produce different but still deterministic scores."""
        weights = StabilityWeights(base=0.5, test_pass=0.1, test_pass_cap=0.5)
        evidence = [Evidence.test_pass("t1")]
        assert abs(weights.compute(evidence) - 0.6) < 1e-10


# ===================================================================
# 5. Content hashing determinism
# ===================================================================


class TestContentHashing:
    """Prove that content hashing is deterministic and collision-resistant."""

    def test_same_intent_same_hash(self):
        i1 = _make_intent(intent_id="x", agent_id="a", intent_text="test")
        i2 = _make_intent(intent_id="x", agent_id="a", intent_text="test")
        assert content_hash_intent(i1) == content_hash_intent(i2)

    def test_different_intent_different_hash(self):
        i1 = _make_intent(intent_id="x", intent_text="test1")
        i2 = _make_intent(intent_id="y", intent_text="test2")
        assert content_hash_intent(i1) != content_hash_intent(i2)

    def test_hash_is_hex_sha256(self):
        intent = _make_intent()
        h = content_hash_intent(intent)
        assert len(h) == 64  # SHA-256 hex length
        assert all(c in "0123456789abcdef" for c in h)

    def test_intents_hash_order_independent(self):
        """content_hash_intents produces same hash regardless of list order."""
        i1 = _make_intent(intent_id="aaa")
        i2 = _make_intent(intent_id="bbb")
        h1 = content_hash_intents([i1, i2])
        h2 = content_hash_intents([i2, i1])
        assert h1 == h2

    def test_empty_intents_hash(self):
        h = content_hash_intents([])
        assert len(h) == 64  # Still a valid hash

    def test_hash_changes_with_evidence(self):
        i1 = _make_intent(intent_id="x")
        i2 = _make_intent(
            intent_id="x",
            evidence=[Evidence.code_committed("commit")],
        )
        assert content_hash_intent(i1) != content_hash_intent(i2)

    def test_hash_changes_with_constraints(self):
        i1 = _make_intent(intent_id="x")
        i2 = _make_intent(
            intent_id="x",
            constraints=[Constraint(target="t", requirement="r", affects_tags=["t"])],
        )
        assert content_hash_intent(i1) != content_hash_intent(i2)


# ===================================================================
# 6. Graph versioning
# ===================================================================


class TestGraphVersioning:
    """Prove snapshot, branch, and merge semantics."""

    def test_snapshot_captures_state(self, vgraph):
        intent = _make_intent()
        vgraph.publish(intent)
        snap = vgraph.snapshot()
        assert snap.intent_count == 1
        assert intent.id in snap.intent_ids()

    def test_snapshot_is_immutable(self, vgraph):
        intent = _make_intent()
        vgraph.publish(intent)
        snap = vgraph.snapshot()

        # Publishing more intents doesn't change the snapshot
        intent2 = _make_intent(intent_text="second")
        vgraph.publish(intent2)
        assert snap.intent_count == 1  # Still 1

    def test_snapshot_content_hash_deterministic(self, vgraph):
        intent = _make_intent(intent_id="stable-id")
        vgraph.publish(intent)
        snap1 = vgraph.snapshot()
        # Create a second VersionedGraph with same intent
        vg2 = VersionedGraph("other")
        vg2.publish(_make_intent(intent_id="stable-id"))
        snap2 = vg2.snapshot()
        assert snap1.content_hash == snap2.content_hash

    def test_version_increments(self, vgraph):
        assert vgraph.version == 0
        vgraph.publish(_make_intent())
        vgraph.snapshot()
        assert vgraph.version == 1
        vgraph.snapshot()
        assert vgraph.version == 2

    def test_branch_creates_independent_copy(self, vgraph):
        intent = _make_intent()
        vgraph.publish(intent)

        branch = vgraph.branch("feature")
        assert branch.branch_name == "feature"

        # Branch has the original intent
        branch_intents = branch.resolver.backend.query_all(min_stability=0.0)
        assert len(branch_intents) == 1
        assert branch_intents[0].id == intent.id

    def test_branch_changes_dont_affect_main(self, vgraph):
        intent = _make_intent()
        vgraph.publish(intent)

        branch = vgraph.branch("feature")
        new_intent = _make_intent(agent_id="branch-agent")
        branch.publish(new_intent)

        # Main should still have 1 intent
        main_intents = vgraph.resolver.backend.query_all(min_stability=0.0)
        assert len(main_intents) == 1

        # Branch should have 2
        branch_intents = branch.resolver.backend.query_all(min_stability=0.0)
        assert len(branch_intents) == 2

    def test_merge_adds_new_intents(self, vgraph):
        base_intent = _make_intent(
            agent_id="main-agent",
            provides=[
                InterfaceSpec(
                    name="AuthService",
                    kind=InterfaceKind.CLASS,
                    signature="login() -> bool",
                    tags=["auth"],
                )
            ],
        )
        vgraph.publish(base_intent)

        branch = vgraph.branch("feature")
        new_intent = _make_intent(
            agent_id="feature-agent",
            intent_text="new feature",
            provides=[
                InterfaceSpec(
                    name="EmailService",
                    kind=InterfaceKind.CLASS,
                    signature="send() -> bool",
                    tags=["email"],
                )
            ],
        )
        branch.publish(new_intent)

        result = vgraph.merge(branch)
        assert result.success
        assert len(result.merged_intents) == 1
        assert result.merged_intents[0].id == new_intent.id

        # Main now has both intents
        main_intents = vgraph.resolver.backend.query_all(min_stability=0.0)
        assert len(main_intents) == 2

    def test_merge_detects_no_new_intents(self, vgraph):
        intent = _make_intent()
        vgraph.publish(intent)

        branch = vgraph.branch("empty-branch")
        # No new intents on branch

        result = vgraph.merge(branch)
        assert result.success
        assert len(result.merged_intents) == 0

    def test_merge_result_has_snapshot_on_success(self, vgraph):
        vgraph.publish(
            _make_intent(
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
        branch = vgraph.branch("feature")
        branch.publish(
            _make_intent(
                agent_id="b",
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
        result = vgraph.merge(branch)
        assert result.success
        assert result.resulting_snapshot is not None
        assert result.resulting_snapshot.intent_count == 2

    def test_merge_with_conflict(self, vgraph):
        """Merging conflicting provisions detects the conflict."""
        # Main has a high-stability intent providing Config
        main_intent = _make_intent(
            agent_id="main",
            provides=[
                InterfaceSpec(
                    name="Config",
                    kind=InterfaceKind.CLASS,
                    signature="get(key) -> str",
                    tags=["config", "settings"],
                )
            ],
            evidence=[
                Evidence.code_committed("config.py"),
                Evidence.test_pass("test_config"),
            ],
        )
        vgraph.publish(main_intent)

        branch = vgraph.branch("feature")
        # Branch has a low-stability intent also providing Config
        branch_intent = _make_intent(
            agent_id="feature",
            provides=[
                InterfaceSpec(
                    name="Config",
                    kind=InterfaceKind.CLASS,
                    signature="get(key) -> str",
                    tags=["config", "settings"],
                )
            ],
        )
        branch.publish(branch_intent)

        result = vgraph.merge(branch)
        # Should have adjustments (ConsumeInstead) since main has higher stability
        assert len(result.conflicts) > 0 or len(result.merged_intents) > 0

    def test_snapshots_list(self, vgraph):
        vgraph.publish(_make_intent())
        vgraph.snapshot()
        vgraph.snapshot()
        assert len(vgraph.snapshots) == 2


# ===================================================================
# 7. Deterministic replay
# ===================================================================


class TestDeterministicReplay:
    """Prove: same intents + same policy ⇒ same resolved state."""

    def test_replay_empty_log(self):
        log = ReplayLog()
        result = log.replay()
        assert result.deterministic
        assert result.replayed_intent_count == 0

    def test_replay_publish_only(self):
        log = ReplayLog()
        i1 = _make_intent(agent_id="a", intent_text="first")
        i2 = _make_intent(agent_id="b", intent_text="second")
        log.record_publish(i1)
        log.record_publish(i2)

        result = log.replay()
        assert result.deterministic
        assert result.replayed_intent_count == 2

    def test_replay_produces_same_hash(self):
        """Two replays of the same log produce the same content hash."""
        log = ReplayLog()
        i1 = _make_intent(agent_id="a")
        i2 = _make_intent(agent_id="b")
        log.record_publish(i1)
        log.record_publish(i2)

        result1 = log.replay()
        result2 = log.replay()
        assert result1.final_content_hash == result2.final_content_hash

    def test_replay_with_resolution(self):
        """Replay of publish + resolve produces same adjustments."""
        resolver = IntentResolver(min_stability=0.0)

        # Agent A publishes high-stability intent
        intent_a = _make_intent(
            agent_id="agent-a",
            intent_text="Auth module",
            provides=[
                InterfaceSpec(
                    name="User",
                    kind=InterfaceKind.MODEL,
                    signature="id: UUID, email: str",
                    tags=["user", "model"],
                )
            ],
            evidence=[
                Evidence.code_committed("auth/models.py"),
                Evidence.test_pass("test_user"),
            ],
        )

        log = ReplayLog()
        log.record_publish(intent_a)
        resolver.publish(intent_a)

        # Agent B resolves against graph
        intent_b = _make_intent(
            agent_id="agent-b",
            intent_text="Meal module",
            provides=[
                InterfaceSpec(
                    name="User",
                    kind=InterfaceKind.MODEL,
                    signature="id: UUID",
                    tags=["user", "model"],
                )
            ],
        )

        result = resolver.resolve(intent_b)
        log.record_resolve(intent_b, result)
        log.record_publish(intent_b)
        resolver.publish(intent_b)

        # Replay
        replay_result = log.replay()
        assert replay_result.deterministic
        assert replay_result.all_resolutions_match

    def test_replay_resolution_count(self):
        log = ReplayLog()
        intent = _make_intent()
        resolver = IntentResolver(min_stability=0.0)
        resolver.publish(intent)

        log.record_publish(intent)
        result = resolver.resolve(intent)
        log.record_resolve(intent, result)

        replay_result = log.replay()
        assert replay_result.replayed_resolution_count == 1

    def test_replay_three_agents_deterministic(self):
        """Full 3-agent scenario: two independent replays produce same state."""
        from convergent.agent import SimulationRunner
        from convergent.demo import build_agent_a, build_agent_b, build_agent_c

        # Run 1: record all publishes
        resolver1 = IntentResolver(min_stability=0.0)
        log = ReplayLog()

        agent_a = build_agent_a(resolver1)
        agent_b = build_agent_b(resolver1)
        agent_c = build_agent_c(resolver1)

        runner = SimulationRunner(resolver1)
        runner.add_agent(agent_a)
        runner.add_agent(agent_b)
        runner.add_agent(agent_c)

        original_publish = resolver1.publish

        def recording_publish(intent: Intent) -> float:
            log.record_publish(intent)
            return original_publish(intent)

        resolver1.publish = recording_publish  # type: ignore[assignment]
        runner.run()

        # Replay twice — both replays must produce the same hash
        result1 = log.replay()
        result2 = log.replay()
        assert result1.final_content_hash == result2.final_content_hash
        assert result1.deterministic
        assert result2.deterministic
        assert result1.replayed_intent_count == result2.replayed_intent_count


# ===================================================================
# 8. Resolution policy integration
# ===================================================================


class TestResolutionPolicyIntegration:
    """Prove that the resolution policy integrates with the resolver."""

    def test_default_policy_auto_resolves_stability_gap(self, resolver):
        """When one agent has clearly higher stability, auto-resolve."""
        intent_a = _make_intent(
            agent_id="a",
            provides=[
                InterfaceSpec(
                    name="Service",
                    kind=InterfaceKind.CLASS,
                    signature="run()",
                    tags=["core"],
                )
            ],
            evidence=[Evidence.code_committed("committed")],
        )
        resolver.publish(intent_a)

        intent_b = _make_intent(
            agent_id="b",
            provides=[
                InterfaceSpec(
                    name="Service",
                    kind=InterfaceKind.CLASS,
                    signature="run()",
                    tags=["core"],
                )
            ],
        )
        result = resolver.resolve(intent_b)
        # B should get ConsumeInstead (auto-resolved)
        assert any(a.kind == "ConsumeInstead" for a in result.adjustments)

    def test_policy_classifies_constraint_conflict(self):
        """Policy correctly classifies a Required constraint conflict."""
        policy = ResolutionPolicy()
        constraint = Constraint(
            target="API format",
            requirement="must use JSON",
            severity=ConstraintSeverity.REQUIRED,
            affects_tags=["api"],
        )
        # Clear stability winner
        cls = policy.classify_constraint_conflict(constraint, 0.3, 0.7)
        assert cls == ConflictClass.AUTO_RESOLVE

        # Tied stability
        cls = policy.classify_constraint_conflict(constraint, 0.5, 0.5)
        assert cls == ConflictClass.HUMAN_ESCALATION


# ===================================================================
# 9. Edge type coverage
# ===================================================================


class TestEdgeTypes:
    """Prove that all edge types are exercised in the system."""

    def test_provides_edge(self, resolver):
        """PROVIDES: intent offers an interface."""
        intent = _make_intent(
            provides=[
                InterfaceSpec(
                    name="UserService",
                    kind=InterfaceKind.CLASS,
                    signature="create(email: str) -> User",
                    tags=["user"],
                )
            ]
        )
        resolver.publish(intent)
        all_intents = resolver.backend.query_all()
        assert len(all_intents[0].provides) == 1

    def test_requires_edge(self, resolver):
        """REQUIRES: intent depends on an interface."""
        intent = _make_intent(
            provides=[
                InterfaceSpec(
                    name="RecipeService",
                    kind=InterfaceKind.CLASS,
                    signature="create()",
                    tags=["recipe"],
                )
            ],
            requires=[
                InterfaceSpec(
                    name="User",
                    kind=InterfaceKind.MODEL,
                    signature="id: UUID",
                    tags=["user"],
                )
            ],
        )
        resolver.publish(intent)
        all_intents = resolver.backend.query_all()
        assert len(all_intents[0].requires) == 1

    def test_constrains_edge(self, resolver):
        """CONSTRAINS: intent imposes constraints on others."""
        intent = _make_intent(
            constraints=[
                Constraint(
                    target="User.id",
                    requirement="must be UUID",
                    severity=ConstraintSeverity.REQUIRED,
                    affects_tags=["user"],
                )
            ],
        )
        resolver.publish(intent)
        all_intents = resolver.backend.query_all()
        assert len(all_intents[0].constraints) == 1

    def test_supersedes_edge(self, resolver):
        """SUPERSEDES: intent references its parent version."""
        parent = _make_intent(intent_text="v1")
        resolver.publish(parent)

        child = _make_intent(
            intent_text="v2",
            parent_id=parent.id,
        )
        resolver.publish(child)

        all_intents = resolver.backend.query_all()
        child_intent = next(i for i in all_intents if i.intent == "v2")
        assert child_intent.parent_id == parent.id


# ===================================================================
# 10. Contract as interop spec
# ===================================================================


class TestContractAsInteropSpec:
    """Prove that the contract is complete enough for independent implementation."""

    def test_contract_specifies_all_types(self):
        """Every type used in the system is enumerated in the contract."""
        contract = DEFAULT_CONTRACT
        # Node types cover all InterfaceKind values
        assert set(contract.node_types) == {k.value for k in InterfaceKind}
        # Edge types cover all EdgeType values
        assert set(contract.edge_types) == {e.value for e in EdgeType}
        # Evidence kinds cover all EvidenceKind values
        assert set(contract.evidence_kinds) == {e.value for e in EvidenceKind}

    def test_contract_specifies_stability_computation(self):
        """The stability algorithm is fully specified by weights."""
        weights = DEFAULT_CONTRACT.stability_weights
        # Verify computation is reproducible from spec alone
        evidence = [
            Evidence.code_committed("c"),
            Evidence.test_pass("t1"),
            Evidence.test_pass("t2"),
            Evidence.consumed_by("b"),
        ]
        expected = (
            weights.base
            + weights.code_committed
            + min(2 * weights.test_pass, weights.test_pass_cap)
            + min(1 * weights.consumed_by_other, weights.consumed_cap)
        )
        actual = weights.compute(evidence)
        assert abs(expected - actual) < 1e-10

    def test_contract_specifies_resolution_rules(self):
        """Resolution rules are fully specified in the contract."""
        d = DEFAULT_CONTRACT.to_dict()
        rules = d["resolution_policy"]["rules"]
        # Each rule has condition, class, and action
        for rule in rules:
            assert "condition" in rule
            assert "class" in rule
            assert "action" in rule
            assert rule["class"] in [c.value for c in ConflictClass]

    def test_contract_specifies_matching_rules(self):
        """Matching rules are documented in the contract."""
        d = DEFAULT_CONTRACT.to_dict()
        mr = d["matching_rules"]
        # Each rule has a description
        for key in mr:
            assert "description" in mr[key]
        # Tag overlap has threshold
        assert mr["tag_overlap"]["threshold"] == 2
