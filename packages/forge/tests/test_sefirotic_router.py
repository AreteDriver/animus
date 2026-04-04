"""Tests for the Sefirotic Router — topology-aware delegation routing."""

import pytest

from animus_forge.agents.sefirotic_router import (
    EDGES,
    PILLAR_MAP,
    ROLE_MAP,
    Pillar,
    Sefirah,
    SefiroticRouter,
)


@pytest.fixture
def router():
    return SefiroticRouter()


class TestTopologyStructure:
    """Verify the Tree of Life graph structure is correctly encoded."""

    def test_22_edges(self):
        assert len(EDGES) == 22

    def test_10_sefirot(self):
        assert len(Sefirah) == 10

    def test_3_pillars(self):
        assert len(Pillar) == 3

    def test_all_sefirot_have_pillar(self):
        for s in Sefirah:
            assert s in PILLAR_MAP

    def test_tiferet_connected_to_all(self, router):
        """Tiferet (hub) must connect to all 9 other sefirot."""
        for s in Sefirah:
            if s != Sefirah.TIFERET:
                assert router.are_connected(Sefirah.TIFERET, s), (
                    f"Tiferet not connected to {s.name}"
                )

    def test_tiferet_degree_9(self, router):
        adj = router._adjacency[Sefirah.TIFERET]
        assert len(adj) == 9

    def test_keter_degree_3(self, router):
        adj = router._adjacency[Sefirah.KETER]
        assert len(adj) == 3

    def test_malkuth_degree_2(self, router):
        adj = router._adjacency[Sefirah.MALKUTH]
        assert len(adj) == 2

    def test_22nd_path_exists(self, router):
        """The symmetry-breaking 22nd path: Chokhmah -> Netzach."""
        assert router.are_connected(Sefirah.CHOKHMAH, Sefirah.NETZACH)

    def test_22nd_path_mirror_missing(self, router):
        """Binah -> Hod should NOT exist (asymmetry)."""
        assert not router.are_connected(Sefirah.BINAH, Sefirah.HOD)

    def test_diameter_is_2(self, router):
        """Max hop count between any two nodes should be 2."""
        max_hops = 0
        for a in Sefirah:
            for b in Sefirah:
                if a != b:
                    hops = router.hop_count(a, b)
                    assert hops > 0
                    max_hops = max(max_hops, hops)
        assert max_hops == 2


class TestPillarAssignment:
    """Verify pillar assignments match the spec."""

    def test_mercy_pillar(self):
        mercy = [s for s, p in PILLAR_MAP.items() if p == Pillar.MERCY]
        assert set(mercy) == {Sefirah.CHOKHMAH, Sefirah.CHESED, Sefirah.NETZACH}

    def test_severity_pillar(self):
        severity = [s for s, p in PILLAR_MAP.items() if p == Pillar.SEVERITY]
        assert set(severity) == {Sefirah.BINAH, Sefirah.GEVURAH, Sefirah.HOD}

    def test_equilibrium_pillar(self):
        equil = [s for s, p in PILLAR_MAP.items() if p == Pillar.EQUILIBRIUM]
        assert set(equil) == {
            Sefirah.KETER, Sefirah.TIFERET, Sefirah.YESOD, Sefirah.MALKUTH,
        }


class TestRouting:
    """Test routing decisions."""

    def test_builder_routes_to_chesed(self, router):
        decision = router.route("builder")
        assert decision.target == Sefirah.CHESED
        assert decision.source == Sefirah.TIFERET

    def test_tester_routes_to_binah(self, router):
        decision = router.route("tester")
        assert decision.target == Sefirah.BINAH

    def test_reviewer_routes_to_gevurah(self, router):
        decision = router.route("reviewer")
        assert decision.target == Sefirah.GEVURAH

    def test_documenter_routes_to_netzach(self, router):
        decision = router.route("documenter")
        assert decision.target == Sefirah.NETZACH

    def test_planner_routes_to_chokhmah(self, router):
        decision = router.route("planner")
        assert decision.target == Sefirah.CHOKHMAH

    def test_all_routes_from_tiferet(self, router):
        """All routes originate from the supervisor (Tiferet)."""
        for role in ROLE_MAP:
            decision = router.route(role)
            assert decision.source == Sefirah.TIFERET

    def test_all_routes_have_direct_connection(self, router):
        """All mapped roles should have direct paths from Tiferet."""
        for role in ROLE_MAP:
            decision = router.route(role)
            assert decision.path_exists, f"No direct path for role {role}"
            assert decision.hop_count == 1

    def test_unknown_role_defaults_to_yesod(self, router):
        decision = router.route("unknown_role")
        assert decision.target == Sefirah.YESOD

    def test_creative_task_alignment(self, router):
        decision = router.route("planner", task_type="creative")
        # Planner -> Chokhmah (Mercy), creative = Mercy -> alignment = 1.0
        assert decision.pillar_alignment == 1.0

    def test_validation_task_alignment(self, router):
        decision = router.route("tester", task_type="validation")
        # Tester -> Binah (Severity), validation = Severity -> alignment = 1.0
        assert decision.pillar_alignment == 1.0

    def test_misaligned_task(self, router):
        decision = router.route("builder", task_type="validation")
        # Builder -> Chesed (Mercy), validation = Severity -> alignment = 0.5
        assert decision.pillar_alignment == 0.5

    def test_topology_weight_positive(self, router):
        for role in ROLE_MAP:
            decision = router.route(role)
            assert decision.topology_weight > 0


class TestSkipConnection:
    """Test the 22nd path (Chokhmah -> Netzach) skip connection."""

    def test_planner_creative_triggers_skip(self, router):
        assert router.should_skip_to_memory("planner", "Create a new feature")

    def test_architect_creative_triggers_skip(self, router):
        assert router.should_skip_to_memory("architect", "Design a new API")

    def test_builder_does_not_skip(self, router):
        assert not router.should_skip_to_memory("builder", "Build a new module")

    def test_tester_does_not_skip(self, router):
        assert not router.should_skip_to_memory("tester", "Create test cases")

    def test_planner_validation_does_not_skip(self, router):
        assert not router.should_skip_to_memory("planner", "Review the test results")


class TestTaskClassification:
    """Test keyword-based task classification."""

    def test_creative_task(self, router):
        assert router.classify_task("Create a new authentication module") == "creative"

    def test_validation_task(self, router):
        assert router.classify_task("Check the test coverage") == "validation"

    def test_safety_task(self, router):
        assert router.classify_task("Review security vulnerabilities") == "safety"

    def test_general_task(self, router):
        assert router.classify_task("What time is it?") == "general"

    def test_mixed_defaults_to_higher_priority(self, router):
        # Safety > creative > validation
        result = router.classify_task("Create a security guard")
        assert result == "safety"


class TestDelegationWeighting:
    """Test topology-weighted delegation ordering."""

    def test_adds_topology_weight(self, router):
        delegations = [{"role": "builder"}, {"role": "tester"}]
        weighted = router.weight_delegations(delegations, "Build and test a feature")
        for d in weighted:
            assert "topology_weight" in d
            assert "sefirotic_path" in d
            assert "pillar" in d

    def test_sorts_by_weight(self, router):
        delegations = [
            {"role": "tester"},   # Validation role for creative task -> misaligned
            {"role": "builder"},  # Expansion role for creative task -> aligned
        ]
        weighted = router.weight_delegations(delegations, "Create a new module")
        # Builder should rank higher for creative task
        assert weighted[0]["role"] == "builder"

    def test_preserves_existing_fields(self, router):
        delegations = [{"role": "builder", "task": "Build X", "custom": 42}]
        weighted = router.weight_delegations(delegations, "Build X")
        assert weighted[0]["task"] == "Build X"
        assert weighted[0]["custom"] == 42

    def test_empty_delegations(self, router):
        result = router.weight_delegations([], "anything")
        assert result == []


class TestTopologySummary:
    """Test topology metadata output."""

    def test_summary_structure(self, router):
        summary = router.get_topology_summary()
        assert summary["nodes"] == 10
        assert summary["edges"] == 22
        assert summary["hub"] == "TIFERET"
        assert summary["hub_connections"] == 9
        assert "mercy" in summary["pillars"]
        assert "severity" in summary["pillars"]
        assert "equilibrium" in summary["pillars"]

    def test_skip_connection_documented(self, router):
        summary = router.get_topology_summary()
        assert "22nd path" in summary["skip_connection"]
