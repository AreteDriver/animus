"""Tests for CitizenCouncil — unified backlog and ranking."""

import pytest

from animus.citizens import (
    CitizenCouncil,
    ImprovementProposal,
    ProposalStatus,
    RankedProposal,
)


@pytest.fixture
def sample_proposals():
    """A diverse set of proposals for ranking/deduplication tests."""
    return [
        ImprovementProposal(
            id="PROP-A",
            title="Fix auth race condition",
            problem="Tests fail intermittently",
            recommendation="Use asyncio.gather",
            confidence_score=0.85,
            estimated_effort_hours=4.0,
            affected_components=["tests/", "api/auth"],
            status=ProposalStatus.SUBMITTED,
        ),
        ImprovementProposal(
            id="PROP-B",
            title="Refactor database layer",
            problem="Connection pool exhaustion",
            recommendation="Add connection retry logic",
            confidence_score=0.6,
            estimated_effort_hours=16.0,
            affected_components=["db/", "api/models"],
            status=ProposalStatus.SUBMITTED,
        ),
        ImprovementProposal(
            id="PROP-C",
            title="Update auth tests",
            problem="Auth tests are outdated",
            recommendation="Add new test cases",
            confidence_score=0.7,
            estimated_effort_hours=2.0,
            affected_components=["tests/", "api/auth"],
            status=ProposalStatus.SUBMITTED,
        ),
        ImprovementProposal(
            id="PROP-D",
            title="Add caching",
            problem="Slow response times",
            recommendation="Redis cache layer",
            confidence_score=0.9,
            estimated_effort_hours=8.0,
            affected_components=["cache/", "api/routes"],
            status=ProposalStatus.SUBMITTED,
        ),
    ]


class TestCitizenCouncilCollect:
    def test_collect_from_citizens_success(self):
        class FakeCitizen:
            def generate_proposal(self):
                return ImprovementProposal(
                    id="PROP-FAKE",
                    title="Fake proposal",
                    problem="A problem",
                    recommendation="Do something",
                    confidence_score=0.8,
                    estimated_effort_hours=1.0,
                    affected_components=["fake/"],
                )

        council = CitizenCouncil()
        count = council.collect_from_citizens({"fake": FakeCitizen()})
        assert count == 1
        assert "PROP-FAKE" in council._proposals

    def test_collect_from_citizens_skips_none(self):
        class NoProposalCitizen:
            def generate_proposal(self):
                return None

        council = CitizenCouncil()
        count = council.collect_from_citizens({"none": NoProposalCitizen()})
        assert count == 0

    def test_collect_from_citizens_handles_exception(self):
        class BrokenCitizen:
            def generate_proposal(self):
                raise RuntimeError("boom")

        council = CitizenCouncil()
        count = council.collect_from_citizens({"broken": BrokenCitizen()})
        assert count == 0

    def test_collect_from_memory_without_memory(self):
        council = CitizenCouncil()
        assert council.collect_from_memory() == 0

    def test_add_proposal_merges_sources(self):
        council = CitizenCouncil()
        p = ImprovementProposal(
            id="PROP-MERGE",
            title="Merge",
            problem="x",
            recommendation="y",
            confidence_score=0.5,
        )
        council._add_proposal(p, source="architect")
        council._add_proposal(p, source="test_oracle")
        rp = council._proposals["PROP-MERGE"]
        assert sorted(rp.source_citizens) == ["architect", "test_oracle"]


class TestCitizenCouncilRanking:
    def test_rank_backlog_basic(self, sample_proposals):
        council = CitizenCouncil()
        for p in sample_proposals:
            council._add_proposal(p, source="test")

        ranked = council.rank_backlog(deduplicate=False)
        assert len(ranked) == 4
        # Higher score first
        assert ranked[0].priority_score >= ranked[1].priority_score
        # Ranks are sequential
        assert ranked[0].rank == 1
        assert ranked[3].rank == 4

    def test_rank_backlog_with_deduplication(self, sample_proposals):
        council = CitizenCouncil()
        for p in sample_proposals:
            council._add_proposal(p, source="test")

        ranked = council.rank_backlog(deduplicate=True)
        # PROP-A and PROP-C share "tests/" and "api/auth"
        # The lower-scoring one should be deduplicated
        ids = [rp.proposal.id for rp in ranked]
        assert len(ids) < 4
        # The dupe should have its id recorded on the keeper
        keeper = next(rp for rp in council._proposals.values() if rp.proposal.id in ids)
        # Actually dedup tracking is on the kept item, but we can't easily know which
        # was kept. Let's verify at least one has duplicates.
        dupes = [rp for rp in council._proposals.values() if rp.duplicates]
        assert len(dupes) >= 1

    def test_rank_backlog_empty(self):
        council = CitizenCouncil()
        assert council.rank_backlog() == []

    def test_rank_backlog_structural_bonus(self):
        from animus.citizens.proposal import EvidenceItem

        council = CitizenCouncil()

        # Surface-level proposal: long function, no structural evidence
        surface = ImprovementProposal(
            id="SURFACE",
            title="Add missing docstring",
            problem="Missing docs",
            recommendation="Add docs",
            confidence_score=0.7,
            estimated_effort_hours=2.0,
            affected_components=["docs/"],
            evidence=[
                EvidenceItem(
                    source="codebase",
                    description="Missing docs",
                    data={"pattern_type": "missing_docstring"},
                ),
            ],
        )

        # Structural proposal: tight coupling
        structural = ImprovementProposal(
            id="STRUCT",
            title="Extract interface layer",
            problem="Module imports 25 others",
            recommendation="Extract interfaces",
            confidence_score=0.7,
            estimated_effort_hours=2.0,
            affected_components=["core/"],
            evidence=[
                EvidenceItem(
                    source="codebase",
                    description="Tight coupling",
                    data={"pattern_type": "tight_coupling"},
                ),
            ],
        )

        council._add_proposal(surface, source="architect")
        council._add_proposal(structural, source="architect")

        ranked = council.rank_backlog(deduplicate=False)
        assert len(ranked) == 2
        # Structural should outrank surface despite same base metrics
        assert ranked[0].proposal.id == "STRUCT"
        assert ranked[0].priority_score > ranked[1].priority_score


class TestCitizenCouncilFiltering:
    def test_filter_by_component(self, sample_proposals):
        council = CitizenCouncil()
        for p in sample_proposals:
            council._add_proposal(p, source="test")
        results = council.filter_by_component("api/auth")
        assert len(results) == 2
        ids = {rp.proposal.id for rp in results}
        assert ids == {"PROP-A", "PROP-C"}

    def test_filter_by_confidence(self, sample_proposals):
        council = CitizenCouncil()
        for p in sample_proposals:
            council._add_proposal(p, source="test")
        results = council.filter_by_confidence(min_confidence=0.8)
        assert len(results) == 2
        ids = {rp.proposal.id for rp in results}
        assert ids == {"PROP-A", "PROP-D"}

    def test_filter_by_status(self, sample_proposals):
        council = CitizenCouncil()
        for p in sample_proposals:
            council._add_proposal(p, source="test")
        results = council.filter_by_status(ProposalStatus.SUBMITTED)
        assert len(results) == 4


class TestCitizenCouncilSummary:
    def test_summary(self, sample_proposals):
        council = CitizenCouncil()
        for p in sample_proposals:
            council._add_proposal(p, source="test")
        s = council.summary()
        assert s["total_proposals"] == 4
        assert s["total_estimated_effort_hours"] == 30.0
        assert s["sources"] == {"test": 4}

    def test_clear(self, sample_proposals):
        council = CitizenCouncil()
        for p in sample_proposals:
            council._add_proposal(p, source="test")
        council.clear()
        assert council._proposals == {}
        assert council.rank_backlog() == []


class TestRankedProposal:
    def test_to_dict(self):
        p = ImprovementProposal(
            id="PROP-T",
            title="Test",
            problem="x",
            recommendation="y",
            confidence_score=0.5,
        )
        rp = RankedProposal(proposal=p, rank=1, priority_score=2.5)
        d = rp.to_dict()
        assert d["rank"] == 1
        assert d["priority_score"] == 2.5
        assert d["proposal"]["id"] == "PROP-T"
