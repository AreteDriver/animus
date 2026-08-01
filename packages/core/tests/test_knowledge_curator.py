"""Tests for KnowledgeCuratorCitizen."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from animus.citizens import (
    ImprovementProposal,
    KnowledgeCuratorCitizen,
    ProposalStatus,
)


class TestKnowledgeCuratorCitizen:
    def test_initialization(self):
        curator = KnowledgeCuratorCitizen(codebase_path="/tmp/test")
        assert curator.codebase_path == Path("/tmp/test")
        assert curator._drifts == []

    def test_initialization_no_paths(self):
        curator = KnowledgeCuratorCitizen()
        assert curator.codebase_path is None
        assert curator.memory is None

    # ------------------------------------------------------------------
    # observe_stale_references
    # ------------------------------------------------------------------

    def test_observe_stale_references_no_memory(self):
        curator = KnowledgeCuratorCitizen(codebase_path="/tmp/test")
        observations = curator.observe_stale_references()
        assert len(observations) == 1
        assert "Memory layer not available" in observations[0].description

    def test_observe_stale_references_no_codebase(self):
        mock_memory = MagicMock()
        curator = KnowledgeCuratorCitizen(memory_layer=mock_memory)
        observations = curator.observe_stale_references()
        assert len(observations) == 1
        assert "Codebase path not configured" in observations[0].description

    def test_observe_stale_references_finds_missing_file(self, tmp_path):
        # Create a codebase with only some files
        codebase = tmp_path / "codebase"
        codebase.mkdir()
        (codebase / "exists.py").write_text("# exists")

        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {
                "id": "mem-1",
                "content": "The `missing.py` file handles authentication.",
            },
            {
                "id": "mem-2",
                "content": "The `exists.py` file handles logging.",
            },
        ]

        curator = KnowledgeCuratorCitizen(
            codebase_path=codebase,
            memory_layer=mock_memory,
        )
        observations = curator.observe_stale_references()

        stale = [o for o in observations if "missing.py" in o.description]
        assert len(stale) == 1
        assert stale[0].context["pattern_type"] == "stale_reference"

        # exists.py should NOT be flagged
        existing = [o for o in observations if "exists.py" in o.description]
        assert len(existing) == 0

    def test_observe_stale_references_ignores_existing(self, tmp_path):
        codebase = tmp_path / "codebase"
        codebase.mkdir()
        (codebase / "real.py").write_text("# exists")

        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"id": "mem-1", "content": "Use `real.py` for this task."},
        ]

        curator = KnowledgeCuratorCitizen(
            codebase_path=codebase,
            memory_layer=mock_memory,
        )
        observations = curator.observe_stale_references()

        stale = [o for o in observations if "stale_reference" in o.context.get("pattern_type", "")]
        assert len(stale) == 0

    # ------------------------------------------------------------------
    # observe_contradictions
    # ------------------------------------------------------------------

    def test_observe_contradictions_no_memory(self):
        curator = KnowledgeCuratorCitizen()
        observations = curator.observe_contradictions()
        assert observations == []

    def test_observe_contradictions_finds_conflict(self):
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {
                "id": "mem-a",
                "content": "AuthModule enables SSO and improves login speed.",
            },
            {
                "id": "mem-b",
                "content": "AuthModule breaks on weekends. It disables caching and is slow.",
            },
        ]

        curator = KnowledgeCuratorCitizen(memory_layer=mock_memory)
        observations = curator.observe_contradictions()

        assert len(observations) >= 1
        assert observations[0].context["pattern_type"] == "contradiction"
        assert "AuthModule" in observations[0].description
        assert observations[0].severity == "high"

    def test_observe_contradictions_none_when_agreement(self):
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"id": "mem-a", "content": "The AuthModule is fast and safe."},
            {"id": "mem-b", "content": "The AuthModule supports SSO and is secure."},
        ]

        curator = KnowledgeCuratorCitizen(memory_layer=mock_memory)
        observations = curator.observe_contradictions()
        # Both positive, no contradiction
        assert not any(o.context.get("pattern_type") == "contradiction" for o in observations)

    # ------------------------------------------------------------------
    # observe_outdated_claims
    # ------------------------------------------------------------------

    def test_observe_outdated_claims_no_memory(self):
        curator = KnowledgeCuratorCitizen()
        observations = curator.observe_outdated_claims()
        assert observations == []

    def test_observe_outdated_claims_finds_stale(self):
        old_date = (datetime.now() - timedelta(days=100)).isoformat()

        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {
                "id": "mem-old",
                "content": "CCP recently deprecated the read_online SSO scope.",
                "created_at": old_date,
            },
            {
                "id": "mem-fresh",
                "content": "CCP recently deprecated the read_online SSO scope.",
                "created_at": datetime.now().isoformat(),
            },
        ]

        curator = KnowledgeCuratorCitizen(memory_layer=mock_memory)
        observations = curator.observe_outdated_claims()

        old_obs = [o for o in observations if o.context.get("memory_id") == "mem-old"]
        assert len(old_obs) == 1
        assert old_obs[0].severity == "high"

    def test_observe_outdated_claims_ignores_timeless(self):
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {
                "id": "mem-1",
                "content": "Python uses indentation for block structure.",
                "created_at": (datetime.now() - timedelta(days=200)).isoformat(),
            },
        ]

        curator = KnowledgeCuratorCitizen(memory_layer=mock_memory)
        observations = curator.observe_outdated_claims()
        assert len(observations) == 0

    # ------------------------------------------------------------------
    # observe_orphan_topics
    # ------------------------------------------------------------------

    def test_observe_orphan_topics_no_codebase(self):
        curator = KnowledgeCuratorCitizen()
        observations = curator.observe_orphan_topics()
        assert len(observations) == 1
        assert "Codebase path not configured" in observations[0].description

    def test_observe_orphan_topics_finds_orphans(self, tmp_path):
        codebase = tmp_path / "codebase"
        codebase.mkdir()

        topics = codebase / "topics"
        topics.mkdir()

        orphan = topics / "orphan.md"
        orphan.write_text("# Orphan Topic\n\nThis topic is not linked.")

        linked = topics / "linked.md"
        linked.write_text("# Linked Topic\n\nSee [orphan](orphan.md) for details.")

        # Also create a parent README that links to linked.md
        readme = topics / "README.md"
        readme.write_text("# Topics\n\n- [Linked](linked.md)")

        curator = KnowledgeCuratorCitizen(codebase_path=codebase)
        observations = curator.observe_orphan_topics()

        orphan_obs = [o for o in observations if "orphan" in o.description.lower()]
        assert len(orphan_obs) == 1
        assert orphan_obs[0].context["pattern_type"] == "orphan_topic"

    def test_observe_orphan_topics_no_topics_dir(self, tmp_path):
        codebase = tmp_path / "codebase"
        codebase.mkdir()

        curator = KnowledgeCuratorCitizen(codebase_path=codebase)
        observations = curator.observe_orphan_topics()
        # No topics/ dir present — gracefully returns empty, not a false-positive finding.
        assert len(observations) == 0

    # ------------------------------------------------------------------
    # analyze
    # ------------------------------------------------------------------

    def test_analyze_no_findings(self, tmp_path):
        codebase = tmp_path / "codebase"
        codebase.mkdir()
        curator = KnowledgeCuratorCitizen(codebase_path=codebase)

        # Patch observation methods to return empty
        with (
            patch.object(curator, "observe_stale_references", return_value=[]),
            patch.object(curator, "observe_contradictions", return_value=[]),
            patch.object(curator, "observe_outdated_claims", return_value=[]),
            patch.object(curator, "observe_orphan_topics", return_value=[]),
        ):
            drifts = curator.analyze()

        assert drifts == []

    def test_analyze_aggregates_by_type(self, tmp_path):
        from animus.citizens.architect import Observation

        codebase = tmp_path / "codebase"
        codebase.mkdir()
        curator = KnowledgeCuratorCitizen(codebase_path=codebase)

        with (
            patch.object(
                curator,
                "observe_stale_references",
                return_value=[
                    Observation(
                        source="knowledge",
                        description="Stale ref 1",
                        severity="medium",
                        context={"pattern_type": "stale_reference"},
                    ),
                ],
            ),
            patch.object(
                curator,
                "observe_contradictions",
                return_value=[
                    Observation(
                        source="knowledge",
                        description="Contradiction 1",
                        severity="high",
                        context={"pattern_type": "contradiction"},
                    ),
                ],
            ),
            patch.object(curator, "observe_outdated_claims", return_value=[]),
            patch.object(curator, "observe_orphan_topics", return_value=[]),
        ):
            drifts = curator.analyze()

        assert len(drifts) == 2
        types = {d.drift_type for d in drifts}
        assert "stale_reference" in types
        assert "contradiction" in types

    # ------------------------------------------------------------------
    # generate_proposal
    # ------------------------------------------------------------------

    def test_generate_proposal_no_findings(self, tmp_path):
        codebase = tmp_path / "codebase"
        codebase.mkdir()
        curator = KnowledgeCuratorCitizen(codebase_path=codebase)

        with patch.object(curator, "analyze", return_value=[]):
            proposal = curator.generate_proposal()
        assert proposal is None

    def test_generate_proposal_with_findings(self, tmp_path):
        from animus.citizens.knowledge_curator import KnowledgeDrift

        codebase = tmp_path / "codebase"
        codebase.mkdir()
        curator = KnowledgeCuratorCitizen(codebase_path=codebase)

        drifts = [
            KnowledgeDrift(
                drift_type="stale_reference",
                description="Memory references old_file.py which no longer exists",
                severity="high",
                affected_memory_id="mem-1",
            ),
            KnowledgeDrift(
                drift_type="stale_reference",
                description="Memory references another_missing.py",
                severity="medium",
                affected_memory_id="mem-2",
            ),
        ]

        with patch.object(curator, "analyze", return_value=drifts):
            proposal = curator.generate_proposal()

        assert proposal is not None
        assert proposal.status == ProposalStatus.DRAFT
        assert isinstance(proposal.id, str)
        assert proposal.id.startswith("ADL-")
        assert len(proposal.evidence) == 2
        assert proposal.affected_components == ["Mind", "Factory"]

    def test_generate_proposal_for_outdated_claim(self, tmp_path):
        from animus.citizens.knowledge_curator import KnowledgeDrift

        codebase = tmp_path / "codebase"
        codebase.mkdir()
        curator = KnowledgeCuratorCitizen(codebase_path=codebase)

        drifts = [
            KnowledgeDrift(
                drift_type="outdated_claim",
                description="Old claim about deprecated feature",
                severity="high",
            ),
        ]

        with patch.object(curator, "analyze", return_value=drifts):
            proposal = curator.generate_proposal()

        assert proposal is not None
        assert proposal.affected_components == ["Mind", "Society"]

    # ------------------------------------------------------------------
    # store_proposal
    # ------------------------------------------------------------------

    def test_store_proposal_without_memory(self):
        curator = KnowledgeCuratorCitizen()
        proposal = ImprovementProposal(id="1", title="T", problem="P")
        assert curator.store_proposal(proposal) is False

    def test_store_proposal_with_memory(self):
        mock_memory = MagicMock()
        curator = KnowledgeCuratorCitizen(memory_layer=mock_memory)
        proposal = ImprovementProposal(id="1", title="T", problem="P", recommendation="R")

        assert curator.store_proposal(proposal) is True
        mock_memory.remember.assert_called_once()
        call_kwargs = mock_memory.remember.call_args.kwargs
        assert "knowledge_curator" in call_kwargs["tags"]
        assert "proposal" in call_kwargs["tags"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def test_extract_topic_anchors(self):
        content = "The AuthModule is fast. Use def authenticate() in auth.py."
        anchors = KnowledgeCuratorCitizen._extract_topic_anchors(content)
        assert "AuthModule" in anchors
        assert "authenticate" in anchors

    def test_suggest_for_drift(self):
        assert "Update memory" in KnowledgeCuratorCitizen._suggest_for_drift("stale_reference")
        assert "Reconcile" in KnowledgeCuratorCitizen._suggest_for_drift("contradiction")
        assert "verification date" in KnowledgeCuratorCitizen._suggest_for_drift("outdated_claim")
        assert "cross-references" in KnowledgeCuratorCitizen._suggest_for_drift("orphan_topic")

    def test_build_problem_recommendation(self):
        from animus.citizens.knowledge_curator import KnowledgeDrift

        drift = KnowledgeDrift(
            drift_type="stale_reference",
            description="Memory references old file",
            severity="high",
        )
        problem, recommendation = KnowledgeCuratorCitizen._build_problem_recommendation(drift)
        assert "no longer exist" in problem
        assert "Audit" in recommendation

    def test_repr(self):
        curator = KnowledgeCuratorCitizen()
        assert "KnowledgeCuratorCitizen" in repr(curator)
