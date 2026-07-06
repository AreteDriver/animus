"""Tests for the Animus Citizens package (Mind Foundation layer)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from animus.citizens import ArchitectCitizen, ForgeCommissioner, ImprovementProposal, ProposalStatus
from animus.citizens.commissioner import CommissionResult
from animus.citizens.proposal import EvidenceItem, ProposalConfidence, RiskAssessment


# ---------------------------------------------------------------------------
# ImprovementProposal tests
# ---------------------------------------------------------------------------


class TestImprovementProposal:
    def test_basic_creation(self):
        proposal = ImprovementProposal(
            id="ADL-20260705-001",
            title="Test Proposal",
            problem="Something is wrong",
        )
        assert proposal.id == "ADL-20260705-001"
        assert proposal.status == ProposalStatus.DRAFT
        assert proposal.confidence_score == 0.5

    def test_confidence_mapping(self):
        proposal = ImprovementProposal(id="1", title="T", problem="P", confidence_score=0.95)
        assert proposal.confidence == ProposalConfidence.VERY_HIGH

        proposal.confidence_score = 0.8
        assert proposal.confidence == ProposalConfidence.HIGH

        proposal.confidence_score = 0.6
        assert proposal.confidence == ProposalConfidence.MEDIUM

        proposal.confidence_score = 0.3
        assert proposal.confidence == ProposalConfidence.LOW

        proposal.confidence_score = 0.1
        assert proposal.confidence == ProposalConfidence.VERY_LOW

    def test_status_update(self):
        proposal = ImprovementProposal(id="1", title="T", problem="P")
        proposal.update_status(ProposalStatus.APPROVED, actor="human")
        assert proposal.status == ProposalStatus.APPROVED
        assert proposal.approved_by == "human"
        assert proposal.approved_at is not None

    def test_serialization_roundtrip(self):
        original = ImprovementProposal(
            id="ADL-20260705-001",
            title="Test Proposal",
            problem="Problem description",
            evidence=[
                EvidenceItem(source="codebase", description="Found issue in file.py")
            ],
            potential_risks=[
                RiskAssessment(
                    description="Might break tests",
                    severity="medium",
                    mitigation="Run full suite",
                )
            ],
            confidence_score=0.75,
            affected_components=["Factory", "Kernel"],
        )

        data = original.to_dict()
        restored = ImprovementProposal.from_dict(data)

        assert restored.id == original.id
        assert restored.title == original.title
        assert len(restored.evidence) == 1
        assert restored.evidence[0].source == "codebase"
        assert len(restored.potential_risks) == 1
        assert restored.potential_risks[0].severity == "medium"
        assert restored.confidence_score == 0.75
        assert restored.affected_components == ["Factory", "Kernel"]


# ---------------------------------------------------------------------------
# ArchitectCitizen tests
# ---------------------------------------------------------------------------


class TestArchitectCitizen:
    def test_initialization(self):
        architect = ArchitectCitizen(codebase_path="/tmp/test")
        assert architect.codebase_path == Path("/tmp/test")
        assert architect._observations == []

    def test_map_priority_to_severity(self):
        assert ArchitectCitizen._map_priority_to_severity(1) == "critical"
        assert ArchitectCitizen._map_priority_to_severity(2) == "high"
        assert ArchitectCitizen._map_priority_to_severity(3) == "medium"
        assert ArchitectCitizen._map_priority_to_severity(4) == "low"
        assert ArchitectCitizen._map_priority_to_severity(5) == "info"
        assert ArchitectCitizen._map_priority_to_severity(99) == "medium"

    def test_observe_conversations_repeated_prompts(self, tmp_path):
        log_dir = tmp_path / "conversations"
        log_dir.mkdir()

        # Create conversation logs with repeated prompts
        for i in range(5):
            log_file = log_dir / f"session_{i}.jsonl"
            entries = [
                json.dumps({"prompt": "How do I configure the API?"}),
                json.dumps({"prompt": "Help me debug this error"}),
                json.dumps({"prompt": "How do I configure the API?"}),
            ]
            log_file.write_text("\n".join(entries))

        architect = ArchitectCitizen(
            codebase_path=tmp_path,
            conversation_log_dir=log_dir,
        )

        observations = architect.observe_conversations()
        assert len(observations) >= 1
        assert observations[0].source == "conversation"
        assert "Repeated prompt detected" in observations[0].description
        assert observations[0].context["count"] >= 3

    def test_observe_conversations_no_logs(self, tmp_path):
        architect = ArchitectCitizen(
            codebase_path=tmp_path,
            conversation_log_dir=tmp_path / "nonexistent",
        )
        observations = architect.observe_conversations()
        assert len(observations) == 1
        assert "not configured" in observations[0].description

    def test_analyze_empty_observations(self, tmp_path):
        architect = ArchitectCitizen(codebase_path=tmp_path)
        # Patch auto-observe methods so they don't add default observations
        with patch.object(architect, "observe_codebase", return_value=[]), \
             patch.object(architect, "observe_conversations", return_value=[]), \
             patch.object(architect, "observe_evaluations", return_value=[]):
            report = architect.analyze()
        assert report.findings == []
        assert report.technical_debt_items == []
        assert report.friction_points == []

    def test_analyze_with_observations(self, tmp_path):
        from animus.citizens.architect import Observation

        architect = ArchitectCitizen(codebase_path=tmp_path)
        architect._observations = [
            Observation(source="codebase", description="High complexity in parser.py", severity="high"),
            Observation(source="conversation", description="Users confused by auth flow", severity="medium"),
        ]

        report = architect.analyze()
        assert len(report.technical_debt_items) == 1
        assert "parser.py" in report.technical_debt_items[0]
        assert len(report.friction_points) == 1
        assert "auth" in report.friction_points[0]
        assert len(report.recommendations) == 2

    def test_generate_proposal_no_findings(self, tmp_path):
        architect = ArchitectCitizen(codebase_path=tmp_path)
        # Patch auto-observe methods so no default observations are added
        with patch.object(architect, "observe_codebase", return_value=[]), \
             patch.object(architect, "observe_conversations", return_value=[]), \
             patch.object(architect, "observe_evaluations", return_value=[]):
            report = architect.analyze()
        proposal = architect.generate_proposal(report)
        assert proposal is None

    def test_generate_proposal_with_findings(self, tmp_path):
        from animus.citizens.architect import Observation

        architect = ArchitectCitizen(codebase_path=tmp_path)
        architect._observations = [
            Observation(source="codebase", description="High complexity in parser.py", severity="high"),
        ]

        report = architect.analyze()
        proposal = architect.generate_proposal(report)
        assert proposal is not None
        assert proposal.status == ProposalStatus.DRAFT
        assert proposal.confidence_score == 0.6
        assert "parser.py" in proposal.problem
        assert proposal.affected_components == ["Factory", "Kernel"]
        assert len(proposal.evidence) == 1
        assert len(proposal.potential_risks) == 2

    def test_store_proposal_without_memory(self, tmp_path):
        architect = ArchitectCitizen(codebase_path=tmp_path)
        proposal = ImprovementProposal(id="1", title="Test", problem="P")
        assert architect.store_proposal(proposal) is False

    def test_store_proposal_with_memory(self, tmp_path):
        mock_memory = MagicMock()
        architect = ArchitectCitizen(codebase_path=tmp_path, memory_layer=mock_memory)
        proposal = ImprovementProposal(id="1", title="Test", problem="P", recommendation="R")

        assert architect.store_proposal(proposal) is True
        mock_memory.store.assert_called_once()
        call_kwargs = mock_memory.store.call_args.kwargs
        assert "architect" in call_kwargs["tags"]
        assert "proposal" in call_kwargs["tags"]

    def test_list_pending_proposals(self, tmp_path):
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {
                "content": "Proposal 1",
                "metadata": {
                    "id": "1",
                    "title": "T1",
                    "status": "submitted",
                    "problem": "P1",
                    "recommendation": "R1",
                },
            },
            {
                "content": "Proposal 2",
                "metadata": {
                    "id": "2",
                    "title": "T2",
                    "status": "complete",
                    "problem": "P2",
                    "recommendation": "R2",
                },
            },
        ]

        architect = ArchitectCitizen(codebase_path=tmp_path, memory_layer=mock_memory)
        pending = architect.list_pending_proposals()
        assert len(pending) == 1
        assert pending[0].id == "1"

    def test_observe_codebase_with_analyzer(self, tmp_path):
        mock_analyzer = MagicMock()
        mock_suggestion = MagicMock()
        mock_suggestion.category = "code_quality"
        mock_suggestion.title = "Refactor long function"
        mock_suggestion.description = "Function is 200 lines"
        mock_suggestion.priority = 2
        mock_suggestion.estimated_lines = 50
        mock_suggestion.reasoning = "Too long"
        mock_suggestion.affected_files = ["main.py"]

        mock_analyzer.analyze.return_value = MagicMock(
            suggestions=[mock_suggestion],
        )

        architect = ArchitectCitizen(codebase_path=tmp_path)
        with patch.object(architect, "_get_analyzer", return_value=mock_analyzer):
            observations = architect.observe_codebase()

        assert len(observations) == 1
        assert observations[0].source == "codebase"
        assert "Refactor long function" in observations[0].description
        assert observations[0].severity == "high"


# ---------------------------------------------------------------------------
# ForgeCommissioner tests
# ---------------------------------------------------------------------------


class TestForgeCommissioner:
    def test_initialization(self):
        commissioner = ForgeCommissioner(codebase_path="/tmp/test")
        assert commissioner.codebase_path == Path("/tmp/test")
        assert commissioner.forge_host == "localhost"
        assert commissioner.forge_port == 7700

    def test_commission_rejects_unapproved_proposal(self, tmp_path):
        commissioner = ForgeCommissioner(codebase_path=tmp_path)
        commissioner._forge_available = False  # Skip health check
        proposal = ImprovementProposal(id="1", title="T", problem="P", status=ProposalStatus.DRAFT)

        result = commissioner.commission(proposal)
        assert result.success is False
        assert "not approved" in result.error

    def test_commission_simulated_when_forge_unavailable(self, tmp_path):
        commissioner = ForgeCommissioner(codebase_path=tmp_path)
        # Force forge unavailable
        commissioner._forge_available = False

        proposal = ImprovementProposal(
            id="1",
            title="T",
            problem="P",
            status=ProposalStatus.APPROVED,
            affected_components=["Mind"],
        )

        result = commissioner.commission(proposal)
        assert result.success is False
        assert result.stage_reached == "simulated"
        assert result.evidence_bundle["simulated"] is True
        assert "Forge unavailable" in result.error

    def test_commission_auto_approve_blocked_without_env(self, tmp_path):
        commissioner = ForgeCommissioner(codebase_path=tmp_path)
        proposal = ImprovementProposal(id="1", title="T", problem="P")

        with pytest.raises(RuntimeError, match="auto_approve=True is blocked"):
            commissioner.commission(proposal, auto_approve=True)

    @patch.dict("os.environ", {"ANIMUS_FORGE_ALLOW_AUTO_APPROVE": "1"})
    def test_commission_auto_approve_with_env(self, tmp_path):
        commissioner = ForgeCommissioner(codebase_path=tmp_path)
        commissioner._forge_available = False

        proposal = ImprovementProposal(
            id="1",
            title="T",
            problem="P",
            status=ProposalStatus.DRAFT,
            affected_components=["Mind"],
        )

        # Should not raise — auto_approve is allowed with env
        result = commissioner.commission(proposal, auto_approve=True)
        assert result.proposal_id == "1"

    def test_create_workflow_config(self, tmp_path):
        commissioner = ForgeCommissioner(codebase_path=tmp_path)
        proposal = ImprovementProposal(
            id="ADL-001",
            title="Refactor parser",
            problem="Parser is too complex",
            recommendation="Split into smaller functions",
            affected_components=["Factory"],
            success_metrics=["Tests pass", "Coverage stable"],
        )

        config = commissioner._create_workflow_config(proposal)
        assert config["name"] == "architect-ADL-001"
        assert config["description"] == "Refactor parser"
        assert len(config["agents"]) == 3
        assert config["agents"][0]["name"] == "analyzer"
        assert config["agents"][1]["name"] == "implementer"
        assert config["agents"][2]["name"] == "evaluator"
        assert config["budget"]["total_tokens"] == 200_000
        assert config["budget"]["max_cost_usd"] == 0.0

    def test_update_proposal_with_evidence_success(self, tmp_path):
        commissioner = ForgeCommissioner(codebase_path=tmp_path)
        proposal = ImprovementProposal(id="1", title="T", problem="P", status=ProposalStatus.APPROVED)
        result = CommissionResult(
            success=True,
            proposal_id="1",
            stage_reached="complete",
            tests_passed=True,
        )

        updated = commissioner.update_proposal_with_evidence(proposal, result)
        assert updated.status == ProposalStatus.COMPLETE
        assert updated.evidence_bundle["success"] is True

    def test_update_proposal_with_evidence_failure(self, tmp_path):
        commissioner = ForgeCommissioner(codebase_path=tmp_path)
        proposal = ImprovementProposal(id="1", title="T", problem="P", status=ProposalStatus.APPROVED)
        result = CommissionResult(
            success=False,
            proposal_id="1",
            stage_reached="testing",
            tests_passed=False,
            error="Tests failed",
        )

        updated = commissioner.update_proposal_with_evidence(proposal, result)
        assert updated.status == ProposalStatus.EVALUATING
        assert updated.evidence_bundle["success"] is False

    @patch("httpx.get")
    def test_check_forge_health(self, mock_get, tmp_path):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        commissioner = ForgeCommissioner(codebase_path=tmp_path)
        assert commissioner._check_forge() is True

    @patch("httpx.get")
    def test_check_forge_unhealthy(self, mock_get, tmp_path):
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_get.return_value = mock_response

        commissioner = ForgeCommissioner(codebase_path=tmp_path)
        assert commissioner._check_forge() is False
