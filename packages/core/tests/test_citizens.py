"""Tests for the Animus Citizens package (Mind Foundation layer)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from animus.citizens import (
    ArchitectCitizen,
    CitizenCouncil,
    ConversationDesignerCitizen,
    ForgeCommissioner,
    ImprovementProposal,
    KnowledgeCuratorCitizen,
    ProposalQueue,
    ProposalStatus,
    TestOracleCitizen,
)
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
        # No logs configured — gracefully returns empty, not a false-positive finding.
        assert len(observations) == 0

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
        # Confidence is now dynamically scored from evidence quality
        assert 0.25 <= proposal.confidence_score <= 1.0
        assert "parser.py" in proposal.problem
        assert proposal.affected_components == ["Factory", "Kernel"]
        assert len(proposal.evidence) >= 1
        assert len(proposal.potential_risks) >= 2
        # Senior skillsets should enrich the recommendation
        assert "Trade-off analysis" in proposal.recommendation or "Estimated effort" in proposal.recommendation

    def test_store_proposal_without_memory(self, tmp_path):
        architect = ArchitectCitizen(codebase_path=tmp_path)
        proposal = ImprovementProposal(id="1", title="Test", problem="P")
        assert architect.store_proposal(proposal) is False

    def test_store_proposal_with_memory(self, tmp_path):
        mock_memory = MagicMock()
        architect = ArchitectCitizen(codebase_path=tmp_path, memory_layer=mock_memory)
        proposal = ImprovementProposal(id="1", title="Test", problem="P", recommendation="R")

        assert architect.store_proposal(proposal) is True
        mock_memory.remember.assert_called_once()
        call_kwargs = mock_memory.remember.call_args.kwargs
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


    def test_senior_dependency_analysis_tight_coupling(self, tmp_path):
        # Create a module that imports many others (>10 threshold)
        pkg = tmp_path / "core"
        pkg.mkdir()
        main = pkg / "main.py"
        main.write_text(
            "import os\nimport sys\nimport json\nimport re\nimport ast\n"
            "import typing\nimport collections\nimport itertools\nimport functools\n"
            "import pathlib\nimport datetime\nimport hashlib\nimport uuid\nimport logging\n"
            "\ndef hello(): pass\n"
        )
        architect = ArchitectCitizen(codebase_path=tmp_path)
        obs = architect._analyze_dependencies()
        assert len(obs) >= 1
        assert any(o.context.get("pattern_type") == "tight_coupling" for o in obs)

    def test_senior_dependency_analysis_circular_import(self, tmp_path):
        pkg = tmp_path / "core"
        pkg.mkdir()
        a = pkg / "a.py"
        b = pkg / "b.py"
        a.write_text("from . import b\n")
        b.write_text("from . import a\n")
        architect = ArchitectCitizen(codebase_path=tmp_path)
        obs = architect._analyze_dependencies()
        assert any(o.context.get("pattern_type") == "circular_import" for o in obs)

    def test_senior_detect_god_class(self, tmp_path):
        pkg = tmp_path / "core"
        pkg.mkdir()
        f = pkg / "big.py"
        methods = "\n".join(f"    def method_{i}(self): pass" for i in range(20))
        f.write_text(f"class GodClass:\n{methods}\n")
        architect = ArchitectCitizen(codebase_path=tmp_path)
        obs = architect._detect_architectural_patterns()
        assert any(o.context.get("pattern_type") == "god_class" for o in obs)

    def test_senior_detect_singleton_abuse(self, tmp_path):
        pkg = tmp_path / "core"
        pkg.mkdir()
        f = pkg / "singleton.py"
        f.write_text("class MySingleton:\n    def __new__(cls): pass\n")
        architect = ArchitectCitizen(codebase_path=tmp_path)
        obs = architect._detect_architectural_patterns()
        assert any(o.context.get("pattern_type") == "singleton_abuse" for o in obs)

    def test_senior_constraint_check_blocks_direct_modification(self, tmp_path):
        architect = ArchitectCitizen(codebase_path=tmp_path)
        proposal = ImprovementProposal(
            id="1",
            title="T",
            problem="P",
            recommendation="Modify directly the source code",
            evaluation_plan="Run tests",
            rollback_plan="Revert",
            affected_components=["Mind"],
        )
        violations = architect._check_architectural_constraints(proposal)
        assert any("modify directly" in v.lower() for v in violations)

    def test_senior_constraint_check_requires_evaluation_plan(self, tmp_path):
        architect = ArchitectCitizen(codebase_path=tmp_path)
        proposal = ImprovementProposal(
            id="1",
            title="T",
            problem="P",
            recommendation="Refactor parser",
            rollback_plan="Revert",
            affected_components=["Mind"],
        )
        violations = architect._check_architectural_constraints(proposal)
        assert any("evaluation plan" in v.lower() for v in violations)

    def test_senior_impact_estimation_empty(self, tmp_path):
        architect = ArchitectCitizen(codebase_path=tmp_path)
        impact = architect._estimate_impact([])
        assert impact["component_count"] == 0
        assert impact["impact_score"] == 0.0

    def test_senior_impact_estimation_with_files(self, tmp_path):
        architect = ArchitectCitizen(codebase_path=tmp_path)
        impact = architect._estimate_impact(["packages/core/animus/identity.py"])
        assert impact["component_count"] >= 1
        assert 0 < impact["impact_score"] <= 1.0

    def test_senior_evidence_quality_score_no_evidence(self, tmp_path):
        architect = ArchitectCitizen(codebase_path=tmp_path)
        score = architect._score_evidence_quality([])
        assert score == 0.3

    def test_senior_evidence_quality_score_with_evidence(self, tmp_path):
        from datetime import datetime
        from animus.citizens.proposal import EvidenceItem

        architect = ArchitectCitizen(codebase_path=tmp_path)
        evidence = [
            EvidenceItem(source="codebase", description="Issue in parser.py", data={"file": "parser.py"}),
            EvidenceItem(source="evaluation", description="Low score", data={"score": 0.5}),
        ]
        score = architect._score_evidence_quality(evidence)
        assert 0.4 < score <= 1.0

    def test_senior_trade_off_analysis(self, tmp_path):
        from animus.citizens.proposal import RiskAssessment

        architect = ArchitectCitizen(codebase_path=tmp_path)
        proposal = ImprovementProposal(
            id="1",
            title="T",
            problem="P",
            estimated_effort_hours=6.0,
            affected_components=["Factory", "Kernel"],
            potential_risks=[
                RiskAssessment(description="Might break tests", severity="medium", mitigation="Run suite", probability=0.3),
            ],
            alternatives_considered=["Status quo"],
        )
        trade_offs = architect._build_trade_off_analysis(proposal)
        assert "6.0 hours" in trade_offs
        assert "Factory" in trade_offs
        assert "Status quo" in trade_offs


# ---------------------------------------------------------------------------
# ForgeCommissioner tests
# ---------------------------------------------------------------------------


class TestForgeCommissioner:
    def test_initialization(self):
        commissioner = ForgeCommissioner(codebase_path="/tmp/test")
        assert commissioner.codebase_path == Path("/tmp/test")
        assert commissioner.forge_host == "localhost"
        assert commissioner.forge_port == 8000

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

    def test_commission_local_engine_bypasses_http(self, tmp_path):
        commissioner = ForgeCommissioner(codebase_path=tmp_path, use_local_engine=True)

        # Mock _execute_local to avoid import errors
        mock_result = CommissionResult(
            success=True,
            proposal_id="1",
            stage_reached="complete",
            tests_passed=True,
        )
        commissioner._execute_local = MagicMock(return_value=mock_result)

        proposal = ImprovementProposal(
            id="1",
            title="T",
            problem="P",
            status=ProposalStatus.APPROVED,
            affected_components=["Mind"],
        )

        result = commissioner.commission(proposal)
        assert result.success is True
        assert result.stage_reached == "complete"
        commissioner._execute_local.assert_called_once_with(proposal)

    def test_execute_local_engine_unavailable(self, tmp_path):
        commissioner = ForgeCommissioner(codebase_path=tmp_path, use_local_engine=True)
        commissioner._get_local_engine = MagicMock(return_value=None)

        proposal = ImprovementProposal(
            id="1",
            title="T",
            problem="P",
            status=ProposalStatus.APPROVED,
            affected_components=["Mind"],
        )

        result = commissioner._execute_local(proposal)
        assert result.success is False
        assert "Local ForgeEngine not available" in result.error

    def test_execute_local_success(self, tmp_path):
        commissioner = ForgeCommissioner(codebase_path=tmp_path, use_local_engine=True)

        mock_engine = MagicMock()
        mock_engine.run.return_value = {
            "status": "success",
            "metrics": {"tokens_used": 1500},
        }
        commissioner._get_local_engine = MagicMock(return_value=mock_engine)

        proposal = ImprovementProposal(
            id="ADL-001",
            title="Refactor parser",
            problem="Parser is too complex",
            status=ProposalStatus.APPROVED,
            affected_components=["Factory"],
        )

        result = commissioner._execute_local(proposal)
        assert result.success is True
        assert result.stage_reached == "success"
        assert result.tests_passed is True
        assert result.benchmark_results == {"tokens_used": 1500}
        mock_engine.run.assert_called_once()


# ---------------------------------------------------------------------------
# ConversationDesignerCitizen tests
# ---------------------------------------------------------------------------


class TestConversationDesignerCitizen:
    def test_observe_repeated_prompts(self, tmp_path):
        log_dir = tmp_path / "conversations"
        log_dir.mkdir()
        log_file = log_dir / "session.jsonl"
        entries = [
            json.dumps({"prompt": "How do I configure the API?"}),
            json.dumps({"prompt": "Help me debug this error"}),
            json.dumps({"prompt": "How do I configure the API?"}),
            json.dumps({"prompt": "How do I configure the API?"}),
        ]
        log_file.write_text("\n".join(entries))

        designer = ConversationDesignerCitizen(conversation_log_dir=log_dir)
        observations = designer.observe_repeated_prompts()
        assert len(observations) >= 1
        assert observations[0].source == "conversation"
        assert "Repeated prompt detected" in observations[0].description
        assert observations[0].context["count"] >= 3

    def test_observe_vague_requests(self, tmp_path):
        log_dir = tmp_path / "conversations"
        log_dir.mkdir()
        log_file = log_dir / "session.jsonl"
        entries = [
            json.dumps({"prompt": "do something here"}),
            json.dumps({"prompt": "do something here"}),
            json.dumps({"prompt": "fix this"}),
        ]
        log_file.write_text("\n".join(entries))

        designer = ConversationDesignerCitizen(conversation_log_dir=log_dir)
        observations = designer.observe_vague_requests()
        assert len(observations) >= 1
        assert observations[0].source == "conversation"
        assert "Vague request detected" in observations[0].description
        assert observations[0].context["pattern_type"] == "vague_request"

    def test_observe_correction_loops(self, tmp_path):
        log_dir = tmp_path / "conversations"
        log_dir.mkdir()
        log_file = log_dir / "session.jsonl"
        entries = [
            json.dumps({"prompt": "no, that's wrong"}),
            json.dumps({"prompt": "actually, I meant something else"}),
            json.dumps({"prompt": "not quite right"}),
        ]
        log_file.write_text("\n".join(entries))

        designer = ConversationDesignerCitizen(conversation_log_dir=log_dir)
        observations = designer.observe_correction_loops()
        assert len(observations) >= 1
        assert observations[0].source == "conversation"
        assert "Correction loop detected" in observations[0].description
        assert observations[0].context["pattern_type"] == "correction_loop"

    def test_generate_proposal(self, tmp_path):
        log_dir = tmp_path / "conversations"
        log_dir.mkdir()
        log_file = log_dir / "session.jsonl"
        entries = [
            json.dumps({"prompt": "How do I configure the API?"}),
            json.dumps({"prompt": "How do I configure the API?"}),
            json.dumps({"prompt": "How do I configure the API?"}),
        ]
        log_file.write_text("\n".join(entries))

        designer = ConversationDesignerCitizen(conversation_log_dir=log_dir)
        proposal = designer.generate_proposal()
        assert proposal is not None
        assert isinstance(proposal, ImprovementProposal)
        assert proposal.status == ProposalStatus.DRAFT
        assert len(proposal.evidence) >= 1

    def test_generate_proposal_no_findings(self, tmp_path):
        designer = ConversationDesignerCitizen(conversation_log_dir=tmp_path / "empty")
        proposal = designer.generate_proposal()
        assert proposal is None

    def test_store_proposal_with_memory(self, tmp_path):
        mock_memory = MagicMock()
        designer = ConversationDesignerCitizen(memory_layer=mock_memory)
        proposal = ImprovementProposal(id="1", title="Test", problem="P", recommendation="R")
        assert designer.store_proposal(proposal) is True
        mock_memory.remember.assert_called_once()

    def test_store_proposal_without_memory(self, tmp_path):
        designer = ConversationDesignerCitizen()
        proposal = ImprovementProposal(id="1", title="Test", problem="P", recommendation="R")
        assert designer.store_proposal(proposal) is False


# ---------------------------------------------------------------------------
# KnowledgeCuratorCitizen tests
# ---------------------------------------------------------------------------


class TestKnowledgeCuratorCitizen:
    def test_observe_stale_references_no_memory(self, tmp_path):
        curator = KnowledgeCuratorCitizen(codebase_path=tmp_path)
        observations = curator.observe_stale_references()
        assert len(observations) == 1
        assert "Memory layer not available" in observations[0].description

    def test_observe_stale_references_with_mock(self, tmp_path):
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"content": "See main.py for details", "id": "mem1"},
        ]
        curator = KnowledgeCuratorCitizen(codebase_path=tmp_path, memory_layer=mock_memory)
        observations = curator.observe_stale_references()
        assert len(observations) >= 1
        assert observations[0].source == "knowledge"
        assert observations[0].context["pattern_type"] == "stale_reference"

    def test_observe_contradictions(self, tmp_path):
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"content": "ModuleA is fast and improves performance", "id": "mem1"},
            {"content": "ModuleA is slow and breaks things", "id": "mem2"},
        ]
        curator = KnowledgeCuratorCitizen(memory_layer=mock_memory)
        observations = curator.observe_contradictions()
        assert len(observations) >= 1
        assert observations[0].source == "knowledge"
        assert "Contradictory claims" in observations[0].description
        assert observations[0].context["pattern_type"] == "contradiction"

    def test_observe_outdated_claims(self, tmp_path):
        mock_memory = MagicMock()
        old_date = (datetime.now() - timedelta(days=100)).isoformat()
        mock_memory.search.return_value = [
            {"content": "CCP recently changed the SSO scopes", "id": "mem1", "created_at": old_date},
        ]
        curator = KnowledgeCuratorCitizen(memory_layer=mock_memory)
        observations = curator.observe_outdated_claims()
        assert len(observations) >= 1
        assert observations[0].source == "knowledge"
        assert observations[0].context["pattern_type"] == "outdated_claim"

    def test_observe_orphan_topics(self, tmp_path):
        topics_dir = tmp_path / "topics"
        topics_dir.mkdir()
        orphan = topics_dir / "orphan.md"
        linked = topics_dir / "linked.md"
        index = topics_dir / "index.md"
        orphan.write_text("# Orphan Topic\n")
        linked.write_text("# Linked Topic\n[index](index.md)\n")
        index.write_text("[Linked Topic](linked.md)\n")

        curator = KnowledgeCuratorCitizen(codebase_path=tmp_path)
        observations = curator.observe_orphan_topics()
        assert len(observations) == 1
        assert observations[0].context["pattern_type"] == "orphan_topic"
        assert "orphan.md" in observations[0].description

    def test_generate_proposal(self, tmp_path):
        mock_memory = MagicMock()
        old_date = (datetime.now() - timedelta(days=100)).isoformat()
        mock_memory.search.return_value = [
            {"content": "CCP recently changed the SSO scopes", "id": "mem1", "created_at": old_date},
        ]
        curator = KnowledgeCuratorCitizen(memory_layer=mock_memory)
        proposal = curator.generate_proposal()
        assert proposal is not None
        assert isinstance(proposal, ImprovementProposal)
        assert proposal.status == ProposalStatus.DRAFT
        assert len(proposal.evidence) >= 1

    def test_generate_proposal_no_findings(self, tmp_path):
        mock_memory = MagicMock()
        mock_memory.search.return_value = []
        curator = KnowledgeCuratorCitizen(codebase_path=tmp_path, memory_layer=mock_memory)
        proposal = curator.generate_proposal()
        assert proposal is None

    def test_store_proposal_with_memory(self, tmp_path):
        mock_memory = MagicMock()
        curator = KnowledgeCuratorCitizen(memory_layer=mock_memory)
        proposal = ImprovementProposal(id="1", title="Test", problem="P", recommendation="R")
        assert curator.store_proposal(proposal) is True
        mock_memory.remember.assert_called_once()

    def test_store_proposal_without_memory(self, tmp_path):
        curator = KnowledgeCuratorCitizen()
        proposal = ImprovementProposal(id="1", title="Test", problem="P", recommendation="R")
        assert curator.store_proposal(proposal) is False


# ---------------------------------------------------------------------------
# TestOracleCitizen tests
# ---------------------------------------------------------------------------


class TestTestOracleCitizen:
    def test_observe_test_failures(self, tmp_path):
        pytest_output = (
            "test_foo.py::test_a PASSED\n"
            "FAILED test_foo.py::test_b\n"
            "FAILED test_bar.py::test_c\n"
            "2 failed, 1 passed in 0.5s\n"
        )
        oracle = TestOracleCitizen(codebase_path=tmp_path)
        observations = oracle.observe_test_failures(pytest_output=pytest_output)
        assert len(observations) >= 1
        assert any(o.context.get("pattern_type") == "test_failure" for o in observations)

    def test_observe_coverage_gaps(self, tmp_path):
        coverage_report = (
            "Name         Stmts   Miss  Cover\n"
            "--------------------------------\n"
            "main.py         10      0     0%\n"
            "utils.py         5      0   100%\n"
            "--------------------------------\n"
            "TOTAL           15     10    33%\n"
        )
        oracle = TestOracleCitizen(codebase_path=tmp_path)
        observations = oracle.observe_coverage_gaps(coverage_report=coverage_report)
        assert len(observations) >= 1
        assert any(
            o.context.get("pattern_type") in ("coverage_drop", "missing_coverage")
            for o in observations
        )

    def test_observe_eval_drift(self, tmp_path):
        eval_results = [
            {"suite": "suite_a", "score": 0.9, "timestamp": "2026-01-01T00:00:00Z"},
            {"suite": "suite_a", "score": 0.7, "timestamp": "2026-01-02T00:00:00Z"},
        ]
        oracle = TestOracleCitizen(codebase_path=tmp_path)
        observations = oracle.observe_eval_drift(eval_results=eval_results)
        assert len(observations) == 1
        assert observations[0].context["pattern_type"] == "eval_drift"
        assert observations[0].context["delta"] == pytest.approx(-0.2)

    def test_generate_proposal(self, tmp_path):
        (tmp_path / "pytest-output.txt").write_text(
            "test_foo.py::test_a FAILED\n"
            "1 failed, 0 passed in 0.5s\n"
        )
        (tmp_path / "coverage.txt").write_text(
            "Name         Stmts   Miss  Cover\n"
            "--------------------------------\n"
            "main.py         10      0     0%\n"
            "--------------------------------\n"
            "TOTAL           15     10    33%\n"
        )
        oracle = TestOracleCitizen(codebase_path=tmp_path)
        proposal = oracle.generate_proposal()
        assert proposal is not None
        assert isinstance(proposal, ImprovementProposal)
        assert proposal.status == ProposalStatus.DRAFT

    def test_generate_proposal_no_findings(self, tmp_path):
        oracle = TestOracleCitizen(codebase_path=tmp_path)
        proposal = oracle.generate_proposal()
        assert proposal is None

    def test_store_proposal_with_memory(self, tmp_path):
        mock_memory = MagicMock()
        oracle = TestOracleCitizen(codebase_path=tmp_path, memory_layer=mock_memory)
        proposal = ImprovementProposal(id="1", title="Test", problem="P", recommendation="R")
        assert oracle.store_proposal(proposal) is True
        mock_memory.remember.assert_called_once()

    def test_store_proposal_without_memory(self, tmp_path):
        oracle = TestOracleCitizen(codebase_path=tmp_path)
        proposal = ImprovementProposal(id="1", title="Test", problem="P", recommendation="R")
        assert oracle.store_proposal(proposal) is False


# ---------------------------------------------------------------------------
# ProposalQueue tests
# ---------------------------------------------------------------------------


class TestProposalQueue:
    def test_submit_approve_commission_complete_lifecycle(self, tmp_path):
        queue = ProposalQueue(storage_path=str(tmp_path / "queue.json"))
        proposal = ImprovementProposal(id="ADL-001", title="T", problem="P")

        qp = queue.submit(proposal)
        assert qp.current_status == ProposalStatus.SUBMITTED
        assert queue.stats()["total"] == 1

        approved = queue.approve("ADL-001", actor="human", reason="LGTM")
        assert approved.current_status == ProposalStatus.APPROVED
        assert approved.proposal.approved_by == "human"

        commissioned = queue.commission("ADL-001", actor="forge")
        assert commissioned.current_status == ProposalStatus.COMMISSIONED

        completed = queue.complete("ADL-001", actor="forge")
        assert completed.current_status == ProposalStatus.COMPLETE
        assert queue.stats()["complete"] == 1

    def test_reject_path(self, tmp_path):
        queue = ProposalQueue(storage_path=str(tmp_path / "queue.json"))
        proposal = ImprovementProposal(id="ADL-002", title="T", problem="P")
        queue.submit(proposal)

        rejected = queue.reject("ADL-002", actor="human", reason="Not now")
        assert rejected.current_status == ProposalStatus.REJECTED
        assert queue.stats()["rejected"] == 1

        # Rejecting again should be no-op
        rejected_again = queue.reject("ADL-002", actor="human", reason="Still no")
        assert rejected_again.current_status == ProposalStatus.REJECTED

    def test_persistence_roundtrip(self, tmp_path):
        storage = tmp_path / "queue.json"
        queue = ProposalQueue(storage_path=str(storage))
        proposal = ImprovementProposal(
            id="ADL-003", title="T", problem="P", recommendation="R"
        )
        queue.submit(proposal, priority=3, tags=["architect", "urgent"])
        queue.approve("ADL-003")

        # Load into fresh queue
        queue2 = ProposalQueue(storage_path=str(storage))
        queue2.load_from_memory()
        loaded = queue2.get("ADL-003")
        assert loaded is not None
        assert loaded.current_status == ProposalStatus.APPROVED
        assert loaded.priority == 3
        assert "architect" in loaded.tags

    def test_stats(self, tmp_path):
        queue = ProposalQueue(storage_path=str(tmp_path / "queue.json"))
        p1 = ImprovementProposal(id="p1", title="T1", problem="P1")
        p2 = ImprovementProposal(id="p2", title="T2", problem="P2")
        p3 = ImprovementProposal(id="p3", title="T3", problem="P3")

        queue.submit(p1)
        queue.submit(p2)
        queue.approve("p2")
        queue.submit(p3)
        queue.complete("p3")

        stats = queue.stats()
        assert stats["total"] == 3
        assert stats["pending"] == 1
        assert stats["approved"] == 1
        assert stats["complete"] == 1
        assert stats["rejected"] == 0


# ---------------------------------------------------------------------------
# CitizenCouncil tests
# ---------------------------------------------------------------------------


class TestCitizenCouncil:
    def test_collect_from_memory(self, tmp_path):
        mock_memory = MagicMock()
        proposal_dict = ImprovementProposal(
            id="ADL-001",
            title="T1",
            problem="P1",
            recommendation="R1",
            affected_components=["Factory"],
        ).to_dict()
        mock_memory.search.return_value = [
            {
                "content": "proposal 1",
                "metadata": proposal_dict,
            }
        ]

        council = CitizenCouncil(memory_layer=mock_memory)
        count = council.collect_from_memory(citizen_names=["architect"])
        assert count == 1
        assert "ADL-001" in council._proposals

    def test_rank_backlog_scoring(self, tmp_path):
        council = CitizenCouncil()
        p1 = ImprovementProposal(
            id="p1",
            title="High severity",
            problem="P1",
            confidence_score=0.9,
            estimated_effort_hours=2.0,
            affected_components=["Factory"],
            evidence=[
                EvidenceItem(source="test", description="Critical issue")
            ],
        )
        p2 = ImprovementProposal(
            id="p2",
            title="Low severity",
            problem="P2",
            confidence_score=0.3,
            estimated_effort_hours=8.0,
            affected_components=["Kernel"],
        )
        council._add_proposal(p1, source="architect")
        council._add_proposal(p2, source="test_oracle")

        ranked = council.rank_backlog(deduplicate=False)
        assert len(ranked) == 2
        assert ranked[0].proposal.id == "p1"
        assert ranked[0].rank == 1
        assert ranked[0].priority_score > ranked[1].priority_score

    def test_deduplication_by_component_overlap(self, tmp_path):
        council = CitizenCouncil()
        p1 = ImprovementProposal(
            id="p1",
            title="First",
            problem="P1",
            confidence_score=0.9,
            estimated_effort_hours=1.0,
            affected_components=["Factory", "Kernel"],
        )
        p2 = ImprovementProposal(
            id="p2",
            title="Second",
            problem="P2",
            confidence_score=0.8,
            estimated_effort_hours=1.0,
            affected_components=["Factory", "Mind"],
        )
        council._add_proposal(p1, source="architect")
        council._add_proposal(p2, source="conversation_designer")

        ranked = council.rank_backlog(deduplicate=True)
        assert len(ranked) == 1
        assert ranked[0].duplicates == ["p2"]
