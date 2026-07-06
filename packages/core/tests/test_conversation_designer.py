"""Tests for ConversationDesignerCitizen."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from animus.citizens import (
    ConversationDesignerCitizen,
    ImprovementProposal,
    ProposalStatus,
)


class TestConversationDesignerCitizen:
    def test_initialization(self):
        designer = ConversationDesignerCitizen(conversation_log_dir="/tmp/logs")
        assert designer.conversation_log_dir == Path("/tmp/logs")
        assert designer._patterns == []

    def test_initialization_no_log_dir(self):
        designer = ConversationDesignerCitizen()
        assert designer.conversation_log_dir is None

    # ------------------------------------------------------------------
    # observe_repeated_prompts
    # ------------------------------------------------------------------

    def test_observe_repeated_prompts_no_logs(self, tmp_path):
        designer = ConversationDesignerCitizen(conversation_log_dir=tmp_path / "nonexistent")
        observations = designer.observe_repeated_prompts()
        # No logs configured — gracefully returns empty, not a false-positive finding.
        assert len(observations) == 0

    def test_observe_repeated_prompts_finds_patterns(self, tmp_path):
        log_dir = tmp_path / "conversations"
        log_dir.mkdir()

        # Create logs with repeated prompts
        for i in range(5):
            log_file = log_dir / f"session_{i}.jsonl"
            entries = [
                {"prompt": "How do I configure the API?"},
                {"prompt": "How do I configure the API?"},  # repeated
                {"prompt": "What is the status of the build?"},
                {"prompt": "How do I configure the API?"},  # repeated again
            ]
            log_file.write_text("\n".join(json.dumps(e) for e in entries))

        designer = ConversationDesignerCitizen(conversation_log_dir=log_dir)
        observations = designer.observe_repeated_prompts()

        assert len(observations) >= 1
        assert observations[0].source == "conversation"
        assert "Repeated prompt detected" in observations[0].description
        assert observations[0].context["count"] >= 3

    def test_observe_repeated_prompts_ignores_short_prompts(self, tmp_path):
        log_dir = tmp_path / "conversations"
        log_dir.mkdir()

        log_file = log_dir / "session.jsonl"
        entries = [{"prompt": "hi"} for _ in range(10)]  # short, should be ignored
        log_file.write_text("\n".join(json.dumps(e) for e in entries))

        designer = ConversationDesignerCitizen(conversation_log_dir=log_dir)
        observations = designer.observe_repeated_prompts()
        # Should only get the "not configured" fallback if no patterns found
        # But short prompts are filtered out, so we should get no repeated patterns
        assert not any("Repeated prompt" in o.description for o in observations)

    # ------------------------------------------------------------------
    # observe_vague_requests
    # ------------------------------------------------------------------

    def test_observe_vague_requests_no_logs(self, tmp_path):
        designer = ConversationDesignerCitizen()
        observations = designer.observe_vague_requests()
        assert observations == []

    def test_observe_vague_requests_finds_patterns(self, tmp_path):
        log_dir = tmp_path / "conversations"
        log_dir.mkdir()

        log_file = log_dir / "session.jsonl"
        entries = [
            {"prompt": "Help me"},
            {"prompt": "Fix this"},
            {"prompt": "Help me"},
            {"prompt": "How do I do this?"},
        ]
        log_file.write_text("\n".join(json.dumps(e) for e in entries))

        designer = ConversationDesignerCitizen(conversation_log_dir=log_dir)
        observations = designer.observe_vague_requests()

        assert len(observations) >= 1
        vague_obs = [o for o in observations if "Vague request" in o.description]
        assert len(vague_obs) >= 1
        assert vague_obs[0].context["pattern_type"] == "vague_request"

    # ------------------------------------------------------------------
    # observe_correction_loops
    # ------------------------------------------------------------------

    def test_observe_correction_loops_no_logs(self, tmp_path):
        designer = ConversationDesignerCitizen()
        observations = designer.observe_correction_loops()
        assert observations == []

    def test_observe_correction_loops_finds_patterns(self, tmp_path):
        log_dir = tmp_path / "conversations"
        log_dir.mkdir()

        log_file = log_dir / "session.jsonl"
        entries = [
            {"prompt": "Do X"},
            {"prompt": "No, I meant do Y"},
            {"prompt": "Actually, do Z instead"},
            {"prompt": "No, that's wrong — do W"},
            {"prompt": "Nope"},
            {"prompt": "Incorrect"},
        ]
        log_file.write_text("\n".join(json.dumps(e) for e in entries))

        designer = ConversationDesignerCitizen(conversation_log_dir=log_dir)
        observations = designer.observe_correction_loops()

        assert len(observations) == 1
        assert "Correction loop" in observations[0].description
        assert observations[0].severity == "high"
        assert observations[0].context["correction_count"] == 5

    # ------------------------------------------------------------------
    # analyze
    # ------------------------------------------------------------------

    def test_analyze_no_patterns(self, tmp_path):
        designer = ConversationDesignerCitizen(conversation_log_dir=tmp_path / "nonexistent")
        patterns = designer.analyze()
        # No logs configured — gracefully returns empty.
        assert len(patterns) == 0

    def test_analyze_with_multiple_patterns(self, tmp_path):
        log_dir = tmp_path / "conversations"
        log_dir.mkdir()

        # Mix of vague + correction patterns
        log_file = log_dir / "session.jsonl"
        entries = [
            {"prompt": "Help me with this"},
            {"prompt": "Help me with this"},
            {"prompt": "No, I meant something else"},
            {"prompt": "Actually, do this"},
            {"prompt": "That's not right"},
        ]
        log_file.write_text("\n".join(json.dumps(e) for e in entries))

        designer = ConversationDesignerCitizen(conversation_log_dir=log_dir)
        patterns = designer.analyze()

        assert len(patterns) >= 2  # vague + correction
        pattern_types = {p.pattern_type for p in patterns}
        assert "vague_request" in pattern_types
        assert "correction_loop" in pattern_types

    # ------------------------------------------------------------------
    # generate_proposal
    # ------------------------------------------------------------------

    def test_generate_proposal_no_patterns(self, tmp_path):
        designer = ConversationDesignerCitizen(conversation_log_dir=tmp_path / "nonexistent")
        # Patch analyze to return empty
        with patch.object(designer, "analyze", return_value=[]):
            proposal = designer.generate_proposal()
        assert proposal is None

    def test_generate_proposal_with_patterns(self, tmp_path):
        log_dir = tmp_path / "conversations"
        log_dir.mkdir()

        log_file = log_dir / "session.jsonl"
        entries = [
            {"prompt": "How do I configure the API?"},
            {"prompt": "How do I configure the API?"},
            {"prompt": "How do I configure the API?"},
        ]
        log_file.write_text("\n".join(json.dumps(e) for e in entries))

        designer = ConversationDesignerCitizen(conversation_log_dir=log_dir)
        proposal = designer.generate_proposal()

        assert proposal is not None
        assert proposal.status == ProposalStatus.DRAFT
        assert isinstance(proposal.id, str)
        assert proposal.id.startswith("ADL-")
        assert len(proposal.evidence) >= 1
        assert "conversation" in proposal.evidence[0].source
        assert len(proposal.potential_risks) == 2
        assert proposal.affected_components == ["Mind", "Society"]

    def test_generate_proposal_for_correction_loop(self, tmp_path):
        log_dir = tmp_path / "conversations"
        log_dir.mkdir()

        log_file = log_dir / "session.jsonl"
        entries = [
            {"prompt": "Do X"},
            {"prompt": "No, that's wrong — do Y"},
            {"prompt": "Actually I meant Z"},
            {"prompt": "Nope, try W"},
            {"prompt": "Wrong again"},
        ]
        log_file.write_text("\n".join(json.dumps(e) for e in entries))

        designer = ConversationDesignerCitizen(conversation_log_dir=log_dir)
        proposal = designer.generate_proposal()

        assert proposal is not None
        assert proposal.affected_components == ["Mind", "Factory"]
        assert "correction" in proposal.problem.lower() or "loop" in proposal.problem.lower()

    # ------------------------------------------------------------------
    # store_proposal
    # ------------------------------------------------------------------

    def test_store_proposal_without_memory(self):
        designer = ConversationDesignerCitizen()
        proposal = ImprovementProposal(id="1", title="T", problem="P")
        assert designer.store_proposal(proposal) is False

    def test_store_proposal_with_memory(self):
        mock_memory = MagicMock()
        designer = ConversationDesignerCitizen(memory_layer=mock_memory)
        proposal = ImprovementProposal(id="1", title="T", problem="P", recommendation="R")

        assert designer.store_proposal(proposal) is True
        mock_memory.remember.assert_called_once()
        call_kwargs = mock_memory.remember.call_args.kwargs
        assert "conversation_designer" in call_kwargs["tags"]
        assert "proposal" in call_kwargs["tags"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def test_aggregate_severity(self):
        from animus.citizens.architect import Observation

        obs = [
            Observation(source="a", description="d", severity="low"),
            Observation(source="a", description="d", severity="high"),
            Observation(source="a", description="d", severity="medium"),
        ]
        assert ConversationDesignerCitizen._aggregate_severity(obs) == "high"

    def test_aggregate_severity_empty(self):
        assert ConversationDesignerCitizen._aggregate_severity([]) == "low"

    def test_suggest_for_pattern(self):
        from animus.citizens.architect import Observation

        obs = Observation(source="a", description="d", severity="low")
        assert "shortcut" in ConversationDesignerCitizen._suggest_for_pattern("repeated_prompt", obs)
        assert "clarifying" in ConversationDesignerCitizen._suggest_for_pattern("vague_request", obs)
        assert "confirmation" in ConversationDesignerCitizen._suggest_for_pattern("correction_loop", obs)

    def test_build_problem_recommendation(self):
        from animus.citizens.conversation_designer import ConversationPattern

        pattern = ConversationPattern(
            pattern_type="repeated_prompt",
            description="Users repeatedly ask how to configure API",
            frequency=5,
            example="How do I configure the API?",
            suggestion="Add a shortcut",
            severity="medium",
        )
        problem, recommendation = ConversationDesignerCitizen._build_problem_recommendation(pattern)
        assert "repeatedly ask" in problem
        assert "command" in recommendation or "template" in recommendation

    def test_repr(self):
        designer = ConversationDesignerCitizen()
        assert "ConversationDesignerCitizen" in repr(designer)
