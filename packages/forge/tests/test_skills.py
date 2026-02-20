"""Tests for skill loading, querying, consensus voting, and enforcement."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_forge.skills import SkillCapability, SkillDefinition, SkillLibrary
from animus_forge.skills.consensus import (
    NUM_VOTERS,
    VOTER_PROMPTS,
    ConsensusLevel,
    ConsensusVerdict,
    ConsensusVoter,
    Vote,
    VoteDecision,
    consensus_level_order,
)
from animus_forge.skills.enforcer import (
    _PATH_RE,
    EnforcementAction,
    EnforcementResult,
    SkillEnforcer,
    Violation,
    ViolationType,
)
from animus_forge.skills.loader import load_registry, load_skill

SKILLS_DIR = Path(__file__).parent.parent / "skills"


@pytest.fixture
def library():
    return SkillLibrary(skills_dir=SKILLS_DIR)


@pytest.fixture
def registry():
    return load_registry(SKILLS_DIR)


# --- Loader tests ---


class TestLoadSkill:
    def test_load_file_operations(self):
        skill = load_skill(SKILLS_DIR / "system" / "file_operations")
        assert skill.name == "file_operations"
        assert skill.agent == "system"
        assert skill.version == "1.0.0"
        assert len(skill.capabilities) > 0
        assert skill.skill_doc  # SKILL.md loaded

    def test_load_web_search(self):
        skill = load_skill(SKILLS_DIR / "browser" / "web_search")
        assert skill.name == "web_search"
        assert skill.agent == "browser"

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_skill(tmp_path / "nonexistent")


class TestLoadRegistry:
    def test_registry_loads_all_skills(self, registry):
        assert registry.version == "1.0.0"
        assert len(registry.skills) >= 5  # 6 active skills in registry

    def test_registry_has_categories(self, registry):
        assert "system" in registry.categories
        assert "browser" in registry.categories

    def test_registry_has_consensus_levels(self, registry):
        assert "any" in registry.consensus_levels
        assert "unanimous" in registry.consensus_levels


# --- Model tests ---


class TestSkillDefinition:
    def test_get_capability(self):
        cap = SkillCapability(name="read_file", description="Read a file")
        skill = SkillDefinition(name="test", agent="system", capabilities=[cap])
        assert skill.get_capability("read_file") is not None
        assert skill.get_capability("nonexistent") is None


class TestSkillRegistry:
    def test_get_skill(self, registry):
        assert registry.get_skill("file_operations") is not None
        assert registry.get_skill("nonexistent_skill") is None

    def test_get_skills_for_agent(self, registry):
        system_skills = registry.get_skills_for_agent("system")
        assert len(system_skills) >= 2
        for s in system_skills:
            assert s.agent == "system"

        browser_skills = registry.get_skills_for_agent("browser")
        assert len(browser_skills) >= 2
        for s in browser_skills:
            assert s.agent == "browser"


# --- Library tests ---


class TestSkillLibrary:
    def test_get_skill(self, library):
        skill = library.get_skill("file_operations")
        assert skill is not None
        assert skill.name == "file_operations"

    def test_get_skill_missing(self, library):
        assert library.get_skill("does_not_exist") is None

    def test_get_skills_for_agent(self, library):
        skills = library.get_skills_for_agent("system")
        assert len(skills) >= 2
        names = [s.name for s in skills]
        assert "file_operations" in names
        assert "process_management" in names

    def test_get_capabilities(self, library):
        caps = library.get_capabilities("file_operations")
        assert len(caps) > 0
        cap_names = [c.name for c in caps]
        assert "read_file" in cap_names
        assert "delete_file" in cap_names

    def test_get_capabilities_missing_skill(self, library):
        assert library.get_capabilities("nonexistent") == []

    def test_get_consensus_level(self, library):
        assert library.get_consensus_level("file_operations", "read_file") == "any"
        assert library.get_consensus_level("file_operations", "delete_file") == "unanimous"

    def test_get_consensus_level_missing(self, library):
        assert library.get_consensus_level("nonexistent", "read_file") is None
        assert library.get_consensus_level("file_operations", "nonexistent") is None

    def test_build_skill_context(self, library):
        ctx = library.build_skill_context("system")
        assert "system agent" in ctx
        assert "file_operations" in ctx
        assert "read_file" in ctx
        assert "Protected paths" in ctx

    def test_build_skill_context_empty_agent(self, library):
        assert library.build_skill_context("nonexistent_agent") == ""


# --- Consensus tests ---


class TestConsensusLevelOrder:
    @pytest.mark.parametrize(
        "level, expected",
        [
            (ConsensusLevel.ANY, 0),
            (ConsensusLevel.MAJORITY, 1),
            (ConsensusLevel.UNANIMOUS, 2),
            (ConsensusLevel.UNANIMOUS_PLUS_USER, 3),
        ],
    )
    def test_order_from_enum(self, level, expected):
        assert consensus_level_order(level) == expected

    @pytest.mark.parametrize(
        "level_str, expected",
        [
            ("any", 0),
            ("majority", 1),
            ("unanimous", 2),
            ("unanimous_plus_user", 3),
        ],
    )
    def test_order_from_string(self, level_str, expected):
        assert consensus_level_order(level_str) == expected

    def test_invalid_string_returns_negative(self):
        assert consensus_level_order("bogus") == -1


class TestVoteDecision:
    def test_enum_values(self):
        assert VoteDecision.APPROVE.value == "approve"
        assert VoteDecision.REJECT.value == "reject"
        assert VoteDecision.ABSTAIN.value == "abstain"


class TestConsensusLevelEnum:
    def test_enum_values(self):
        assert ConsensusLevel.ANY.value == "any"
        assert ConsensusLevel.MAJORITY.value == "majority"
        assert ConsensusLevel.UNANIMOUS.value == "unanimous"
        assert ConsensusLevel.UNANIMOUS_PLUS_USER.value == "unanimous_plus_user"


class TestVote:
    def test_defaults(self):
        vote = Vote(voter_id=0, decision=VoteDecision.APPROVE)
        assert vote.reasoning == ""
        assert vote.error == ""

    def test_full_construction(self):
        vote = Vote(
            voter_id=1,
            decision=VoteDecision.REJECT,
            reasoning="Too risky",
            error="",
        )
        assert vote.voter_id == 1
        assert vote.decision == VoteDecision.REJECT
        assert vote.reasoning == "Too risky"


class TestConsensusVerdict:
    def test_approve_count_includes_abstains(self):
        votes = [
            Vote(voter_id=0, decision=VoteDecision.APPROVE),
            Vote(voter_id=1, decision=VoteDecision.ABSTAIN),
            Vote(voter_id=2, decision=VoteDecision.REJECT),
        ]
        verdict = ConsensusVerdict(level=ConsensusLevel.MAJORITY, approved=True, votes=votes)
        assert verdict.approve_count == 2  # APPROVE + ABSTAIN
        assert verdict.reject_count == 1

    def test_all_approve(self):
        votes = [Vote(voter_id=i, decision=VoteDecision.APPROVE) for i in range(3)]
        verdict = ConsensusVerdict(level=ConsensusLevel.UNANIMOUS, approved=True, votes=votes)
        assert verdict.approve_count == 3
        assert verdict.reject_count == 0

    def test_all_reject(self):
        votes = [Vote(voter_id=i, decision=VoteDecision.REJECT) for i in range(3)]
        verdict = ConsensusVerdict(level=ConsensusLevel.UNANIMOUS, approved=False, votes=votes)
        assert verdict.approve_count == 0
        assert verdict.reject_count == 3

    def test_to_dict(self):
        votes = [
            Vote(voter_id=0, decision=VoteDecision.APPROVE, reasoning="Safe"),
            Vote(voter_id=1, decision=VoteDecision.REJECT, reasoning="Risky"),
        ]
        verdict = ConsensusVerdict(
            level=ConsensusLevel.MAJORITY,
            approved=True,
            votes=votes,
            requires_user_confirmation=False,
        )
        d = verdict.to_dict()
        assert d["level"] == "majority"
        assert d["approved"] is True
        assert d["approve_count"] == 1
        assert d["reject_count"] == 1
        assert d["requires_user_confirmation"] is False
        assert len(d["votes"]) == 2
        assert d["votes"][0]["decision"] == "approve"
        assert d["votes"][1]["reasoning"] == "Risky"

    def test_to_dict_with_user_confirmation(self):
        verdict = ConsensusVerdict(
            level=ConsensusLevel.UNANIMOUS_PLUS_USER,
            approved=True,
            votes=[],
            requires_user_confirmation=True,
        )
        d = verdict.to_dict()
        assert d["requires_user_confirmation"] is True


class TestParseVote:
    def test_approve(self):
        vote = ConsensusVoter._parse_vote(0, "APPROVE\nLooks safe to me.")
        assert vote.decision == VoteDecision.APPROVE
        assert vote.reasoning == "Looks safe to me."
        assert vote.voter_id == 0

    def test_reject(self):
        vote = ConsensusVoter._parse_vote(1, "REJECT\nToo dangerous.")
        assert vote.decision == VoteDecision.REJECT
        assert vote.reasoning == "Too dangerous."

    def test_approve_case_insensitive(self):
        vote = ConsensusVoter._parse_vote(0, "approve\nOK")
        assert vote.decision == VoteDecision.APPROVE

    def test_reject_case_insensitive(self):
        vote = ConsensusVoter._parse_vote(0, "Reject\nNope")
        assert vote.decision == VoteDecision.REJECT

    def test_approve_no_reasoning(self):
        vote = ConsensusVoter._parse_vote(0, "APPROVE")
        assert vote.decision == VoteDecision.APPROVE
        assert vote.reasoning == ""

    def test_empty_output_abstains(self):
        vote = ConsensusVoter._parse_vote(2, "")
        assert vote.decision == VoteDecision.ABSTAIN
        assert vote.error == "empty output"

    def test_whitespace_only_abstains(self):
        vote = ConsensusVoter._parse_vote(0, "   \n  ")
        assert vote.decision == VoteDecision.ABSTAIN
        assert vote.error == "empty output"

    def test_unparseable_output_abstains(self):
        vote = ConsensusVoter._parse_vote(0, "MAYBE\nI'm not sure")
        assert vote.decision == VoteDecision.ABSTAIN
        assert vote.error == "unparseable"
        assert "MAYBE" in vote.reasoning

    def test_extra_whitespace_trimmed(self):
        vote = ConsensusVoter._parse_vote(0, "  APPROVE  \n  Looks good  ")
        assert vote.decision == VoteDecision.APPROVE
        assert vote.reasoning == "Looks good"


class TestGetVoterPrompt:
    def test_voter_0_safety(self):
        prompt = ConsensusVoter._get_voter_prompt(0)
        assert "SAFETY" in prompt

    def test_voter_1_correctness(self):
        prompt = ConsensusVoter._get_voter_prompt(1)
        assert "CORRECTNESS" in prompt

    def test_voter_2_alignment(self):
        prompt = ConsensusVoter._get_voter_prompt(2)
        assert "ALIGNMENT" in prompt

    def test_voter_id_wraps_around(self):
        # voter_id=3 wraps to index 0
        assert ConsensusVoter._get_voter_prompt(3) == VOTER_PROMPTS[0]
        assert ConsensusVoter._get_voter_prompt(4) == VOTER_PROMPTS[1]


class TestAggregate:
    def _make_votes(self, decisions):
        return [Vote(voter_id=i, decision=d) for i, d in enumerate(decisions)]

    # --- ANY level ---

    def test_any_one_approve(self):
        votes = self._make_votes([VoteDecision.APPROVE])
        verdict = ConsensusVoter._aggregate(ConsensusLevel.ANY, votes)
        assert verdict.approved is True
        assert verdict.requires_user_confirmation is False

    def test_any_one_reject(self):
        votes = self._make_votes([VoteDecision.REJECT])
        verdict = ConsensusVoter._aggregate(ConsensusLevel.ANY, votes)
        assert verdict.approved is False

    def test_any_one_abstain_counts_as_approve(self):
        votes = self._make_votes([VoteDecision.ABSTAIN])
        verdict = ConsensusVoter._aggregate(ConsensusLevel.ANY, votes)
        assert verdict.approved is True

    # --- MAJORITY level ---

    def test_majority_two_approve(self):
        votes = self._make_votes([VoteDecision.APPROVE, VoteDecision.APPROVE, VoteDecision.REJECT])
        verdict = ConsensusVoter._aggregate(ConsensusLevel.MAJORITY, votes)
        assert verdict.approved is True

    def test_majority_two_reject(self):
        votes = self._make_votes([VoteDecision.REJECT, VoteDecision.REJECT, VoteDecision.APPROVE])
        verdict = ConsensusVoter._aggregate(ConsensusLevel.MAJORITY, votes)
        assert verdict.approved is False

    def test_majority_abstain_counts_as_approve(self):
        votes = self._make_votes([VoteDecision.ABSTAIN, VoteDecision.APPROVE, VoteDecision.REJECT])
        verdict = ConsensusVoter._aggregate(ConsensusLevel.MAJORITY, votes)
        assert verdict.approved is True

    def test_majority_one_approve_one_abstain_one_reject(self):
        votes = self._make_votes([VoteDecision.APPROVE, VoteDecision.ABSTAIN, VoteDecision.REJECT])
        verdict = ConsensusVoter._aggregate(ConsensusLevel.MAJORITY, votes)
        assert verdict.approved is True  # 2 approves (1 real + 1 abstain)

    def test_majority_all_abstain(self):
        votes = self._make_votes([VoteDecision.ABSTAIN, VoteDecision.ABSTAIN, VoteDecision.ABSTAIN])
        verdict = ConsensusVoter._aggregate(ConsensusLevel.MAJORITY, votes)
        assert verdict.approved is True  # All 3 abstains count as approve

    # --- UNANIMOUS level ---

    def test_unanimous_all_approve(self):
        votes = self._make_votes([VoteDecision.APPROVE, VoteDecision.APPROVE, VoteDecision.APPROVE])
        verdict = ConsensusVoter._aggregate(ConsensusLevel.UNANIMOUS, votes)
        assert verdict.approved is True
        assert verdict.requires_user_confirmation is False

    def test_unanimous_one_reject(self):
        votes = self._make_votes([VoteDecision.APPROVE, VoteDecision.APPROVE, VoteDecision.REJECT])
        verdict = ConsensusVoter._aggregate(ConsensusLevel.UNANIMOUS, votes)
        assert verdict.approved is False

    def test_unanimous_abstain_counts_as_approve(self):
        votes = self._make_votes([VoteDecision.APPROVE, VoteDecision.APPROVE, VoteDecision.ABSTAIN])
        verdict = ConsensusVoter._aggregate(ConsensusLevel.UNANIMOUS, votes)
        assert verdict.approved is True

    # --- UNANIMOUS_PLUS_USER level ---

    def test_unanimous_plus_user_all_approve(self):
        votes = self._make_votes([VoteDecision.APPROVE, VoteDecision.APPROVE, VoteDecision.APPROVE])
        verdict = ConsensusVoter._aggregate(ConsensusLevel.UNANIMOUS_PLUS_USER, votes)
        assert verdict.approved is True
        assert verdict.requires_user_confirmation is True

    def test_unanimous_plus_user_one_reject(self):
        votes = self._make_votes([VoteDecision.APPROVE, VoteDecision.APPROVE, VoteDecision.REJECT])
        verdict = ConsensusVoter._aggregate(ConsensusLevel.UNANIMOUS_PLUS_USER, votes)
        assert verdict.approved is False
        assert verdict.requires_user_confirmation is False  # Not approved, so no user confirm

    def test_unanimous_plus_user_level_stored(self):
        votes = self._make_votes([VoteDecision.APPROVE, VoteDecision.APPROVE, VoteDecision.APPROVE])
        verdict = ConsensusVoter._aggregate(ConsensusLevel.UNANIMOUS_PLUS_USER, votes)
        assert verdict.level == ConsensusLevel.UNANIMOUS_PLUS_USER


class TestConsensusVoterSync:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.generate_completion.return_value = {
            "success": True,
            "output": "APPROVE\nLooks safe.",
        }
        return client

    @pytest.fixture
    def voter(self, mock_client):
        return ConsensusVoter(mock_client)

    def test_vote_any_calls_one_voter(self, voter, mock_client):
        verdict = voter.vote("delete file", ConsensusLevel.ANY)
        assert mock_client.generate_completion.call_count == 1
        assert verdict.approved is True
        assert len(verdict.votes) == 1

    def test_vote_majority_calls_three_voters(self, voter, mock_client):
        verdict = voter.vote("delete file", ConsensusLevel.MAJORITY)
        assert mock_client.generate_completion.call_count == NUM_VOTERS
        assert verdict.approved is True
        assert len(verdict.votes) == NUM_VOTERS

    def test_vote_unanimous_calls_three_voters(self, voter, mock_client):
        _ = voter.vote("delete file", ConsensusLevel.UNANIMOUS)
        assert mock_client.generate_completion.call_count == NUM_VOTERS

    def test_vote_with_string_level(self, voter, mock_client):
        verdict = voter.vote("delete file", "majority")
        assert verdict.approved is True
        assert mock_client.generate_completion.call_count == NUM_VOTERS

    def test_vote_with_role(self, voter, mock_client):
        verdict = voter.vote("delete file", ConsensusLevel.ANY, role="builder")
        assert verdict.approved is True

    def test_vote_client_failure_returns_abstain(self, mock_client):
        mock_client.generate_completion.return_value = {
            "success": False,
            "error": "connection timeout",
        }
        voter = ConsensusVoter(mock_client)
        verdict = voter.vote("delete file", ConsensusLevel.ANY)
        assert verdict.votes[0].decision == VoteDecision.ABSTAIN
        assert "connection timeout" in verdict.votes[0].error

    def test_vote_no_output_returns_abstain(self, mock_client):
        mock_client.generate_completion.return_value = {
            "success": True,
            "output": "",
        }
        voter = ConsensusVoter(mock_client)
        verdict = voter.vote("delete file", ConsensusLevel.ANY)
        assert verdict.votes[0].decision == VoteDecision.ABSTAIN

    def test_vote_exception_returns_abstain(self, mock_client):
        mock_client.generate_completion.side_effect = RuntimeError("boom")
        voter = ConsensusVoter(mock_client)
        verdict = voter.vote("delete file", ConsensusLevel.ANY)
        assert verdict.votes[0].decision == VoteDecision.ABSTAIN
        assert "boom" in verdict.votes[0].error

    def test_vote_reject_output(self, mock_client):
        mock_client.generate_completion.return_value = {
            "success": True,
            "output": "REJECT\nThis is dangerous.",
        }
        voter = ConsensusVoter(mock_client)
        verdict = voter.vote("rm -rf /", ConsensusLevel.ANY)
        assert verdict.approved is False
        assert verdict.votes[0].decision == VoteDecision.REJECT

    def test_vote_mixed_results(self, mock_client):
        responses = [
            {"success": True, "output": "APPROVE\nSafe."},
            {"success": True, "output": "REJECT\nDangerous."},
            {"success": True, "output": "APPROVE\nOK."},
        ]
        mock_client.generate_completion.side_effect = responses
        voter = ConsensusVoter(mock_client)
        verdict = voter.vote("questionable op", ConsensusLevel.MAJORITY)
        assert verdict.approved is True  # 2 approve, 1 reject

    def test_vote_majority_denied(self, mock_client):
        responses = [
            {"success": True, "output": "REJECT\nNo."},
            {"success": True, "output": "REJECT\nNope."},
            {"success": True, "output": "APPROVE\nOK."},
        ]
        mock_client.generate_completion.side_effect = responses
        voter = ConsensusVoter(mock_client)
        verdict = voter.vote("bad op", ConsensusLevel.MAJORITY)
        assert verdict.approved is False


class TestConsensusVoterAsync:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.generate_completion_async = AsyncMock(
            return_value={
                "success": True,
                "output": "APPROVE\nLooks safe.",
            }
        )
        return client

    @pytest.fixture
    def voter(self, mock_client):
        return ConsensusVoter(mock_client)

    def test_vote_async_any(self, voter, mock_client):
        verdict = asyncio.run(voter.vote_async("delete file", ConsensusLevel.ANY))
        assert mock_client.generate_completion_async.call_count == 1
        assert verdict.approved is True

    def test_vote_async_majority(self, voter, mock_client):
        verdict = asyncio.run(voter.vote_async("delete file", ConsensusLevel.MAJORITY))
        assert mock_client.generate_completion_async.call_count == NUM_VOTERS
        assert verdict.approved is True

    def test_vote_async_with_string_level(self, voter, mock_client):
        verdict = asyncio.run(voter.vote_async("op", "unanimous"))
        assert verdict.approved is True

    def test_vote_async_failure_returns_abstain(self, mock_client):
        mock_client.generate_completion_async = AsyncMock(
            return_value={"success": False, "error": "timeout"}
        )
        voter = ConsensusVoter(mock_client)
        verdict = asyncio.run(voter.vote_async("delete file", ConsensusLevel.ANY))
        assert verdict.votes[0].decision == VoteDecision.ABSTAIN

    def test_vote_async_exception_returns_abstain(self, mock_client):
        mock_client.generate_completion_async = AsyncMock(side_effect=RuntimeError("async boom"))
        voter = ConsensusVoter(mock_client)
        verdict = asyncio.run(voter.vote_async("delete file", ConsensusLevel.ANY))
        assert verdict.votes[0].decision == VoteDecision.ABSTAIN
        assert "async boom" in verdict.votes[0].error


class TestConsensusVoterCostTracking:
    def test_cost_tracking_called_on_success(self):
        with patch("animus_forge.skills.consensus.ConsensusVoter._track_voter_cost") as mock_track:
            client = MagicMock()
            client.generate_completion.return_value = {
                "success": True,
                "output": "APPROVE\nOK.",
            }
            voter = ConsensusVoter(client)
            voter.vote("test op", ConsensusLevel.ANY, role="tester")
            mock_track.assert_called_once_with(0, "APPROVE\nOK.", "tester")

    def test_cost_tracking_does_not_fail_on_import_error(self):
        """_track_voter_cost swallows import errors gracefully."""
        with patch(
            "animus_forge.skills.consensus.ConsensusVoter._track_voter_cost",
            wraps=ConsensusVoter._track_voter_cost,
        ):
            # Should not raise even if cost_tracker is not importable
            ConsensusVoter._track_voter_cost(0, "APPROVE\nOK.", "test")


class TestConsensusAuditLog:
    def test_log_verdict_emits_audit_log(self):
        votes = [Vote(voter_id=0, decision=VoteDecision.APPROVE, reasoning="OK")]
        verdict = ConsensusVerdict(level=ConsensusLevel.ANY, approved=True, votes=votes)
        with patch("animus_forge.skills.consensus.audit_logger") as mock_logger:
            ConsensusVoter._log_verdict(verdict, "test op", "builder", 0.123)
            mock_logger.info.assert_called_once()
            args, kwargs = mock_logger.info.call_args
            assert args[0] == "consensus_vote"
            record = kwargs["extra"]["consensus"]
            assert record["role"] == "builder"
            assert record["approved"] is True
            assert record["elapsed_seconds"] == 0.123

    def test_log_verdict_truncates_operation(self):
        votes = [Vote(voter_id=0, decision=VoteDecision.APPROVE)]
        verdict = ConsensusVerdict(level=ConsensusLevel.ANY, approved=True, votes=votes)
        long_op = "x" * 500
        with patch("animus_forge.skills.consensus.audit_logger") as mock_logger:
            ConsensusVoter._log_verdict(verdict, long_op, "", 0.0)
            record = mock_logger.info.call_args[1]["extra"]["consensus"]
            assert len(record["operation"]) == 200


# --- Enforcer tests ---


@pytest.fixture
def mock_library():
    """Create a mock SkillLibrary with configurable skills."""
    lib = MagicMock(spec=SkillLibrary)
    return lib


def _make_skill(
    name="test_skill",
    agent="system",
    protected_paths=None,
    blocked_patterns=None,
):
    return SkillDefinition(
        name=name,
        agent=agent,
        protected_paths=protected_paths or [],
        blocked_patterns=blocked_patterns or [],
    )


class TestEnforcementEnums:
    def test_enforcement_action_values(self):
        assert EnforcementAction.ALLOW.value == "allow"
        assert EnforcementAction.WARN.value == "warn"
        assert EnforcementAction.BLOCK.value == "block"

    def test_violation_type_values(self):
        assert ViolationType.PROTECTED_PATH.value == "protected_path"
        assert ViolationType.BLOCKED_PATTERN.value == "blocked_pattern"


class TestViolation:
    def test_construction(self):
        v = Violation(
            type=ViolationType.PROTECTED_PATH,
            severity="high",
            message="Bad path",
            matched_text="/etc/passwd",
            skill="file_ops",
        )
        assert v.type == ViolationType.PROTECTED_PATH
        assert v.severity == "high"
        assert v.skill == "file_ops"

    def test_default_skill(self):
        v = Violation(
            type=ViolationType.BLOCKED_PATTERN,
            severity="critical",
            message="Blocked",
            matched_text="rm -rf",
        )
        assert v.skill == ""


class TestEnforcementResult:
    def test_allow_result_passed(self):
        result = EnforcementResult(action=EnforcementAction.ALLOW)
        assert result.passed is True
        assert result.has_violations is False

    def test_warn_result_not_passed(self):
        result = EnforcementResult(
            action=EnforcementAction.WARN,
            violations=[
                Violation(
                    type=ViolationType.PROTECTED_PATH,
                    severity="high",
                    message="path match",
                    matched_text="/root/.ssh",
                )
            ],
        )
        assert result.passed is False
        assert result.has_violations is True

    def test_block_result_not_passed(self):
        result = EnforcementResult(action=EnforcementAction.BLOCK)
        assert result.passed is False

    def test_to_dict(self):
        v = Violation(
            type=ViolationType.BLOCKED_PATTERN,
            severity="critical",
            message="Found blocked pattern",
            matched_text="rm -rf /",
            skill="file_ops",
        )
        result = EnforcementResult(action=EnforcementAction.BLOCK, violations=[v])
        d = result.to_dict()
        assert d["action"] == "block"
        assert d["passed"] is False
        assert len(d["violations"]) == 1
        assert d["violations"][0]["type"] == "blocked_pattern"
        assert d["violations"][0]["severity"] == "critical"
        assert d["violations"][0]["skill"] == "file_ops"

    def test_to_dict_no_violations(self):
        result = EnforcementResult(action=EnforcementAction.ALLOW)
        d = result.to_dict()
        assert d["violations"] == []
        assert d["passed"] is True


class TestPathRegex:
    """Test the _PATH_RE regex used to extract paths from output."""

    def test_absolute_path(self):
        matches = _PATH_RE.findall("Reading /etc/passwd now")
        assert "/etc/passwd" in matches

    def test_home_tilde_path(self):
        matches = _PATH_RE.findall("Saving to ~/documents/report.txt")
        assert "~/documents/report.txt" in matches

    def test_multiple_paths(self):
        text = "Copying /src/main.py to /dst/main.py"
        matches = _PATH_RE.findall(text)
        assert "/src/main.py" in matches
        assert "/dst/main.py" in matches

    def test_no_paths(self):
        matches = _PATH_RE.findall("This has no paths at all")
        assert matches == []

    def test_path_at_start_of_line(self):
        matches = _PATH_RE.findall("/usr/bin/python3")
        assert "/usr/bin/python3" in matches


class TestSkillEnforcerCheckOutput:
    @pytest.fixture
    def enforcer_with_skills(self):
        """Create an enforcer with a mock library returning specific skills."""

        def _make(protected_paths=None, blocked_patterns=None, role="builder"):
            lib = MagicMock(spec=SkillLibrary)
            skill = _make_skill(
                protected_paths=protected_paths or [],
                blocked_patterns=blocked_patterns or [],
            )
            lib.get_skills_for_agent.return_value = [skill]
            role_map = {role: ["system"]}
            return SkillEnforcer(lib, role_map)

        return _make

    def test_empty_output_always_allowed(self, mock_library):
        enforcer = SkillEnforcer(mock_library, {"builder": ["system"]})
        result = enforcer.check_output("builder", "")
        assert result.passed is True
        assert result.action == EnforcementAction.ALLOW

    def test_no_agent_mapping_always_allowed(self, mock_library):
        enforcer = SkillEnforcer(mock_library)
        result = enforcer.check_output("unknown_role", "some output")
        assert result.passed is True

    def test_no_violations_allows(self, enforcer_with_skills):
        enforcer = enforcer_with_skills(protected_paths=["/secret/*"])
        result = enforcer.check_output("builder", "Reading /public/data.txt")
        assert result.passed is True

    def test_protected_path_violation(self, enforcer_with_skills):
        enforcer = enforcer_with_skills(protected_paths=["/etc/passwd"])
        result = enforcer.check_output("builder", "I will read /etc/passwd now")
        assert result.passed is False
        assert result.action == EnforcementAction.WARN
        assert len(result.violations) == 1
        assert result.violations[0].type == ViolationType.PROTECTED_PATH
        assert result.violations[0].severity == "high"
        assert result.violations[0].matched_text == "/etc/passwd"

    def test_protected_path_fnmatch_wildcard(self, enforcer_with_skills):
        enforcer = enforcer_with_skills(protected_paths=["/etc/*"])
        result = enforcer.check_output("builder", "Checking /etc/shadow for permissions")
        assert result.passed is False
        assert result.violations[0].matched_text == "/etc/shadow"

    def test_protected_path_fnmatch_glob_star(self, enforcer_with_skills):
        enforcer = enforcer_with_skills(protected_paths=["/root/.ssh/*"])
        result = enforcer.check_output("builder", "Found key at /root/.ssh/id_rsa")
        assert result.passed is False

    def test_protected_path_exact_match(self, enforcer_with_skills):
        enforcer = enforcer_with_skills(protected_paths=["/var/log/syslog"])
        result = enforcer.check_output("builder", "Tailing /var/log/syslog for errors")
        assert result.passed is False

    def test_no_protected_paths_skips_check(self, enforcer_with_skills):
        enforcer = enforcer_with_skills(protected_paths=[])
        result = enforcer.check_output("builder", "Reading /etc/passwd now")
        assert result.passed is True

    def test_blocked_pattern_violation(self, enforcer_with_skills):
        enforcer = enforcer_with_skills(blocked_patterns=[r"rm\s+-rf\s+/"])
        result = enforcer.check_output("builder", "Running: rm -rf / to clean up")
        assert result.passed is False
        assert result.action == EnforcementAction.BLOCK
        assert len(result.violations) == 1
        assert result.violations[0].type == ViolationType.BLOCKED_PATTERN
        assert result.violations[0].severity == "critical"

    def test_blocked_pattern_no_match(self, enforcer_with_skills):
        enforcer = enforcer_with_skills(blocked_patterns=[r"rm\s+-rf\s+/"])
        result = enforcer.check_output("builder", "Using ls -la to list files")
        assert result.passed is True

    def test_no_blocked_patterns_skips_check(self, enforcer_with_skills):
        enforcer = enforcer_with_skills(blocked_patterns=[])
        result = enforcer.check_output("builder", "rm -rf / oops")
        assert result.passed is True

    def test_multiple_blocked_patterns(self, enforcer_with_skills):
        enforcer = enforcer_with_skills(blocked_patterns=[r"rm\s+-rf", r"DROP\s+TABLE"])
        result = enforcer.check_output("builder", "Running rm -rf and DROP TABLE users")
        assert result.action == EnforcementAction.BLOCK
        assert len(result.violations) == 2

    def test_blocked_pattern_blocks_even_with_path_violation(self, enforcer_with_skills):
        """When both path and pattern violations exist, action is BLOCK."""
        enforcer = enforcer_with_skills(
            protected_paths=["/etc/*"],
            blocked_patterns=[r"sudo\s+"],
        )
        result = enforcer.check_output("builder", "Running sudo cat /etc/shadow")
        assert result.action == EnforcementAction.BLOCK
        assert len(result.violations) == 2

    def test_path_only_violations_produce_warn(self, enforcer_with_skills):
        enforcer = enforcer_with_skills(
            protected_paths=["/etc/*"],
            blocked_patterns=[],
        )
        result = enforcer.check_output("builder", "Reading /etc/hosts")
        assert result.action == EnforcementAction.WARN

    def test_invalid_regex_pattern_skipped(self, enforcer_with_skills):
        """Invalid regex in blocked_patterns is skipped, not raised."""
        enforcer = enforcer_with_skills(blocked_patterns=["[invalid(regex"])
        result = enforcer.check_output("builder", "Some output text")
        assert result.passed is True

    def test_multiple_agents_checked(self, mock_library):
        skill1 = _make_skill(name="skill1", protected_paths=["/etc/*"])
        skill2 = _make_skill(name="skill2", blocked_patterns=[r"DROP\s+TABLE"])

        def get_skills(agent_name):
            if agent_name == "system":
                return [skill1]
            elif agent_name == "database":
                return [skill2]
            return []

        mock_library.get_skills_for_agent.side_effect = get_skills
        role_map = {"devops": ["system", "database"]}
        enforcer = SkillEnforcer(mock_library, role_map)

        result = enforcer.check_output("devops", "Running DROP TABLE on /etc/hosts")
        assert result.action == EnforcementAction.BLOCK
        assert len(result.violations) == 2


class TestSkillEnforcerPatternCache:
    def test_pattern_cached_after_first_compile(self, mock_library):
        skill = _make_skill(blocked_patterns=[r"rm\s+-rf"])
        mock_library.get_skills_for_agent.return_value = [skill]
        enforcer = SkillEnforcer(mock_library, {"builder": ["system"]})

        # First call compiles and caches
        enforcer.check_output("builder", "rm -rf /tmp")
        assert r"rm\s+-rf" in enforcer._pattern_cache

        # Second call reuses cache
        enforcer.check_output("builder", "rm -rf /var")
        # Verify pattern is still in cache (not recompiled)
        assert r"rm\s+-rf" in enforcer._pattern_cache

    def test_invalid_pattern_cached_as_none(self, mock_library):
        skill = _make_skill(blocked_patterns=["[bad(regex"])
        mock_library.get_skills_for_agent.return_value = [skill]
        enforcer = SkillEnforcer(mock_library, {"builder": ["system"]})

        enforcer.check_output("builder", "some output")
        assert enforcer._pattern_cache["[bad(regex"] is None

    def test_cached_none_pattern_reused(self, mock_library):
        skill = _make_skill(blocked_patterns=["[bad(regex"])
        mock_library.get_skills_for_agent.return_value = [skill]
        enforcer = SkillEnforcer(mock_library, {"builder": ["system"]})

        enforcer.check_output("builder", "output 1")
        enforcer.check_output("builder", "output 2")
        # The invalid pattern should be cached as None, not re-compiled
        assert "[bad(regex" in enforcer._pattern_cache


class TestSkillEnforcerProtectedPathEdgeCases:
    @pytest.fixture
    def enforcer_with_paths(self):
        def _make(paths):
            lib = MagicMock(spec=SkillLibrary)
            skill = _make_skill(protected_paths=paths)
            lib.get_skills_for_agent.return_value = [skill]
            return SkillEnforcer(lib, {"builder": ["system"]})

        return _make

    def test_tilde_home_path_protected(self, enforcer_with_paths):
        enforcer = enforcer_with_paths(["~/.ssh/*"])
        result = enforcer.check_output("builder", "Reading ~/.ssh/authorized_keys")
        assert result.passed is False

    def test_one_violation_per_path(self, enforcer_with_paths):
        """Even if a path matches multiple protected patterns, only one violation per path."""
        enforcer = enforcer_with_paths(["/etc/*", "/etc/shadow"])
        result = enforcer.check_output("builder", "Reading /etc/shadow")
        # Should match first pattern and break
        assert len(result.violations) == 1

    def test_multiple_paths_in_output(self, enforcer_with_paths):
        enforcer = enforcer_with_paths(["/etc/*"])
        result = enforcer.check_output("builder", "Reading /etc/passwd and /etc/shadow")
        assert len(result.violations) == 2

    def test_path_not_in_protected_set(self, enforcer_with_paths):
        enforcer = enforcer_with_paths(["/etc/shadow"])
        result = enforcer.check_output("builder", "Reading /var/log/syslog")
        assert result.passed is True
