"""Tests for SupervisorAgent skill library integration."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from animus_forge.agents.supervisor import SupervisorAgent
from animus_forge.skills.library import SkillLibrary
from animus_forge.skills.models import (
    SkillDefinition,
    SkillRouting,
)


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.complete = AsyncMock(return_value="Agent response")
    return provider


@pytest.fixture
def mock_skill_library():
    lib = MagicMock(spec=SkillLibrary)
    lib.build_routing_summary.return_value = (
        "# Available Skills\n\n"
        "## file-ops\nFilesystem operations\n"
        "**Use when:** Filesystem operations\n"
    )
    lib.build_skill_context.return_value = ""
    lib.find_skills_for_task.return_value = []
    return lib


class TestSupervisorSkillLibraryInit:
    def test_accepts_skill_library(self, mock_provider, mock_skill_library):
        sup = SupervisorAgent(provider=mock_provider, skill_library=mock_skill_library)
        assert sup._skill_library is mock_skill_library

    def test_skill_library_defaults_to_none(self, mock_provider):
        sup = SupervisorAgent(provider=mock_provider)
        assert sup._skill_library is None


class TestBuildSystemPromptWithSkills:
    def test_includes_routing_summary(self, mock_provider, mock_skill_library):
        sup = SupervisorAgent(provider=mock_provider, skill_library=mock_skill_library)
        prompt = sup._build_system_prompt()
        assert "Available Skills" in prompt
        assert "file-ops" in prompt

    def test_no_skill_library_no_routing(self, mock_provider):
        sup = SupervisorAgent(provider=mock_provider)
        prompt = sup._build_system_prompt()
        assert "Available Skills" not in prompt

    def test_empty_routing_summary_not_appended(self, mock_provider, mock_skill_library):
        mock_skill_library.build_routing_summary.return_value = ""
        sup = SupervisorAgent(provider=mock_provider, skill_library=mock_skill_library)
        prompt = sup._build_system_prompt()
        # Should not add extra newlines for empty summary
        assert not prompt.endswith("\n\n")


class TestGetAgentPromptWithSkills:
    def test_injects_skill_context(self, mock_provider, mock_skill_library):
        mock_skill_library.build_skill_context.return_value = (
            "# Skills for system agent\n\n## file-ops\nFile operations\n"
        )
        sup = SupervisorAgent(provider=mock_provider, skill_library=mock_skill_library)
        prompt = sup._get_agent_prompt("system")
        # Should have the base prompt + skill context
        assert "Skills for system agent" in prompt
        mock_skill_library.build_skill_context.assert_called_with("system")

    def test_no_skill_context_when_empty(self, mock_provider, mock_skill_library):
        mock_skill_library.build_skill_context.return_value = ""
        sup = SupervisorAgent(provider=mock_provider, skill_library=mock_skill_library)
        prompt = sup._get_agent_prompt("builder")
        # Should just be the base builder prompt
        assert "Builder agent" in prompt
        assert "Skills for" not in prompt

    def test_no_skill_library_returns_base_prompt(self, mock_provider):
        sup = SupervisorAgent(provider=mock_provider)
        prompt = sup._get_agent_prompt("planner")
        assert "Planning agent" in prompt


class TestExecuteDelegationsConsensus:
    @pytest.fixture
    def supervisor_with_skills(self, mock_provider):
        lib = MagicMock(spec=SkillLibrary)
        lib.find_skills_for_task.return_value = [
            SkillDefinition(
                name="secure-ops",
                agent="system",
                consensus_level="unanimous",
                routing=SkillRouting(use_when=["secure operations"]),
            )
        ]
        lib.build_skill_context.return_value = ""
        sup = SupervisorAgent(provider=mock_provider, skill_library=lib)
        return sup

    def test_consensus_annotation_added(self, supervisor_with_skills, mock_provider):
        """When a matched skill requires non-any consensus, the delegation is annotated."""
        import asyncio

        delegations = [{"agent": "system", "task": "Perform secure operations on files"}]
        context = [{"role": "system", "content": "test"}]

        asyncio.run(
            supervisor_with_skills._execute_delegations(delegations, context, lambda x: None)
        )

        # Check that the delegation was annotated with consensus info
        assert delegations[0].get("_skill_consensus") == "unanimous"
        assert delegations[0].get("_skill_name") == "secure-ops"

    def test_no_consensus_annotation_for_any(self, mock_provider):
        """Skills with consensus_level='any' don't get annotated."""
        import asyncio

        lib = MagicMock(spec=SkillLibrary)
        lib.find_skills_for_task.return_value = [
            SkillDefinition(
                name="basic-ops",
                agent="system",
                consensus_level="any",
                routing=SkillRouting(use_when=["basic operations"]),
            )
        ]
        lib.build_skill_context.return_value = ""
        sup = SupervisorAgent(provider=mock_provider, skill_library=lib)

        delegations = [{"agent": "system", "task": "basic operations"}]
        context = [{"role": "system", "content": "test"}]

        asyncio.run(sup._execute_delegations(delegations, context, lambda x: None))
        assert "_skill_consensus" not in delegations[0]

    def test_no_library_no_annotation(self, mock_provider):
        """Without a skill library, delegations are not annotated."""
        import asyncio

        sup = SupervisorAgent(provider=mock_provider)
        delegations = [{"agent": "system", "task": "anything"}]
        context = [{"role": "system", "content": "test"}]

        asyncio.run(sup._execute_delegations(delegations, context, lambda x: None))
        assert "_skill_consensus" not in delegations[0]
