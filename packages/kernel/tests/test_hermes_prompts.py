"""Tests for Hermes-format prompt loader."""

from __future__ import annotations

import pytest

from animus_kernel.agents.prompts.hermes import get_role_prompt


class TestGetRolePrompt:
    @pytest.mark.parametrize(
        "role",
        ["planner", "builder", "tester", "reviewer", "architect", "documenter"],
    )
    def test_loads_known_roles(self, role):
        prompt = get_role_prompt(role)
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "<system>" in prompt or "</system>" in prompt

    def test_case_insensitive(self):
        lower = get_role_prompt("builder")
        upper = get_role_prompt("BUILDER")
        assert lower == upper

    def test_unknown_role_raises(self):
        with pytest.raises(ValueError, match="No Hermes prompt for role"):
            get_role_prompt("nonexistent")
