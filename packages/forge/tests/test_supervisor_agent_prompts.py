"""Tests for file-based agent prompt loading in SupervisorAgent."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import animus_forge.agents.supervisor as supervisor_module
from animus_forge.agents.supervisor import (
    SupervisorAgent,
    _load_agent_prompts,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset the module-level prompt cache between tests."""
    supervisor_module._agent_prompts_cache = None
    yield
    supervisor_module._agent_prompts_cache = None


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.complete = AsyncMock(return_value="Agent response")
    return provider


@pytest.fixture
def prompts_file(tmp_path):
    """Create a temporary agent_prompts.json."""
    data = {
        "planner": {
            "name": "Custom Planner",
            "description": "Custom planning agent",
            "system_prompt": "You are a custom planning agent from file.",
        },
        "builder": {
            "name": "Custom Builder",
            "description": "Custom builder agent",
            "system_prompt": "You are a custom builder agent from file.",
        },
    }
    path = tmp_path / "agent_prompts.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestLoadAgentPrompts:
    """Tests for the _load_agent_prompts function."""

    def test_loads_valid_json_file(self, prompts_file):
        result = _load_agent_prompts(path=prompts_file)
        assert "planner" in result
        assert result["planner"]["system_prompt"] == "You are a custom planning agent from file."

    def test_returns_empty_dict_for_missing_file(self, tmp_path):
        missing = tmp_path / "nonexistent.json"
        result = _load_agent_prompts(path=missing)
        assert result == {}

    def test_returns_empty_dict_for_invalid_json(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{invalid json", encoding="utf-8")
        result = _load_agent_prompts(path=bad_file)
        assert result == {}

    def test_returns_empty_dict_for_non_dict_json(self, tmp_path):
        list_file = tmp_path / "list.json"
        list_file.write_text('["not", "a", "dict"]', encoding="utf-8")
        result = _load_agent_prompts(path=list_file)
        assert result == {}

    def test_caches_result_on_default_path(self):
        """When called without explicit path, result is cached."""
        with patch.object(supervisor_module, "_AGENT_PROMPTS_PATH") as mock_path:
            mock_path.read_text.return_value = json.dumps({"test": {"system_prompt": "cached"}})
            result1 = _load_agent_prompts()
            assert "test" in result1
            # Second call should use cache, not re-read
            result2 = _load_agent_prompts()
            assert result2 is result1
            mock_path.read_text.assert_called_once()

    def test_explicit_path_bypasses_cache(self, prompts_file):
        """Explicit path never uses or sets the module cache."""
        result1 = _load_agent_prompts(path=prompts_file)
        assert supervisor_module._agent_prompts_cache is None
        result2 = _load_agent_prompts(path=prompts_file)
        assert result1 == result2

    def test_caches_empty_on_file_error(self):
        """When default path fails, cache is set to empty dict."""
        with patch.object(supervisor_module, "_AGENT_PROMPTS_PATH") as mock_path:
            mock_path.read_text.side_effect = OSError("not found")
            result = _load_agent_prompts()
            assert result == {}
            assert supervisor_module._agent_prompts_cache == {}


class TestGetAgentPromptFromFile:
    """Tests for _get_agent_prompt loading from file."""

    def test_uses_file_prompt_when_role_found(self, mock_provider, prompts_file):
        with patch.object(supervisor_module, "_AGENT_PROMPTS_PATH", prompts_file):
            sup = SupervisorAgent(provider=mock_provider)
            prompt = sup._get_agent_prompt("planner")
            assert "custom planning agent from file" in prompt

    def test_falls_back_to_hardcoded_when_role_not_in_file(self, mock_provider, prompts_file):
        """Role exists in hardcoded but not in file -> hardcoded used."""
        with patch.object(supervisor_module, "_AGENT_PROMPTS_PATH", prompts_file):
            sup = SupervisorAgent(provider=mock_provider)
            # "reviewer" is not in our test JSON
            prompt = sup._get_agent_prompt("reviewer")
            assert "Reviewer agent" in prompt

    def test_falls_back_to_hardcoded_when_file_missing(self, mock_provider, tmp_path):
        missing = tmp_path / "nonexistent.json"
        with patch.object(supervisor_module, "_AGENT_PROMPTS_PATH", missing):
            sup = SupervisorAgent(provider=mock_provider)
            prompt = sup._get_agent_prompt("planner")
            assert "Planning agent" in prompt

    def test_falls_back_for_unknown_role(self, mock_provider, prompts_file):
        with patch.object(supervisor_module, "_AGENT_PROMPTS_PATH", prompts_file):
            sup = SupervisorAgent(provider=mock_provider)
            prompt = sup._get_agent_prompt("unknown_role")
            assert "helpful unknown_role agent" in prompt

    def test_file_prompt_with_skill_library(self, mock_provider, prompts_file):
        """Skill context is appended to file-loaded prompts."""
        skill_lib = MagicMock()
        skill_lib.build_skill_context.return_value = "Extra skill context"
        skill_lib.build_routing_summary.return_value = ""
        skill_lib.find_skills_for_task.return_value = []
        with patch.object(supervisor_module, "_AGENT_PROMPTS_PATH", prompts_file):
            sup = SupervisorAgent(provider=mock_provider, skill_library=skill_lib)
            prompt = sup._get_agent_prompt("planner")
            assert "custom planning agent from file" in prompt
            assert "Extra skill context" in prompt

    def test_file_entry_missing_system_prompt_key_falls_back(self, mock_provider, tmp_path):
        """If JSON entry lacks 'system_prompt' key, fall back to hardcoded."""
        bad_data = {"planner": {"name": "No prompt here"}}
        path = tmp_path / "agent_prompts.json"
        path.write_text(json.dumps(bad_data), encoding="utf-8")
        with patch.object(supervisor_module, "_AGENT_PROMPTS_PATH", path):
            sup = SupervisorAgent(provider=mock_provider)
            prompt = sup._get_agent_prompt("planner")
            assert "Planning agent" in prompt

    def test_file_entry_not_dict_falls_back(self, mock_provider, tmp_path):
        """If JSON entry is a string instead of dict, fall back to hardcoded."""
        bad_data = {"planner": "just a string"}
        path = tmp_path / "agent_prompts.json"
        path.write_text(json.dumps(bad_data), encoding="utf-8")
        with patch.object(supervisor_module, "_AGENT_PROMPTS_PATH", path):
            sup = SupervisorAgent(provider=mock_provider)
            prompt = sup._get_agent_prompt("planner")
            assert "Planning agent" in prompt
