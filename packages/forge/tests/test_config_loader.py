"""Tests for animus_forge.agents.config_loader.

Covers load_agent_configs() with defaults, explicit paths, YAML merging,
invalid file handling, _find_config() search logic, new roles, and
partial overrides.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from animus_forge.agents.agent_config import DEFAULT_AGENT_CONFIGS
from animus_forge.agents.config_loader import _find_config, load_agent_configs


class TestLoadAgentConfigsDefaults:
    """Returns DEFAULT_AGENT_CONFIGS when no YAML is found."""

    def test_returns_defaults_when_no_yaml(self, tmp_path):
        """No YAML files in search dirs -> default configs."""
        result = load_agent_configs(base_dir=tmp_path)
        assert set(result.keys()) == set(DEFAULT_AGENT_CONFIGS.keys())
        for role, cfg in result.items():
            assert cfg.role == role

    def test_returns_dict_not_same_object(self, tmp_path):
        """Returned dict is a copy, not the global DEFAULT_AGENT_CONFIGS."""
        result = load_agent_configs(base_dir=tmp_path)
        assert result is not DEFAULT_AGENT_CONFIGS

    def test_returns_defaults_when_path_does_not_exist(self, tmp_path):
        """Explicit path to non-existent file -> defaults."""
        result = load_agent_configs(path=tmp_path / "missing.yaml")
        assert set(result.keys()) == set(DEFAULT_AGENT_CONFIGS.keys())


class TestLoadAgentConfigsExplicitPath:
    """Loads from explicit YAML path."""

    def test_loads_from_explicit_path(self, tmp_path):
        """load_agent_configs(path=...) reads the specified file."""
        yaml_file = tmp_path / "custom.yaml"
        yaml_file.write_text("agents:\n  builder:\n    max_tool_iterations: 20\n")
        result = load_agent_configs(path=yaml_file)
        assert result["builder"].max_tool_iterations == 20

    def test_explicit_path_relative_to_base_dir(self, tmp_path):
        """Relative path is resolved against base_dir."""
        sub = tmp_path / "config"
        sub.mkdir()
        yaml_file = sub / "agents.yaml"
        yaml_file.write_text("agents:\n  tester:\n    timeout_seconds: 999\n")
        result = load_agent_configs(path=Path("config/agents.yaml"), base_dir=tmp_path)
        assert result["tester"].timeout_seconds == 999


class TestYAMLMerging:
    """Merges YAML overrides with defaults."""

    def test_merges_with_defaults(self, tmp_path):
        """YAML overrides merge onto DEFAULT_AGENT_CONFIGS."""
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(
            "agents:\n  builder:\n    enable_shell: false\n    max_tool_iterations: 5\n"
        )
        result = load_agent_configs(path=yaml_file)
        # Overridden fields
        assert result["builder"].enable_shell is False
        assert result["builder"].max_tool_iterations == 5
        # Non-overridden roles still present
        assert "tester" in result
        assert "reviewer" in result

    def test_partial_override_only_changes_specified(self, tmp_path):
        """Only specified fields change; others keep default values."""
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text("agents:\n  reviewer:\n    timeout_seconds: 500\n")
        result = load_agent_configs(path=yaml_file)
        reviewer = result["reviewer"]
        default_reviewer = DEFAULT_AGENT_CONFIGS["reviewer"]
        # Overridden
        assert reviewer.timeout_seconds == 500
        # Unchanged
        assert reviewer.enable_shell == default_reviewer.enable_shell
        assert reviewer.denied_tools == default_reviewer.denied_tools
        assert reviewer.max_tool_iterations == default_reviewer.max_tool_iterations

    def test_new_role_created_with_defaults(self, tmp_path):
        """A role in YAML but not in defaults gets created with AgentConfig defaults."""
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(
            "agents:\n  debugger:\n    enable_shell: true\n    timeout_seconds: 60\n"
        )
        result = load_agent_configs(path=yaml_file)
        assert "debugger" in result
        cfg = result["debugger"]
        assert cfg.role == "debugger"
        assert cfg.enable_shell is True
        assert cfg.timeout_seconds == 60
        # Defaults for unset fields
        assert cfg.max_tool_iterations == 8
        assert cfg.max_output_chars == 50000

    def test_allowed_tools_override(self, tmp_path):
        """allowed_tools in YAML replaces the default list."""
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(
            "agents:\n  tester:\n    allowed_tools:\n      - read_file\n      - run_command\n"
        )
        result = load_agent_configs(path=yaml_file)
        assert result["tester"].allowed_tools == ["read_file", "run_command"]

    def test_model_override(self, tmp_path):
        """model field overrides from YAML."""
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text("agents:\n  builder:\n    model: gpt-4o\n")
        result = load_agent_configs(path=yaml_file)
        assert result["builder"].model == "gpt-4o"

    def test_workspace_override(self, tmp_path):
        """workspace field creates a Path from string."""
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text("agents:\n  builder:\n    workspace: /tmp/sandbox\n")
        result = load_agent_configs(path=yaml_file)
        assert result["builder"].workspace == Path("/tmp/sandbox")

    def test_top_level_agents_section_optional(self, tmp_path):
        """If 'agents' key missing, treat root dict as agents section."""
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text("builder:\n  max_tool_iterations: 15\n")
        result = load_agent_configs(path=yaml_file)
        assert result["builder"].max_tool_iterations == 15


class TestInvalidYAML:
    """Handles missing, malformed, and invalid YAML gracefully."""

    def test_invalid_yaml_syntax_returns_defaults(self, tmp_path):
        """Malformed YAML returns defaults without crashing."""
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(": [invalid yaml {{{\n")
        result = load_agent_configs(path=yaml_file)
        assert set(result.keys()) == set(DEFAULT_AGENT_CONFIGS.keys())

    def test_yaml_with_list_root_returns_defaults(self, tmp_path):
        """YAML root is a list instead of dict -> defaults."""
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text("- builder\n- tester\n")
        result = load_agent_configs(path=yaml_file)
        assert set(result.keys()) == set(DEFAULT_AGENT_CONFIGS.keys())

    def test_agents_section_is_list_returns_defaults(self, tmp_path):
        """agents section is a list instead of dict -> defaults."""
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text("agents:\n  - builder\n  - tester\n")
        result = load_agent_configs(path=yaml_file)
        assert set(result.keys()) == set(DEFAULT_AGENT_CONFIGS.keys())

    def test_role_with_non_dict_overrides_skipped(self, tmp_path):
        """A role whose value is not a dict is silently skipped."""
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text(
            "agents:\n  builder: just_a_string\n  tester:\n    timeout_seconds: 42\n"
        )
        result = load_agent_configs(path=yaml_file)
        # builder should keep default (override was invalid)
        assert (
            result["builder"].max_tool_iterations
            == DEFAULT_AGENT_CONFIGS["builder"].max_tool_iterations
        )
        # tester gets the valid override
        assert result["tester"].timeout_seconds == 42

    def test_pyyaml_not_installed_returns_defaults(self, tmp_path):
        """When PyYAML is not installed, returns defaults."""
        yaml_file = tmp_path / "agents.yaml"
        yaml_file.write_text("agents:\n  builder:\n    model: foo\n")

        with patch.dict("sys.modules", {"yaml": None}):
            # Force re-import to hit the ImportError path
            import importlib

            from animus_forge.agents import config_loader

            importlib.reload(config_loader)
            result = config_loader.load_agent_configs(path=yaml_file)

        assert set(result.keys()) == set(DEFAULT_AGENT_CONFIGS.keys())


class TestFindConfig:
    """Tests for _find_config() search logic."""

    def test_explicit_path_exists(self, tmp_path):
        """Returns the path when it exists."""
        f = tmp_path / "agents.yaml"
        f.write_text("")
        assert _find_config(f, None) == f

    def test_explicit_path_missing(self, tmp_path):
        """Returns None when explicit path doesn't exist."""
        assert _find_config(tmp_path / "nope.yaml", None) is None

    def test_searches_default_paths(self, tmp_path):
        """Finds agents.yaml in default search locations."""
        (tmp_path / "agents.yaml").write_text("")
        result = _find_config(None, tmp_path)
        assert result == tmp_path / "agents.yaml"

    def test_searches_config_subdir(self, tmp_path):
        """Finds config/agents.yaml."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "agents.yaml").write_text("")
        result = _find_config(None, tmp_path)
        assert result == config_dir / "agents.yaml"

    def test_searches_gorgon_subdir(self, tmp_path):
        """Finds .gorgon/agents.yaml."""
        gorgon_dir = tmp_path / ".gorgon"
        gorgon_dir.mkdir()
        (gorgon_dir / "agents.yaml").write_text("")
        result = _find_config(None, tmp_path)
        assert result == gorgon_dir / "agents.yaml"

    def test_returns_none_when_nothing_found(self, tmp_path):
        """Returns None when no default paths match."""
        assert _find_config(None, tmp_path) is None

    def test_explicit_relative_resolved_against_base_dir(self, tmp_path):
        """Relative explicit path is resolved against base_dir."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "my.yaml").write_text("")
        result = _find_config(Path("sub/my.yaml"), tmp_path)
        assert result == sub / "my.yaml"
