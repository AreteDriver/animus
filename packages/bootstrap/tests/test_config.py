"""Tests for the Animus Bootstrap configuration system."""

from __future__ import annotations

import os
import platform
from pathlib import Path

import pytest
import tomli_w

from animus_bootstrap.config import AnimusConfig, ConfigManager
from animus_bootstrap.config.defaults import DEFAULT_CONFIG
from animus_bootstrap.config.manager import _deep_merge
from animus_bootstrap.config.schema import (
    AnimusSection,
    ApiSection,
    ForgeSection,
    IdentitySection,
    IntelligenceSection,
    MCPConfig,
    MemorySection,
    PersonaProfileConfig,
    PersonasSection,
    PersonaVoiceConfig,
    ProactiveCheckConfig,
    ProactiveSection,
    ServicesSection,
)

# ------------------------------------------------------------------
# ConfigManager — load
# ------------------------------------------------------------------


class TestLoadNoFile:
    """Loading when no config file exists returns pure defaults."""

    def test_returns_animus_config(self, tmp_path: Path) -> None:
        mgr = ConfigManager(config_dir=tmp_path)
        cfg = mgr.load()
        assert isinstance(cfg, AnimusConfig)

    def test_default_version(self, tmp_path: Path) -> None:
        cfg = ConfigManager(config_dir=tmp_path).load()
        assert cfg.animus.version == "0.1.0"

    def test_default_first_run(self, tmp_path: Path) -> None:
        cfg = ConfigManager(config_dir=tmp_path).load()
        assert cfg.animus.first_run is True

    def test_default_data_dir(self, tmp_path: Path) -> None:
        cfg = ConfigManager(config_dir=tmp_path).load()
        assert cfg.animus.data_dir == "~/.local/share/animus"

    def test_default_api_keys_empty(self, tmp_path: Path) -> None:
        cfg = ConfigManager(config_dir=tmp_path).load()
        assert cfg.api.anthropic_key == ""
        assert cfg.api.openai_key == ""

    def test_default_forge_disabled(self, tmp_path: Path) -> None:
        cfg = ConfigManager(config_dir=tmp_path).load()
        assert cfg.forge.enabled is False
        assert cfg.forge.host == "localhost"
        assert cfg.forge.port == 8000

    def test_default_memory_backend(self, tmp_path: Path) -> None:
        cfg = ConfigManager(config_dir=tmp_path).load()
        assert cfg.memory.backend == "sqlite"
        assert cfg.memory.max_context_tokens == 100_000

    def test_default_identity_empty(self, tmp_path: Path) -> None:
        cfg = ConfigManager(config_dir=tmp_path).load()
        assert cfg.identity.name == ""
        assert cfg.identity.timezone == ""
        assert cfg.identity.locale == ""

    def test_default_services(self, tmp_path: Path) -> None:
        cfg = ConfigManager(config_dir=tmp_path).load()
        assert cfg.services.autostart is True
        assert cfg.services.port == 7700
        assert cfg.services.log_level == "info"
        assert cfg.services.update_check is True


# ------------------------------------------------------------------
# ConfigManager — save / reload round-trip
# ------------------------------------------------------------------


class TestSaveAndReload:
    """Writing config and reading it back preserves all values."""

    def test_round_trip_defaults(self, tmp_path: Path) -> None:
        mgr = ConfigManager(config_dir=tmp_path)
        original = mgr.load()
        mgr.save(original)
        reloaded = mgr.load()
        assert reloaded.model_dump() == original.model_dump()

    def test_round_trip_custom_values(self, tmp_path: Path) -> None:
        mgr = ConfigManager(config_dir=tmp_path)
        cfg = AnimusConfig(
            animus=AnimusSection(version="2.0.0", first_run=False, data_dir="/tmp/animus-data"),
            api=ApiSection(anthropic_key="sk-ant-xxx", openai_key="sk-oai-yyy"),
            forge=ForgeSection(enabled=True, host="0.0.0.0", port=9000, api_key="forge-key"),
            memory=MemorySection(backend="chroma", path="/tmp/mem.db", max_context_tokens=50_000),
            identity=IdentitySection(name="Arete", timezone="US/Eastern", locale="en_US"),
            services=ServicesSection(
                autostart=False, port=8080, log_level="debug", update_check=False
            ),
        )
        mgr.save(cfg)
        reloaded = mgr.load()
        assert reloaded.animus.version == "2.0.0"
        assert reloaded.animus.first_run is False
        assert reloaded.api.anthropic_key == "sk-ant-xxx"
        assert reloaded.api.openai_key == "sk-oai-yyy"
        assert reloaded.forge.enabled is True
        assert reloaded.forge.host == "0.0.0.0"
        assert reloaded.forge.port == 9000
        assert reloaded.forge.api_key == "forge-key"
        assert reloaded.memory.backend == "chroma"
        assert reloaded.memory.max_context_tokens == 50_000
        assert reloaded.identity.name == "Arete"
        assert reloaded.identity.timezone == "US/Eastern"
        assert reloaded.identity.locale == "en_US"
        assert reloaded.services.autostart is False
        assert reloaded.services.port == 8080
        assert reloaded.services.log_level == "debug"
        assert reloaded.services.update_check is False

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "config"
        mgr = ConfigManager(config_dir=nested)
        mgr.save(AnimusConfig())
        assert mgr.get_config_path().is_file()

    def test_partial_toml_merges_with_defaults(self, tmp_path: Path) -> None:
        """A TOML file with only some sections still loads correctly."""
        config_path = tmp_path / "config.toml"
        partial = {"animus": {"first_run": False}, "forge": {"enabled": True}}
        with open(config_path, "wb") as fh:
            tomli_w.dump(partial, fh)

        mgr = ConfigManager(config_dir=tmp_path)
        cfg = mgr.load()
        # Overridden values
        assert cfg.animus.first_run is False
        assert cfg.forge.enabled is True
        # Defaults preserved
        assert cfg.animus.version == "0.1.0"
        assert cfg.forge.host == "localhost"
        assert cfg.memory.backend == "sqlite"
        assert cfg.services.port == 7700


# ------------------------------------------------------------------
# ConfigManager — chmod 600 on Linux
# ------------------------------------------------------------------


@pytest.mark.skipif(platform.system() == "Windows", reason="chmod not applicable on Windows")
class TestFilePermissions:
    """Config file is restricted to owner-only read/write on Unix."""

    def test_chmod_600_after_save(self, tmp_path: Path) -> None:
        mgr = ConfigManager(config_dir=tmp_path)
        mgr.save(AnimusConfig())
        path = mgr.get_config_path()
        mode = os.stat(path).st_mode
        assert mode & 0o777 == 0o600

    def test_chmod_600_preserved_on_overwrite(self, tmp_path: Path) -> None:
        mgr = ConfigManager(config_dir=tmp_path)
        mgr.save(AnimusConfig())
        mgr.save(AnimusConfig(animus=AnimusSection(first_run=False)))
        path = mgr.get_config_path()
        mode = os.stat(path).st_mode
        assert mode & 0o777 == 0o600


# ------------------------------------------------------------------
# ConfigManager — exists
# ------------------------------------------------------------------


class TestExists:
    def test_false_when_no_file(self, tmp_path: Path) -> None:
        mgr = ConfigManager(config_dir=tmp_path)
        assert mgr.exists() is False

    def test_true_after_save(self, tmp_path: Path) -> None:
        mgr = ConfigManager(config_dir=tmp_path)
        mgr.save(AnimusConfig())
        assert mgr.exists() is True


# ------------------------------------------------------------------
# ConfigManager — get_config_path / get_data_dir
# ------------------------------------------------------------------


class TestPaths:
    def test_get_config_path(self, tmp_path: Path) -> None:
        mgr = ConfigManager(config_dir=tmp_path)
        assert mgr.get_config_path() == tmp_path / "config.toml"

    def test_get_data_dir_creates_directory(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "custom-data"
        mgr = ConfigManager(config_dir=tmp_path)
        cfg = AnimusConfig(animus=AnimusSection(data_dir=str(data_dir)))
        mgr.save(cfg)
        result = mgr.get_data_dir()
        assert result == data_dir
        assert result.is_dir()

    def test_get_data_dir_idempotent(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        mgr = ConfigManager(config_dir=tmp_path)
        cfg = AnimusConfig(animus=AnimusSection(data_dir=str(data_dir)))
        mgr.save(cfg)
        mgr.get_data_dir()
        mgr.get_data_dir()
        assert data_dir.is_dir()


# ------------------------------------------------------------------
# Nested config sections
# ------------------------------------------------------------------


class TestNestedSections:
    """Verify each nested section model works independently."""

    def test_animus_section_defaults(self) -> None:
        s = AnimusSection()
        assert s.version == "0.1.0"
        assert s.first_run is True
        assert s.data_dir == "~/.local/share/animus"

    def test_api_section_defaults(self) -> None:
        s = ApiSection()
        assert s.anthropic_key == ""
        assert s.openai_key == ""

    def test_forge_section_defaults(self) -> None:
        s = ForgeSection()
        assert s.enabled is False
        assert s.host == "localhost"
        assert s.port == 8000
        assert s.api_key == ""

    def test_memory_section_defaults(self) -> None:
        s = MemorySection()
        assert s.backend == "sqlite"
        assert s.path == "~/.local/share/animus/memory.db"
        assert s.max_context_tokens == 100_000

    def test_identity_section_defaults(self) -> None:
        s = IdentitySection()
        assert s.name == ""
        assert s.timezone == ""
        assert s.locale == ""

    def test_services_section_defaults(self) -> None:
        s = ServicesSection()
        assert s.autostart is True
        assert s.port == 7700
        assert s.log_level == "info"
        assert s.update_check is True

    def test_config_model_dump_structure(self) -> None:
        cfg = AnimusConfig()
        data = cfg.model_dump()
        assert set(data.keys()) == {
            "animus",
            "api",
            "forge",
            "memory",
            "identity",
            "services",
            "gateway",
            "channels",
            "intelligence",
            "proactive",
            "personas",
            "ollama",
            "self_improvement",
        }
        assert isinstance(data["animus"], dict)
        assert isinstance(data["forge"], dict)

    def test_get_data_path_expands_tilde(self) -> None:
        cfg = AnimusConfig()
        path = cfg.get_data_path()
        assert "~" not in str(path)

    def test_get_memory_path_expands_tilde(self) -> None:
        cfg = AnimusConfig()
        path = cfg.get_memory_path()
        assert "~" not in str(path)


# ------------------------------------------------------------------
# Deep merge helper
# ------------------------------------------------------------------


class TestDeepMerge:
    def test_override_flat_key(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"a": 10}
        result = _deep_merge(base, override)
        assert result == {"a": 10, "b": 2}

    def test_nested_merge(self) -> None:
        base = {"x": {"y": 1, "z": 2}}
        override = {"x": {"y": 99}}
        result = _deep_merge(base, override)
        assert result == {"x": {"y": 99, "z": 2}}

    def test_new_keys_from_override(self) -> None:
        base = {"a": 1}
        override = {"b": 2}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_empty_override(self) -> None:
        base = {"a": {"b": 1}}
        result = _deep_merge(base, {})
        assert result == {"a": {"b": 1}}

    def test_empty_base(self) -> None:
        override = {"a": {"b": 1}}
        result = _deep_merge({}, override)
        assert result == {"a": {"b": 1}}


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    def test_corrupt_toml_returns_defaults(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text("this is not valid [[[ toml", encoding="utf-8")
        mgr = ConfigManager(config_dir=tmp_path)
        cfg = mgr.load()
        assert cfg.animus.version == "0.1.0"

    def test_default_config_dict_matches_schema(self) -> None:
        """DEFAULT_CONFIG values match AnimusConfig defaults."""
        from_defaults = AnimusConfig(**DEFAULT_CONFIG)
        from_empty = AnimusConfig()
        assert from_defaults.model_dump() == from_empty.model_dump()

    def test_config_manager_default_dir(self) -> None:
        """Default config dir points to ~/.config/animus."""
        mgr = ConfigManager()
        expected = Path("~/.config/animus").expanduser() / "config.toml"
        assert mgr.get_config_path() == expected


# ------------------------------------------------------------------
# Intelligence & Proactive section defaults
# ------------------------------------------------------------------


class TestIntelligenceSectionDefaults:
    """Verify IntelligenceSection and related models have correct defaults."""

    def test_intelligence_section_defaults(self) -> None:
        s = IntelligenceSection()
        assert s.enabled is True
        assert s.memory_backend == "sqlite"
        assert s.memory_db_path == "~/.local/share/animus/intelligence.db"
        assert s.tool_approval_default == "auto"
        assert s.max_tool_calls_per_turn == 5
        assert s.tool_timeout_seconds == 30
        assert isinstance(s.mcp, MCPConfig)

    def test_mcp_config_defaults(self) -> None:
        m = MCPConfig()
        assert m.config_path == "~/.config/animus/mcp.json"
        assert m.auto_discover is True

    def test_proactive_section_defaults(self) -> None:
        s = ProactiveSection()
        assert s.enabled is True
        assert s.quiet_hours_start == "22:00"
        assert s.quiet_hours_end == "07:00"
        assert s.timezone == "UTC"
        assert s.checks == {}

    def test_proactive_check_config_defaults(self) -> None:
        c = ProactiveCheckConfig()
        assert c.enabled is True
        assert c.schedule == ""
        assert c.channels == []

    def test_config_with_intelligence_section(self) -> None:
        cfg = AnimusConfig()
        assert isinstance(cfg.intelligence, IntelligenceSection)
        assert cfg.intelligence.enabled is True
        assert cfg.intelligence.mcp.auto_discover is True

    def test_config_with_proactive_section(self) -> None:
        cfg = AnimusConfig()
        assert isinstance(cfg.proactive, ProactiveSection)
        assert cfg.proactive.enabled is True
        assert cfg.proactive.timezone == "UTC"

    def test_intelligence_in_model_dump(self) -> None:
        cfg = AnimusConfig()
        data = cfg.model_dump()
        assert "intelligence" in data
        assert data["intelligence"]["enabled"] is True
        assert data["intelligence"]["mcp"]["auto_discover"] is True

    def test_proactive_in_model_dump(self) -> None:
        cfg = AnimusConfig()
        data = cfg.model_dump()
        assert "proactive" in data
        assert data["proactive"]["quiet_hours_start"] == "22:00"
        assert data["proactive"]["checks"] == {}

    def test_custom_intelligence_values(self) -> None:
        s = IntelligenceSection(
            enabled=False,
            memory_backend="chromadb",
            max_tool_calls_per_turn=10,
            tool_timeout_seconds=60,
        )
        assert s.enabled is False
        assert s.memory_backend == "chromadb"
        assert s.max_tool_calls_per_turn == 10
        assert s.tool_timeout_seconds == 60

    def test_custom_proactive_with_checks(self) -> None:
        s = ProactiveSection(
            checks={
                "morning_brief": ProactiveCheckConfig(
                    enabled=True, schedule="0 7 * * *", channels=["webchat"]
                ),
            },
        )
        assert "morning_brief" in s.checks
        assert s.checks["morning_brief"].schedule == "0 7 * * *"
        assert s.checks["morning_brief"].channels == ["webchat"]


# ------------------------------------------------------------------
# Personas section defaults
# ------------------------------------------------------------------


class TestPersonasSectionDefaults:
    """Verify PersonasSection and related models have correct defaults."""

    def test_personas_section_defaults(self) -> None:
        s = PersonasSection()
        assert s.enabled is True
        assert s.default_name == "Animus"
        assert s.default_tone == "balanced"
        assert s.default_max_response_length == "medium"
        assert s.default_emoji_policy == "minimal"
        assert s.default_system_prompt == "You are Animus, a personal AI assistant."
        assert s.profiles == {}

    def test_persona_profile_config_defaults(self) -> None:
        p = PersonaProfileConfig()
        assert p.name == ""
        assert p.description == ""
        assert p.system_prompt == ""
        assert p.tone == "balanced"
        assert p.knowledge_domains == []
        assert p.excluded_topics == []
        assert p.channel_bindings == {}

    def test_persona_voice_config_defaults(self) -> None:
        v = PersonaVoiceConfig()
        assert v.tone == "balanced"
        assert v.max_response_length == "medium"
        assert v.emoji_policy == "minimal"
        assert v.language == "en"
        assert v.custom_instructions == ""

    def test_config_with_personas_section(self) -> None:
        cfg = AnimusConfig()
        assert isinstance(cfg.personas, PersonasSection)
        assert cfg.personas.enabled is True
        assert cfg.personas.default_name == "Animus"
        assert cfg.personas.profiles == {}

    def test_personas_in_model_dump(self) -> None:
        cfg = AnimusConfig()
        data = cfg.model_dump()
        assert "personas" in data
        assert data["personas"]["enabled"] is True
        assert data["personas"]["default_name"] == "Animus"
        assert data["personas"]["profiles"] == {}

    def test_custom_personas_with_profiles(self) -> None:
        s = PersonasSection(
            default_name="CustomBot",
            profiles={
                "coder": PersonaProfileConfig(
                    name="CodeBot",
                    tone="technical",
                    knowledge_domains=["python", "rust"],
                    channel_bindings={"discord": True},
                ),
            },
        )
        assert s.default_name == "CustomBot"
        assert "coder" in s.profiles
        assert s.profiles["coder"].name == "CodeBot"
        assert s.profiles["coder"].tone == "technical"
        assert s.profiles["coder"].knowledge_domains == ["python", "rust"]
        assert s.profiles["coder"].channel_bindings == {"discord": True}
