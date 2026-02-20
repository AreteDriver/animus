"""Tests for YAML config support, env var placeholders, get_config alias, and new fields."""

import os
from pathlib import Path
from unittest.mock import patch

from animus_forge.config.settings import (
    _ENV_VAR_PLACEHOLDER_RE,
    _YAML_SEARCH_PATHS,
    Settings,
    _find_yaml_config,
    get_config,
    get_settings,
)

# =============================================================================
# Helpers
# =============================================================================

_ENV_KEYS_TO_CLEAR = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "NOTION_TOKEN",
    "GITHUB_TOKEN",
    "GMAIL_CREDENTIALS_PATH",
    "SECRET_KEY",
    "DATABASE_URL",
    "PRODUCTION",
    "DEBUG",
    "ALLOW_DEMO_AUTH",
    "REQUIRE_SECURE_CONFIG",
    "API_CREDENTIALS",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "REDIS_URL",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_ALLOWED_USERS",
    "TELEGRAM_ADMIN_USERS",
    "DISCORD_BOT_TOKEN",
    "DISCORD_ALLOWED_USERS",
    "DISCORD_ADMIN_USERS",
    "DISCORD_ALLOWED_GUILDS",
    "SETTINGS_ENCRYPTION_KEY",
    "JWT_SECRET",
]


def _make_settings(tmp_path, yaml_content=None, keep_env=(), **overrides):
    """Create Settings with optional YAML file and temp dirs.

    If yaml_content is provided, writes it to tmp_path/gorgon.yaml and patches
    the search paths so Settings discovers it.

    Args:
        keep_env: Env var names to NOT clear (e.g. for testing env overrides).
    """
    defaults = dict(
        _env_file=None,
        logs_dir=tmp_path / "logs",
        prompts_dir=tmp_path / "prompts",
        workflows_dir=tmp_path / "workflows",
        schedules_dir=tmp_path / "schedules",
        webhooks_dir=tmp_path / "webhooks",
        jobs_dir=tmp_path / "jobs",
        plugins_dir=tmp_path / "plugins",
    )
    defaults.update(overrides)

    yaml_path = None
    if yaml_content is not None:
        yaml_path = tmp_path / "gorgon.yaml"
        yaml_path.write_text(yaml_content)

    saved = {}
    for key in _ENV_KEYS_TO_CLEAR:
        if key in keep_env:
            continue
        if key in os.environ:
            saved[key] = os.environ.pop(key)
    try:
        if yaml_path is not None:
            with patch(
                "animus_forge.config.settings._find_yaml_config",
                return_value=yaml_path,
            ):
                return Settings(**defaults)
        else:
            with patch(
                "animus_forge.config.settings._find_yaml_config",
                return_value=None,
            ):
                return Settings(**defaults)
    finally:
        os.environ.update(saved)


# =============================================================================
# TestFindYamlConfig
# =============================================================================


class TestFindYamlConfig:
    """Tests for the _find_yaml_config helper."""

    def test_returns_none_when_no_yaml_exists(self, tmp_path):
        fake_paths = [tmp_path / "a.yaml", tmp_path / "b.yaml"]
        with patch("animus_forge.config.settings._YAML_SEARCH_PATHS", fake_paths):
            assert _find_yaml_config() is None

    def test_returns_first_matching_path(self, tmp_path):
        first = tmp_path / "gorgon.yaml"
        second = tmp_path / "config" / "gorgon.yaml"
        second.parent.mkdir()
        # Create both
        first.write_text("debug: true\n")
        second.write_text("debug: false\n")
        with patch("animus_forge.config.settings._YAML_SEARCH_PATHS", [first, second]):
            assert _find_yaml_config() == first

    def test_skips_missing_returns_second(self, tmp_path):
        first = tmp_path / "missing.yaml"
        second = tmp_path / "gorgon.yaml"
        second.write_text("debug: true\n")
        with patch("animus_forge.config.settings._YAML_SEARCH_PATHS", [first, second]):
            assert _find_yaml_config() == second

    def test_search_paths_has_expected_entries(self):
        assert len(_YAML_SEARCH_PATHS) == 3
        assert _YAML_SEARCH_PATHS[0] == Path("gorgon.yaml")
        assert _YAML_SEARCH_PATHS[1] == Path("config/gorgon.yaml")
        assert "gorgon" in str(_YAML_SEARCH_PATHS[2])

    def test_ignores_directories(self, tmp_path):
        dir_path = tmp_path / "gorgon.yaml"
        dir_path.mkdir()
        with patch("animus_forge.config.settings._YAML_SEARCH_PATHS", [dir_path]):
            assert _find_yaml_config() is None


# =============================================================================
# TestYamlSettingsSource
# =============================================================================


class TestYamlSettingsSource:
    """Tests for YAML loading as a settings source."""

    def test_yaml_overrides_defaults(self, tmp_path):
        s = _make_settings(tmp_path, yaml_content="log_level: DEBUG\n")
        assert s.log_level == "DEBUG"

    def test_yaml_sets_app_name(self, tmp_path):
        s = _make_settings(tmp_path, yaml_content="app_name: MyApp\n")
        assert s.app_name == "MyApp"

    def test_env_overrides_yaml(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        s = _make_settings(tmp_path, yaml_content="log_level: DEBUG\n", keep_env=("LOG_LEVEL",))
        assert s.log_level == "WARNING"

    def test_init_overrides_yaml(self, tmp_path):
        s = _make_settings(tmp_path, yaml_content="log_level: DEBUG\n", log_level="ERROR")
        assert s.log_level == "ERROR"

    def test_unknown_yaml_fields_ignored(self, tmp_path):
        s = _make_settings(
            tmp_path,
            yaml_content="totally_unknown_field: whatever\nlog_level: DEBUG\n",
        )
        assert s.log_level == "DEBUG"
        assert not hasattr(s, "totally_unknown_field")

    def test_empty_yaml_uses_defaults(self, tmp_path):
        s = _make_settings(tmp_path, yaml_content="")
        assert s.app_name == "Gorgon"

    def test_no_yaml_uses_defaults(self, tmp_path):
        s = _make_settings(tmp_path)
        assert s.app_name == "Gorgon"

    def test_yaml_numeric_value(self, tmp_path):
        s = _make_settings(tmp_path, yaml_content="ratelimit_openai_rpm: 120\n")
        assert s.ratelimit_openai_rpm == 120

    def test_yaml_boolean_value(self, tmp_path):
        s = _make_settings(tmp_path, yaml_content="debug: true\n")
        assert s.debug is True

    def test_yaml_multiple_fields(self, tmp_path):
        yaml = "log_level: DEBUG\napp_name: TestApp\ndebug: true\n"
        s = _make_settings(tmp_path, yaml_content=yaml)
        assert s.log_level == "DEBUG"
        assert s.app_name == "TestApp"
        assert s.debug is True


# =============================================================================
# TestEnvVarPlaceholderResolution
# =============================================================================


class TestEnvVarPlaceholderResolution:
    """Tests for ${VAR} placeholder stripping."""

    def test_placeholder_regex_matches(self):
        assert _ENV_VAR_PLACEHOLDER_RE.match("${FOO_BAR}")
        assert _ENV_VAR_PLACEHOLDER_RE.match("${A}")
        assert _ENV_VAR_PLACEHOLDER_RE.match("${MY_VAR_123}")

    def test_placeholder_regex_rejects(self):
        assert not _ENV_VAR_PLACEHOLDER_RE.match("plain-value")
        assert not _ENV_VAR_PLACEHOLDER_RE.match("${}")
        assert not _ENV_VAR_PLACEHOLDER_RE.match("${123}")
        assert not _ENV_VAR_PLACEHOLDER_RE.match("prefix${VAR}")
        assert not _ENV_VAR_PLACEHOLDER_RE.match("${VAR}suffix")

    def test_placeholder_stripped_to_none(self, tmp_path):
        yaml = "redis_url: ${REDIS_URL}\n"
        s = _make_settings(tmp_path, yaml_content=yaml)
        assert s.redis_url is None

    def test_env_wins_over_placeholder(self, tmp_path, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "redis://real-host:6379/0")
        yaml = "redis_url: ${REDIS_URL}\n"
        s = _make_settings(tmp_path, yaml_content=yaml, keep_env=("REDIS_URL",))
        assert s.redis_url == "redis://real-host:6379/0"

    def test_real_value_not_stripped(self, tmp_path):
        yaml = "redis_url: redis://localhost:6379/0\n"
        s = _make_settings(tmp_path, yaml_content=yaml)
        assert s.redis_url == "redis://localhost:6379/0"


# =============================================================================
# TestGetConfigAlias
# =============================================================================


class TestGetConfigAlias:
    """Tests for the get_config alias."""

    def test_get_config_is_get_settings(self):
        assert get_config is get_settings

    def test_get_config_returns_settings(self):
        get_config.cache_clear()
        s = get_config()
        assert isinstance(s, Settings)

    def test_get_config_same_instance_as_get_settings(self):
        get_config.cache_clear()
        s1 = get_config()
        s2 = get_settings()
        assert s1 is s2


# =============================================================================
# TestNewSettingsFields — defaults
# =============================================================================


class TestNewSettingsFieldDefaults:
    """All 16 new fields have correct None defaults."""

    def test_azure_openai_api_key(self, tmp_path):
        s = _make_settings(tmp_path)
        assert s.azure_openai_api_key is None

    def test_azure_openai_endpoint(self, tmp_path):
        s = _make_settings(tmp_path)
        assert s.azure_openai_endpoint is None

    def test_azure_openai_deployment(self, tmp_path):
        s = _make_settings(tmp_path)
        assert s.azure_openai_deployment is None

    def test_google_application_credentials(self, tmp_path):
        s = _make_settings(tmp_path)
        assert s.google_application_credentials is None

    def test_google_cloud_project(self, tmp_path):
        s = _make_settings(tmp_path)
        assert s.google_cloud_project is None

    def test_google_cloud_location(self, tmp_path):
        s = _make_settings(tmp_path)
        assert s.google_cloud_location is None

    def test_redis_url(self, tmp_path):
        s = _make_settings(tmp_path)
        assert s.redis_url is None

    def test_telegram_bot_token(self, tmp_path):
        s = _make_settings(tmp_path)
        assert s.telegram_bot_token is None

    def test_telegram_allowed_users(self, tmp_path):
        s = _make_settings(tmp_path)
        assert s.telegram_allowed_users is None

    def test_telegram_admin_users(self, tmp_path):
        s = _make_settings(tmp_path)
        assert s.telegram_admin_users is None

    def test_discord_bot_token(self, tmp_path):
        s = _make_settings(tmp_path)
        assert s.discord_bot_token is None

    def test_discord_allowed_users(self, tmp_path):
        s = _make_settings(tmp_path)
        assert s.discord_allowed_users is None

    def test_discord_admin_users(self, tmp_path):
        s = _make_settings(tmp_path)
        assert s.discord_admin_users is None

    def test_discord_allowed_guilds(self, tmp_path):
        s = _make_settings(tmp_path)
        assert s.discord_allowed_guilds is None

    def test_settings_encryption_key(self, tmp_path):
        s = _make_settings(tmp_path)
        assert s.settings_encryption_key is None

    def test_jwt_secret(self, tmp_path):
        s = _make_settings(tmp_path)
        assert s.jwt_secret is None


# =============================================================================
# TestNewSettingsFields — YAML/env override
# =============================================================================


class TestNewSettingsFieldOverrides:
    """New fields can be set via YAML and env vars."""

    def test_azure_key_from_yaml(self, tmp_path):
        s = _make_settings(tmp_path, yaml_content="azure_openai_api_key: test-key\n")
        assert s.azure_openai_api_key == "test-key"

    def test_redis_url_from_yaml(self, tmp_path):
        s = _make_settings(tmp_path, yaml_content="redis_url: redis://myhost:6379/0\n")
        assert s.redis_url == "redis://myhost:6379/0"

    def test_telegram_token_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok-123")
        s = _make_settings(tmp_path, keep_env=("TELEGRAM_BOT_TOKEN",))
        assert s.telegram_bot_token == "tok-123"

    def test_discord_guilds_from_yaml(self, tmp_path):
        s = _make_settings(tmp_path, yaml_content="discord_allowed_guilds: '111,222'\n")
        assert s.discord_allowed_guilds == "111,222"

    def test_jwt_secret_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JWT_SECRET", "my-jwt-secret")
        s = _make_settings(tmp_path, keep_env=("JWT_SECRET",))
        assert s.jwt_secret == "my-jwt-secret"

    def test_google_cloud_project_from_yaml(self, tmp_path):
        s = _make_settings(tmp_path, yaml_content="google_cloud_project: my-project\n")
        assert s.google_cloud_project == "my-project"
