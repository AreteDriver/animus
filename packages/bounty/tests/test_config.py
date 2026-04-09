"""Tests for animus_bounty.config."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import yaml

from animus_bounty.config import (
    AlertConfig,
    BountyConfig,
    FilterConfig,
    LLMConfig,
    PlatformCredentials,
    SkillProfile,
)


class TestLLMConfig:
    def test_defaults(self):
        c = LLMConfig()
        assert c.provider == "ollama"
        assert c.model == "qwen2.5:14b"
        assert c.timeout == 60.0

    def test_env_override(self):
        with patch.dict(
            os.environ, {"BOUNTY_LLM_PROVIDER": "anthropic", "BOUNTY_LLM_MODEL": "gpt4"}
        ):
            c = LLMConfig()
            assert c.provider == "anthropic"
            assert c.model == "gpt4"

    def test_env_ollama_url(self):
        with patch.dict(os.environ, {"BOUNTY_OLLAMA_URL": "http://custom:11434"}):
            c = LLMConfig()
            assert c.ollama_url == "http://custom:11434"


class TestSkillProfile:
    def test_defaults(self):
        sp = SkillProfile()
        assert "python" in sp.languages
        assert "fastapi" in sp.frameworks
        assert "ai" in sp.domains

    def test_all_skills(self):
        sp = SkillProfile(languages=["python"], frameworks=["flask"], domains=["web"])
        skills = sp.all_skills()
        assert skills == ["python", "flask", "web"]

    def test_empty(self):
        sp = SkillProfile(languages=[], frameworks=[], domains=[])
        assert sp.all_skills() == []


class TestFilterConfig:
    def test_defaults(self):
        fc = FilterConfig()
        assert fc.min_reward_usd == 5.0
        assert fc.auto_claim_low_effort is True
        assert "wontfix" in fc.exclude_labels

    def test_custom(self):
        fc = FilterConfig(min_reward_usd=100.0, platforms=["github"])
        assert fc.min_reward_usd == 100.0
        assert fc.platforms == ["github"]


class TestAlertConfig:
    def test_defaults(self):
        ac = AlertConfig()
        assert ac.discord_webhook_url == ""
        assert ac.high_value_threshold_usd == 100.0

    def test_env_override(self):
        with patch.dict(os.environ, {"BOUNTY_DISCORD_WEBHOOK": "https://discord.com/hook"}):
            ac = AlertConfig()
            assert ac.discord_webhook_url == "https://discord.com/hook"


class TestPlatformCredentials:
    def test_defaults(self):
        pc = PlatformCredentials()
        assert pc.github_token == ""
        assert pc.gitcoin_api_key == ""

    def test_env_override(self):
        with patch.dict(
            os.environ,
            {"GITHUB_TOKEN": "ghp_test", "GITCOIN_API_KEY": "gk_test", "REPLIT_TOKEN": "rp_test"},
        ):
            pc = PlatformCredentials()
            assert pc.github_token == "ghp_test"
            assert pc.gitcoin_api_key == "gk_test"
            assert pc.replit_token == "rp_test"


class TestBountyConfig:
    def test_defaults(self):
        c = BountyConfig()
        assert c.data_dir == Path.home() / ".animus" / "bounty"
        assert c.scan_interval_hours == 4

    def test_env_data_dir(self, tmp_path):
        with patch.dict(os.environ, {"BOUNTY_DATA_DIR": str(tmp_path)}):
            c = BountyConfig()
            assert c.data_dir == tmp_path

    def test_ensure_dirs(self, tmp_data_dir):
        c = BountyConfig(data_dir=tmp_data_dir / "new_dir")
        c.ensure_dirs()
        assert c.data_dir.exists()

    def test_config_file_path(self, sample_config):
        assert sample_config.config_file == sample_config.data_dir / "config.yaml"

    def test_from_yaml(self, tmp_data_dir):
        config_data = {
            "scan_interval_hours": 8,
            "skills": {
                "languages": ["rust"],
                "frameworks": ["bevy"],
                "domains": ["gamedev"],
            },
            "filters": {
                "min_reward_usd": 50.0,
                "max_difficulty": "medium",
                "auto_claim_low_effort": False,
                "auto_claim_max_reward_usd": 100.0,
                "platforms": ["github"],
                "exclude_labels": [],
            },
        }
        config_file = tmp_data_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        c = BountyConfig.from_yaml(config_file)
        assert c.scan_interval_hours == 8
        assert c.skills.languages == ["rust"]
        assert c.filters.min_reward_usd == 50.0
        assert not c.filters.auto_claim_low_effort

    def test_from_yaml_empty(self, tmp_data_dir):
        config_file = tmp_data_dir / "config.yaml"
        config_file.write_text("")
        c = BountyConfig.from_yaml(config_file)
        assert c.scan_interval_hours == 4  # default

    def test_from_yaml_with_data_dir(self, tmp_data_dir):
        config_data = {"data_dir": str(tmp_data_dir / "custom")}
        config_file = tmp_data_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        c = BountyConfig.from_yaml(config_file)
        assert c.data_dir == tmp_data_dir / "custom"

    def test_to_dict(self, sample_config):
        d = sample_config.to_dict()
        assert "skills" in d
        assert "filters" in d
        assert "scan_interval_hours" in d
        assert d["filters"]["min_reward_usd"] == 5.0

    def test_from_dict_partial(self):
        c = BountyConfig._from_dict({"llm": {"provider": "anthropic", "model": "claude"}})
        assert c.llm.provider == "anthropic"
        assert c.skills.languages  # defaults preserved

    def test_from_dict_with_alerts(self):
        c = BountyConfig._from_dict({"alerts": {"high_value_threshold_usd": 500.0}})
        assert c.alerts.high_value_threshold_usd == 500.0

    def test_from_dict_with_credentials(self):
        c = BountyConfig._from_dict({"credentials": {"github_token": "test"}})
        assert c.credentials.github_token == "test"
