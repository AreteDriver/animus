"""Tests for configuration."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import yaml

from animus_validator.config import (
    AlertConfig,
    CompoundConfig,
    DiscoveryConfig,
    MonitorConfig,
    ValidatorConfig,
)


class TestCompoundConfig:
    def test_defaults(self):
        c = CompoundConfig()
        assert c.enabled is True
        assert c.min_reward_threshold == 0.01
        assert c.gas_buffer_multiplier == 2.0
        assert c.check_interval_hours == 6

    def test_env_override(self):
        with patch.dict(os.environ, {"VALIDATOR_COMPOUND_THRESHOLD": "0.05"}):
            c = CompoundConfig()
            assert c.min_reward_threshold == 0.05

    def test_env_not_set(self):
        with patch.dict(os.environ, {}, clear=True):
            c = CompoundConfig()
            assert c.min_reward_threshold == 0.01


class TestMonitorConfig:
    def test_defaults(self):
        c = MonitorConfig()
        assert c.check_interval_minutes == 15
        assert c.uptime_alert_threshold == 95.0
        assert c.slashing_alert is True
        assert c.downtime_alert_minutes == 30


class TestAlertConfig:
    def test_defaults(self):
        c = AlertConfig()
        assert c.discord_webhook_url == ""
        assert c.alert_on_slashing is True
        assert c.daily_summary is True

    def test_env_override(self):
        with patch.dict(os.environ, {"VALIDATOR_DISCORD_WEBHOOK": "https://discord.com/webhook"}):
            c = AlertConfig()
            assert c.discord_webhook_url == "https://discord.com/webhook"


class TestDiscoveryConfig:
    def test_defaults(self):
        c = DiscoveryConfig()
        assert c.enabled is True
        assert c.scan_interval_hours == 24
        assert c.min_apy_threshold == 3.0
        assert c.max_risk_score == 0.5


class TestValidatorConfig:
    def test_defaults(self):
        config = ValidatorConfig()
        assert config.data_dir == Path.home() / ".animus" / "validator"
        assert isinstance(config.compound, CompoundConfig)
        assert isinstance(config.monitor, MonitorConfig)
        assert isinstance(config.alerts, AlertConfig)
        assert isinstance(config.discovery, DiscoveryConfig)

    def test_env_data_dir(self):
        with patch.dict(os.environ, {"VALIDATOR_DATA_DIR": "/tmp/test-validator"}):
            config = ValidatorConfig()
            assert config.data_dir == Path("/tmp/test-validator")

    def test_from_yaml(self, tmp_path):
        yaml_data = {
            "data_dir": str(tmp_path / "custom"),
            "compound": {"enabled": False, "min_reward_threshold": 0.1},
            "monitor": {"check_interval_minutes": 30},
            "alerts": {"daily_summary": False},
            "discovery": {"min_apy_threshold": 5.0},
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(yaml_data, f)

        config = ValidatorConfig.from_yaml(config_file)
        assert config.data_dir == tmp_path / "custom"
        assert config.compound.enabled is False
        assert config.compound.min_reward_threshold == 0.1
        assert config.monitor.check_interval_minutes == 30
        assert config.alerts.daily_summary is False
        assert config.discovery.min_apy_threshold == 5.0

    def test_from_yaml_empty(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            f.write("")

        config = ValidatorConfig.from_yaml(config_file)
        assert isinstance(config.compound, CompoundConfig)

    def test_from_yaml_partial(self, tmp_path):
        yaml_data = {"compound": {"enabled": False}}
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(yaml_data, f)

        config = ValidatorConfig.from_yaml(config_file)
        assert config.compound.enabled is False
        assert isinstance(config.monitor, MonitorConfig)

    def test_ensure_dirs(self, tmp_path):
        config = ValidatorConfig(data_dir=tmp_path / "validator")
        config.ensure_dirs()
        assert (tmp_path / "validator").is_dir()
        assert (tmp_path / "validator" / "positions").is_dir()
        assert (tmp_path / "validator" / "rewards").is_dir()
        assert (tmp_path / "validator" / "compounds").is_dir()

    def test_file_paths(self):
        config = ValidatorConfig()
        assert config.positions_file.name == "stakes.jsonl"
        assert config.rewards_file.name == "history.jsonl"
        assert config.compounds_file.name == "actions.jsonl"
        assert config.nodes_file.name == "nodes.jsonl"
        assert config.config_file.name == "config.yaml"

    def test_from_dict_no_data_dir(self):
        config = ValidatorConfig._from_dict({})
        assert config.data_dir == Path.home() / ".animus" / "validator"
