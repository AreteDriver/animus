"""Tests for configuration."""

import os
from pathlib import Path
from unittest.mock import patch

import yaml

from animus_prospector.config import (
    AlertConfig,
    EvaluationConfig,
    HunterConfig,
    LLMConfig,
    ProspectorConfig,
)


class TestLLMConfig:
    def test_defaults(self):
        c = LLMConfig()
        assert c.provider == "ollama"
        assert c.model == "qwen2.5:14b"
        assert c.timeout == 60.0

    def test_env_override(self):
        with patch.dict(
            os.environ,
            {
                "PROSPECTOR_LLM_PROVIDER": "anthropic",
                "PROSPECTOR_OLLAMA_URL": "http://gpu:11434",
                "PROSPECTOR_LLM_MODEL": "llama3",
            },
        ):
            c = LLMConfig()
            assert c.provider == "anthropic"
            assert c.ollama_url == "http://gpu:11434"
            assert c.model == "llama3"


class TestHunterConfig:
    def test_defaults(self):
        c = HunterConfig()
        assert c.enabled is True
        assert c.scan_interval_hours == 6
        assert c.max_results_per_scan == 50
        assert c.sources == {}


class TestAlertConfig:
    def test_defaults(self):
        c = AlertConfig()
        assert c.discord_webhook_url == ""
        assert c.alert_on_high_value is True
        assert c.high_value_threshold_usd == 10.0

    def test_env_override(self):
        with patch.dict(os.environ, {"PROSPECTOR_DISCORD_WEBHOOK": "https://hook"}):
            c = AlertConfig()
            assert c.discord_webhook_url == "https://hook"


class TestEvaluationConfig:
    def test_defaults(self):
        c = EvaluationConfig()
        assert c.auto_execute_min_roi == 0.5
        assert c.auto_execute_max_risk == 0.2
        assert c.skip_max_effort_minutes == 120
        assert len(c.scam_keywords) > 5

    def test_scam_keywords(self):
        c = EvaluationConfig()
        assert "send eth to receive" in c.scam_keywords
        assert "private key" in c.scam_keywords


class TestProspectorConfig:
    def test_defaults(self):
        c = ProspectorConfig()
        assert isinstance(c.llm, LLMConfig)
        assert isinstance(c.hunters, HunterConfig)
        assert isinstance(c.evaluation, EvaluationConfig)

    def test_env_override(self):
        with patch.dict(os.environ, {"PROSPECTOR_DATA_DIR": "/tmp/test_prospector"}):
            c = ProspectorConfig()
            assert c.data_dir == Path("/tmp/test_prospector")

    def test_ensure_dirs(self, tmp_data_dir):
        c = ProspectorConfig(data_dir=tmp_data_dir)
        c.ensure_dirs()
        assert (tmp_data_dir / "opportunities").exists()
        assert (tmp_data_dir / "evaluations").exists()

    def test_file_paths(self, tmp_data_dir):
        c = ProspectorConfig(data_dir=tmp_data_dir)
        assert c.opportunities_file == tmp_data_dir / "opportunities" / "feed.jsonl"
        assert c.evaluations_file == tmp_data_dir / "evaluations" / "results.jsonl"
        assert c.config_file == tmp_data_dir / "config.yaml"

    def test_from_yaml(self, tmp_path):
        yaml_data = {
            "llm": {"provider": "anthropic", "model": "haiku"},
            "hunters": {"scan_interval_hours": 12},
            "alerts": {"daily_digest": False},
            "evaluation": {"auto_execute_min_roi": 1.0},
            "data_dir": str(tmp_path / "custom"),
        }
        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_data, f)

        c = ProspectorConfig.from_yaml(yaml_file)
        assert c.llm.provider == "anthropic"
        assert c.hunters.scan_interval_hours == 12
        assert c.data_dir == tmp_path / "custom"

    def test_from_yaml_empty(self, tmp_path):
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        c = ProspectorConfig.from_yaml(yaml_file)
        assert isinstance(c.llm, LLMConfig)

    def test_from_yaml_minimal(self, tmp_path):
        yaml_file = tmp_path / "min.yaml"
        yaml_file.write_text("llm:\n  model: test")
        c = ProspectorConfig.from_yaml(yaml_file)
        assert c.llm.model == "test"
