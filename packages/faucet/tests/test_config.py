"""Tests for configuration."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import yaml

from animus_faucet.config import (
    AlertConfig,
    ConsolidationConfig,
    FaucetConfig,
    ProxyConfig,
    VisionConfig,
)


class TestProxyConfig:
    def test_defaults(self):
        p = ProxyConfig()
        assert p.enabled is False
        assert p.provider == "brightdata"
        assert p.proxy_url is None

    def test_proxy_url_with_auth(self):
        p = ProxyConfig(
            enabled=True, endpoint="proxy.example.com:8080", username="user", password="pass"
        )
        assert p.proxy_url == "http://user:pass@proxy.example.com:8080"

    def test_proxy_url_without_auth(self):
        p = ProxyConfig(enabled=True, endpoint="proxy.example.com:8080")
        assert p.proxy_url == "http://proxy.example.com:8080"

    def test_proxy_url_disabled(self):
        p = ProxyConfig(enabled=False, endpoint="proxy.example.com:8080")
        assert p.proxy_url is None

    def test_proxy_url_no_endpoint(self):
        p = ProxyConfig(enabled=True, endpoint="")
        assert p.proxy_url is None

    def test_env_override(self):
        with patch.dict(
            os.environ,
            {
                "FAUCET_PROXY_ENABLED": "true",
                "FAUCET_PROXY_ENDPOINT": "env-proxy:9090",
                "FAUCET_PROXY_USERNAME": "envuser",
                "FAUCET_PROXY_PASSWORD": "envpass",
            },
        ):
            p = ProxyConfig()
            assert p.enabled is True
            assert p.endpoint == "env-proxy:9090"
            assert p.username == "envuser"
            assert p.password == "envpass"


class TestVisionConfig:
    def test_defaults(self):
        v = VisionConfig()
        assert v.enabled is False
        assert v.model == "llava"
        assert v.audio_fallback is True

    def test_env_override(self):
        with patch.dict(
            os.environ,
            {
                "FAUCET_VISION_ENABLED": "1",
                "FAUCET_OLLAMA_URL": "http://gpu:11434",
                "FAUCET_VISION_MODEL": "bakllava",
            },
        ):
            v = VisionConfig()
            assert v.enabled is True
            assert v.ollama_url == "http://gpu:11434"
            assert v.model == "bakllava"


class TestConsolidationConfig:
    def test_defaults(self):
        c = ConsolidationConfig()
        assert c.enabled is True
        assert c.min_balance_usd == 1.0
        assert c.gas_price_max_gwei == 30
        assert c.cold_wallet_addresses == {}

    def test_env_cold_wallets(self):
        wallets = {"eth": "0xabc", "btc": "bc1xyz"}
        with patch.dict(os.environ, {"FAUCET_COLD_WALLETS": json.dumps(wallets)}):
            c = ConsolidationConfig()
            assert c.cold_wallet_addresses == wallets


class TestAlertConfig:
    def test_defaults(self):
        a = AlertConfig()
        assert a.discord_webhook_url == ""
        assert a.alert_on_ban is True
        assert a.daily_summary is True

    def test_env_override(self):
        with patch.dict(os.environ, {"FAUCET_DISCORD_WEBHOOK": "https://discord.com/hook"}):
            a = AlertConfig()
            assert a.discord_webhook_url == "https://discord.com/hook"


class TestFaucetConfig:
    def test_defaults(self):
        c = FaucetConfig()
        assert c.jitter_seconds == 300
        assert c.headless is True
        assert c.auto_disable_threshold == 10
        assert c.faucets == []

    def test_env_override(self):
        with patch.dict(
            os.environ,
            {
                "FAUCET_DATA_DIR": "/tmp/faucet_test",
                "FAUCET_JITTER_SECONDS": "60",
            },
        ):
            c = FaucetConfig()
            assert c.data_dir == Path("/tmp/faucet_test")
            assert c.jitter_seconds == 60

    def test_ensure_dirs(self, tmp_data_dir):
        c = FaucetConfig(data_dir=tmp_data_dir)
        c.ensure_dirs()
        assert (tmp_data_dir / "claims").exists()
        assert (tmp_data_dir / "health").exists()

    def test_file_paths(self, tmp_data_dir):
        c = FaucetConfig(data_dir=tmp_data_dir)
        assert c.claims_file == tmp_data_dir / "claims" / "history.json"
        assert c.health_file == tmp_data_dir / "health" / "status.json"
        assert c.faucets_file == tmp_data_dir / "faucets.yaml"

    def test_from_yaml(self, tmp_path):
        yaml_data = {
            "jitter_seconds": 120,
            "headless": False,
            "auto_disable_threshold": 5,
            "data_dir": str(tmp_path / "custom"),
            "proxy": {"enabled": True, "endpoint": "proxy:8080"},
            "vision": {"enabled": True, "model": "bakllava"},
            "consolidation": {"min_balance_usd": 5.0},
            "alerts": {"alert_on_ban": False},
            "faucets": [
                {
                    "name": "test",
                    "url": "http://faucet.test",
                    "chain": "btc",
                    "driver_type": "api",
                },
            ],
        }
        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_data, f)

        c = FaucetConfig.from_yaml(yaml_file)
        assert c.jitter_seconds == 120
        assert c.headless is False
        assert c.auto_disable_threshold == 5
        assert c.data_dir == tmp_path / "custom"
        assert len(c.faucets) == 1
        assert c.faucets[0].name == "test"

    def test_from_yaml_empty(self, tmp_path):
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        c = FaucetConfig.from_yaml(yaml_file)
        assert c.faucets == []
        assert c.jitter_seconds == 300

    def test_from_yaml_minimal(self, tmp_path):
        yaml_file = tmp_path / "min.yaml"
        yaml_file.write_text("faucets: []")
        c = FaucetConfig.from_yaml(yaml_file)
        assert c.faucets == []

    def test_with_faucets(self, sample_faucet):
        c = FaucetConfig(faucets=[sample_faucet])
        assert len(c.faucets) == 1
        assert c.faucets[0].name == "test_faucet"
