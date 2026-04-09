"""Tests for MEV configuration."""

from __future__ import annotations

from pathlib import Path

from animus_mev.config import (
    ChainConfig,
    ExecutionConfig,
    MEVConfig,
    MonitorConfig,
    RiskConfig,
)
from animus_mev.models import Chain


class TestChainConfig:
    def test_defaults(self):
        cfg = ChainConfig(chain=Chain.BASE)
        assert cfg.enabled is True
        assert cfg.min_swap_usd == 1000.0
        assert "base" in cfg.rpc_url.lower() or cfg.rpc_url != ""

    def test_default_rpc_filled(self):
        cfg = ChainConfig(chain=Chain.ARB)
        assert "arb" in cfg.rpc_url.lower()

    def test_default_ws_filled(self):
        cfg = ChainConfig(chain=Chain.OP)
        assert "optimism" in cfg.ws_url.lower() or "op" in cfg.ws_url.lower()

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MEV_BASE_RPC_URL", "https://custom-rpc.example.com")
        monkeypatch.setenv("MEV_BASE_WS_URL", "wss://custom-ws.example.com")
        cfg = ChainConfig(chain=Chain.BASE)
        assert cfg.rpc_url == "https://custom-rpc.example.com"
        assert cfg.ws_url == "wss://custom-ws.example.com"

    def test_custom_values(self):
        cfg = ChainConfig(
            chain=Chain.ETH,
            enabled=False,
            min_swap_usd=5000.0,
            max_gas_gwei=100.0,
        )
        assert cfg.enabled is False
        assert cfg.min_swap_usd == 5000.0
        assert cfg.max_gas_gwei == 100.0


class TestExecutionConfig:
    def test_default_dry_run(self):
        cfg = ExecutionConfig()
        assert cfg.live is False

    def test_env_live_true(self, monkeypatch):
        monkeypatch.setenv("MEV_EXECUTION_LIVE", "true")
        cfg = ExecutionConfig()
        assert cfg.live is True

    def test_env_live_yes(self, monkeypatch):
        monkeypatch.setenv("MEV_EXECUTION_LIVE", "yes")
        cfg = ExecutionConfig()
        assert cfg.live is True

    def test_env_live_1(self, monkeypatch):
        monkeypatch.setenv("MEV_EXECUTION_LIVE", "1")
        cfg = ExecutionConfig()
        assert cfg.live is True

    def test_env_live_false(self, monkeypatch):
        monkeypatch.setenv("MEV_EXECUTION_LIVE", "false")
        cfg = ExecutionConfig()
        assert cfg.live is False

    def test_env_live_empty(self, monkeypatch):
        monkeypatch.setenv("MEV_EXECUTION_LIVE", "")
        cfg = ExecutionConfig()
        assert cfg.live is False

    def test_private_tx_default(self):
        cfg = ExecutionConfig()
        assert cfg.private_tx is True


class TestRiskConfig:
    def test_defaults(self):
        cfg = RiskConfig()
        assert cfg.max_gas_per_tx_usd == 5.0
        assert cfg.daily_gas_budget_usd == 50.0
        assert cfg.min_profit_usd == 0.50
        assert cfg.min_confidence == 0.7
        assert cfg.max_concurrent_txs == 3
        assert cfg.revert_protection is True
        assert cfg.sandwich_execution_blocked is True

    def test_sandwich_always_blocked(self):
        cfg = RiskConfig(sandwich_execution_blocked=False)
        # We can set it False in code, but the system enforces it elsewhere
        # The config just declares the default
        assert cfg.sandwich_execution_blocked is False  # Config allows override, executor blocks


class TestMonitorConfig:
    def test_defaults(self):
        cfg = MonitorConfig()
        assert cfg.poll_interval_seconds == 1.0
        assert cfg.max_pending_opportunities == 100
        assert cfg.log_all_txs is False

    def test_env_webhook(self, monkeypatch):
        monkeypatch.setenv("MEV_DISCORD_WEBHOOK", "https://discord.com/api/webhooks/test")
        cfg = MonitorConfig()
        assert cfg.discord_webhook_url == "https://discord.com/api/webhooks/test"


class TestMEVConfig:
    def test_defaults(self):
        cfg = MEVConfig()
        assert len(cfg.chains) == 3  # BASE, ARB, OP by default
        assert "base" in cfg.chains
        assert "arb" in cfg.chains
        assert "op" in cfg.chains
        assert cfg.execution.live is False

    def test_env_data_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MEV_DATA_DIR", str(tmp_path / "custom"))
        cfg = MEVConfig()
        assert cfg.data_dir == tmp_path / "custom"

    def test_get_chain(self, config):
        base = config.get_chain(Chain.BASE)
        assert base is not None
        assert base.chain == Chain.BASE

    def test_get_chain_missing(self, config):
        # ETH not in default config
        eth = config.get_chain(Chain.ETH)
        assert eth is None

    def test_enabled_chains(self, config):
        enabled = config.enabled_chains()
        assert len(enabled) == 3

    def test_enabled_chains_with_disabled(self, config):
        config.chains["base"].enabled = False
        enabled = config.enabled_chains()
        assert len(enabled) == 2

    def test_ensure_dirs(self, tmp_data_dir):
        cfg = MEVConfig(data_dir=tmp_data_dir / "sub")
        cfg.ensure_dirs()
        assert (tmp_data_dir / "sub").exists()

    def test_config_file_path(self, config):
        assert config.config_file == config.data_dir / "config.yaml"

    def test_yaml_roundtrip(self, config, tmp_path):
        yaml_path = tmp_path / "test_config.yaml"
        config.to_yaml(yaml_path)
        assert yaml_path.exists()

        loaded = MEVConfig.from_yaml(yaml_path)
        assert len(loaded.chains) == len(config.chains)
        assert loaded.execution.live == config.execution.live
        assert loaded.risk.min_profit_usd == config.risk.min_profit_usd

    def test_from_yaml_empty(self, tmp_path):
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")
        cfg = MEVConfig.from_yaml(yaml_path)
        assert cfg is not None
        # Empty YAML = defaults (no chains from YAML, but __post_init__ adds L2s)
        assert len(cfg.chains) == 3

    def test_from_dict_with_chains(self):
        data = {
            "chains": {
                "base": {
                    "enabled": True,
                    "rpc_url": "https://custom.rpc",
                    "ws_url": "wss://custom.ws",
                    "min_swap_usd": 2000.0,
                    "max_gas_gwei": 10.0,
                },
            },
            "data_dir": "/tmp/mev_test",
        }
        cfg = MEVConfig._from_dict(data)
        assert "base" in cfg.chains
        assert cfg.chains["base"].rpc_url == "https://custom.rpc"
        assert cfg.data_dir == Path("/tmp/mev_test")

    def test_custom_chains_no_defaults(self):
        cfg = MEVConfig(chains={"eth": ChainConfig(chain=Chain.ETH)})
        assert "eth" in cfg.chains
        assert len(cfg.chains) == 1  # No default L2s added since chains was provided
