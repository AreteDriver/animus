"""Tests for arbitrage configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from animus_arbitrage.config import (
    AlertConfig,
    ArbitrageConfig,
    ExecutionConfig,
    FeedConfig,
    RiskConfig,
    WalletConfig,
)
from animus_arbitrage.models import DEX, Chain, ExecutionMode


class TestExecutionConfig:
    def test_defaults(self):
        config = ExecutionConfig()
        assert config.mode == ExecutionMode.DRY_RUN
        assert config.max_trade_size_usd == 100.0
        assert config.slippage_tolerance_pct == 0.5
        assert config.is_live is False

    def test_live_mode(self):
        config = ExecutionConfig(mode=ExecutionMode.LIVE)
        assert config.is_live is True

    def test_env_override_live(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ARBITRAGE_EXECUTION_MODE", "live")
        config = ExecutionConfig()
        assert config.mode == ExecutionMode.LIVE
        assert config.is_live is True

    def test_env_override_dry_run(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ARBITRAGE_EXECUTION_MODE", "dry_run")
        config = ExecutionConfig(mode=ExecutionMode.LIVE)
        assert config.mode == ExecutionMode.DRY_RUN

    def test_env_override_unknown(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ARBITRAGE_EXECUTION_MODE", "unknown")
        config = ExecutionConfig()
        assert config.mode == ExecutionMode.DRY_RUN


class TestRiskConfig:
    def test_defaults(self):
        config = RiskConfig()
        assert config.max_position_usd == 500.0
        assert config.daily_loss_limit_usd == 50.0
        assert config.min_spread_pct == 0.3
        assert config.min_liquidity_usd == 10000.0
        assert config.min_confidence == 0.7
        assert config.cooldown_seconds == 60.0
        assert config.max_trades_per_hour == 10
        assert config.max_concurrent_trades == 2


class TestAlertConfig:
    def test_defaults(self):
        config = AlertConfig()
        assert config.enabled is True
        assert config.high_spread_threshold_pct == 1.0
        assert config.discord_webhook_url == ""

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ARBITRAGE_DISCORD_WEBHOOK", "https://discord.com/api/webhooks/test")
        config = AlertConfig()
        assert config.discord_webhook_url == "https://discord.com/api/webhooks/test"


class TestFeedConfig:
    def test_defaults(self):
        config = FeedConfig()
        assert config.dex == DEX.UNISWAP_V3
        assert config.chain == Chain.ETH
        assert config.enabled is True
        assert config.poll_interval_seconds == 30.0

    def test_custom(self):
        config = FeedConfig(dex=DEX.JUPITER, chain=Chain.SOL, poll_interval_seconds=10.0)
        assert config.dex == DEX.JUPITER
        assert config.chain == Chain.SOL


class TestWalletConfig:
    def test_defaults(self):
        config = WalletConfig()
        assert config.chain == Chain.ETH
        assert config.address == ""
        assert config.private_key_env_var == ""


class TestArbitrageConfig:
    def test_defaults(self):
        config = ArbitrageConfig()
        assert config.execution.mode == ExecutionMode.DRY_RUN
        assert config.risk.max_position_usd == 500.0
        assert config.feeds == []
        assert config.wallets == []

    def test_env_data_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("ARBITRAGE_DATA_DIR", str(tmp_path / "custom"))
        config = ArbitrageConfig()
        assert config.data_dir == tmp_path / "custom"

    def test_ensure_dirs(self, tmp_data_dir: Path):
        config = ArbitrageConfig(data_dir=tmp_data_dir / "new_dir")
        config.ensure_dirs()
        assert config.data_dir.exists()

    def test_config_file_path(self, tmp_data_dir: Path):
        config = ArbitrageConfig(data_dir=tmp_data_dir)
        assert config.config_file == tmp_data_dir / "config.yaml"

    def test_to_dict(self, config: ArbitrageConfig):
        d = config.to_dict()
        assert "execution" in d
        assert "risk" in d
        assert "alerts" in d
        assert d["execution"]["mode"] == "dry_run"
        assert "feeds_count" in d
        assert "wallets_count" in d

    def test_from_yaml(self, tmp_data_dir: Path):
        yaml_data = {
            "execution": {
                "mode": "dry_run",
                "max_trade_size_usd": 200.0,
                "slippage_tolerance_pct": 1.0,
                "confirmation_timeout_seconds": 60.0,
            },
            "risk": {
                "max_position_usd": 1000.0,
                "daily_loss_limit_usd": 100.0,
                "min_spread_pct": 0.5,
                "min_liquidity_usd": 50000.0,
                "min_confidence": 0.8,
                "cooldown_seconds": 30.0,
                "max_trades_per_hour": 20,
                "max_concurrent_trades": 3,
            },
            "feeds": [
                {
                    "dex": "uniswap_v3",
                    "chain": "eth",
                    "enabled": True,
                    "poll_interval_seconds": 15.0,
                    "rpc_url": "",
                },
                {
                    "dex": "jupiter",
                    "chain": "sol",
                    "enabled": True,
                    "poll_interval_seconds": 10.0,
                    "rpc_url": "",
                },
            ],
            "wallets": [
                {"chain": "eth", "address": "0xabc", "private_key_env_var": "ETH_KEY"},
            ],
        }
        yaml_path = tmp_data_dir / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f)

        config = ArbitrageConfig.from_yaml(yaml_path)
        assert config.execution.max_trade_size_usd == 200.0
        assert config.risk.max_position_usd == 1000.0
        assert len(config.feeds) == 2
        assert config.feeds[0].dex == DEX.UNISWAP_V3
        assert config.feeds[1].dex == DEX.JUPITER
        assert len(config.wallets) == 1
        assert config.wallets[0].address == "0xabc"

    def test_from_yaml_minimal(self, tmp_data_dir: Path):
        yaml_path = tmp_data_dir / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump({}, f)

        config = ArbitrageConfig.from_yaml(yaml_path)
        assert config.execution.mode == ExecutionMode.DRY_RUN
        assert config.risk.max_position_usd == 500.0

    def test_from_yaml_with_data_dir(self, tmp_data_dir: Path):
        yaml_path = tmp_data_dir / "config.yaml"
        custom_dir = str(tmp_data_dir / "custom")
        with open(yaml_path, "w") as f:
            yaml.dump({"data_dir": custom_dir}, f)

        config = ArbitrageConfig.from_yaml(yaml_path)
        assert config.data_dir == Path(custom_dir)

    def test_from_yaml_empty_file(self, tmp_data_dir: Path):
        yaml_path = tmp_data_dir / "config.yaml"
        yaml_path.write_text("")
        config = ArbitrageConfig.from_yaml(yaml_path)
        assert config.execution.mode == ExecutionMode.DRY_RUN
