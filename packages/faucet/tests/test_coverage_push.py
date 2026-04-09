"""Extra tests to push coverage above 97%."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from animus_faucet.config import FaucetConfig, ProxyConfig
from animus_faucet.drivers.cli_driver import CLIDriver
from animus_faucet.models import (
    Chain,
    ClaimResult,
    ClaimStatus,
    DriverType,
    FaucetDefinition,
)
from animus_faucet.proxy.rotator import ProxyRotator
from animus_faucet.scheduler import FaucetScheduler
from animus_faucet.wallet.manager import WalletManager


class TestSchedulerJitter:
    """Cover jitter path (scheduler.py:91-93)."""

    @pytest.mark.asyncio
    async def test_claim_with_jitter(self, tmp_data_dir, sample_faucet):
        config = FaucetConfig(
            data_dir=tmp_data_dir,
            faucets=[sample_faucet],
            jitter_seconds=1,  # tiny jitter
        )
        config.ensure_dirs()
        wm = WalletManager(tmp_data_dir)
        wm.set_address(Chain.ETH, "0xtest")
        sched = FaucetScheduler(config, wm)

        mock_driver = AsyncMock()
        mock_driver.claim = AsyncMock(
            return_value=ClaimResult(
                faucet_name="test_faucet",
                chain=Chain.ETH,
                status=ClaimStatus.SUCCESS,
                amount_usd=0.05,
            )
        )
        sched._drivers["test_faucet"] = mock_driver

        # Patch sleep to avoid actual delay
        with patch("animus_faucet.scheduler.asyncio.sleep", new_callable=AsyncMock):
            result = await sched.claim_faucet(sample_faucet)

        assert result.succeeded is True

    @pytest.mark.asyncio
    async def test_claim_failure_logs(self, tmp_data_dir, sample_faucet):
        """Cover scheduler.py:111 — failure log."""
        config = FaucetConfig(data_dir=tmp_data_dir, faucets=[sample_faucet], jitter_seconds=0)
        config.ensure_dirs()
        wm = WalletManager(tmp_data_dir)
        wm.set_address(Chain.ETH, "0xtest")
        sched = FaucetScheduler(config, wm)

        mock_driver = AsyncMock()
        mock_driver.claim = AsyncMock(
            return_value=ClaimResult(
                faucet_name="test_faucet",
                chain=Chain.ETH,
                status=ClaimStatus.NETWORK_ERROR,
                error_detail="Connection refused",
            )
        )
        sched._drivers["test_faucet"] = mock_driver

        result = await sched.claim_faucet(sample_faucet)
        assert result.status == ClaimStatus.NETWORK_ERROR


class TestCLIDriverOSError:
    """Cover cli_driver.py:96-97 OSError path and amount edge cases."""

    @pytest.mark.asyncio
    async def test_claim_os_error(self, sample_config):
        faucet = FaucetDefinition(
            name="oserr",
            url="",
            chain=Chain.SUI_TESTNET,
            driver_type=DriverType.CLI,
            driver_config={
                "command": "some_cmd",
                "timeout": 5,
            },
        )
        driver = CLIDriver(faucet, sample_config)

        with patch(
            "animus_faucet.drivers.cli_driver.asyncio.create_subprocess_shell",
            side_effect=OSError("exec failed"),
        ):
            result = await driver.claim("0xaddr")

        assert result.status == ClaimStatus.UNKNOWN_ERROR
        assert "exec failed" in result.error_detail

    @pytest.mark.asyncio
    async def test_health_check_os_error(self, sample_config):
        """Cover cli_driver.py:112-113."""
        faucet = FaucetDefinition(
            name="hcerr",
            url="",
            chain=Chain.SUI_TESTNET,
            driver_type=DriverType.CLI,
            driver_config={"command": "some_cmd"},
        )
        driver = CLIDriver(faucet, sample_config)

        with patch(
            "animus_faucet.drivers.cli_driver.asyncio.create_subprocess_exec",
            side_effect=OSError("exec failed"),
        ):
            assert await driver.health_check() is False

    @pytest.mark.asyncio
    async def test_amount_value_error(self, sample_config):
        """Cover cli_driver.py:86-87 — ValueError on float parse."""
        faucet = FaucetDefinition(
            name="valerr",
            url="",
            chain=Chain.SUI_TESTNET,
            driver_type=DriverType.CLI,
            driver_config={
                "command": "echo 'Request successful: not_a_number SUI'",
                "success_pattern": "Request successful",
                "amount_pattern": r"successful: (\w+) SUI",
                "timeout": 5,
            },
        )
        driver = CLIDriver(faucet, sample_config)
        result = await driver.claim("0xaddr")
        assert result.succeeded is True
        assert result.amount is None


class TestFaucetToolsClaim:
    """Cover faucet_tools.py claim paths."""

    def _setup(self, tmp_data_dir):
        import animus_faucet.tools.faucet_tools as ft

        ft._config = None
        ft._scheduler = None
        ft._wallet_manager = None

        faucet = FaucetDefinition(
            name="test_api",
            url="http://x",
            chain=Chain.ETH,
            driver_type=DriverType.API,
            driver_config={},
        )
        config = FaucetConfig(data_dir=tmp_data_dir, faucets=[faucet], jitter_seconds=0)
        config.ensure_dirs()
        ft._config = config
        ft._wallet_manager = WalletManager(tmp_data_dir)
        ft._wallet_manager.set_address(Chain.ETH, "0xtest")
        ft._scheduler = FaucetScheduler(config, ft._wallet_manager)
        return ft

    def test_claim_specific_found(self, tmp_data_dir):
        ft = self._setup(tmp_data_dir)
        from animus_faucet.tools.faucet_tools import faucet_claim

        mock_result = ClaimResult(
            faucet_name="test_api",
            chain=Chain.ETH,
            status=ClaimStatus.SUCCESS,
            amount_usd=0.01,
        )

        with patch.object(
            ft._scheduler, "claim_faucet", new_callable=AsyncMock, return_value=mock_result
        ):
            with patch("animus_faucet.tools.faucet_tools.asyncio") as mock_asyncio:
                mock_asyncio.get_event_loop.return_value.run_until_complete.return_value = (
                    mock_result
                )
                result = faucet_claim({"faucet_name": "test_api"})

        assert result["success"] is True

    def test_claim_all(self, tmp_data_dir):
        self._setup(tmp_data_dir)
        from animus_faucet.tools.faucet_tools import faucet_claim

        mock_results = [
            ClaimResult(
                faucet_name="test_api",
                chain=Chain.ETH,
                status=ClaimStatus.SUCCESS,
            )
        ]

        with patch("animus_faucet.tools.faucet_tools.asyncio") as mock_asyncio:
            mock_asyncio.get_event_loop.return_value.run_until_complete.return_value = mock_results
            result = faucet_claim({})

        assert result["success"] is True
        assert result["total"] == 1


class TestFaucetToolsInitWithYaml:
    """Cover faucet_tools.py:35-36 — yaml config path."""

    def test_init_with_yaml(self, tmp_data_dir):
        import animus_faucet.tools.faucet_tools as ft

        ft._config = None
        ft._scheduler = None
        ft._wallet_manager = None

        # Write a config yaml
        yaml_data = {
            "faucets": [
                {
                    "name": "yaml_faucet",
                    "url": "http://test",
                    "chain": "btc",
                    "driver_type": "api",
                }
            ],
            "jitter_seconds": 0,
        }
        config_file = tmp_data_dir / "faucets.yaml"
        with open(config_file, "w") as f:
            yaml.dump(yaml_data, f)

        with patch("animus_faucet.tools.faucet_tools.FaucetConfig") as mock_cls:
            mock_config = MagicMock()
            mock_config.data_dir = tmp_data_dir
            mock_config.faucets_file = config_file
            mock_config.faucets = []

            mock_config_from_yaml = MagicMock()
            mock_config_from_yaml.data_dir = tmp_data_dir
            mock_config_from_yaml.faucets = [FaucetDefinition.from_dict(yaml_data["faucets"][0])]

            mock_cls.return_value = mock_config
            mock_cls.from_yaml.return_value = mock_config_from_yaml

            from animus_faucet.tools.faucet_tools import _ensure_initialized

            sched, cfg, wm = _ensure_initialized()
            mock_cls.from_yaml.assert_called_once_with(config_file)


class TestProxyRotatorEdge:
    """Cover proxy rotator.py:98."""

    def test_brightdata_session_rotation(self):
        config = ProxyConfig(
            enabled=True,
            provider="brightdata",
            endpoint="brd.superproxy.io:22225",
            username="brd-customer-abc",
            password="pass123",
            rotation_interval_seconds=0,
        )
        rotator = ProxyRotator(config)
        rotator._state.last_rotated_at = 0
        proxy = rotator.get_proxy()
        assert proxy is not None
        assert "session" in proxy
        assert "brd-customer-abc" in proxy
