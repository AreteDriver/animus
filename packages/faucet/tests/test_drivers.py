"""Tests for faucet drivers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_faucet.drivers.api_driver import APIDriver, _interpolate
from animus_faucet.drivers.base import BaseDriver
from animus_faucet.drivers.cli_driver import CLIDriver
from animus_faucet.models import (
    Chain,
    ClaimStatus,
    DriverType,
    FaucetDefinition,
)


class TestBaseDriver:
    def test_success_helper(self, sample_faucet, sample_config):
        class ConcreteDriver(BaseDriver):
            async def claim(self, wallet_address):
                return self._success(amount=1.0, tx_hash="0x123")

        driver = ConcreteDriver(sample_faucet, sample_config)
        result = asyncio.get_event_loop().run_until_complete(driver.claim("0xaddr"))
        assert result.succeeded is True
        assert result.amount == 1.0
        assert result.faucet_name == "test_faucet"
        assert result.chain == Chain.ETH

    def test_failure_helper(self, sample_faucet, sample_config):
        class ConcreteDriver(BaseDriver):
            async def claim(self, wallet_address):
                return self._failure(ClaimStatus.BANNED, "IP blocked")

        driver = ConcreteDriver(sample_faucet, sample_config)
        result = asyncio.get_event_loop().run_until_complete(driver.claim("0xaddr"))
        assert result.succeeded is False
        assert result.status == ClaimStatus.BANNED
        assert result.error_detail == "IP blocked"

    def test_default_health_check(self, sample_faucet, sample_config):
        class ConcreteDriver(BaseDriver):
            async def claim(self, wallet_address):
                return self._success()

        driver = ConcreteDriver(sample_faucet, sample_config)
        assert asyncio.get_event_loop().run_until_complete(driver.health_check()) is True


class TestInterpolate:
    def test_simple(self):
        result = _interpolate({"address": "{address}"}, "0xabc")
        assert result == {"address": "0xabc"}

    def test_nested(self):
        result = _interpolate({"outer": {"recipient": "{address}"}}, "0xabc")
        assert result == {"outer": {"recipient": "0xabc"}}

    def test_non_string_preserved(self):
        result = _interpolate({"amount": 100, "flag": True}, "0xabc")
        assert result == {"amount": 100, "flag": True}

    def test_no_placeholder(self):
        result = _interpolate({"key": "value"}, "0xabc")
        assert result == {"key": "value"}


class TestAPIDriver:
    @pytest.fixture
    def api_faucet(self):
        return FaucetDefinition(
            name="api_test",
            url="https://faucet.example.com/claim",
            chain=Chain.ETH,
            driver_type=DriverType.API,
            driver_config={
                "method": "POST",
                "address_field": "wallet",
                "success_field": "ok",
                "amount_field": "payout",
                "headers": {"X-Api-Key": "test123"},
            },
        )

    @pytest.fixture
    def api_driver(self, api_faucet, sample_config):
        return APIDriver(api_faucet, sample_config)

    def test_init(self, api_driver):
        assert api_driver.method == "POST"
        assert api_driver.address_field == "wallet"
        assert api_driver.success_field == "ok"
        assert api_driver.amount_field == "payout"
        assert api_driver.extra_headers == {"X-Api-Key": "test123"}

    @pytest.mark.asyncio
    async def test_claim_success(self, api_driver):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ok": True,
            "payout": 0.001,
            "tx_hash": "0xfeed",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("animus_faucet.drivers.api_driver.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await api_driver.claim("0xmywallet")

        assert result.succeeded is True
        assert result.amount == 0.001
        assert result.tx_hash == "0xfeed"

    @pytest.mark.asyncio
    async def test_claim_rate_limited(self, api_driver):
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch("animus_faucet.drivers.api_driver.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await api_driver.claim("0xmywallet")

        assert result.status == ClaimStatus.RATE_LIMITED

    @pytest.mark.asyncio
    async def test_claim_banned(self, api_driver):
        mock_response = MagicMock()
        mock_response.status_code = 403

        with patch("animus_faucet.drivers.api_driver.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await api_driver.claim("0xmywallet")

        assert result.status == ClaimStatus.BANNED

    @pytest.mark.asyncio
    async def test_claim_server_error(self, api_driver):
        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch("animus_faucet.drivers.api_driver.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await api_driver.claim("0xmywallet")

        assert result.status == ClaimStatus.FAUCET_DOWN

    @pytest.mark.asyncio
    async def test_claim_connect_error(self, api_driver):
        import httpx

        with patch("animus_faucet.drivers.api_driver.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await api_driver.claim("0xmywallet")

        assert result.status == ClaimStatus.NETWORK_ERROR

    @pytest.mark.asyncio
    async def test_claim_timeout(self, api_driver):
        import httpx

        with patch("animus_faucet.drivers.api_driver.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await api_driver.claim("0xmywallet")

        assert result.status == ClaimStatus.NETWORK_ERROR

    @pytest.mark.asyncio
    async def test_claim_http_status_error(self, api_driver):
        import httpx

        with patch("animus_faucet.drivers.api_driver.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "err", request=MagicMock(), response=mock_resp
            )
            mock_resp.json.return_value = {}
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await api_driver.claim("0xmywallet")

        assert result.status == ClaimStatus.UNKNOWN_ERROR

    @pytest.mark.asyncio
    async def test_claim_api_error_response(self, api_driver):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": False, "error": "Rate limit exceeded"}
        mock_response.raise_for_status = MagicMock()

        with patch("animus_faucet.drivers.api_driver.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await api_driver.claim("0xmywallet")

        assert result.succeeded is False
        assert result.error_detail == "Rate limit exceeded"

    @pytest.mark.asyncio
    async def test_claim_get_method(self, sample_config):
        faucet = FaucetDefinition(
            name="get_faucet",
            url="https://faucet.example.com/claim",
            chain=Chain.BTC,
            driver_type=DriverType.API,
            driver_config={"method": "GET"},
        )
        driver = APIDriver(faucet, sample_config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"txHash": "0xget"}
        mock_response.raise_for_status = MagicMock()

        with patch("animus_faucet.drivers.api_driver.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await driver.claim("0xaddr")

        assert result.succeeded is True
        assert result.tx_hash == "0xget"

    @pytest.mark.asyncio
    async def test_claim_with_payload_template(self, sample_config):
        faucet = FaucetDefinition(
            name="template_faucet",
            url="https://faucet.sui.io",
            chain=Chain.SUI_TESTNET,
            driver_type=DriverType.API,
            driver_config={
                "payload_template": {"FixedAmountRequest": {"recipient": "{address}"}},
            },
        )
        driver = APIDriver(faucet, sample_config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"transferredGasObjects": []}
        mock_response.raise_for_status = MagicMock()

        with patch("animus_faucet.drivers.api_driver.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await driver.claim("0xsui_addr")

        assert result.succeeded is True
        # Verify the template was interpolated
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["FixedAmountRequest"]["recipient"] == "0xsui_addr"

    @pytest.mark.asyncio
    async def test_health_check_success(self, api_driver):
        with patch("animus_faucet.drivers.api_driver.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_client.head = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            assert await api_driver.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, api_driver):
        import httpx

        with patch("animus_faucet.drivers.api_driver.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.head = AsyncMock(side_effect=httpx.ConnectError("down"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            assert await api_driver.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_timeout(self, api_driver):
        import httpx

        with patch("animus_faucet.drivers.api_driver.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.head = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            assert await api_driver.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_server_error(self, api_driver):
        with patch("animus_faucet.drivers.api_driver.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 502
            mock_client.head = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            assert await api_driver.health_check() is False

    def test_parse_response_no_success_field(self, sample_config):
        faucet = FaucetDefinition(
            name="nosf",
            url="http://x",
            chain=Chain.BTC,
            driver_type=DriverType.API,
            driver_config={"amount_field": "coins"},
        )
        driver = APIDriver(faucet, sample_config)
        result = driver._parse_response({"coins": "0.005", "hash": "0xh"})
        assert result.succeeded is True
        assert result.amount == 0.005
        assert result.tx_hash == "0xh"

    def test_parse_response_api_error_message(self, sample_config):
        faucet = FaucetDefinition(
            name="errmsg",
            url="http://x",
            chain=Chain.BTC,
            driver_type=DriverType.API,
            driver_config={"success_field": "ok"},
        )
        driver = APIDriver(faucet, sample_config)
        result = driver._parse_response({"ok": False, "message": "Try again later"})
        assert result.succeeded is False
        assert result.error_detail == "Try again later"

    def test_parse_response_unknown_api_error(self, sample_config):
        faucet = FaucetDefinition(
            name="unkerr",
            url="http://x",
            chain=Chain.BTC,
            driver_type=DriverType.API,
            driver_config={"success_field": "ok"},
        )
        driver = APIDriver(faucet, sample_config)
        result = driver._parse_response({"ok": False})
        assert result.succeeded is False
        assert result.error_detail == "Unknown API error"


class TestCLIDriver:
    @pytest.fixture
    def cli_driver(self, sample_cli_faucet, sample_config):
        return CLIDriver(sample_cli_faucet, sample_config)

    def test_init(self, cli_driver):
        assert "echo" in cli_driver.command_template
        assert cli_driver.success_pattern == "Request successful"
        assert cli_driver.timeout == 10

    @pytest.mark.asyncio
    async def test_claim_success(self, cli_driver):
        result = await cli_driver.claim("0xsui_addr")
        assert result.succeeded is True
        assert result.amount == 1.0

    @pytest.mark.asyncio
    async def test_claim_no_command(self, sample_config):
        faucet = FaucetDefinition(
            name="empty",
            url="",
            chain=Chain.SUI_TESTNET,
            driver_type=DriverType.CLI,
            driver_config={"command": ""},
        )
        driver = CLIDriver(faucet, sample_config)
        result = await driver.claim("0xaddr")
        assert result.status == ClaimStatus.UNKNOWN_ERROR
        assert "No command configured" in result.error_detail

    @pytest.mark.asyncio
    async def test_claim_command_fails(self, sample_config):
        faucet = FaucetDefinition(
            name="fail",
            url="",
            chain=Chain.SUI_TESTNET,
            driver_type=DriverType.CLI,
            driver_config={
                "command": "false",  # exits with code 1
                "success_pattern": "ok",
                "timeout": 5,
            },
        )
        driver = CLIDriver(faucet, sample_config)
        result = await driver.claim("0xaddr")
        assert result.status == ClaimStatus.UNKNOWN_ERROR
        assert "Exit code 1" in result.error_detail

    @pytest.mark.asyncio
    async def test_claim_pattern_not_found(self, sample_config):
        faucet = FaucetDefinition(
            name="nomatch",
            url="",
            chain=Chain.SUI_TESTNET,
            driver_type=DriverType.CLI,
            driver_config={
                "command": "echo 'something else'",
                "success_pattern": "Request successful",
                "timeout": 5,
            },
        )
        driver = CLIDriver(faucet, sample_config)
        result = await driver.claim("0xaddr")
        assert result.status == ClaimStatus.UNKNOWN_ERROR
        assert "Success pattern not found" in result.error_detail

    @pytest.mark.asyncio
    async def test_claim_timeout(self, sample_config):
        faucet = FaucetDefinition(
            name="timeout",
            url="",
            chain=Chain.SUI_TESTNET,
            driver_type=DriverType.CLI,
            driver_config={
                "command": "sleep 10",
                "success_pattern": "ok",
                "timeout": 1,
            },
        )
        driver = CLIDriver(faucet, sample_config)
        result = await driver.claim("0xaddr")
        assert result.status == ClaimStatus.NETWORK_ERROR
        assert "timed out" in result.error_detail

    @pytest.mark.asyncio
    async def test_claim_no_amount_pattern(self, sample_config):
        faucet = FaucetDefinition(
            name="noamt",
            url="",
            chain=Chain.SUI_TESTNET,
            driver_type=DriverType.CLI,
            driver_config={
                "command": "echo 'Request successful'",
                "success_pattern": "Request successful",
                "timeout": 5,
            },
        )
        driver = CLIDriver(faucet, sample_config)
        result = await driver.claim("0xaddr")
        assert result.succeeded is True
        assert result.amount is None

    @pytest.mark.asyncio
    async def test_claim_amount_parse_failure(self, sample_config):
        faucet = FaucetDefinition(
            name="badamt",
            url="",
            chain=Chain.SUI_TESTNET,
            driver_type=DriverType.CLI,
            driver_config={
                "command": "echo 'Request successful: abc SUI'",
                "success_pattern": "Request successful",
                "amount_pattern": r"bad_group_pattern",
                "timeout": 5,
            },
        )
        driver = CLIDriver(faucet, sample_config)
        result = await driver.claim("0xaddr")
        assert result.succeeded is True
        assert result.amount is None

    @pytest.mark.asyncio
    async def test_health_check_success(self, sample_config):
        faucet = FaucetDefinition(
            name="hc",
            url="",
            chain=Chain.SUI_TESTNET,
            driver_type=DriverType.CLI,
            driver_config={"command": "echo test"},
        )
        driver = CLIDriver(faucet, sample_config)
        assert await driver.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_no_command(self, sample_config):
        faucet = FaucetDefinition(
            name="hc_empty",
            url="",
            chain=Chain.SUI_TESTNET,
            driver_type=DriverType.CLI,
            driver_config={"command": ""},
        )
        driver = CLIDriver(faucet, sample_config)
        assert await driver.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_missing_binary(self, sample_config):
        faucet = FaucetDefinition(
            name="hc_missing",
            url="",
            chain=Chain.SUI_TESTNET,
            driver_type=DriverType.CLI,
            driver_config={"command": "nonexistent_binary_xyz123 --help"},
        )
        driver = CLIDriver(faucet, sample_config)
        assert await driver.health_check() is False

    @pytest.mark.asyncio
    async def test_claim_no_success_pattern(self, sample_config):
        faucet = FaucetDefinition(
            name="nopattern",
            url="",
            chain=Chain.SUI_TESTNET,
            driver_type=DriverType.CLI,
            driver_config={
                "command": "echo 'done'",
                "success_pattern": "",
                "timeout": 5,
            },
        )
        driver = CLIDriver(faucet, sample_config)
        result = await driver.claim("0xaddr")
        assert result.succeeded is True
