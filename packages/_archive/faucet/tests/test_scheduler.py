"""Tests for the faucet scheduler."""

from unittest.mock import AsyncMock

import pytest

from animus_faucet.models import (
    Chain,
    ClaimResult,
    ClaimStatus,
    DriverType,
    FaucetDefinition,
    FaucetHealth,
)
from animus_faucet.scheduler import DRIVER_MAP, FaucetScheduler
from animus_faucet.wallet.manager import WalletManager


@pytest.fixture
def wallet_manager(tmp_data_dir):
    wm = WalletManager(tmp_data_dir)
    wm.set_address(Chain.ETH, "0xtest_wallet")
    wm.set_address(Chain.BTC, "bc1test")
    wm.set_address(Chain.SUI_TESTNET, "0xsui_test")
    return wm


@pytest.fixture
def scheduler(sample_config, wallet_manager):
    return FaucetScheduler(sample_config, wallet_manager)


class TestDriverMap:
    def test_all_driver_types_mapped(self):
        for dt in DriverType:
            assert dt in DRIVER_MAP, f"Missing driver for {dt}"


class TestFaucetScheduler:
    @pytest.mark.asyncio
    async def test_claim_faucet_success(self, scheduler, sample_faucet):
        mock_driver = AsyncMock()
        mock_driver.claim = AsyncMock(
            return_value=ClaimResult(
                faucet_name="test_faucet",
                chain=Chain.ETH,
                status=ClaimStatus.SUCCESS,
                amount=0.001,
                amount_usd=0.05,
            )
        )
        scheduler._drivers["test_faucet"] = mock_driver

        result = await scheduler.claim_faucet(sample_faucet)
        assert result.succeeded is True
        assert result.amount == 0.001

    @pytest.mark.asyncio
    async def test_claim_faucet_no_wallet(self, sample_config, tmp_data_dir):
        wm = WalletManager(tmp_data_dir)  # empty wallets
        sched = FaucetScheduler(sample_config, wm)
        faucet = sample_config.faucets[0]
        result = await sched.claim_faucet(faucet)
        assert result.status == ClaimStatus.UNKNOWN_ERROR
        assert "No wallet address" in result.error_detail

    @pytest.mark.asyncio
    async def test_claim_faucet_auto_disabled(self, scheduler, sample_faucet, tmp_data_dir):
        # Set health to banned
        health = FaucetHealth(name="test_faucet", enabled=True, banned=True)
        scheduler.health_store.save(health)

        result = await scheduler.claim_faucet(sample_faucet)
        assert result.status == ClaimStatus.BANNED
        assert "Auto-disabled" in result.error_detail

    @pytest.mark.asyncio
    async def test_claim_faucet_auto_disabled_failures(self, scheduler, sample_faucet):
        health = FaucetHealth(name="test_faucet", enabled=True, consecutive_failures=10)
        scheduler.health_store.save(health)

        result = await scheduler.claim_faucet(sample_faucet)
        assert result.status == ClaimStatus.RATE_LIMITED

    @pytest.mark.asyncio
    async def test_claim_all(self, scheduler):
        mock_driver = AsyncMock()
        mock_driver.claim = AsyncMock(
            return_value=ClaimResult(
                faucet_name="test_faucet",
                chain=Chain.ETH,
                status=ClaimStatus.SUCCESS,
            )
        )
        scheduler._drivers["test_faucet"] = mock_driver

        results = await scheduler.claim_all()
        assert len(results) == 1
        assert results[0].succeeded is True

    @pytest.mark.asyncio
    async def test_claim_all_skips_disabled(self, sample_config, wallet_manager):
        disabled_faucet = FaucetDefinition(
            name="disabled",
            url="http://x",
            chain=Chain.BTC,
            driver_type=DriverType.API,
            enabled=False,
        )
        sample_config.faucets.append(disabled_faucet)
        sched = FaucetScheduler(sample_config, wallet_manager)

        mock_driver = AsyncMock()
        mock_driver.claim = AsyncMock(
            return_value=ClaimResult(
                faucet_name="test_faucet",
                chain=Chain.ETH,
                status=ClaimStatus.SUCCESS,
            )
        )
        sched._drivers["test_faucet"] = mock_driver

        results = await sched.claim_all()
        assert len(results) == 1  # only enabled faucet

    @pytest.mark.asyncio
    async def test_run_once(self, scheduler):
        mock_driver = AsyncMock()
        mock_driver.claim = AsyncMock(
            return_value=ClaimResult(
                faucet_name="test_faucet",
                chain=Chain.ETH,
                status=ClaimStatus.SUCCESS,
            )
        )
        scheduler._drivers["test_faucet"] = mock_driver

        results = await scheduler.run_once()
        assert len(results) == 1

    def test_get_summary_empty(self, scheduler):
        summary = scheduler.get_summary()
        assert summary["total_faucets"] == 1
        assert summary["enabled"] == 1
        assert summary["total_earned_usd"] == 0.0
        assert len(summary["faucets"]) == 1
        assert summary["faucets"][0].get("no_data") is True

    def test_get_summary_with_health(self, scheduler):
        health = FaucetHealth(
            name="test_faucet",
            enabled=True,
            success_count=10,
            total_earned_usd=0.5,
        )
        scheduler.health_store.save(health)

        summary = scheduler.get_summary()
        assert summary["total_earned_usd"] == 0.5
        assert summary["faucets"][0]["success_count"] == 10

    def test_get_driver_creates_instance(self, scheduler, sample_faucet):
        driver = scheduler._get_driver(sample_faucet)
        assert driver is not None
        # Second call returns cached
        assert scheduler._get_driver(sample_faucet) is driver

    def test_get_driver_unknown_type(self, scheduler):
        faucet = FaucetDefinition(
            name="bad",
            url="http://x",
            chain=Chain.BTC,
            driver_type=DriverType.API,
        )
        # Monkey-patch an invalid type
        faucet.driver_type = "invalid"
        with pytest.raises(ValueError):
            scheduler._get_driver(faucet)

    @pytest.mark.asyncio
    async def test_health_updated_on_claim(self, scheduler, sample_faucet):
        mock_driver = AsyncMock()
        mock_driver.claim = AsyncMock(
            return_value=ClaimResult(
                faucet_name="test_faucet",
                chain=Chain.ETH,
                status=ClaimStatus.SUCCESS,
                amount_usd=0.05,
            )
        )
        scheduler._drivers["test_faucet"] = mock_driver

        await scheduler.claim_faucet(sample_faucet)

        health = scheduler.health_store.get("test_faucet")
        assert health is not None
        assert health.success_count == 1
        assert health.total_earned_usd == 0.05
