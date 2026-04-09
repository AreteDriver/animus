"""Tests for data models."""

from datetime import datetime

import pytest

from animus_faucet.models import (
    CaptchaType,
    Chain,
    ClaimResult,
    ClaimStatus,
    DriverType,
    FaucetDefinition,
    FaucetHealth,
    TreasuryBalance,
)


class TestChain:
    def test_values(self):
        assert Chain.BTC.value == "btc"
        assert Chain.ETH.value == "eth"
        assert Chain.SUI_TESTNET.value == "sui_testnet"
        assert Chain.BASE_SEPOLIA.value == "base_sepolia"

    def test_from_string(self):
        assert Chain("btc") == Chain.BTC
        assert Chain("sui_testnet") == Chain.SUI_TESTNET

    def test_invalid_chain(self):
        with pytest.raises(ValueError):
            Chain("invalid")


class TestCaptchaType:
    def test_values(self):
        assert CaptchaType.NONE.value == "none"
        assert CaptchaType.HCAPTCHA.value == "hcaptcha"
        assert CaptchaType.RECAPTCHA_V2.value == "recaptcha_v2"
        assert CaptchaType.TURNSTILE.value == "turnstile"
        assert CaptchaType.CUSTOM.value == "custom"
        assert CaptchaType.RECAPTCHA_V3.value == "recaptcha_v3"


class TestClaimStatus:
    def test_all_statuses(self):
        assert ClaimStatus.SUCCESS.value == "success"
        assert ClaimStatus.CAPTCHA_FAILED.value == "captcha_failed"
        assert ClaimStatus.RATE_LIMITED.value == "rate_limited"
        assert ClaimStatus.BANNED.value == "banned"
        assert ClaimStatus.NETWORK_ERROR.value == "network_error"
        assert ClaimStatus.FAUCET_EMPTY.value == "faucet_empty"
        assert ClaimStatus.FAUCET_DOWN.value == "faucet_down"
        assert ClaimStatus.UNKNOWN_ERROR.value == "unknown_error"


class TestDriverType:
    def test_values(self):
        assert DriverType.API.value == "api"
        assert DriverType.BROWSER.value == "browser"
        assert DriverType.CLI.value == "cli"


class TestFaucetDefinition:
    def test_creation(self, sample_faucet):
        assert sample_faucet.name == "test_faucet"
        assert sample_faucet.chain == Chain.ETH
        assert sample_faucet.driver_type == DriverType.API
        assert sample_faucet.captcha_type == CaptchaType.NONE
        assert sample_faucet.enabled is True

    def test_to_dict(self, sample_faucet):
        d = sample_faucet.to_dict()
        assert d["name"] == "test_faucet"
        assert d["chain"] == "eth"
        assert d["driver_type"] == "api"
        assert d["captcha_type"] == "none"
        assert d["enabled"] is True
        assert d["requires_proxy"] is False
        assert d["schedule_cron"] == "0 * * * *"
        assert d["expected_payout_usd"] == 0.05
        assert d["notes"] == ""

    def test_from_dict(self, sample_faucet):
        d = sample_faucet.to_dict()
        restored = FaucetDefinition.from_dict(d)
        assert restored.name == sample_faucet.name
        assert restored.chain == sample_faucet.chain
        assert restored.driver_type == sample_faucet.driver_type
        assert restored.captcha_type == sample_faucet.captcha_type
        assert restored.enabled == sample_faucet.enabled
        assert restored.driver_config == sample_faucet.driver_config

    def test_from_dict_defaults(self):
        d = {"name": "min", "url": "http://x", "chain": "btc", "driver_type": "api"}
        f = FaucetDefinition.from_dict(d)
        assert f.captcha_type == CaptchaType.NONE
        assert f.schedule_cron == "0 * * * *"
        assert f.expected_payout_usd == 0.01
        assert f.enabled is True
        assert f.requires_proxy is False
        assert f.notes == ""
        assert f.driver_config == {}

    def test_defaults(self):
        f = FaucetDefinition(name="x", url="http://x", chain=Chain.BTC, driver_type=DriverType.API)
        assert f.captcha_type == CaptchaType.NONE
        assert f.schedule_cron == "0 * * * *"
        assert f.enabled is True


class TestClaimResult:
    def test_success(self, sample_claim_success):
        assert sample_claim_success.succeeded is True
        assert sample_claim_success.status == ClaimStatus.SUCCESS
        assert sample_claim_success.amount == 0.001
        assert sample_claim_success.tx_hash == "0xabc123"

    def test_failure(self, sample_claim_failure):
        assert sample_claim_failure.succeeded is False
        assert sample_claim_failure.status == ClaimStatus.RATE_LIMITED

    def test_to_dict(self, sample_claim_success):
        d = sample_claim_success.to_dict()
        assert d["faucet_name"] == "test_faucet"
        assert d["chain"] == "eth"
        assert d["status"] == "success"
        assert d["amount"] == 0.001
        assert d["amount_usd"] == 0.05
        assert d["tx_hash"] == "0xabc123"
        assert "timestamp" in d

    def test_from_dict(self, sample_claim_success):
        d = sample_claim_success.to_dict()
        restored = ClaimResult.from_dict(d)
        assert restored.faucet_name == sample_claim_success.faucet_name
        assert restored.chain == sample_claim_success.chain
        assert restored.status == sample_claim_success.status
        assert restored.amount == sample_claim_success.amount
        assert restored.succeeded is True

    def test_from_dict_minimal(self):
        d = {
            "faucet_name": "x",
            "chain": "btc",
            "status": "network_error",
            "timestamp": "2026-01-01T00:00:00",
        }
        r = ClaimResult.from_dict(d)
        assert r.amount is None
        assert r.tx_hash is None
        assert r.succeeded is False

    def test_default_timestamp(self):
        r = ClaimResult(faucet_name="x", chain=Chain.BTC, status=ClaimStatus.SUCCESS)
        assert isinstance(r.timestamp, datetime)

    def test_proxy_used(self):
        r = ClaimResult(
            faucet_name="x",
            chain=Chain.BTC,
            status=ClaimStatus.SUCCESS,
            proxy_used="http://proxy:8080",
        )
        d = r.to_dict()
        assert d["proxy_used"] == "http://proxy:8080"


class TestFaucetHealth:
    def test_initial_state(self):
        h = FaucetHealth(name="test", enabled=True)
        assert h.success_rate == 0.0
        assert h.should_disable is False
        assert h.success_count == 0
        assert h.fail_count == 0
        assert h.total_earned_usd == 0.0
        assert h.consecutive_failures == 0
        assert h.banned is False

    def test_record_success(self, sample_claim_success):
        h = FaucetHealth(name="test_faucet", enabled=True)
        h.record_claim(sample_claim_success)
        assert h.success_count == 1
        assert h.fail_count == 0
        assert h.consecutive_failures == 0
        assert h.total_earned_usd == 0.05
        assert h.success_rate == 1.0

    def test_record_failure(self, sample_claim_failure):
        h = FaucetHealth(name="test_faucet", enabled=True)
        h.record_claim(sample_claim_failure)
        assert h.success_count == 0
        assert h.fail_count == 1
        assert h.consecutive_failures == 1

    def test_record_ban(self):
        h = FaucetHealth(name="test", enabled=True)
        banned_result = ClaimResult(
            faucet_name="test",
            chain=Chain.BTC,
            status=ClaimStatus.BANNED,
        )
        h.record_claim(banned_result)
        assert h.banned is True
        assert h.should_disable is True

    def test_auto_disable_after_10_failures(self, sample_claim_failure):
        h = FaucetHealth(name="test_faucet", enabled=True)
        for _ in range(10):
            h.record_claim(sample_claim_failure)
        assert h.consecutive_failures == 10
        assert h.should_disable is True

    def test_consecutive_failures_reset_on_success(
        self, sample_claim_failure, sample_claim_success
    ):
        h = FaucetHealth(name="test_faucet", enabled=True)
        for _ in range(5):
            h.record_claim(sample_claim_failure)
        assert h.consecutive_failures == 5
        h.record_claim(sample_claim_success)
        assert h.consecutive_failures == 0
        assert h.should_disable is False

    def test_success_rate_mixed(self, sample_claim_success, sample_claim_failure):
        h = FaucetHealth(name="test_faucet", enabled=True)
        h.record_claim(sample_claim_success)
        h.record_claim(sample_claim_failure)
        assert h.success_rate == 0.5

    def test_to_dict(self):
        h = FaucetHealth(name="test", enabled=True, success_count=5, fail_count=2)
        d = h.to_dict()
        assert d["name"] == "test"
        assert d["enabled"] is True
        assert d["success_count"] == 5
        assert d["fail_count"] == 2

    def test_from_dict(self):
        d = {
            "name": "test",
            "enabled": True,
            "last_claim_at": "2026-01-01T00:00:00",
            "last_status": "success",
            "success_count": 10,
            "fail_count": 2,
            "total_earned_usd": 0.5,
            "consecutive_failures": 0,
            "banned": False,
        }
        h = FaucetHealth.from_dict(d)
        assert h.name == "test"
        assert h.success_count == 10
        assert h.last_status == ClaimStatus.SUCCESS

    def test_from_dict_none_fields(self):
        d = {
            "name": "test",
            "enabled": True,
            "last_claim_at": None,
            "last_status": None,
        }
        h = FaucetHealth.from_dict(d)
        assert h.last_claim_at is None
        assert h.last_status is None

    def test_record_claim_no_usd(self):
        h = FaucetHealth(name="test", enabled=True)
        result = ClaimResult(
            faucet_name="test",
            chain=Chain.BTC,
            status=ClaimStatus.SUCCESS,
            amount=100,
            amount_usd=None,
        )
        h.record_claim(result)
        assert h.total_earned_usd == 0.0
        assert h.success_count == 1


class TestTreasuryBalance:
    def test_creation(self):
        b = TreasuryBalance(
            chain=Chain.ETH,
            address="0xabc",
            balance=1.5,
            balance_usd=3000.0,
        )
        assert b.chain == Chain.ETH
        assert b.balance == 1.5

    def test_to_dict(self):
        b = TreasuryBalance(chain=Chain.BTC, address="bc1qxyz", balance=0.001, balance_usd=60.0)
        d = b.to_dict()
        assert d["chain"] == "btc"
        assert d["address"] == "bc1qxyz"
        assert d["balance"] == 0.001
        assert d["balance_usd"] == 60.0

    def test_from_dict(self):
        d = {
            "chain": "eth",
            "address": "0xabc",
            "balance": 2.0,
            "balance_usd": 4000.0,
            "last_updated": "2026-01-01T00:00:00",
        }
        b = TreasuryBalance.from_dict(d)
        assert b.chain == Chain.ETH
        assert b.balance == 2.0

    def test_default_timestamp(self):
        b = TreasuryBalance(chain=Chain.BTC, address="x", balance=0.0)
        assert isinstance(b.last_updated, datetime)

    def test_none_usd(self):
        b = TreasuryBalance(chain=Chain.BTC, address="x", balance=0.5)
        assert b.balance_usd is None
        d = b.to_dict()
        assert d["balance_usd"] is None
