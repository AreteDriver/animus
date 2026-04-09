"""Shared test fixtures for animus_faucet."""

from __future__ import annotations

import pytest

from animus_faucet.config import FaucetConfig
from animus_faucet.models import (
    CaptchaType,
    Chain,
    ClaimResult,
    ClaimStatus,
    DriverType,
    FaucetDefinition,
)


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary data directory for tests."""
    data_dir = tmp_path / "faucet_data"
    data_dir.mkdir()
    (data_dir / "claims").mkdir()
    (data_dir / "health").mkdir()
    return data_dir


@pytest.fixture
def sample_faucet():
    """A sample API faucet definition."""
    return FaucetDefinition(
        name="test_faucet",
        url="https://faucet.example.com/api/claim",
        chain=Chain.ETH,
        driver_type=DriverType.API,
        captcha_type=CaptchaType.NONE,
        schedule_cron="0 * * * *",
        expected_payout_usd=0.05,
        enabled=True,
        driver_config={
            "method": "POST",
            "address_field": "address",
            "success_field": "ok",
            "amount_field": "amount",
        },
    )


@pytest.fixture
def sample_browser_faucet():
    """A sample browser faucet definition."""
    return FaucetDefinition(
        name="test_browser_faucet",
        url="https://freebitco.in",
        chain=Chain.BTC,
        driver_type=DriverType.BROWSER,
        captcha_type=CaptchaType.HCAPTCHA,
        schedule_cron="0 * * * *",
        expected_payout_usd=0.01,
        enabled=True,
        requires_proxy=True,
        driver_config={
            "address_selector": "#btc_address",
            "submit_selector": "#claim",
            "success_selector": ".result",
        },
    )


@pytest.fixture
def sample_cli_faucet():
    """A sample CLI faucet definition."""
    return FaucetDefinition(
        name="test_cli_faucet",
        url="",
        chain=Chain.SUI_TESTNET,
        driver_type=DriverType.CLI,
        captcha_type=CaptchaType.NONE,
        schedule_cron="0 */2 * * *",
        enabled=True,
        driver_config={
            "command": "echo 'Request successful: 1.0 SUI'",
            "success_pattern": "Request successful",
            "amount_pattern": r"([\d.]+) SUI",
            "timeout": 10,
        },
    )


@pytest.fixture
def sample_config(tmp_data_dir, sample_faucet):
    """A sample FaucetConfig with one faucet."""
    config = FaucetConfig(
        data_dir=tmp_data_dir,
        faucets=[sample_faucet],
        jitter_seconds=0,  # no jitter in tests
    )
    return config


@pytest.fixture
def sample_claim_success():
    """A successful claim result."""
    return ClaimResult(
        faucet_name="test_faucet",
        chain=Chain.ETH,
        status=ClaimStatus.SUCCESS,
        amount=0.001,
        amount_usd=0.05,
        tx_hash="0xabc123",
    )


@pytest.fixture
def sample_claim_failure():
    """A failed claim result."""
    return ClaimResult(
        faucet_name="test_faucet",
        chain=Chain.ETH,
        status=ClaimStatus.RATE_LIMITED,
        error_detail="HTTP 429",
    )
