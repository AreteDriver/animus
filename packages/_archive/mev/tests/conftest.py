"""Shared test fixtures for Animus MEV."""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

from animus_mev.config import MEVConfig
from animus_mev.models import (
    Chain,
    ExtractionResult,
    MempoolTx,
    MEVOpportunity,
    MEVType,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove MEV env vars to avoid test pollution."""
    for key in list(os.environ):
        if key.startswith("MEV_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary data directory for tests."""
    d = tmp_path / "mev_data"
    d.mkdir()
    return d


@pytest.fixture
def config(tmp_data_dir):
    """MEV config for testing."""
    cfg = MEVConfig(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def sample_tx():
    """A sample DEX swap transaction on Base."""
    return MempoolTx(
        chain=Chain.BASE,
        tx_hash="0xabc123def456",
        from_addr="0x1111111111111111111111111111111111111111",
        to_addr="0x2626664c2603336e57b271c5c0b26f421741e481",  # Uniswap V3 on Base
        value=int(5e18),  # 5 ETH
        gas_price=int(0.01e9),  # 0.01 gwei
        input_data="0x04e45aaf" + "0" * 128,  # exactInputSingle selector
        block_number=12345678,
        timestamp=datetime.now(UTC),
    )


@pytest.fixture
def small_tx():
    """A small transaction that shouldn't trigger strategies."""
    return MempoolTx(
        chain=Chain.BASE,
        tx_hash="0xsmall123",
        from_addr="0x2222222222222222222222222222222222222222",
        to_addr="0x3333333333333333333333333333333333333333",
        value=int(0.001e18),  # 0.001 ETH
        gas_price=int(0.01e9),
        input_data="0x",
        block_number=12345679,
        timestamp=datetime.now(UTC),
    )


@pytest.fixture
def sample_opportunity(sample_tx):
    """A sample backrun MEV opportunity."""
    return MEVOpportunity(
        mev_type=MEVType.BACKRUN,
        chain=Chain.BASE,
        trigger_tx=sample_tx,
        estimated_profit_usd=5.0,
        gas_cost_usd=0.10,
        confidence=0.85,
        expires_at_block=12345679,
    )


@pytest.fixture
def sandwich_opportunity(sample_tx):
    """A sandwich detection opportunity — should NEVER execute."""
    return MEVOpportunity(
        mev_type=MEVType.SANDWICH_DETECT,
        chain=Chain.BASE,
        trigger_tx=sample_tx,
        estimated_profit_usd=50.0,
        gas_cost_usd=0.20,
        confidence=0.9,
        expires_at_block=12345679,
    )


@pytest.fixture
def sample_result():
    """A sample extraction result."""
    return ExtractionResult(
        opportunity_id="abc123",
        executed=True,
        tx_hash="0xresult456",
        actual_profit_usd=4.5,
        gas_used_usd=0.08,
        block_number=12345679,
    )
