"""Shared test fixtures for animus_arbitrage tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from animus_arbitrage.config import ArbitrageConfig, ExecutionConfig, RiskConfig
from animus_arbitrage.models import (
    DEX,
    ArbitrageOpportunity,
    Chain,
    ExecutionMode,
    PriceQuote,
    TokenPair,
    TradeResult,
)
from animus_arbitrage.store import ArbitrageStore


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Temporary data directory for tests."""
    data_dir = tmp_path / "arbitrage_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def config(tmp_data_dir: Path) -> ArbitrageConfig:
    """Default test configuration."""
    return ArbitrageConfig(
        data_dir=tmp_data_dir,
        execution=ExecutionConfig(mode=ExecutionMode.DRY_RUN),
        risk=RiskConfig(
            min_spread_pct=0.3,
            min_liquidity_usd=1000.0,
            min_confidence=0.1,
            cooldown_seconds=0.0,
            max_trades_per_hour=100,
        ),
    )


@pytest.fixture
def live_config(tmp_data_dir: Path) -> ArbitrageConfig:
    """Live mode configuration for testing."""
    return ArbitrageConfig(
        data_dir=tmp_data_dir,
        execution=ExecutionConfig(mode=ExecutionMode.LIVE),
        risk=RiskConfig(
            min_spread_pct=0.3,
            min_liquidity_usd=1000.0,
            min_confidence=0.1,
            cooldown_seconds=0.0,
            max_trades_per_hour=100,
        ),
    )


@pytest.fixture
def store(tmp_data_dir: Path) -> ArbitrageStore:
    """Fresh arbitrage store."""
    return ArbitrageStore(tmp_data_dir)


@pytest.fixture
def eth_usdc_pair() -> TokenPair:
    """ETH/USDC on Ethereum."""
    return TokenPair(base_token="ETH", quote_token="USDC", chain=Chain.ETH)


@pytest.fixture
def sol_usdc_pair() -> TokenPair:
    """SOL/USDC on Solana."""
    return TokenPair(base_token="SOL", quote_token="USDC", chain=Chain.SOL)


@pytest.fixture
def sui_usdc_pair() -> TokenPair:
    """SUI/USDC on Sui."""
    return TokenPair(base_token="SUI", quote_token="USDC", chain=Chain.SUI)


def make_quote(
    dex: DEX = DEX.UNISWAP_V3,
    pair: TokenPair | None = None,
    price: float = 2000.0,
    liquidity: float = 100_000.0,
    gas: float = 5.0,
    chain: Chain = Chain.ETH,
) -> PriceQuote:
    """Helper to create test price quotes."""
    if pair is None:
        pair = TokenPair(base_token="ETH", quote_token="USDC", chain=chain)
    return PriceQuote(
        dex=dex,
        pair=pair,
        price=price,
        liquidity_usd=liquidity,
        timestamp=datetime.now(UTC),
        gas_estimate_usd=gas,
    )


def make_opportunity(
    spread_pct: float = 1.0,
    profit_after_gas: float = 5.0,
    confidence: float = 0.8,
    trade_size: float = 100.0,
    expired: bool = False,
) -> ArbitrageOpportunity:
    """Helper to create test opportunities."""
    pair = TokenPair(base_token="ETH", quote_token="USDC", chain=Chain.ETH)
    now = datetime.now(UTC)

    buy_quote = make_quote(dex=DEX.UNISWAP_V3, pair=pair, price=2000.0, gas=2.0)
    sell_quote = make_quote(
        dex=DEX.SUSHISWAP, pair=pair, price=2000.0 * (1 + spread_pct / 100), gas=2.0
    )

    expires_at = now - timedelta(minutes=5) if expired else now + timedelta(minutes=5)

    return ArbitrageOpportunity(
        buy_quote=buy_quote,
        sell_quote=sell_quote,
        spread_pct=spread_pct,
        profit_after_gas_usd=profit_after_gas,
        confidence=confidence,
        detected_at=now,
        expires_at=expires_at,
        trade_size_usd=trade_size,
    )


def make_result(
    opportunity_id: str = "test123",
    executed: bool = False,
    profit: float = 5.0,
    error: str | None = None,
) -> TradeResult:
    """Helper to create test trade results."""
    return TradeResult(
        opportunity_id=opportunity_id,
        executed=executed,
        mode=ExecutionMode.DRY_RUN,
        actual_profit_usd=profit,
        error=error,
    )
