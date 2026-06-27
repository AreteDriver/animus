"""Domain models for DEX arbitrage detection and execution."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class DEX(str, Enum):
    """Supported decentralized exchanges."""

    UNISWAP_V3 = "uniswap_v3"
    SUSHISWAP = "sushiswap"
    CURVE = "curve"
    JUPITER = "jupiter"
    CETUS = "cetus"
    TURBOS = "turbos"
    RAYDIUM = "raydium"


class Chain(str, Enum):
    """Supported blockchain networks."""

    ETH = "eth"
    BASE = "base"
    ARB = "arb"
    OP = "op"
    SOL = "sol"
    SUI = "sui"


class ExecutionMode(str, Enum):
    """Trade execution mode."""

    DRY_RUN = "dry_run"
    LIVE = "live"


class TokenPair(BaseModel):
    """A trading pair on a specific chain."""

    base_token: str = Field(description="Base token symbol (e.g. ETH)")
    quote_token: str = Field(description="Quote token symbol (e.g. USDC)")
    chain: Chain = Field(description="Blockchain network")
    base_address: str = Field(default="", description="Base token contract address")
    quote_address: str = Field(default="", description="Quote token contract address")

    @property
    def symbol(self) -> str:
        return f"{self.base_token}/{self.quote_token}"

    def __hash__(self) -> int:
        return hash((self.base_token, self.quote_token, self.chain))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TokenPair):
            return NotImplemented
        return (
            self.base_token == other.base_token
            and self.quote_token == other.quote_token
            and self.chain == other.chain
        )


class PriceQuote(BaseModel):
    """A price quote from a specific DEX."""

    dex: DEX
    pair: TokenPair
    price: float = Field(gt=0, description="Price of base in quote terms")
    liquidity_usd: float = Field(ge=0, description="Available liquidity in USD")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    gas_estimate_usd: float = Field(ge=0, default=0.0, description="Estimated gas cost in USD")
    block_number: int | None = Field(default=None, description="Block at quote time")

    def to_dict(self) -> dict:
        return {
            "dex": self.dex.value,
            "pair": {
                "base_token": self.pair.base_token,
                "quote_token": self.pair.quote_token,
                "chain": self.pair.chain.value,
            },
            "price": self.price,
            "liquidity_usd": self.liquidity_usd,
            "timestamp": self.timestamp.isoformat(),
            "gas_estimate_usd": self.gas_estimate_usd,
            "block_number": self.block_number,
        }


class ArbitrageOpportunity(BaseModel):
    """A detected arbitrage opportunity between two DEXes."""

    opportunity_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    buy_quote: PriceQuote
    sell_quote: PriceQuote
    spread_pct: float = Field(description="Price spread as percentage")
    profit_after_gas_usd: float = Field(description="Expected profit after gas costs")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    detected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = Field(default=None, description="Opportunity expiration time")
    trade_size_usd: float = Field(default=0.0, description="Recommended trade size in USD")

    @property
    def is_cross_chain(self) -> bool:
        return self.buy_quote.pair.chain != self.sell_quote.pair.chain

    def to_dict(self) -> dict:
        return {
            "opportunity_id": self.opportunity_id,
            "buy_dex": self.buy_quote.dex.value,
            "sell_dex": self.sell_quote.dex.value,
            "buy_chain": self.buy_quote.pair.chain.value,
            "sell_chain": self.sell_quote.pair.chain.value,
            "pair": self.buy_quote.pair.symbol,
            "buy_price": self.buy_quote.price,
            "sell_price": self.sell_quote.price,
            "spread_pct": self.spread_pct,
            "profit_after_gas_usd": self.profit_after_gas_usd,
            "confidence": self.confidence,
            "detected_at": self.detected_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "trade_size_usd": self.trade_size_usd,
            "is_cross_chain": self.is_cross_chain,
        }


class TradeResult(BaseModel):
    """Result of an executed (or dry-run) trade."""

    result_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    opportunity_id: str
    executed: bool = Field(description="Whether the trade was actually executed")
    mode: ExecutionMode = Field(default=ExecutionMode.DRY_RUN)
    buy_tx: str | None = Field(default=None, description="Buy transaction hash")
    sell_tx: str | None = Field(default=None, description="Sell transaction hash")
    actual_profit_usd: float = Field(default=0.0, description="Realized profit in USD")
    slippage_pct: float = Field(default=0.0, description="Actual slippage percentage")
    error: str | None = Field(default=None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def success(self) -> bool:
        return self.executed and self.error is None

    def to_dict(self) -> dict:
        return {
            "result_id": self.result_id,
            "opportunity_id": self.opportunity_id,
            "executed": self.executed,
            "mode": self.mode.value,
            "buy_tx": self.buy_tx,
            "sell_tx": self.sell_tx,
            "actual_profit_usd": self.actual_profit_usd,
            "slippage_pct": self.slippage_pct,
            "error": self.error,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
        }
