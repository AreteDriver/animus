"""Data models for Animus MEV."""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime


class MEVType(str, enum.Enum):
    """Types of MEV opportunities."""

    BACKRUN = "backrun"
    ARBITRAGE = "arbitrage"
    LIQUIDATION = "liquidation"
    SANDWICH_DETECT = "sandwich_detect"  # Monitor only — never execute


class Chain(str, enum.Enum):
    """Supported blockchain networks."""

    BASE = "base"
    ARB = "arb"
    OP = "op"
    ETH = "eth"


# Well-known DEX router addresses per chain
DEX_ROUTERS: dict[Chain, set[str]] = {
    Chain.BASE: {
        "0x2626664c2603336e57b271c5c0b26f421741e481",  # Uniswap V3 SwapRouter
        "0x327df1e6de05895d2ab08513aadd9313fe505d86",  # BaseSwap
        "0xcf77a3ba9a5ca399b7c97c74d54e5b1beb874e43",  # Aerodrome
    },
    Chain.ARB: {
        "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",  # Uniswap V3 SwapRouter02
        "0x1b02da8cb0d097eb8d57a175b88c7d8b47997506",  # SushiSwap
        "0xe592427a0aece92de3edee1f18e0157c05861564",  # Uniswap V3 SwapRouter
    },
    Chain.OP: {
        "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",  # Uniswap V3 SwapRouter02
        "0xa062ae8a9c5e11aaa026fc2670b0d65ccc8b2858",  # Velodrome V2
    },
    Chain.ETH: {
        "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",  # Uniswap V3 SwapRouter02
        "0xe592427a0aece92de3edee1f18e0157c05861564",  # Uniswap V3 SwapRouter
        "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Uniswap V2 Router
    },
}

# Common DEX swap function selectors
SWAP_SELECTORS: dict[str, str] = {
    "0x38ed1739": "swapExactTokensForTokens",
    "0x8803dbee": "swapTokensForExactTokens",
    "0x7ff36ab5": "swapExactETHForTokens",
    "0x18cbafe5": "swapExactTokensForETH",
    "0x04e45aaf": "exactInputSingle",
    "0xb858183f": "exactInput",
    "0x5023b4df": "exactOutputSingle",
    "0x09b81346": "exactOutput",
}


@dataclass
class MempoolTx:
    """A pending transaction observed in the mempool."""

    chain: Chain
    tx_hash: str
    from_addr: str
    to_addr: str
    value: int  # wei
    gas_price: int  # wei
    input_data: str
    block_number: int | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def dex_detected(self) -> bool:
        """Check if this tx targets a known DEX router."""
        routers = DEX_ROUTERS.get(self.chain, set())
        return self.to_addr.lower() in routers

    @property
    def swap_selector(self) -> str | None:
        """Extract the function selector if it matches a known swap."""
        if len(self.input_data) < 10:
            return None
        selector = self.input_data[:10].lower()
        return SWAP_SELECTORS.get(selector)

    @property
    def swap_amount_usd(self) -> float:
        """Estimate swap size in USD from tx value. Rough heuristic."""
        # ETH value heuristic: value in wei -> ETH -> approx USD
        if self.value > 0:
            eth_amount = self.value / 1e18
            return eth_amount * 3000.0  # approximate ETH price
        return 0.0

    def to_dict(self) -> dict:
        return {
            "chain": self.chain.value,
            "tx_hash": self.tx_hash,
            "from_addr": self.from_addr,
            "to_addr": self.to_addr,
            "value": self.value,
            "gas_price": self.gas_price,
            "input_data": self.input_data[:66] if len(self.input_data) > 66 else self.input_data,
            "block_number": self.block_number,
            "timestamp": self.timestamp.isoformat(),
            "dex_detected": self.dex_detected,
            "swap_selector": self.swap_selector,
            "swap_amount_usd": self.swap_amount_usd,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MempoolTx:
        return cls(
            chain=Chain(data["chain"]),
            tx_hash=data["tx_hash"],
            from_addr=data["from_addr"],
            to_addr=data["to_addr"],
            value=data["value"],
            gas_price=data["gas_price"],
            input_data=data.get("input_data", "0x"),
            block_number=data.get("block_number"),
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if data.get("timestamp")
                else datetime.now(UTC)
            ),
        )


@dataclass
class MEVOpportunity:
    """A detected MEV opportunity."""

    mev_type: MEVType
    chain: Chain
    trigger_tx: MempoolTx
    estimated_profit_usd: float
    gas_cost_usd: float
    confidence: float  # 0-1
    expires_at_block: int | None = None
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict = field(default_factory=dict)

    @property
    def opportunity_id(self) -> str:
        """Deterministic ID from trigger tx hash and type."""
        raw = f"{self.mev_type.value}:{self.trigger_tx.tx_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def net_profit_usd(self) -> float:
        return self.estimated_profit_usd - self.gas_cost_usd

    @property
    def is_profitable(self) -> bool:
        return self.net_profit_usd > 0

    def to_dict(self) -> dict:
        return {
            "opportunity_id": self.opportunity_id,
            "mev_type": self.mev_type.value,
            "chain": self.chain.value,
            "trigger_tx": self.trigger_tx.to_dict(),
            "estimated_profit_usd": self.estimated_profit_usd,
            "gas_cost_usd": self.gas_cost_usd,
            "net_profit_usd": self.net_profit_usd,
            "confidence": self.confidence,
            "expires_at_block": self.expires_at_block,
            "detected_at": self.detected_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MEVOpportunity:
        return cls(
            mev_type=MEVType(data["mev_type"]),
            chain=Chain(data["chain"]),
            trigger_tx=MempoolTx.from_dict(data["trigger_tx"]),
            estimated_profit_usd=data["estimated_profit_usd"],
            gas_cost_usd=data["gas_cost_usd"],
            confidence=data.get("confidence", 0.0),
            expires_at_block=data.get("expires_at_block"),
            detected_at=(
                datetime.fromisoformat(data["detected_at"])
                if data.get("detected_at")
                else datetime.now(UTC)
            ),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExtractionResult:
    """Result of an MEV extraction attempt."""

    opportunity_id: str
    executed: bool
    tx_hash: str | None = None
    actual_profit_usd: float = 0.0
    gas_used_usd: float = 0.0
    block_number: int | None = None
    error: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def net_profit_usd(self) -> float:
        return self.actual_profit_usd - self.gas_used_usd

    @property
    def success(self) -> bool:
        return self.executed and self.error is None and self.net_profit_usd > 0

    def to_dict(self) -> dict:
        return {
            "opportunity_id": self.opportunity_id,
            "executed": self.executed,
            "tx_hash": self.tx_hash,
            "actual_profit_usd": self.actual_profit_usd,
            "gas_used_usd": self.gas_used_usd,
            "net_profit_usd": self.net_profit_usd,
            "block_number": self.block_number,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExtractionResult:
        return cls(
            opportunity_id=data["opportunity_id"],
            executed=data["executed"],
            tx_hash=data.get("tx_hash"),
            actual_profit_usd=data.get("actual_profit_usd", 0.0),
            gas_used_usd=data.get("gas_used_usd", 0.0),
            block_number=data.get("block_number"),
            error=data.get("error"),
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if data.get("timestamp")
                else datetime.now(UTC)
            ),
        )
