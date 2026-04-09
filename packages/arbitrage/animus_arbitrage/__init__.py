"""Animus Arbitrage — Autonomous DEX arbitrage detection and execution."""

__version__ = "0.1.0"

from animus_arbitrage.config import ArbitrageConfig
from animus_arbitrage.detector import ArbitrageDetector
from animus_arbitrage.executor import TradeExecutor
from animus_arbitrage.models import (
    DEX,
    ArbitrageOpportunity,
    Chain,
    ExecutionMode,
    PriceQuote,
    TokenPair,
    TradeResult,
)
from animus_arbitrage.risk import RiskManager
from animus_arbitrage.store import ArbitrageStore
from animus_arbitrage.tracker import PnLTracker

__all__ = [
    "ArbitrageConfig",
    "ArbitrageDetector",
    "ArbitrageOpportunity",
    "Chain",
    "DEX",
    "ExecutionMode",
    "PnLTracker",
    "PriceQuote",
    "RiskManager",
    "TokenPair",
    "TradeExecutor",
    "TradeResult",
    "ArbitrageStore",
    "__version__",
]
