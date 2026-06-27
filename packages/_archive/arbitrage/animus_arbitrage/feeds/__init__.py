"""Price feed adapters for DEX protocols."""

from animus_arbitrage.feeds.base import BaseFeed
from animus_arbitrage.feeds.cetus import CetusFeed
from animus_arbitrage.feeds.jupiter import JupiterFeed
from animus_arbitrage.feeds.uniswap import UniswapV3Feed

__all__ = [
    "BaseFeed",
    "CetusFeed",
    "JupiterFeed",
    "UniswapV3Feed",
]
