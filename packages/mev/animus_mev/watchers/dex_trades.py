"""DEX trade event watcher."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from animus_mev.config import ChainConfig
from animus_mev.models import MempoolTx
from animus_mev.watchers.base import BaseWatcher

logger = logging.getLogger("animus_mev.watchers.dex_trades")

# Uniswap V3 Swap event topic
SWAP_EVENT_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"


class DexTradeWatcher(BaseWatcher):
    """Watch DEX swap events via eth_subscribe logs.

    Monitors Swap events from known DEX routers to detect
    large trades that create MEV opportunities.
    """

    def __init__(self, chain_config: ChainConfig):
        super().__init__(chain_config)
        self._ws = None

    async def connect(self) -> None:
        """Connect and subscribe to DEX swap logs."""
        try:
            import websockets

            self._ws = await websockets.connect(self.chain_config.ws_url)
            subscribe_msg = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_subscribe",
                "params": [
                    "logs",
                    {
                        "topics": [SWAP_EVENT_TOPIC],
                    },
                ],
            }
            await self._ws.send(json.dumps(subscribe_msg))
            response = await self._ws.recv()
            data = json.loads(response)
            sub_id = data.get("result")
            logger.info(
                "Subscribed to DEX swaps on %s (sub_id=%s)",
                self.chain_config.chain.value,
                sub_id,
            )
        except ImportError:
            logger.error("websockets package not installed")
            raise
        except Exception:
            logger.exception("Failed to connect for DEX watching on %s", self.chain_config.ws_url)
            raise

    async def disconnect(self) -> None:
        """Close WebSocket."""
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                logger.debug("Error closing DEX watcher WebSocket", exc_info=True)
            self._ws = None

    async def watch(self) -> AsyncIterator[MempoolTx]:
        """Yield DEX swap events as MempoolTx objects."""
        if not self._ws:
            return

        while self._running:
            try:
                raw = await self._ws.recv()
                msg = json.loads(raw)

                if "params" not in msg:
                    continue

                log_data = msg["params"].get("result", {})
                tx = _parse_swap_log(log_data, self.chain_config)
                if tx:
                    yield tx

            except Exception:
                if self._running:
                    logger.debug("DEX watch error", exc_info=True)
                break


def _parse_swap_log(log_data: dict, chain_config: ChainConfig) -> MempoolTx | None:
    """Parse a swap log event into a MempoolTx."""
    try:
        tx_hash = log_data.get("transactionHash", "")
        address = log_data.get("address", "").lower()
        block_hex = log_data.get("blockNumber", "0x0")
        block_number = int(block_hex, 16) if block_hex else None
        data = log_data.get("data", "0x")

        # Estimate value from log data (amount0/amount1 in Uniswap V3 Swap)
        value = _estimate_swap_value(data)

        return MempoolTx(
            chain=chain_config.chain,
            tx_hash=tx_hash,
            from_addr="",  # Log events don't include sender
            to_addr=address,
            value=value,
            gas_price=0,
            input_data=data,
            block_number=block_number,
            timestamp=datetime.now(UTC),
        )
    except (ValueError, TypeError):
        logger.debug("Failed to parse swap log")
        return None


def _estimate_swap_value(data: str) -> int:
    """Estimate swap value from Uniswap V3 Swap event data.

    The data field contains: amount0 (int256), amount1 (int256),
    sqrtPriceX96 (uint160), liquidity (uint128), tick (int24).
    We use amount0 as a rough proxy.
    """
    if len(data) < 66:
        return 0
    try:
        # First 32 bytes (64 hex chars after 0x) is amount0
        amount0_hex = data[2:66]
        amount0 = int(amount0_hex, 16)
        # Handle signed int256
        if amount0 >= 2**255:
            amount0 = amount0 - 2**256
        return abs(amount0)
    except (ValueError, IndexError):
        return 0
