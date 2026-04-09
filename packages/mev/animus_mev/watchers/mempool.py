"""Mempool transaction watcher via WebSocket."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from animus_mev.config import ChainConfig
from animus_mev.models import Chain, MempoolTx
from animus_mev.watchers.base import BaseWatcher

logger = logging.getLogger("animus_mev.watchers.mempool")


class MempoolWatcher(BaseWatcher):
    """Watch pending transactions via WebSocket subscription.

    Subscribes to newPendingTransactions on the configured chain
    and yields MempoolTx objects for each pending tx.
    """

    def __init__(self, chain_config: ChainConfig):
        super().__init__(chain_config)
        self._ws = None
        self._subscription_id: str | None = None

    async def connect(self) -> None:
        """Connect to WebSocket endpoint."""
        try:
            import websockets

            self._ws = await websockets.connect(self.chain_config.ws_url)
            # Subscribe to pending transactions
            subscribe_msg = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_subscribe",
                "params": ["newPendingTransactions"],
            }
            await self._ws.send(json.dumps(subscribe_msg))
            response = await self._ws.recv()
            data = json.loads(response)
            self._subscription_id = data.get("result")
            logger.info(
                "Subscribed to mempool on %s (sub_id=%s)",
                self.chain_config.chain.value,
                self._subscription_id,
            )
        except ImportError:
            logger.error("websockets package not installed")
            raise
        except Exception:
            logger.exception("Failed to connect to %s", self.chain_config.ws_url)
            raise

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                logger.debug("Error closing WebSocket", exc_info=True)
            self._ws = None
            self._subscription_id = None

    async def _get_tx_details(self, tx_hash: str) -> dict | None:
        """Fetch full transaction details via eth_getTransactionByHash."""
        if not self._ws:
            return None
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "eth_getTransactionByHash",
            "params": [tx_hash],
        }
        await self._ws.send(json.dumps(request))
        response = await self._ws.recv()
        data = json.loads(response)
        return data.get("result")

    async def watch(self) -> AsyncIterator[MempoolTx]:
        """Yield pending transactions from the mempool."""
        if not self._ws:
            return

        while self._running:
            try:
                raw = await self._ws.recv()
                msg = json.loads(raw)

                # Subscription notifications have params.result = tx_hash
                if "params" in msg:
                    tx_hash = msg["params"].get("result", "")
                    if not tx_hash:
                        continue

                    tx_data = await self._get_tx_details(tx_hash)
                    if not tx_data:
                        continue

                    tx = _parse_tx(tx_data, self.chain_config.chain)
                    if tx:
                        yield tx

            except Exception:
                if self._running:
                    logger.debug("Mempool watch error", exc_info=True)
                break


def _parse_tx(tx_data: dict, chain: Chain) -> MempoolTx | None:
    """Parse raw JSON-RPC tx data into MempoolTx."""

    try:
        return MempoolTx(
            chain=chain,
            tx_hash=tx_data.get("hash", ""),
            from_addr=tx_data.get("from", ""),
            to_addr=tx_data.get("to", "") or "",
            value=int(tx_data.get("value", "0x0"), 16),
            gas_price=int(tx_data.get("gasPrice", "0x0"), 16),
            input_data=tx_data.get("input", "0x"),
            block_number=(int(tx_data["blockNumber"], 16) if tx_data.get("blockNumber") else None),
            timestamp=datetime.now(UTC),
        )
    except (ValueError, TypeError):
        logger.debug("Failed to parse tx: %s", tx_data.get("hash", "unknown"))
        return None
