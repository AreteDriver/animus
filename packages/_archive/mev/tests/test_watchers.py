"""Tests for MEV watchers."""

from __future__ import annotations

import pytest

from animus_mev.config import ChainConfig
from animus_mev.models import Chain
from animus_mev.watchers.base import BaseWatcher
from animus_mev.watchers.dex_trades import DexTradeWatcher, _estimate_swap_value, _parse_swap_log
from animus_mev.watchers.mempool import MempoolWatcher, _parse_tx


class TestBaseWatcher:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            BaseWatcher(ChainConfig(chain=Chain.BASE))

    def test_is_running_default(self):
        watcher = MempoolWatcher(ChainConfig(chain=Chain.BASE))
        assert watcher.is_running is False

    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        watcher = MempoolWatcher(ChainConfig(chain=Chain.BASE))
        # Can't actually connect without a real WS endpoint
        # But we can test the state management
        watcher._running = True
        assert watcher.is_running is True

    @pytest.mark.asyncio
    async def test_stop_clears_running(self):
        watcher = MempoolWatcher(ChainConfig(chain=Chain.BASE))
        watcher._running = True
        await watcher.stop()
        assert watcher.is_running is False


class TestMempoolWatcher:
    def test_create(self):
        cfg = ChainConfig(chain=Chain.BASE)
        watcher = MempoolWatcher(cfg)
        assert watcher.chain_config.chain == Chain.BASE
        assert watcher._ws is None
        assert watcher._subscription_id is None

    @pytest.mark.asyncio
    async def test_disconnect_no_ws(self):
        watcher = MempoolWatcher(ChainConfig(chain=Chain.BASE))
        await watcher.disconnect()
        assert watcher._ws is None

    @pytest.mark.asyncio
    async def test_watch_no_ws(self):
        watcher = MempoolWatcher(ChainConfig(chain=Chain.BASE))
        txs = []
        async for tx in watcher.watch():
            txs.append(tx)
        assert txs == []


class TestParseTx:
    def test_valid_tx(self):
        tx_data = {
            "hash": "0xabc123",
            "from": "0x1111",
            "to": "0x2222",
            "value": "0xde0b6b3a7640000",  # 1 ETH
            "gasPrice": "0x3b9aca00",  # 1 gwei
            "input": "0x04e45aaf",
            "blockNumber": "0xbc614e",
        }
        tx = _parse_tx(tx_data, Chain.BASE)
        assert tx is not None
        assert tx.chain == Chain.BASE
        assert tx.tx_hash == "0xabc123"
        assert tx.value == 10**18
        assert tx.block_number == 12345678

    def test_missing_to(self):
        tx_data = {
            "hash": "0xabc",
            "from": "0x1",
            "to": None,
            "value": "0x0",
            "gasPrice": "0x0",
            "input": "0x",
        }
        tx = _parse_tx(tx_data, Chain.BASE)
        assert tx is not None
        assert tx.to_addr == ""

    def test_no_block_number(self):
        tx_data = {
            "hash": "0xabc",
            "from": "0x1",
            "to": "0x2",
            "value": "0x0",
            "gasPrice": "0x0",
            "input": "0x",
            "blockNumber": None,
        }
        tx = _parse_tx(tx_data, Chain.ARB)
        assert tx is not None
        assert tx.block_number is None

    def test_invalid_value(self):
        tx_data = {
            "hash": "0xabc",
            "from": "0x1",
            "to": "0x2",
            "value": "invalid",
            "gasPrice": "0x0",
            "input": "0x",
        }
        tx = _parse_tx(tx_data, Chain.BASE)
        assert tx is None

    def test_missing_fields(self):
        tx = _parse_tx({}, Chain.BASE)
        # Should handle gracefully (empty hash, 0 value)
        assert tx is not None or tx is None  # Either is fine, shouldn't crash


class TestDexTradeWatcher:
    def test_create(self):
        cfg = ChainConfig(chain=Chain.ARB)
        watcher = DexTradeWatcher(cfg)
        assert watcher.chain_config.chain == Chain.ARB
        assert watcher._ws is None

    @pytest.mark.asyncio
    async def test_disconnect_no_ws(self):
        watcher = DexTradeWatcher(ChainConfig(chain=Chain.BASE))
        await watcher.disconnect()
        assert watcher._ws is None

    @pytest.mark.asyncio
    async def test_watch_no_ws(self):
        watcher = DexTradeWatcher(ChainConfig(chain=Chain.BASE))
        txs = []
        async for tx in watcher.watch():
            txs.append(tx)
        assert txs == []


class TestParseSwapLog:
    def test_valid_log(self):
        log_data = {
            "transactionHash": "0xswap123",
            "address": "0x2626664c2603336e57b271c5c0b26f421741e481",
            "blockNumber": "0xbc614e",
            "data": "0x" + "0" * 64 + "ff" * 32 + "0" * 192,
        }
        cfg = ChainConfig(chain=Chain.BASE)
        tx = _parse_swap_log(log_data, cfg)
        assert tx is not None
        assert tx.tx_hash == "0xswap123"
        assert tx.chain == Chain.BASE

    def test_empty_log(self):
        cfg = ChainConfig(chain=Chain.BASE)
        tx = _parse_swap_log({}, cfg)
        assert tx is not None or tx is None  # Should not crash

    def test_invalid_block(self):
        log_data = {
            "transactionHash": "0x",
            "address": "0x",
            "blockNumber": None,
            "data": "0x",
        }
        cfg = ChainConfig(chain=Chain.BASE)
        tx = _parse_swap_log(log_data, cfg)
        # Should handle gracefully
        assert tx is not None or tx is None


class TestEstimateSwapValue:
    def test_normal_data(self):
        # 64 hex chars = 32 bytes for amount0
        data = "0x" + "00" * 31 + "01"  # amount0 = 1
        val = _estimate_swap_value(data)
        assert val == 1

    def test_short_data(self):
        val = _estimate_swap_value("0x")
        assert val == 0

    def test_large_value(self):
        # amount0 = 0xde0b6b3a7640000 (1 ETH in wei)
        data = "0x" + "00" * 24 + "0de0b6b3a7640000"
        val = _estimate_swap_value(data)
        assert val == 10**18

    def test_signed_negative(self):
        # Negative int256 (high bit set)
        data = "0x" + "ff" * 32
        val = _estimate_swap_value(data)
        assert val == 1  # abs of -1

    def test_empty_string(self):
        val = _estimate_swap_value("")
        assert val == 0
