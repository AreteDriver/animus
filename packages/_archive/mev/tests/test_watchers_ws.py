"""Tests for WebSocket-based watchers with mocked connections."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from animus_mev.config import ChainConfig
from animus_mev.models import Chain
from animus_mev.watchers.dex_trades import DexTradeWatcher
from animus_mev.watchers.mempool import MempoolWatcher


class TestMempoolWatcherConnect:
    @pytest.mark.asyncio
    async def test_connect_success(self):
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=json.dumps({"result": "sub_123"}))

        with patch("websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
            cfg = ChainConfig(chain=Chain.BASE, ws_url="wss://test.example.com")
            watcher = MempoolWatcher(cfg)
            await watcher.connect()

            assert watcher._ws is mock_ws
            assert watcher._subscription_id == "sub_123"
            mock_ws.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_exception(self):
        with patch(
            "websockets.connect",
            new_callable=AsyncMock,
            side_effect=ConnectionRefusedError("refused"),
        ):
            cfg = ChainConfig(chain=Chain.BASE, ws_url="wss://bad.example.com")
            watcher = MempoolWatcher(cfg)
            with pytest.raises(ConnectionRefusedError):
                await watcher.connect()

    @pytest.mark.asyncio
    async def test_disconnect_with_ws(self):
        mock_ws = AsyncMock()
        cfg = ChainConfig(chain=Chain.BASE)
        watcher = MempoolWatcher(cfg)
        watcher._ws = mock_ws
        watcher._subscription_id = "sub_123"
        await watcher.disconnect()
        mock_ws.close.assert_called_once()
        assert watcher._ws is None
        assert watcher._subscription_id is None

    @pytest.mark.asyncio
    async def test_disconnect_with_ws_error(self):
        mock_ws = AsyncMock()
        mock_ws.close = AsyncMock(side_effect=Exception("close error"))
        cfg = ChainConfig(chain=Chain.BASE)
        watcher = MempoolWatcher(cfg)
        watcher._ws = mock_ws
        await watcher.disconnect()
        assert watcher._ws is None

    @pytest.mark.asyncio
    async def test_get_tx_details(self):
        mock_ws = AsyncMock()
        tx_data = {
            "hash": "0xabc",
            "from": "0x1",
            "to": "0x2",
            "value": "0x0",
            "gasPrice": "0x0",
            "input": "0x",
        }
        mock_ws.recv = AsyncMock(return_value=json.dumps({"result": tx_data}))
        cfg = ChainConfig(chain=Chain.BASE)
        watcher = MempoolWatcher(cfg)
        watcher._ws = mock_ws
        result = await watcher._get_tx_details("0xabc")
        assert result == tx_data

    @pytest.mark.asyncio
    async def test_get_tx_details_no_ws(self):
        cfg = ChainConfig(chain=Chain.BASE)
        watcher = MempoolWatcher(cfg)
        result = await watcher._get_tx_details("0xabc")
        assert result is None


class TestMempoolWatcherWatch:
    @pytest.mark.asyncio
    async def test_watch_receives_tx(self):
        tx_data = {
            "hash": "0xfeed123",
            "from": "0x1111",
            "to": "0x2222",
            "value": "0xde0b6b3a7640000",
            "gasPrice": "0x3b9aca00",
            "input": "0x04e45aaf",
            "blockNumber": "0xbc614e",
        }

        subscription_msg = json.dumps({"params": {"result": "0xfeed123"}})
        tx_response = json.dumps({"result": tx_data})

        mock_ws = AsyncMock()
        recv_values = [subscription_msg, tx_response]
        call_count = 0

        async def mock_recv():
            nonlocal call_count
            if call_count < len(recv_values):
                val = recv_values[call_count]
                call_count += 1
                return val
            raise Exception("stop")

        mock_ws.recv = mock_recv
        mock_ws.send = AsyncMock()

        cfg = ChainConfig(chain=Chain.BASE)
        watcher = MempoolWatcher(cfg)
        watcher._ws = mock_ws
        watcher._running = True

        txs = []
        async for tx in watcher.watch():
            txs.append(tx)
            break  # Just get one

        assert len(txs) == 1
        assert txs[0].tx_hash == "0xfeed123"

    @pytest.mark.asyncio
    async def test_watch_empty_params(self):
        mock_ws = AsyncMock()
        msg = json.dumps({"params": {"result": ""}})

        call_count = 0

        async def mock_recv():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return msg
            raise Exception("stop")

        mock_ws.recv = mock_recv
        cfg = ChainConfig(chain=Chain.BASE)
        watcher = MempoolWatcher(cfg)
        watcher._ws = mock_ws
        watcher._running = True

        txs = []
        async for tx in watcher.watch():
            txs.append(tx)
        assert txs == []

    @pytest.mark.asyncio
    async def test_watch_tx_details_none(self):
        """When tx details can't be fetched, skip the tx."""
        mock_ws = AsyncMock()
        sub_msg = json.dumps({"params": {"result": "0xabc"}})
        null_response = json.dumps({"result": None})

        call_count = 0

        async def mock_recv():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return sub_msg
            if call_count == 2:
                return null_response
            raise Exception("stop")

        mock_ws.recv = mock_recv
        mock_ws.send = AsyncMock()

        cfg = ChainConfig(chain=Chain.BASE)
        watcher = MempoolWatcher(cfg)
        watcher._ws = mock_ws
        watcher._running = True

        txs = []
        async for tx in watcher.watch():
            txs.append(tx)
        assert txs == []

    @pytest.mark.asyncio
    async def test_watch_parse_failure(self):
        """When tx has invalid data, skip gracefully."""
        mock_ws = AsyncMock()
        sub_msg = json.dumps({"params": {"result": "0xbad"}})
        bad_tx = json.dumps(
            {"result": {"hash": "0xbad", "from": "0x1", "to": "0x2", "value": "invalid"}}
        )

        call_count = 0

        async def mock_recv():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return sub_msg
            if call_count == 2:
                return bad_tx
            raise Exception("stop")

        mock_ws.recv = mock_recv
        mock_ws.send = AsyncMock()

        cfg = ChainConfig(chain=Chain.BASE)
        watcher = MempoolWatcher(cfg)
        watcher._ws = mock_ws
        watcher._running = True

        txs = []
        async for tx in watcher.watch():
            txs.append(tx)
        assert txs == []

    @pytest.mark.asyncio
    async def test_watch_exception_not_running(self):
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=Exception("connection lost"))

        cfg = ChainConfig(chain=Chain.BASE)
        watcher = MempoolWatcher(cfg)
        watcher._ws = mock_ws
        watcher._running = False  # Not running — should not log

        txs = []
        async for tx in watcher.watch():
            txs.append(tx)
        assert txs == []

    @pytest.mark.asyncio
    async def test_watch_exception_while_running(self):
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=Exception("connection lost"))

        cfg = ChainConfig(chain=Chain.BASE)
        watcher = MempoolWatcher(cfg)
        watcher._ws = mock_ws
        watcher._running = True  # Running — should log

        txs = []
        async for tx in watcher.watch():
            txs.append(tx)
        assert txs == []


class TestDexTradeWatcherConnect:
    @pytest.mark.asyncio
    async def test_connect_success(self):
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=json.dumps({"result": "sub_456"}))

        with patch("websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
            cfg = ChainConfig(chain=Chain.ARB, ws_url="wss://test.example.com")
            watcher = DexTradeWatcher(cfg)
            await watcher.connect()

            assert watcher._ws is mock_ws

    @pytest.mark.asyncio
    async def test_connect_exception(self):
        with patch(
            "websockets.connect",
            new_callable=AsyncMock,
            side_effect=ConnectionRefusedError("nope"),
        ):
            cfg = ChainConfig(chain=Chain.OP, ws_url="wss://bad.example.com")
            watcher = DexTradeWatcher(cfg)
            with pytest.raises(ConnectionRefusedError):
                await watcher.connect()

    @pytest.mark.asyncio
    async def test_disconnect_with_ws(self):
        mock_ws = AsyncMock()
        cfg = ChainConfig(chain=Chain.BASE)
        watcher = DexTradeWatcher(cfg)
        watcher._ws = mock_ws
        await watcher.disconnect()
        mock_ws.close.assert_called_once()
        assert watcher._ws is None

    @pytest.mark.asyncio
    async def test_disconnect_with_ws_error(self):
        mock_ws = AsyncMock()
        mock_ws.close = AsyncMock(side_effect=Exception("boom"))
        cfg = ChainConfig(chain=Chain.BASE)
        watcher = DexTradeWatcher(cfg)
        watcher._ws = mock_ws
        await watcher.disconnect()
        assert watcher._ws is None


class TestDexTradeWatcherWatch:
    @pytest.mark.asyncio
    async def test_watch_receives_swap(self):
        log_data = {
            "transactionHash": "0xswap789",
            "address": "0x2626664c2603336e57b271c5c0b26f421741e481",
            "blockNumber": "0xbc614e",
            "data": "0x" + "00" * 31 + "01" + "ff" * 32 + "00" * 96,
        }
        msg = json.dumps({"params": {"result": log_data}})

        call_count = 0

        async def mock_recv():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return msg
            raise Exception("stop")

        mock_ws = AsyncMock()
        mock_ws.recv = mock_recv

        cfg = ChainConfig(chain=Chain.BASE)
        watcher = DexTradeWatcher(cfg)
        watcher._ws = mock_ws
        watcher._running = True

        txs = []
        async for tx in watcher.watch():
            txs.append(tx)
            break

        assert len(txs) == 1
        assert txs[0].tx_hash == "0xswap789"

    @pytest.mark.asyncio
    async def test_watch_no_params(self):
        msg = json.dumps({"id": 1, "result": "something"})

        call_count = 0

        async def mock_recv():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return msg
            raise Exception("stop")

        mock_ws = AsyncMock()
        mock_ws.recv = mock_recv

        cfg = ChainConfig(chain=Chain.BASE)
        watcher = DexTradeWatcher(cfg)
        watcher._ws = mock_ws
        watcher._running = True

        txs = []
        async for tx in watcher.watch():
            txs.append(tx)
        assert txs == []

    @pytest.mark.asyncio
    async def test_watch_exception_while_running(self):
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=Exception("connection lost"))

        cfg = ChainConfig(chain=Chain.BASE)
        watcher = DexTradeWatcher(cfg)
        watcher._ws = mock_ws
        watcher._running = True

        txs = []
        async for tx in watcher.watch():
            txs.append(tx)
        assert txs == []

    @pytest.mark.asyncio
    async def test_watch_exception_not_running(self):
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=Exception("err"))

        cfg = ChainConfig(chain=Chain.BASE)
        watcher = DexTradeWatcher(cfg)
        watcher._ws = mock_ws
        watcher._running = False

        txs = []
        async for tx in watcher.watch():
            txs.append(tx)
        assert txs == []


class TestWatcherStartStop:
    @pytest.mark.asyncio
    async def test_start_calls_connect(self):
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=json.dumps({"result": "sub"}))

        with patch("websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
            cfg = ChainConfig(chain=Chain.BASE, ws_url="wss://test.example.com")
            watcher = MempoolWatcher(cfg)
            await watcher.start()
            assert watcher.is_running is True

    @pytest.mark.asyncio
    async def test_stop_calls_disconnect(self):
        cfg = ChainConfig(chain=Chain.BASE)
        watcher = MempoolWatcher(cfg)
        watcher._running = True
        await watcher.stop()
        assert watcher.is_running is False
