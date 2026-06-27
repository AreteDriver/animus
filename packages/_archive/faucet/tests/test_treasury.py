"""Tests for treasury management."""

from unittest.mock import AsyncMock

import pytest

from animus_faucet.config import ConsolidationConfig, FaucetConfig
from animus_faucet.models import Chain, TreasuryBalance
from animus_faucet.treasury import ChainAdapter, Treasury
from animus_faucet.wallet.manager import WalletManager


@pytest.fixture
def wallet_manager(tmp_data_dir):
    wm = WalletManager(tmp_data_dir)
    wm.set_address(Chain.ETH, "0xhot_wallet")
    wm.set_address(Chain.BTC, "bc1hot")
    return wm


@pytest.fixture
def config(tmp_data_dir):
    return FaucetConfig(
        data_dir=tmp_data_dir,
        consolidation=ConsolidationConfig(
            enabled=True,
            min_balance_usd=1.0,
            cold_wallet_addresses={"eth": "0xcold_wallet"},
        ),
    )


@pytest.fixture
def treasury(config, wallet_manager):
    return Treasury(config, wallet_manager)


class TestChainAdapter:
    @pytest.mark.asyncio
    async def test_abstract_methods(self):
        adapter = ChainAdapter()
        with pytest.raises(NotImplementedError):
            await adapter.get_balance("0x")
        with pytest.raises(NotImplementedError):
            await adapter.to_usd(1.0)
        with pytest.raises(NotImplementedError):
            await adapter.transfer("0x1", "0x2", 1.0)

    @pytest.mark.asyncio
    async def test_gas_price_default(self):
        adapter = ChainAdapter()
        assert await adapter.get_gas_price() is None


class TestTreasury:
    @pytest.mark.asyncio
    async def test_refresh_balances(self, treasury):
        mock_adapter = AsyncMock()
        mock_adapter.get_balance = AsyncMock(return_value=1.5)
        mock_adapter.to_usd = AsyncMock(return_value=3000.0)
        treasury.register_adapter(Chain.ETH, mock_adapter)

        balances = await treasury.refresh_balances()
        assert len(balances) == 1
        assert balances[0].chain == Chain.ETH
        assert balances[0].balance == 1.5
        assert balances[0].balance_usd == 3000.0

    @pytest.mark.asyncio
    async def test_refresh_balances_no_adapter(self, treasury):
        # BTC has no adapter registered
        balances = await treasury.refresh_balances()
        assert len(balances) == 0

    @pytest.mark.asyncio
    async def test_refresh_balances_error(self, treasury):
        mock_adapter = AsyncMock()
        mock_adapter.get_balance = AsyncMock(side_effect=Exception("RPC down"))
        treasury.register_adapter(Chain.ETH, mock_adapter)

        balances = await treasury.refresh_balances()
        assert len(balances) == 0  # error handled gracefully

    @pytest.mark.asyncio
    async def test_should_consolidate_above_threshold(self, treasury):
        treasury.balance_tracker.update_balance(
            TreasuryBalance(chain=Chain.ETH, address="0xhot", balance=0.1, balance_usd=200.0)
        )
        assert await treasury.should_consolidate(Chain.ETH) is True

    @pytest.mark.asyncio
    async def test_should_consolidate_below_threshold(self, treasury):
        treasury.balance_tracker.update_balance(
            TreasuryBalance(chain=Chain.ETH, address="0xhot", balance=0.0001, balance_usd=0.2)
        )
        assert await treasury.should_consolidate(Chain.ETH) is False

    @pytest.mark.asyncio
    async def test_should_consolidate_no_balance(self, treasury):
        assert await treasury.should_consolidate(Chain.SOL) is False

    @pytest.mark.asyncio
    async def test_should_consolidate_disabled(self, config, wallet_manager):
        config.consolidation.enabled = False
        t = Treasury(config, wallet_manager)
        t.balance_tracker.update_balance(
            TreasuryBalance(chain=Chain.ETH, address="0xhot", balance=100, balance_usd=200000)
        )
        assert await t.should_consolidate(Chain.ETH) is False

    @pytest.mark.asyncio
    async def test_should_consolidate_none_usd(self, treasury):
        treasury.balance_tracker.update_balance(
            TreasuryBalance(chain=Chain.ETH, address="0xhot", balance=1.0, balance_usd=None)
        )
        assert await treasury.should_consolidate(Chain.ETH) is False

    @pytest.mark.asyncio
    async def test_consolidate_success(self, treasury):
        mock_adapter = AsyncMock()
        mock_adapter.get_balance = AsyncMock(return_value=0.5)
        mock_adapter.transfer = AsyncMock(return_value="0xtxhash")
        treasury.register_adapter(Chain.ETH, mock_adapter)

        tx = await treasury.consolidate(Chain.ETH)
        assert tx == "0xtxhash"
        mock_adapter.transfer.assert_called_once_with("0xhot_wallet", "0xcold_wallet", 0.5)

    @pytest.mark.asyncio
    async def test_consolidate_no_cold_wallet(self, treasury):
        mock_adapter = AsyncMock()
        treasury.register_adapter(Chain.BTC, mock_adapter)
        tx = await treasury.consolidate(Chain.BTC)
        assert tx is None

    @pytest.mark.asyncio
    async def test_consolidate_no_adapter(self, treasury):
        tx = await treasury.consolidate(Chain.SOL)
        assert tx is None

    @pytest.mark.asyncio
    async def test_consolidate_no_hot_wallet(self, config, tmp_data_dir):
        wm = WalletManager(tmp_data_dir)  # no wallets
        t = Treasury(config, wm)
        mock_adapter = AsyncMock()
        t.register_adapter(Chain.ETH, mock_adapter)
        tx = await t.consolidate(Chain.ETH)
        assert tx is None

    @pytest.mark.asyncio
    async def test_consolidate_zero_balance(self, treasury):
        mock_adapter = AsyncMock()
        mock_adapter.get_balance = AsyncMock(return_value=0)
        treasury.register_adapter(Chain.ETH, mock_adapter)
        tx = await treasury.consolidate(Chain.ETH)
        assert tx is None

    @pytest.mark.asyncio
    async def test_consolidate_error(self, treasury):
        mock_adapter = AsyncMock()
        mock_adapter.get_balance = AsyncMock(return_value=1.0)
        mock_adapter.transfer = AsyncMock(side_effect=Exception("nonce too low"))
        treasury.register_adapter(Chain.ETH, mock_adapter)
        tx = await treasury.consolidate(Chain.ETH)
        assert tx is None

    def test_get_portfolio_empty(self, treasury):
        portfolio = treasury.get_portfolio()
        assert portfolio["total_usd"] == 0.0
        assert portfolio["chains"] == 0

    def test_get_portfolio_with_data(self, treasury):
        treasury.balance_tracker.update_balance(
            TreasuryBalance(chain=Chain.ETH, address="0x1", balance=1.0, balance_usd=2000.0)
        )
        treasury.balance_tracker.update_balance(
            TreasuryBalance(chain=Chain.BTC, address="bc1", balance=0.01, balance_usd=600.0)
        )
        portfolio = treasury.get_portfolio()
        assert portfolio["total_usd"] == 2600.0
        assert portfolio["chains"] == 2
        assert len(portfolio["balances"]) == 2
