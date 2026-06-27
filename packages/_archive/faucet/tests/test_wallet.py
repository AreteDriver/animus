"""Tests for wallet manager and balance tracker."""

from animus_faucet.models import Chain, TreasuryBalance
from animus_faucet.wallet.manager import BalanceTracker, WalletManager


class TestWalletManager:
    def test_set_and_get(self, tmp_data_dir):
        wm = WalletManager(tmp_data_dir)
        wm.set_address(Chain.ETH, "0xabc123")
        assert wm.get_address(Chain.ETH) == "0xabc123"

    def test_get_nonexistent(self, tmp_data_dir):
        wm = WalletManager(tmp_data_dir)
        assert wm.get_address(Chain.BTC) is None

    def test_has_address(self, tmp_data_dir):
        wm = WalletManager(tmp_data_dir)
        assert wm.has_address(Chain.ETH) is False
        wm.set_address(Chain.ETH, "0xabc")
        assert wm.has_address(Chain.ETH) is True

    def test_list_wallets(self, tmp_data_dir):
        wm = WalletManager(tmp_data_dir)
        wm.set_address(Chain.ETH, "0xabc")
        wm.set_address(Chain.BTC, "bc1xyz")
        wallets = wm.list_wallets()
        assert len(wallets) == 2
        assert wallets["eth"] == "0xabc"
        assert wallets["btc"] == "bc1xyz"

    def test_remove_address(self, tmp_data_dir):
        wm = WalletManager(tmp_data_dir)
        wm.set_address(Chain.ETH, "0xabc")
        assert wm.remove_address(Chain.ETH) is True
        assert wm.get_address(Chain.ETH) is None

    def test_remove_nonexistent(self, tmp_data_dir):
        wm = WalletManager(tmp_data_dir)
        assert wm.remove_address(Chain.BTC) is False

    def test_persistence(self, tmp_data_dir):
        wm1 = WalletManager(tmp_data_dir)
        wm1.set_address(Chain.SUI_TESTNET, "0xsui123")

        wm2 = WalletManager(tmp_data_dir)
        assert wm2.get_address(Chain.SUI_TESTNET) == "0xsui123"

    def test_overwrite_address(self, tmp_data_dir):
        wm = WalletManager(tmp_data_dir)
        wm.set_address(Chain.ETH, "0xold")
        wm.set_address(Chain.ETH, "0xnew")
        assert wm.get_address(Chain.ETH) == "0xnew"


class TestBalanceTracker:
    def test_update_and_get(self, tmp_data_dir):
        bt = BalanceTracker(tmp_data_dir)
        balance = TreasuryBalance(chain=Chain.ETH, address="0xabc", balance=1.5, balance_usd=3000.0)
        bt.update_balance(balance)
        retrieved = bt.get_balance(Chain.ETH)
        assert retrieved is not None
        assert retrieved.balance == 1.5
        assert retrieved.balance_usd == 3000.0

    def test_get_nonexistent(self, tmp_data_dir):
        bt = BalanceTracker(tmp_data_dir)
        assert bt.get_balance(Chain.BTC) is None

    def test_get_all_balances(self, tmp_data_dir):
        bt = BalanceTracker(tmp_data_dir)
        bt.update_balance(
            TreasuryBalance(chain=Chain.ETH, address="0x1", balance=1.0, balance_usd=2000.0)
        )
        bt.update_balance(
            TreasuryBalance(chain=Chain.BTC, address="bc1", balance=0.01, balance_usd=600.0)
        )
        all_b = bt.get_all_balances()
        assert len(all_b) == 2

    def test_total_usd(self, tmp_data_dir):
        bt = BalanceTracker(tmp_data_dir)
        bt.update_balance(
            TreasuryBalance(chain=Chain.ETH, address="0x1", balance=1.0, balance_usd=2000.0)
        )
        bt.update_balance(
            TreasuryBalance(chain=Chain.BTC, address="bc1", balance=0.01, balance_usd=600.0)
        )
        assert bt.total_usd() == 2600.0

    def test_total_usd_with_none(self, tmp_data_dir):
        bt = BalanceTracker(tmp_data_dir)
        bt.update_balance(
            TreasuryBalance(chain=Chain.ETH, address="0x1", balance=1.0, balance_usd=None)
        )
        assert bt.total_usd() == 0.0

    def test_persistence(self, tmp_data_dir):
        bt1 = BalanceTracker(tmp_data_dir)
        bt1.update_balance(
            TreasuryBalance(chain=Chain.ETH, address="0x1", balance=5.0, balance_usd=10000.0)
        )

        bt2 = BalanceTracker(tmp_data_dir)
        assert bt2.get_balance(Chain.ETH).balance == 5.0

    def test_update_overwrites(self, tmp_data_dir):
        bt = BalanceTracker(tmp_data_dir)
        bt.update_balance(
            TreasuryBalance(chain=Chain.ETH, address="0x1", balance=1.0, balance_usd=2000.0)
        )
        bt.update_balance(
            TreasuryBalance(chain=Chain.ETH, address="0x1", balance=2.0, balance_usd=4000.0)
        )
        assert bt.get_balance(Chain.ETH).balance == 2.0
