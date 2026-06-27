"""Tests for Animus tool definitions."""

from unittest.mock import MagicMock, patch

from animus_faucet.config import FaucetConfig
from animus_faucet.models import Chain, ClaimResult, ClaimStatus
from animus_faucet.tools.faucet_tools import (
    FAUCET_TOOLS,
    _ensure_initialized,
    faucet_add_wallet,
    faucet_claim,
    faucet_earnings,
    faucet_list_wallets,
    faucet_status,
)


def _reset_globals():
    """Reset module-level singletons between tests."""
    import animus_faucet.tools.faucet_tools as ft

    ft._config = None
    ft._scheduler = None
    ft._wallet_manager = None


class TestToolDefinitions:
    def test_all_tools_have_required_fields(self):
        for tool in FAUCET_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            assert "handler" in tool
            assert "category" in tool
            assert tool["category"] == "faucet"

    def test_tool_count(self):
        assert len(FAUCET_TOOLS) == 5

    def test_tool_names(self):
        names = {t["name"] for t in FAUCET_TOOLS}
        assert "faucet_claim" in names
        assert "faucet_status" in names
        assert "faucet_add_wallet" in names
        assert "faucet_list_wallets" in names
        assert "faucet_earnings" in names


class TestEnsureInitialized:
    def test_lazy_init(self, tmp_data_dir):
        _reset_globals()
        import animus_faucet.tools.faucet_tools as ft

        config = FaucetConfig(data_dir=tmp_data_dir)
        config.ensure_dirs()

        ft._config = config
        from animus_faucet.scheduler import FaucetScheduler
        from animus_faucet.wallet.manager import WalletManager

        ft._wallet_manager = WalletManager(tmp_data_dir)
        ft._scheduler = FaucetScheduler(config, ft._wallet_manager)

        sched, cfg, wm = _ensure_initialized()
        assert sched is not None
        assert cfg is not None
        assert wm is not None

    def test_default_init(self, tmp_path):
        _reset_globals()

        with patch.object(FaucetConfig, "ensure_dirs"):
            with patch("animus_faucet.tools.faucet_tools.FaucetConfig") as mock_cls:
                mock_config = MagicMock()
                mock_config.data_dir = tmp_path
                mock_config.faucets_file = tmp_path / "faucets.yaml"
                mock_config.faucets = []
                mock_cls.return_value = mock_config

                sched, cfg, wm = _ensure_initialized()
                assert sched is not None


class TestFaucetAddWallet:
    def setup_method(self):
        _reset_globals()

    def _setup_tools(self, tmp_data_dir):
        import animus_faucet.tools.faucet_tools as ft
        from animus_faucet.scheduler import FaucetScheduler
        from animus_faucet.wallet.manager import WalletManager

        config = FaucetConfig(data_dir=tmp_data_dir, faucets=[])
        config.ensure_dirs()
        ft._config = config
        ft._wallet_manager = WalletManager(tmp_data_dir)
        ft._scheduler = FaucetScheduler(config, ft._wallet_manager)

    def test_add_wallet_success(self, tmp_data_dir):
        self._setup_tools(tmp_data_dir)
        result = faucet_add_wallet({"chain": "eth", "address": "0xabc"})
        assert result["success"] is True
        assert result["chain"] == "eth"
        assert result["address"] == "0xabc"

    def test_add_wallet_missing_params(self, tmp_data_dir):
        self._setup_tools(tmp_data_dir)
        result = faucet_add_wallet({"chain": "eth"})
        assert result["success"] is False
        assert "required" in result["error"]

    def test_add_wallet_invalid_chain(self, tmp_data_dir):
        self._setup_tools(tmp_data_dir)
        result = faucet_add_wallet({"chain": "invalid_chain", "address": "0x"})
        assert result["success"] is False
        assert "Invalid chain" in result["error"]

    def test_add_wallet_missing_both(self, tmp_data_dir):
        self._setup_tools(tmp_data_dir)
        result = faucet_add_wallet({})
        assert result["success"] is False


class TestFaucetListWallets:
    def setup_method(self):
        _reset_globals()

    def test_list_wallets(self, tmp_data_dir):
        import animus_faucet.tools.faucet_tools as ft
        from animus_faucet.scheduler import FaucetScheduler
        from animus_faucet.wallet.manager import WalletManager

        config = FaucetConfig(data_dir=tmp_data_dir, faucets=[])
        config.ensure_dirs()
        ft._config = config
        ft._wallet_manager = WalletManager(tmp_data_dir)
        ft._wallet_manager.set_address(Chain.ETH, "0xtest")
        ft._scheduler = FaucetScheduler(config, ft._wallet_manager)

        result = faucet_list_wallets({})
        assert result["success"] is True
        assert result["wallets"]["eth"] == "0xtest"


class TestFaucetStatus:
    def setup_method(self):
        _reset_globals()

    def test_status(self, tmp_data_dir):
        import animus_faucet.tools.faucet_tools as ft
        from animus_faucet.scheduler import FaucetScheduler
        from animus_faucet.wallet.manager import WalletManager

        config = FaucetConfig(data_dir=tmp_data_dir, faucets=[])
        config.ensure_dirs()
        ft._config = config
        ft._wallet_manager = WalletManager(tmp_data_dir)
        ft._scheduler = FaucetScheduler(config, ft._wallet_manager)

        result = faucet_status({})
        assert result["success"] is True
        assert "summary" in result


class TestFaucetClaim:
    def setup_method(self):
        _reset_globals()

    def test_claim_specific_not_found(self, tmp_data_dir):
        import animus_faucet.tools.faucet_tools as ft
        from animus_faucet.scheduler import FaucetScheduler
        from animus_faucet.wallet.manager import WalletManager

        config = FaucetConfig(data_dir=tmp_data_dir, faucets=[])
        config.ensure_dirs()
        ft._config = config
        ft._wallet_manager = WalletManager(tmp_data_dir)
        ft._scheduler = FaucetScheduler(config, ft._wallet_manager)

        result = faucet_claim({"faucet_name": "nonexistent"})
        assert result["success"] is False
        assert "not found" in result["error"]


class TestFaucetEarnings:
    def setup_method(self):
        _reset_globals()

    def test_earnings_empty(self, tmp_data_dir):
        import animus_faucet.tools.faucet_tools as ft
        from animus_faucet.scheduler import FaucetScheduler
        from animus_faucet.wallet.manager import WalletManager

        config = FaucetConfig(data_dir=tmp_data_dir, faucets=[])
        config.ensure_dirs()
        ft._config = config
        ft._wallet_manager = WalletManager(tmp_data_dir)
        ft._scheduler = FaucetScheduler(config, ft._wallet_manager)

        result = faucet_earnings({})
        assert result["success"] is True
        assert result["total_earned_usd"] == 0
        assert result["total_claims"] == 0

    def test_earnings_with_data(self, tmp_data_dir):
        import animus_faucet.tools.faucet_tools as ft
        from animus_faucet.scheduler import FaucetScheduler
        from animus_faucet.store import ClaimStore
        from animus_faucet.wallet.manager import WalletManager

        config = FaucetConfig(data_dir=tmp_data_dir, faucets=[])
        config.ensure_dirs()
        ft._config = config
        ft._wallet_manager = WalletManager(tmp_data_dir)
        ft._scheduler = FaucetScheduler(config, ft._wallet_manager)

        store = ClaimStore(tmp_data_dir)
        store.record(
            ClaimResult(
                faucet_name="test",
                chain=Chain.BTC,
                status=ClaimStatus.SUCCESS,
                amount_usd=0.05,
            )
        )
        store.record(
            ClaimResult(
                faucet_name="test",
                chain=Chain.BTC,
                status=ClaimStatus.RATE_LIMITED,
            )
        )

        result = faucet_earnings({"days": 30})
        assert result["success"] is True
        assert result["total_claims"] == 2
        assert result["successful_claims"] == 1
        assert result["total_earned_usd"] == 0.05
        assert result["by_faucet"]["test"] == 0.05
