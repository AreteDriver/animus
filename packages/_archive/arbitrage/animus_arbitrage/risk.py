"""Risk management for arbitrage trading."""

from __future__ import annotations

import logging
import time

from animus_arbitrage.config import ArbitrageConfig
from animus_arbitrage.models import ArbitrageOpportunity

logger = logging.getLogger("animus_arbitrage.risk")


class RiskManager:
    """Enforces risk limits on arbitrage trades.

    Tracks: position sizes, daily P&L, trade frequency, cooldowns.
    All limits are configurable via ArbitrageConfig.risk.
    """

    def __init__(self, config: ArbitrageConfig):
        self.config = config
        self._daily_pnl: float = 0.0
        self._daily_loss: float = 0.0
        self._current_position_usd: float = 0.0
        self._trades_this_hour: int = 0
        self._active_trades: int = 0
        self._last_trade_time: float = 0.0
        self._hour_start: float = time.monotonic()
        self._day_start: float = time.monotonic()

    def check_trade(self, opportunity: ArbitrageOpportunity) -> tuple[bool, str]:
        """Check if a trade passes all risk limits.

        Returns (allowed, reason). If allowed is False, reason explains why.
        """
        self._maybe_reset_counters()

        risk = self.config.risk

        # Max position size
        new_position = self._current_position_usd + opportunity.trade_size_usd
        if new_position > risk.max_position_usd:
            return False, (f"Position limit: ${new_position:.2f} > ${risk.max_position_usd:.2f}")

        # Daily loss limit
        if self._daily_loss >= risk.daily_loss_limit_usd:
            return False, (
                f"Daily loss limit: ${self._daily_loss:.2f} >= ${risk.daily_loss_limit_usd:.2f}"
            )

        # Minimum spread
        if opportunity.spread_pct < risk.min_spread_pct:
            return False, (
                f"Spread too low: {opportunity.spread_pct:.2f}% < {risk.min_spread_pct:.2f}%"
            )

        # Minimum confidence
        if opportunity.confidence < risk.min_confidence:
            return False, (
                f"Confidence too low: {opportunity.confidence:.2f} < {risk.min_confidence:.2f}"
            )

        # Trade frequency
        if self._trades_this_hour >= risk.max_trades_per_hour:
            return False, (
                f"Hourly trade limit: {self._trades_this_hour} >= {risk.max_trades_per_hour}"
            )

        # Concurrent trades
        if self._active_trades >= risk.max_concurrent_trades:
            return False, (
                f"Concurrent limit: {self._active_trades} >= {risk.max_concurrent_trades}"
            )

        # Cooldown
        now = time.monotonic()
        elapsed = now - self._last_trade_time
        if self._last_trade_time > 0 and elapsed < risk.cooldown_seconds:
            remaining = risk.cooldown_seconds - elapsed
            return False, f"Cooldown: {remaining:.0f}s remaining"

        # Max trade size
        if opportunity.trade_size_usd > self.config.execution.max_trade_size_usd:
            return False, (
                f"Trade size: ${opportunity.trade_size_usd:.2f} "
                f"> ${self.config.execution.max_trade_size_usd:.2f}"
            )

        return True, "OK"

    def record_trade(self, opportunity: ArbitrageOpportunity) -> None:
        """Record a trade for tracking limits."""
        self._current_position_usd += opportunity.trade_size_usd
        self._trades_this_hour += 1
        self._active_trades += 1
        self._last_trade_time = time.monotonic()
        self._daily_pnl += opportunity.profit_after_gas_usd

    def record_trade_complete(self, trade_size_usd: float, actual_profit_usd: float) -> None:
        """Record trade completion to release position and update P&L."""
        self._current_position_usd = max(0.0, self._current_position_usd - trade_size_usd)
        self._active_trades = max(0, self._active_trades - 1)
        self._daily_pnl += actual_profit_usd
        if actual_profit_usd < 0:
            self._daily_loss += abs(actual_profit_usd)

    def _maybe_reset_counters(self) -> None:
        """Reset hourly/daily counters if periods have elapsed."""
        now = time.monotonic()

        # Reset hourly counter
        if now - self._hour_start >= 3600:
            self._trades_this_hour = 0
            self._hour_start = now

        # Reset daily counters
        if now - self._day_start >= 86400:
            self._daily_pnl = 0.0
            self._daily_loss = 0.0
            self._day_start = now

    @property
    def status(self) -> dict:
        """Current risk manager state."""
        self._maybe_reset_counters()
        risk = self.config.risk
        return {
            "current_position_usd": self._current_position_usd,
            "max_position_usd": risk.max_position_usd,
            "daily_pnl_usd": self._daily_pnl,
            "daily_loss_usd": self._daily_loss,
            "daily_loss_limit_usd": risk.daily_loss_limit_usd,
            "trades_this_hour": self._trades_this_hour,
            "max_trades_per_hour": risk.max_trades_per_hour,
            "active_trades": self._active_trades,
            "max_concurrent_trades": risk.max_concurrent_trades,
            "cooldown_seconds": risk.cooldown_seconds,
        }
