"""Persistent JSONL storage for arbitrage data."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from animus_arbitrage.models import ArbitrageOpportunity, TradeResult

logger = logging.getLogger("animus_arbitrage.store")


class ArbitrageStore:
    """Append-only JSONL store for opportunities and trade results."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._opportunities_file = data_dir / "opportunities.jsonl"
        self._results_file = data_dir / "results.jsonl"
        self._seen_ids: set[str] = set()
        self._ensure_dir()
        self._load_seen_ids()

    def _ensure_dir(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _load_seen_ids(self) -> None:
        """Load existing opportunity IDs for dedup."""
        if not self._opportunities_file.exists():
            return
        with open(self._opportunities_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        self._seen_ids.add(data.get("opportunity_id", ""))
                    except json.JSONDecodeError:
                        continue

    def record_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """Record an opportunity. Returns True if new, False if duplicate."""
        if opportunity.opportunity_id in self._seen_ids:
            return False
        with open(self._opportunities_file, "a") as f:
            f.write(json.dumps(opportunity.to_dict()) + "\n")
        self._seen_ids.add(opportunity.opportunity_id)
        return True

    def record_opportunities(self, opportunities: list[ArbitrageOpportunity]) -> int:
        """Record multiple opportunities. Returns count of new ones."""
        count = 0
        for opp in opportunities:
            if self.record_opportunity(opp):
                count += 1
        return count

    def load_opportunities(self) -> list[ArbitrageOpportunity]:
        """Load all recorded opportunities (deserialized as dicts)."""
        if not self._opportunities_file.exists():
            return []
        results: list[ArbitrageOpportunity] = []
        with open(self._opportunities_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        results.append(_opportunity_from_dict(data))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        return results

    def opportunity_count(self) -> int:
        """Total number of recorded opportunities."""
        return len(self._seen_ids)

    def record_result(self, result: TradeResult) -> None:
        """Record a trade result."""
        with open(self._results_file, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

    def load_results(self) -> list[TradeResult]:
        """Load all trade results."""
        if not self._results_file.exists():
            return []
        results: list[TradeResult] = []
        with open(self._results_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        results.append(_result_from_dict(data))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        return results

    def result_count(self) -> int:
        """Total number of trade results."""
        return len(self.load_results())


def _opportunity_from_dict(data: dict) -> ArbitrageOpportunity:
    """Reconstruct an ArbitrageOpportunity from its serialized dict."""
    from datetime import datetime

    from animus_arbitrage.models import DEX, Chain, PriceQuote, TokenPair

    # Parse the pair info from the stored format
    pair_str = data.get("pair", "?/?")
    parts = pair_str.split("/")
    base_token = parts[0] if len(parts) > 0 else "?"
    quote_token = parts[1] if len(parts) > 1 else "?"

    buy_chain = Chain(data.get("buy_chain", "eth"))
    sell_chain = Chain(data.get("sell_chain", "eth"))

    buy_pair = TokenPair(base_token=base_token, quote_token=quote_token, chain=buy_chain)
    sell_pair = TokenPair(base_token=base_token, quote_token=quote_token, chain=sell_chain)

    buy_quote = PriceQuote(
        dex=DEX(data["buy_dex"]),
        pair=buy_pair,
        price=data["buy_price"],
        liquidity_usd=0.0,
        gas_estimate_usd=0.0,
    )
    sell_quote = PriceQuote(
        dex=DEX(data["sell_dex"]),
        pair=sell_pair,
        price=data["sell_price"],
        liquidity_usd=0.0,
        gas_estimate_usd=0.0,
    )

    detected_at = datetime.fromisoformat(data["detected_at"]) if "detected_at" in data else None
    expires_at = datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None

    kwargs: dict = {
        "opportunity_id": data["opportunity_id"],
        "buy_quote": buy_quote,
        "sell_quote": sell_quote,
        "spread_pct": data["spread_pct"],
        "profit_after_gas_usd": data["profit_after_gas_usd"],
        "confidence": data["confidence"],
        "trade_size_usd": data.get("trade_size_usd", 0.0),
    }
    if detected_at is not None:
        kwargs["detected_at"] = detected_at
    if expires_at is not None:
        kwargs["expires_at"] = expires_at

    return ArbitrageOpportunity(**kwargs)


def _result_from_dict(data: dict) -> TradeResult:
    """Reconstruct a TradeResult from its serialized dict."""
    from datetime import datetime

    from animus_arbitrage.models import ExecutionMode

    kwargs: dict = {
        "result_id": data.get("result_id", ""),
        "opportunity_id": data["opportunity_id"],
        "executed": data["executed"],
        "mode": ExecutionMode(data.get("mode", "dry_run")),
        "buy_tx": data.get("buy_tx"),
        "sell_tx": data.get("sell_tx"),
        "actual_profit_usd": data.get("actual_profit_usd", 0.0),
        "slippage_pct": data.get("slippage_pct", 0.0),
        "error": data.get("error"),
    }
    if "timestamp" in data:
        kwargs["timestamp"] = datetime.fromisoformat(data["timestamp"])

    return TradeResult(**kwargs)
