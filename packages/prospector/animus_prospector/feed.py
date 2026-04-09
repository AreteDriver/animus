"""Unified opportunity feed — the main interface for consumers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from animus_prospector.models import (
    Action,
    EvaluationResult,
    Opportunity,
)
from animus_prospector.store import EvaluationStore, OpportunityStore

if TYPE_CHECKING:
    pass

logger = logging.getLogger("animus_prospector.feed")


class OpportunityFeed:
    """Unified view of all opportunities with filtering and ranking.

    This is what consumers (CLI, Animus tools, faucet module) use
    to get actionable opportunities.
    """

    def __init__(self, data_dir: Path):
        self.opportunity_store = OpportunityStore(data_dir)
        self.evaluation_store = EvaluationStore(data_dir)

    def get_actionable(self) -> list[tuple[Opportunity, EvaluationResult | None]]:
        """Get opportunities recommended for action (auto-execute or add-to-faucet)."""
        results = []
        for opp in self.opportunity_store.load_all():
            eval_result = self.evaluation_store.get_for_opportunity(opp.opportunity_id)
            if eval_result and eval_result.recommended_action in (
                Action.AUTO_EXECUTE,
                Action.ADD_TO_FAUCET,
            ):
                results.append((opp, eval_result))
        return sorted(results, key=lambda x: -(x[1].roi_score if x[1] else 0))

    def get_flagged(self) -> list[tuple[Opportunity, EvaluationResult | None]]:
        """Get opportunities flagged for human review."""
        results = []
        for opp in self.opportunity_store.load_all():
            eval_result = self.evaluation_store.get_for_opportunity(opp.opportunity_id)
            if eval_result and eval_result.recommended_action == Action.FLAG_FOR_REVIEW:
                results.append((opp, eval_result))
        return sorted(results, key=lambda x: -(x[1].roi_score if x[1] else 0))

    def get_new_faucets(self) -> list[Opportunity]:
        """Get opportunities that should be added to the faucet module."""
        results = []
        for opp in self.opportunity_store.load_all():
            eval_result = self.evaluation_store.get_for_opportunity(opp.opportunity_id)
            if eval_result and eval_result.recommended_action == Action.ADD_TO_FAUCET:
                results.append(opp)
        return results

    def get_summary(self) -> dict:
        """Summary stats across all opportunities."""
        all_opps = self.opportunity_store.load_all()
        all_evals = self.evaluation_store.load_all()

        by_type: dict[str, int] = {}
        by_action: dict[str, int] = {}
        total_value = 0.0

        for opp in all_opps:
            t = opp.opportunity_type.value
            by_type[t] = by_type.get(t, 0) + 1

        for ev in all_evals:
            a = ev.recommended_action.value
            by_action[a] = by_action.get(a, 0) + 1
            if ev.recommended_action in (Action.AUTO_EXECUTE, Action.ADD_TO_FAUCET):
                total_value += ev.roi_score

        return {
            "total_opportunities": len(all_opps),
            "total_evaluations": len(all_evals),
            "by_type": by_type,
            "by_action": by_action,
            "estimated_actionable_value": total_value,
        }

    def record_opportunity(self, opp: Opportunity) -> bool:
        """Record an opportunity (with dedup). Returns True if new."""
        return self.opportunity_store.record(opp)

    def record_evaluation(self, result: EvaluationResult) -> None:
        """Record an evaluation result."""
        self.evaluation_store.record(result)
