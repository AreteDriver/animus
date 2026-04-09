"""LLM-powered opportunity evaluator."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from animus_prospector.models import (
    Action,
    EvaluationResult,
    Opportunity,
)

if TYPE_CHECKING:
    from animus_prospector.config import ProspectorConfig

logger = logging.getLogger("animus_prospector.evaluator")

EVALUATION_PROMPT = """You are evaluating a crypto opportunity for legitimacy and value.

Opportunity:
- Title: {title}
- URL: {url}
- Type: {opportunity_type}
- Chain: {chain}
- Source: {source_type} ({source_url})
- Tags: {tags}
- Raw data: {raw_data}

Evaluate this opportunity and respond with ONLY a JSON object (no markdown):
{{
  "scam_probability": <float 0-1, how likely this is a scam>,
  "estimated_value_usd": <float, realistic expected value>,
  "effort_minutes": <int, time required to complete>,
  "auto_actionable": <bool, can a bot do this without human input>,
  "reasoning": "<one sentence explaining your assessment>",
  "confidence": <float 0-1, how confident you are in this assessment>
}}

Key scam indicators: asking for private keys, seed phrases, upfront payments,
"double your crypto", unrealistic returns, urgency pressure, unknown domains.
Key legitimacy indicators: known platforms (Layer3, Galxe, Coinbase), established
projects, testnet interactions, verified social accounts, reasonable rewards."""


class Evaluator:
    """Evaluates opportunities using LLM analysis and heuristic rules.

    Two-stage evaluation:
    1. Fast heuristic pass (keyword scam detection, dedup)
    2. LLM deep evaluation for opportunities that pass stage 1
    """

    def __init__(self, config: ProspectorConfig):
        self.config = config

    async def evaluate(self, opportunity: Opportunity) -> EvaluationResult:
        """Evaluate a single opportunity."""
        # Stage 1: Fast heuristic check
        heuristic = self._heuristic_check(opportunity)
        if heuristic is not None:
            return heuristic

        # Stage 2: LLM evaluation
        try:
            return await self._llm_evaluate(opportunity)
        except Exception as e:
            logger.warning(f"LLM evaluation failed for {opportunity.title}: {e}")
            return self._fallback_evaluation(opportunity)

    async def evaluate_batch(self, opportunities: list[Opportunity]) -> list[EvaluationResult]:
        """Evaluate a batch of opportunities."""
        results = []
        for opp in opportunities:
            result = await self.evaluate(opp)
            results.append(result)
        return results

    def _heuristic_check(self, opportunity: Opportunity) -> EvaluationResult | None:
        """Fast scam/spam detection without LLM."""
        text = (opportunity.title + " " + opportunity.url + " " + str(opportunity.raw_data)).lower()

        # Check scam keywords
        for keyword in self.config.evaluation.scam_keywords:
            if keyword.lower() in text:
                return EvaluationResult(
                    opportunity_id=opportunity.opportunity_id,
                    roi_score=0.0,
                    scam_probability=0.9,
                    recommended_action=Action.SKIP,
                    reasoning=f"Scam keyword detected: '{keyword}'",
                    confidence=0.8,
                )

        # Check if expired
        if opportunity.is_expired:
            return EvaluationResult(
                opportunity_id=opportunity.opportunity_id,
                roi_score=0.0,
                scam_probability=0.0,
                recommended_action=Action.EXPIRED,
                reasoning="Opportunity has expired",
                confidence=1.0,
            )

        return None  # Pass to LLM

    async def _llm_evaluate(self, opportunity: Opportunity) -> EvaluationResult:
        """Use LLM to deeply evaluate an opportunity."""
        prompt = EVALUATION_PROMPT.format(
            title=opportunity.title,
            url=opportunity.url,
            opportunity_type=opportunity.opportunity_type.value,
            chain=opportunity.chain.value,
            source_type=opportunity.source_type.value,
            source_url=opportunity.source_url,
            tags=", ".join(opportunity.tags),
            raw_data=json.dumps(opportunity.raw_data)[:500],
        )

        response_text = await self._call_llm(prompt)
        return self._parse_llm_response(opportunity, response_text)

    async def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM provider."""
        import httpx

        if self.config.llm.provider == "ollama":
            async with httpx.AsyncClient(timeout=self.config.llm.timeout) as client:
                resp = await client.post(
                    f"{self.config.llm.ollama_url}/api/generate",
                    json={
                        "model": self.config.llm.model,
                        "prompt": prompt,
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                return resp.json().get("response", "")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm.provider}")

    def _parse_llm_response(self, opportunity: Opportunity, response: str) -> EvaluationResult:
        """Parse LLM JSON response into EvaluationResult."""
        try:
            # Try direct JSON parse
            data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: extract JSON from response text
            json_match = re.search(r"\{[^{}]+\}", response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    return self._fallback_evaluation(opportunity)
            else:
                return self._fallback_evaluation(opportunity)

        scam_prob = float(data.get("scam_probability", 0.5))
        value = float(data.get("estimated_value_usd", 0.0))
        effort = int(data.get("effort_minutes", 30))
        auto = bool(data.get("auto_actionable", False))
        reasoning = str(data.get("reasoning", "LLM evaluation"))
        confidence = float(data.get("confidence", 0.5))

        # Update opportunity with LLM estimates
        opportunity.estimated_value_usd = value
        opportunity.effort_minutes = effort
        opportunity.risk_score = scam_prob
        opportunity.auto_actionable = auto

        action = self._decide_action(opportunity, scam_prob)

        return EvaluationResult(
            opportunity_id=opportunity.opportunity_id,
            roi_score=opportunity.roi_score,
            scam_probability=scam_prob,
            recommended_action=action,
            reasoning=reasoning,
            confidence=confidence,
        )

    def _decide_action(self, opportunity: Opportunity, scam_prob: float) -> Action:
        """Decide what to do based on evaluation thresholds."""
        cfg = self.config.evaluation

        if scam_prob > cfg.auto_execute_max_risk:
            return Action.SKIP

        if opportunity.effort_minutes > cfg.skip_max_effort_minutes:
            return Action.SKIP

        if opportunity.estimated_value_usd < cfg.flag_min_value_usd:
            return Action.SKIP

        if (
            opportunity.auto_actionable
            and opportunity.roi_score >= cfg.auto_execute_min_roi
            and scam_prob <= cfg.auto_execute_max_risk
        ):
            from animus_prospector.models import OpportunityType

            if opportunity.opportunity_type == OpportunityType.FAUCET:
                return Action.ADD_TO_FAUCET
            return Action.AUTO_EXECUTE

        return Action.FLAG_FOR_REVIEW

    def _fallback_evaluation(self, opportunity: Opportunity) -> EvaluationResult:
        """Conservative evaluation when LLM fails."""
        return EvaluationResult(
            opportunity_id=opportunity.opportunity_id,
            roi_score=opportunity.roi_score,
            scam_probability=0.5,
            recommended_action=Action.FLAG_FOR_REVIEW,
            reasoning="LLM evaluation failed — flagging for manual review",
            confidence=0.2,
        )
