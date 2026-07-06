"""Forge Commissioner — commissions Forge to implement Architect proposals.

This is the bridge between the Architect Citizen (observation/proposal)
and the Factory layer (Forge execution). It preserves sovereignty by
requiring explicit human approval before any commission is issued.

The Commissioner NEVER modifies code directly. It only:
1. Receives an approved ImprovementProposal
2. Transforms it into a Forge workflow configuration
3. Submits the workflow to Forge for execution
4. Returns the evidence bundle
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from animus.citizens.proposal import ImprovementProposal, ProposalStatus
from animus.logging import get_logger

logger = get_logger("citizens.commissioner")


@dataclass
class CommissionResult:
    """Result of commissioning Forge to implement a proposal."""

    success: bool
    proposal_id: str
    stage_reached: str = ""
    evidence_bundle: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    tests_passed: bool = False
    benchmark_results: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "proposal_id": self.proposal_id,
            "stage_reached": self.stage_reached,
            "evidence_bundle": self.evidence_bundle,
            "error": self.error,
            "tests_passed": self.tests_passed,
            "benchmark_results": self.benchmark_results,
            "timestamp": self.timestamp.isoformat(),
        }


class ForgeCommissioner:
    """Commissioner for Forge implementation of Architect proposals.

    Usage:
        commissioner = ForgeCommissioner(codebase_path="~/projects/animus")
        result = commissioner.commission(proposal, auto_approve=False)

    The commissioner requires explicit human approval. It will raise
    RuntimeError if auto_approve=True without the env opt-in.
    """

    def __init__(
        self,
        codebase_path: Path | str = "~/projects/animus",
        forge_host: str = "localhost",
        forge_port: int = 7700,
    ):
        self.codebase_path = Path(codebase_path).expanduser()
        self.forge_host = forge_host
        self.forge_port = forge_port
        self._forge_available: bool | None = None

    def _check_forge(self) -> bool:
        """Check if Forge is available.

        Returns:
            True if Forge can be reached.
        """
        if self._forge_available is not None:
            return self._forge_available

        try:
            import httpx

            response = httpx.get(
                f"http://{self.forge_host}:{self.forge_port}/health",
                timeout=5.0,
            )
            self._forge_available = response.status_code == 200
            return self._forge_available
        except Exception:
            self._forge_available = False
            return False

    def _create_workflow_config(self, proposal: ImprovementProposal) -> dict[str, Any]:
        """Transform an ImprovementProposal into a Forge workflow configuration.

        Args:
            proposal: Approved improvement proposal.

        Returns:
            Workflow configuration dictionary.
        """
        return {
            "name": f"architect-{proposal.id}",
            "description": proposal.title,
            "agents": [
                {
                    "name": "analyzer",
                    "system_prompt": (
                        "You are the implementation agent for an Architect proposal. "
                        f"Problem: {proposal.problem}\n"
                        f"Recommendation: {proposal.recommendation}\n"
                        f"Affected components: {', '.join(proposal.affected_components)}"
                    ),
                    "model": "ollama",
                    "temperature": 0.2,
                    "max_tokens": 4000,
                },
                {
                    "name": "implementer",
                    "system_prompt": (
                        "You implement the approved changes. "
                        "Follow the recommendation exactly. "
                        "Write tests for all changes. "
                        "Do not modify files outside the affected components."
                    ),
                    "model": "ollama",
                    "temperature": 0.1,
                    "max_tokens": 8000,
                },
                {
                    "name": "evaluator",
                    "system_prompt": (
                        "You verify the implementation. "
                        f"Success criteria: {', '.join(proposal.success_metrics)}\n"
                        "Run tests. Check benchmarks. Report pass/fail."
                    ),
                    "model": "ollama",
                    "temperature": 0.0,
                    "max_tokens": 2000,
                },
            ],
            "budget": {
                "total_tokens": 200_000,
                "max_cost_usd": 0.0,  # Local-only
            },
            "gates": [
                {
                    "name": "test_gate",
                    "condition": "evaluator.output.contains('PASS')",
                    "on_fail": "halt",
                }
            ],
        }

    def commission(
        self,
        proposal: ImprovementProposal,
        auto_approve: bool = False,
    ) -> CommissionResult:
        """Commission Forge to implement an approved proposal.

        Args:
            proposal: An ImprovementProposal with status APPROVED.
            auto_approve: If True, bypass approval checks. **Test-only.**

        Returns:
            CommissionResult with evidence bundle.

        Raises:
            RuntimeError: If proposal is not approved or auto_approve
                is True without env opt-in.
        """
        if auto_approve and __import__("os").environ.get("ANIMUS_FORGE_ALLOW_AUTO_APPROVE") != "1":
            raise RuntimeError(
                "auto_approve=True is blocked. "
                "Set ANIMUS_FORGE_ALLOW_AUTO_APPROVE=1 only in test environments. "
                "Human approval is mandatory for all commissions."
            )

        if proposal.status != ProposalStatus.APPROVED and not auto_approve:
            return CommissionResult(
                success=False,
                proposal_id=proposal.id,
                error=f"Proposal status is {proposal.status.value}, not approved",
            )

        if not self._check_forge():
            logger.warning("Forge not available — returning simulated commission result")
            return self._simulate_commission(proposal)

        try:
            return self._execute_commission(proposal)
        except Exception as e:
            logger.error(f"Commission failed: {e}")
            return CommissionResult(
                success=False,
                proposal_id=proposal.id,
                error=str(e),
            )

    def _execute_commission(self, proposal: ImprovementProposal) -> CommissionResult:
        """Execute commission via Forge API.

        Args:
            proposal: Approved proposal.

        Returns:
            Commission result.
        """
        import httpx

        workflow = self._create_workflow_config(proposal)

        # Submit workflow to Forge
        try:
            response = httpx.post(
                f"http://{self.forge_host}:{self.forge_port}/workflows/execute",
                json=workflow,
                timeout=300.0,
            )
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPStatusError as e:
            return CommissionResult(
                success=False,
                proposal_id=proposal.id,
                error=f"Forge HTTP error: {e.response.status_code} — {e.response.text}",
            )
        except Exception as e:
            return CommissionResult(
                success=False,
                proposal_id=proposal.id,
                error=f"Forge communication failed: {e}",
            )

        # Parse result
        evidence = {
            "workflow_name": workflow["name"],
            "forge_response": result,
            "affected_components": proposal.affected_components,
        }

        tests_passed = result.get("status") == "complete" and not result.get("errors")
        stage = result.get("status", "unknown")

        return CommissionResult(
            success=tests_passed,
            proposal_id=proposal.id,
            stage_reached=stage,
            evidence_bundle=evidence,
            tests_passed=tests_passed,
            benchmark_results=result.get("metrics", {}),
        )

    def _simulate_commission(self, proposal: ImprovementProposal) -> CommissionResult:
        """Simulate a commission when Forge is unavailable.

        This is used for development and testing when Forge
        is not running. It produces a realistic result structure
        without executing any code.
        """
        logger.info(f"Simulating commission for proposal {proposal.id}")

        evidence = {
            "simulated": True,
            "workflow_name": f"architect-{proposal.id}",
            "agents": ["analyzer", "implementer", "evaluator"],
            "affected_components": proposal.affected_components,
            "recommendation": proposal.recommendation,
        }

        return CommissionResult(
            success=False,  # Simulated commissions are never "real" success
            proposal_id=proposal.id,
            stage_reached="simulated",
            evidence_bundle=evidence,
            tests_passed=False,
            error="Forge unavailable — commission was simulated, not executed",
        )

    def update_proposal_with_evidence(
        self,
        proposal: ImprovementProposal,
        result: CommissionResult,
    ) -> ImprovementProposal:
        """Update a proposal with commission evidence.

        Args:
            proposal: Original proposal.
            result: Commission result.

        Returns:
            Updated proposal with evidence bundle attached.
        """
        proposal.evidence_bundle = result.to_dict()

        if result.success:
            proposal.update_status(ProposalStatus.COMPLETE, actor="forge")
        else:
            proposal.update_status(ProposalStatus.EVALUATING, actor="forge")

        return proposal
