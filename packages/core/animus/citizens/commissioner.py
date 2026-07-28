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
        forge_port: int = 8000,
        use_local_engine: bool = False,
        cognitive: Any = None,
    ):
        self.codebase_path = Path(codebase_path).expanduser()
        self.forge_host = forge_host
        self.forge_port = forge_port
        self.use_local_engine = use_local_engine
        self._cognitive = cognitive
        self._forge_available: bool | None = None
        self._cached_token: str | None = None
        self._local_engine: Any | None = None

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
            "id": f"architect-{proposal.id}",
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

        if self.use_local_engine:
            logger.info("Using local ForgeEngine for commission (bypassing HTTP)")
            return self._execute_local(proposal)

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

    def _get_local_engine(self) -> Any:
        if self._local_engine is not None:
            return self._local_engine
        try:
            from animus_forge.engine import ForgeEngine  # boundary-ok: citizen degrades gracefully without Forge
            from animus_forge.state import AppState  # boundary-ok: citizen degrades gracefully without Forge

            state = AppState()
            if self._cognitive:
                state.cognitive = self._cognitive
            self._local_engine = ForgeEngine(state=state)
            return self._local_engine
        except Exception:
            logger.exception("Failed to initialise local ForgeEngine")
            self._local_engine = None
            return None

    def _execute_local(self, proposal: ImprovementProposal) -> CommissionResult:
        """Execute a proposal using the local ForgeEngine (bypasses HTTP)."""
        engine = self._get_local_engine()
        if not engine:
            return CommissionResult(
                success=False,
                proposal_id=proposal.id,
                error="Local ForgeEngine not available",
            )

        workflow = self._create_workflow_config(proposal)
        try:
            result = engine.run(workflow)
            success = result.get("status") in ("complete", "success")
            evidence = {
                "workflow_name": workflow["name"],
                "workflow_id": workflow["id"],
                "forge_response": result,
                "affected_components": proposal.affected_components,
            }
            return CommissionResult(
                success=success,
                proposal_id=proposal.id,
                stage_reached=result.get("status", "unknown"),
                evidence_bundle=evidence,
                tests_passed=success,
                benchmark_results=result.get("metrics", {}),
            )
        except Exception as e:
            logger.exception("Local ForgeEngine execution failed")
            return CommissionResult(
                success=False,
                proposal_id=proposal.id,
                error=f"Local execution failed: {e}",
            )

    def _auth_header(self) -> dict[str, str]:
        """Build Authorization header.

        Priority:
        1. Cached token from previous login
        2. ANIMUS_FORGE_API_TOKEN env var
        3. Login with FORGE_API_USER + FORGE_API_PASS
        """
        if self._cached_token:
            return {"Authorization": f"Bearer {self._cached_token}"}

        token = __import__("os").environ.get("ANIMUS_FORGE_API_TOKEN", "")
        if token:
            self._cached_token = token
            return {"Authorization": f"Bearer {token}"}

        # Attempt login with credentials
        cred_token = self._login_with_credentials()
        if cred_token:
            self._cached_token = cred_token
            return {"Authorization": f"Bearer {cred_token}"}

        return {}

    def _login_with_credentials(self) -> str | None:
        """Login to Forge using FORGE_API_USER + FORGE_API_PASS env vars.

        Returns:
            Access token string, or None if credentials not configured.
        """
        import os

        user = os.environ.get("FORGE_API_USER", "")
        pw = os.environ.get("FORGE_API_PASS", "")
        if not user or not pw:
            return None

        try:
            import httpx

            resp = httpx.post(
                f"http://{self.forge_host}:{self.forge_port}/v1/auth/login",
                json={"user_id": user, "password": pw},
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            token = data.get("access_token")
            if token:
                logger.info(f"Forge login succeeded for user '{user}'")
                return token
        except Exception as e:
            logger.warning(f"Forge login failed for user '{user}': {e}")
        return None

    def _execute_commission(self, proposal: ImprovementProposal) -> CommissionResult:
        """Execute commission via Forge API.

        Two-step process:
        1. POST workflow config to /workflows → receive workflow_id
        2. POST {workflow_id} to /workflows/execute → receive execution result

        Args:
            proposal: Approved proposal.

        Returns:
            Commission result.
        """
        import httpx

        workflow = self._create_workflow_config(proposal)
        headers = self._auth_header()
        base_url = f"http://{self.forge_host}:{self.forge_port}/v1"

        # Step 1: Register workflow
        try:
            register_resp = httpx.post(
                f"{base_url}/workflows",
                json=workflow,
                headers=headers,
                timeout=30.0,
            )
            register_resp.raise_for_status()
            register_data = register_resp.json()
            workflow_id = register_data.get("workflow_id") or register_data.get("id")
            if not workflow_id:
                return CommissionResult(
                    success=False,
                    proposal_id=proposal.id,
                    error=f"Forge did not return workflow_id. Response: {register_data}",
                )
        except httpx.HTTPStatusError as e:
            return CommissionResult(
                success=False,
                proposal_id=proposal.id,
                error=f"Forge register error: {e.response.status_code} — {e.response.text}",
            )
        except Exception as e:
            return CommissionResult(
                success=False,
                proposal_id=proposal.id,
                error=f"Forge register failed: {e}",
            )

        # Step 2: Execute workflow by ID
        try:
            exec_resp = httpx.post(
                f"{base_url}/workflows/execute",
                json={"workflow_id": workflow_id},
                headers=headers,
                timeout=300.0,
            )
            exec_resp.raise_for_status()
            result = exec_resp.json()
        except httpx.HTTPStatusError as e:
            return CommissionResult(
                success=False,
                proposal_id=proposal.id,
                error=f"Forge execute error: {e.response.status_code} — {e.response.text}",
            )
        except Exception as e:
            return CommissionResult(
                success=False,
                proposal_id=proposal.id,
                error=f"Forge execute failed: {e}",
            )

        # Parse result
        evidence = {
            "workflow_name": workflow["name"],
            "workflow_id": workflow_id,
            "forge_response": result,
            "affected_components": proposal.affected_components,
        }

        tests_passed = result.get("status") in ("complete", "success") and not result.get("errors")
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
