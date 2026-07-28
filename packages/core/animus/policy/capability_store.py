"""Capability grant storage.

In-memory implementation backed by optional schema validation via
``animus_contracts``. Will be replaced with PostgreSQL-backed store
in Phase 3 completion.

Kernel-native policy layer (v2.3, 2026-07-06).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from animus.logging import get_logger
from animus_types import ValidationError as _ContractValidationError

logger = get_logger("policy.capability_store")

try:
    from animus_contracts import validate as _validate_contract

    _HAS_CONTRACTS = True
except ImportError:  # pragma: no cover
    _HAS_CONTRACTS = False


@dataclass
class CapabilityGrant:
    """A scoped authorization grant."""

    grant_id: str
    principal: str
    scope: list[str]
    resource: str
    action: list[str]
    granted_by: str
    granted_at: datetime
    expires_at: datetime | None = None
    budget: dict[str, Any] | None = None
    conditions: dict[str, Any] | None = None
    revoked_at: datetime | None = None
    revoked_by: str | None = None
    revocation_reason: str | None = None


class CapabilityGrantStore:
    """In-memory capability grant store (scaffold).

    Validates grants against ``capability_grant.schema.json`` on creation.
    """

    def __init__(self) -> None:
        self._grants: dict[str, CapabilityGrant] = {}

    def create(self, grant: CapabilityGrant) -> None:
        """Store a grant after schema validation."""
        if _HAS_CONTRACTS:
            grant_dict = {
                "grant_id": grant.grant_id,
                "principal": grant.principal,
                "scope": grant.scope,
                "resource": grant.resource,
                "action": grant.action,
                "granted_by": grant.granted_by,
                "granted_at": grant.granted_at.isoformat(),
                "expires_at": grant.expires_at.isoformat() if grant.expires_at else None,
                "budget": grant.budget,
                "conditions": grant.conditions,
                "revoked_at": grant.revoked_at.isoformat() if grant.revoked_at else None,
                "revoked_by": grant.revoked_by,
                "revocation_reason": grant.revocation_reason,
            }
            try:
                _validate_contract(grant_dict, "capability_grant")
            except _ContractValidationError as exc:
                raise ValueError(f"Grant failed schema validation: {exc.errors}") from exc

        self._grants[grant.grant_id] = grant
        logger.debug(f"Created grant {grant.grant_id} for principal {grant.principal}")

    def find_grants(self, principal: str, workspace_id: str) -> list[CapabilityGrant]:
        """Return all grants for a principal in a workspace."""
        return [
            g for g in self._grants.values()
            if g.principal == principal
            and (
                workspace_id in g.resource
                or g.resource == "*"
                or workspace_id in (g.conditions.get("allowed_workspaces", []) if g.conditions else [])
            )
        ]

    def revoke(self, grant_id: str, revoked_by: str, reason: str) -> bool:
        """Revoke a grant. Returns True if found."""
        grant = self._grants.get(grant_id)
        if not grant:
            return False
        grant.revoked_at = datetime.now(timezone.utc)
        grant.revoked_by = revoked_by
        grant.revocation_reason = reason
        logger.debug(f"Revoked grant {grant_id} by {revoked_by}")
        return True

    def list_all(self) -> list[CapabilityGrant]:
        """Return all grants (for admin / audit)."""
        return list(self._grants.values())
