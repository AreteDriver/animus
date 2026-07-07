"""Adversarial tests for :class:`PolicyDecisionPoint`.

Covers default-deny, expiry, revocation, action scope, schema restrictions,
high-risk escalation, and multiple grants.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from animus.policy import (
    CapabilityGrant,
    CapabilityGrantStore,
    Decision,
    DenialReason,
    PolicyDecisionPoint,
)


@pytest.fixture
def store():
    return CapabilityGrantStore()


@pytest.fixture
def pdp(store):
    return PolicyDecisionPoint(store)


@pytest.fixture
def active_grant(store):
    g = CapabilityGrant(
        grant_id="grant-001",
        principal="user-alice",
        scope=["memory"],
        resource="ws-test",
        action=["read", "create", "update"],
        granted_by="admin",
        granted_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    store.create(g)
    return g


class TestDefaultDeny:
    """No grants → DENY."""

    def test_no_grants(self, pdp):
        result = pdp.evaluate(
            principal="user-bob",
            action="read",
            resource="mem-001",
            workspace_id="ws-test",
        )
        assert result.decision == Decision.DENY
        assert result.denial_reason_code == DenialReason.MISSING_SCOPE

    def test_no_matching_workspace(self, pdp, active_grant):
        result = pdp.evaluate(
            principal="user-alice",
            action="read",
            resource="mem-001",
            workspace_id="ws-other",
        )
        assert result.decision == Decision.DENY


class TestExpiryAndRevocation:
    """Expired or revoked grants do not permit."""

    def test_expired_grant(self, store, pdp):
        g = CapabilityGrant(
            grant_id="grant-expired",
            principal="user-alice",
            scope=["memory"],
            resource="ws-test",
            action=["read"],
            granted_by="admin",
            granted_at=datetime.now(timezone.utc) - timedelta(hours=2),
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        store.create(g)
        result = pdp.evaluate(
            principal="user-alice",
            action="read",
            resource="mem-001",
            workspace_id="ws-test",
        )
        assert result.decision == Decision.DENY
        assert result.denial_reason_code == DenialReason.CAPABILITY_REVOKED

    def test_revoked_grant(self, store, pdp, active_grant):
        store.revoke(active_grant.grant_id, "admin", "security incident")
        result = pdp.evaluate(
            principal="user-alice",
            action="read",
            resource="mem-001",
            workspace_id="ws-test",
        )
        assert result.decision == Decision.DENY
        assert result.denial_reason_code == DenialReason.CAPABILITY_REVOKED


class TestActionScope:
    """Grants must explicitly permit the requested action."""

    def test_permitted_action(self, pdp, active_grant):
        result = pdp.evaluate(
            principal="user-alice",
            action="read",
            resource="mem-001",
            workspace_id="ws-test",
        )
        assert result.decision == Decision.ALLOW

    def test_unpermitted_action(self, pdp, active_grant):
        result = pdp.evaluate(
            principal="user-alice",
            action="delete",
            resource="mem-001",
            workspace_id="ws-test",
        )
        assert result.decision == Decision.DENY
        assert result.denial_reason_code == DenialReason.MISSING_SCOPE


class TestSchemaRestrictions:
    """Schema whitelisting via conditions."""

    def test_allowed_schema(self, store, pdp):
        g = CapabilityGrant(
            grant_id="grant-schema",
            principal="user-alice",
            scope=["memory"],
            resource="ws-test",
            action=["read"],
            granted_by="admin",
            granted_at=datetime.now(timezone.utc),
            conditions={"allowed_schemas": ["memory_candidate"]},
        )
        store.create(g)
        result = pdp.evaluate(
            principal="user-alice",
            action="read",
            resource="mem-001",
            workspace_id="ws-test",
            schema_id="memory_candidate",
        )
        assert result.decision == Decision.ALLOW

    def test_disallowed_schema(self, store, pdp):
        g = CapabilityGrant(
            grant_id="grant-schema-deny",
            principal="user-alice",
            scope=["memory"],
            resource="ws-test",
            action=["read"],
            granted_by="admin",
            granted_at=datetime.now(timezone.utc),
            conditions={"allowed_schemas": ["memory_candidate"]},
        )
        store.create(g)
        result = pdp.evaluate(
            principal="user-alice",
            action="read",
            resource="mem-001",
            workspace_id="ws-test",
            schema_id="observation",
        )
        assert result.decision == Decision.DENY
        assert result.denial_reason_code == DenialReason.UNKNOWN_SCHEMA

    def test_no_schema_restrictions(self, pdp, active_grant):
        # Grant has no conditions.allowed_schemas → schema check skipped
        result = pdp.evaluate(
            principal="user-alice",
            action="read",
            resource="mem-001",
            workspace_id="ws-test",
            schema_id="any_schema",
        )
        assert result.decision == Decision.ALLOW


class TestHighRiskEscalation:
    """High-risk actions escalate regardless of grants."""

    def test_delete_escalates(self, store, pdp):
        g = CapabilityGrant(
            grant_id="grant-delete",
            principal="user-alice",
            scope=["memory"],
            resource="ws-test",
            action=["delete"],
            granted_by="admin",
            granted_at=datetime.now(timezone.utc),
        )
        store.create(g)
        result = pdp.evaluate(
            principal="user-alice",
            action="delete",
            resource="mem-001",
            workspace_id="ws-test",
        )
        assert result.decision == Decision.ESCALATE
        assert result.denial_reason_code == DenialReason.ESCALATION_REQUIRED
        assert any(o["obligation_type"] == "approve" for o in result.obligations)

    @pytest.mark.parametrize("action", ["execute", "delegate", "export"])
    def test_other_high_risk_escalates(self, store, pdp, action):
        g = CapabilityGrant(
            grant_id=f"grant-{action}",
            principal="user-alice",
            scope=["memory"],
            resource="ws-test",
            action=[action],
            granted_by="admin",
            granted_at=datetime.now(timezone.utc),
        )
        store.create(g)
        result = pdp.evaluate(
            principal="user-alice",
            action=action,
            resource="mem-001",
            workspace_id="ws-test",
        )
        assert result.decision == Decision.ESCALATE


class TestMultipleGrants:
    """Multiple grants: one active permit suffices."""

    def test_one_active_one_expired(self, store, pdp):
        expired = CapabilityGrant(
            grant_id="grant-old",
            principal="user-alice",
            scope=["memory"],
            resource="ws-test",
            action=["read"],
            granted_by="admin",
            granted_at=datetime.now(timezone.utc) - timedelta(hours=2),
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        active = CapabilityGrant(
            grant_id="grant-new",
            principal="user-alice",
            scope=["memory"],
            resource="ws-test",
            action=["create"],
            granted_by="admin",
            granted_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        store.create(expired)
        store.create(active)
        result = pdp.evaluate(
            principal="user-alice",
            action="create",
            resource="mem-001",
            workspace_id="ws-test",
        )
        assert result.decision == Decision.ALLOW


class TestStoreAdmin:
    """CapabilityGrantStore admin operations."""

    def test_list_all(self, store, active_grant):
        all_grants = store.list_all()
        assert len(all_grants) == 1
        assert all_grants[0].grant_id == active_grant.grant_id

    def test_revoke_missing(self, store):
        assert store.revoke("nonexistent", "admin", "test") is False
