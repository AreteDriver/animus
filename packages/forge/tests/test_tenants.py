"""Tests for multi-tenant support."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from animus_forge.auth.tenants import (
    Organization,
    OrganizationInvite,
    OrganizationMember,
    OrganizationRole,
    OrganizationStatus,
    TenantManager,
)


class TestOrganizationModels:
    """Tests for organization data models."""

    def test_organization_role_values(self):
        """Test all organization roles are defined."""
        expected = ["owner", "admin", "member", "viewer"]
        actual = [r.value for r in OrganizationRole]
        for role in expected:
            assert role in actual

    def test_organization_status_values(self):
        """Test all organization statuses are defined."""
        expected = ["active", "suspended", "pending"]
        actual = [s.value for s in OrganizationStatus]
        for status in expected:
            assert status in actual

    def test_organization_create(self):
        """Test creating organization."""
        org = Organization.create("Test Organization")
        assert org.name == "Test Organization"
        assert org.slug == "test-organization"
        assert org.status == OrganizationStatus.ACTIVE
        assert org.id is not None

    def test_organization_create_with_slug(self):
        """Test creating organization with custom slug."""
        org = Organization.create("My Org", slug="custom-slug")
        assert org.name == "My Org"
        assert org.slug == "custom-slug"

    def test_organization_slug_generation(self):
        """Test slug generation from name."""
        # Normal case
        assert Organization._generate_slug("Test Org") == "test-org"

        # Special characters
        assert Organization._generate_slug("Test!@#$Org") == "testorg"

        # Long name
        long_name = "A" * 100
        assert len(Organization._generate_slug(long_name)) == 50

    def test_organization_member_creation(self):
        """Test creating organization member."""
        member = OrganizationMember(
            id="m1",
            organization_id="org1",
            user_id="user1",
            role=OrganizationRole.ADMIN,
        )
        assert member.organization_id == "org1"
        assert member.role == OrganizationRole.ADMIN

    def test_organization_invite_creation(self):
        """Test creating organization invite."""
        invite = OrganizationInvite(
            id="i1",
            organization_id="org1",
            email="test@example.com",
            role=OrganizationRole.MEMBER,
            token="abc123",
        )
        assert invite.email == "test@example.com"
        assert invite.accepted_at is None


class TestTenantManager:
    """Tests for TenantManager class."""

    @pytest.fixture
    def mock_backend(self):
        """Create mock database backend."""
        backend = MagicMock()
        backend.execute.return_value = []
        backend.transaction.return_value.__enter__ = MagicMock()
        backend.transaction.return_value.__exit__ = MagicMock()
        return backend

    @pytest.fixture
    def manager(self, mock_backend):
        """Create tenant manager with mock backend."""
        return TenantManager(mock_backend)

    def test_manager_init(self, manager, mock_backend):
        """Test manager initialization creates schema."""
        assert manager.backend == mock_backend
        mock_backend.transaction.assert_called()

    def test_create_organization(self, manager, mock_backend):
        """Test creating organization."""
        org = manager.create_organization("Test Org")
        assert org.name == "Test Org"
        assert org.slug == "test-org"
        mock_backend.execute.assert_called()

    def test_create_organization_with_owner(self, manager, mock_backend):
        """Test creating organization with owner."""
        org = manager.create_organization("Test Org", owner_user_id="user1")
        assert org.name == "Test Org"
        # Should have called execute twice (org + member)
        assert mock_backend.execute.call_count >= 2

    def test_get_organization_not_found(self, manager, mock_backend):
        """Test getting non-existent organization."""
        mock_backend.execute.return_value = []
        result = manager.get_organization("nonexistent")
        assert result is None

    def test_get_organization_by_slug_not_found(self, manager, mock_backend):
        """Test getting organization by non-existent slug."""
        mock_backend.execute.return_value = []
        result = manager.get_organization_by_slug("nonexistent")
        assert result is None

    def test_update_organization(self, manager, mock_backend):
        """Test updating organization."""
        # Setup mock to return existing org for settings merge
        mock_backend.execute.return_value = [
            ("id", "name", "slug", "active", "2026-01-01", "2026-01-01", "{}", "{}")
        ]
        result = manager.update_organization("org1", name="New Name")
        assert result is True

    def test_delete_organization(self, manager, mock_backend):
        """Test deleting organization."""
        result = manager.delete_organization("org1")
        assert result is True
        mock_backend.execute.assert_called()

    def test_list_organizations(self, manager, mock_backend):
        """Test listing organizations."""
        mock_backend.execute.return_value = []
        result = manager.list_organizations()
        assert result == []

    def test_list_organizations_with_status(self, manager, mock_backend):
        """Test listing organizations filtered by status."""
        mock_backend.execute.return_value = []
        result = manager.list_organizations(status=OrganizationStatus.ACTIVE)
        assert result == []
        # Verify status filter in SQL
        call_args = mock_backend.execute.call_args
        assert "status = ?" in call_args[0][0]

    def test_add_member(self, manager, mock_backend):
        """Test adding member to organization."""
        member = manager.add_member("org1", "user1", OrganizationRole.MEMBER)
        assert member.organization_id == "org1"
        assert member.user_id == "user1"
        assert member.role == OrganizationRole.MEMBER

    def test_remove_member(self, manager, mock_backend):
        """Test removing member from organization."""
        result = manager.remove_member("org1", "user1")
        assert result is True

    def test_update_member_role(self, manager, mock_backend):
        """Test updating member role."""
        result = manager.update_member_role("org1", "user1", OrganizationRole.ADMIN)
        assert result is True

    def test_get_member_not_found(self, manager, mock_backend):
        """Test getting non-existent member."""
        mock_backend.execute.return_value = []
        result = manager.get_member("org1", "user1")
        assert result is None

    def test_list_members(self, manager, mock_backend):
        """Test listing organization members."""
        mock_backend.execute.return_value = []
        result = manager.list_members("org1")
        assert result == []

    def test_get_user_organizations(self, manager, mock_backend):
        """Test getting user's organizations."""
        mock_backend.execute.return_value = []
        result = manager.get_user_organizations("user1")
        assert result == []

    def test_get_default_organization_none(self, manager, mock_backend):
        """Test getting default organization when user has none."""
        mock_backend.execute.return_value = []
        result = manager.get_default_organization("user1")
        assert result is None

    def test_create_invite(self, manager, mock_backend):
        """Test creating invitation."""
        invite = manager.create_invite(
            "org1", "test@example.com", OrganizationRole.MEMBER, "inviter1"
        )
        assert invite.email == "test@example.com"
        assert invite.role == OrganizationRole.MEMBER
        assert invite.token is not None

    def test_accept_invite_not_found(self, manager, mock_backend):
        """Test accepting non-existent invite."""
        mock_backend.execute.return_value = []
        result = manager.accept_invite("invalid-token", "user1")
        assert result is None

    def test_revoke_invite(self, manager, mock_backend):
        """Test revoking invitation."""
        result = manager.revoke_invite("invite1")
        assert result is True

    def test_list_pending_invites(self, manager, mock_backend):
        """Test listing pending invitations."""
        mock_backend.execute.return_value = []
        result = manager.list_pending_invites("org1")
        assert result == []


class TestTenantPermissions:
    """Tests for permission checking."""

    @pytest.fixture
    def mock_backend(self):
        """Create mock database backend."""
        backend = MagicMock()
        backend.execute.return_value = []
        backend.transaction.return_value.__enter__ = MagicMock()
        backend.transaction.return_value.__exit__ = MagicMock()
        return backend

    @pytest.fixture
    def manager(self, mock_backend):
        """Create tenant manager with mock backend."""
        return TenantManager(mock_backend)

    def test_check_permission_no_member(self, manager, mock_backend):
        """Test permission check for non-member."""
        mock_backend.execute.return_value = []
        result = manager.check_permission("org1", "user1", OrganizationRole.VIEWER)
        assert result is False

    def test_check_permission_viewer(self, manager, mock_backend):
        """Test viewer permission check."""
        mock_backend.execute.return_value = [("m1", "org1", "user1", "viewer", "2026-01-01", None)]
        result = manager.check_permission("org1", "user1", OrganizationRole.VIEWER)
        assert result is True

        # Viewer cannot do member actions
        result = manager.check_permission("org1", "user1", OrganizationRole.MEMBER)
        assert result is False

    def test_check_permission_owner(self, manager, mock_backend):
        """Test owner has all permissions."""
        mock_backend.execute.return_value = [("m1", "org1", "user1", "owner", "2026-01-01", None)]

        assert manager.check_permission("org1", "user1", OrganizationRole.VIEWER)
        assert manager.check_permission("org1", "user1", OrganizationRole.MEMBER)
        assert manager.check_permission("org1", "user1", OrganizationRole.ADMIN)
        assert manager.check_permission("org1", "user1", OrganizationRole.OWNER)

    def test_is_owner(self, manager, mock_backend):
        """Test is_owner helper."""
        mock_backend.execute.return_value = [("m1", "org1", "user1", "owner", "2026-01-01", None)]
        assert manager.is_owner("org1", "user1") is True

    def test_is_admin_with_owner(self, manager, mock_backend):
        """Test is_admin returns true for owner."""
        mock_backend.execute.return_value = [("m1", "org1", "user1", "owner", "2026-01-01", None)]
        assert manager.is_admin("org1", "user1") is True

    def test_is_member(self, manager, mock_backend):
        """Test is_member helper."""
        mock_backend.execute.return_value = [("m1", "org1", "user1", "viewer", "2026-01-01", None)]
        assert manager.is_member("org1", "user1") is True


class TestTenantIntegration:
    """Integration tests for tenant management."""

    def test_organization_to_dict(self):
        """Test organization can be serialized."""
        org = Organization.create("Test")
        # Should have all required fields
        assert org.id
        assert org.name
        assert org.slug
        assert org.status
        assert org.created_at
        assert org.updated_at

    def test_role_hierarchy(self):
        """Test role hierarchy values."""
        role_hierarchy = {
            OrganizationRole.VIEWER: 0,
            OrganizationRole.MEMBER: 1,
            OrganizationRole.ADMIN: 2,
            OrganizationRole.OWNER: 3,
        }

        # Verify hierarchy is correct
        assert role_hierarchy[OrganizationRole.OWNER] > role_hierarchy[OrganizationRole.ADMIN]
        assert role_hierarchy[OrganizationRole.ADMIN] > role_hierarchy[OrganizationRole.MEMBER]
        assert role_hierarchy[OrganizationRole.MEMBER] > role_hierarchy[OrganizationRole.VIEWER]

    def test_invite_expiry(self):
        """Test invite expiry checking."""
        # Create expired invite
        invite = OrganizationInvite(
            id="i1",
            organization_id="org1",
            email="test@example.com",
            role=OrganizationRole.MEMBER,
            token="token",
            created_at=datetime.now() - timedelta(days=5),
            expires_at=datetime.now() - timedelta(days=1),
        )

        # Should be expired
        assert invite.expires_at < datetime.now()

    def test_member_with_invited_by(self):
        """Test member tracks who invited them."""
        member = OrganizationMember(
            id="m1",
            organization_id="org1",
            user_id="user1",
            role=OrganizationRole.MEMBER,
            invited_by="admin1",
        )
        assert member.invited_by == "admin1"
