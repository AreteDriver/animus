"""Multi-tenant support for organization isolation."""

from __future__ import annotations

import json
import logging
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from animus_forge.state.backends import DatabaseBackend

logger = logging.getLogger(__name__)


class OrganizationRole(str, Enum):
    """User roles within an organization."""

    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class OrganizationStatus(str, Enum):
    """Organization status."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING = "pending"


@dataclass
class Organization:
    """An organization (tenant) in the system."""

    id: str
    name: str
    slug: str  # URL-safe identifier
    status: OrganizationStatus = OrganizationStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    settings: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def create(cls, name: str, slug: str | None = None) -> Organization:
        """Create a new organization."""
        if not slug:
            slug = cls._generate_slug(name)
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            slug=slug,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    @staticmethod
    def _generate_slug(name: str) -> str:
        """Generate URL-safe slug from name."""
        slug = name.lower().replace(" ", "-")
        slug = "".join(c for c in slug if c.isalnum() or c == "-")
        return slug[:50]


@dataclass
class OrganizationMember:
    """A user's membership in an organization."""

    id: str
    organization_id: str
    user_id: str
    role: OrganizationRole
    joined_at: datetime = field(default_factory=datetime.now)
    invited_by: str | None = None


@dataclass
class OrganizationInvite:
    """An invitation to join an organization."""

    id: str
    organization_id: str
    email: str
    role: OrganizationRole
    token: str
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    accepted_at: datetime | None = None
    invited_by: str | None = None


class TenantManager:
    """Manages organizations and memberships.

    Provides CRUD operations for organizations and handles
    user membership management with role-based access.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS organizations (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        slug TEXT UNIQUE NOT NULL,
        status TEXT DEFAULT 'active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        settings TEXT,
        metadata TEXT
    );

    CREATE TABLE IF NOT EXISTS organization_members (
        id TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'member',
        joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        invited_by TEXT,
        UNIQUE(organization_id, user_id),
        FOREIGN KEY(organization_id) REFERENCES organizations(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS organization_invites (
        id TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL,
        email TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'member',
        token TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP,
        accepted_at TIMESTAMP,
        invited_by TEXT,
        FOREIGN KEY(organization_id) REFERENCES organizations(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_org_members_user ON organization_members(user_id);
    CREATE INDEX IF NOT EXISTS idx_org_members_org ON organization_members(organization_id);
    CREATE INDEX IF NOT EXISTS idx_org_invites_token ON organization_invites(token);
    CREATE INDEX IF NOT EXISTS idx_org_invites_email ON organization_invites(email);
    CREATE INDEX IF NOT EXISTS idx_organizations_slug ON organizations(slug);
    """

    def __init__(self, backend: DatabaseBackend):
        """Initialize tenant manager.

        Args:
            backend: Database backend for persistence.
        """
        self.backend = backend
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Ensure tenant tables exist."""
        try:
            with self.backend.transaction() as conn:
                for statement in self.SCHEMA.split(";"):
                    statement = statement.strip()
                    if statement:
                        conn.execute(statement)
            logger.info("Tenant schema initialized")
        except Exception as e:
            logger.error(f"Failed to initialize tenant schema: {e}")

    # Organization CRUD

    def create_organization(
        self,
        name: str,
        slug: str | None = None,
        owner_user_id: str | None = None,
        settings: dict | None = None,
    ) -> Organization:
        """Create a new organization.

        Args:
            name: Organization name.
            slug: URL-safe identifier (auto-generated if not provided).
            owner_user_id: User ID to set as owner (optional).
            settings: Initial organization settings.

        Returns:
            Created organization.
        """
        org = Organization.create(name, slug)
        if settings:
            org.settings = settings

        sql = """
            INSERT INTO organizations (id, name, slug, status, created_at, updated_at, settings, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.backend.execute(
            sql,
            [
                org.id,
                org.name,
                org.slug,
                org.status.value,
                org.created_at.isoformat(),
                org.updated_at.isoformat(),
                json.dumps(org.settings),
                json.dumps(org.metadata),
            ],
        )

        # Add owner if specified
        if owner_user_id:
            self.add_member(org.id, owner_user_id, OrganizationRole.OWNER)

        logger.info(f"Created organization {org.name} ({org.id})")
        return org

    def get_organization(self, org_id: str) -> Organization | None:
        """Get organization by ID.

        Args:
            org_id: Organization ID.

        Returns:
            Organization or None if not found.
        """
        sql = """
            SELECT id, name, slug, status, created_at, updated_at, settings, metadata
            FROM organizations WHERE id = ?
        """
        rows = self.backend.execute(sql, [org_id])
        if rows:
            return self._row_to_organization(rows[0])
        return None

    def get_organization_by_slug(self, slug: str) -> Organization | None:
        """Get organization by slug.

        Args:
            slug: Organization slug.

        Returns:
            Organization or None if not found.
        """
        sql = """
            SELECT id, name, slug, status, created_at, updated_at, settings, metadata
            FROM organizations WHERE slug = ?
        """
        rows = self.backend.execute(sql, [slug])
        if rows:
            return self._row_to_organization(rows[0])
        return None

    def update_organization(
        self,
        org_id: str,
        name: str | None = None,
        status: OrganizationStatus | None = None,
        settings: dict | None = None,
    ) -> bool:
        """Update organization.

        Args:
            org_id: Organization ID.
            name: New name.
            status: New status.
            settings: Updated settings (merged with existing).

        Returns:
            True if updated.
        """
        updates = ["updated_at = ?"]
        params = [datetime.now().isoformat()]

        if name:
            updates.append("name = ?")
            params.append(name)

        if status:
            updates.append("status = ?")
            params.append(status.value)

        if settings is not None:
            # Merge with existing settings
            org = self.get_organization(org_id)
            if org:
                merged = {**org.settings, **settings}
                updates.append("settings = ?")
                params.append(json.dumps(merged))

        params.append(org_id)
        sql = f"UPDATE organizations SET {', '.join(updates)} WHERE id = ?"

        try:
            self.backend.execute(sql, params)
            return True
        except Exception as e:
            logger.error(f"Failed to update organization: {e}")
            return False

    def delete_organization(self, org_id: str) -> bool:
        """Delete organization and all related data.

        Args:
            org_id: Organization ID.

        Returns:
            True if deleted.
        """
        try:
            # Members and invites cascade delete
            self.backend.execute("DELETE FROM organizations WHERE id = ?", [org_id])
            logger.info(f"Deleted organization {org_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete organization: {e}")
            return False

    def list_organizations(
        self,
        status: OrganizationStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Organization]:
        """List organizations.

        Args:
            status: Filter by status.
            limit: Max results.
            offset: Skip results.

        Returns:
            List of organizations.
        """
        sql = """
            SELECT id, name, slug, status, created_at, updated_at, settings, metadata
            FROM organizations
        """
        params = []

        if status:
            sql += " WHERE status = ?"
            params.append(status.value)

        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        results = []
        rows = self.backend.execute(sql, params)
        for row in rows:
            results.append(self._row_to_organization(row))
        return results

    # Membership management

    def add_member(
        self,
        org_id: str,
        user_id: str,
        role: OrganizationRole = OrganizationRole.MEMBER,
        invited_by: str | None = None,
    ) -> OrganizationMember:
        """Add a member to an organization.

        Args:
            org_id: Organization ID.
            user_id: User ID to add.
            role: Member role.
            invited_by: User ID who invited this member.

        Returns:
            Created membership.
        """
        member = OrganizationMember(
            id=str(uuid.uuid4()),
            organization_id=org_id,
            user_id=user_id,
            role=role,
            joined_at=datetime.now(),
            invited_by=invited_by,
        )

        sql = """
            INSERT INTO organization_members (id, organization_id, user_id, role, joined_at, invited_by)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        self.backend.execute(
            sql,
            [
                member.id,
                member.organization_id,
                member.user_id,
                member.role.value,
                member.joined_at.isoformat(),
                member.invited_by,
            ],
        )

        logger.info(f"Added {user_id} to organization {org_id} as {role.value}")
        return member

    def remove_member(self, org_id: str, user_id: str) -> bool:
        """Remove a member from an organization.

        Args:
            org_id: Organization ID.
            user_id: User ID to remove.

        Returns:
            True if removed.
        """
        sql = "DELETE FROM organization_members WHERE organization_id = ? AND user_id = ?"
        try:
            self.backend.execute(sql, [org_id, user_id])
            return True
        except Exception as e:
            logger.error(f"Failed to remove member: {e}")
            return False

    def update_member_role(self, org_id: str, user_id: str, role: OrganizationRole) -> bool:
        """Update a member's role.

        Args:
            org_id: Organization ID.
            user_id: User ID.
            role: New role.

        Returns:
            True if updated.
        """
        sql = "UPDATE organization_members SET role = ? WHERE organization_id = ? AND user_id = ?"
        try:
            self.backend.execute(sql, [role.value, org_id, user_id])
            return True
        except Exception as e:
            logger.error(f"Failed to update member role: {e}")
            return False

    def get_member(self, org_id: str, user_id: str) -> OrganizationMember | None:
        """Get membership for a user in an organization.

        Args:
            org_id: Organization ID.
            user_id: User ID.

        Returns:
            Membership or None.
        """
        sql = """
            SELECT id, organization_id, user_id, role, joined_at, invited_by
            FROM organization_members
            WHERE organization_id = ? AND user_id = ?
        """
        rows = self.backend.execute(sql, [org_id, user_id])
        if rows:
            return self._row_to_member(rows[0])
        return None

    def list_members(self, org_id: str) -> list[OrganizationMember]:
        """List all members of an organization.

        Args:
            org_id: Organization ID.

        Returns:
            List of memberships.
        """
        sql = """
            SELECT id, organization_id, user_id, role, joined_at, invited_by
            FROM organization_members
            WHERE organization_id = ?
            ORDER BY role, joined_at
        """
        results = []
        rows = self.backend.execute(sql, [org_id])
        for row in rows:
            results.append(self._row_to_member(row))
        return results

    def get_user_organizations(self, user_id: str) -> list[tuple[Organization, OrganizationRole]]:
        """Get all organizations a user belongs to.

        Args:
            user_id: User ID.

        Returns:
            List of (organization, role) tuples.
        """
        sql = """
            SELECT o.id, o.name, o.slug, o.status, o.created_at, o.updated_at,
                   o.settings, o.metadata, m.role
            FROM organizations o
            JOIN organization_members m ON o.id = m.organization_id
            WHERE m.user_id = ?
            ORDER BY o.name
        """
        results = []
        rows = self.backend.execute(sql, [user_id])
        for row in rows:
            org = self._row_to_organization(row[:8])
            role = OrganizationRole(row[8])
            results.append((org, role))
        return results

    def get_default_organization(self, user_id: str) -> Organization | None:
        """Get user's default (first) organization.

        Args:
            user_id: User ID.

        Returns:
            Default organization or None.
        """
        orgs = self.get_user_organizations(user_id)
        if orgs:
            return orgs[0][0]
        return None

    # Invite management

    def create_invite(
        self,
        org_id: str,
        email: str,
        role: OrganizationRole = OrganizationRole.MEMBER,
        invited_by: str | None = None,
        expires_hours: int = 72,
    ) -> OrganizationInvite:
        """Create an invitation to join an organization.

        Args:
            org_id: Organization ID.
            email: Email to invite.
            role: Role to assign.
            invited_by: User ID who sent invite.
            expires_hours: Hours until invite expires.

        Returns:
            Created invite.
        """
        from datetime import timedelta

        invite = OrganizationInvite(
            id=str(uuid.uuid4()),
            organization_id=org_id,
            email=email.lower(),
            role=role,
            token=secrets.token_urlsafe(32),
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=expires_hours),
            invited_by=invited_by,
        )

        sql = """
            INSERT INTO organization_invites
            (id, organization_id, email, role, token, created_at, expires_at, invited_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.backend.execute(
            sql,
            [
                invite.id,
                invite.organization_id,
                invite.email,
                invite.role.value,
                invite.token,
                invite.created_at.isoformat(),
                invite.expires_at.isoformat() if invite.expires_at else None,
                invite.invited_by,
            ],
        )

        logger.info(f"Created invite for {email} to organization {org_id}")
        return invite

    def accept_invite(self, token: str, user_id: str) -> OrganizationMember | None:
        """Accept an invitation.

        Args:
            token: Invite token.
            user_id: User accepting the invite.

        Returns:
            Membership if accepted, None if invalid/expired.
        """
        sql = """
            SELECT id, organization_id, email, role, token, created_at, expires_at, invited_by
            FROM organization_invites
            WHERE token = ? AND accepted_at IS NULL
        """
        rows = self.backend.execute(sql, [token])
        if not rows:
            return None

        invite = self._row_to_invite(rows[0])

        # Check expiry
        if invite.expires_at and invite.expires_at < datetime.now():
            return None

        # Add member
        member = self.add_member(
            invite.organization_id,
            user_id,
            invite.role,
            invite.invited_by,
        )

        # Mark invite as accepted
        self.backend.execute(
            "UPDATE organization_invites SET accepted_at = ? WHERE id = ?",
            [datetime.now().isoformat(), invite.id],
        )

        return member

    def revoke_invite(self, invite_id: str) -> bool:
        """Revoke an invitation.

        Args:
            invite_id: Invite ID.

        Returns:
            True if revoked.
        """
        try:
            self.backend.execute("DELETE FROM organization_invites WHERE id = ?", [invite_id])
            return True
        except Exception as e:
            logger.error(f"Failed to revoke invite: {e}")
            return False

    def list_pending_invites(self, org_id: str) -> list[OrganizationInvite]:
        """List pending invitations for an organization.

        Args:
            org_id: Organization ID.

        Returns:
            List of pending invites.
        """
        sql = """
            SELECT id, organization_id, email, role, token, created_at, expires_at,
                   accepted_at, invited_by
            FROM organization_invites
            WHERE organization_id = ? AND accepted_at IS NULL
            ORDER BY created_at DESC
        """
        results = []
        rows = self.backend.execute(sql, [org_id])
        for row in rows:
            results.append(self._row_to_invite(row))
        return results

    # Permission checks

    def check_permission(
        self,
        org_id: str,
        user_id: str,
        required_role: OrganizationRole,
    ) -> bool:
        """Check if user has required role in organization.

        Args:
            org_id: Organization ID.
            user_id: User ID.
            required_role: Minimum required role.

        Returns:
            True if user has permission.
        """
        member = self.get_member(org_id, user_id)
        if not member:
            return False

        role_hierarchy = {
            OrganizationRole.VIEWER: 0,
            OrganizationRole.MEMBER: 1,
            OrganizationRole.ADMIN: 2,
            OrganizationRole.OWNER: 3,
        }

        return role_hierarchy[member.role] >= role_hierarchy[required_role]

    def is_owner(self, org_id: str, user_id: str) -> bool:
        """Check if user is organization owner."""
        return self.check_permission(org_id, user_id, OrganizationRole.OWNER)

    def is_admin(self, org_id: str, user_id: str) -> bool:
        """Check if user is organization admin or owner."""
        return self.check_permission(org_id, user_id, OrganizationRole.ADMIN)

    def is_member(self, org_id: str, user_id: str) -> bool:
        """Check if user is organization member."""
        return self.check_permission(org_id, user_id, OrganizationRole.VIEWER)

    # Helper methods

    def _row_to_organization(self, row: tuple) -> Organization:
        """Convert database row to Organization."""
        return Organization(
            id=row[0],
            name=row[1],
            slug=row[2],
            status=OrganizationStatus(row[3]) if row[3] else OrganizationStatus.ACTIVE,
            created_at=datetime.fromisoformat(row[4]) if row[4] else datetime.now(),
            updated_at=datetime.fromisoformat(row[5]) if row[5] else datetime.now(),
            settings=json.loads(row[6]) if row[6] else {},
            metadata=json.loads(row[7]) if row[7] else {},
        )

    def _row_to_member(self, row: tuple) -> OrganizationMember:
        """Convert database row to OrganizationMember."""
        return OrganizationMember(
            id=row[0],
            organization_id=row[1],
            user_id=row[2],
            role=OrganizationRole(row[3]) if row[3] else OrganizationRole.MEMBER,
            joined_at=datetime.fromisoformat(row[4]) if row[4] else datetime.now(),
            invited_by=row[5],
        )

    def _row_to_invite(self, row: tuple) -> OrganizationInvite:
        """Convert database row to OrganizationInvite."""
        return OrganizationInvite(
            id=row[0],
            organization_id=row[1],
            email=row[2],
            role=OrganizationRole(row[3]) if row[3] else OrganizationRole.MEMBER,
            token=row[4],
            created_at=datetime.fromisoformat(row[5]) if row[5] else datetime.now(),
            expires_at=datetime.fromisoformat(row[6]) if row[6] else None,
            accepted_at=datetime.fromisoformat(row[7])
            if row[7]
            else None
            if len(row) > 7
            else None,
            invited_by=row[8] if len(row) > 8 else None,
        )
