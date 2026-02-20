"""MCP Connector Manager.

Handles registration, storage, and connection testing for MCP servers.
"""

from __future__ import annotations

import base64
import json
import logging
import secrets
import uuid
from datetime import datetime

from animus_forge.state.backends import DatabaseBackend

from .models import (
    Credential,
    CredentialCreateInput,
    MCPConnectionTestResult,
    MCPResource,
    MCPServer,
    MCPServerCreateInput,
    MCPServerStatus,
    MCPServerUpdateInput,
    MCPTool,
)

logger = logging.getLogger(__name__)


class MCPConnectorManager:
    """Manager for MCP connector registrations."""

    def __init__(self, backend: DatabaseBackend):
        """Initialize the connector manager.

        Args:
            backend: Database backend for persistence
        """
        self.backend = backend

    # =========================================================================
    # MCP Server CRUD
    # =========================================================================

    def list_servers(self) -> list[MCPServer]:
        """List all registered MCP servers.

        Returns:
            List of MCPServer objects
        """
        rows = self.backend.fetchall(
            """
            SELECT id, name, url, type, status, description, auth_type,
                   credential_id, tools, resources, last_connected, error,
                   created_at, updated_at
            FROM mcp_servers
            ORDER BY created_at DESC
            """
        )
        return [self._row_to_server(row) for row in rows]

    def get_server(self, server_id: str) -> MCPServer | None:
        """Get an MCP server by ID.

        Args:
            server_id: Server ID

        Returns:
            MCPServer or None if not found
        """
        row = self.backend.fetchone(
            """
            SELECT id, name, url, type, status, description, auth_type,
                   credential_id, tools, resources, last_connected, error,
                   created_at, updated_at
            FROM mcp_servers
            WHERE id = ?
            """,
            (server_id,),
        )
        return self._row_to_server(row) if row else None

    def create_server(self, data: MCPServerCreateInput) -> MCPServer:
        """Register a new MCP server.

        Args:
            data: Server creation input

        Returns:
            Created MCPServer
        """
        server_id = str(uuid.uuid4())
        now = datetime.now()

        # Determine initial status based on auth requirements
        status = (
            MCPServerStatus.NOT_CONFIGURED
            if data.authType != "none" and not data.credentialId
            else MCPServerStatus.DISCONNECTED
        )

        with self.backend.transaction():
            self.backend.execute(
                """
                INSERT INTO mcp_servers
                    (id, name, url, type, status, description, auth_type,
                     credential_id, tools, resources, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    server_id,
                    data.name,
                    data.url,
                    data.type,
                    status.value,
                    data.description or "",
                    data.authType,
                    data.credentialId,
                    "[]",  # tools
                    "[]",  # resources
                    now.isoformat(),
                    now.isoformat(),
                ),
            )

        logger.info(f"Created MCP server: {data.name} ({server_id})")
        return self.get_server(server_id)

    def update_server(self, server_id: str, data: MCPServerUpdateInput) -> MCPServer | None:
        """Update an MCP server registration.

        Args:
            server_id: Server ID
            data: Update input

        Returns:
            Updated MCPServer or None if not found
        """
        existing = self.get_server(server_id)
        if not existing:
            return None

        # Build update fields
        updates = []
        params = []

        if data.name is not None:
            updates.append("name = ?")
            params.append(data.name)
        if data.url is not None:
            updates.append("url = ?")
            params.append(data.url)
        if data.type is not None:
            updates.append("type = ?")
            params.append(data.type)
        if data.authType is not None:
            updates.append("auth_type = ?")
            params.append(data.authType)
        if data.credentialId is not None:
            updates.append("credential_id = ?")
            params.append(data.credentialId if data.credentialId else None)
        if data.description is not None:
            updates.append("description = ?")
            params.append(data.description)

        if not updates:
            return existing

        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(server_id)

        with self.backend.transaction():
            self.backend.execute(
                f"UPDATE mcp_servers SET {', '.join(updates)} WHERE id = ?",
                tuple(params),
            )

        logger.info(f"Updated MCP server: {server_id}")
        return self.get_server(server_id)

    def delete_server(self, server_id: str) -> bool:
        """Delete an MCP server registration.

        Args:
            server_id: Server ID

        Returns:
            True if deleted, False if not found
        """
        existing = self.get_server(server_id)
        if not existing:
            return False

        with self.backend.transaction():
            self.backend.execute("DELETE FROM mcp_servers WHERE id = ?", (server_id,))

        logger.info(f"Deleted MCP server: {server_id}")
        return True

    def test_connection(self, server_id: str) -> MCPConnectionTestResult:
        """Test connection to an MCP server.

        Attempts to connect and discover tools/resources.

        Args:
            server_id: Server ID

        Returns:
            MCPConnectionTestResult with success/failure and discovered tools
        """
        server = self.get_server(server_id)
        if not server:
            return MCPConnectionTestResult(
                success=False, error="Server not found", tools=[], resources=[]
            )

        # Check if credentials are required but not configured
        if server.authType != "none" and not server.credentialId:
            return MCPConnectionTestResult(
                success=False,
                error="Credentials required but not configured",
                tools=[],
                resources=[],
            )

        try:
            # Build auth headers if needed
            headers = self._build_auth_headers(server)

            # Attempt real MCP discovery, fall back to simulation
            discovered_tools, discovered_resources = self._discover_server(server, headers)

            # Update server status and tools
            with self.backend.transaction():
                self.backend.execute(
                    """
                    UPDATE mcp_servers
                    SET status = ?, tools = ?, resources = ?,
                        last_connected = ?, error = NULL, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        MCPServerStatus.CONNECTED.value,
                        json.dumps([t.model_dump() for t in discovered_tools]),
                        json.dumps([r.model_dump() for r in discovered_resources]),
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                        server_id,
                    ),
                )

            logger.info(f"MCP connection test successful: {server.name}")
            return MCPConnectionTestResult(
                success=True, tools=discovered_tools, resources=discovered_resources
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"MCP connection test failed for {server.name}: {error_msg}")

            # Update server status with error
            with self.backend.transaction():
                self.backend.execute(
                    """
                    UPDATE mcp_servers
                    SET status = ?, error = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        MCPServerStatus.ERROR.value,
                        error_msg,
                        datetime.now().isoformat(),
                        server_id,
                    ),
                )

            return MCPConnectionTestResult(success=False, error=error_msg, tools=[], resources=[])

    def get_server_by_name(self, name: str) -> MCPServer | None:
        """Get an MCP server by name (case-insensitive).

        Args:
            name: Server name to look up

        Returns:
            MCPServer or None if not found
        """
        row = self.backend.fetchone(
            """
            SELECT id, name, url, type, status, description, auth_type,
                   credential_id, tools, resources, last_connected, error,
                   created_at, updated_at
            FROM mcp_servers
            WHERE LOWER(name) = LOWER(?)
            """,
            (name,),
        )
        return self._row_to_server(row) if row else None

    def get_tools(self, server_id: str) -> list[MCPTool]:
        """Get tools for an MCP server.

        Args:
            server_id: Server ID

        Returns:
            List of MCPTool objects
        """
        server = self.get_server(server_id)
        if not server:
            return []
        return server.tools

    # =========================================================================
    # Credentials CRUD
    # =========================================================================

    def list_credentials(self) -> list[Credential]:
        """List all credentials (values not exposed).

        Returns:
            List of Credential objects
        """
        rows = self.backend.fetchall(
            """
            SELECT id, name, type, service, created_at, last_used
            FROM credentials
            ORDER BY created_at DESC
            """
        )
        return [self._row_to_credential(row) for row in rows]

    def get_credential(self, credential_id: str) -> Credential | None:
        """Get a credential by ID (value not exposed).

        Args:
            credential_id: Credential ID

        Returns:
            Credential or None if not found
        """
        row = self.backend.fetchone(
            """
            SELECT id, name, type, service, created_at, last_used
            FROM credentials
            WHERE id = ?
            """,
            (credential_id,),
        )
        return self._row_to_credential(row) if row else None

    def create_credential(self, data: CredentialCreateInput) -> Credential:
        """Create a new credential.

        Args:
            data: Credential creation input

        Returns:
            Created Credential (value not exposed)
        """
        credential_id = str(uuid.uuid4())
        now = datetime.now()

        # Simple encryption for demo - in production use proper encryption
        encrypted_value = self._encrypt_value(data.value)

        with self.backend.transaction():
            self.backend.execute(
                """
                INSERT INTO credentials
                    (id, name, type, service, encrypted_value, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    credential_id,
                    data.name,
                    data.type,
                    data.service,
                    encrypted_value,
                    now.isoformat(),
                ),
            )

        logger.info(f"Created credential: {data.name} ({credential_id})")
        return self.get_credential(credential_id)

    def delete_credential(self, credential_id: str) -> bool:
        """Delete a credential.

        Args:
            credential_id: Credential ID

        Returns:
            True if deleted, False if not found
        """
        existing = self.get_credential(credential_id)
        if not existing:
            return False

        with self.backend.transaction():
            # Clear credential references from servers
            self.backend.execute(
                """
                UPDATE mcp_servers
                SET credential_id = NULL, status = ?
                WHERE credential_id = ?
                """,
                (MCPServerStatus.NOT_CONFIGURED.value, credential_id),
            )
            self.backend.execute("DELETE FROM credentials WHERE id = ?", (credential_id,))

        logger.info(f"Deleted credential: {credential_id}")
        return True

    def get_credential_value(self, credential_id: str) -> str | None:
        """Get decrypted credential value (internal use only).

        Args:
            credential_id: Credential ID

        Returns:
            Decrypted value or None if not found
        """
        row = self.backend.fetchone(
            "SELECT encrypted_value FROM credentials WHERE id = ?",
            (credential_id,),
        )
        if not row:
            return None

        # Update last used timestamp
        self.backend.execute(
            "UPDATE credentials SET last_used = ? WHERE id = ?",
            (datetime.now().isoformat(), credential_id),
        )

        return self._decrypt_value(row["encrypted_value"])

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _row_to_server(self, row: dict) -> MCPServer:
        """Convert database row to MCPServer model."""
        tools_json = row.get("tools", "[]")
        resources_json = row.get("resources", "[]")

        try:
            tools_data = json.loads(tools_json) if tools_json else []
            resources_data = json.loads(resources_json) if resources_json else []
        except json.JSONDecodeError:
            tools_data = []
            resources_data = []

        return MCPServer(
            id=row["id"],
            name=row["name"],
            url=row["url"],
            type=row["type"],
            status=row["status"],
            description=row.get("description"),
            authType=row["auth_type"],
            credentialId=row.get("credential_id"),
            tools=[MCPTool(**t) for t in tools_data],
            resources=[MCPResource(**r) for r in resources_data],
            lastConnected=self._parse_datetime(row.get("last_connected")),
            error=row.get("error"),
            createdAt=self._parse_datetime(row.get("created_at")) or datetime.now(),
            updatedAt=self._parse_datetime(row.get("updated_at")) or datetime.now(),
        )

    def _row_to_credential(self, row: dict) -> Credential:
        """Convert database row to Credential model."""
        return Credential(
            id=row["id"],
            name=row["name"],
            type=row["type"],
            service=row["service"],
            createdAt=self._parse_datetime(row.get("created_at")) or datetime.now(),
            lastUsed=self._parse_datetime(row.get("last_used")),
        )

    def _parse_datetime(self, value: str | None) -> datetime | None:
        """Parse datetime string from database."""
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    def _encrypt_value(self, value: str) -> str:
        """Simple encryption for credential values.

        Note: In production, use proper encryption (e.g., Fernet with key management).
        """
        # Simple base64 encoding with salt for demo
        salt = secrets.token_bytes(16)
        salted = salt + value.encode("utf-8")
        return base64.b64encode(salted).decode("ascii")

    def _decrypt_value(self, encrypted: str) -> str:
        """Decrypt credential value."""
        try:
            decoded = base64.b64decode(encrypted)
            # Remove 16-byte salt
            return decoded[16:].decode("utf-8")
        except Exception:
            return ""

    def _build_auth_headers(self, server: MCPServer) -> dict[str, str] | None:
        """Build HTTP auth headers from server credentials."""
        if server.authType == "none" or not server.credentialId:
            return None
        value = self.get_credential_value(server.credentialId)
        if not value:
            return None
        if server.authType == "bearer":
            return {"Authorization": f"Bearer {value}"}
        if server.authType == "api_key":
            return {"X-API-Key": value}
        return None

    def _discover_server(
        self, server: MCPServer, headers: dict[str, str] | None
    ) -> tuple[list[MCPTool], list[MCPResource]]:
        """Discover tools/resources via real SDK or simulation fallback.

        Returns:
            Tuple of (tools, resources)
        """
        from .client import HAS_MCP_SDK, MCPClientError, discover_tools

        if not HAS_MCP_SDK:
            logger.info(
                "MCP SDK not installed — using simulated discovery for %s",
                server.name,
            )
            return self._simulate_tool_discovery(server), []

        try:
            result = discover_tools(
                server_type=server.type,
                server_url=server.url,
                headers=headers,
            )
            tools = [MCPTool(**t) for t in result.get("tools", [])]
            resources = [MCPResource(**r) for r in result.get("resources", [])]
            return tools, resources
        except MCPClientError as exc:
            logger.warning(
                "Real MCP discovery failed for %s, falling back to simulation: %s",
                server.name,
                exc,
            )
            return self._simulate_tool_discovery(server), []

    def _simulate_tool_discovery(self, server: MCPServer) -> list[MCPTool]:
        """Simulate MCP tool discovery for demo purposes.

        In production, this would actually connect to the MCP server
        and perform proper tool discovery.
        """
        # Return tools based on server name/type for demo
        name_lower = server.name.lower()

        if "github" in name_lower:
            return [
                MCPTool(
                    name="list_repos",
                    description="List repositories for the authenticated user",
                    inputSchema={"type": "object", "properties": {}},
                ),
                MCPTool(
                    name="create_pr",
                    description="Create a pull request",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string"},
                            "title": {"type": "string"},
                            "body": {"type": "string"},
                        },
                        "required": ["repo", "title"],
                    },
                ),
                MCPTool(
                    name="list_issues",
                    description="List issues for a repository",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string"},
                            "state": {
                                "type": "string",
                                "enum": ["open", "closed", "all"],
                            },
                        },
                        "required": ["repo"],
                    },
                ),
            ]
        elif "filesystem" in name_lower:
            return [
                MCPTool(
                    name="read_file",
                    description="Read contents of a file",
                    inputSchema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                ),
                MCPTool(
                    name="write_file",
                    description="Write contents to a file",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                ),
                MCPTool(
                    name="list_directory",
                    description="List contents of a directory",
                    inputSchema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                ),
            ]
        elif "notion" in name_lower:
            return [
                MCPTool(
                    name="search_pages",
                    description="Search Notion pages",
                    inputSchema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                ),
                MCPTool(
                    name="get_page",
                    description="Get a Notion page by ID",
                    inputSchema={
                        "type": "object",
                        "properties": {"page_id": {"type": "string"}},
                        "required": ["page_id"],
                    },
                ),
            ]
        elif "slack" in name_lower:
            return [
                MCPTool(
                    name="send_message",
                    description="Send a message to a Slack channel",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "channel": {"type": "string"},
                            "text": {"type": "string"},
                        },
                        "required": ["channel", "text"],
                    },
                ),
                MCPTool(
                    name="list_channels",
                    description="List Slack channels",
                    inputSchema={"type": "object", "properties": {}},
                ),
            ]
        else:
            # Generic/custom server - return empty tools
            return []
