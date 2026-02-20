"""Tests for MCP connector management."""

import base64
import json
from unittest.mock import patch

import pytest

from animus_forge.mcp import (
    CredentialCreateInput,
    MCPConnectorManager,
    MCPServerCreateInput,
    MCPServerStatus,
)
from animus_forge.mcp.models import MCPServerUpdateInput
from animus_forge.state import SQLiteBackend


@pytest.fixture
def backend(tmp_path):
    """Create a test database backend."""
    db_path = tmp_path / "test.db"
    backend = SQLiteBackend(str(db_path))

    # Run schema migration
    schema = """
    CREATE TABLE IF NOT EXISTS mcp_servers (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        url TEXT NOT NULL,
        type TEXT NOT NULL DEFAULT 'sse',
        status TEXT NOT NULL DEFAULT 'not_configured',
        description TEXT DEFAULT '',
        auth_type TEXT NOT NULL DEFAULT 'none',
        credential_id TEXT,
        tools TEXT DEFAULT '[]',
        resources TEXT DEFAULT '[]',
        last_connected TIMESTAMP,
        error TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS credentials (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        service TEXT NOT NULL,
        encrypted_value TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_used TIMESTAMP
    );
    """
    backend.executescript(schema)
    return backend


@pytest.fixture
def manager(backend):
    """Create an MCP manager with test backend."""
    return MCPConnectorManager(backend)


class TestMCPServerCRUD:
    """Tests for MCP server CRUD operations."""

    def test_create_server(self, manager):
        """Test creating an MCP server."""
        data = MCPServerCreateInput(
            name="GitHub",
            url="https://mcp.github.com/sse",
            type="sse",
            authType="bearer",
            description="GitHub MCP server",
        )
        server = manager.create_server(data)

        assert server is not None
        assert server.id is not None
        assert server.name == "GitHub"
        assert server.url == "https://mcp.github.com/sse"
        assert server.type == "sse"
        assert server.authType == "bearer"
        assert server.status == "not_configured"  # No credential

    def test_create_server_no_auth(self, manager):
        """Test creating server without auth requirement."""
        data = MCPServerCreateInput(
            name="Filesystem",
            url="stdio://mcp-filesystem",
            type="stdio",
            authType="none",
        )
        server = manager.create_server(data)

        assert server.status == "disconnected"  # Ready to connect

    def test_list_servers(self, manager):
        """Test listing servers."""
        # Create multiple servers
        manager.create_server(
            MCPServerCreateInput(name="Server1", url="https://s1.com", type="sse")
        )
        manager.create_server(
            MCPServerCreateInput(name="Server2", url="https://s2.com", type="sse")
        )

        servers = manager.list_servers()
        assert len(servers) == 2
        names = [s.name for s in servers]
        assert "Server1" in names
        assert "Server2" in names

    def test_get_server(self, manager):
        """Test getting a specific server."""
        created = manager.create_server(
            MCPServerCreateInput(name="TestServer", url="https://test.com", type="sse")
        )

        fetched = manager.get_server(created.id)
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.name == "TestServer"

    def test_get_server_not_found(self, manager):
        """Test getting non-existent server."""
        result = manager.get_server("nonexistent-id")
        assert result is None

    def test_update_server(self, manager):
        """Test updating a server."""
        from animus_forge.mcp.models import MCPServerUpdateInput

        created = manager.create_server(
            MCPServerCreateInput(name="Original", url="https://test.com", type="sse")
        )

        updated = manager.update_server(
            created.id, MCPServerUpdateInput(name="Updated", description="New desc")
        )

        assert updated is not None
        assert updated.name == "Updated"
        assert updated.description == "New desc"
        assert updated.url == "https://test.com"  # Unchanged

    def test_delete_server(self, manager):
        """Test deleting a server."""
        created = manager.create_server(
            MCPServerCreateInput(name="ToDelete", url="https://test.com", type="sse")
        )

        result = manager.delete_server(created.id)
        assert result is True

        # Verify deleted
        fetched = manager.get_server(created.id)
        assert fetched is None

    def test_delete_server_not_found(self, manager):
        """Test deleting non-existent server."""
        result = manager.delete_server("nonexistent-id")
        assert result is False


class TestConnectionTest:
    """Tests for connection testing."""

    def test_connection_test_no_auth_required(self, manager):
        """Test connection with server that doesn't require auth."""
        server = manager.create_server(
            MCPServerCreateInput(
                name="Filesystem",
                url="stdio://mcp-filesystem",
                type="stdio",
                authType="none",
            )
        )

        result = manager.test_connection(server.id)
        assert result.success is True
        assert result.error is None
        # Should discover simulated tools for filesystem
        assert len(result.tools) > 0

    def test_connection_test_missing_credentials(self, manager):
        """Test connection when credentials are required but missing."""
        server = manager.create_server(
            MCPServerCreateInput(
                name="GitHub",
                url="https://mcp.github.com/sse",
                type="sse",
                authType="bearer",
            )
        )

        result = manager.test_connection(server.id)
        assert result.success is False
        assert "Credentials required" in result.error

    def test_connection_test_not_found(self, manager):
        """Test connection for non-existent server."""
        result = manager.test_connection("nonexistent-id")
        assert result.success is False
        assert "not found" in result.error.lower()


class TestCredentials:
    """Tests for credential management."""

    def test_create_credential(self, manager):
        """Test creating a credential."""
        data = CredentialCreateInput(
            name="GitHub Token",
            type="bearer",
            service="github",
            value="ghp_xxxxxxxxxxxx",
        )
        credential = manager.create_credential(data)

        assert credential is not None
        assert credential.id is not None
        assert credential.name == "GitHub Token"
        assert credential.type == "bearer"
        assert credential.service == "github"
        # Value should not be exposed
        assert not hasattr(credential, "value") or credential.model_dump().get("value") is None

    def test_list_credentials(self, manager):
        """Test listing credentials."""
        manager.create_credential(
            CredentialCreateInput(name="Cred1", type="bearer", service="s1", value="xxx")
        )
        manager.create_credential(
            CredentialCreateInput(name="Cred2", type="api_key", service="s2", value="yyy")
        )

        credentials = manager.list_credentials()
        assert len(credentials) == 2

    def test_get_credential_value(self, manager):
        """Test retrieving credential value (internal use)."""
        created = manager.create_credential(
            CredentialCreateInput(name="Test", type="bearer", service="test", value="secret123")
        )

        # Internal method to retrieve decrypted value
        value = manager.get_credential_value(created.id)
        assert value == "secret123"

    def test_delete_credential(self, manager):
        """Test deleting a credential."""
        created = manager.create_credential(
            CredentialCreateInput(name="ToDelete", type="bearer", service="x", value="y")
        )

        result = manager.delete_credential(created.id)
        assert result is True

        fetched = manager.get_credential(created.id)
        assert fetched is None

    def test_delete_credential_updates_servers(self, manager):
        """Test that deleting credential updates referencing servers."""
        # Create credential
        cred = manager.create_credential(
            CredentialCreateInput(name="Token", type="bearer", service="github", value="xxx")
        )

        # Create server with this credential
        server = manager.create_server(
            MCPServerCreateInput(
                name="GitHub",
                url="https://mcp.github.com",
                type="sse",
                authType="bearer",
                credentialId=cred.id,
            )
        )

        # Delete credential
        manager.delete_credential(cred.id)

        # Server should be updated
        updated_server = manager.get_server(server.id)
        assert updated_server.credentialId is None
        assert updated_server.status == "not_configured"


class TestGetCredential:
    """Tests for get_credential unit method."""

    def test_get_credential_returns_metadata(self, manager):
        """Test get_credential returns credential metadata without value."""
        created = manager.create_credential(
            CredentialCreateInput(
                name="GitHub Token",
                type="bearer",
                service="github",
                value="ghp_secret_value",
            )
        )

        credential = manager.get_credential(created.id)
        assert credential is not None
        assert credential.id == created.id
        assert credential.name == "GitHub Token"
        assert credential.type == "bearer"
        assert credential.service == "github"
        assert credential.createdAt is not None
        # Value must never be exposed on the Credential model
        assert "value" not in credential.model_dump()

    def test_get_credential_nonexistent_returns_none(self, manager):
        """Test get_credential with non-existent ID returns None."""
        result = manager.get_credential("does-not-exist-id")
        assert result is None


class TestEncryptionDecryption:
    """Tests for _encrypt_value and _decrypt_value edge cases."""

    def test_encrypt_produces_different_output_each_call(self, manager):
        """Test _encrypt_value produces different output due to random salt."""
        value = "same-secret-value"
        encrypted1 = manager._encrypt_value(value)
        encrypted2 = manager._encrypt_value(value)

        # Random salt means same plaintext produces different ciphertext
        assert encrypted1 != encrypted2

    def test_encrypt_decrypt_roundtrip(self, manager):
        """Test encrypt then decrypt returns original value."""
        original = "my-api-key-12345!@#$%"
        encrypted = manager._encrypt_value(original)
        decrypted = manager._decrypt_value(encrypted)

        assert decrypted == original

    def test_encrypt_decrypt_roundtrip_unicode(self, manager):
        """Test roundtrip with unicode characters."""
        original = "token-with-unicode-\u00e9\u00e8\u00ea"
        encrypted = manager._encrypt_value(original)
        decrypted = manager._decrypt_value(encrypted)

        assert decrypted == original

    def test_encrypt_decrypt_roundtrip_empty_string(self, manager):
        """Test roundtrip with empty string."""
        encrypted = manager._encrypt_value("")
        decrypted = manager._decrypt_value(encrypted)

        assert decrypted == ""

    def test_decrypt_corrupted_base64(self, manager):
        """Test _decrypt_value with corrupted/non-base64 data returns empty string."""
        result = manager._decrypt_value("not-valid-base64!!!")
        assert result == ""

    def test_decrypt_truncated_data(self, manager):
        """Test _decrypt_value with data shorter than 16-byte salt."""
        # Encode only 8 bytes (less than the 16-byte salt)
        short_data = base64.b64encode(b"short").decode("ascii")
        result = manager._decrypt_value(short_data)
        # Should return empty string after removing salt, or handle gracefully
        assert isinstance(result, str)

    def test_decrypt_malformed_utf8(self, manager):
        """Test _decrypt_value with invalid UTF-8 after salt removal."""
        # 16 bytes salt + invalid UTF-8 bytes
        bad_payload = b"\x00" * 16 + b"\xff\xfe\xfd"
        encoded = base64.b64encode(bad_payload).decode("ascii")
        result = manager._decrypt_value(encoded)
        # Exception handler returns empty string
        assert result == ""

    def test_get_credential_value_nonexistent(self, manager):
        """Test get_credential_value with non-existent ID returns None."""
        result = manager.get_credential_value("nonexistent-id")
        assert result is None


class TestRowToServerMalformedJSON:
    """Tests for _row_to_server with malformed JSON in tools/resources."""

    def _make_row(self, **overrides):
        """Helper to build a minimal valid row dict."""
        row = {
            "id": "test-id",
            "name": "Test",
            "url": "https://test.com",
            "type": "sse",
            "status": "disconnected",
            "description": "",
            "auth_type": "none",
            "credential_id": None,
            "tools": "[]",
            "resources": "[]",
            "last_connected": None,
            "error": None,
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }
        row.update(overrides)
        return row

    def test_malformed_tools_json(self, manager):
        """Test _row_to_server with malformed JSON in tools column."""
        row = self._make_row(tools="{bad json!!!")
        server = manager._row_to_server(row)

        assert server.id == "test-id"
        assert server.tools == []  # Falls back to empty list

    def test_malformed_resources_json(self, manager):
        """Test _row_to_server with malformed JSON in resources column."""
        row = self._make_row(resources="not-json")
        server = manager._row_to_server(row)

        assert server.resources == []  # Falls back to empty list

    def test_null_tools_and_resources(self, manager):
        """Test _row_to_server with null tools and resources."""
        row = self._make_row(tools=None, resources=None)
        server = manager._row_to_server(row)

        assert server.tools == []
        assert server.resources == []

    def test_empty_string_tools_and_resources(self, manager):
        """Test _row_to_server with empty string tools and resources."""
        row = self._make_row(tools="", resources="")
        server = manager._row_to_server(row)

        assert server.tools == []
        assert server.resources == []

    def test_valid_tools_json(self, manager):
        """Test _row_to_server correctly parses valid tools JSON."""
        tools_data = [{"name": "read_file", "description": "Read a file", "inputSchema": {}}]
        row = self._make_row(tools=json.dumps(tools_data))
        server = manager._row_to_server(row)

        assert len(server.tools) == 1
        assert server.tools[0].name == "read_file"

    def test_both_tools_and_resources_malformed(self, manager):
        """Test _row_to_server when both tools and resources are malformed."""
        row = self._make_row(tools="[invalid", resources="{also bad")
        server = manager._row_to_server(row)

        # Both fallback to empty
        assert server.tools == []
        assert server.resources == []


class TestSimulateToolDiscovery:
    """Tests for _simulate_tool_discovery for each server type."""

    def _create_server_with_name(self, manager, name):
        """Helper to create a server and return it."""
        server = manager.create_server(
            MCPServerCreateInput(
                name=name,
                url=f"https://{name.lower()}.example.com",
                type="sse",
                authType="none",
            )
        )
        return manager.get_server(server.id)

    def test_github_server_tools(self, manager):
        """Test tool discovery for GitHub server type."""
        server = self._create_server_with_name(manager, "GitHub MCP")
        tools = manager._simulate_tool_discovery(server)

        assert len(tools) == 3
        tool_names = [t.name for t in tools]
        assert "list_repos" in tool_names
        assert "create_pr" in tool_names
        assert "list_issues" in tool_names

        # Verify create_pr has required fields in schema
        pr_tool = next(t for t in tools if t.name == "create_pr")
        assert "required" in pr_tool.inputSchema
        assert "repo" in pr_tool.inputSchema["required"]
        assert "title" in pr_tool.inputSchema["required"]

    def test_filesystem_server_tools(self, manager):
        """Test tool discovery for filesystem server type."""
        server = self._create_server_with_name(manager, "Filesystem Local")
        tools = manager._simulate_tool_discovery(server)

        assert len(tools) == 3
        tool_names = [t.name for t in tools]
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "list_directory" in tool_names

    def test_notion_server_tools(self, manager):
        """Test tool discovery for Notion server type."""
        server = self._create_server_with_name(manager, "Notion Workspace")
        tools = manager._simulate_tool_discovery(server)

        assert len(tools) == 2
        tool_names = [t.name for t in tools]
        assert "search_pages" in tool_names
        assert "get_page" in tool_names

    def test_slack_server_tools(self, manager):
        """Test tool discovery for Slack server type."""
        server = self._create_server_with_name(manager, "Slack Integration")
        tools = manager._simulate_tool_discovery(server)

        assert len(tools) == 2
        tool_names = [t.name for t in tools]
        assert "send_message" in tool_names
        assert "list_channels" in tool_names

    def test_unknown_server_returns_empty_tools(self, manager):
        """Test tool discovery for unknown/custom server returns empty list."""
        server = self._create_server_with_name(manager, "Custom MCP Server")
        tools = manager._simulate_tool_discovery(server)

        assert tools == []

    def test_case_insensitive_matching(self, manager):
        """Test that server name matching is case-insensitive."""
        server = self._create_server_with_name(manager, "GITHUB Enterprise")
        tools = manager._simulate_tool_discovery(server)

        assert len(tools) == 3  # Should still match github


class TestEdgeCases:
    """Tests for edge cases in MCP operations."""

    def test_create_server_with_all_optional_fields(self, manager):
        """Test create_server with all optional fields populated."""
        cred = manager.create_credential(
            CredentialCreateInput(name="Token", type="bearer", service="github", value="xxx")
        )

        data = MCPServerCreateInput(
            name="Full Server",
            url="https://full.example.com/mcp",
            type="websocket",
            authType="bearer",
            credentialId=cred.id,
            description="A fully configured MCP server",
        )
        server = manager.create_server(data)

        assert server is not None
        assert server.name == "Full Server"
        assert server.type == "websocket"
        assert server.authType == "bearer"
        assert server.credentialId == cred.id
        assert server.description == "A fully configured MCP server"
        # Has credential, so status should be disconnected (not not_configured)
        assert server.status == "disconnected"

    def test_update_server_empty_update(self, manager):
        """Test update_server with no changes returns existing server."""
        created = manager.create_server(
            MCPServerCreateInput(
                name="Unchanged",
                url="https://test.com",
                type="sse",
                authType="none",
            )
        )

        # Empty update - all fields None
        result = manager.update_server(created.id, MCPServerUpdateInput())

        assert result is not None
        assert result.id == created.id
        assert result.name == "Unchanged"

    def test_update_server_not_found(self, manager):
        """Test update_server with non-existent ID returns None."""
        result = manager.update_server("nonexistent-id", MCPServerUpdateInput(name="New"))
        assert result is None

    def test_update_server_all_fields(self, manager):
        """Test update_server with every field changed."""
        created = manager.create_server(
            MCPServerCreateInput(
                name="Original",
                url="https://old.com",
                type="sse",
                authType="none",
            )
        )

        updated = manager.update_server(
            created.id,
            MCPServerUpdateInput(
                name="New Name",
                url="https://new.com",
                type="websocket",
                authType="api_key",
                credentialId="",  # Empty string to clear
                description="Updated desc",
            ),
        )

        assert updated.name == "New Name"
        assert updated.url == "https://new.com"
        assert updated.type == "websocket"
        assert updated.authType == "api_key"
        assert updated.description == "Updated desc"

    def test_list_servers_empty(self, manager):
        """Test list_servers when no servers exist."""
        servers = manager.list_servers()
        assert servers == []

    def test_list_credentials_empty(self, manager):
        """Test list_credentials when no credentials exist."""
        credentials = manager.list_credentials()
        assert credentials == []

    def test_delete_credential_not_found(self, manager):
        """Test deleting a non-existent credential returns False."""
        result = manager.delete_credential("nonexistent-id")
        assert result is False

    def test_delete_credential_referenced_by_multiple_servers(self, manager):
        """Test deleting credential updates all referencing servers."""
        cred = manager.create_credential(
            CredentialCreateInput(
                name="Shared Token", type="bearer", service="test", value="shared"
            )
        )

        server1 = manager.create_server(
            MCPServerCreateInput(
                name="Server A",
                url="https://a.com",
                type="sse",
                authType="bearer",
                credentialId=cred.id,
            )
        )
        server2 = manager.create_server(
            MCPServerCreateInput(
                name="Server B",
                url="https://b.com",
                type="sse",
                authType="bearer",
                credentialId=cred.id,
            )
        )

        # Delete the shared credential
        manager.delete_credential(cred.id)

        # Both servers should lose their credential reference
        s1 = manager.get_server(server1.id)
        s2 = manager.get_server(server2.id)
        assert s1.credentialId is None
        assert s1.status == "not_configured"
        assert s2.credentialId is None
        assert s2.status == "not_configured"

    def test_get_tools_for_nonexistent_server(self, manager):
        """Test get_tools returns empty list for non-existent server."""
        tools = manager.get_tools("nonexistent-id")
        assert tools == []

    def test_connection_test_updates_server_status(self, manager):
        """Test that successful connection test updates server to connected."""
        server = manager.create_server(
            MCPServerCreateInput(
                name="Filesystem Test",
                url="stdio://mcp-filesystem",
                type="stdio",
                authType="none",
            )
        )

        result = manager.test_connection(server.id)
        assert result.success is True

        # Verify server status updated in DB
        updated = manager.get_server(server.id)
        assert updated.status == MCPServerStatus.CONNECTED.value
        assert updated.lastConnected is not None
        assert updated.error is None

    def test_parse_datetime_invalid_value(self, manager):
        """Test _parse_datetime with invalid datetime string."""
        result = manager._parse_datetime("not-a-date")
        assert result is None

    def test_parse_datetime_none(self, manager):
        """Test _parse_datetime with None."""
        result = manager._parse_datetime(None)
        assert result is None

    def test_parse_datetime_with_z_suffix(self, manager):
        """Test _parse_datetime handles Z timezone suffix."""
        result = manager._parse_datetime("2024-01-15T10:30:00Z")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15


class TestBuildAuthHeaders:
    """Tests for _build_auth_headers."""

    def _create_server_with_auth(self, manager, auth_type, credential_id=None):
        """Helper to create a server with auth config."""
        server = manager.create_server(
            MCPServerCreateInput(
                name="Auth Test",
                url="https://example.com/mcp",
                type="sse",
                authType=auth_type,
                credentialId=credential_id,
            )
        )
        return manager.get_server(server.id)

    def test_no_auth_returns_none(self, manager):
        """No auth type returns None."""
        server = self._create_server_with_auth(manager, "none")
        assert manager._build_auth_headers(server) is None

    def test_bearer_auth(self, manager):
        """Bearer auth returns Authorization header."""
        cred = manager.create_credential(
            CredentialCreateInput(name="Token", type="bearer", service="test", value="my-secret")
        )
        server = self._create_server_with_auth(manager, "bearer", cred.id)
        headers = manager._build_auth_headers(server)
        assert headers == {"Authorization": "Bearer my-secret"}

    def test_api_key_auth(self, manager):
        """API key auth returns X-API-Key header."""
        cred = manager.create_credential(
            CredentialCreateInput(name="Key", type="api_key", service="test", value="key-123")
        )
        server = self._create_server_with_auth(manager, "api_key", cred.id)
        headers = manager._build_auth_headers(server)
        assert headers == {"X-API-Key": "key-123"}

    def test_missing_credential_id(self, manager):
        """Auth type set but no credential ID returns None."""
        server = self._create_server_with_auth(manager, "bearer", None)
        assert manager._build_auth_headers(server) is None

    def test_unsupported_auth_type(self, manager):
        """Unsupported auth type returns None."""
        cred = manager.create_credential(
            CredentialCreateInput(name="OAuth", type="bearer", service="test", value="tok")
        )
        server = self._create_server_with_auth(manager, "oauth", cred.id)
        headers = manager._build_auth_headers(server)
        assert headers is None


class TestDiscoverServer:
    """Tests for _discover_server with real/fallback paths."""

    def _create_test_server(self, manager):
        """Helper to create a test server."""
        server = manager.create_server(
            MCPServerCreateInput(
                name="Test Server",
                url="https://example.com/mcp",
                type="sse",
                authType="none",
            )
        )
        return manager.get_server(server.id)

    def test_fallback_to_simulation_when_sdk_missing(self, manager):
        """Should use simulated discovery when SDK is not installed."""
        server = self._create_test_server(manager)
        with patch("animus_forge.mcp.client.HAS_MCP_SDK", False):
            tools, resources = manager._discover_server(server, None)
        # Unknown server type → empty simulated tools
        assert tools == []
        assert resources == []

    def test_real_discovery_success(self, manager):
        """Should use real discovery when SDK is available."""
        server = self._create_test_server(manager)
        mock_result = {
            "tools": [{"name": "read", "description": "Read file", "inputSchema": {}}],
            "resources": [
                {
                    "uri": "file:///tmp",
                    "name": "tmp",
                    "mimeType": None,
                    "description": None,
                }
            ],
        }
        with patch("animus_forge.mcp.client.HAS_MCP_SDK", True):
            with patch("animus_forge.mcp.client.discover_tools", return_value=mock_result):
                tools, resources = manager._discover_server(server, None)

        assert len(tools) == 1
        assert tools[0].name == "read"
        assert len(resources) == 1
        assert resources[0].name == "tmp"

    def test_real_discovery_failure_falls_back(self, manager):
        """Should fall back to simulation when real discovery fails."""
        server = manager.create_server(
            MCPServerCreateInput(
                name="GitHub Fallback",
                url="https://github.example.com",
                type="sse",
                authType="none",
            )
        )
        server = manager.get_server(server.id)

        from animus_forge.mcp.client import MCPClientError

        with patch("animus_forge.mcp.client.HAS_MCP_SDK", True):
            with patch(
                "animus_forge.mcp.client.discover_tools",
                side_effect=MCPClientError("Connection refused"),
            ):
                tools, resources = manager._discover_server(server, None)

        # Falls back to simulation — "github" in name → simulated tools
        assert len(tools) == 3
        assert resources == []

    def test_test_connection_uses_discover_server(self, manager):
        """test_connection() should use _discover_server internally."""
        server = manager.create_server(
            MCPServerCreateInput(
                name="Filesystem SDK",
                url="stdio://mcp-filesystem",
                type="stdio",
                authType="none",
            )
        )

        mock_result = {
            "tools": [{"name": "read_file", "description": "Read", "inputSchema": {}}],
            "resources": [],
        }
        with patch("animus_forge.mcp.client.HAS_MCP_SDK", True):
            with patch("animus_forge.mcp.client.discover_tools", return_value=mock_result):
                result = manager.test_connection(server.id)

        assert result.success is True
        assert len(result.tools) == 1
        assert result.tools[0].name == "read_file"
