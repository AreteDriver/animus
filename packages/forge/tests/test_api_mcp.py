"""Tests for MCP API endpoints."""

import os
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from animus_forge.auth import create_access_token
from animus_forge.mcp import MCPConnectorManager
from animus_forge.state.backends import SQLiteBackend


@pytest.fixture
def backend():
    """Create a temporary SQLite backend with MCP tables."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        backend = SQLiteBackend(db_path=db_path)

        # Create required tables
        schema = """
        -- Schema migrations tracking table
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            description TEXT
        );

        -- MCP Servers table
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

        -- Credentials table
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
        yield backend
        backend.close()


@pytest.fixture
def mcp_manager(backend):
    """Create an MCP manager with test backend."""
    return MCPConnectorManager(backend)


@pytest.fixture
def client(backend, mcp_manager, monkeypatch):
    """Create a test client with MCP manager injected."""
    # Enable demo auth
    monkeypatch.setenv("ALLOW_DEMO_AUTH", "true")

    # Clear settings cache
    from animus_forge.config.settings import get_settings

    get_settings.cache_clear()

    # Patch the mcp_manager global variable
    with patch("animus_forge.api.get_database", return_value=backend):
        with patch("animus_forge.api.run_migrations", return_value=[]):
            import animus_forge.api_state as api_state
            from animus_forge.api import app
            from animus_forge.api_state import limiter
            from animus_forge.security.brute_force import get_brute_force_protection

            # Disable rate limiting for tests
            limiter.enabled = False

            # Reset brute force protection state for tests
            protection = get_brute_force_protection()
            protection._attempts.clear()
            protection._total_blocked = 0
            protection._total_allowed = 0

            test_client = TestClient(app)

            # Reset shutting_down flag — previous test files' TestClient
            # lifespan shutdown sets this to True on the shared api_state
            api_state._app_state["shutting_down"] = False

            # Set mcp_manager AFTER TestClient creation so lifespan
            # doesn't overwrite it with a manager using the wrong backend
            original_mcp_manager = api_state.mcp_manager
            api_state.mcp_manager = mcp_manager

            yield test_client

            # Restore original
            api_state.mcp_manager = original_mcp_manager


@pytest.fixture
def auth_headers():
    """Create auth headers with valid token."""
    token = create_access_token("test-user")
    return {"Authorization": f"Bearer {token}"}


class TestMCPEndpoints:
    """Tests for MCP server endpoints."""

    def test_list_servers_requires_auth(self, client):
        """Test that listing servers requires authentication."""
        response = client.get("/v1/mcp/servers")
        assert response.status_code == 401

    def test_list_servers_empty(self, client, auth_headers):
        """Test listing servers when none exist."""
        response = client.get("/v1/mcp/servers", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_create_server(self, client, auth_headers):
        """Test creating an MCP server."""
        server_data = {
            "name": "Test Server",
            "url": "https://test.example.com/mcp",
            "type": "sse",
            "authType": "none",
            "description": "Test description",
        }
        response = client.post("/v1/mcp/servers", json=server_data, headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "Test Server"
        assert data["url"] == "https://test.example.com/mcp"
        assert data["type"] == "sse"
        assert data["status"] == "disconnected"  # No auth required

    def test_create_server_with_auth_requirement(self, client, auth_headers):
        """Test creating server that requires auth."""
        server_data = {
            "name": "GitHub",
            "url": "https://mcp.github.com/sse",
            "type": "sse",
            "authType": "bearer",
        }
        response = client.post("/v1/mcp/servers", json=server_data, headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "not_configured"  # Needs credentials

    def test_get_server(self, client, auth_headers):
        """Test getting a specific server."""
        # Create server first
        server_data = {
            "name": "Get Test",
            "url": "https://test.com",
            "type": "sse",
            "authType": "none",
        }
        create_response = client.post("/v1/mcp/servers", json=server_data, headers=auth_headers)
        server_id = create_response.json()["id"]

        # Get server
        response = client.get(f"/v1/mcp/servers/{server_id}", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["name"] == "Get Test"

    def test_get_server_not_found(self, client, auth_headers):
        """Test getting non-existent server."""
        response = client.get("/v1/mcp/servers/nonexistent", headers=auth_headers)
        assert response.status_code == 404

    def test_update_server(self, client, auth_headers):
        """Test updating a server."""
        # Create server
        server_data = {
            "name": "Original",
            "url": "https://test.com",
            "type": "sse",
            "authType": "none",
        }
        create_response = client.post("/v1/mcp/servers", json=server_data, headers=auth_headers)
        server_id = create_response.json()["id"]

        # Update server
        update_data = {"name": "Updated", "description": "New description"}
        response = client.put(
            f"/v1/mcp/servers/{server_id}", json=update_data, headers=auth_headers
        )
        assert response.status_code == 200
        assert response.json()["name"] == "Updated"
        assert response.json()["description"] == "New description"

    def test_delete_server(self, client, auth_headers):
        """Test deleting a server."""
        # Create server
        server_data = {
            "name": "To Delete",
            "url": "https://test.com",
            "type": "sse",
            "authType": "none",
        }
        create_response = client.post("/v1/mcp/servers", json=server_data, headers=auth_headers)
        server_id = create_response.json()["id"]

        # Delete server
        response = client.delete(f"/v1/mcp/servers/{server_id}", headers=auth_headers)
        assert response.status_code == 200

        # Verify deleted
        get_response = client.get(f"/v1/mcp/servers/{server_id}", headers=auth_headers)
        assert get_response.status_code == 404

    def test_test_connection(self, client, auth_headers):
        """Test connection testing."""
        # Create server that doesn't require auth
        server_data = {
            "name": "Filesystem",
            "url": "stdio://mcp-filesystem",
            "type": "stdio",
            "authType": "none",
        }
        create_response = client.post("/v1/mcp/servers", json=server_data, headers=auth_headers)
        server_id = create_response.json()["id"]

        # Test connection
        response = client.post(f"/v1/mcp/servers/{server_id}/test", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "tools" in data

    def test_get_tools(self, client, auth_headers):
        """Test getting server tools."""
        # Create and connect server
        server_data = {
            "name": "Filesystem",
            "url": "stdio://mcp-filesystem",
            "type": "stdio",
            "authType": "none",
        }
        create_response = client.post("/v1/mcp/servers", json=server_data, headers=auth_headers)
        server_id = create_response.json()["id"]

        # Test connection to populate tools
        client.post(f"/v1/mcp/servers/{server_id}/test", headers=auth_headers)

        # Get tools
        response = client.get(f"/v1/mcp/servers/{server_id}/tools", headers=auth_headers)
        assert response.status_code == 200
        tools = response.json()
        assert isinstance(tools, list)
        # Should have tools after connection test
        assert len(tools) > 0

    def test_discover_tools(self, client, auth_headers):
        """Test live tool discovery endpoint."""
        # Create server
        server_data = {
            "name": "Filesystem",
            "url": "stdio://mcp-filesystem",
            "type": "stdio",
            "authType": "none",
        }
        create_response = client.post("/v1/mcp/servers", json=server_data, headers=auth_headers)
        server_id = create_response.json()["id"]

        # Discover tools (uses simulated fallback since SDK not installed)
        response = client.post(f"/v1/mcp/servers/{server_id}/discover", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert "resources" in data
        assert isinstance(data["tools"], list)

    def test_discover_tools_not_found(self, client, auth_headers):
        """Test discover endpoint with nonexistent server."""
        response = client.post("/v1/mcp/servers/nonexistent/discover", headers=auth_headers)
        assert response.status_code == 404


class TestCredentialsEndpoints:
    """Tests for credential endpoints."""

    def test_list_credentials_empty(self, client, auth_headers):
        """Test listing credentials when none exist."""
        response = client.get("/v1/credentials", headers=auth_headers)
        assert response.status_code == 200
        assert response.json() == []

    def test_create_credential(self, client, auth_headers):
        """Test creating a credential."""
        cred_data = {
            "name": "Test Token",
            "type": "bearer",
            "service": "test",
            "value": "secret123",
        }
        response = client.post("/v1/credentials", json=cred_data, headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "Test Token"
        assert data["type"] == "bearer"
        assert data["service"] == "test"
        # Value should not be exposed
        assert "value" not in data or data.get("value") is None

    def test_delete_credential(self, client, auth_headers):
        """Test deleting a credential."""
        # Create credential
        cred_data = {
            "name": "To Delete",
            "type": "bearer",
            "service": "test",
            "value": "secret",
        }
        create_response = client.post("/v1/credentials", json=cred_data, headers=auth_headers)
        cred_id = create_response.json()["id"]

        # Delete
        response = client.delete(f"/v1/credentials/{cred_id}", headers=auth_headers)
        assert response.status_code == 200

        # Verify deleted
        get_response = client.get(f"/v1/credentials/{cred_id}", headers=auth_headers)
        assert get_response.status_code == 404
