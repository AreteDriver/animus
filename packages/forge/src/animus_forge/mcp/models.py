"""Pydantic models for MCP connectors."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class MCPServerStatus(str, Enum):
    """MCP server connection status."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    CONNECTING = "connecting"
    NOT_CONFIGURED = "not_configured"


class MCPServerType(str, Enum):
    """MCP server connection type."""

    SSE = "sse"
    STDIO = "stdio"
    WEBSOCKET = "websocket"


class MCPAuthType(str, Enum):
    """MCP authentication type."""

    NONE = "none"
    BEARER = "bearer"
    API_KEY = "api_key"
    OAUTH = "oauth"


class MCPTool(BaseModel):
    """MCP tool definition."""

    name: str
    description: str
    inputSchema: dict = Field(default_factory=dict)


class MCPResource(BaseModel):
    """MCP resource definition."""

    uri: str
    name: str
    mimeType: str | None = None
    description: str | None = None


class MCPServer(BaseModel):
    """MCP server registration."""

    model_config = ConfigDict(use_enum_values=True)

    id: str
    name: str
    url: str
    type: MCPServerType = MCPServerType.SSE
    status: MCPServerStatus = MCPServerStatus.NOT_CONFIGURED
    description: str | None = None
    authType: MCPAuthType = MCPAuthType.NONE
    credentialId: str | None = None
    tools: list[MCPTool] = Field(default_factory=list)
    resources: list[MCPResource] = Field(default_factory=list)
    lastConnected: datetime | None = None
    error: str | None = None
    createdAt: datetime = Field(default_factory=datetime.now)
    updatedAt: datetime = Field(default_factory=datetime.now)


class MCPServerCreateInput(BaseModel):
    """Input for creating an MCP server registration."""

    model_config = ConfigDict(use_enum_values=True)

    name: str
    url: str
    type: MCPServerType = MCPServerType.SSE
    authType: MCPAuthType = MCPAuthType.NONE
    credentialId: str | None = None
    description: str | None = None


class MCPServerUpdateInput(BaseModel):
    """Input for updating an MCP server registration."""

    model_config = ConfigDict(use_enum_values=True)

    name: str | None = None
    url: str | None = None
    type: MCPServerType | None = None
    authType: MCPAuthType | None = None
    credentialId: str | None = None
    description: str | None = None


class CredentialType(str, Enum):
    """Credential type."""

    BEARER = "bearer"
    API_KEY = "api_key"
    OAUTH = "oauth"


class Credential(BaseModel):
    """Stored credential (value never exposed)."""

    model_config = ConfigDict(use_enum_values=True)

    id: str
    name: str
    type: CredentialType
    service: str
    createdAt: datetime = Field(default_factory=datetime.now)
    lastUsed: datetime | None = None


class CredentialCreateInput(BaseModel):
    """Input for creating a credential."""

    model_config = ConfigDict(use_enum_values=True)

    name: str
    type: CredentialType
    service: str
    value: str  # The actual credential value (will be encrypted)


class MCPConnectionTestResult(BaseModel):
    """Result of testing an MCP connection."""

    success: bool
    error: str | None = None
    tools: list[MCPTool] = Field(default_factory=list)
    resources: list[MCPResource] = Field(default_factory=list)
