"""Plugin marketplace models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PluginCategory(str, Enum):
    """Plugin categories for marketplace organization."""

    INTEGRATION = "integration"
    DATA_TRANSFORM = "data_transform"
    MONITORING = "monitoring"
    SECURITY = "security"
    WORKFLOW = "workflow"
    AI_PROVIDER = "ai_provider"
    STORAGE = "storage"
    NOTIFICATION = "notification"
    ANALYTICS = "analytics"
    OTHER = "other"


class PluginSource(str, Enum):
    """Source types for plugin installation."""

    MARKETPLACE = "marketplace"
    LOCAL = "local"
    GITHUB = "github"
    PYPI = "pypi"
    URL = "url"


class PluginMetadata(BaseModel):
    """Metadata for a plugin package."""

    name: str = Field(..., description="Unique plugin name")
    version: str = Field(..., description="Semantic version")
    description: str = Field(default="", description="Plugin description")
    author: str = Field(default="", description="Plugin author")
    author_email: str | None = Field(default=None, description="Author email")
    license: str = Field(default="MIT", description="License type")
    repository_url: str | None = Field(default=None, description="Source repository")
    documentation_url: str | None = Field(default=None, description="Documentation URL")
    homepage_url: str | None = Field(default=None, description="Homepage URL")
    tags: list[str] = Field(default_factory=list, description="Plugin tags")
    keywords: list[str] = Field(default_factory=list, description="Search keywords")
    category: PluginCategory = Field(default=PluginCategory.OTHER, description="Plugin category")
    requirements: list[str] = Field(default_factory=list, description="Python dependencies")
    min_gorgon_version: str | None = Field(default=None, description="Minimum Gorgon version")
    entry_point: str = Field(default="Plugin", description="Plugin class name")
    provides_handlers: list[str] = Field(default_factory=list, description="Step handlers provided")
    provides_hooks: list[str] = Field(default_factory=list, description="Lifecycle hooks provided")


class PluginRelease(BaseModel):
    """A specific release version of a plugin."""

    id: str = Field(..., description="Release ID")
    plugin_name: str = Field(..., description="Plugin name")
    version: str = Field(..., description="Release version")
    released_at: datetime = Field(default_factory=datetime.now, description="Release date")
    download_url: str = Field(..., description="Download URL")
    checksum: str = Field(..., description="SHA256 checksum")
    signature: str | None = Field(default=None, description="GPG signature")
    changelog: str = Field(default="", description="Release notes")
    file_size: int = Field(default=0, description="File size in bytes")
    compatible_versions: list[str] = Field(
        default_factory=list, description="Compatible Gorgon versions"
    )
    metadata: PluginMetadata | None = Field(default=None, description="Full plugin metadata")


class PluginListing(BaseModel):
    """Marketplace listing for a plugin."""

    id: str = Field(..., description="Listing ID")
    name: str = Field(..., description="Plugin name")
    display_name: str = Field(..., description="Display name")
    description: str = Field(default="", description="Short description")
    long_description: str = Field(default="", description="Full description/README")
    author: str = Field(default="", description="Author name")
    category: PluginCategory = Field(default=PluginCategory.OTHER, description="Category")
    tags: list[str] = Field(default_factory=list, description="Tags")
    downloads: int = Field(default=0, description="Total downloads")
    rating: float = Field(default=0.0, ge=0.0, le=5.0, description="Average rating")
    review_count: int = Field(default=0, description="Number of reviews")
    latest_version: str = Field(..., description="Latest version")
    published_at: datetime = Field(default_factory=datetime.now, description="First published")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last updated")
    verified: bool = Field(default=False, description="Verified by Gorgon team")
    featured: bool = Field(default=False, description="Featured plugin")
    repository_url: str | None = Field(default=None, description="Source repository")
    releases: list[PluginRelease] = Field(default_factory=list, description="Available releases")


class PluginInstallation(BaseModel):
    """Tracks a locally installed plugin."""

    id: str = Field(..., description="Installation ID")
    plugin_name: str = Field(..., description="Plugin name")
    version: str = Field(..., description="Installed version")
    installed_at: datetime = Field(default_factory=datetime.now, description="Installation date")
    updated_at: datetime | None = Field(default=None, description="Last update date")
    enabled: bool = Field(default=True, description="Whether plugin is enabled")
    config: dict[str, Any] = Field(default_factory=dict, description="Plugin configuration")
    local_path: str = Field(..., description="Path to plugin files")
    source: PluginSource = Field(default=PluginSource.LOCAL, description="Installation source")
    source_url: str | None = Field(default=None, description="Source URL if remote")
    checksum: str | None = Field(default=None, description="Installed file checksum")
    auto_update: bool = Field(default=False, description="Auto-update enabled")


class PluginSearchResult(BaseModel):
    """Search results from marketplace."""

    query: str = Field(..., description="Search query")
    total: int = Field(default=0, description="Total matching plugins")
    page: int = Field(default=1, description="Current page")
    per_page: int = Field(default=20, description="Results per page")
    results: list[PluginListing] = Field(default_factory=list, description="Matching plugins")


class PluginInstallRequest(BaseModel):
    """Request to install a plugin."""

    name: str = Field(..., description="Plugin name")
    version: str | None = Field(default=None, description="Version (latest if None)")
    source: PluginSource = Field(
        default=PluginSource.MARKETPLACE, description="Installation source"
    )
    source_url: str | None = Field(default=None, description="URL for remote sources")
    config: dict[str, Any] = Field(default_factory=dict, description="Initial configuration")
    enable: bool = Field(default=True, description="Enable after install")
    auto_update: bool = Field(default=False, description="Enable auto-updates")


class PluginUpdateRequest(BaseModel):
    """Request to update a plugin."""

    name: str = Field(..., description="Plugin name")
    version: str | None = Field(default=None, description="Target version (latest if None)")
    config: dict[str, Any] | None = Field(default=None, description="Updated configuration")
