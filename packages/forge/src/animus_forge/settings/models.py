"""User settings models."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class NotificationSettings(BaseModel):
    """Notification preferences."""

    execution_complete: bool = True
    execution_failed: bool = True
    budget_alert: bool = True


class UserPreferences(BaseModel):
    """User preferences model."""

    user_id: str
    theme: Literal["light", "dark", "system"] = "system"
    compact_view: bool = False
    show_costs: bool = True
    default_page_size: int = Field(default=20, ge=10, le=100)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class UserPreferencesUpdate(BaseModel):
    """User preferences update request."""

    theme: Literal["light", "dark", "system"] | None = None
    compact_view: bool | None = None
    show_costs: bool | None = None
    default_page_size: int | None = Field(default=None, ge=10, le=100)
    notifications: NotificationSettings | None = None


class APIKeyInfo(BaseModel):
    """API key metadata (no raw key exposed)."""

    id: int
    provider: str
    key_prefix: str  # e.g., "sk-...abc" - first few and last few chars
    created_at: datetime
    updated_at: datetime


class APIKeyCreate(BaseModel):
    """Request to create/update an API key."""

    provider: Literal["openai", "anthropic", "github"]
    key: str = Field(..., min_length=1)


class APIKeyStatus(BaseModel):
    """Status of configured API keys."""

    openai: bool = False
    anthropic: bool = False
    github: bool = False
