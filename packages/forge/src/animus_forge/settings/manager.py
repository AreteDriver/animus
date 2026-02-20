"""Settings manager for user preferences and API keys."""

import base64
import logging
from datetime import datetime

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from animus_forge.state.backends import DatabaseBackend

from .models import (
    APIKeyCreate,
    APIKeyInfo,
    APIKeyStatus,
    NotificationSettings,
    UserPreferences,
    UserPreferencesUpdate,
)

logger = logging.getLogger(__name__)


class SettingsManager:
    """Manages user preferences and API key storage."""

    def __init__(self, backend: DatabaseBackend):
        """Initialize settings manager.

        Args:
            backend: Database backend for persistence
        """
        self._backend = backend
        self._encryption_key = self._derive_encryption_key()

    def _derive_encryption_key(self) -> bytes:
        """Derive encryption key from secret.

        Uses SETTINGS_ENCRYPTION_KEY env var, or generates from JWT_SECRET.
        """
        # Try dedicated encryption key first
        from animus_forge.config.settings import get_settings

        settings = get_settings()
        key = settings.settings_encryption_key
        if key:
            # If it's already a valid Fernet key, use it directly
            if len(key) == 44:
                try:
                    Fernet(key.encode())
                    return key.encode()
                except Exception:
                    pass  # Graceful degradation: invalid Fernet key falls through to derived key

        # Fall back to deriving from JWT_SECRET or a default
        secret = settings.jwt_secret or "gorgon-default-secret-change-me"
        salt = b"gorgon-settings-salt"

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(secret.encode()))
        return key

    def _encrypt(self, plaintext: str) -> str:
        """Encrypt a string."""
        f = Fernet(self._encryption_key)
        return f.encrypt(plaintext.encode()).decode()

    def _decrypt(self, ciphertext: str) -> str:
        """Decrypt a string."""
        f = Fernet(self._encryption_key)
        return f.decrypt(ciphertext.encode()).decode()

    def _mask_key(self, key: str) -> str:
        """Create a masked version of an API key.

        Shows first 7 chars and last 4 chars, e.g., "sk-proj...xyz1"
        """
        if len(key) <= 12:
            return key[:3] + "..." + key[-2:]
        return key[:7] + "..." + key[-4:]

    # =========================================================================
    # User Preferences
    # =========================================================================

    def get_preferences(self, user_id: str) -> UserPreferences:
        """Get user preferences, creating defaults if not exists.

        Args:
            user_id: User identifier

        Returns:
            User preferences
        """
        row = self._backend.fetchone(
            "SELECT * FROM user_preferences WHERE user_id = ?",
            (user_id,),
        )

        if not row:
            # Create default preferences
            prefs = UserPreferences(user_id=user_id)
            self._save_preferences(prefs)
            return prefs

        return UserPreferences(
            user_id=row["user_id"],
            theme=row["theme"],
            compact_view=bool(row["compact_view"]),
            show_costs=bool(row["show_costs"]),
            default_page_size=row["default_page_size"],
            notifications=NotificationSettings(
                execution_complete=bool(row["notify_execution_complete"]),
                execution_failed=bool(row["notify_execution_failed"]),
                budget_alert=bool(row["notify_budget_alert"]),
            ),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )

    def update_preferences(self, user_id: str, update: UserPreferencesUpdate) -> UserPreferences:
        """Update user preferences.

        Args:
            user_id: User identifier
            update: Partial preferences update

        Returns:
            Updated preferences
        """
        # Get existing preferences (creates if not exists)
        current = self.get_preferences(user_id)

        # Apply updates
        update_dict = update.model_dump(exclude_unset=True)
        if "theme" in update_dict:
            current.theme = update_dict["theme"]
        if "compact_view" in update_dict:
            current.compact_view = update_dict["compact_view"]
        if "show_costs" in update_dict:
            current.show_costs = update_dict["show_costs"]
        if "default_page_size" in update_dict:
            current.default_page_size = update_dict["default_page_size"]
        if "notifications" in update_dict and update_dict["notifications"]:
            current.notifications = NotificationSettings(**update_dict["notifications"])

        current.updated_at = datetime.now()
        self._save_preferences(current)
        return current

    def _save_preferences(self, prefs: UserPreferences) -> None:
        """Save preferences to database."""
        now = datetime.now().isoformat()
        self._backend.execute(
            """
            INSERT INTO user_preferences (
                user_id, theme, compact_view, show_costs, default_page_size,
                notify_execution_complete, notify_execution_failed, notify_budget_alert,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                theme = excluded.theme,
                compact_view = excluded.compact_view,
                show_costs = excluded.show_costs,
                default_page_size = excluded.default_page_size,
                notify_execution_complete = excluded.notify_execution_complete,
                notify_execution_failed = excluded.notify_execution_failed,
                notify_budget_alert = excluded.notify_budget_alert,
                updated_at = excluded.updated_at
            """,
            (
                prefs.user_id,
                prefs.theme,
                int(prefs.compact_view),
                int(prefs.show_costs),
                prefs.default_page_size,
                int(prefs.notifications.execution_complete),
                int(prefs.notifications.execution_failed),
                int(prefs.notifications.budget_alert),
                prefs.created_at.isoformat() if prefs.created_at else now,
                now,
            ),
        )
        # Commit the transaction
        with self._backend.transaction():
            pass

    # =========================================================================
    # API Keys
    # =========================================================================

    def get_api_keys(self, user_id: str) -> list[APIKeyInfo]:
        """Get all API keys for a user (metadata only, no raw keys).

        Args:
            user_id: User identifier

        Returns:
            List of API key info (without raw keys)
        """
        rows = self._backend.fetchall(
            "SELECT id, provider, key_prefix, created_at, updated_at "
            "FROM user_api_keys WHERE user_id = ?",
            (user_id,),
        )
        return [
            APIKeyInfo(
                id=row["id"],
                provider=row["provider"],
                key_prefix=row["key_prefix"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def get_api_key_status(self, user_id: str) -> APIKeyStatus:
        """Get status of which API keys are configured.

        Args:
            user_id: User identifier

        Returns:
            Status object indicating which providers have keys
        """
        rows = self._backend.fetchall(
            "SELECT provider FROM user_api_keys WHERE user_id = ?",
            (user_id,),
        )
        providers = {row["provider"] for row in rows}
        return APIKeyStatus(
            openai="openai" in providers,
            anthropic="anthropic" in providers,
            github="github" in providers,
        )

    def set_api_key(self, user_id: str, key_create: APIKeyCreate) -> APIKeyInfo:
        """Set or update an API key.

        Args:
            user_id: User identifier
            key_create: Provider and raw key

        Returns:
            API key info (without raw key)
        """
        encrypted = self._encrypt(key_create.key)
        key_prefix = self._mask_key(key_create.key)
        now = datetime.now().isoformat()

        self._backend.execute(
            """
            INSERT INTO user_api_keys (
                user_id, provider, encrypted_key, key_prefix, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, provider) DO UPDATE SET
                encrypted_key = excluded.encrypted_key,
                key_prefix = excluded.key_prefix,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                key_create.provider,
                encrypted,
                key_prefix,
                now,
                now,
            ),
        )
        with self._backend.transaction():
            pass

        # Fetch the saved record
        row = self._backend.fetchone(
            "SELECT id, provider, key_prefix, created_at, updated_at "
            "FROM user_api_keys WHERE user_id = ? AND provider = ?",
            (user_id, key_create.provider),
        )
        return APIKeyInfo(
            id=row["id"],
            provider=row["provider"],
            key_prefix=row["key_prefix"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def delete_api_key(self, user_id: str, provider: str) -> bool:
        """Delete an API key.

        Args:
            user_id: User identifier
            provider: Provider name (openai, anthropic, github)

        Returns:
            True if deleted, False if not found
        """
        cursor = self._backend.execute(
            "DELETE FROM user_api_keys WHERE user_id = ? AND provider = ?",
            (user_id, provider),
        )
        with self._backend.transaction():
            pass
        return cursor.rowcount > 0

    def get_decrypted_api_key(self, user_id: str, provider: str) -> str | None:
        """Get decrypted API key for internal use (not exposed via API).

        Args:
            user_id: User identifier
            provider: Provider name

        Returns:
            Decrypted API key or None if not found
        """
        row = self._backend.fetchone(
            "SELECT encrypted_key FROM user_api_keys WHERE user_id = ? AND provider = ?",
            (user_id, provider),
        )
        if not row:
            return None
        return self._decrypt(row["encrypted_key"])
