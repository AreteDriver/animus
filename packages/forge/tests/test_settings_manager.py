"""Comprehensive tests for settings/manager.py and settings/models.py."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from animus_forge.settings.manager import SettingsManager
from animus_forge.settings.models import (
    APIKeyCreate,
    APIKeyInfo,
    APIKeyStatus,
    NotificationSettings,
    UserPreferences,
    UserPreferencesUpdate,
)

# =============================================================================
# Model Tests — NotificationSettings
# =============================================================================


class TestNotificationSettings:
    """Tests for NotificationSettings model."""

    def test_defaults_all_true(self):
        n = NotificationSettings()
        assert n.execution_complete is True
        assert n.execution_failed is True
        assert n.budget_alert is True

    def test_custom_values(self):
        n = NotificationSettings(
            execution_complete=False, execution_failed=False, budget_alert=False
        )
        assert n.execution_complete is False
        assert n.execution_failed is False
        assert n.budget_alert is False

    def test_partial_override(self):
        n = NotificationSettings(budget_alert=False)
        assert n.execution_complete is True
        assert n.budget_alert is False

    def test_serialization(self):
        n = NotificationSettings()
        data = n.model_dump()
        assert data == {
            "execution_complete": True,
            "execution_failed": True,
            "budget_alert": True,
        }


# =============================================================================
# Model Tests — UserPreferences
# =============================================================================


class TestUserPreferences:
    """Tests for UserPreferences model."""

    def test_required_user_id(self):
        p = UserPreferences(user_id="user-1")
        assert p.user_id == "user-1"

    def test_missing_user_id_raises(self):
        with pytest.raises(ValidationError):
            UserPreferences()

    def test_default_theme(self):
        p = UserPreferences(user_id="u1")
        assert p.theme == "system"

    def test_valid_themes(self):
        for theme in ("light", "dark", "system"):
            p = UserPreferences(user_id="u1", theme=theme)
            assert p.theme == theme

    def test_invalid_theme(self):
        with pytest.raises(ValidationError):
            UserPreferences(user_id="u1", theme="neon")

    def test_default_compact_view(self):
        p = UserPreferences(user_id="u1")
        assert p.compact_view is False

    def test_default_show_costs(self):
        p = UserPreferences(user_id="u1")
        assert p.show_costs is True

    def test_default_page_size(self):
        p = UserPreferences(user_id="u1")
        assert p.default_page_size == 20

    def test_page_size_min(self):
        with pytest.raises(ValidationError):
            UserPreferences(user_id="u1", default_page_size=5)

    def test_page_size_max(self):
        with pytest.raises(ValidationError):
            UserPreferences(user_id="u1", default_page_size=200)

    def test_page_size_boundary_low(self):
        p = UserPreferences(user_id="u1", default_page_size=10)
        assert p.default_page_size == 10

    def test_page_size_boundary_high(self):
        p = UserPreferences(user_id="u1", default_page_size=100)
        assert p.default_page_size == 100

    def test_default_notifications(self):
        p = UserPreferences(user_id="u1")
        assert isinstance(p.notifications, NotificationSettings)
        assert p.notifications.execution_complete is True

    def test_custom_notifications(self):
        n = NotificationSettings(budget_alert=False)
        p = UserPreferences(user_id="u1", notifications=n)
        assert p.notifications.budget_alert is False

    def test_default_timestamps(self):
        p = UserPreferences(user_id="u1")
        assert p.created_at is None
        assert p.updated_at is None

    def test_custom_timestamps(self):
        now = datetime.now()
        p = UserPreferences(user_id="u1", created_at=now, updated_at=now)
        assert p.created_at == now
        assert p.updated_at == now

    def test_serialization_roundtrip(self):
        p = UserPreferences(user_id="u1", theme="dark", compact_view=True)
        data = p.model_dump()
        p2 = UserPreferences(**data)
        assert p2.user_id == "u1"
        assert p2.theme == "dark"
        assert p2.compact_view is True


# =============================================================================
# Model Tests — UserPreferencesUpdate
# =============================================================================


class TestUserPreferencesUpdate:
    """Tests for UserPreferencesUpdate model."""

    def test_all_none_by_default(self):
        u = UserPreferencesUpdate()
        assert u.theme is None
        assert u.compact_view is None
        assert u.show_costs is None
        assert u.default_page_size is None
        assert u.notifications is None

    def test_partial_update(self):
        u = UserPreferencesUpdate(theme="dark")
        assert u.theme == "dark"
        assert u.compact_view is None

    def test_exclude_unset(self):
        u = UserPreferencesUpdate(theme="dark")
        data = u.model_dump(exclude_unset=True)
        assert "theme" in data
        assert "compact_view" not in data

    def test_invalid_theme(self):
        with pytest.raises(ValidationError):
            UserPreferencesUpdate(theme="invalid")

    def test_page_size_validation(self):
        with pytest.raises(ValidationError):
            UserPreferencesUpdate(default_page_size=5)

    def test_notifications_update(self):
        n = NotificationSettings(budget_alert=False)
        u = UserPreferencesUpdate(notifications=n)
        assert u.notifications.budget_alert is False

    def test_all_fields_set(self):
        u = UserPreferencesUpdate(
            theme="light",
            compact_view=True,
            show_costs=False,
            default_page_size=50,
            notifications=NotificationSettings(execution_complete=False),
        )
        data = u.model_dump(exclude_unset=True)
        assert len(data) == 5


# =============================================================================
# Model Tests — APIKeyInfo
# =============================================================================


class TestAPIKeyInfo:
    """Tests for APIKeyInfo model."""

    def test_create(self):
        now = datetime.now()
        info = APIKeyInfo(
            id=1,
            provider="openai",
            key_prefix="sk-...xyz",
            created_at=now,
            updated_at=now,
        )
        assert info.id == 1
        assert info.provider == "openai"
        assert info.key_prefix == "sk-...xyz"

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            APIKeyInfo(id=1, provider="openai")


# =============================================================================
# Model Tests — APIKeyCreate
# =============================================================================


class TestAPIKeyCreate:
    """Tests for APIKeyCreate model."""

    def test_valid_providers(self):
        for provider in ("openai", "anthropic", "github"):
            k = APIKeyCreate(provider=provider, key="test-key")
            assert k.provider == provider

    def test_invalid_provider(self):
        with pytest.raises(ValidationError):
            APIKeyCreate(provider="invalid", key="test-key")

    def test_empty_key_rejected(self):
        with pytest.raises(ValidationError):
            APIKeyCreate(provider="openai", key="")

    def test_key_stored(self):
        k = APIKeyCreate(provider="openai", key="sk-test-123")
        assert k.key == "sk-test-123"


# =============================================================================
# Model Tests — APIKeyStatus
# =============================================================================


class TestAPIKeyStatus:
    """Tests for APIKeyStatus model."""

    def test_defaults_all_false(self):
        s = APIKeyStatus()
        assert s.openai is False
        assert s.anthropic is False
        assert s.github is False

    def test_partial_true(self):
        s = APIKeyStatus(openai=True)
        assert s.openai is True
        assert s.anthropic is False

    def test_all_true(self):
        s = APIKeyStatus(openai=True, anthropic=True, github=True)
        assert s.openai is True
        assert s.anthropic is True
        assert s.github is True


# =============================================================================
# SettingsManager — Fixtures
# =============================================================================


class _NoOpContextManager:
    """A reusable no-op context manager for mocking transaction()."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.fixture
def mock_backend():
    """Create a mock DatabaseBackend."""
    backend = MagicMock()
    backend.transaction.side_effect = lambda: _NoOpContextManager()
    return backend


@pytest.fixture
def manager(mock_backend):
    """Create SettingsManager with mocked backend and settings."""
    mock_settings = MagicMock()
    mock_settings.settings_encryption_key = None
    mock_settings.jwt_secret = "test-secret-key-for-testing"
    with patch("animus_forge.config.settings.get_settings", return_value=mock_settings):
        mgr = SettingsManager(mock_backend)
    return mgr


# =============================================================================
# SettingsManager — Encryption
# =============================================================================


class TestSettingsManagerEncryption:
    """Tests for encryption/decryption in SettingsManager."""

    def test_encrypt_decrypt_roundtrip(self, manager):
        plaintext = "sk-test-api-key-12345"
        encrypted = manager._encrypt(plaintext)
        assert encrypted != plaintext
        decrypted = manager._decrypt(encrypted)
        assert decrypted == plaintext

    def test_different_plaintexts_produce_different_ciphertext(self, manager):
        c1 = manager._encrypt("key-a")
        c2 = manager._encrypt("key-b")
        assert c1 != c2

    def test_derive_key_from_jwt_secret(self):
        mock_settings = MagicMock()
        mock_settings.settings_encryption_key = None
        mock_settings.jwt_secret = "my-jwt-secret"
        with patch("animus_forge.config.settings.get_settings", return_value=mock_settings):
            mgr = SettingsManager(MagicMock())
            assert mgr._encryption_key is not None
            assert len(mgr._encryption_key) == 44  # base64-encoded 32 bytes

    def test_derive_key_from_valid_fernet_key(self):
        from cryptography.fernet import Fernet

        fernet_key = Fernet.generate_key().decode()
        mock_settings = MagicMock()
        mock_settings.settings_encryption_key = fernet_key
        mock_settings.jwt_secret = None
        with patch("animus_forge.config.settings.get_settings", return_value=mock_settings):
            mgr = SettingsManager(MagicMock())
            assert mgr._encryption_key == fernet_key.encode()

    def test_derive_key_invalid_encryption_key_falls_back(self):
        mock_settings = MagicMock()
        mock_settings.settings_encryption_key = "not-a-valid-fernet-key"
        mock_settings.jwt_secret = None
        with patch("animus_forge.config.settings.get_settings", return_value=mock_settings):
            mgr = SettingsManager(MagicMock())
            # Should fall back to JWT_SECRET derivation
            assert mgr._encryption_key is not None

    def test_derive_key_default_secret(self):
        mock_settings = MagicMock()
        mock_settings.settings_encryption_key = None
        mock_settings.jwt_secret = None
        with patch("animus_forge.config.settings.get_settings", return_value=mock_settings):
            mgr = SettingsManager(MagicMock())
            assert mgr._encryption_key is not None


# =============================================================================
# SettingsManager — Key Masking
# =============================================================================


class TestSettingsManagerMasking:
    """Tests for _mask_key method."""

    def test_long_key_masking(self, manager):
        masked = manager._mask_key("sk-proj-abcdefghijklmnop")
        assert masked == "sk-proj...mnop"

    def test_short_key_masking(self, manager):
        masked = manager._mask_key("short")
        assert masked == "sho...rt"

    def test_exactly_twelve_chars(self, manager):
        masked = manager._mask_key("123456789012")
        assert masked == "123...12"

    def test_boundary_thirteen_chars(self, manager):
        masked = manager._mask_key("1234567890123")
        assert masked == "1234567...0123"


# =============================================================================
# SettingsManager — User Preferences
# =============================================================================


class TestSettingsManagerPreferences:
    """Tests for preference CRUD operations."""

    def test_get_preferences_creates_default_when_not_found(self, manager, mock_backend):
        mock_backend.fetchone.return_value = None
        prefs = manager.get_preferences("user-1")
        assert prefs.user_id == "user-1"
        assert prefs.theme == "system"
        # Should have called execute to save defaults
        mock_backend.execute.assert_called()

    def test_get_preferences_returns_existing(self, manager, mock_backend):
        mock_backend.fetchone.return_value = {
            "user_id": "user-1",
            "theme": "dark",
            "compact_view": 1,
            "show_costs": 0,
            "default_page_size": 50,
            "notify_execution_complete": 1,
            "notify_execution_failed": 0,
            "notify_budget_alert": 1,
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-02T00:00:00",
        }
        prefs = manager.get_preferences("user-1")
        assert prefs.theme == "dark"
        assert prefs.compact_view is True
        assert prefs.show_costs is False
        assert prefs.default_page_size == 50
        assert prefs.notifications.execution_failed is False

    def test_update_preferences_theme(self, manager, mock_backend):
        mock_backend.fetchone.return_value = None
        update = UserPreferencesUpdate(theme="dark")
        result = manager.update_preferences("user-1", update)
        assert result.theme == "dark"

    def test_update_preferences_compact_view(self, manager, mock_backend):
        mock_backend.fetchone.return_value = None
        update = UserPreferencesUpdate(compact_view=True)
        result = manager.update_preferences("user-1", update)
        assert result.compact_view is True

    def test_update_preferences_show_costs(self, manager, mock_backend):
        mock_backend.fetchone.return_value = None
        update = UserPreferencesUpdate(show_costs=False)
        result = manager.update_preferences("user-1", update)
        assert result.show_costs is False

    def test_update_preferences_page_size(self, manager, mock_backend):
        mock_backend.fetchone.return_value = None
        update = UserPreferencesUpdate(default_page_size=50)
        result = manager.update_preferences("user-1", update)
        assert result.default_page_size == 50

    def test_update_preferences_notifications(self, manager, mock_backend):
        mock_backend.fetchone.return_value = None
        notif = NotificationSettings(budget_alert=False)
        update = UserPreferencesUpdate(notifications=notif)
        result = manager.update_preferences("user-1", update)
        assert result.notifications.budget_alert is False

    def test_update_preferences_sets_updated_at(self, manager, mock_backend):
        mock_backend.fetchone.return_value = None
        update = UserPreferencesUpdate(theme="light")
        result = manager.update_preferences("user-1", update)
        assert result.updated_at is not None

    def test_update_preserves_unset_fields(self, manager, mock_backend):
        mock_backend.fetchone.return_value = None
        # First get creates defaults (theme="system", show_costs=True)
        update = UserPreferencesUpdate(theme="dark")
        result = manager.update_preferences("user-1", update)
        # show_costs should remain default True
        assert result.show_costs is True
        assert result.theme == "dark"

    def test_save_preferences_calls_execute(self, manager, mock_backend):
        mock_backend.fetchone.return_value = None
        manager.get_preferences("user-1")
        # _save_preferences calls execute with INSERT...ON CONFLICT
        assert mock_backend.execute.call_count >= 1
        sql = mock_backend.execute.call_args_list[0][0][0]
        assert "INSERT INTO user_preferences" in sql

    def test_save_preferences_calls_transaction(self, manager, mock_backend):
        mock_backend.fetchone.return_value = None
        manager.get_preferences("user-1")
        mock_backend.transaction.assert_called()


# =============================================================================
# SettingsManager — API Keys
# =============================================================================


class TestSettingsManagerAPIKeys:
    """Tests for API key CRUD operations."""

    def test_get_api_keys_empty(self, manager, mock_backend):
        mock_backend.fetchall.return_value = []
        keys = manager.get_api_keys("user-1")
        assert keys == []

    def test_get_api_keys_returns_list(self, manager, mock_backend):
        now = datetime.now()
        mock_backend.fetchall.return_value = [
            {
                "id": 1,
                "provider": "openai",
                "key_prefix": "sk-...xyz",
                "created_at": now,
                "updated_at": now,
            },
            {
                "id": 2,
                "provider": "anthropic",
                "key_prefix": "sk-ant...abc",
                "created_at": now,
                "updated_at": now,
            },
        ]
        keys = manager.get_api_keys("user-1")
        assert len(keys) == 2
        assert keys[0].provider == "openai"
        assert keys[1].provider == "anthropic"

    def test_get_api_keys_are_api_key_info(self, manager, mock_backend):
        now = datetime.now()
        mock_backend.fetchall.return_value = [
            {
                "id": 1,
                "provider": "openai",
                "key_prefix": "sk-...xyz",
                "created_at": now,
                "updated_at": now,
            }
        ]
        keys = manager.get_api_keys("user-1")
        assert isinstance(keys[0], APIKeyInfo)

    def test_get_api_key_status_none_configured(self, manager, mock_backend):
        mock_backend.fetchall.return_value = []
        status = manager.get_api_key_status("user-1")
        assert status.openai is False
        assert status.anthropic is False
        assert status.github is False

    def test_get_api_key_status_some_configured(self, manager, mock_backend):
        mock_backend.fetchall.return_value = [
            {"provider": "openai"},
            {"provider": "github"},
        ]
        status = manager.get_api_key_status("user-1")
        assert status.openai is True
        assert status.anthropic is False
        assert status.github is True

    def test_set_api_key_encrypts_and_saves(self, manager, mock_backend):
        now = datetime.now()
        mock_backend.fetchone.return_value = {
            "id": 1,
            "provider": "openai",
            "key_prefix": "sk-proj...mnop",
            "created_at": now,
            "updated_at": now,
        }
        key_create = APIKeyCreate(provider="openai", key="sk-proj-abcdefghijklmnop")
        result = manager.set_api_key("user-1", key_create)

        assert result.provider == "openai"
        assert result.key_prefix == "sk-proj...mnop"
        # Verify encryption happened
        call_args = mock_backend.execute.call_args_list[0][0]
        sql = call_args[0]
        params = call_args[1]
        assert "INSERT INTO user_api_keys" in sql
        # The encrypted key should not be plaintext
        assert params[2] != "sk-proj-abcdefghijklmnop"

    def test_set_api_key_returns_api_key_info(self, manager, mock_backend):
        now = datetime.now()
        mock_backend.fetchone.return_value = {
            "id": 1,
            "provider": "openai",
            "key_prefix": "sk-...xyz",
            "created_at": now,
            "updated_at": now,
        }
        key_create = APIKeyCreate(provider="openai", key="sk-test-key-12345")
        result = manager.set_api_key("user-1", key_create)
        assert isinstance(result, APIKeyInfo)

    def test_delete_api_key_found(self, manager, mock_backend):
        cursor_mock = MagicMock()
        cursor_mock.rowcount = 1
        mock_backend.execute.return_value = cursor_mock
        result = manager.delete_api_key("user-1", "openai")
        assert result is True

    def test_delete_api_key_not_found(self, manager, mock_backend):
        cursor_mock = MagicMock()
        cursor_mock.rowcount = 0
        mock_backend.execute.return_value = cursor_mock
        result = manager.delete_api_key("user-1", "openai")
        assert result is False

    def test_delete_api_key_calls_correct_query(self, manager, mock_backend):
        cursor_mock = MagicMock()
        cursor_mock.rowcount = 1
        mock_backend.execute.return_value = cursor_mock
        manager.delete_api_key("user-1", "openai")
        sql, params = mock_backend.execute.call_args[0]
        assert "DELETE FROM user_api_keys" in sql
        assert params == ("user-1", "openai")

    def test_get_decrypted_api_key_found(self, manager, mock_backend):
        plaintext_key = "sk-test-my-api-key"
        encrypted = manager._encrypt(plaintext_key)
        mock_backend.fetchone.return_value = {"encrypted_key": encrypted}
        result = manager.get_decrypted_api_key("user-1", "openai")
        assert result == plaintext_key

    def test_get_decrypted_api_key_not_found(self, manager, mock_backend):
        mock_backend.fetchone.return_value = None
        result = manager.get_decrypted_api_key("user-1", "openai")
        assert result is None

    def test_get_decrypted_api_key_query(self, manager, mock_backend):
        mock_backend.fetchone.return_value = None
        manager.get_decrypted_api_key("user-1", "anthropic")
        sql, params = mock_backend.fetchone.call_args[0]
        assert "SELECT encrypted_key" in sql
        assert params == ("user-1", "anthropic")


# =============================================================================
# SettingsManager — Edge Cases
# =============================================================================


class TestSettingsManagerEdgeCases:
    """Edge case tests."""

    def test_encrypt_empty_string(self, manager):
        encrypted = manager._encrypt("")
        decrypted = manager._decrypt(encrypted)
        assert decrypted == ""

    def test_encrypt_unicode(self, manager):
        text = "Hello, World!"
        encrypted = manager._encrypt(text)
        decrypted = manager._decrypt(encrypted)
        assert decrypted == text

    def test_mask_key_very_short(self, manager):
        masked = manager._mask_key("ab")
        # len <= 12 branch: first 3 + "..." + last 2
        assert "..." in masked

    def test_mask_key_single_char(self, manager):
        masked = manager._mask_key("a")
        assert "..." in masked

    def test_get_preferences_row_missing_optional_fields(self, manager, mock_backend):
        """Test graceful handling when row.get returns None for optional fields."""
        mock_backend.fetchone.return_value = {
            "user_id": "user-1",
            "theme": "system",
            "compact_view": 0,
            "show_costs": 1,
            "default_page_size": 20,
            "notify_execution_complete": 1,
            "notify_execution_failed": 1,
            "notify_budget_alert": 1,
        }
        prefs = manager.get_preferences("user-1")
        assert prefs.created_at is None
        assert prefs.updated_at is None
