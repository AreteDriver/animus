"""Tests for the per-install salt helper (roadmap A7)."""

from __future__ import annotations

from animus_forge.security.install_salt import get_install_salt


class TestInstallSalt:
    def test_generates_and_persists_stable_salt(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANIMUS_CONFIG_DIR", str(tmp_path))
        s1 = get_install_salt("unit_test_a")
        assert isinstance(s1, bytes) and len(s1) == 16
        # Persisted: a second call returns the SAME salt.
        s2 = get_install_salt("unit_test_a")
        assert s1 == s2
        assert (tmp_path / "unit_test_a.salt").exists()

    def test_different_names_get_different_salts(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANIMUS_CONFIG_DIR", str(tmp_path))
        assert get_install_salt("aaa") != get_install_salt("bbb")

    def test_env_override_wins_and_is_not_persisted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANIMUS_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("LEGACY_SALT", "gorgon-field-encryption-v1")
        salt = get_install_salt("field_encryption", env_var="LEGACY_SALT")
        assert salt == b"gorgon-field-encryption-v1"
        # Override path does not write a salt file.
        assert not (tmp_path / "field_encryption.salt").exists()

    def test_salt_file_has_restrictive_permissions(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANIMUS_CONFIG_DIR", str(tmp_path))
        get_install_salt("perm_test")
        mode = (tmp_path / "perm_test.salt").stat().st_mode & 0o777
        assert mode == 0o600

    def test_fallback_when_unpersistable(self, tmp_path, monkeypatch):
        # Point config dir at a path that can't be created (a file, not a dir).
        blocker = tmp_path / "blocker"
        blocker.write_text("not a dir")
        monkeypatch.setenv("ANIMUS_CONFIG_DIR", str(blocker / "sub"))
        salt = get_install_salt("fallback_test")
        # Degrades to a stable non-persisted value rather than raising.
        assert salt == b"animus-install-fallback-fallback_test"
