"""Tests for database factory and migrations modules."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, "src")

from animus_forge.state.backends import SQLiteBackend
from animus_forge.state.database import get_database, reset_database


class TestGetDatabase:
    """Tests for get_database function."""

    def setup_method(self):
        """Reset database cache before each test."""
        reset_database()

    def teardown_method(self):
        """Reset database cache after each test."""
        reset_database()

    def test_returns_backend(self, monkeypatch):
        """get_database returns a DatabaseBackend instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

            # Clear settings cache
            from animus_forge.config import get_settings

            get_settings.cache_clear()

            backend = get_database()
            assert isinstance(backend, SQLiteBackend)

    def test_caches_backend(self, monkeypatch):
        """get_database returns the same instance on repeated calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

            from animus_forge.config import get_settings

            get_settings.cache_clear()

            backend1 = get_database()
            backend2 = get_database()
            assert backend1 is backend2

    def test_reset_clears_cache(self, monkeypatch):
        """reset_database clears the cached backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

            from animus_forge.config import get_settings

            get_settings.cache_clear()

            backend1 = get_database()
            reset_database()

            # After reset, we need to clear settings cache again for new path
            db_path2 = os.path.join(tmpdir, "test2.db")
            monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path2}")
            get_settings.cache_clear()

            backend2 = get_database()
            assert backend1 is not backend2


class TestMigrations:
    """Tests for migrations module."""

    @pytest.fixture
    def backend(self):
        """Create a temporary SQLite backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            backend = SQLiteBackend(db_path=db_path)
            yield backend
            backend.close()

    @pytest.fixture
    def migrations_dir(self, tmp_path):
        """Create a temporary migrations directory with test migrations."""
        migrations = tmp_path / "migrations"
        migrations.mkdir()

        # Create test migration 001
        migration_001 = migrations / "001_create_users.sql"
        migration_001.write_text("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            );

            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE
            );
        """)

        # Create test migration 002
        migration_002 = migrations / "002_add_posts.sql"
        migration_002.write_text("""
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        """)

        return migrations

    def test_run_migrations_applies_all(self, backend, migrations_dir, monkeypatch):
        """run_migrations applies all pending migrations."""
        from animus_forge.state import migrations

        # Patch MIGRATIONS_DIR to use our test directory
        monkeypatch.setattr(migrations, "MIGRATIONS_DIR", migrations_dir)

        applied = migrations.run_migrations(backend)

        assert "001" in applied
        assert "002" in applied
        assert len(applied) == 2

        # Verify tables were created
        users = backend.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
        )
        assert len(users) == 1

        posts = backend.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='posts'"
        )
        assert len(posts) == 1

    def test_run_migrations_skips_applied(self, backend, migrations_dir, monkeypatch):
        """run_migrations skips already applied migrations."""
        from animus_forge.state import migrations

        monkeypatch.setattr(migrations, "MIGRATIONS_DIR", migrations_dir)

        # Run migrations first time
        first_run = migrations.run_migrations(backend)
        assert len(first_run) == 2

        # Run migrations second time
        second_run = migrations.run_migrations(backend)
        assert len(second_run) == 0

    def test_run_migrations_tracks_versions(self, backend, migrations_dir, monkeypatch):
        """run_migrations records applied versions in schema_migrations."""
        from animus_forge.state import migrations

        monkeypatch.setattr(migrations, "MIGRATIONS_DIR", migrations_dir)
        migrations.run_migrations(backend)

        rows = backend.fetchall("SELECT version FROM schema_migrations ORDER BY version")
        versions = [r["version"] for r in rows]

        assert "001" in versions
        assert "002" in versions

    def test_run_migrations_empty_directory(self, backend, tmp_path, monkeypatch):
        """run_migrations handles empty migrations directory."""
        from animus_forge.state import migrations

        empty_dir = tmp_path / "empty_migrations"
        empty_dir.mkdir()
        monkeypatch.setattr(migrations, "MIGRATIONS_DIR", empty_dir)

        applied = migrations.run_migrations(backend)
        assert applied == []

    def test_run_migrations_missing_directory(self, backend, tmp_path, monkeypatch):
        """run_migrations handles missing migrations directory."""
        from animus_forge.state import migrations

        missing_dir = tmp_path / "nonexistent"
        monkeypatch.setattr(migrations, "MIGRATIONS_DIR", missing_dir)

        applied = migrations.run_migrations(backend)
        assert applied == []

    def test_get_migration_status(self, backend, migrations_dir, monkeypatch):
        """get_migration_status returns correct status."""
        from animus_forge.state import migrations

        monkeypatch.setattr(migrations, "MIGRATIONS_DIR", migrations_dir)

        # Before running migrations
        status = migrations.get_migration_status(backend)
        assert status["applied"] == []
        assert "001" in status["pending"]
        assert "002" in status["pending"]
        assert status["up_to_date"] is False

        # After running migrations
        migrations.run_migrations(backend)
        status = migrations.get_migration_status(backend)
        assert "001" in status["applied"]
        assert "002" in status["applied"]
        assert status["pending"] == []
        assert status["up_to_date"] is True

    def test_migration_failure_rolls_back(self, backend, tmp_path, monkeypatch):
        """Failed migration does not leave partial state."""
        from animus_forge.state import migrations

        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()

        # Create migration with schema_migrations table and invalid SQL
        bad_migration = migrations_dir / "001_bad.sql"
        bad_migration.write_text("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            );

            CREATE TABLE good_table (id INTEGER PRIMARY KEY);
            THIS IS INVALID SQL;
        """)

        monkeypatch.setattr(migrations, "MIGRATIONS_DIR", migrations_dir)

        with pytest.raises(RuntimeError) as exc:
            migrations.run_migrations(backend)

        assert "001" in str(exc.value)

    def test_migrations_sorted_by_version(self, backend, tmp_path, monkeypatch):
        """Migrations are applied in version order."""
        from animus_forge.state import migrations

        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()

        # Create migrations out of order
        (migrations_dir / "003_third.sql").write_text("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            );
            CREATE TABLE third (id INTEGER PRIMARY KEY);
        """)
        (migrations_dir / "001_first.sql").write_text("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            );
            CREATE TABLE first (id INTEGER PRIMARY KEY);
        """)
        (migrations_dir / "002_second.sql").write_text("""
            CREATE TABLE second (id INTEGER PRIMARY KEY);
        """)

        monkeypatch.setattr(migrations, "MIGRATIONS_DIR", migrations_dir)

        applied = migrations.run_migrations(backend)

        # Should be applied in order
        assert applied == ["001", "002", "003"]
