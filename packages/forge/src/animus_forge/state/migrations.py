"""Database migration runner."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .backends import DatabaseBackend, PostgresBackend

logger = logging.getLogger(__name__)

# Migration files location relative to package root
MIGRATIONS_DIR = Path(__file__).parent.parent.parent.parent / "migrations"


def _adapt_sql_for_postgres(sql: str) -> str:
    """Adapt SQLite SQL syntax for PostgreSQL.

    Args:
        sql: SQL script with SQLite syntax

    Returns:
        SQL adapted for PostgreSQL
    """
    # Replace INTEGER PRIMARY KEY AUTOINCREMENT with SERIAL PRIMARY KEY
    sql = re.sub(
        r"INTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT",
        "SERIAL PRIMARY KEY",
        sql,
        flags=re.IGNORECASE,
    )
    return sql


def _get_applied_migrations(backend: DatabaseBackend) -> set[str]:
    """Get list of already applied migration versions.

    Args:
        backend: Database backend

    Returns:
        Set of applied migration version strings
    """
    # For PostgreSQL, we need to check if table exists first to avoid
    # aborting the transaction when the table doesn't exist
    if isinstance(backend, PostgresBackend):
        try:
            result = backend.fetchone(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'schema_migrations'
                )
                """
            )
            if not result or not result.get("exists", False):
                return set()
        except Exception:
            return set()

    try:
        rows = backend.fetchall("SELECT version FROM schema_migrations")
        return {row["version"] for row in rows}
    except Exception:
        # Table doesn't exist yet (SQLite)
        return set()


def _get_migration_files() -> list[tuple[str, Path]]:
    """Get sorted list of migration files.

    Returns:
        List of (version, path) tuples sorted by version
    """
    if not MIGRATIONS_DIR.exists():
        return []

    migrations = []
    for file_path in MIGRATIONS_DIR.glob("*.sql"):
        # Extract version from filename (e.g., "001_initial_schema.sql" -> "001")
        match = re.match(r"^(\d+)_", file_path.name)
        if match:
            version = match.group(1)
            migrations.append((version, file_path))

    return sorted(migrations, key=lambda x: x[0])


def run_migrations(backend: DatabaseBackend) -> list[str]:
    """Run pending database migrations.

    Applies all migration files that haven't been applied yet.
    Migrations are tracked in the schema_migrations table.

    Args:
        backend: Database backend to run migrations on

    Returns:
        List of applied migration versions
    """
    applied = []
    already_applied = _get_applied_migrations(backend)
    migration_files = _get_migration_files()

    if not migration_files:
        logger.info("No migration files found")
        return applied

    for version, file_path in migration_files:
        if version in already_applied:
            logger.debug(f"Migration {version} already applied, skipping")
            continue

        logger.info(f"Applying migration {version}: {file_path.name}")

        try:
            sql = file_path.read_text()

            # Adapt SQL for PostgreSQL if needed
            if isinstance(backend, PostgresBackend):
                sql = _adapt_sql_for_postgres(sql)

            # Execute the migration
            with backend.transaction():
                backend.executescript(sql)

                # Record the migration
                backend.execute(
                    "INSERT INTO schema_migrations (version, description) VALUES (?, ?)",
                    (version, file_path.stem),
                )

            applied.append(version)
            logger.info(f"Migration {version} applied successfully")

        except Exception as e:
            logger.error(f"Migration {version} failed: {e}")
            raise RuntimeError(f"Migration {version} failed: {e}") from e

    if applied:
        logger.info(f"Applied {len(applied)} migration(s): {', '.join(applied)}")
    else:
        logger.info("Database schema is up to date")

    return applied


def get_migration_status(backend: DatabaseBackend) -> dict:
    """Get current migration status.

    Args:
        backend: Database backend

    Returns:
        Dictionary with migration status info
    """
    applied = _get_applied_migrations(backend)
    available = _get_migration_files()

    pending = [v for v, _ in available if v not in applied]

    return {
        "applied": sorted(applied),
        "pending": pending,
        "up_to_date": len(pending) == 0,
    }
