"""Global database backend instance."""

from functools import lru_cache

from .backends import DatabaseBackend, create_backend


@lru_cache
def get_database() -> DatabaseBackend:
    """Get cached database backend from settings.

    Returns a singleton DatabaseBackend instance configured from
    the DATABASE_URL setting. The backend is cached for reuse
    across the application.

    Returns:
        DatabaseBackend: Configured database backend
    """
    from animus_forge.config import get_settings

    return create_backend(get_settings().database_url)


def reset_database() -> None:
    """Reset cached database connection.

    Clears the cached database backend, forcing a new connection
    on the next get_database() call. Useful for testing.
    """
    get_database.cache_clear()
