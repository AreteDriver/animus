"""Plugin marketplace for discovering and browsing plugins."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from .models import (
    PluginCategory,
    PluginListing,
    PluginMetadata,
    PluginRelease,
    PluginSearchResult,
)

if TYPE_CHECKING:
    from animus_forge.state.backends import DatabaseBackend

logger = logging.getLogger(__name__)


class PluginMarketplace:
    """Marketplace for discovering and browsing plugins.

    Provides search, browse, and catalog functionality for plugins.
    Can work with local catalog or remote registry.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS plugin_catalog (
        id TEXT PRIMARY KEY,
        name TEXT UNIQUE NOT NULL,
        display_name TEXT NOT NULL,
        description TEXT,
        long_description TEXT,
        author TEXT,
        category TEXT DEFAULT 'other',
        tags TEXT,  -- JSON array
        downloads INTEGER DEFAULT 0,
        rating REAL DEFAULT 0.0,
        review_count INTEGER DEFAULT 0,
        latest_version TEXT NOT NULL,
        published_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        verified BOOLEAN DEFAULT FALSE,
        featured BOOLEAN DEFAULT FALSE,
        repository_url TEXT,
        metadata TEXT,  -- JSON object
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS plugin_releases (
        id TEXT PRIMARY KEY,
        plugin_name TEXT NOT NULL,
        version TEXT NOT NULL,
        released_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        download_url TEXT NOT NULL,
        checksum TEXT NOT NULL,
        signature TEXT,
        changelog TEXT,
        file_size INTEGER DEFAULT 0,
        compatible_versions TEXT,  -- JSON array
        metadata TEXT,  -- JSON object
        UNIQUE(plugin_name, version),
        FOREIGN KEY(plugin_name) REFERENCES plugin_catalog(name) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_plugin_catalog_category ON plugin_catalog(category);
    CREATE INDEX IF NOT EXISTS idx_plugin_catalog_featured ON plugin_catalog(featured);
    CREATE INDEX IF NOT EXISTS idx_plugin_catalog_downloads ON plugin_catalog(downloads DESC);
    CREATE INDEX IF NOT EXISTS idx_plugin_catalog_rating ON plugin_catalog(rating DESC);
    CREATE INDEX IF NOT EXISTS idx_plugin_releases_plugin ON plugin_releases(plugin_name);
    """

    def __init__(self, backend: DatabaseBackend):
        """Initialize marketplace with database backend.

        Args:
            backend: Database backend for persistence.
        """
        self.backend = backend
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Ensure marketplace tables exist."""
        try:
            with self.backend.transaction() as conn:
                for statement in self.SCHEMA.split(";"):
                    statement = statement.strip()
                    if statement:
                        conn.execute(statement)
            logger.info("Plugin marketplace schema initialized")
        except Exception as e:
            logger.error(f"Failed to initialize marketplace schema: {e}")

    def search(
        self,
        query: str,
        category: PluginCategory | None = None,
        tags: list[str] | None = None,
        verified_only: bool = False,
        page: int = 1,
        per_page: int = 20,
        sort_by: str = "downloads",
    ) -> PluginSearchResult:
        """Search for plugins in the marketplace.

        Args:
            query: Search query (matches name, description, tags).
            category: Filter by category.
            tags: Filter by tags (any match).
            verified_only: Only show verified plugins.
            page: Page number (1-indexed).
            per_page: Results per page.
            sort_by: Sort field (downloads, rating, updated_at, name).

        Returns:
            Search results with pagination.
        """
        conditions = []
        params = []

        if query:
            conditions.append(
                "(name LIKE ? OR display_name LIKE ? OR description LIKE ? OR tags LIKE ?)"
            )
            like_query = f"%{query}%"
            params.extend([like_query] * 4)

        if category:
            conditions.append("category = ?")
            params.append(category.value)

        if tags:
            tag_conditions = []
            for tag in tags:
                tag_conditions.append("tags LIKE ?")
                params.append(f'%"{tag}"%')
            conditions.append(f"({' OR '.join(tag_conditions)})")

        if verified_only:
            conditions.append("verified = 1")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Validate sort field
        valid_sorts = {
            "downloads": "downloads DESC",
            "rating": "rating DESC",
            "updated_at": "updated_at DESC",
            "name": "name ASC",
        }
        order_clause = valid_sorts.get(sort_by, "downloads DESC")

        # Count total
        count_sql = f"SELECT COUNT(*) FROM plugin_catalog WHERE {where_clause}"
        total = 0
        try:
            result = self.backend.execute(count_sql, params)
            if result:
                total = result[0][0]
        except Exception as e:
            logger.error(f"Failed to count plugins: {e}")

        # Fetch page
        offset = (page - 1) * per_page
        select_sql = f"""
            SELECT id, name, display_name, description, long_description, author,
                   category, tags, downloads, rating, review_count, latest_version,
                   published_at, updated_at, verified, featured, repository_url
            FROM plugin_catalog
            WHERE {where_clause}
            ORDER BY {order_clause}
            LIMIT ? OFFSET ?
        """
        params.extend([per_page, offset])

        results = []
        try:
            rows = self.backend.execute(select_sql, params)
            for row in rows:
                listing = self._row_to_listing(row)
                results.append(listing)
        except Exception as e:
            logger.error(f"Failed to search plugins: {e}")

        return PluginSearchResult(
            query=query,
            total=total,
            page=page,
            per_page=per_page,
            results=results,
        )

    def browse(
        self,
        category: PluginCategory | None = None,
        featured_only: bool = False,
        page: int = 1,
        per_page: int = 20,
    ) -> PluginSearchResult:
        """Browse plugins by category.

        Args:
            category: Category to browse (None for all).
            featured_only: Only show featured plugins.
            page: Page number.
            per_page: Results per page.

        Returns:
            Plugin listings.
        """
        return self.search(
            query="",
            category=category,
            verified_only=False,
            page=page,
            per_page=per_page,
        )

    def get_featured(self, limit: int = 10) -> list[PluginListing]:
        """Get featured plugins.

        Args:
            limit: Maximum number of plugins.

        Returns:
            Featured plugin listings.
        """
        sql = """
            SELECT id, name, display_name, description, long_description, author,
                   category, tags, downloads, rating, review_count, latest_version,
                   published_at, updated_at, verified, featured, repository_url
            FROM plugin_catalog
            WHERE featured = 1
            ORDER BY downloads DESC
            LIMIT ?
        """
        results = []
        try:
            rows = self.backend.execute(sql, [limit])
            for row in rows:
                results.append(self._row_to_listing(row))
        except Exception as e:
            logger.error(f"Failed to get featured plugins: {e}")
        return results

    def get_popular(self, limit: int = 10) -> list[PluginListing]:
        """Get most popular plugins by downloads.

        Args:
            limit: Maximum number of plugins.

        Returns:
            Popular plugin listings.
        """
        sql = """
            SELECT id, name, display_name, description, long_description, author,
                   category, tags, downloads, rating, review_count, latest_version,
                   published_at, updated_at, verified, featured, repository_url
            FROM plugin_catalog
            ORDER BY downloads DESC
            LIMIT ?
        """
        results = []
        try:
            rows = self.backend.execute(sql, [limit])
            for row in rows:
                results.append(self._row_to_listing(row))
        except Exception as e:
            logger.error(f"Failed to get popular plugins: {e}")
        return results

    def get_plugin(self, name: str) -> PluginListing | None:
        """Get a specific plugin by name.

        Args:
            name: Plugin name.

        Returns:
            Plugin listing or None if not found.
        """
        sql = """
            SELECT id, name, display_name, description, long_description, author,
                   category, tags, downloads, rating, review_count, latest_version,
                   published_at, updated_at, verified, featured, repository_url
            FROM plugin_catalog
            WHERE name = ?
        """
        try:
            rows = self.backend.execute(sql, [name])
            if rows:
                listing = self._row_to_listing(rows[0])
                # Fetch releases
                listing.releases = self.get_releases(name)
                return listing
        except Exception as e:
            logger.error(f"Failed to get plugin {name}: {e}")
        return None

    def get_releases(self, plugin_name: str) -> list[PluginRelease]:
        """Get all releases for a plugin.

        Args:
            plugin_name: Plugin name.

        Returns:
            List of releases (newest first).
        """
        sql = """
            SELECT id, plugin_name, version, released_at, download_url, checksum,
                   signature, changelog, file_size, compatible_versions, metadata
            FROM plugin_releases
            WHERE plugin_name = ?
            ORDER BY released_at DESC
        """
        results = []
        try:
            rows = self.backend.execute(sql, [plugin_name])
            for row in rows:
                results.append(self._row_to_release(row))
        except Exception as e:
            logger.error(f"Failed to get releases for {plugin_name}: {e}")
        return results

    def get_release(self, plugin_name: str, version: str) -> PluginRelease | None:
        """Get a specific release.

        Args:
            plugin_name: Plugin name.
            version: Release version.

        Returns:
            Release or None if not found.
        """
        sql = """
            SELECT id, plugin_name, version, released_at, download_url, checksum,
                   signature, changelog, file_size, compatible_versions, metadata
            FROM plugin_releases
            WHERE plugin_name = ? AND version = ?
        """
        try:
            rows = self.backend.execute(sql, [plugin_name, version])
            if rows:
                return self._row_to_release(rows[0])
        except Exception as e:
            logger.error(f"Failed to get release {plugin_name}@{version}: {e}")
        return None

    def add_plugin(self, listing: PluginListing) -> bool:
        """Add a plugin to the catalog.

        Args:
            listing: Plugin listing to add.

        Returns:
            True if added successfully.
        """
        sql = """
            INSERT INTO plugin_catalog (
                id, name, display_name, description, long_description, author,
                category, tags, downloads, rating, review_count, latest_version,
                published_at, updated_at, verified, featured, repository_url
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            self.backend.execute(
                sql,
                [
                    listing.id,
                    listing.name,
                    listing.display_name,
                    listing.description,
                    listing.long_description,
                    listing.author,
                    listing.category.value,
                    json.dumps(listing.tags),
                    listing.downloads,
                    listing.rating,
                    listing.review_count,
                    listing.latest_version,
                    listing.published_at.isoformat(),
                    listing.updated_at.isoformat(),
                    listing.verified,
                    listing.featured,
                    listing.repository_url,
                ],
            )
            logger.info(f"Added plugin {listing.name} to catalog")
            return True
        except Exception as e:
            logger.error(f"Failed to add plugin {listing.name}: {e}")
            return False

    def add_release(self, release: PluginRelease) -> bool:
        """Add a release to the catalog.

        Args:
            release: Release to add.

        Returns:
            True if added successfully.
        """
        sql = """
            INSERT INTO plugin_releases (
                id, plugin_name, version, released_at, download_url, checksum,
                signature, changelog, file_size, compatible_versions, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            metadata_json = None
            if release.metadata:
                metadata_json = release.metadata.model_dump_json()

            self.backend.execute(
                sql,
                [
                    release.id,
                    release.plugin_name,
                    release.version,
                    release.released_at.isoformat(),
                    release.download_url,
                    release.checksum,
                    release.signature,
                    release.changelog,
                    release.file_size,
                    json.dumps(release.compatible_versions),
                    metadata_json,
                ],
            )
            logger.info(f"Added release {release.plugin_name}@{release.version}")
            return True
        except Exception as e:
            logger.error(f"Failed to add release: {e}")
            return False

    def update_plugin(self, name: str, **kwargs) -> bool:
        """Update plugin catalog entry.

        Args:
            name: Plugin name.
            **kwargs: Fields to update.

        Returns:
            True if updated successfully.
        """
        valid_fields = {
            "display_name",
            "description",
            "long_description",
            "author",
            "category",
            "tags",
            "downloads",
            "rating",
            "review_count",
            "latest_version",
            "verified",
            "featured",
            "repository_url",
        }
        updates = []
        params = []
        for field, value in kwargs.items():
            if field in valid_fields:
                updates.append(f"{field} = ?")
                if field == "tags" and isinstance(value, list):
                    value = json.dumps(value)
                elif field == "category" and hasattr(value, "value"):
                    value = value.value
                params.append(value)

        if not updates:
            return False

        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(name)

        sql = f"UPDATE plugin_catalog SET {', '.join(updates)} WHERE name = ?"
        try:
            self.backend.execute(sql, params)
            logger.info(f"Updated plugin {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to update plugin {name}: {e}")
            return False

    def increment_downloads(self, name: str) -> bool:
        """Increment download count for a plugin.

        Args:
            name: Plugin name.

        Returns:
            True if updated successfully.
        """
        sql = "UPDATE plugin_catalog SET downloads = downloads + 1 WHERE name = ?"
        try:
            self.backend.execute(sql, [name])
            return True
        except Exception as e:
            logger.error(f"Failed to increment downloads for {name}: {e}")
            return False

    def get_categories(self) -> dict[str, int]:
        """Get category counts.

        Returns:
            Dict mapping category to plugin count.
        """
        sql = "SELECT category, COUNT(*) FROM plugin_catalog GROUP BY category"
        counts = {}
        try:
            rows = self.backend.execute(sql, [])
            for row in rows:
                counts[row[0]] = row[1]
        except Exception as e:
            logger.error(f"Failed to get categories: {e}")
        return counts

    def _row_to_listing(self, row: tuple) -> PluginListing:
        """Convert database row to PluginListing."""
        return PluginListing(
            id=row[0],
            name=row[1],
            display_name=row[2],
            description=row[3] or "",
            long_description=row[4] or "",
            author=row[5] or "",
            category=PluginCategory(row[6]) if row[6] else PluginCategory.OTHER,
            tags=json.loads(row[7]) if row[7] else [],
            downloads=row[8] or 0,
            rating=row[9] or 0.0,
            review_count=row[10] or 0,
            latest_version=row[11],
            published_at=datetime.fromisoformat(row[12]) if row[12] else datetime.now(),
            updated_at=datetime.fromisoformat(row[13]) if row[13] else datetime.now(),
            verified=bool(row[14]),
            featured=bool(row[15]),
            repository_url=row[16],
        )

    def _row_to_release(self, row: tuple) -> PluginRelease:
        """Convert database row to PluginRelease."""
        metadata = None
        if row[10]:
            try:
                metadata = PluginMetadata.model_validate_json(row[10])
            except Exception:
                pass  # Non-critical fallback: metadata parsing optional, continue with None

        return PluginRelease(
            id=row[0],
            plugin_name=row[1],
            version=row[2],
            released_at=datetime.fromisoformat(row[3]) if row[3] else datetime.now(),
            download_url=row[4],
            checksum=row[5],
            signature=row[6],
            changelog=row[7] or "",
            file_size=row[8] or 0,
            compatible_versions=json.loads(row[9]) if row[9] else [],
            metadata=metadata,
        )
