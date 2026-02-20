"""Notion API client wrapper with sync and async support.

Includes response caching for frequently accessed data like database schemas.
"""

from typing import Any, Optional

from animus_forge.api_clients.resilience import resilient_call, resilient_call_async
from animus_forge.cache import cached
from animus_forge.config import get_settings
from animus_forge.errors import MaxRetriesError
from animus_forge.utils.retry import async_with_retry, with_retry

try:
    from notion_client import APIResponseError
    from notion_client import AsyncClient as AsyncNotionClient
    from notion_client import Client as NotionClient
except ImportError:
    NotionClient = None
    AsyncNotionClient = None
    APIResponseError = Exception


class NotionClientWrapper:
    """Wrapper for Notion API with sync and async support.

    Provides both synchronous and asynchronous methods for API calls.
    Async methods are suffixed with '_async' (e.g., query_database_async).
    """

    def __init__(self):
        settings = get_settings()
        if settings.notion_token and NotionClient:
            self.client = NotionClient(auth=settings.notion_token)
        else:
            self.client = None
        self._async_client: AsyncNotionClient | None = None

    @property
    def async_client(self) -> Optional["AsyncNotionClient"]:
        """Lazy-load async client on first access."""
        if self._async_client is None and AsyncNotionClient:
            settings = get_settings()
            if settings.notion_token:
                self._async_client = AsyncNotionClient(auth=settings.notion_token)
        return self._async_client

    def is_configured(self) -> bool:
        """Check if Notion client is configured."""
        return self.client is not None

    # -------------------------------------------------------------------------
    # Database Operations
    # -------------------------------------------------------------------------

    def query_database(
        self,
        database_id: str,
        filter: dict | None = None,
        sorts: list[dict] | None = None,
        page_size: int = 100,
    ) -> list[dict]:
        """Query a Notion database with optional filter and sort."""
        if not self.is_configured():
            return []

        try:
            return self._query_database_with_retry(database_id, filter, sorts, page_size)
        except (APIResponseError, MaxRetriesError) as e:
            return [{"error": str(e)}]

    @resilient_call("notion")
    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _query_database_with_retry(
        self,
        database_id: str,
        filter: dict | None,
        sorts: list[dict] | None,
        page_size: int,
    ) -> list[dict]:
        """Query database with retry logic and resilience."""
        query_params: dict[str, Any] = {
            "database_id": database_id,
            "page_size": page_size,
        }
        if filter:
            query_params["filter"] = filter
        if sorts:
            query_params["sorts"] = sorts

        results = self.client.databases.query(**query_params)
        return [self._parse_page_properties(page) for page in results.get("results", [])]

    def get_database_schema(self, database_id: str) -> dict | None:
        """Get database schema (properties and their types).

        Results are cached for 1 hour since schemas rarely change.
        """
        if not self.is_configured():
            return None

        try:
            return self._get_database_schema_cached(database_id)
        except (APIResponseError, MaxRetriesError) as e:
            return {"error": str(e)}

    @cached(ttl=3600, prefix="notion:schema")  # Cache for 1 hour
    def _get_database_schema_cached(self, database_id: str) -> dict:
        """Get database schema with caching and retry."""
        return self._get_database_schema_with_retry(database_id)

    @resilient_call("notion")
    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _get_database_schema_with_retry(self, database_id: str) -> dict:
        """Get database schema with retry logic and resilience."""
        db = self.client.databases.retrieve(database_id=database_id)
        return {
            "id": db["id"],
            "title": self._extract_title(db.get("title", [])),
            "properties": {
                name: {"type": prop["type"], "id": prop["id"]}
                for name, prop in db.get("properties", {}).items()
            },
        }

    def create_database_entry(self, database_id: str, properties: dict[str, Any]) -> dict | None:
        """Create a database entry with custom properties."""
        if not self.is_configured():
            return None

        try:
            return self._create_database_entry_with_retry(database_id, properties)
        except (APIResponseError, MaxRetriesError) as e:
            return {"error": str(e)}

    @resilient_call("notion")
    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _create_database_entry_with_retry(
        self, database_id: str, properties: dict[str, Any]
    ) -> dict:
        """Create database entry with retry logic and resilience."""
        page = self.client.pages.create(
            parent={"database_id": database_id},
            properties=properties,
        )
        return {"id": page["id"], "url": page["url"]}

    # -------------------------------------------------------------------------
    # Page Operations
    # -------------------------------------------------------------------------

    def read_page_content(self, page_id: str) -> list[dict]:
        """Read all blocks from a page."""
        if not self.is_configured():
            return []

        try:
            return self._read_page_content_with_retry(page_id)
        except (APIResponseError, MaxRetriesError) as e:
            return [{"error": str(e)}]

    @resilient_call("notion")
    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _read_page_content_with_retry(self, page_id: str) -> list[dict]:
        """Read page content with retry logic and resilience."""
        blocks = self.client.blocks.children.list(block_id=page_id)
        return [self._parse_block(block) for block in blocks.get("results", [])]

    def get_page(self, page_id: str) -> dict | None:
        """Get page metadata and properties."""
        if not self.is_configured():
            return None

        try:
            return self._get_page_with_retry(page_id)
        except (APIResponseError, MaxRetriesError) as e:
            return {"error": str(e)}

    @resilient_call("notion")
    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _get_page_with_retry(self, page_id: str) -> dict:
        """Get page with retry logic."""
        page = self.client.pages.retrieve(page_id=page_id)
        return self._parse_page_properties(page)

    def update_page(self, page_id: str, properties: dict[str, Any]) -> dict | None:
        """Update page properties."""
        if not self.is_configured():
            return None

        try:
            return self._update_page_with_retry(page_id, properties)
        except (APIResponseError, MaxRetriesError) as e:
            return {"error": str(e)}

    @resilient_call("notion")
    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _update_page_with_retry(self, page_id: str, properties: dict[str, Any]) -> dict:
        """Update page with retry logic."""
        page = self.client.pages.update(page_id=page_id, properties=properties)
        return {"id": page["id"], "url": page["url"]}

    def archive_page(self, page_id: str) -> dict | None:
        """Archive (soft delete) a page."""
        if not self.is_configured():
            return None

        try:
            return self._archive_page_with_retry(page_id)
        except (APIResponseError, MaxRetriesError) as e:
            return {"error": str(e)}

    @resilient_call("notion")
    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _archive_page_with_retry(self, page_id: str) -> dict:
        """Archive page with retry logic."""
        page = self.client.pages.update(page_id=page_id, archived=True)
        return {"id": page["id"], "archived": True}

    # -------------------------------------------------------------------------
    # Block Operations
    # -------------------------------------------------------------------------

    def delete_block(self, block_id: str) -> dict | None:
        """Delete a block."""
        if not self.is_configured():
            return None

        try:
            return self._delete_block_with_retry(block_id)
        except (APIResponseError, MaxRetriesError) as e:
            return {"error": str(e)}

    @resilient_call("notion")
    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _delete_block_with_retry(self, block_id: str) -> dict:
        """Delete block with retry logic."""
        self.client.blocks.delete(block_id=block_id)
        return {"deleted": True, "block_id": block_id}

    def update_block(self, block_id: str, content: str) -> dict | None:
        """Update a paragraph block's content."""
        if not self.is_configured():
            return None

        try:
            return self._update_block_with_retry(block_id, content)
        except (APIResponseError, MaxRetriesError) as e:
            return {"error": str(e)}

    @resilient_call("notion")
    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _update_block_with_retry(self, block_id: str, content: str) -> dict:
        """Update block with retry logic."""
        block = self.client.blocks.update(
            block_id=block_id,
            paragraph={"rich_text": [{"type": "text", "text": {"content": content}}]},
        )
        return {"id": block["id"], "type": block["type"]}

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _parse_block(self, block: dict) -> dict:
        """Parse a block into a simplified format."""
        block_type = block.get("type", "unknown")
        parsed = {
            "id": block["id"],
            "type": block_type,
            "has_children": block.get("has_children", False),
        }

        # Extract text content from common block types
        if block_type in (
            "paragraph",
            "heading_1",
            "heading_2",
            "heading_3",
            "bulleted_list_item",
            "numbered_list_item",
            "quote",
            "callout",
        ):
            rich_text = block.get(block_type, {}).get("rich_text", [])
            parsed["text"] = self._extract_rich_text(rich_text)
        elif block_type == "code":
            code_block = block.get("code", {})
            parsed["text"] = self._extract_rich_text(code_block.get("rich_text", []))
            parsed["language"] = code_block.get("language", "plain text")
        elif block_type == "to_do":
            todo = block.get("to_do", {})
            parsed["text"] = self._extract_rich_text(todo.get("rich_text", []))
            parsed["checked"] = todo.get("checked", False)

        return parsed

    def _parse_page_properties(self, page: dict) -> dict:
        """Parse page properties into a simplified format."""
        parsed = {
            "id": page["id"],
            "url": page.get("url", ""),
            "created_time": page.get("created_time", ""),
            "last_edited_time": page.get("last_edited_time", ""),
            "properties": {},
        }

        for name, prop in page.get("properties", {}).items():
            parsed["properties"][name] = self._extract_property_value(prop)

        return parsed

    def _extract_property_value(self, prop: dict) -> Any:
        """Extract value from a Notion property."""
        prop_type = prop.get("type", "")

        # Simple direct extractions
        simple_types = {"number", "checkbox", "url", "email", "phone_number"}
        if prop_type in simple_types:
            return prop.get(prop_type) if prop_type != "checkbox" else prop.get("checkbox", False)

        # Rich text types
        if prop_type in ("title", "rich_text"):
            return self._extract_rich_text(prop.get(prop_type, []))

        # Types that extract .name from nested object
        name_types = {"select", "status"}
        if prop_type in name_types:
            obj = prop.get(prop_type)
            return obj.get("name") if obj else None

        # Array extractions
        if prop_type == "multi_select":
            return [s.get("name") for s in prop.get("multi_select", [])]
        if prop_type == "relation":
            return [r.get("id") for r in prop.get("relation", [])]

        # Date extraction
        if prop_type == "date":
            date = prop.get("date")
            return date.get("start") if date else None

        # Formula extraction
        if prop_type == "formula":
            formula = prop.get("formula", {})
            return formula.get(formula.get("type"))

        return None

    def _extract_rich_text(self, rich_text: list[dict]) -> str:
        """Extract plain text from rich text array."""
        return "".join(t.get("text", {}).get("content", "") for t in rich_text)

    def _extract_title(self, title: list[dict]) -> str:
        """Extract title text."""
        return self._extract_rich_text(title)

    def create_page(self, parent_id: str, title: str, content: str) -> dict | None:
        """Create a page in Notion."""
        if not self.is_configured():
            return None

        try:
            return self._create_page_with_retry(parent_id, title, content)
        except (APIResponseError, MaxRetriesError) as e:
            return {"error": str(e)}

    @resilient_call("notion")
    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _create_page_with_retry(self, parent_id: str, title: str, content: str) -> dict:
        """Create page with retry logic."""
        page = self.client.pages.create(
            parent={"database_id": parent_id},
            properties={"Name": {"title": [{"text": {"content": title}}]}},
            children=[
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {"content": content}}]},
                }
            ],
        )
        return {"id": page["id"], "url": page["url"]}

    def append_to_page(self, page_id: str, content: str) -> dict | None:
        """Append content to an existing Notion page."""
        if not self.is_configured():
            return None

        try:
            return self._append_to_page_with_retry(page_id, content)
        except (APIResponseError, MaxRetriesError) as e:
            return {"error": str(e)}

    @resilient_call("notion")
    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _append_to_page_with_retry(self, page_id: str, content: str) -> dict:
        """Append to page with retry logic."""
        block = self.client.blocks.children.append(
            block_id=page_id,
            children=[
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {"content": content}}]},
                }
            ],
        )
        return {"success": True, "block_id": block["results"][0]["id"]}

    def search_pages(self, query: str) -> list[dict]:
        """Search for pages in Notion."""
        if not self.is_configured():
            return []

        try:
            return self._search_pages_with_retry(query)
        except (APIResponseError, MaxRetriesError):
            return []

    @resilient_call("notion")
    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _search_pages_with_retry(self, query: str) -> list[dict]:
        """Search pages with retry logic."""
        results = self.client.search(query=query, filter={"property": "object", "value": "page"})
        return [
            {
                "id": page["id"],
                "title": page.get("properties", {})
                .get("Name", {})
                .get("title", [{}])[0]
                .get("text", {})
                .get("content", "Untitled"),
                "url": page.get("url", ""),
            }
            for page in results.get("results", [])
        ]

    # -------------------------------------------------------------------------
    # Async Database Operations
    # -------------------------------------------------------------------------

    def is_async_configured(self) -> bool:
        """Check if async Notion client is configured."""
        return self.async_client is not None

    async def query_database_async(
        self,
        database_id: str,
        filter: dict | None = None,
        sorts: list[dict] | None = None,
        page_size: int = 100,
    ) -> list[dict]:
        """Query a Notion database with optional filter and sort (async)."""
        if not self.is_async_configured():
            return []

        try:
            return await self._query_database_with_retry_async(
                database_id, filter, sorts, page_size
            )
        except (APIResponseError, MaxRetriesError) as e:
            return [{"error": str(e)}]

    @resilient_call_async("notion")
    @async_with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    async def _query_database_with_retry_async(
        self,
        database_id: str,
        filter: dict | None,
        sorts: list[dict] | None,
        page_size: int,
    ) -> list[dict]:
        """Query database with retry logic (async)."""
        query_params: dict[str, Any] = {
            "database_id": database_id,
            "page_size": page_size,
        }
        if filter:
            query_params["filter"] = filter
        if sorts:
            query_params["sorts"] = sorts

        results = await self.async_client.databases.query(**query_params)
        return [self._parse_page_properties(page) for page in results.get("results", [])]

    async def get_database_schema_async(self, database_id: str) -> dict | None:
        """Get database schema (properties and their types) (async)."""
        if not self.is_async_configured():
            return None

        try:
            return await self._get_database_schema_with_retry_async(database_id)
        except (APIResponseError, MaxRetriesError) as e:
            return {"error": str(e)}

    @resilient_call_async("notion")
    @async_with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    async def _get_database_schema_with_retry_async(self, database_id: str) -> dict:
        """Get database schema with retry logic (async)."""
        db = await self.async_client.databases.retrieve(database_id=database_id)
        return {
            "id": db["id"],
            "title": self._extract_title(db.get("title", [])),
            "properties": {
                name: {"type": prop["type"], "id": prop["id"]}
                for name, prop in db.get("properties", {}).items()
            },
        }

    # -------------------------------------------------------------------------
    # Async Page Operations
    # -------------------------------------------------------------------------

    async def get_page_async(self, page_id: str) -> dict | None:
        """Get page metadata and properties (async)."""
        if not self.is_async_configured():
            return None

        try:
            return await self._get_page_with_retry_async(page_id)
        except (APIResponseError, MaxRetriesError) as e:
            return {"error": str(e)}

    @resilient_call_async("notion")
    @async_with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    async def _get_page_with_retry_async(self, page_id: str) -> dict:
        """Get page with retry logic (async)."""
        page = await self.async_client.pages.retrieve(page_id=page_id)
        return self._parse_page_properties(page)

    async def read_page_content_async(self, page_id: str) -> list[dict]:
        """Read all blocks from a page (async)."""
        if not self.is_async_configured():
            return []

        try:
            return await self._read_page_content_with_retry_async(page_id)
        except (APIResponseError, MaxRetriesError) as e:
            return [{"error": str(e)}]

    @resilient_call_async("notion")
    @async_with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    async def _read_page_content_with_retry_async(self, page_id: str) -> list[dict]:
        """Read page content with retry logic (async)."""
        blocks = await self.async_client.blocks.children.list(block_id=page_id)
        return [self._parse_block(block) for block in blocks.get("results", [])]

    async def create_page_async(self, parent_id: str, title: str, content: str) -> dict | None:
        """Create a page in Notion (async)."""
        if not self.is_async_configured():
            return None

        try:
            return await self._create_page_with_retry_async(parent_id, title, content)
        except (APIResponseError, MaxRetriesError) as e:
            return {"error": str(e)}

    @resilient_call_async("notion")
    @async_with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    async def _create_page_with_retry_async(self, parent_id: str, title: str, content: str) -> dict:
        """Create page with retry logic (async)."""
        page = await self.async_client.pages.create(
            parent={"database_id": parent_id},
            properties={"Name": {"title": [{"text": {"content": title}}]}},
            children=[
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {"content": content}}]},
                }
            ],
        )
        return {"id": page["id"], "url": page["url"]}

    async def update_page_async(self, page_id: str, properties: dict[str, Any]) -> dict | None:
        """Update page properties (async)."""
        if not self.is_async_configured():
            return None

        try:
            return await self._update_page_with_retry_async(page_id, properties)
        except (APIResponseError, MaxRetriesError) as e:
            return {"error": str(e)}

    @resilient_call_async("notion")
    @async_with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    async def _update_page_with_retry_async(self, page_id: str, properties: dict[str, Any]) -> dict:
        """Update page with retry logic (async)."""
        page = await self.async_client.pages.update(page_id=page_id, properties=properties)
        return {"id": page["id"], "url": page["url"]}

    # -------------------------------------------------------------------------
    # Async Search
    # -------------------------------------------------------------------------

    async def search_pages_async(self, query: str) -> list[dict]:
        """Search for pages in Notion (async)."""
        if not self.is_async_configured():
            return []

        try:
            return await self._search_pages_with_retry_async(query)
        except (APIResponseError, MaxRetriesError):
            return []

    @resilient_call_async("notion")
    @async_with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    async def _search_pages_with_retry_async(self, query: str) -> list[dict]:
        """Search pages with retry logic (async)."""
        results = await self.async_client.search(
            query=query, filter={"property": "object", "value": "page"}
        )
        return [
            {
                "id": page["id"],
                "title": page.get("properties", {})
                .get("Name", {})
                .get("title", [{}])[0]
                .get("text", {})
                .get("content", "Untitled"),
                "url": page.get("url", ""),
            }
            for page in results.get("results", [])
        ]
