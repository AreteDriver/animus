"""Tests for Notion API client wrapper."""

from unittest.mock import Mock, patch

import pytest


class TestNotionClientWrapper:
    """Tests for NotionClientWrapper."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings with Notion token."""
        with patch("animus_forge.api_clients.notion_client.get_settings") as mock:
            mock.return_value = Mock(notion_token="secret_test_token")
            yield mock

    @pytest.fixture
    def mock_notion_client(self):
        """Mock the Notion SDK client."""
        with patch("animus_forge.api_clients.notion_client.NotionClient") as mock:
            yield mock

    @pytest.fixture
    def client(self, mock_settings, mock_notion_client):
        """Create a NotionClientWrapper with mocked dependencies."""
        from animus_forge.api_clients.notion_client import NotionClientWrapper

        return NotionClientWrapper()

    # -------------------------------------------------------------------------
    # Configuration Tests
    # -------------------------------------------------------------------------

    def test_is_configured_with_token(self, mock_settings, mock_notion_client):
        """Test client is configured when token is present."""
        from animus_forge.api_clients.notion_client import NotionClientWrapper

        wrapper = NotionClientWrapper()
        assert wrapper.is_configured() is True

    def test_is_not_configured_without_token(self, mock_notion_client):
        """Test client is not configured when token is missing."""
        with patch("animus_forge.api_clients.notion_client.get_settings") as mock:
            mock.return_value = Mock(notion_token=None)
            from animus_forge.api_clients.notion_client import NotionClientWrapper

            wrapper = NotionClientWrapper()
            assert wrapper.is_configured() is False

    # -------------------------------------------------------------------------
    # Database Operations Tests
    # -------------------------------------------------------------------------

    def test_query_database_success(self, client):
        """Test successful database query."""
        client.client.databases.query.return_value = {
            "results": [
                {
                    "id": "page-1",
                    "url": "https://notion.so/page-1",
                    "created_time": "2024-01-01T00:00:00.000Z",
                    "last_edited_time": "2024-01-02T00:00:00.000Z",
                    "properties": {
                        "Name": {
                            "type": "title",
                            "title": [{"text": {"content": "Test Page"}}],
                        },
                        "Status": {"type": "select", "select": {"name": "Done"}},
                    },
                }
            ]
        }

        results = client.query_database("db-123")

        assert len(results) == 1
        assert results[0]["id"] == "page-1"
        assert results[0]["properties"]["Name"] == "Test Page"
        assert results[0]["properties"]["Status"] == "Done"

    def test_query_database_with_filter(self, client):
        """Test database query with filter."""
        client.client.databases.query.return_value = {"results": []}

        filter_params = {"property": "Status", "select": {"equals": "Done"}}
        client.query_database("db-123", filter=filter_params)

        client.client.databases.query.assert_called_once()
        call_kwargs = client.client.databases.query.call_args[1]
        assert call_kwargs["filter"] == filter_params

    def test_query_database_with_sorts(self, client):
        """Test database query with sorts."""
        client.client.databases.query.return_value = {"results": []}

        sorts = [{"property": "Created", "direction": "descending"}]
        client.query_database("db-123", sorts=sorts)

        call_kwargs = client.client.databases.query.call_args[1]
        assert call_kwargs["sorts"] == sorts

    def test_query_database_not_configured(self, mock_notion_client):
        """Test query_database returns empty when not configured."""
        with patch("animus_forge.api_clients.notion_client.get_settings") as mock:
            mock.return_value = Mock(notion_token=None)
            from animus_forge.api_clients.notion_client import NotionClientWrapper

            wrapper = NotionClientWrapper()
            results = wrapper.query_database("db-123")
            assert results == []

    def test_get_database_schema_success(self, client):
        """Test successful database schema retrieval."""
        client.client.databases.retrieve.return_value = {
            "id": "db-123",
            "title": [{"text": {"content": "My Database"}}],
            "properties": {
                "Name": {"type": "title", "id": "title"},
                "Status": {"type": "select", "id": "abc123"},
                "Tags": {"type": "multi_select", "id": "def456"},
            },
        }

        schema = client.get_database_schema("db-123")

        assert schema["id"] == "db-123"
        assert schema["title"] == "My Database"
        assert schema["properties"]["Name"]["type"] == "title"
        assert schema["properties"]["Status"]["type"] == "select"
        assert schema["properties"]["Tags"]["type"] == "multi_select"

    def test_create_database_entry_success(self, client):
        """Test successful database entry creation."""
        client.client.pages.create.return_value = {
            "id": "new-page-id",
            "url": "https://notion.so/new-page",
        }

        properties = {
            "Name": {"title": [{"text": {"content": "New Entry"}}]},
            "Status": {"select": {"name": "In Progress"}},
        }
        result = client.create_database_entry("db-123", properties)

        assert result["id"] == "new-page-id"
        assert result["url"] == "https://notion.so/new-page"

    # -------------------------------------------------------------------------
    # Page Operations Tests
    # -------------------------------------------------------------------------

    def test_create_page_success(self, client):
        """Test successful page creation."""
        client.client.pages.create.return_value = {
            "id": "page-id",
            "url": "https://notion.so/page",
        }

        result = client.create_page("db-123", "Test Title", "Test content")

        assert result["id"] == "page-id"
        assert result["url"] == "https://notion.so/page"

    def test_get_page_success(self, client):
        """Test successful page retrieval."""
        client.client.pages.retrieve.return_value = {
            "id": "page-123",
            "url": "https://notion.so/page-123",
            "created_time": "2024-01-01T00:00:00.000Z",
            "last_edited_time": "2024-01-02T00:00:00.000Z",
            "properties": {
                "Name": {"type": "title", "title": [{"text": {"content": "My Page"}}]},
            },
        }

        result = client.get_page("page-123")

        assert result["id"] == "page-123"
        assert result["properties"]["Name"] == "My Page"

    def test_read_page_content_success(self, client):
        """Test successful page content reading."""
        client.client.blocks.children.list.return_value = {
            "results": [
                {
                    "id": "block-1",
                    "type": "paragraph",
                    "has_children": False,
                    "paragraph": {"rich_text": [{"text": {"content": "First paragraph"}}]},
                },
                {
                    "id": "block-2",
                    "type": "heading_1",
                    "has_children": False,
                    "heading_1": {"rich_text": [{"text": {"content": "Main Heading"}}]},
                },
                {
                    "id": "block-3",
                    "type": "code",
                    "has_children": False,
                    "code": {
                        "rich_text": [{"text": {"content": "print('hello')"}}],
                        "language": "python",
                    },
                },
                {
                    "id": "block-4",
                    "type": "to_do",
                    "has_children": False,
                    "to_do": {
                        "rich_text": [{"text": {"content": "Complete task"}}],
                        "checked": True,
                    },
                },
            ]
        }

        blocks = client.read_page_content("page-123")

        assert len(blocks) == 4
        assert blocks[0]["text"] == "First paragraph"
        assert blocks[1]["text"] == "Main Heading"
        assert blocks[2]["text"] == "print('hello')"
        assert blocks[2]["language"] == "python"
        assert blocks[3]["text"] == "Complete task"
        assert blocks[3]["checked"] is True

    def test_update_page_success(self, client):
        """Test successful page update."""
        client.client.pages.update.return_value = {
            "id": "page-123",
            "url": "https://notion.so/page-123",
        }

        properties = {"Status": {"select": {"name": "Completed"}}}
        result = client.update_page("page-123", properties)

        assert result["id"] == "page-123"
        client.client.pages.update.assert_called_once_with(
            page_id="page-123", properties=properties
        )

    def test_archive_page_success(self, client):
        """Test successful page archiving."""
        client.client.pages.update.return_value = {
            "id": "page-123",
            "archived": True,
        }

        result = client.archive_page("page-123")

        assert result["id"] == "page-123"
        assert result["archived"] is True
        client.client.pages.update.assert_called_once_with(page_id="page-123", archived=True)

    def test_append_to_page_success(self, client):
        """Test successful content append to page."""
        client.client.blocks.children.append.return_value = {"results": [{"id": "new-block-id"}]}

        result = client.append_to_page("page-123", "Additional content")

        assert result["success"] is True
        assert result["block_id"] == "new-block-id"

    # -------------------------------------------------------------------------
    # Block Operations Tests
    # -------------------------------------------------------------------------

    def test_delete_block_success(self, client):
        """Test successful block deletion."""
        client.client.blocks.delete.return_value = {}

        result = client.delete_block("block-123")

        assert result["deleted"] is True
        assert result["block_id"] == "block-123"
        client.client.blocks.delete.assert_called_once_with(block_id="block-123")

    def test_update_block_success(self, client):
        """Test successful block update."""
        client.client.blocks.update.return_value = {
            "id": "block-123",
            "type": "paragraph",
        }

        result = client.update_block("block-123", "Updated content")

        assert result["id"] == "block-123"
        assert result["type"] == "paragraph"

    # -------------------------------------------------------------------------
    # Search Tests
    # -------------------------------------------------------------------------

    def test_search_pages_success(self, client):
        """Test successful page search."""
        client.client.search.return_value = {
            "results": [
                {
                    "id": "page-1",
                    "url": "https://notion.so/page-1",
                    "properties": {"Name": {"title": [{"text": {"content": "Search Result 1"}}]}},
                },
                {
                    "id": "page-2",
                    "url": "https://notion.so/page-2",
                    "properties": {"Name": {"title": [{"text": {"content": "Search Result 2"}}]}},
                },
            ]
        }

        results = client.search_pages("test query")

        assert len(results) == 2
        assert results[0]["id"] == "page-1"
        assert results[0]["title"] == "Search Result 1"
        assert results[1]["id"] == "page-2"
        assert results[1]["title"] == "Search Result 2"

    def test_search_pages_not_configured(self, mock_notion_client):
        """Test search returns empty when not configured."""
        with patch("animus_forge.api_clients.notion_client.get_settings") as mock:
            mock.return_value = Mock(notion_token=None)
            from animus_forge.api_clients.notion_client import NotionClientWrapper

            wrapper = NotionClientWrapper()
            results = wrapper.search_pages("query")
            assert results == []

    # -------------------------------------------------------------------------
    # Property Parsing Tests
    # -------------------------------------------------------------------------

    def test_extract_property_value_types(self, client):
        """Test parsing of various property types."""
        # Number
        assert client._extract_property_value({"type": "number", "number": 42}) == 42

        # Checkbox
        assert client._extract_property_value({"type": "checkbox", "checkbox": True}) is True

        # URL
        assert (
            client._extract_property_value({"type": "url", "url": "https://example.com"})
            == "https://example.com"
        )

        # Email
        assert (
            client._extract_property_value({"type": "email", "email": "test@example.com"})
            == "test@example.com"
        )

        # Phone
        assert (
            client._extract_property_value({"type": "phone_number", "phone_number": "555-1234"})
            == "555-1234"
        )

        # Date
        assert (
            client._extract_property_value({"type": "date", "date": {"start": "2024-01-01"}})
            == "2024-01-01"
        )

        # Multi-select
        multi_select = {
            "type": "multi_select",
            "multi_select": [{"name": "Tag1"}, {"name": "Tag2"}],
        }
        assert client._extract_property_value(multi_select) == ["Tag1", "Tag2"]

        # Relation
        relation = {
            "type": "relation",
            "relation": [{"id": "page-1"}, {"id": "page-2"}],
        }
        assert client._extract_property_value(relation) == ["page-1", "page-2"]

        # Status
        status = {"type": "status", "status": {"name": "In Progress"}}
        assert client._extract_property_value(status) == "In Progress"

        # Formula (number result)
        formula = {"type": "formula", "formula": {"type": "number", "number": 100}}
        assert client._extract_property_value(formula) == 100

    def test_extract_rich_text(self, client):
        """Test rich text extraction."""
        rich_text = [
            {"text": {"content": "Hello "}},
            {"text": {"content": "World"}},
        ]
        assert client._extract_rich_text(rich_text) == "Hello World"

    def test_extract_rich_text_empty(self, client):
        """Test rich text extraction with empty input."""
        assert client._extract_rich_text([]) == ""

    # -------------------------------------------------------------------------
    # Error Handling Tests
    # -------------------------------------------------------------------------

    def test_query_database_api_error(self, client):
        """Test query_database handles API errors."""
        from animus_forge.errors import MaxRetriesError

        client.client.databases.query.side_effect = MaxRetriesError("API error", attempts=3)

        results = client.query_database("db-123")

        assert len(results) == 1
        assert "error" in results[0]

    def test_create_page_api_error(self, client):
        """Test create_page handles API errors."""
        from animus_forge.errors import MaxRetriesError

        client.client.pages.create.side_effect = MaxRetriesError("API error", attempts=3)

        result = client.create_page("db-123", "Title", "Content")

        assert "error" in result

    def test_get_page_not_configured(self, mock_notion_client):
        """Test get_page returns None when not configured."""
        with patch("animus_forge.api_clients.notion_client.get_settings") as mock:
            mock.return_value = Mock(notion_token=None)
            from animus_forge.api_clients.notion_client import NotionClientWrapper

            wrapper = NotionClientWrapper()
            result = wrapper.get_page("page-123")
            assert result is None


class TestNotionClientIntegration:
    """Integration-style tests for Notion client workflows."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings with Notion token."""
        with patch("animus_forge.api_clients.notion_client.get_settings") as mock:
            mock.return_value = Mock(notion_token="secret_test_token")
            yield mock

    @pytest.fixture
    def mock_notion_client(self):
        """Mock the Notion SDK client."""
        with patch("animus_forge.api_clients.notion_client.NotionClient") as mock:
            yield mock

    @pytest.fixture
    def client(self, mock_settings, mock_notion_client):
        """Create a NotionClientWrapper with mocked dependencies."""
        from animus_forge.api_clients.notion_client import NotionClientWrapper

        return NotionClientWrapper()

    def test_full_workflow_create_and_update(self, client):
        """Test creating a page and then updating it."""
        # Create page
        client.client.pages.create.return_value = {
            "id": "new-page-id",
            "url": "https://notion.so/new-page",
        }

        create_result = client.create_page("db-123", "Initial Title", "Initial content")
        assert create_result["id"] == "new-page-id"

        # Update page
        client.client.pages.update.return_value = {
            "id": "new-page-id",
            "url": "https://notion.so/new-page",
        }

        update_result = client.update_page(
            "new-page-id", {"Status": {"select": {"name": "Published"}}}
        )
        assert update_result["id"] == "new-page-id"

    def test_full_workflow_query_and_archive(self, client):
        """Test querying a database and archiving matching pages."""
        # Query database
        client.client.databases.query.return_value = {
            "results": [
                {
                    "id": "old-page-1",
                    "url": "https://notion.so/old-page-1",
                    "created_time": "2023-01-01T00:00:00.000Z",
                    "last_edited_time": "2023-01-01T00:00:00.000Z",
                    "properties": {
                        "Name": {
                            "type": "title",
                            "title": [{"text": {"content": "Old Page"}}],
                        },
                        "Status": {"type": "select", "select": {"name": "Archived"}},
                    },
                }
            ]
        }

        results = client.query_database(
            "db-123", filter={"property": "Status", "select": {"equals": "Archived"}}
        )
        assert len(results) == 1

        # Archive the page
        client.client.pages.update.return_value = {
            "id": "old-page-1",
            "archived": True,
        }

        archive_result = client.archive_page(results[0]["id"])
        assert archive_result["archived"] is True
