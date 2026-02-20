"""Tests for prompts/template_manager.py module."""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError as PydanticValidationError

from animus_forge.errors import ValidationError
from animus_forge.prompts.template_manager import PromptTemplate, PromptTemplateManager

# =============================================================================
# PromptTemplate Model Tests
# =============================================================================


class TestPromptTemplateCreation:
    """Tests for PromptTemplate model creation and validation."""

    def test_create_minimal_template(self):
        """Test creating a template with only required fields."""
        template = PromptTemplate(
            id="test_template",
            name="Test Template",
            description="A test template",
            user_prompt="Hello {name}",
        )
        assert template.id == "test_template"
        assert template.name == "Test Template"
        assert template.description == "A test template"
        assert template.user_prompt == "Hello {name}"

    def test_create_full_template(self):
        """Test creating a template with all fields."""
        template = PromptTemplate(
            id="full_template",
            name="Full Template",
            description="A template with all fields",
            system_prompt="You are a helpful assistant.",
            user_prompt="Process this: {input}",
            variables=["input"],
            model="gpt-4",
            temperature=0.5,
        )
        assert template.id == "full_template"
        assert template.system_prompt == "You are a helpful assistant."
        assert template.variables == ["input"]
        assert template.model == "gpt-4"
        assert template.temperature == 0.5

    def test_default_values(self):
        """Test that default values are set correctly."""
        template = PromptTemplate(
            id="defaults",
            name="Defaults",
            description="Test defaults",
            user_prompt="Hello",
        )
        assert template.system_prompt is None
        assert template.variables == []
        assert template.model == "gpt-4o-mini"
        assert template.temperature == 0.7

    def test_missing_required_field_id(self):
        """Test that missing id raises validation error."""
        with pytest.raises(PydanticValidationError) as exc_info:
            PromptTemplate(
                name="Test",
                description="Test",
                user_prompt="Hello",
            )
        assert "id" in str(exc_info.value)

    def test_missing_required_field_name(self):
        """Test that missing name raises validation error."""
        with pytest.raises(PydanticValidationError) as exc_info:
            PromptTemplate(
                id="test",
                description="Test",
                user_prompt="Hello",
            )
        assert "name" in str(exc_info.value)

    def test_missing_required_field_description(self):
        """Test that missing description raises validation error."""
        with pytest.raises(PydanticValidationError) as exc_info:
            PromptTemplate(
                id="test",
                name="Test",
                user_prompt="Hello",
            )
        assert "description" in str(exc_info.value)

    def test_missing_required_field_user_prompt(self):
        """Test that missing user_prompt raises validation error."""
        with pytest.raises(PydanticValidationError) as exc_info:
            PromptTemplate(
                id="test",
                name="Test",
                description="Test",
            )
        assert "user_prompt" in str(exc_info.value)


class TestPromptTemplateIdValidation:
    """Tests for template ID validation."""

    def test_valid_id_alphanumeric(self):
        """Test that alphanumeric ID is valid."""
        template = PromptTemplate(
            id="template123",
            name="Test",
            description="Test",
            user_prompt="Hello",
        )
        assert template.id == "template123"

    def test_valid_id_with_underscore(self):
        """Test that ID with underscore is valid."""
        template = PromptTemplate(
            id="my_template",
            name="Test",
            description="Test",
            user_prompt="Hello",
        )
        assert template.id == "my_template"

    def test_valid_id_with_hyphen(self):
        """Test that ID with hyphen is valid."""
        template = PromptTemplate(
            id="my-template",
            name="Test",
            description="Test",
            user_prompt="Hello",
        )
        assert template.id == "my-template"

    def test_invalid_id_empty(self):
        """Test that empty ID raises validation error."""
        with pytest.raises((PydanticValidationError, ValidationError)):
            PromptTemplate(
                id="",
                name="Test",
                description="Test",
                user_prompt="Hello",
            )

    def test_invalid_id_starts_with_number(self):
        """Test that ID starting with number raises validation error."""
        with pytest.raises((PydanticValidationError, ValidationError)):
            PromptTemplate(
                id="123template",
                name="Test",
                description="Test",
                user_prompt="Hello",
            )

    def test_invalid_id_path_traversal(self):
        """Test that path traversal in ID raises validation error."""
        with pytest.raises((PydanticValidationError, ValidationError)):
            PromptTemplate(
                id="../../../etc/passwd",
                name="Test",
                description="Test",
                user_prompt="Hello",
            )

    def test_invalid_id_with_slash(self):
        """Test that ID with slash raises validation error."""
        with pytest.raises((PydanticValidationError, ValidationError)):
            PromptTemplate(
                id="path/to/template",
                name="Test",
                description="Test",
                user_prompt="Hello",
            )

    def test_invalid_id_with_special_chars(self):
        """Test that ID with special characters raises validation error."""
        with pytest.raises((PydanticValidationError, ValidationError)):
            PromptTemplate(
                id="template@#$",
                name="Test",
                description="Test",
                user_prompt="Hello",
            )


class TestPromptTemplateFormat:
    """Tests for PromptTemplate.format() method."""

    def test_format_single_variable(self):
        """Test formatting with a single variable."""
        template = PromptTemplate(
            id="test",
            name="Test",
            description="Test",
            user_prompt="Hello {name}!",
            variables=["name"],
        )
        result = template.format(name="World")
        assert result == "Hello World!"

    def test_format_multiple_variables(self):
        """Test formatting with multiple variables."""
        template = PromptTemplate(
            id="test",
            name="Test",
            description="Test",
            user_prompt="Hello {first} {last}, you are {age} years old.",
            variables=["first", "last", "age"],
        )
        result = template.format(first="John", last="Doe", age=30)
        assert result == "Hello John Doe, you are 30 years old."

    def test_format_no_variables(self):
        """Test formatting with no variables."""
        template = PromptTemplate(
            id="test",
            name="Test",
            description="Test",
            user_prompt="Hello World!",
        )
        result = template.format()
        assert result == "Hello World!"

    def test_format_missing_variable_raises_error(self):
        """Test that missing variable raises KeyError."""
        template = PromptTemplate(
            id="test",
            name="Test",
            description="Test",
            user_prompt="Hello {name}!",
            variables=["name"],
        )
        with pytest.raises(KeyError):
            template.format()

    def test_format_extra_variables_ignored(self):
        """Test that extra variables are ignored."""
        template = PromptTemplate(
            id="test",
            name="Test",
            description="Test",
            user_prompt="Hello {name}!",
            variables=["name"],
        )
        result = template.format(name="World", extra="ignored")
        assert result == "Hello World!"

    def test_format_repeated_variable(self):
        """Test formatting with repeated variables."""
        template = PromptTemplate(
            id="test",
            name="Test",
            description="Test",
            user_prompt="{name} said: Hello {name}!",
            variables=["name"],
        )
        result = template.format(name="Alice")
        assert result == "Alice said: Hello Alice!"


class TestPromptTemplateModelDump:
    """Tests for PromptTemplate serialization."""

    def test_model_dump(self):
        """Test that model_dump returns expected structure."""
        template = PromptTemplate(
            id="test",
            name="Test",
            description="Test description",
            system_prompt="System",
            user_prompt="User {var}",
            variables=["var"],
            model="gpt-4",
            temperature=0.5,
        )
        data = template.model_dump()
        assert data["id"] == "test"
        assert data["name"] == "Test"
        assert data["description"] == "Test description"
        assert data["system_prompt"] == "System"
        assert data["user_prompt"] == "User {var}"
        assert data["variables"] == ["var"]
        assert data["model"] == "gpt-4"
        assert data["temperature"] == 0.5


# =============================================================================
# PromptTemplateManager Tests
# =============================================================================


@pytest.fixture
def temp_templates_dir(tmp_path):
    """Create a temporary templates directory."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    return templates_dir


@pytest.fixture
def manager(temp_templates_dir):
    """Create a PromptTemplateManager with temporary directory."""
    with patch("animus_forge.prompts.template_manager.get_settings") as mock_settings:
        mock_settings.return_value.prompts_dir = temp_templates_dir
        return PromptTemplateManager()


@pytest.fixture
def sample_template():
    """Create a sample PromptTemplate."""
    return PromptTemplate(
        id="sample",
        name="Sample Template",
        description="A sample template for testing",
        system_prompt="You are a helpful assistant.",
        user_prompt="Process this: {input}",
        variables=["input"],
        model="gpt-4",
        temperature=0.7,
    )


class TestPromptTemplateManagerInit:
    """Tests for PromptTemplateManager initialization."""

    def test_init_creates_directory(self, tmp_path):
        """Test that init creates templates directory if not exists."""
        templates_dir = tmp_path / "new_templates"
        with patch("animus_forge.prompts.template_manager.get_settings") as mock_settings:
            mock_settings.return_value.prompts_dir = templates_dir
            _manager = PromptTemplateManager()
            assert templates_dir.exists()

    def test_init_uses_existing_directory(self, temp_templates_dir):
        """Test that init uses existing templates directory."""
        with patch("animus_forge.prompts.template_manager.get_settings") as mock_settings:
            mock_settings.return_value.prompts_dir = temp_templates_dir
            manager = PromptTemplateManager()
            assert manager.templates_dir == temp_templates_dir


class TestPromptTemplateManagerSave:
    """Tests for PromptTemplateManager.save_template()."""

    def test_save_template_success(self, manager, sample_template, temp_templates_dir):
        """Test saving a template successfully."""
        result = manager.save_template(sample_template)
        assert result is True
        assert (temp_templates_dir / "sample.json").exists()

    def test_save_template_creates_valid_json(self, manager, sample_template, temp_templates_dir):
        """Test that saved template is valid JSON."""
        manager.save_template(sample_template)
        with open(temp_templates_dir / "sample.json") as f:
            data = json.load(f)
        assert data["id"] == "sample"
        assert data["name"] == "Sample Template"

    def test_save_template_overwrites_existing(self, manager, sample_template, temp_templates_dir):
        """Test that saving overwrites existing template."""
        manager.save_template(sample_template)

        # Modify and save again
        modified = PromptTemplate(
            id="sample",
            name="Modified Template",
            description="Modified",
            user_prompt="Modified prompt",
        )
        result = manager.save_template(modified)
        assert result is True

        with open(temp_templates_dir / "sample.json") as f:
            data = json.load(f)
        assert data["name"] == "Modified Template"

    def test_save_template_invalid_id_returns_false(self, manager):
        """Test that invalid template ID returns False."""
        # Create template with patched validation that would fail
        template = MagicMock()
        template.id = "../../../etc/passwd"
        template.model_dump.return_value = {"id": "../../../etc/passwd"}

        result = manager.save_template(template)
        assert result is False

    def test_save_template_file_error_returns_false(
        self, manager, sample_template, temp_templates_dir
    ):
        """Test that file write error returns False."""
        # Make directory read-only
        temp_templates_dir.chmod(0o444)
        try:
            result = manager.save_template(sample_template)
            assert result is False
        finally:
            temp_templates_dir.chmod(0o755)


class TestPromptTemplateManagerLoad:
    """Tests for PromptTemplateManager.load_template()."""

    def test_load_template_success(self, manager, sample_template, temp_templates_dir):
        """Test loading a template successfully."""
        manager.save_template(sample_template)
        loaded = manager.load_template("sample")
        assert loaded is not None
        assert loaded.id == "sample"
        assert loaded.name == "Sample Template"

    def test_load_template_not_found(self, manager):
        """Test loading non-existent template returns None."""
        loaded = manager.load_template("nonexistent")
        assert loaded is None

    def test_load_template_invalid_id_returns_none(self, manager):
        """Test that invalid template ID returns None."""
        loaded = manager.load_template("../../../etc/passwd")
        assert loaded is None

    def test_load_template_corrupted_json_returns_none(self, manager, temp_templates_dir):
        """Test that corrupted JSON returns None."""
        # Create corrupted JSON file
        with open(temp_templates_dir / "corrupted.json", "w") as f:
            f.write("not valid json {{{")

        loaded = manager.load_template("corrupted")
        assert loaded is None

    def test_load_template_invalid_data_returns_none(self, manager, temp_templates_dir):
        """Test that invalid template data returns None."""
        # Create JSON with missing required fields
        with open(temp_templates_dir / "invalid.json", "w") as f:
            json.dump({"id": "invalid"}, f)

        loaded = manager.load_template("invalid")
        assert loaded is None

    def test_load_template_preserves_all_fields(self, manager, sample_template, temp_templates_dir):
        """Test that all fields are preserved on load."""
        manager.save_template(sample_template)
        loaded = manager.load_template("sample")

        assert loaded.id == sample_template.id
        assert loaded.name == sample_template.name
        assert loaded.description == sample_template.description
        assert loaded.system_prompt == sample_template.system_prompt
        assert loaded.user_prompt == sample_template.user_prompt
        assert loaded.variables == sample_template.variables
        assert loaded.model == sample_template.model
        assert loaded.temperature == sample_template.temperature


class TestPromptTemplateManagerList:
    """Tests for PromptTemplateManager.list_templates()."""

    def test_list_templates_empty(self, manager):
        """Test listing templates when directory is empty."""
        templates = manager.list_templates()
        assert templates == []

    def test_list_templates_single(self, manager, sample_template):
        """Test listing a single template."""
        manager.save_template(sample_template)
        templates = manager.list_templates()
        assert len(templates) == 1
        assert templates[0]["id"] == "sample"
        assert templates[0]["name"] == "Sample Template"
        assert templates[0]["description"] == "A sample template for testing"

    def test_list_templates_multiple(self, manager):
        """Test listing multiple templates."""
        for i in range(3):
            template = PromptTemplate(
                id=f"template{i}",
                name=f"Template {i}",
                description=f"Description {i}",
                user_prompt="Hello",
            )
            manager.save_template(template)

        templates = manager.list_templates()
        assert len(templates) == 3
        ids = {t["id"] for t in templates}
        assert ids == {"template0", "template1", "template2"}

    def test_list_templates_ignores_corrupted(self, manager, temp_templates_dir):
        """Test that corrupted files are ignored in listing."""
        # Create valid template
        template = PromptTemplate(
            id="valid",
            name="Valid",
            description="Valid template",
            user_prompt="Hello",
        )
        manager.save_template(template)

        # Create corrupted file
        with open(temp_templates_dir / "corrupted.json", "w") as f:
            f.write("not valid json")

        templates = manager.list_templates()
        assert len(templates) == 1
        assert templates[0]["id"] == "valid"

    def test_list_templates_ignores_non_json(self, manager, temp_templates_dir):
        """Test that non-JSON files are ignored."""
        template = PromptTemplate(
            id="valid",
            name="Valid",
            description="Valid",
            user_prompt="Hello",
        )
        manager.save_template(template)

        # Create non-JSON file
        (temp_templates_dir / "readme.txt").write_text("Not a template")

        templates = manager.list_templates()
        assert len(templates) == 1


class TestPromptTemplateManagerDelete:
    """Tests for PromptTemplateManager.delete_template()."""

    def test_delete_template_success(self, manager, sample_template, temp_templates_dir):
        """Test deleting a template successfully."""
        manager.save_template(sample_template)
        assert (temp_templates_dir / "sample.json").exists()

        result = manager.delete_template("sample")
        assert result is True
        assert not (temp_templates_dir / "sample.json").exists()

    def test_delete_template_not_found(self, manager):
        """Test deleting non-existent template returns False."""
        result = manager.delete_template("nonexistent")
        assert result is False

    def test_delete_template_invalid_id(self, manager):
        """Test that invalid template ID returns False."""
        result = manager.delete_template("../../../etc/passwd")
        assert result is False

    def test_delete_template_removes_from_list(self, manager, sample_template):
        """Test that deleted template is removed from list."""
        manager.save_template(sample_template)
        assert len(manager.list_templates()) == 1

        manager.delete_template("sample")
        assert len(manager.list_templates()) == 0


class TestPromptTemplateManagerDefaultTemplates:
    """Tests for PromptTemplateManager.create_default_templates()."""

    def test_create_default_templates(self, manager):
        """Test creating default templates."""
        manager.create_default_templates()
        templates = manager.list_templates()
        assert len(templates) == 4

        ids = {t["id"] for t in templates}
        expected = {"email_summary", "sop_generator", "meeting_notes", "code_review"}
        assert ids == expected

    def test_default_template_email_summary(self, manager):
        """Test email_summary default template."""
        manager.create_default_templates()
        template = manager.load_template("email_summary")

        assert template is not None
        assert template.name == "Email Summary"
        assert "email" in template.description.lower()
        assert "{email_content}" in template.user_prompt
        assert "email_content" in template.variables

    def test_default_template_sop_generator(self, manager):
        """Test sop_generator default template."""
        manager.create_default_templates()
        template = manager.load_template("sop_generator")

        assert template is not None
        assert template.name == "SOP Generator"
        assert "{task_description}" in template.user_prompt
        assert "task_description" in template.variables

    def test_default_template_meeting_notes(self, manager):
        """Test meeting_notes default template."""
        manager.create_default_templates()
        template = manager.load_template("meeting_notes")

        assert template is not None
        assert template.name == "Meeting Notes"
        assert "{transcript}" in template.user_prompt
        assert "transcript" in template.variables

    def test_default_template_code_review(self, manager):
        """Test code_review default template."""
        manager.create_default_templates()
        template = manager.load_template("code_review")

        assert template is not None
        assert template.name == "Code Review"
        assert "{code}" in template.user_prompt
        assert "code" in template.variables

    def test_default_templates_idempotent(self, manager):
        """Test that calling create_default_templates multiple times is safe."""
        manager.create_default_templates()
        manager.create_default_templates()
        templates = manager.list_templates()
        assert len(templates) == 4


class TestPromptTemplateManagerGetTemplatePath:
    """Tests for PromptTemplateManager._get_template_path()."""

    def test_get_template_path_valid(self, manager, temp_templates_dir):
        """Test getting path for valid template ID."""
        path = manager._get_template_path("my_template")
        assert path == temp_templates_dir / "my_template.json"

    def test_get_template_path_with_hyphen(self, manager, temp_templates_dir):
        """Test getting path for ID with hyphen."""
        path = manager._get_template_path("my-template")
        assert path == temp_templates_dir / "my-template.json"

    def test_get_template_path_invalid_raises(self, manager):
        """Test that invalid ID raises ValidationError."""
        with pytest.raises(ValidationError):
            manager._get_template_path("../../../etc")

    def test_get_template_path_empty_raises(self, manager):
        """Test that empty ID raises ValidationError."""
        with pytest.raises(ValidationError):
            manager._get_template_path("")


# =============================================================================
# Security Tests
# =============================================================================


class TestSecurityPathTraversal:
    """Security tests for path traversal prevention."""

    def test_save_path_traversal_blocked(self, manager):
        """Test that path traversal is blocked on save."""
        # This would fail at PromptTemplate creation due to ID validation
        # But we test the manager layer as well
        result = manager.save_template(MagicMock(id="../../../etc/passwd", model_dump=lambda: {}))
        assert result is False

    def test_load_path_traversal_blocked(self, manager):
        """Test that path traversal is blocked on load."""
        result = manager.load_template("../../../etc/passwd")
        assert result is None

    def test_delete_path_traversal_blocked(self, manager):
        """Test that path traversal is blocked on delete."""
        result = manager.delete_template("../../../etc/passwd")
        assert result is False

    def test_absolute_path_blocked(self, manager):
        """Test that absolute path in ID is blocked."""
        result = manager.load_template("/etc/passwd")
        assert result is None

    def test_null_byte_injection_blocked(self, manager):
        """Test that null byte injection is blocked."""
        result = manager.load_template("template\x00.json")
        assert result is None


class TestSecurityInputValidation:
    """Security tests for input validation."""

    def test_special_characters_in_id_blocked(self, manager):
        """Test that special characters in ID are blocked."""
        invalid_ids = [
            "template;rm -rf",
            "template$(whoami)",
            "template`id`",
            "template|cat /etc/passwd",
            "template&&echo pwned",
        ]
        for invalid_id in invalid_ids:
            result = manager.load_template(invalid_id)
            assert result is None, f"ID should be blocked: {invalid_id}"

    def test_long_id_handled(self, manager):
        """Test that very long IDs are handled."""
        long_id = "a" * 1000
        result = manager.load_template(long_id)
        assert result is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestPromptTemplateManagerIntegration:
    """Integration tests for full workflow."""

    def test_full_workflow(self, manager):
        """Test complete create-read-update-delete workflow."""
        # Create
        template = PromptTemplate(
            id="workflow_test",
            name="Workflow Test",
            description="Testing full workflow",
            user_prompt="Hello {name}",
            variables=["name"],
        )
        assert manager.save_template(template) is True

        # Read
        loaded = manager.load_template("workflow_test")
        assert loaded is not None
        assert loaded.name == "Workflow Test"

        # Update
        updated = PromptTemplate(
            id="workflow_test",
            name="Updated Workflow",
            description="Updated description",
            user_prompt="Goodbye {name}",
            variables=["name"],
        )
        assert manager.save_template(updated) is True

        loaded = manager.load_template("workflow_test")
        assert loaded.name == "Updated Workflow"
        assert loaded.user_prompt == "Goodbye {name}"

        # Delete
        assert manager.delete_template("workflow_test") is True
        assert manager.load_template("workflow_test") is None

    def test_format_loaded_template(self, manager):
        """Test formatting a template after loading."""
        template = PromptTemplate(
            id="format_test",
            name="Format Test",
            description="Test formatting",
            user_prompt="Hello {name}, welcome to {place}!",
            variables=["name", "place"],
        )
        manager.save_template(template)

        loaded = manager.load_template("format_test")
        result = loaded.format(name="Alice", place="Wonderland")
        assert result == "Hello Alice, welcome to Wonderland!"

    def test_multiple_managers_same_directory(self, temp_templates_dir):
        """Test that multiple managers can access same directory."""
        with patch("animus_forge.prompts.template_manager.get_settings") as mock_settings:
            mock_settings.return_value.prompts_dir = temp_templates_dir

            manager1 = PromptTemplateManager()
            manager2 = PromptTemplateManager()

            # Save with manager1
            template = PromptTemplate(
                id="shared",
                name="Shared Template",
                description="Shared",
                user_prompt="Hello",
            )
            manager1.save_template(template)

            # Load with manager2
            loaded = manager2.load_template("shared")
            assert loaded is not None
            assert loaded.name == "Shared Template"
