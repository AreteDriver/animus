"""Prompt template management."""

import json
import logging
import stat
from typing import Any

from pydantic import BaseModel, Field, field_validator

from animus_forge.config import get_settings
from animus_forge.errors import ValidationError
from animus_forge.utils.validation import (
    PathValidator,
    sanitize_prompt_variable,
    validate_identifier,
)

logger = logging.getLogger(__name__)


class PromptTemplate(BaseModel):
    """A reusable prompt template."""

    id: str = Field(..., description="Unique template identifier")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Template description")
    system_prompt: str | None = Field(None, description="System prompt")
    user_prompt: str = Field(..., description="User prompt template")
    variables: list[str] = Field(default_factory=list, description="Variables in the prompt")
    model: str = Field("gpt-4o-mini", description="Default model to use")
    temperature: float = Field(0.7, description="Default temperature")

    @field_validator("id")
    @classmethod
    def validate_template_id(cls, v: str) -> str:
        """Validate template ID is safe for use as filename."""
        return validate_identifier(v, name="template_id")

    def format(self, **kwargs: Any) -> str:
        """Format the prompt with variables.

        All values are sanitized to prevent format string injection.
        Curly braces in user input are escaped so they cannot reference
        Python object attributes or other template variables.
        """
        sanitized = {k: sanitize_prompt_variable(str(v)) for k, v in kwargs.items()}
        return self.user_prompt.format(**sanitized)


class PromptTemplateManager:
    """Manages prompt templates stored as JSON files."""

    def __init__(self):
        settings = get_settings()
        self.templates_dir = settings.prompts_dir
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self._path_validator = PathValidator(
            self.templates_dir,
            allowed_extensions={".json"},
        )

    def _get_template_path(self, template_id: str):
        """Get validated path for a template ID."""
        # Validate the ID first
        validate_identifier(template_id, name="template_id")
        return self.templates_dir / f"{template_id}.json"

    def save_template(self, template: PromptTemplate) -> bool:
        """Save a template to disk.

        The template ID is validated to prevent path traversal.
        """
        try:
            file_path = self._get_template_path(template.id)
            dir_mode = file_path.parent.stat().st_mode
            if not (dir_mode & stat.S_IWUSR):
                raise OSError(f"Directory not writable: {file_path.parent}")
            with open(file_path, "w") as f:
                json.dump(template.model_dump(), f, indent=2)
            return True
        except ValidationError as e:
            logger.error(f"Template ID validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to save template: {e}")
            return False

    def load_template(self, template_id: str) -> PromptTemplate | None:
        """Load a template from disk.

        The template ID is validated to prevent path traversal.
        """
        try:
            file_path = self._get_template_path(template_id)
            with open(file_path) as f:
                data = json.load(f)
            return PromptTemplate(**data)
        except ValidationError as e:
            logger.error(f"Template ID validation failed: {e}")
            return None
        except Exception as e:
            logger.debug(f"Failed to load template {template_id}: {e}")
            return None

    def list_templates(self) -> list[dict]:
        """List all available templates."""
        templates = []
        for file_path in self.templates_dir.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                templates.append(
                    {
                        "id": data.get("id"),
                        "name": data.get("name"),
                        "description": data.get("description"),
                    }
                )
            except Exception:
                continue
        return templates

    def delete_template(self, template_id: str) -> bool:
        """Delete a template.

        The template ID is validated to prevent path traversal.
        """
        try:
            file_path = self._get_template_path(template_id)
            file_path.unlink()
            return True
        except ValidationError as e:
            logger.error(f"Template ID validation failed: {e}")
            return False
        except Exception as e:
            logger.debug(f"Failed to delete template {template_id}: {e}")
            return False

    def create_default_templates(self):
        """Create default templates."""
        default_templates = [
            PromptTemplate(
                id="email_summary",
                name="Email Summary",
                description="Summarize email content",
                system_prompt="You are a helpful assistant that creates concise email summaries.",
                user_prompt="Please summarize this email:\n\n{email_content}",
                variables=["email_content"],
            ),
            PromptTemplate(
                id="sop_generator",
                name="SOP Generator",
                description="Generate Standard Operating Procedures",
                system_prompt="You are an expert at creating clear, detailed Standard Operating Procedures.",
                user_prompt="Create a detailed SOP for: {task_description}",
                variables=["task_description"],
            ),
            PromptTemplate(
                id="meeting_notes",
                name="Meeting Notes",
                description="Generate meeting notes from transcript",
                system_prompt="You are an expert at organizing meeting notes.",
                user_prompt="Create structured meeting notes from this transcript:\n\n{transcript}\n\nInclude: key points, action items, and decisions.",
                variables=["transcript"],
            ),
            PromptTemplate(
                id="code_review",
                name="Code Review",
                description="Generate code review comments",
                system_prompt="You are an experienced software engineer reviewing code.",
                user_prompt="Review this code and provide constructive feedback:\n\n{code}",
                variables=["code"],
            ),
        ]

        for template in default_templates:
            self.save_template(template)
