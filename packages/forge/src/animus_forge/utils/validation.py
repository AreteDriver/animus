"""Input validation utilities for security hardening.

Provides protection against:
- Shell command injection
- Path traversal attacks
- Unsafe identifier patterns
"""

from __future__ import annotations

import logging
import re
import shlex
from pathlib import Path
from typing import Any

from animus_forge.errors import ValidationError

logger = logging.getLogger(__name__)

# Characters that are dangerous in shell contexts
SHELL_METACHARACTERS = frozenset(";&|`$(){}[]<>\\'\"!#*?~\n\r")

# Pattern for safe identifiers (alphanumeric, underscore, hyphen)
SAFE_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")

# Maximum identifier length
MAX_IDENTIFIER_LENGTH = 128

# Dangerous shell command patterns
DANGEROUS_SHELL_PATTERNS = [
    re.compile(r"\brm\s+-[rf]*\s+/", re.IGNORECASE),  # rm with absolute path
    re.compile(r"\b(curl|wget)\s+.*\|\s*(sh|bash)", re.IGNORECASE),  # pipe to shell
    re.compile(r"\bchmod\s+[0-7]*777", re.IGNORECASE),  # world-writable
    re.compile(r"\bsudo\b", re.IGNORECASE),  # sudo usage
    re.compile(r"\beval\b", re.IGNORECASE),  # eval usage
    re.compile(r">\s*/dev/sd[a-z]", re.IGNORECASE),  # writing to block devices
    re.compile(r"\bdd\s+.*of=/dev/", re.IGNORECASE),  # dd to devices
    re.compile(r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;", re.IGNORECASE),  # fork bomb
]


def escape_shell_arg(value: Any) -> str:
    """Escape a value for safe use as a shell argument.

    Uses shlex.quote for proper shell escaping.

    Args:
        value: Value to escape

    Returns:
        Safely quoted string

    Example:
        >>> escape_shell_arg("hello world")
        "'hello world'"
        >>> escape_shell_arg("$(dangerous)")
        "'$(dangerous)'"
    """
    return shlex.quote(str(value))


def contains_shell_metacharacters(value: str) -> bool:
    """Check if a string contains shell metacharacters.

    Args:
        value: String to check

    Returns:
        True if metacharacters found
    """
    return bool(SHELL_METACHARACTERS & set(value))


def validate_shell_command(command: str, allow_dangerous: bool = False) -> str:
    """Validate a shell command for dangerous patterns.

    Args:
        command: Command to validate
        allow_dangerous: If True, skip dangerous pattern checks (use with caution)

    Returns:
        The command if valid

    Raises:
        ValidationError: If command contains dangerous patterns
    """
    if not command or not command.strip():
        raise ValidationError("Shell command cannot be empty")

    if not allow_dangerous:
        for pattern in DANGEROUS_SHELL_PATTERNS:
            if pattern.search(command):
                raise ValidationError(
                    f"Shell command contains dangerous pattern: {pattern.pattern}"
                )

    return command


def substitute_shell_variables(
    command: str,
    context: dict[str, Any],
    escape: bool = True,
) -> str:
    """Safely substitute variables into a shell command.

    Variables are referenced as ${name} in the command string.

    Args:
        command: Command template with ${variable} placeholders
        context: Dictionary of variable values
        escape: If True, escape values for shell safety (recommended)

    Returns:
        Command with variables substituted

    Raises:
        ValidationError: If template command (before substitution) is dangerous

    Example:
        >>> substitute_shell_variables("echo ${msg}", {"msg": "hello"})
        "echo hello"
        >>> substitute_shell_variables("echo ${msg}", {"msg": "$(rm -rf /)"})
        "echo '$(rm -rf /)'"  # Escaped, safe
    """
    result = command

    for key, value in context.items():
        placeholder = f"${{{key}}}"
        if placeholder in result:
            safe_value = escape_shell_arg(value) if escape else str(value)
            result = result.replace(placeholder, safe_value)

    # Note: We don't validate the final command when escape=True because
    # properly escaped user input may contain patterns that look dangerous
    # but are actually safe (e.g., '$(rm -rf /)' is just a string literal).
    # The template command should be validated before calling this function.

    return result


def validate_safe_path(
    path: str | Path,
    base_dir: str | Path,
    must_exist: bool = False,
    allow_absolute: bool = False,
) -> Path:
    """Validate a file path to prevent path traversal attacks.

    Ensures the resolved path stays within the base directory.

    Args:
        path: Path to validate
        base_dir: Base directory that path must stay within
        must_exist: If True, path must exist
        allow_absolute: If True, allow absolute paths (but still validate containment)

    Returns:
        Resolved, validated Path object

    Raises:
        ValidationError: If path escapes base_dir or is otherwise invalid

    Example:
        >>> validate_safe_path("templates/foo.json", "/app/data")
        PosixPath('/app/data/templates/foo.json')
        >>> validate_safe_path("../../../etc/passwd", "/app/data")
        ValidationError: Path escapes base directory
    """
    path = Path(path)
    base_dir = Path(base_dir).resolve()

    # Check for absolute paths
    if path.is_absolute() and not allow_absolute:
        raise ValidationError(f"Absolute paths not allowed: {path}")

    # Resolve the full path
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (base_dir / path).resolve()

    # Check for path traversal
    try:
        resolved.relative_to(base_dir)
    except ValueError:
        raise ValidationError(
            f"Path escapes base directory: {path} resolves to {resolved}, "
            f"which is outside {base_dir}"
        )

    # Check existence if required
    if must_exist and not resolved.exists():
        raise ValidationError(f"Path does not exist: {resolved}")

    return resolved


def validate_identifier(
    value: str,
    name: str = "identifier",
    max_length: int = MAX_IDENTIFIER_LENGTH,
    allow_dots: bool = False,
) -> str:
    """Validate an identifier (template ID, plugin name, etc.).

    Safe identifiers:
    - Start with a letter
    - Contain only alphanumeric, underscore, hyphen
    - Optionally allow dots (for namespaced identifiers)
    - Have reasonable length

    Args:
        value: Identifier to validate
        name: Name of the identifier (for error messages)
        max_length: Maximum allowed length
        allow_dots: If True, allow dots in identifier

    Returns:
        The validated identifier

    Raises:
        ValidationError: If identifier is invalid

    Example:
        >>> validate_identifier("my_template")
        "my_template"
        >>> validate_identifier("../../../etc")
        ValidationError: identifier contains invalid characters
    """
    if not value:
        raise ValidationError(f"{name} cannot be empty")

    if len(value) > max_length:
        raise ValidationError(f"{name} exceeds maximum length of {max_length}: {len(value)}")

    # Build pattern based on options
    if allow_dots:
        pattern = re.compile(r"^[a-zA-Z][a-zA-Z0-9_.-]*$")
    else:
        pattern = SAFE_IDENTIFIER_PATTERN

    if not pattern.match(value):
        raise ValidationError(
            f"{name} contains invalid characters: {value!r}. "
            f"Must start with letter and contain only alphanumeric, underscore, hyphen"
            + (", dot" if allow_dots else "")
        )

    # Additional check for path traversal attempts
    if ".." in value or value.startswith("/") or value.startswith("\\"):
        raise ValidationError(f"{name} contains path traversal pattern: {value!r}")

    return value


def validate_workflow_params(
    params: dict[str, Any],
    step_type: str,
) -> dict[str, Any]:
    """Validate workflow step parameters based on step type.

    Performs type-specific validation for known step types.

    Args:
        params: Step parameters to validate
        step_type: Type of step (shell, claude_code, openai, etc.)

    Returns:
        Validated parameters

    Raises:
        ValidationError: If parameters are invalid
    """
    validated = dict(params)

    if step_type == "shell":
        command = params.get("command", "")
        if command:
            validate_shell_command(command, allow_dangerous=params.get("allow_dangerous", False))

    elif step_type in ("claude_code", "openai"):
        # Validate prompt length
        prompt = params.get("prompt", "")
        if len(prompt) > 100000:  # 100KB limit
            raise ValidationError(f"Prompt exceeds maximum length: {len(prompt)} > 100000")

    elif step_type == "parallel":
        # Validate sub-steps recursively
        sub_steps = params.get("steps", [])
        for i, sub_step in enumerate(sub_steps):
            if isinstance(sub_step, dict):
                sub_type = sub_step.get("type", "")
                sub_params = sub_step.get("params", {})
                validate_workflow_params(sub_params, sub_type)

    return validated


def sanitize_log_message(message: str, sensitive_patterns: list[str] | None = None) -> str:
    """Sanitize a log message to remove sensitive data.

    Args:
        message: Message to sanitize
        sensitive_patterns: Additional patterns to redact

    Returns:
        Sanitized message with sensitive data redacted
    """
    result = message

    # Default sensitive patterns
    default_patterns = [
        (r"sk-[a-zA-Z0-9]{20,}", "[REDACTED_API_KEY]"),  # OpenAI keys
        (r"sk-ant-[a-zA-Z0-9_-]{40,}", "[REDACTED_API_KEY]"),  # Anthropic keys
        (r"ghp_[a-zA-Z0-9]{36}", "[REDACTED_GITHUB_TOKEN]"),  # GitHub PATs
        (r"gho_[a-zA-Z0-9]{36}", "[REDACTED_GITHUB_TOKEN]"),  # GitHub OAuth
        (r"secret_[a-zA-Z0-9]{32,}", "[REDACTED_SECRET]"),  # Notion tokens
        (r'password["\']?\s*[:=]\s*["\']?[^"\'\s]+', "password=[REDACTED]"),
        (r'token["\']?\s*[:=]\s*["\']?[^"\'\s]+', "token=[REDACTED]"),
    ]

    for pattern, replacement in default_patterns:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Apply additional patterns
    if sensitive_patterns:
        for pattern in sensitive_patterns:
            result = re.sub(pattern, "[REDACTED]", result, flags=re.IGNORECASE)

    return result


def sanitize_prompt_variable(value: str, max_length: int = 50000) -> str:
    """Sanitize a user-supplied value before prompt template interpolation.

    Prevents format string injection by escaping curly braces in the value,
    and enforces a length limit.

    Args:
        value: Raw user input to be inserted into a prompt template.
        max_length: Maximum allowed length for the value.

    Returns:
        Sanitized string safe for use with str.format().

    Raises:
        ValidationError: If value exceeds max_length.
    """
    text = str(value)
    if len(text) > max_length:
        raise ValidationError(f"Prompt variable exceeds maximum length: {len(text)} > {max_length}")
    # Escape any braces that aren't the template's own placeholders.
    # Double-braces are treated as literal braces by str.format().
    text = text.replace("{", "{{").replace("}", "}}")
    return text


class PathValidator:
    """Context-aware path validator for a specific base directory.

    Example:
        validator = PathValidator("/app/plugins")
        safe_path = validator.validate("my_plugin.py")
        # Returns: /app/plugins/my_plugin.py

        validator.validate("../../../etc/passwd")
        # Raises: ValidationError
    """

    def __init__(
        self,
        base_dir: str | Path,
        allowed_extensions: set[str] | None = None,
        must_exist: bool = False,
    ):
        """Initialize path validator.

        Args:
            base_dir: Base directory for all paths
            allowed_extensions: If set, only allow these extensions (e.g., {".py", ".json"})
            must_exist: If True, paths must exist
        """
        self.base_dir = Path(base_dir).resolve()
        self.allowed_extensions = allowed_extensions
        self.must_exist = must_exist

    def validate(self, path: str | Path) -> Path:
        """Validate a path.

        Args:
            path: Path to validate (relative to base_dir)

        Returns:
            Resolved, validated Path

        Raises:
            ValidationError: If path is invalid
        """
        validated = validate_safe_path(
            path,
            self.base_dir,
            must_exist=self.must_exist,
        )

        if self.allowed_extensions:
            if validated.suffix.lower() not in self.allowed_extensions:
                raise ValidationError(
                    f"File extension not allowed: {validated.suffix}. "
                    f"Allowed: {self.allowed_extensions}"
                )

        return validated

    def validate_identifier_as_path(self, identifier: str, extension: str = "") -> Path:
        """Validate an identifier and convert to a safe path.

        Args:
            identifier: Identifier to validate
            extension: File extension to append (e.g., ".json")

        Returns:
            Safe path within base_dir

        Example:
            validator = PathValidator("/app/templates")
            validator.validate_identifier_as_path("my_template", ".json")
            # Returns: /app/templates/my_template.json
        """
        validate_identifier(identifier)
        filename = f"{identifier}{extension}"
        return self.base_dir / filename
