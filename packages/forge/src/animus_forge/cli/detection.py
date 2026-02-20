"""Codebase detection utilities for CLI context-aware prompts."""

from __future__ import annotations

import json
from pathlib import Path


def _detect_python_framework(path: Path) -> str | None:
    """Detect Python framework from pyproject.toml."""
    pyproject = path / "pyproject.toml"
    if not pyproject.exists():
        return None
    try:
        content = pyproject.read_text().lower()
        frameworks = [
            ("fastapi", "fastapi"),
            ("django", "django"),
            ("flask", "flask"),
            ("streamlit", "streamlit"),
        ]
        for keyword, framework in frameworks:
            if keyword in content:
                return framework
    except Exception:
        pass  # Non-critical fallback: pyproject.toml unreadable, framework detection skipped
    return None


def _detect_js_framework(path: Path) -> str | None:
    """Detect JavaScript/TypeScript framework from package.json."""
    try:
        pkg = json.loads((path / "package.json").read_text())
        deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
        frameworks = [("react", "react"), ("vue", "vue"), ("next", "nextjs")]
        for keyword, framework in frameworks:
            if keyword in deps:
                return framework
    except Exception:
        pass  # Non-critical fallback: package.json unreadable, framework detection skipped
    return None


def _detect_language_and_framework(path: Path) -> tuple[str, str | None]:
    """Detect primary language and framework.

    Returns:
        Tuple of (language, framework)
    """
    if (path / "pyproject.toml").exists() or (path / "setup.py").exists():
        return "python", _detect_python_framework(path)
    if (path / "Cargo.toml").exists():
        return "rust", None
    if (path / "package.json").exists():
        return "typescript", _detect_js_framework(path)
    if (path / "go.mod").exists():
        return "go", None
    return "unknown", None


def _get_key_structure(path: Path, limit: int = 20) -> list[str]:
    """Get key directories and files in the codebase."""
    structure = []
    key_dirs = {"src", "lib", "app", "tests", "docs"}
    code_exts = {".py", ".rs", ".ts", ".js", ".go"}

    for item in path.iterdir():
        if item.name.startswith("."):
            continue
        if item.is_dir() and item.name in key_dirs:
            structure.append(f"{item.name}/")
        elif item.is_file() and item.suffix in code_exts:
            structure.append(item.name)
    return structure[:limit]


def _get_readme_content(path: Path, max_chars: int = 500) -> str | None:
    """Get README content if present."""
    readme_names = ("README.md", "README.rst", "README.txt", "README")
    for name in readme_names:
        readme_path = path / name
        if readme_path.exists():
            try:
                return readme_path.read_text()[:max_chars]
            except Exception:
                pass  # Non-critical fallback: README unreadable, try next filename variant
    return None


def detect_codebase_context(path: Path = None) -> dict:
    """Auto-detect codebase context for better agent prompts.

    Returns context dict with:
    - language: Primary language (python, rust, typescript, etc.)
    - framework: Detected framework (fastapi, react, etc.)
    - structure: Key directories and files
    - readme: First 500 chars of README if present
    """
    path = path or Path.cwd()
    language, framework = _detect_language_and_framework(path)

    return {
        "path": str(path),
        "language": language,
        "framework": framework,
        "structure": _get_key_structure(path),
        "readme": _get_readme_content(path),
    }


def format_context_for_prompt(context: dict) -> str:
    """Format codebase context for agent prompts."""
    lines = [f"Codebase: {context['path']}"]
    lines.append(f"Language: {context['language']}")
    if context["framework"]:
        lines.append(f"Framework: {context['framework']}")
    if context["structure"]:
        lines.append(f"Structure: {', '.join(context['structure'][:10])}")
    return "\n".join(lines)
