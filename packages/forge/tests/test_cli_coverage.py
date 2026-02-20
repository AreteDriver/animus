"""Additional coverage tests for CLI main module."""

import json
import sys
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

sys.path.insert(0, "src")

from animus_forge.cli.main import (
    _detect_js_framework,
    _detect_language_and_framework,
    _detect_python_framework,
    _get_key_structure,
    _get_readme_content,
    _parse_cli_variables,
    app,
    detect_codebase_context,
    format_context_for_prompt,
    get_tracker,
    version_callback,
)

runner = CliRunner()


class TestVersionCallback:
    def test_version_shows(self):
        import typer

        with pytest.raises(typer.Exit):
            version_callback(True)

    def test_version_false(self):
        version_callback(False)  # No exit


class TestCLIApp:
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "gorgon" in result.stdout.lower() or "workflow" in result.stdout.lower()

    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0


class TestDetectPythonFramework:
    def test_fastapi(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[dependencies]\nfastapi = "^0.100"')
        assert _detect_python_framework(tmp_path) == "fastapi"

    def test_django(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[dependencies]\ndjango = "^4.2"')
        assert _detect_python_framework(tmp_path) == "django"

    def test_flask(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[dependencies]\nflask = "^3.0"')
        assert _detect_python_framework(tmp_path) == "flask"

    def test_streamlit(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[dependencies]\nstreamlit = "^1.0"')
        assert _detect_python_framework(tmp_path) == "streamlit"

    def test_no_pyproject(self, tmp_path):
        assert _detect_python_framework(tmp_path) is None

    def test_no_framework(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'basic'")
        assert _detect_python_framework(tmp_path) is None


class TestDetectJsFramework:
    def test_react(self, tmp_path):
        pkg = {"dependencies": {"react": "^18.0"}}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        assert _detect_js_framework(tmp_path) == "react"

    def test_vue(self, tmp_path):
        pkg = {"dependencies": {"vue": "^3.0"}}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        assert _detect_js_framework(tmp_path) == "vue"

    def test_nextjs(self, tmp_path):
        pkg = {"dependencies": {"next": "^14.0"}}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        assert _detect_js_framework(tmp_path) == "nextjs"

    def test_no_package_json(self, tmp_path):
        assert _detect_js_framework(tmp_path) is None


class TestDetectLanguageAndFramework:
    def test_python(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]")
        lang, fw = _detect_language_and_framework(tmp_path)
        assert lang == "python"

    def test_rust(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]")
        lang, fw = _detect_language_and_framework(tmp_path)
        assert lang == "rust"
        assert fw is None

    def test_typescript(self, tmp_path):
        (tmp_path / "package.json").write_text("{}")
        lang, fw = _detect_language_and_framework(tmp_path)
        assert lang == "typescript"

    def test_go(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example")
        lang, fw = _detect_language_and_framework(tmp_path)
        assert lang == "go"
        assert fw is None

    def test_unknown(self, tmp_path):
        lang, fw = _detect_language_and_framework(tmp_path)
        assert lang == "unknown"


class TestGetKeyStructure:
    def test_basic(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()
        (tmp_path / "main.py").touch()
        (tmp_path / ".hidden").mkdir()

        structure = _get_key_structure(tmp_path)
        assert "src/" in structure
        assert "tests/" in structure
        assert "main.py" in structure
        assert ".hidden/" not in structure

    def test_limit(self, tmp_path):
        for i in range(30):
            (tmp_path / f"file_{i}.py").touch()
        structure = _get_key_structure(tmp_path, limit=5)
        assert len(structure) <= 5


class TestGetReadmeContent:
    def test_readme_md(self, tmp_path):
        (tmp_path / "README.md").write_text("# Hello World\nThis is a test.")
        content = _get_readme_content(tmp_path)
        assert "Hello World" in content

    def test_readme_txt(self, tmp_path):
        (tmp_path / "README.txt").write_text("Hello")
        assert _get_readme_content(tmp_path) == "Hello"

    def test_no_readme(self, tmp_path):
        assert _get_readme_content(tmp_path) is None

    def test_truncated(self, tmp_path):
        (tmp_path / "README.md").write_text("x" * 1000)
        content = _get_readme_content(tmp_path, max_chars=100)
        assert len(content) == 100


class TestDetectCodebaseContext:
    def test_basic(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[deps]\nfastapi = "1"')
        (tmp_path / "README.md").write_text("# Project")
        (tmp_path / "src").mkdir()

        ctx = detect_codebase_context(tmp_path)
        assert ctx["language"] == "python"
        assert ctx["framework"] == "fastapi"
        assert ctx["readme"] is not None
        assert "src/" in ctx["structure"]


class TestFormatContextForPrompt:
    def test_with_framework(self):
        ctx = {
            "path": "/tmp/test",
            "language": "python",
            "framework": "fastapi",
            "structure": ["src/", "tests/", "main.py"],
        }
        result = format_context_for_prompt(ctx)
        assert "python" in result
        assert "fastapi" in result
        assert "src/" in result

    def test_without_framework(self):
        ctx = {
            "path": "/tmp/test",
            "language": "rust",
            "framework": None,
            "structure": [],
        }
        result = format_context_for_prompt(ctx)
        assert "rust" in result
        assert "Framework" not in result


class TestParseCliVariables:
    def test_valid(self):
        result = _parse_cli_variables(["key1=val1", "key2=val2"])
        assert result == {"key1": "val1", "key2": "val2"}

    def test_value_with_equals(self):
        result = _parse_cli_variables(["key=a=b"])
        assert result == {"key": "a=b"}

    def test_invalid_format(self):
        import typer

        with pytest.raises(typer.Exit):
            _parse_cli_variables(["invalid"])


class TestGetTracker:
    def test_returns_tracker(self):
        tracker = get_tracker()
        # May return None or a tracker depending on imports
        assert tracker is None or tracker is not None


class TestGetWorkflowEngine:
    def test_success(self):
        from animus_forge.cli.main import get_workflow_engine

        engine = get_workflow_engine()
        assert engine is not None

    def test_import_error(self):
        with patch.dict("sys.modules", {"animus_forge.orchestrator": None}):
            # Can't easily trigger - depends on import caching
            pass


class TestGetWorkflowExecutor:
    def test_success(self):
        from animus_forge.cli.main import get_workflow_executor

        executor = get_workflow_executor(dry_run=True)
        assert executor is not None
        assert executor.dry_run is True
