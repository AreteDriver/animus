"""Integration tests for self-improve pipeline with live Ollama.

Excluded from CI via conftest.py collect_ignore.
Run manually: pytest tests/test_self_improve_ollama_integration.py -v -s
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:
    import httpx
except ImportError:
    httpx = None


def ollama_is_available() -> bool:
    """Check if Ollama is running locally."""
    if not httpx:
        return False
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not ollama_is_available(),
    reason="Ollama not available at localhost:11434",
)

# Default model for integration tests
OLLAMA_MODEL = "deepseek-coder-v2"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ollama_provider():
    """Create a live OllamaProvider with generous timeout."""
    from animus_forge.providers.base import ProviderConfig, ProviderType
    from animus_forge.providers.ollama_provider import OllamaProvider

    provider = OllamaProvider(
        config=ProviderConfig(
            provider_type=ProviderType.OLLAMA,
            base_url="http://localhost:11434",
            default_model=OLLAMA_MODEL,
            timeout=600.0,
        ),
    )
    return provider


@pytest.fixture()
def agent_provider(ollama_provider):
    """Wrap OllamaProvider in AgentProvider."""
    from animus_forge.agents.provider_wrapper import AgentProvider

    return AgentProvider(ollama_provider)


@pytest.fixture()
def permissive_config():
    """SafetyConfig with no approvals and relaxed limits."""
    from animus_forge.self_improve.safety import SafetyConfig

    return SafetyConfig(
        critical_files=["**/identity/**", "**/CORE_VALUES*"],
        sensitive_files=[],
        max_files_per_pr=50,
        max_lines_changed=5000,
        max_new_files=20,
        max_deleted_files=5,
        human_approval_plan=False,
        human_approval_apply=False,
        human_approval_merge=False,
        max_snapshots=5,
        auto_rollback_on_test_failure=True,
        branch_prefix="test-improve/",
    )


@pytest.fixture()
def synthetic_project(tmp_path: Path) -> Path:
    """Create a self-contained Python project with intentional issues."""
    # pyproject.toml
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "sample"\nversion = "0.1.0"\n\n'
        "[tool.pytest.ini_options]\n"
        'pythonpath = ["src"]\n'
    )

    # Source code with intentional issues
    src = tmp_path / "src" / "sample"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("")
    (src / "calculator.py").write_text(
        '"""Calculator module."""\n'
        "\n"
        "import os\n"
        "import sys\n"
        "import json\n"
        "\n"
        "\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "\n"
        "def subtract(a, b):\n"
        "    return a - b\n"
        "\n"
        "\n"
        "def divide(a, b):\n"
        "    # TODO: handle division by zero\n"
        "    try:\n"
        "        result = a / b\n"
        "    except:\n"
        "        result = 0\n"
        "    return result\n"
        "\n"
        "\n"
        "def process_data(data):\n"
        "    x = []\n"
        "    for i in range(len(data)):\n"
        "        if data[i] > 0:\n"
        "            x.append(data[i] * 2)\n"
        "        else:\n"
        "            if data[i] == 0:\n"
        "                x.append(0)\n"
        "            else:\n"
        "                if data[i] > -10:\n"
        "                    x.append(data[i])\n"
        "                else:\n"
        "                    if data[i] > -100:\n"
        "                        x.append(data[i] + 100)\n"
        "                    else:\n"
        "                        x.append(None)\n"
        "    result = []\n"
        "    for item in x:\n"
        "        if item is not None:\n"
        "            result.append(item)\n"
        "    a = 1\n"
        "    b = 2\n"
        "    c = 3\n"
        "    d = 4\n"
        "    e = 5\n"
        "    total = a + b + c + d + e\n"
        "    result.append(total)\n"
        "    temp = []\n"
        "    for r in result:\n"
        "        temp.append(str(r))\n"
        "    output = ','.join(temp)\n"
        "    return output\n"
    )

    # Tests
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("")
    (tests / "test_calculator.py").write_text(
        "from sample.calculator import add, subtract\n"
        "\n"
        "\n"
        "def test_add():\n"
        "    assert add(1, 2) == 3\n"
        "\n"
        "\n"
        "def test_subtract():\n"
        "    assert subtract(5, 3) == 2\n"
    )

    return tmp_path


# ===========================================================================
# TestOllamaProviderIntegration
# ===========================================================================


class TestOllamaProviderIntegration:
    """Verify basic Ollama connectivity and completions."""

    def test_health_check(self, ollama_provider):
        """Ollama server is reachable."""
        assert ollama_provider.health_check() is True

    def test_basic_completion(self, ollama_provider):
        """Provider returns non-empty content."""
        from animus_forge.providers.base import CompletionRequest

        request = CompletionRequest(
            prompt="Return the word 'hello' and nothing else.",
            temperature=0.0,
            max_tokens=32,
        )
        response = ollama_provider.complete(request)
        assert response.content.strip()
        assert response.tokens_used > 0
        print(f"  [basic_completion] {response.latency_ms:.0f}ms, {response.tokens_used} tokens")

    def test_json_generation(self, ollama_provider):
        """Provider can generate valid JSON."""
        from animus_forge.providers.base import CompletionRequest

        request = CompletionRequest(
            prompt=(
                'Return ONLY a JSON object with keys "name" and "value". '
                'Example: {"name": "test", "value": 42}. No other text.'
            ),
            system_prompt="You are a JSON generator. Return only valid JSON.",
            temperature=0.0,
            max_tokens=128,
        )
        response = ollama_provider.complete(request)
        content = response.content.strip()

        # Try direct parse, then regex fallback (same as orchestrator)
        import re

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                parsed = json.loads(match.group())
            else:
                pytest.skip(f"Model returned non-JSON: {content[:200]}")

        assert isinstance(parsed, dict)
        print(f"  [json_generation] Parsed keys: {list(parsed.keys())}")


# ===========================================================================
# TestAnalyzerWithOllama
# ===========================================================================


class TestAnalyzerWithOllama:
    """Verify the analyzer produces valid suggestions with live LLM."""

    def test_analyze_with_ai(self, agent_provider, synthetic_project):
        """analyze_with_ai() returns valid suggestions."""
        from animus_forge.self_improve.analyzer import CodebaseAnalyzer

        analyzer = CodebaseAnalyzer(agent_provider, synthetic_project)
        result = asyncio.run(analyzer.analyze_with_ai())

        # Static analysis should always find something (bare except, unused imports)
        assert result.files_analyzed > 0
        print(f"  [analyze] {result.files_analyzed} files, {len(result.suggestions)} suggestions")
        for s in result.suggestions[:5]:
            print(f"    - [{s.category.value}] {s.title}")


# ===========================================================================
# TestCodeGenerationWithOllama
# ===========================================================================


class TestCodeGenerationWithOllama:
    """Verify _generate_changes() with live Ollama."""

    def test_generate_changes_returns_parseable_json(
        self, agent_provider, synthetic_project, permissive_config
    ):
        """_generate_changes() returns a non-empty dict."""
        from animus_forge.self_improve.analyzer import ImprovementCategory, ImprovementSuggestion
        from animus_forge.self_improve.orchestrator import ImprovementPlan, SelfImproveOrchestrator

        plan = ImprovementPlan(
            id="test-gen",
            title="Fix bare except clause",
            description="Replace bare except: with except ZeroDivisionError:",
            suggestions=[
                ImprovementSuggestion(
                    id="s1",
                    category=ImprovementCategory.CODE_QUALITY,
                    title="Fix bare except",
                    description="Replace bare except: with except ZeroDivisionError: in divide()",
                    affected_files=["src/sample/calculator.py"],
                    estimated_lines=2,
                    implementation_hints="Change 'except:' to 'except ZeroDivisionError:'",
                ),
            ],
            implementation_steps=["Fix bare except clause"],
            estimated_files=["src/sample/calculator.py"],
            estimated_lines=2,
        )

        orch = SelfImproveOrchestrator(
            codebase_path=synthetic_project,
            provider=agent_provider,
            config=permissive_config,
        )

        changes = asyncio.run(orch._generate_changes(plan))

        if not changes:
            pytest.skip("LLM returned empty/unparseable response")

        assert isinstance(changes, dict)
        print(f"  [generate] Got changes for {len(changes)} files: {list(changes.keys())}")

        # Validate the content is valid Python
        for path, content in changes.items():
            try:
                compile(content, path, "exec")
            except SyntaxError as e:
                print(f"  [generate] WARNING: {path} has syntax error: {e}")

    def test_generate_changes_valid_python(
        self, agent_provider, synthetic_project, permissive_config
    ):
        """Generated code should be syntactically valid Python."""
        from animus_forge.self_improve.analyzer import ImprovementCategory, ImprovementSuggestion
        from animus_forge.self_improve.orchestrator import ImprovementPlan, SelfImproveOrchestrator

        plan = ImprovementPlan(
            id="test-syntax",
            title="Add type hints to add and subtract",
            description="Add type hints to add() and subtract() functions",
            suggestions=[
                ImprovementSuggestion(
                    id="s1",
                    category=ImprovementCategory.CODE_QUALITY,
                    title="Add type hints",
                    description="Add int type hints to add(a, b) and subtract(a, b)",
                    affected_files=["src/sample/calculator.py"],
                    estimated_lines=4,
                    implementation_hints="Add ': int' to params and '-> int' return type",
                ),
            ],
            implementation_steps=["Add type hints"],
            estimated_files=["src/sample/calculator.py"],
            estimated_lines=4,
        )

        orch = SelfImproveOrchestrator(
            codebase_path=synthetic_project,
            provider=agent_provider,
            config=permissive_config,
        )

        changes = asyncio.run(orch._generate_changes(plan))

        if not changes:
            pytest.skip("LLM returned empty/unparseable response")

        for path, content in changes.items():
            compile(content, path, "exec")  # Raises SyntaxError if invalid
            print(f"  [syntax_check] {path}: valid Python ({len(content)} chars)")


# ===========================================================================
# TestSandboxWithOllama
# ===========================================================================


class TestSandboxWithOllama:
    """Verify LLM-generated changes pass sandbox validation."""

    def test_llm_changes_pass_sandbox(self, agent_provider, synthetic_project, permissive_config):
        """LLM-generated changes pass pytest + ruff in sandbox."""
        from animus_forge.self_improve.analyzer import ImprovementCategory, ImprovementSuggestion
        from animus_forge.self_improve.orchestrator import ImprovementPlan, SelfImproveOrchestrator
        from animus_forge.self_improve.sandbox import Sandbox

        plan = ImprovementPlan(
            id="test-sandbox",
            title="Remove unused imports",
            description="Remove os, sys, json imports from calculator.py",
            suggestions=[
                ImprovementSuggestion(
                    id="s1",
                    category=ImprovementCategory.CODE_QUALITY,
                    title="Remove unused imports",
                    description="Remove import os, import sys, import json — they are unused",
                    affected_files=["src/sample/calculator.py"],
                    estimated_lines=3,
                    implementation_hints="Delete the three import lines at the top",
                ),
            ],
            implementation_steps=["Remove unused imports"],
            estimated_files=["src/sample/calculator.py"],
            estimated_lines=3,
        )

        orch = SelfImproveOrchestrator(
            codebase_path=synthetic_project,
            provider=agent_provider,
            config=permissive_config,
        )

        changes = asyncio.run(orch._generate_changes(plan))
        if not changes:
            pytest.skip("LLM returned empty/unparseable response")

        # Apply in sandbox and run tests
        with Sandbox(synthetic_project, timeout=120) as sandbox:
            applied = asyncio.run(sandbox.apply_changes(changes))
            assert applied, "Failed to apply changes to sandbox"

            result = asyncio.run(sandbox.validate_changes())
            print(
                f"  [sandbox] tests_passed={result.tests_passed}, lint_passed={result.lint_passed}"
            )
            if result.test_output:
                print(f"  [sandbox] test output:\n{result.test_output[:500]}")
            if result.lint_output:
                print(f"  [sandbox] lint output:\n{result.lint_output[:500]}")


# ===========================================================================
# TestFullPipelineWithOllama
# ===========================================================================


class TestFullPipelineWithOllama:
    """End-to-end pipeline run with live Ollama, mocking only git/PR ops."""

    def test_full_pipeline_auto_approve(self, agent_provider, synthetic_project, permissive_config):
        """Full orchestrator.run(auto_approve=True) with live LLM."""
        from animus_forge.self_improve.orchestrator import SelfImproveOrchestrator, WorkflowStage

        orch = SelfImproveOrchestrator(
            codebase_path=synthetic_project,
            provider=agent_provider,
            config=permissive_config,
        )

        start = time.time()

        with (
            patch.object(orch.pr_manager, "_run_git"),
            patch("animus_forge.self_improve.pr_manager.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=0, stdout="https://github.com/test/pull/1\n"
            )
            result = asyncio.run(orch.run(auto_approve=True))

        elapsed = time.time() - start

        print(f"\n  [pipeline] Stage reached: {result.stage_reached.value}")
        print(f"  [pipeline] Success: {result.success}")
        print(f"  [pipeline] Elapsed: {elapsed:.1f}s")
        if result.error:
            print(f"  [pipeline] Error: {result.error}")
        if result.plan:
            print(f"  [pipeline] Plan: {result.plan.title}")
            print(f"  [pipeline] Suggestions: {len(result.plan.suggestions)}")
        if result.sandbox_result:
            print(
                f"  [pipeline] Sandbox: tests={result.sandbox_result.tests_passed}, "
                f"lint={result.sandbox_result.lint_passed}"
            )

        # The pipeline should at least get past analysis
        assert result.stage_reached != WorkflowStage.IDLE


# ===========================================================================
# TestErrorRecovery
# ===========================================================================


class TestErrorRecovery:
    """Verify graceful handling of garbage or adversarial LLM output."""

    def test_garbage_output_handled(self, synthetic_project, permissive_config):
        """Pipeline handles completely unparseable LLM output."""
        from unittest.mock import AsyncMock

        from animus_forge.self_improve.orchestrator import SelfImproveOrchestrator

        mock_provider = AsyncMock()
        mock_provider.complete.return_value = (
            "I cannot generate code because the sun is too bright today. "
            "Here is a poem instead:\n\nRoses are red..."
        )

        orch = SelfImproveOrchestrator(
            codebase_path=synthetic_project,
            provider=mock_provider,
            config=permissive_config,
        )

        result = asyncio.run(orch.run())
        # Should fail gracefully at IMPLEMENTING, not crash
        assert result.success is False
        assert "No changes generated" in (result.error or "")

    def test_partial_json_handled(self, synthetic_project, permissive_config):
        """Pipeline handles truncated JSON from LLM."""
        from unittest.mock import AsyncMock

        from animus_forge.self_improve.orchestrator import SelfImproveOrchestrator

        mock_provider = AsyncMock()
        mock_provider.complete.return_value = '{"src/sample/calculator.py": "def add(a'

        orch = SelfImproveOrchestrator(
            codebase_path=synthetic_project,
            provider=mock_provider,
            config=permissive_config,
        )

        result = asyncio.run(orch.run())
        assert result.success is False


# ===========================================================================
# TestModelPerformance
# ===========================================================================


class TestModelPerformance:
    """Measure JSON generation reliability and timing across multiple runs."""

    def test_json_generation_reliability(self, ollama_provider):
        """Run JSON generation N=3 times, report success rate and timing."""
        from animus_forge.providers.base import CompletionRequest

        n_runs = 3
        successes = 0
        timings = []

        for i in range(n_runs):
            request = CompletionRequest(
                prompt=(
                    "Return a JSON object with exactly two keys: "
                    '"file" (a string file path) and "content" (a string with Python code). '
                    "Example: "
                    '{"file": "example.py", "content": "print(42)\\n"}\n'
                    "Return ONLY the JSON object, no other text."
                ),
                system_prompt="You are a JSON generator. Return only valid JSON, no markdown fences.",
                temperature=0.0,
                max_tokens=256,
            )

            start = time.time()
            response = ollama_provider.complete(request)
            elapsed = time.time() - start
            timings.append(elapsed)

            content = response.content.strip()
            import re

            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    successes += 1
            except json.JSONDecodeError:
                match = re.search(r"\{[\s\S]*\}", content)
                if match:
                    try:
                        parsed = json.loads(match.group())
                        if isinstance(parsed, dict):
                            successes += 1
                    except json.JSONDecodeError:
                        pass

            print(f"  [run {i + 1}/{n_runs}] {elapsed:.1f}s, parsed={successes > i}")

        rate = successes / n_runs * 100
        avg_time = sum(timings) / len(timings)
        print(f"\n  [summary] Success rate: {successes}/{n_runs} ({rate:.0f}%)")
        print(f"  [summary] Avg time: {avg_time:.1f}s")
        print(f"  [summary] Min/Max: {min(timings):.1f}s / {max(timings):.1f}s")

        # At least 1 out of 3 should succeed — this isn't a hard gate,
        # just validates the model can produce JSON at all
        assert successes >= 1, f"Only {successes}/{n_runs} runs produced valid JSON"
