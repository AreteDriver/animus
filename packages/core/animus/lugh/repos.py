"""
Animus Harvest Tool

Scans external repos with anchormd and extracts learnable patterns,
architectures, and tools. Stores findings in animus memory.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from animus.tools import Tool, ToolResult

logger = logging.getLogger(__name__)

# Pattern categories we extract
PATTERN_CATEGORIES = [
    "architecture",
    "dependencies",
    "testing",
    "ci_cd",
    "code_conventions",
    "tooling",
]


@dataclass
class HarvestResult:
    """Structured output from a repo harvest."""

    repo: str
    score: int = 0
    architecture: str = ""
    notable_patterns: list[str] = field(default_factory=list)
    tools_worth_adopting: list[str] = field(default_factory=list)
    testing_approach: str = ""
    comparison: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "score": self.score,
            "architecture": self.architecture,
            "notable_patterns": self.notable_patterns,
            "tools_worth_adopting": self.tools_worth_adopting,
            "testing_approach": self.testing_approach,
            "comparison": self.comparison,
        }


def _normalize_target(target: str) -> str:
    """Normalize a GitHub target to a clone URL.

    Accepts:
      - Full URL: https://github.com/user/repo
      - Short form: user/repo

    Returns:
      Clone URL as https://github.com/user/repo.git
    """
    target = target.strip().rstrip("/")

    # Already a full URL
    if target.startswith("https://") or target.startswith("git@"):
        if not target.endswith(".git"):
            target += ".git"
        return target

    # Short form: user/repo
    if "/" in target and not target.startswith("/"):
        parts = target.split("/")
        if len(parts) == 2:
            return f"https://github.com/{parts[0]}/{parts[1]}.git"

    raise ValueError(f"Invalid target: '{target}'. Use 'user/repo' or full GitHub URL.")


def _extract_repo_name(target: str) -> str:
    """Extract 'user/repo' from a target string."""
    target = target.strip().rstrip("/")

    # Remove .git suffix
    if target.endswith(".git"):
        target = target[:-4]

    # Full URL
    if "github.com" in target:
        match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", target)
        if match:
            return match.group(1)

    # Already user/repo
    if "/" in target and not target.startswith("/"):
        parts = target.split("/")
        if len(parts) == 2:
            return target

    return target


def _clone_repo(clone_url: str, dest: Path, depth: int = 1) -> None:
    """Shallow clone a repo into dest directory."""
    cmd = ["git", "clone", "--depth", str(depth), clone_url, str(dest)]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")


def _scan_with_anchormd(repo_path: Path) -> dict[str, Any]:
    """Run anchormd scanner and analyzers on a repo path.

    Returns dict with structure and analysis data.
    Falls back to basic scanning if anchormd is not installed.
    """
    try:
        from anchormd.analyzers import run_all
        from anchormd.models import ForgeConfig
        from anchormd.scanner import CodebaseScanner

        config = ForgeConfig(
            root_path=repo_path,
            output_path=repo_path / "CLAUDE.md",
        )
        scanner = CodebaseScanner(config)
        structure = scanner.scan()
        analyses = run_all(structure, config)

        return {
            "total_files": structure.total_files,
            "total_lines": structure.total_lines,
            "languages": structure.languages,
            "primary_language": structure.primary_language,
            "version": structure.version,
            "description": structure.description,
            "dependencies": structure.declared_dependencies,
            "analyses": [
                {
                    "category": a.category,
                    "findings": a.findings,
                    "confidence": a.confidence,
                    "section_content": a.section_content,
                }
                for a in analyses
            ],
        }
    except ImportError:
        logger.info("anchormd not installed, using basic scanning")
        return _basic_scan(repo_path)


def _basic_scan(repo_path: Path) -> dict[str, Any]:
    """Basic scanning fallback when anchormd is not available."""
    languages: dict[str, int] = {}
    total_files = 0
    total_lines = 0

    ext_to_lang = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".rs": "Rust",
        ".go": "Go",
        ".java": "Java",
        ".rb": "Ruby",
        ".cpp": "C++",
        ".c": "C",
        ".cs": "C#",
        ".swift": "Swift",
        ".kt": "Kotlin",
    }

    skip_dirs = {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        "dist",
        "build",
        ".next",
        "target",
    }

    for path in repo_path.rglob("*"):
        if any(part in skip_dirs for part in path.parts):
            continue
        if not path.is_file():
            continue
        total_files += 1
        ext = path.suffix.lower()
        lang = ext_to_lang.get(ext)
        if lang:
            languages[lang] = languages.get(lang, 0) + 1
        try:
            line_count = len(path.read_bytes().split(b"\n"))
            total_lines += line_count
        except (OSError, UnicodeDecodeError):
            pass

    primary = max(languages, key=languages.get) if languages else None

    # Detect dependencies
    dependencies = _detect_dependencies(repo_path)

    # Detect patterns
    analyses = _detect_patterns(repo_path)

    return {
        "total_files": total_files,
        "total_lines": total_lines,
        "languages": languages,
        "primary_language": primary,
        "version": None,
        "description": None,
        "dependencies": dependencies,
        "analyses": analyses,
    }


def _detect_dependencies(repo_path: Path) -> dict[str, list[str]]:
    """Detect declared dependencies from manifest files."""
    deps: dict[str, list[str]] = {}

    # Python: pyproject.toml / requirements.txt
    pyproject = repo_path / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomli

            with open(pyproject, "rb") as f:
                data = tomli.load(f)
            project_deps = data.get("project", {}).get("dependencies", [])
            if project_deps:
                deps["python"] = [re.split(r"[<>=!;\[]", d)[0].strip() for d in project_deps]
        except (ImportError, Exception):
            pass

    req_txt = repo_path / "requirements.txt"
    if req_txt.exists() and "python" not in deps:
        try:
            lines = req_txt.read_text().splitlines()
            deps["python"] = [
                re.split(r"[<>=!;\[]", line)[0].strip()
                for line in lines
                if line.strip() and not line.startswith("#")
            ]
        except OSError:
            pass

    # Node: package.json
    pkg_json = repo_path / "package.json"
    if pkg_json.exists():
        try:
            data = json.loads(pkg_json.read_text())
            node_deps = list(data.get("dependencies", {}).keys())
            node_deps += list(data.get("devDependencies", {}).keys())
            if node_deps:
                deps["node"] = node_deps
        except (json.JSONDecodeError, OSError):
            pass

    # Rust: Cargo.toml
    cargo = repo_path / "Cargo.toml"
    if cargo.exists():
        try:
            import tomli

            with open(cargo, "rb") as f:
                data = tomli.load(f)
            rust_deps = list(data.get("dependencies", {}).keys())
            if rust_deps:
                deps["rust"] = rust_deps
        except (ImportError, Exception):
            pass

    # Go: go.mod
    go_mod = repo_path / "go.mod"
    if go_mod.exists():
        try:
            content = go_mod.read_text()
            go_deps = re.findall(r"^\s+(\S+)\s+v", content, re.MULTILINE)
            if go_deps:
                deps["go"] = go_deps
        except OSError:
            pass

    return deps


def _detect_patterns(repo_path: Path) -> list[dict[str, Any]]:
    """Detect notable patterns in the codebase."""
    patterns: list[dict[str, Any]] = []

    # CI/CD detection
    ci_findings: dict[str, Any] = {}
    ci_dirs = [
        (".github/workflows", "GitHub Actions"),
        (".gitlab-ci.yml", "GitLab CI"),
        (".circleci", "CircleCI"),
        ("Jenkinsfile", "Jenkins"),
    ]
    for ci_path, ci_name in ci_dirs:
        full = repo_path / ci_path
        if full.exists():
            ci_findings["ci_system"] = ci_name
            if full.is_dir():
                ci_findings["workflow_count"] = len(list(full.glob("*.yml"))) + len(
                    list(full.glob("*.yaml"))
                )
            break
    if ci_findings:
        patterns.append(
            {
                "category": "ci_cd",
                "findings": ci_findings,
                "confidence": 0.9,
                "section_content": f"CI: {ci_findings.get('ci_system', 'unknown')}",
            }
        )

    # Testing detection
    test_findings: dict[str, Any] = {}
    test_dirs = ["tests", "test", "spec", "__tests__"]
    for td in test_dirs:
        test_dir = repo_path / td
        if test_dir.is_dir():
            test_files = list(test_dir.rglob("test_*.py")) + list(test_dir.rglob("*_test.py"))
            test_files += list(test_dir.rglob("*.test.ts")) + list(test_dir.rglob("*.test.js"))
            test_files += list(test_dir.rglob("*.spec.ts")) + list(test_dir.rglob("*.spec.js"))
            test_findings["test_dir"] = td
            test_findings["test_file_count"] = len(test_files)
            break

    # Check for test config files
    test_configs = {
        "pytest.ini": "pytest",
        "setup.cfg": "pytest",
        "pyproject.toml": "pytest",
        "jest.config.js": "jest",
        "jest.config.ts": "jest",
        "vitest.config.ts": "vitest",
        ".mocharc.yml": "mocha",
    }
    for config_file, framework in test_configs.items():
        if (repo_path / config_file).exists():
            test_findings["test_framework"] = framework
            break

    if test_findings:
        patterns.append(
            {
                "category": "testing",
                "findings": test_findings,
                "confidence": 0.8,
                "section_content": f"Testing: {test_findings.get('test_framework', 'unknown')}",
            }
        )

    # Architecture patterns
    arch_findings: dict[str, Any] = {}
    framework_markers = {
        "fastapi": ["from fastapi", "FastAPI("],
        "django": ["from django", "INSTALLED_APPS"],
        "flask": ["from flask", "Flask("],
        "express": ["require('express')", "from 'express'"],
        "nextjs": ["next.config"],
        "react": ["from 'react'", 'from "react"'],
    }

    for framework, markers in framework_markers.items():
        for marker in markers:
            # Check a sample of files
            for ext in ["*.py", "*.js", "*.ts", "*.tsx"]:
                for f in list(repo_path.rglob(ext))[:50]:
                    try:
                        if marker in f.read_text(errors="ignore"):
                            arch_findings.setdefault("frameworks", []).append(framework)
                            break
                    except OSError:
                        pass
                if framework in arch_findings.get("frameworks", []):
                    break

    # Deduplicate frameworks
    if "frameworks" in arch_findings:
        arch_findings["frameworks"] = list(set(arch_findings["frameworks"]))

    # Check for Docker
    if (repo_path / "Dockerfile").exists() or (repo_path / "docker-compose.yml").exists():
        arch_findings["containerized"] = True

    if arch_findings:
        patterns.append(
            {
                "category": "architecture",
                "findings": arch_findings,
                "confidence": 0.7,
                "section_content": str(arch_findings),
            }
        )

    return patterns


def _compute_score(scan_data: dict[str, Any]) -> int:
    """Compute a quality score 0-100 for the repo."""
    score = 50  # baseline

    # Has tests
    for a in scan_data.get("analyses", []):
        if a.get("category") == "testing":
            test_count = a.get("findings", {}).get("test_file_count", 0)
            if test_count > 0:
                score += 10
            if test_count > 20:
                score += 5
            if a.get("findings", {}).get("test_framework"):
                score += 5

    # Has CI
    for a in scan_data.get("analyses", []):
        if a.get("category") == "ci_cd":
            score += 10

    # Has multiple languages (polyglot = interesting)
    lang_count = len(scan_data.get("languages", {}))
    if lang_count >= 2:
        score += 5
    if lang_count >= 3:
        score += 5

    # Has dependencies declared
    if scan_data.get("dependencies"):
        score += 5

    # Size factor — too small is less interesting
    total_files = scan_data.get("total_files", 0)
    if total_files > 10:
        score += 5

    return min(score, 100)


def _extract_architecture(scan_data: dict[str, Any]) -> str:
    """Summarize the architecture as a string."""
    parts: list[str] = []

    # Frameworks
    for a in scan_data.get("analyses", []):
        if a.get("category") == "architecture":
            frameworks = a.get("findings", {}).get("frameworks", [])
            parts.extend(f.title() for f in frameworks)

    # Primary language
    primary = scan_data.get("primary_language")
    if primary and primary not in parts:
        parts.insert(0, primary)

    # Container
    for a in scan_data.get("analyses", []):
        if a.get("category") == "architecture":
            if a.get("findings", {}).get("containerized"):
                parts.append("Docker")

    return " + ".join(parts) if parts else "Unknown"


def _extract_notable_patterns(scan_data: dict[str, Any]) -> list[str]:
    """Extract notable patterns from scan data."""
    patterns: list[str] = []

    for a in scan_data.get("analyses", []):
        category = a.get("category", "")
        findings = a.get("findings", {})

        if category == "ci_cd":
            ci = findings.get("ci_system")
            wf_count = findings.get("workflow_count", 0)
            if ci:
                patterns.append(f"{ci} with {wf_count} workflows")

        if category == "testing":
            framework = findings.get("test_framework")
            count = findings.get("test_file_count", 0)
            if framework:
                patterns.append(f"{framework} testing ({count} test files)")

        if category == "architecture":
            if findings.get("containerized"):
                patterns.append("Dockerized deployment")
            frameworks = findings.get("frameworks", [])
            for f in frameworks:
                patterns.append(f"{f.title()} framework")

    # Check for anchormd-specific analyses
    for a in scan_data.get("analyses", []):
        section = a.get("section_content", "")
        if (
            section
            and len(section) > 20
            and a.get("category")
            not in (
                "ci_cd",
                "testing",
                "architecture",
            )
        ):
            patterns.append(f"{a['category']}: {section[:100]}")

    return patterns


def _extract_testing_approach(scan_data: dict[str, Any]) -> str:
    """Summarize the testing approach."""
    for a in scan_data.get("analyses", []):
        if a.get("category") == "testing":
            findings = a.get("findings", {})
            framework = findings.get("test_framework", "unknown")
            count = findings.get("test_file_count", 0)
            test_dir = findings.get("test_dir", "tests")
            return f"{framework} framework, {count} test files in {test_dir}/"
    return "No tests detected"


def _find_novel_tools(scan_data: dict[str, Any], our_tools: set[str] | None = None) -> list[str]:
    """Find tools/libraries in the target repo we don't use."""
    if our_tools is None:
        # Known tools from our stack
        our_tools = {
            "pytest",
            "ruff",
            "mypy",
            "fastapi",
            "pydantic",
            "rich",
            "typer",
            "httpx",
            "chromadb",
            "ollama",
            "anthropic",
            "sqlalchemy",
            "alembic",
            "stripe",
            "uvicorn",
            "gunicorn",
            "docker",
            "github-actions",
            "pre-commit",
            "black",
            "isort",
            "flask",
            "django",
            "react",
            "next",
            "vite",
            "tailwindcss",
        }

    novel: list[str] = []
    all_deps: list[str] = []

    for lang_deps in scan_data.get("dependencies", {}).values():
        all_deps.extend(lang_deps)

    for dep in all_deps:
        dep_lower = dep.lower().strip()
        if dep_lower and dep_lower not in our_tools and not dep_lower.startswith("-"):
            novel.append(dep)

    return novel[:20]  # Cap at 20 most interesting


def _compare_with_ours(
    scan_data: dict[str, Any],
) -> dict[str, list[str]]:
    """Compare target repo patterns against our projects."""
    comparison: dict[str, list[str]] = {
        "things_they_do_better": [],
        "things_we_do_better": [],
        "novel_approaches": [],
    }

    # Check CI sophistication
    for a in scan_data.get("analyses", []):
        if a.get("category") == "ci_cd":
            wf_count = a.get("findings", {}).get("workflow_count", 0)
            if wf_count > 5:
                comparison["things_they_do_better"].append(f"More CI workflows ({wf_count})")

    # We always have these (from our stack)
    comparison["things_we_do_better"].extend(
        [
            "Consistent conventional commits across all repos",
            "anchormd CLAUDE.md generation",
            "Multi-package monorepo architecture",
        ]
    )

    # Novel dependencies
    novel = _find_novel_tools(scan_data)
    if novel:
        comparison["novel_approaches"].append(f"Uses tools we don't: {', '.join(novel[:5])}")

    # Architecture patterns
    for a in scan_data.get("analyses", []):
        if a.get("category") == "architecture":
            frameworks = a.get("findings", {}).get("frameworks", [])
            for f in frameworks:
                if f not in ("fastapi", "react", "nextjs"):
                    comparison["novel_approaches"].append(f"Uses {f.title()} (not in our stack)")

    return comparison


def harvest_repo(
    target: str,
    compare: bool = True,
    depth: str = "quick",
    memory_layer: Any | None = None,
) -> HarvestResult:
    """Main harvest function. Clone, scan, extract patterns.

    Args:
        target: GitHub repo URL or user/repo string.
        compare: Whether to compare against our projects.
        depth: "quick" for shallow scan, "deep" for full clone.
        memory_layer: Optional MemoryLayer to store findings.

    Returns:
        HarvestResult with structured findings.
    """
    repo_name = _extract_repo_name(target)
    clone_url = _normalize_target(target)
    clone_depth = 1 if depth == "quick" else 0

    result = HarvestResult(repo=repo_name)

    tmp_dir = tempfile.mkdtemp(prefix="animus_harvest_")
    repo_path = Path(tmp_dir) / repo_name.split("/")[-1]

    try:
        # Clone
        logger.info("Cloning %s to %s", clone_url, repo_path)
        _clone_repo(clone_url, repo_path, depth=clone_depth)

        # Scan
        logger.info("Scanning %s", repo_path)
        scan_data = _scan_with_anchormd(repo_path)

        # Extract findings
        result.score = _compute_score(scan_data)
        result.architecture = _extract_architecture(scan_data)
        result.notable_patterns = _extract_notable_patterns(scan_data)
        result.tools_worth_adopting = _find_novel_tools(scan_data)
        result.testing_approach = _extract_testing_approach(scan_data)

        if compare:
            result.comparison = _compare_with_ours(scan_data)

        # Store in memory if available
        if memory_layer is not None:
            _store_in_memory(memory_layer, result)

        return result

    finally:
        # Always clean up
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _store_in_memory(memory_layer: Any, result: HarvestResult) -> None:
    """Store harvest findings in animus memory."""
    try:
        from animus.memory import MemoryType

        content = (
            f"Harvested repo: {result.repo}\n"
            f"Score: {result.score}/100\n"
            f"Architecture: {result.architecture}\n"
            f"Notable: {', '.join(result.notable_patterns[:5])}\n"
            f"Tools to consider: {', '.join(result.tools_worth_adopting[:5])}\n"
            f"Testing: {result.testing_approach}"
        )

        memory_layer.remember(
            content=content,
            memory_type=MemoryType.SEMANTIC,
            tags=["harvest", "discovery", result.repo.replace("/", "-")],
            source="harvest",
        )
        logger.info("Stored harvest findings in memory for %s", result.repo)
    except Exception as e:
        logger.warning("Failed to store harvest in memory: %s", e)


# ---- Tool handler for ToolRegistry integration ----


def _tool_harvest(params: dict) -> ToolResult:
    """Tool handler for animus_harvest."""
    target = params.get("target")
    if not target:
        return ToolResult(
            tool_name="animus_harvest",
            success=False,
            output=None,
            error="Missing required parameter: target",
        )

    compare = params.get("compare", True)
    depth = params.get("depth", "quick")

    if depth not in ("quick", "deep"):
        return ToolResult(
            tool_name="animus_harvest",
            success=False,
            output=None,
            error="depth must be 'quick' or 'deep'",
        )

    try:
        result = harvest_repo(
            target=target,
            compare=compare,
            depth=depth,
        )
        return ToolResult(
            tool_name="animus_harvest",
            success=True,
            output=json.dumps(result.to_dict(), indent=2),
        )
    except ValueError as e:
        return ToolResult(
            tool_name="animus_harvest",
            success=False,
            output=None,
            error=str(e),
        )
    except RuntimeError as e:
        return ToolResult(
            tool_name="animus_harvest",
            success=False,
            output=None,
            error=str(e),
        )
    except Exception as e:
        logger.exception("Harvest failed for %s", target)
        return ToolResult(
            tool_name="animus_harvest",
            success=False,
            output=None,
            error=f"Harvest failed: {e}",
        )


# Tool definition for registration
HARVEST_TOOL = Tool(
    name="animus_harvest",
    description=(
        "Scan external repos with anchormd and extract learnable patterns, architectures, and tools"
    ),
    parameters={
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "GitHub repo URL or username/repo",
            },
            "compare": {
                "type": "boolean",
                "description": "Compare against our projects (default: true)",
            },
            "depth": {
                "type": "string",
                "enum": ["quick", "deep"],
                "description": "Scan depth: quick (shallow clone) or deep (full clone)",
            },
        },
        "required": ["target"],
    },
    handler=_tool_harvest,
    category="analysis",
)
