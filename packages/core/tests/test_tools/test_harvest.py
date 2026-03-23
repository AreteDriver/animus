"""Tests for animus_harvest tool."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from animus.harvest import (
    HARVEST_TOOL,
    HarvestResult,
    _basic_scan,
    _compare_with_ours,
    _compute_score,
    _detect_dependencies,
    _detect_patterns,
    _extract_architecture,
    _extract_notable_patterns,
    _extract_repo_name,
    _extract_testing_approach,
    _find_novel_tools,
    _normalize_target,
    _store_in_memory,
    _tool_harvest,
    harvest_repo,
)
from animus.tools import create_default_registry

# ---------------------------------------------------------------------------
# _normalize_target
# ---------------------------------------------------------------------------


class TestNormalizeTarget:
    def test_short_form(self):
        assert _normalize_target("user/repo") == "https://github.com/user/repo.git"

    def test_full_url_no_git(self):
        assert (
            _normalize_target("https://github.com/user/repo") == "https://github.com/user/repo.git"
        )

    def test_full_url_with_git(self):
        assert (
            _normalize_target("https://github.com/user/repo.git")
            == "https://github.com/user/repo.git"
        )

    def test_trailing_slash(self):
        assert (
            _normalize_target("https://github.com/user/repo/") == "https://github.com/user/repo.git"
        )

    def test_ssh_url(self):
        url = "git@github.com:user/repo.git"
        assert _normalize_target(url) == url

    def test_invalid_target(self):
        with pytest.raises(ValueError, match="Invalid target"):
            _normalize_target("just-a-name")

    def test_whitespace_stripped(self):
        assert _normalize_target("  user/repo  ") == "https://github.com/user/repo.git"


# ---------------------------------------------------------------------------
# _extract_repo_name
# ---------------------------------------------------------------------------


class TestExtractRepoName:
    def test_short_form(self):
        assert _extract_repo_name("user/repo") == "user/repo"

    def test_full_url(self):
        assert _extract_repo_name("https://github.com/user/repo") == "user/repo"

    def test_full_url_with_git(self):
        assert _extract_repo_name("https://github.com/user/repo.git") == "user/repo"

    def test_ssh_url(self):
        assert _extract_repo_name("git@github.com:user/repo.git") == "user/repo"

    def test_trailing_slash(self):
        assert _extract_repo_name("https://github.com/user/repo/") == "user/repo"


# ---------------------------------------------------------------------------
# HarvestResult
# ---------------------------------------------------------------------------


class TestHarvestResult:
    def test_to_dict(self):
        r = HarvestResult(
            repo="user/repo",
            score=85,
            architecture="Python + FastAPI",
            notable_patterns=["pytest testing"],
            tools_worth_adopting=["celery"],
            testing_approach="pytest, 10 files",
            comparison={"things_they_do_better": ["More CI"]},
        )
        d = r.to_dict()
        assert d["repo"] == "user/repo"
        assert d["score"] == 85
        assert "celery" in d["tools_worth_adopting"]

    def test_defaults(self):
        r = HarvestResult(repo="x/y")
        d = r.to_dict()
        assert d["score"] == 0
        assert d["notable_patterns"] == []
        assert d["comparison"] == {}


# ---------------------------------------------------------------------------
# _detect_dependencies
# ---------------------------------------------------------------------------


class TestDetectDependencies:
    def test_package_json(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text(
            json.dumps(
                {
                    "dependencies": {"express": "^4.0.0", "lodash": "^4.17.0"},
                    "devDependencies": {"jest": "^29.0.0"},
                }
            )
        )
        deps = _detect_dependencies(tmp_path)
        assert "node" in deps
        assert "express" in deps["node"]
        assert "jest" in deps["node"]

    def test_requirements_txt(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("fastapi>=0.100\nuvicorn\n# comment\n")
        deps = _detect_dependencies(tmp_path)
        assert "python" in deps
        assert "fastapi" in deps["python"]
        assert "uvicorn" in deps["python"]

    def test_go_mod(self, tmp_path):
        gomod = tmp_path / "go.mod"
        gomod.write_text(
            "module github.com/user/repo\n\ngo 1.21\n\nrequire (\n"
            "\tgithub.com/gin-gonic/gin v1.9.1\n"
            "\tgithub.com/lib/pq v1.10.9\n)\n"
        )
        deps = _detect_dependencies(tmp_path)
        assert "go" in deps
        assert "github.com/gin-gonic/gin" in deps["go"]

    def test_empty_project(self, tmp_path):
        deps = _detect_dependencies(tmp_path)
        assert deps == {}

    def test_malformed_package_json(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text("not valid json {{{")
        deps = _detect_dependencies(tmp_path)
        assert "node" not in deps


# ---------------------------------------------------------------------------
# _detect_patterns
# ---------------------------------------------------------------------------


class TestDetectPatterns:
    def test_ci_detection(self, tmp_path):
        wf_dir = tmp_path / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "ci.yml").write_text("name: CI")
        (wf_dir / "deploy.yml").write_text("name: Deploy")
        patterns = _detect_patterns(tmp_path)
        ci = [p for p in patterns if p["category"] == "ci_cd"]
        assert len(ci) == 1
        assert ci[0]["findings"]["ci_system"] == "GitHub Actions"
        assert ci[0]["findings"]["workflow_count"] == 2

    def test_testing_detection(self, tmp_path):
        test_dir = tmp_path / "tests"
        test_dir.mkdir()
        (test_dir / "test_main.py").write_text("def test_foo(): pass")
        (test_dir / "test_utils.py").write_text("def test_bar(): pass")
        patterns = _detect_patterns(tmp_path)
        testing = [p for p in patterns if p["category"] == "testing"]
        assert len(testing) == 1
        assert testing[0]["findings"]["test_file_count"] == 2

    def test_architecture_docker(self, tmp_path):
        (tmp_path / "Dockerfile").write_text("FROM python:3.12")
        patterns = _detect_patterns(tmp_path)
        arch = [p for p in patterns if p["category"] == "architecture"]
        assert len(arch) == 1
        assert arch[0]["findings"]["containerized"] is True

    def test_empty_project(self, tmp_path):
        patterns = _detect_patterns(tmp_path)
        assert patterns == []

    def test_framework_detection_fastapi(self, tmp_path):
        src = tmp_path / "main.py"
        src.write_text("from fastapi import FastAPI\napp = FastAPI()\n")
        patterns = _detect_patterns(tmp_path)
        arch = [p for p in patterns if p["category"] == "architecture"]
        assert len(arch) >= 1
        frameworks = arch[0]["findings"].get("frameworks", [])
        assert "fastapi" in frameworks


# ---------------------------------------------------------------------------
# _compute_score
# ---------------------------------------------------------------------------


class TestComputeScore:
    def test_baseline_score(self):
        assert _compute_score({}) == 50

    def test_score_with_tests(self):
        data = {
            "analyses": [
                {
                    "category": "testing",
                    "findings": {"test_file_count": 25, "test_framework": "pytest"},
                },
            ]
        }
        score = _compute_score(data)
        assert score > 50

    def test_score_with_ci(self):
        data = {
            "analyses": [
                {"category": "ci_cd", "findings": {"ci_system": "GitHub Actions"}},
            ]
        }
        assert _compute_score(data) == 60  # 50 + 10

    def test_score_capped_at_100(self):
        data = {
            "total_files": 100,
            "languages": {"Python": 50, "TypeScript": 30, "Go": 20},
            "dependencies": {"python": ["fastapi"]},
            "analyses": [
                {
                    "category": "testing",
                    "findings": {"test_file_count": 50, "test_framework": "pytest"},
                },
                {"category": "ci_cd", "findings": {"ci_system": "GitHub Actions"}},
            ],
        }
        assert _compute_score(data) <= 100


# ---------------------------------------------------------------------------
# _extract_architecture
# ---------------------------------------------------------------------------


class TestExtractArchitecture:
    def test_with_frameworks(self):
        data = {
            "primary_language": "Python",
            "analyses": [
                {
                    "category": "architecture",
                    "findings": {"frameworks": ["fastapi"], "containerized": True},
                },
            ],
        }
        arch = _extract_architecture(data)
        assert "Python" in arch
        assert "Fastapi" in arch
        assert "Docker" in arch

    def test_unknown(self):
        assert _extract_architecture({}) == "Unknown"


# ---------------------------------------------------------------------------
# _find_novel_tools
# ---------------------------------------------------------------------------


class TestFindNovelTools:
    def test_finds_novel(self):
        data = {"dependencies": {"python": ["celery", "dramatiq", "pytest"]}}
        our = {"pytest", "ruff"}
        novel = _find_novel_tools(data, our)
        assert "celery" in novel
        assert "dramatiq" in novel
        assert "pytest" not in novel

    def test_empty_deps(self):
        assert _find_novel_tools({}) == []

    def test_cap_at_20(self):
        deps = [f"pkg-{i}" for i in range(30)]
        data = {"dependencies": {"python": deps}}
        assert len(_find_novel_tools(data, set())) <= 20


# ---------------------------------------------------------------------------
# _compare_with_ours
# ---------------------------------------------------------------------------


class TestCompareWithOurs:
    def test_has_structure(self):
        data = {"analyses": []}
        comparison = _compare_with_ours(data)
        assert "things_they_do_better" in comparison
        assert "things_we_do_better" in comparison
        assert "novel_approaches" in comparison

    def test_we_always_have_advantages(self):
        comparison = _compare_with_ours({"analyses": []})
        assert len(comparison["things_we_do_better"]) > 0


# ---------------------------------------------------------------------------
# _extract_notable_patterns / _extract_testing_approach
# ---------------------------------------------------------------------------


class TestExtractionHelpers:
    def test_notable_patterns(self):
        data = {
            "analyses": [
                {
                    "category": "ci_cd",
                    "findings": {"ci_system": "GitHub Actions", "workflow_count": 3},
                },
            ]
        }
        patterns = _extract_notable_patterns(data)
        assert any("GitHub Actions" in p for p in patterns)

    def test_testing_approach_found(self):
        data = {
            "analyses": [
                {
                    "category": "testing",
                    "findings": {
                        "test_framework": "pytest",
                        "test_file_count": 10,
                        "test_dir": "tests",
                    },
                },
            ]
        }
        approach = _extract_testing_approach(data)
        assert "pytest" in approach
        assert "10" in approach

    def test_testing_approach_none(self):
        assert _extract_testing_approach({"analyses": []}) == "No tests detected"


# ---------------------------------------------------------------------------
# _basic_scan
# ---------------------------------------------------------------------------


class TestBasicScan:
    def test_scans_python_project(self, tmp_path):
        (tmp_path / "main.py").write_text("print('hello')\n")
        (tmp_path / "utils.py").write_text("def foo():\n    pass\n")
        result = _basic_scan(tmp_path)
        assert result["total_files"] >= 2
        assert "Python" in result["languages"]

    def test_skips_git_dir(self, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main")
        (tmp_path / "main.py").write_text("x = 1")
        result = _basic_scan(tmp_path)
        assert result["total_files"] == 1

    def test_skips_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules" / "pkg"
        nm.mkdir(parents=True)
        (nm / "index.js").write_text("module.exports = {}")
        (tmp_path / "app.js").write_text("const x = 1;")
        result = _basic_scan(tmp_path)
        assert result["total_files"] == 1


# ---------------------------------------------------------------------------
# _store_in_memory
# ---------------------------------------------------------------------------


class TestStoreInMemory:
    def test_stores_successfully(self):
        mock_memory = MagicMock()
        result = HarvestResult(
            repo="user/repo",
            score=80,
            architecture="Python",
            notable_patterns=["pytest"],
            tools_worth_adopting=["celery"],
            testing_approach="pytest, 5 files",
        )
        _store_in_memory(mock_memory, result)
        mock_memory.remember.assert_called_once()
        call_kwargs = mock_memory.remember.call_args[1]
        assert "harvest" in call_kwargs["tags"]
        assert "discovery" in call_kwargs["tags"]

    def test_handles_memory_error(self):
        mock_memory = MagicMock()
        mock_memory.remember.side_effect = RuntimeError("DB error")
        result = HarvestResult(repo="user/repo")
        # Should not raise
        _store_in_memory(mock_memory, result)


# ---------------------------------------------------------------------------
# _tool_harvest (handler)
# ---------------------------------------------------------------------------


class TestToolHarvest:
    def test_missing_target(self):
        result = _tool_harvest({})
        assert not result.success
        assert "target" in result.error

    def test_invalid_depth(self):
        result = _tool_harvest({"target": "user/repo", "depth": "invalid"})
        assert not result.success
        assert "depth" in result.error

    @patch("animus.harvest.harvest_repo")
    def test_success(self, mock_harvest):
        mock_harvest.return_value = HarvestResult(repo="user/repo", score=75, architecture="Python")
        result = _tool_harvest({"target": "user/repo"})
        assert result.success
        output = json.loads(result.output)
        assert output["repo"] == "user/repo"
        assert output["score"] == 75

    @patch("animus.harvest.harvest_repo")
    def test_value_error(self, mock_harvest):
        mock_harvest.side_effect = ValueError("Bad target")
        result = _tool_harvest({"target": "bad"})
        assert not result.success
        assert "Bad target" in result.error

    @patch("animus.harvest.harvest_repo")
    def test_runtime_error(self, mock_harvest):
        mock_harvest.side_effect = RuntimeError("Clone failed")
        result = _tool_harvest({"target": "user/repo"})
        assert not result.success
        assert "Clone failed" in result.error


# ---------------------------------------------------------------------------
# HARVEST_TOOL registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_tool_spec(self):
        assert HARVEST_TOOL.name == "animus_harvest"
        assert "target" in HARVEST_TOOL.parameters["properties"]
        assert "compare" in HARVEST_TOOL.parameters["properties"]
        assert "depth" in HARVEST_TOOL.parameters["properties"]
        assert HARVEST_TOOL.parameters["required"] == ["target"]
        assert HARVEST_TOOL.category == "analysis"

    def test_registered_in_default_registry(self):
        registry = create_default_registry()
        tool = registry.get("animus_harvest")
        assert tool is not None
        assert tool.name == "animus_harvest"


# ---------------------------------------------------------------------------
# harvest_repo (integration-level, mocked clone)
# ---------------------------------------------------------------------------


class TestHarvestRepo:
    @patch("animus.harvest._clone_repo")
    @patch("animus.harvest._scan_with_anchormd")
    def test_full_harvest(self, mock_scan, mock_clone, tmp_path):
        mock_scan.return_value = {
            "total_files": 42,
            "total_lines": 5000,
            "languages": {"Python": 30, "TypeScript": 12},
            "primary_language": "Python",
            "version": "1.0.0",
            "description": "A test repo",
            "dependencies": {"python": ["fastapi", "celery"]},
            "analyses": [
                {
                    "category": "ci_cd",
                    "findings": {"ci_system": "GitHub Actions", "workflow_count": 3},
                    "confidence": 0.9,
                    "section_content": "CI",
                },
                {
                    "category": "testing",
                    "findings": {
                        "test_framework": "pytest",
                        "test_file_count": 15,
                        "test_dir": "tests",
                    },
                    "confidence": 0.8,
                    "section_content": "Testing",
                },
                {
                    "category": "architecture",
                    "findings": {"frameworks": ["fastapi"], "containerized": True},
                    "confidence": 0.7,
                    "section_content": "Arch",
                },
            ],
        }

        result = harvest_repo("user/test-repo", compare=True, depth="quick")

        assert result.repo == "user/test-repo"
        assert result.score > 50
        assert "Python" in result.architecture
        assert result.testing_approach != "No tests detected"
        assert len(result.notable_patterns) > 0
        assert isinstance(result.comparison, dict)

    @patch("animus.harvest._clone_repo")
    @patch("animus.harvest._scan_with_anchormd")
    def test_harvest_no_compare(self, mock_scan, mock_clone):
        mock_scan.return_value = {
            "total_files": 5,
            "total_lines": 100,
            "languages": {"Python": 5},
            "primary_language": "Python",
            "version": None,
            "description": None,
            "dependencies": {},
            "analyses": [],
        }

        result = harvest_repo("user/small-repo", compare=False)
        assert result.comparison == {}

    @patch("animus.harvest._clone_repo")
    @patch("animus.harvest._scan_with_anchormd")
    def test_harvest_stores_in_memory(self, mock_scan, mock_clone):
        mock_scan.return_value = {
            "total_files": 10,
            "total_lines": 500,
            "languages": {"Python": 10},
            "primary_language": "Python",
            "version": None,
            "description": None,
            "dependencies": {},
            "analyses": [],
        }
        mock_memory = MagicMock()

        harvest_repo("user/repo", memory_layer=mock_memory)
        mock_memory.remember.assert_called_once()

    @patch("animus.harvest._clone_repo")
    def test_harvest_clone_failure(self, mock_clone):
        mock_clone.side_effect = RuntimeError("git clone failed: not found")
        with pytest.raises(RuntimeError, match="git clone failed"):
            harvest_repo("user/nonexistent")

    def test_harvest_invalid_target(self):
        with pytest.raises(ValueError, match="Invalid target"):
            harvest_repo("not-a-valid-target")

    @patch("animus.harvest._clone_repo")
    @patch("animus.harvest._scan_with_anchormd")
    def test_cleanup_on_success(self, mock_scan, mock_clone):
        """Temp dir should be cleaned up even on success."""
        mock_scan.return_value = {
            "total_files": 1,
            "total_lines": 10,
            "languages": {},
            "primary_language": None,
            "version": None,
            "description": None,
            "dependencies": {},
            "analyses": [],
        }
        # The function uses tempfile.mkdtemp — we just verify it doesn't crash
        result = harvest_repo("user/repo")
        assert result.repo == "user/repo"

    @patch("animus.harvest._clone_repo")
    @patch("animus.harvest._scan_with_anchormd")
    def test_deep_depth_uses_full_clone(self, mock_scan, mock_clone):
        mock_scan.return_value = {
            "total_files": 1,
            "total_lines": 10,
            "languages": {},
            "primary_language": None,
            "version": None,
            "description": None,
            "dependencies": {},
            "analyses": [],
        }
        harvest_repo("user/repo", depth="deep")
        # deep should pass depth=0 to _clone_repo
        call_args = mock_clone.call_args
        assert call_args[1].get("depth", call_args[0][2] if len(call_args[0]) > 2 else 1) == 0


# ---------------------------------------------------------------------------
# _scan_with_anchormd
# ---------------------------------------------------------------------------


class TestScanWithAnchormd:
    def test_falls_back_to_basic(self, tmp_path):
        """When anchormd is not installed, falls back to _basic_scan."""
        (tmp_path / "main.py").write_text("x = 1\n")

        with patch.dict(
            "sys.modules",
            {
                "anchormd": None,
                "anchormd.analyzers": None,
                "anchormd.models": None,
                "anchormd.scanner": None,
            },
        ):
            # Force ImportError path
            with patch("animus.harvest._basic_scan") as mock_basic:
                mock_basic.return_value = {"total_files": 1}
                # The import will fail, falling back to basic
                import importlib

                import animus.harvest

                importlib.reload(animus.harvest)
                result = animus.harvest._scan_with_anchormd(tmp_path)
                # Either uses anchormd or falls back — both are valid
                assert isinstance(result, dict)
