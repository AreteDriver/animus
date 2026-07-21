"""Tests for sandbox codebase analyzer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from animus_kernel.sandbox.analyzer import (
    AnalysisResult,
    CodebaseAnalyzer,
    ImprovementCategory,
    ImprovementSuggestion,
)


# ═══════════════════════════════════════════════════════════════════
# CodebaseAnalyzer static analysis tests
# ═══════════════════════════════════════════════════════════════════


class TestCodebaseAnalyzerStatic:
    def test_analyze_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = CodebaseAnalyzer(codebase_path=tmpdir)
            result = analyzer.analyze()
            assert isinstance(result, AnalysisResult)
            assert result.files_analyzed == 0
            assert result.issues_found == 0

    def test_detect_long_function(self):
        # Add a trailing short function so the analyzer's regex loop
        # has a chance to measure long_one's length.
        code = '\n'.join([
            "def short():",
            "    pass",
            "",
            "def long_one():",
        ] + ["    print('line')"] * 60 + [
            "",
            "def trailing():",
            "    pass",
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            (src / "module.py").write_text(code)

            analyzer = CodebaseAnalyzer(codebase_path=tmpdir)
            result = analyzer.analyze(categories=[ImprovementCategory.CODE_QUALITY])

            long_funcs = [s for s in result.suggestions if "Long function" in s.title]
            assert len(long_funcs) == 1
            assert "long_one" in long_funcs[0].title

    def test_detect_todo(self):
        code = "def foo():\n    # TODO: fix this later\n    pass\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            (src / "module.py").write_text(code)

            analyzer = CodebaseAnalyzer(codebase_path=tmpdir)
            result = analyzer.analyze(categories=[ImprovementCategory.CODE_QUALITY])

            todos = [s for s in result.suggestions if "TODO" in s.title]
            assert len(todos) == 1
            assert "fix this later" in todos[0].description

    def test_detect_bare_except(self):
        code = """def risky():
    try:
        pass
    except:
        pass
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            (src / "module.py").write_text(code)

            analyzer = CodebaseAnalyzer(codebase_path=tmpdir)
            result = analyzer.analyze(categories=[ImprovementCategory.CODE_QUALITY])

            bare = [s for s in result.suggestions if "Bare except" in s.title]
            assert len(bare) == 1

    def test_no_false_positive_on_specific_except(self):
        code = """def safe():
    try:
        pass
    except Exception:
        pass
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            (src / "module.py").write_text(code)

            analyzer = CodebaseAnalyzer(codebase_path=tmpdir)
            result = analyzer.analyze(categories=[ImprovementCategory.CODE_QUALITY])

            bare = [s for s in result.suggestions if "Bare except" in s.title]
            assert len(bare) == 0

    def test_detect_missing_module_docstring(self):
        code = "def hello():\n    pass\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            (src / "module.py").write_text(code)

            analyzer = CodebaseAnalyzer(codebase_path=tmpdir)
            result = analyzer.analyze(categories=[ImprovementCategory.DOCUMENTATION])

            missing = [s for s in result.suggestions if "Missing module docstring" in s.title]
            assert len(missing) == 1

    def test_no_false_positive_on_existing_module_docstring(self):
        code = '"""This module says hello."""\n\ndef hello():\n    pass\n'

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            (src / "module.py").write_text(code)

            analyzer = CodebaseAnalyzer(codebase_path=tmpdir)
            result = analyzer.analyze(categories=[ImprovementCategory.DOCUMENTATION])

            missing = [s for s in result.suggestions if "Missing module docstring" in s.title]
            assert len(missing) == 0

    def test_detect_missing_function_docstring(self):
        code = '"""Module."""\n\ndef hello():\n    pass\n'

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            (src / "module.py").write_text(code)

            analyzer = CodebaseAnalyzer(codebase_path=tmpdir)
            result = analyzer.analyze(categories=[ImprovementCategory.DOCUMENTATION])

            missing = [s for s in result.suggestions if "Missing docstring: hello" in s.title]
            assert len(missing) == 1

    def test_skip_private_function_docstring(self):
        code = '"""Module."""\n\ndef _private():\n    pass\n'

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            (src / "module.py").write_text(code)

            analyzer = CodebaseAnalyzer(codebase_path=tmpdir)
            result = analyzer.analyze(categories=[ImprovementCategory.DOCUMENTATION])

            missing = [s for s in result.suggestions if "_private" in s.title]
            assert len(missing) == 0

    def test_detect_missing_tests(self):
        code = '"""Module."""\n\ndef add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n'

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            pkg = src / "mypackage"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / "math.py").write_text(code)

            analyzer = CodebaseAnalyzer(codebase_path=tmpdir)
            result = analyzer.analyze(categories=[ImprovementCategory.TEST_COVERAGE])

            missing = [s for s in result.suggestions if "Missing tests" in s.title]
            assert len(missing) == 1
            assert "math.py" in missing[0].title
            assert "add" in missing[0].implementation_hints
            assert "sub" in missing[0].implementation_hints

    def test_skip_test_files(self):
        code = '"""Module."""\n\ndef add(a, b):\n    return a + b\n'

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            (src / "test_module.py").write_text(code)

            analyzer = CodebaseAnalyzer(codebase_path=tmpdir)
            result = analyzer.analyze(categories=[ImprovementCategory.TEST_COVERAGE])

            missing = [s for s in result.suggestions if "Missing tests" in s.title]
            assert len(missing) == 0

    def test_skip_self_improve_files(self):
        code = "def helper():\n    pass\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            self_improve = src / "self_improve"
            self_improve.mkdir()
            (self_improve / "helper.py").write_text(code)

            analyzer = CodebaseAnalyzer(codebase_path=tmpdir)
            result = analyzer.analyze()

            # Should not analyze files inside self_improve directory
            affected = [s.affected_files for s in result.suggestions]
            flat = [f for files in affected for f in files]
            assert not any("self_improve" in f for f in flat)

    def test_focus_paths(self):
        code = "def hello():\n    pass\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            (src / "target.py").write_text(code)
            (src / "other.py").write_text(code)

            analyzer = CodebaseAnalyzer(codebase_path=tmpdir)
            result = analyzer.analyze(focus_paths=["src/target.py"])

            affected = [f for s in result.suggestions for f in s.affected_files]
            assert any("target.py" in f for f in affected)
            assert not any("other.py" in f for f in affected)

    def test_category_filtering(self):
        code = '"""Module."""\n\ndef hello():\n    pass\n'

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            (src / "module.py").write_text(code)

            analyzer = CodebaseAnalyzer(codebase_path=tmpdir)
            doc_result = analyzer.analyze(categories=[ImprovementCategory.DOCUMENTATION])

            # Should only find doc issues
            assert all(s.category == ImprovementCategory.DOCUMENTATION for s in doc_result.suggestions)


# ═══════════════════════════════════════════════════════════════════
# Performance analysis tests
# ═══════════════════════════════════════════════════════════════════


class TestCodebaseAnalyzerPerformance:
    def test_detect_nested_loops(self):
        code = """def process():
    items = [1, 2, 3]
    for i in items:
        for j in items:
            print(i, j)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            (src / "module.py").write_text(code)

            analyzer = CodebaseAnalyzer(codebase_path=tmpdir)
            result = analyzer.analyze(categories=[ImprovementCategory.PERFORMANCE])

            nested = [s for s in result.suggestions if "Nested loop" in s.title]
            assert len(nested) == 1
            assert nested[0].category == ImprovementCategory.PERFORMANCE
            assert nested[0].priority == 2

    def test_detect_string_concat_in_loop(self):
        code = """def build():
    result = ""
    for i in range(10):
        result = result + str(i)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            (src / "module.py").write_text(code)

            analyzer = CodebaseAnalyzer(codebase_path=tmpdir)
            result = analyzer.analyze(categories=[ImprovementCategory.PERFORMANCE])

            concat = [s for s in result.suggestions if "String concatenation" in s.title]
            assert len(concat) == 1
            assert concat[0].category == ImprovementCategory.PERFORMANCE
            assert "join" in concat[0].implementation_hints

    def test_detect_list_insert_zero_in_loop(self):
        code = """def prepend():
    out = []
    for i in range(10):
        out.insert(0, i)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            (src / "module.py").write_text(code)

            analyzer = CodebaseAnalyzer(codebase_path=tmpdir)
            result = analyzer.analyze(categories=[ImprovementCategory.PERFORMANCE])

            inserts = [s for s in result.suggestions if "list.insert(0" in s.title]
            assert len(inserts) == 1
            assert inserts[0].category == ImprovementCategory.PERFORMANCE
            assert inserts[0].priority == 1

    def test_detect_list_append_comprehension_opportunity(self):
        code = """def collect():
    out = []
    for i in range(10):
        out.append(i * 2)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            (src / "module.py").write_text(code)

            analyzer = CodebaseAnalyzer(codebase_path=tmpdir)
            result = analyzer.analyze(categories=[ImprovementCategory.PERFORMANCE])

            comp = [s for s in result.suggestions if "List append" in s.title]
            assert len(comp) == 1
            assert comp[0].category == ImprovementCategory.PERFORMANCE
            assert "comprehension" in comp[0].description.lower()

    def test_no_performance_issues_on_clean_code(self):
        code = """def fast():
    return [i * 2 for i in range(10)]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            (src / "module.py").write_text(code)

            analyzer = CodebaseAnalyzer(codebase_path=tmpdir)
            result = analyzer.analyze(categories=[ImprovementCategory.PERFORMANCE])

            perf = [s for s in result.suggestions if s.category == ImprovementCategory.PERFORMANCE]
            assert len(perf) == 0

    def test_self_targeting_skips_self_improve_by_default(self):
        code = "def helper():\n    pass\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            self_improve = src / "self_improve"
            self_improve.mkdir()
            (self_improve / "helper.py").write_text(code)

            analyzer = CodebaseAnalyzer(codebase_path=tmpdir)
            result = analyzer.analyze()
            affected = [f for s in result.suggestions for f in s.affected_files]
            assert not any("self_improve" in f for f in affected)

    def test_self_targeting_allows_self_improve_when_enabled(self):
        code = 'def helper():\n    # TODO: optimize\n    pass\n'

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            src.mkdir()
            self_improve = src / "self_improve"
            self_improve.mkdir()
            (self_improve / "helper.py").write_text(code)

            analyzer = CodebaseAnalyzer(codebase_path=tmpdir, allow_self_targeting=True)
            result = analyzer.analyze(categories=[ImprovementCategory.CODE_QUALITY])
            affected = [f for s in result.suggestions for f in s.affected_files]
            assert any("self_improve" in f for f in affected)


# ═══════════════════════════════════════════════════════════════════
# ImprovementSuggestion dataclass tests
# ═══════════════════════════════════════════════════════════════════


class TestImprovementSuggestion:
    def test_default_priority_and_lines(self):
        s = ImprovementSuggestion(
            id="test-1",
            category=ImprovementCategory.REFACTORING,
            title="Test",
            description="Desc",
            affected_files=["a.py"],
        )
        assert s.priority == 3
        assert s.estimated_lines == 0

    def test_custom_values(self):
        s = ImprovementSuggestion(
            id="test-2",
            category=ImprovementCategory.PERFORMANCE,
            title="Slow loop",
            description="Use list comprehension",
            affected_files=["a.py"],
            priority=1,
            estimated_lines=5,
            reasoning="N+1 query pattern",
            implementation_hints="Replace with list comp",
        )
        assert s.priority == 1
        assert s.estimated_lines == 5
