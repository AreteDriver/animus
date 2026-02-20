"""Tests for CLI rich output utilities."""

import pytest

import animus_forge.cli.rich_output as rich_output_mod
from animus_forge.cli.rich_output import (
    OutputStyle,
    RichOutput,
    StepProgress,
    get_output,
    print_error,
    print_header,
    print_info,
    print_success,
    print_table,
    print_warning,
)


class TestOutputStyle:
    """Tests for OutputStyle enum."""

    def test_all_styles_defined(self):
        assert OutputStyle.SUCCESS.value == "green"
        assert OutputStyle.ERROR.value == "red"
        assert OutputStyle.WARNING.value == "yellow"
        assert OutputStyle.INFO.value == "blue"
        assert OutputStyle.MUTED.value == "dim"
        assert OutputStyle.HIGHLIGHT.value == "bold cyan"


class TestStepProgress:
    """Tests for StepProgress dataclass."""

    def test_creation(self):
        sp = StepProgress("build", "Building", "running")
        assert sp.step_id == "build"
        assert sp.step_name == "Building"
        assert sp.status == "running"
        assert sp.message is None
        assert sp.duration_ms is None

    def test_with_optional_fields(self):
        sp = StepProgress("test", "Testing", "completed", message="All passed", duration_ms=1500)
        assert sp.message == "All passed"
        assert sp.duration_ms == 1500


class TestRichOutputPlain:
    """Tests for RichOutput in plain text mode."""

    @pytest.fixture
    def output(self):
        return RichOutput(force_plain=True)

    def test_force_plain(self, output):
        assert not output.use_rich
        assert output.console is None

    def test_print_plain(self, output, capsys):
        output.print("hello")
        assert "hello" in capsys.readouterr().out

    def test_print_with_prefix(self, output, capsys):
        output.print("msg", prefix=">>")
        assert ">> msg" in capsys.readouterr().out

    def test_success(self, output, capsys):
        output.success("done")
        assert "done" in capsys.readouterr().out

    def test_error(self, output, capsys):
        output.error("fail")
        assert "fail" in capsys.readouterr().out

    def test_warning(self, output, capsys):
        output.warning("warn")
        assert "warn" in capsys.readouterr().out

    def test_info(self, output, capsys):
        output.info("note")
        assert "note" in capsys.readouterr().out

    def test_header_no_subtitle(self, output, capsys):
        output.header("Title")
        out = capsys.readouterr().out
        assert "Title" in out
        assert "=" in out

    def test_header_with_subtitle(self, output, capsys):
        output.header("Title", "Sub")
        out = capsys.readouterr().out
        assert "Title" in out
        assert "Sub" in out

    def test_table_plain(self, output, capsys):
        output.table(["Name", "Age"], [["Alice", "30"], ["Bob", "25"]])
        out = capsys.readouterr().out
        assert "Alice" in out
        assert "Bob" in out
        assert "Name" in out

    def test_table_with_title(self, output, capsys):
        output.table(["Col"], [["val"]], title="My Table")
        out = capsys.readouterr().out
        assert "My Table" in out

    def test_code_plain(self, output, capsys):
        output.code("x = 1", "python")
        out = capsys.readouterr().out
        assert "x = 1" in out
        assert "python" in out

    def test_markdown_plain(self, output, capsys):
        output.markdown("# Hello")
        assert "# Hello" in capsys.readouterr().out

    def test_tree_plain(self, output, capsys):
        output.tree("Root", {"child1": "val1", "child2": {"nested": "val2"}})
        out = capsys.readouterr().out
        assert "Root" in out
        assert "child1" in out

    def test_spinner_plain(self, output, capsys):
        with output.spinner("Working..."):
            pass
        out = capsys.readouterr().out
        assert "Working..." in out

    def test_progress_plain(self, output, capsys):
        with output.progress("Loading", total=10) as update:
            update(5)
        out = capsys.readouterr().out
        assert "Loading" in out

    def test_workflow_progress_plain(self, output, capsys):
        steps = [
            StepProgress("plan", "Plan", "completed", duration_ms=100),
            StepProgress("build", "Build", "running", message="in progress"),
            StepProgress("test", "Test", "pending"),
            StepProgress("review", "Review", "failed"),
        ]
        output.workflow_progress("My Workflow", steps)
        out = capsys.readouterr().out
        assert "My Workflow" in out
        assert "[x]" in out  # completed
        assert "[~]" in out  # running
        assert "[ ]" in out  # pending
        assert "[!]" in out  # failed

    def test_agent_status_plain(self, output, capsys):
        output.agent_status("builder", "running", "writing code")
        out = capsys.readouterr().out
        assert "builder" in out
        assert "running" in out
        assert "writing code" in out

    def test_agent_status_unknown_role(self, output, capsys):
        output.agent_status("custom_role", "starting")
        out = capsys.readouterr().out
        assert "custom_role" in out

    def test_divider_plain(self, output, capsys):
        output.divider()
        assert "-" in capsys.readouterr().out

    def test_newline(self, output, capsys):
        output.newline()
        assert "\n" in capsys.readouterr().out


class TestRichOutputRich:
    """Tests for RichOutput with Rich available (mocked)."""

    @pytest.fixture
    def output(self):
        """Create RichOutput with Rich enabled."""
        out = RichOutput(force_plain=False)
        if not out.use_rich:
            pytest.skip("Rich not installed")
        return out

    def test_rich_enabled(self, output):
        assert output.use_rich
        assert output.console is not None

    def test_print_with_style(self, output):
        # Should not raise
        output.print("styled", OutputStyle.SUCCESS)

    def test_header_rich(self, output):
        output.header("Title", "Subtitle")

    def test_table_rich(self, output):
        output.table(["A", "B"], [["1", "2"]], title="T")

    def test_code_rich(self, output):
        output.code("x = 1")

    def test_markdown_rich(self, output):
        output.markdown("**bold**")

    def test_tree_rich(self, output):
        output.tree("Root", {"a": "1", "b": {"c": "2"}})

    def test_workflow_progress_rich(self, output):
        steps = [
            StepProgress("s1", "Step 1", "completed", duration_ms=500),
            StepProgress("s2", "Step 2", "running", message="busy"),
            StepProgress("s3", "Step 3", "pending"),
        ]
        output.workflow_progress("WF", steps)

    def test_agent_status_rich(self, output):
        output.agent_status("planner", "completed", "done")

    def test_divider_rich(self, output):
        output.divider()


class TestGetOutput:
    """Tests for get_output and convenience functions."""

    def test_get_output_singleton(self):
        mod = rich_output_mod

        mod._output = None
        out1 = get_output(force_plain=True)
        out2 = get_output()
        assert out1 is out2
        mod._output = None  # cleanup

    def test_get_output_force_plain_resets(self):
        mod = rich_output_mod

        mod._output = None
        get_output()
        out = get_output(force_plain=True)
        assert not out.use_rich
        mod._output = None

    def test_convenience_functions(self, capsys):
        mod = rich_output_mod

        mod._output = None
        mod._output = RichOutput(force_plain=True)

        print_success("ok")
        print_error("err")
        print_warning("warn")
        print_info("info")
        print_header("h")
        print_table(["A"], [["1"]])

        out = capsys.readouterr().out
        assert "ok" in out
        assert "err" in out
        mod._output = None
