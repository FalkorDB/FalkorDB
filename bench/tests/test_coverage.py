"""Parsing of `llvm-cov report`, which used to be two awk one-liners.

The awk indexed `$8`/`$9` with nothing recording what those columns were, and it
matched any line containing `graph/src` — including llvm-cov's wrapped-path rows
and the TOTAL row, which is why this is worth a test rather than a comment.
"""

from falkorbench import coverage

# Real `llvm-cov report` shape: the column order is
# Filename | Regions Missed Cover | Functions Missed Executed | Lines Missed Cover | ...
REPORT = """\
Filename                      Regions  Missed Regions  Cover  Functions  Missed Functions  Executed  Lines  Missed Lines  Cover  Branches  Missed Branches  Cover
graph/src/runtime/runtime.rs      500              50  90.00%        100                10    90.00%   1000           250  75.00%       200               40  80.00%
graph/src/planner.rs              200              20  90.00%         40                 4    90.00%    400            40  90.00%        80               16  80.00%
graph/src/tiny.rs                   5               0 100.00%          2                 0   100.00%     10             0 100.00%         2                0 100.00%
other/src/elsewhere.rs            100              10  90.00%         20                 2    90.00%    100            50  50.00%        20                4  80.00%
TOTAL                             805              80  90.06%        162                16    90.12%   1510           340  77.48%       302               60  80.13%
"""


class TestParseReport:
    def test_only_the_requested_prefix_is_counted(self):
        cov = coverage.parse_report(REPORT)
        assert [f.path for f in cov.files] == [
            "graph/src/runtime/runtime.rs",
            "graph/src/planner.rs",
            "graph/src/tiny.rs",
        ]

    def test_totals_come_from_the_lines_columns_not_regions(self):
        """The distinction the awk got right by accident and documented nowhere:
        columns 8/9 are lines, not regions or functions."""
        cov = coverage.parse_report(REPORT)
        assert cov.lines == 1000 + 400 + 10
        assert cov.missed == 250 + 40 + 0
        assert cov.covered == 1410 - 290

    def test_percentage(self):
        cov = coverage.parse_report(REPORT)
        assert round(cov.percent, 1) == round(100 * 1120 / 1410, 1)

    def test_total_row_is_excluded(self):
        """It contains no `graph/src`, but if the prefix ever widened, double
        counting it would silently inflate every future number."""
        assert all(f.path != "TOTAL" for f in coverage.parse_report(REPORT).files)

    def test_wrapped_path_rows_are_skipped(self):
        """llvm-cov puts a very long filename on its own line, with the numbers on
        the next. The bare-path line has too few columns to be data."""
        wrapped = REPORT + "graph/src/a/very/long/path/that/llvm/wrapped.rs\n"
        cov = coverage.parse_report(wrapped)
        assert len(cov.files) == 3

    def test_non_numeric_columns_do_not_raise(self):
        assert coverage.parse_report("graph/src/x.rs  a b c d e f g h i\n").files == []

    def test_empty_report_is_zero_not_a_crash(self):
        cov = coverage.parse_report("")
        assert cov.files == []
        assert cov.lines == 0
        assert cov.percent == 0.0


class TestLeastCovered:
    def test_small_files_are_excluded_and_order_is_worst_first(self):
        cov = coverage.parse_report(REPORT)
        worst = cov.least_covered(min_lines=200)
        # tiny.rs (10 lines) is below the floor; runtime.rs (75%) is worse than
        # planner.rs (90%).
        assert [f.path for f in worst] == [
            "graph/src/runtime/runtime.rs",
            "graph/src/planner.rs",
        ]

    def test_limit_is_respected(self):
        assert len(coverage.parse_report(REPORT).least_covered(min_lines=1, limit=1)) == 1


class TestBuildFlags:
    def test_instrumentation_is_always_requested(self):
        flags, _ = coverage.build_flags()
        assert "-C instrument-coverage" in flags

    def test_linux_adds_the_duplicate_symbol_workaround_and_a_cxx(self):
        """The embedded RediSearch static libs fail to link without it; macOS's
        linker neither takes the flag nor needs it."""
        import sys

        flags, env = coverage.build_flags()
        if sys.platform == "darwin":
            assert "allow-multiple-definition" not in flags
            assert "CXX" not in env
        else:
            assert "allow-multiple-definition" in flags
            assert env["CXX"]


class TestLlvmToolDiscovery:
    def test_the_toolchain_copy_wins_over_a_system_one(self, tmp_path, monkeypatch):
        """LLVM's profile format is coupled to the rustc that wrote the .profraw,
        so a system LLVM of a different major version fails with "unsupported
        instrumentation profile format version". PATH must be the fallback, never
        the preference — on a dev machine `which` finds a real llvm-profdata that
        cannot read these profiles.
        """
        toolchain = tmp_path / ".rustup/toolchains/stable/lib/rustlib/bin"
        toolchain.mkdir(parents=True)
        wanted = toolchain / "llvm-profdata"
        wanted.write_text("#!/bin/sh\n")
        wanted.chmod(0o755)

        system = tmp_path / "usr-local-bin"
        system.mkdir()
        decoy = system / "llvm-profdata"
        decoy.write_text("#!/bin/sh\n")
        decoy.chmod(0o755)

        monkeypatch.setattr(coverage.Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setenv("PATH", str(system))
        assert coverage._llvm_tool("llvm-profdata") == wanted

    def test_path_is_used_when_the_toolchain_has_none(self, tmp_path, monkeypatch):
        (tmp_path / ".rustup/toolchains").mkdir(parents=True)
        system = tmp_path / "bin"
        system.mkdir()
        tool = system / "llvm-cov"
        tool.write_text("#!/bin/sh\n")
        tool.chmod(0o755)

        monkeypatch.setattr(coverage.Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setenv("PATH", str(system))
        assert coverage._llvm_tool("llvm-cov") == tool

    def test_missing_everywhere_names_the_remedy(self, tmp_path, monkeypatch):
        monkeypatch.setattr(coverage.Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        try:
            coverage._llvm_tool("llvm-profdata")
        except RuntimeError as e:
            assert "llvm-tools-preview" in str(e)
        else:
            raise AssertionError("expected RuntimeError")
