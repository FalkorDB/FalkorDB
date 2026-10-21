"""The CI comment's guards, over CSV fixtures.

Named after the failures they prevent. Several of these describe real reports
that were posted before the guard existed — a `RETURN 1` row at 4.0M
instructions against a true cost near 150k, a 46% "regression" on a PR that
changed no engine code, a 60x "win" that was an error reply.
"""

import json

import pytest

from falkorbench import report
from falkorbench.metrics import Row


def rows(**by_query: dict[str, float | None]) -> dict[str, Row]:
    return {name: Row(**vals) for name, vals in by_query.items()}


def text_of(rep: report.Report) -> str:
    return rep.text()


class TestMeasureSection:
    def test_missing_pr_side_is_fatal(self):
        """No PR CSV means there is no reading at all — the run must fail, not
        post a comment that looks fine."""
        rep = report.build({"base": rows(q={"alloc_bytes": 1.0})}, {})
        assert rep.fatal
        assert "No PR measurement" in text_of(rep)

    def test_missing_base_side_is_fatal(self):
        rep = report.build({"pr": rows(q={"alloc_bytes": 1.0})}, {})
        assert any("base" in f for f in rep.fatal)

    def test_missing_c_side_costs_only_its_column(self):
        """A C-side failure must not fail the run. `:edge-c` is a moving
        third-party tag, so this happens eventually on unrelated PRs."""
        pr = rows(**{"RETURN 1": {"alloc_bytes": 100000.0, "ms": 1.0}})
        base = rows(**{"RETURN 1": {"alloc_bytes": 100000.0, "ms": 1.0}})
        rep = report.build(
            {"pr": pr, "base": base},
            {"pr": rows(q={"instr": 1.0}), "base": rows(q={"instr": 1.0})},
        )
        assert rep.fatal == []
        assert "vs the C engine" not in text_of(rep)

    def test_ranks_on_instructions_when_available(self):
        pr = rows(q={"instr": 200.0, "alloc_bytes": 1.0})
        base = rows(q={"instr": 100.0, "alloc_bytes": 1.0})
        out = text_of(report.build({"pr": pr, "base": base}, {}))
        assert "Ranked by **instructions**" in out

    def test_falls_back_to_allocated_bytes_never_wall_clock(self):
        """Instructions absent (no PMU) must not silently become a wall-clock
        ranking, which would turn the gate into a noise detector."""
        pr = rows(q={"instr": None, "alloc_bytes": 100000.0, "ms": 5.0})
        base = rows(q={"instr": None, "alloc_bytes": 100000.0, "ms": 1.0})
        out = text_of(report.build({"pr": pr, "base": base}, {}))
        assert "Ranked by **allocated bytes**" in out

    def test_rows_below_the_alloc_floor_are_measured_but_not_ranked(self):
        """A ratio over a ~2 KB denominator is noise dressed as signal."""
        pr = rows(small={"alloc_bytes": 2048.0}, big={"alloc_bytes": 200000.0})
        base = rows(small={"alloc_bytes": 1024.0}, big={"alloc_bytes": 100000.0})
        out = text_of(report.build({"pr": pr, "base": base}, {}))
        assert "below the 64 KB floor" in out
        assert "1 row compared vs base" in out

    def test_no_movers_says_so_explicitly(self):
        pr = rows(q={"alloc_bytes": 100000.0})
        base = rows(q={"alloc_bytes": 100000.0})
        out = text_of(report.build({"pr": pr, "base": base}, {}))
        assert "No row moved more than 5%" in out

    def test_geomean_is_reported(self):
        pr = rows(a={"alloc_bytes": 200000.0}, b={"alloc_bytes": 100000.0})
        base = rows(a={"alloc_bytes": 100000.0}, b={"alloc_bytes": 200000.0})
        out = text_of(report.build({"pr": pr, "base": base}, {}))
        # 2.0 and 0.5 cancel.
        assert "geomean 1.000" in out


class TestWallClock:
    def test_control_row_normalisation_cancels_the_host_offset(self):
        """Both rows are 1.5x slower on the PR host. That is the host, not the
        engine, and must produce no outliers — the un-normalised version of this
        reported a 46% regression on a PR that changed no engine code.
        """
        pr = rows(
            **{"RETURN 1": {"ms": 1.5, "alloc_bytes": 1.0}, "q": {"ms": 3.0, "alloc_bytes": 1.0}}
        )
        base = rows(
            **{"RETURN 1": {"ms": 1.0, "alloc_bytes": 1.0}, "q": {"ms": 2.0, "alloc_bytes": 1.0}}
        )
        out = text_of(report.build({"pr": pr, "base": base}, {}))
        assert "No query moved more than 50%" in out

    def test_a_genuine_outlier_survives_normalisation(self):
        pr = rows(
            **{"RETURN 1": {"ms": 1.0, "alloc_bytes": 1.0}, "q": {"ms": 10.0, "alloc_bytes": 1.0}}
        )
        base = rows(
            **{"RETURN 1": {"ms": 1.0, "alloc_bytes": 1.0}, "q": {"ms": 1.0, "alloc_bytes": 1.0}}
        )
        out = text_of(report.build({"pr": pr, "base": base}, {}))
        assert "Wall-clock outliers" in out
        assert "10.00x" in out

    def test_no_wall_clock_section_without_a_control_row(self):
        """Reporting an uncorrected cross-host ratio is worse than reporting
        nothing."""
        pr = rows(q={"ms": 3.0, "alloc_bytes": 1.0})
        base = rows(q={"ms": 1.0, "alloc_bytes": 1.0})
        out = text_of(report.build({"pr": pr, "base": base}, {}))
        assert "Wall-clock outliers" not in out


class TestCallgrindSection:
    def test_no_movers_message(self):
        pr = rows(a={"instr": 1000.0})
        base = rows(a={"instr": 1000.0})
        out = text_of(report.build({"pr": rows(q={"alloc_bytes": 1.0})}, {"pr": pr, "base": base}))
        assert "No query moved more than 0.5%" in out

    def test_movers_table_lists_only_what_moved(self):
        pr = rows(moved={"instr": 1100.0}, still={"instr": 1000.0})
        base = rows(moved={"instr": 1000.0}, still={"instr": 1000.0})
        out = text_of(report.build({"pr": rows(q={"alloc_bytes": 1.0})}, {"pr": pr, "base": base}))
        assert "Moved more than 0.5% (1 of 2)" in out

    def test_missing_callgrind_output_is_fatal(self):
        rep = report.build(
            {"pr": rows(q={"alloc_bytes": 1.0}), "base": rows(q={"alloc_bytes": 1.0})}, {}
        )
        assert any("callgrind" in f for f in rep.fatal)

    def test_c_column_absent_says_why(self):
        pr = rows(a={"instr": 1000.0})
        base = rows(a={"instr": 1000.0})
        out = text_of(report.build({"pr": rows(q={"alloc_bytes": 1.0})}, {"pr": pr, "base": base}))
        assert "The C engine is not in this table" in out
        assert "331,579,187" in out

    def test_single_c_pass_is_n_a_not_a_number(self):
        """One number with nothing to check it against is exactly where the noise
        hides: an earlier report printed 43 of 93 rows as single-pass, including
        `RETURN 1` at 4.0M instructions against a true cost near 150k."""
        cg = {
            "pr": rows(a={"instr": 1000.0}),
            "base": rows(a={"instr": 1000.0}),
            "c": rows(a={"instr": 900000.0}),
        }
        out = text_of(report.build({"pr": rows(q={"alloc_bytes": 1.0})}, cg))
        assert "n/a" in out

    def test_c_error_replies_are_suppressed(self):
        """The C engine rejects a few things this set exercises; its error path
        costs ~500-2500 instructions against ~149k for a real query. Reporting
        that as a ratio invents a 60x win."""
        cg = {
            "pr": rows(a={"instr": 100000.0}),
            "base": rows(a={"instr": 100000.0}),
            "c": rows(a={"instr": 1200.0}),
            "c2": rows(a={"instr": 1300.0}),
        }
        out = text_of(report.build({"pr": rows(q={"alloc_bytes": 1.0})}, cg))
        assert "n/a" in out

    def test_disagreeing_c_passes_show_a_range(self):
        cg = {
            "pr": rows(a={"instr": 100000.0}),
            "base": rows(a={"instr": 100000.0}),
            "c": rows(a={"instr": 100000.0}),
            "c2": rows(a={"instr": 200000.0}),
        }
        out = text_of(report.build({"pr": rows(q={"alloc_bytes": 1.0})}, cg))
        assert "100,000–200,000" in out

    def test_agreeing_c_passes_show_a_midpoint(self):
        cg = {
            "pr": rows(a={"instr": 100000.0}),
            "base": rows(a={"instr": 100000.0}),
            "c": rows(a={"instr": 100000.0}),
            "c2": rows(a={"instr": 100500.0}),
        }
        out = text_of(report.build({"pr": rows(q={"alloc_bytes": 1.0})}, cg))
        assert "100,250" in out


class TestProvenanceAndCoverage:
    def test_provenance_table_and_metadata_key_handling(self, tmp_path):
        path = tmp_path / "prov.json"
        path.write_text(
            json.dumps(
                {
                    "pr": {"image": "ghcr.io/x:rc-pr-1", "digest": "sha256:abcdef1234567890"},
                    "_head_sha": "0123456789abcdef",
                }
            )
        )
        rep = report.Report()
        report.provenance_section(rep, str(path))
        out = rep.text()
        assert "rc-pr-1" in out
        # `_`-prefixed keys are metadata, not sides, and must not become rows.
        assert "| _head_sha |" not in out
        # The head SHA is shown abbreviated to 9 characters.
        assert "012345678" in out

    def test_missing_files_are_silently_skipped(self):
        rep = report.Report()
        report.provenance_section(rep, "/nonexistent.json")
        report.coverage_section(rep, "/nonexistent.txt")
        assert rep.lines == []


class TestMarker:
    def test_marker_is_always_last(self):
        rep = report.build({"pr": rows(q={"alloc_bytes": 1.0})}, {})
        assert rep.text().rstrip().endswith(report.MARKER)


@pytest.mark.parametrize("side", ["pr", "base"])
def test_each_required_measure_side_is_individually_fatal(side):
    both = {"pr": rows(q={"alloc_bytes": 1.0}), "base": rows(q={"alloc_bytes": 1.0})}
    del both[side]
    assert report.build(both, {}).fatal
