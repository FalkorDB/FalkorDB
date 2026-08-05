"""The CSV format is a wire format, so it gets a golden test.

CI artifacts, local baselines and `report`'s globs all read these files, and the
pre-refactor harness is what produced the ones already in flight. The header,
the column order, and — critically — the empty-cell-means-absent convention must
not drift. `fixtures/legacy_measure.csv` is real output from the pre-refactor
harness; if a change makes it stop round-tripping, that change is a wire-format
break rather than a cleanup.
"""

from pathlib import Path

from falkorbench import metrics
from falkorbench.metrics import Row
from falkorbench.model import CSV_FIELDS

FIXTURES = Path(__file__).parent / "fixtures"


def test_header_matches_the_legacy_layout():
    header = (FIXTURES / "legacy_measure.csv").read_text().splitlines()[0]
    assert header == ",".join(CSV_FIELDS)


def test_legacy_output_round_trips_unchanged(tmp_path):
    """Read a real pre-refactor CSV, write it back, and get the same bytes.

    This is the whole invariant in one assertion: same columns, same order, same
    empty cells, same numeric formatting.
    """
    src = FIXTURES / "legacy_measure.csv"
    rows = metrics.read_rows(str(src))
    out = tmp_path / "again.csv"
    metrics.write_rows(str(out), rows.items(), CSV_FIELDS)
    assert out.read_text() == src.read_text()


def test_absent_metrics_are_written_as_empty_not_zero(tmp_path):
    """Writing 0 for an absent metric would make it read as a real measurement
    and flag every comparison against it as an infinite change."""
    out = tmp_path / "a.csv"
    metrics.write_rows(str(out), [("q", Row(instr=None, ms=1.0))], ("query", "instr", "ms"))
    assert out.read_text().splitlines()[1] == "q,,1.0"


def test_a_populated_metric_survives_the_round_trip(tmp_path):
    out = tmp_path / "a.csv"
    metrics.write_rows(str(out), [("q", Row(instr=123.5))], ("query", "instr"))
    assert metrics.read_rows(str(out))["q"]["instr"] == 123.5


def test_zero_survives_as_zero(tmp_path):
    out = tmp_path / "a.csv"
    metrics.write_rows(str(out), [("q", Row(instr=0.0))], ("query", "instr"))
    assert metrics.read_rows(str(out))["q"]["instr"] == 0.0


def test_callgrind_csv_is_the_two_column_form():
    """The callgrind jobs write only query,instr — `report` merges them by glob,
    so a widened schema there would be a silent format change too."""
    rows = metrics.read_rows(str(FIXTURES / "legacy_callgrind.csv"))
    assert rows
    assert all(r.get("instr") is not None for r in rows.values())
