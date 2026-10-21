"""The CSV layout is a wire format, so it is asserted rather than assumed.

CI artifacts, local baselines and the reporter's globs all read these files, so
the column order and the empty-cell-means-absent convention cannot drift.

Each property is stated directly. An earlier version of this file pasted in a
captured CSV from the pre-refactor harness and round-tripped it, which sounded
like stronger evidence than it was: those bytes only exercised the same code
paths the one-row cases below do, and the header assertion merely checked that
the paste matched `CSV_FIELDS`.
"""

from falkorbench import metrics
from falkorbench.metrics import Row
from falkorbench.model import CSV_FIELDS

# Spelled out, not derived. Deriving it from the code under test would assert
# nothing; this fails if a column is renamed, reordered, added or dropped — the
# change that silently breaks a stored baseline, or an artifact produced by the
# other side of a comparison.
WIRE_FORMAT = (
    "query",
    "instr",
    "cycles",
    "branches",
    "br_miss",
    "l1d_miss",
    "alloc_bytes",
    "dealloc_bytes",
    "ms",
)


def test_column_order_is_the_wire_format():
    assert CSV_FIELDS == WIRE_FORMAT


def test_a_full_row_round_trips_byte_for_byte(tmp_path):
    """Reading then writing must reproduce the input exactly.

    One row carrying every shape that appears in practice: a full-precision float
    repr (these are raw measurements, so no rounding may creep in), a zero, and
    the empty cells a host with no PMU or a libc-malloc redis produces.
    """
    src = (
        "query,instr,cycles,branches,br_miss,l1d_miss,alloc_bytes,dealloc_bytes,ms\n"
        "RETURN 1,341338.41717068205,0.0,,,,65536.0,32768.0,0.34755706787109375\n"
    )
    path = tmp_path / "measure.csv"
    path.write_text(src)

    out = tmp_path / "again.csv"
    metrics.write_rows(str(out), metrics.read_rows(str(path)).items(), CSV_FIELDS)
    assert out.read_text() == src


def test_the_callgrind_form_keeps_its_integer_spelling(tmp_path):
    """The callgrind jobs write only query,instr, as whole numbers. Emitting
    90512.0 instead of 90512 would change an artifact the reporter parses and a
    human diffs, even though float() reads both.
    """
    src = "query,instr\nRETURN 1,90512\narithmetic,1465420\n"
    path = tmp_path / "cg.csv"
    path.write_text(src)

    rows = metrics.read_rows(str(path))
    # The same rounding the callgrind command applies before writing.
    rounded = [(q, Row(instr=round(r["instr"]))) for q, r in rows.items()]
    out = tmp_path / "cg-again.csv"
    metrics.write_rows(str(out), rounded, ("query", "instr"))
    assert out.read_text() == src


def test_absent_is_written_empty_not_zero(tmp_path):
    """Writing 0 for an absent metric makes it read back as a real measurement,
    and every comparison against it becomes an infinite change."""
    out = tmp_path / "a.csv"
    metrics.write_rows(str(out), [("q", Row(instr=None, ms=1.0))], ("query", "instr", "ms"))
    assert out.read_text().splitlines()[1] == "q,,1.0"


def test_zero_is_written_and_read_as_zero(tmp_path):
    """The other half of the same contract: zero is a legitimate measurement and
    must stay distinguishable from absent."""
    out = tmp_path / "a.csv"
    metrics.write_rows(str(out), [("q", Row(instr=0.0))], ("query", "instr"))
    assert out.read_text().splitlines()[1] == "q,0.0"
    assert metrics.read_rows(str(out))["q"]["instr"] == 0.0


def test_a_column_missing_from_the_file_reads_as_absent(tmp_path):
    """A CSV written by a harness lacking a column must not raise; that metric is
    simply unavailable."""
    out = tmp_path / "a.csv"
    out.write_text("query,instr\nq,5\n")
    row = metrics.read_rows(str(out))["q"]
    assert row["instr"] == 5.0
    assert row.get("alloc_bytes") is None
