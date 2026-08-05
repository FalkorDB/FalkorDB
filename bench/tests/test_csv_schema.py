"""The CSV format is a wire format, so it gets a golden test.

CI artifacts, local baselines and the reporter's globs all read these files, and
the pre-refactor harness is what produced the ones already in flight. The header,
the column order, and — critically — the empty-cell-means-absent convention must
not drift.

`LEGACY_MEASURE` below is **verbatim output from the pre-refactor
`bench/run_bench.py`**, captured on an M3 Pro (hence populated instr/cycles and
empty allocation columns: Homebrew redis links libc malloc, not jemalloc). It is
inlined rather than kept as a fixture file so the bytes under test sit next to
the assertion about them.
"""

from falkorbench import metrics
from falkorbench.metrics import Row
from falkorbench.model import CSV_FIELDS

LEGACY_MEASURE = (
    "query,instr,cycles,branches,br_miss,l1d_miss,alloc_bytes,dealloc_bytes,ms\n"
    "RETURN 1,341338.41717068205,153989.93832491103,,,,,,0.34755706787109375\n"
    "arithmetic,1759305.5396216155,366965.332761427,,,,,,0.09987616539001465\n"
    "two-hop,16802398.42278204,2531877.3572816886,,,,,,0.6198580265045166\n"
    "agg count,5132906.819137941,827752.041357211,,,,,,0.20900797843933105\n"
    "shortestPath,491121.6225908206,173781.81119624345,,,,,,0.06075310707092285\n"
    "CREATE + DELETE,14502555.943684824,2349116.585262806,,,,,,0.5756368637084961\n"
    "algo.pageRank,17477716.059407204,3621296.187948728,,,,,,0.8758740425109863\n"
    "write 100,22562801.277057037,3575944.67291875,,,,,,0.8788762092590332\n"
)

# The pre-refactor callgrind writer emitted whole integers via "%.0f" — the
# reporter parses this with float(), but the artifact must keep the integer
# spelling so a diff of two runs stays readable.
LEGACY_CALLGRIND = "query,instr\nRETURN 1,90512\narithmetic,1465420\n"


def write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    return str(p)


def test_header_matches_the_legacy_layout():
    assert LEGACY_MEASURE.splitlines()[0] == ",".join(CSV_FIELDS)


def test_legacy_output_round_trips_byte_for_byte(tmp_path):
    """Read real pre-refactor output, write it back, get the same bytes.

    The whole invariant in one assertion: same columns, same order, same empty
    cells, same numeric formatting.
    """
    src = write(tmp_path, "legacy.csv", LEGACY_MEASURE)
    rows = metrics.read_rows(src)
    out = tmp_path / "again.csv"
    metrics.write_rows(str(out), rows.items(), CSV_FIELDS)
    assert out.read_text() == LEGACY_MEASURE


def test_callgrind_output_round_trips_byte_for_byte(tmp_path):
    """Including the integer spelling — writing 90512.0 would be a format change
    to an artifact other jobs consume."""
    src = write(tmp_path, "cg.csv", LEGACY_CALLGRIND)
    rows = metrics.read_rows(src)
    # Same rounding the callgrind command applies before writing.
    rounded = [(q, Row(instr=round(r["instr"]))) for q, r in rows.items()]
    out = tmp_path / "cg-again.csv"
    metrics.write_rows(str(out), rounded, ("query", "instr"))
    assert out.read_text() == LEGACY_CALLGRIND


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
