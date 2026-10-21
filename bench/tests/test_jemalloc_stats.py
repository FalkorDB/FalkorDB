"""Guards on the jemalloc MALLOC-STATS parse behind `alloc_bytes`.

Every row below is verbatim from a real `MEMORY MALLOC-STATS` reply captured
mid-benchmark, including the fused row that made the metric lie. The parse used
to split these lines on whitespace, which silently fails once a `(#/sec)` rate
reaches 8 digits and fills its fixed-width field: the rate runs into the
`nmalloc` before it, and `nmalloc` reads as a 16-digit number. The parsed
cumulative total jumped ~7e16 bytes when a hot size class crossed 10M
allocs/sec and fell back when it cooled, so `measure`'s per-query *deltas* came
out in the petabytes -- or negative -- and the compare gate read that as a
30,000x memory regression on whatever queries straddled the transition.
"""

from falkorbench.client import _jemalloc_row
from falkorbench.client import _jemalloc_table_columns

BINS_HEADER = (
    "bins:           size ind    allocated      nmalloc (#/sec)      "
    "ndalloc (#/sec)    nrequests   (#/sec)  nshards      curregs     c"
)
# Rates of 6 digits: whitespace-separated, so both parses agree.
BINS_ROW_SEPARATED = (
    "                   8   0       188344      1379247  689623      "
    "1355704  677852     64650963  32325481        1        23543      "
)
# The pathological one: rates are 10,974,939 and 10,973,952 -- 8 digits, which
# exactly fills the `(#/sec)` field, leaving no space before it.
BINS_ROW_FUSED = (
    "                  32   3        63200     2194987910974939     "
    "2194790410973952     69572003  34786001        1         1975      "
)
LARGE_HEADER = (
    "large:          size ind    allocated      nmalloc (#/sec)      "
    "ndalloc (#/sec)    nrequests (#/sec)  curlextents"
)
LARGE_ROW = (
    "               49152  45       933888        24209   12104        "
    "24190   12095        24209     12104        1           19      "
)

WANTED = ("size", "nmalloc", "ndalloc")


def test_separated_row_reads_its_counters():
    cols = _jemalloc_table_columns(BINS_HEADER)
    assert _jemalloc_row(BINS_ROW_SEPARATED, cols, WANTED) == {
        "size": 8,
        "nmalloc": 1_379_247,
        "ndalloc": 1_355_704,
    }


def test_fused_rate_column_does_not_corrupt_nmalloc():
    """The regression this file exists for.

    `BINS_ROW_FUSED.split()` yields `2194987910974939` in the nmalloc position;
    the real value is 21,949,879 with a 10,974,939/sec rate glued to it.
    """
    cols = _jemalloc_table_columns(BINS_HEADER)
    row = _jemalloc_row(BINS_ROW_FUSED, cols, WANTED)
    assert row == {"size": 32, "nmalloc": 21_949_879, "ndalloc": 21_947_904}
    # And the contribution stays a believable number of bytes, not 7e16.
    assert row["size"] * row["nmalloc"] < 10**10
    # Guard the premise, so this test still means something if jemalloc's
    # formatting changes: whitespace splitting really does fuse the fields.
    assert BINS_ROW_FUSED.split()[3] == "2194987910974939"


def test_ndalloc_never_exceeds_nmalloc_on_a_fused_row():
    """A size class cannot be freed more times than it was allocated. The old
    parse broke this (it read `nrequests` as ndalloc at one point), and it is
    the cheapest invariant that catches a column mix-up."""
    cols = _jemalloc_table_columns(BINS_HEADER)
    for line in (BINS_ROW_SEPARATED, BINS_ROW_FUSED):
        row = _jemalloc_row(line, cols, WANTED)
        assert row is not None
        assert row["ndalloc"] <= row["nmalloc"]


def test_large_table_has_its_own_column_layout():
    """`large:` drops `nshards`/`curregs` and ends in `curlextents`, so its
    columns must come from its own header rather than the `bins:` one."""
    cols = _jemalloc_table_columns(LARGE_HEADER)
    assert _jemalloc_row(LARGE_ROW, cols, WANTED) == {
        "size": 49_152,
        "nmalloc": 24_209,
        "ndalloc": 24_190,
    }


def test_non_data_lines_are_rejected():
    cols = _jemalloc_table_columns(BINS_HEADER)
    for line in (
        "total:                        7191648           23524     23524  ",
        "                     ---",
        "",
        "active:                       9830400",
    ):
        assert _jemalloc_row(line, cols, WANTED) is None


def test_missing_counter_column_yields_no_row():
    """An `extents:`-shaped header has no nmalloc/ndalloc; asking for them must
    fail closed rather than pick up whatever sits at that offset."""
    extents = (
        "extents:        size ind       ndirty        dirty       nmuzzy    "
        "    muzzy    nretained     retained       ntotal"
    )
    cols = _jemalloc_table_columns(extents)
    assert "nmalloc" not in cols
    assert _jemalloc_row("               8   0            1         4096", cols, WANTED) is None
