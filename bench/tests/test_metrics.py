"""The guards in `metrics` — each one was previously described only in a comment.

Every test here names the failure it prevents. That is the point: before this
file, the reasoning behind these behaviours lived in prose and nothing would
notice if a refactor undid one.
"""

from falkorbench import metrics
from falkorbench.metrics import Row


def csv_file(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body)
    return str(path)


class TestParsing:
    def test_empty_cell_is_none_not_zero(self, tmp_path):
        """An unavailable metric must not read as 0.

        Zero is a legitimate measurement. Conflating them made a row with no
        instruction counter look like an infinite regression against a baseline
        that had one.
        """
        path = csv_file(tmp_path, "a.csv", "query,instr,ms\nq,,1.5\n")
        rows = metrics.read_rows(path)
        assert rows["q"]["instr"] is None
        assert rows["q"]["ms"] == 1.5

    def test_zero_is_preserved_as_a_measurement(self, tmp_path):
        path = csv_file(tmp_path, "a.csv", "query,instr,ms\nq,0,1.5\n")
        assert metrics.read_rows(path)["q"]["instr"] == 0.0

    def test_garbage_is_none_rather_than_an_exception(self, tmp_path):
        """A malformed cell must not take the whole report down."""
        path = csv_file(tmp_path, "a.csv", "query,instr,ms\nq,not-a-number,1.5\n")
        assert metrics.read_rows(path)["q"]["instr"] is None

    def test_rows_without_a_query_name_are_dropped(self, tmp_path):
        path = csv_file(tmp_path, "a.csv", "query,instr\n,123\nq,456\n")
        assert list(metrics.read_rows(path)) == ["q"]

    def test_glob_merges_shards(self, tmp_path):
        """The callgrind subset arrives as cg-<side>-1..N.csv from parallel jobs."""
        csv_file(tmp_path, "cg-pr-1.csv", "query,instr\na,1\n")
        csv_file(tmp_path, "cg-pr-2.csv", "query,instr\nb,2\n")
        rows = metrics.read_rows(str(tmp_path / "cg-pr-*.csv"))
        assert sorted(rows) == ["a", "b"]

    def test_a_failed_shard_costs_only_its_own_rows(self, tmp_path):
        csv_file(tmp_path, "cg-pr-1.csv", "query,instr\na,1\n")
        rows = metrics.read_rows(str(tmp_path / "cg-pr-*.csv"))
        assert list(rows) == ["a"]

    def test_missing_pattern_is_empty_not_an_error(self):
        assert metrics.read_rows(None) == {}
        assert metrics.read_rows("/nonexistent/*.csv") == {}


class TestRatio:
    def test_basic(self):
        assert metrics.ratio(Row(instr=100.0), Row(instr=150.0), "instr") == 1.5

    def test_non_positive_baseline_is_uncomparable(self):
        """A zero or negative baseline makes the ratio meaningless and
        sign-flipped, so the pair is left uncompared rather than reported as a
        bogus improvement."""
        assert metrics.ratio(Row(instr=0.0), Row(instr=150.0), "instr") is None
        assert metrics.ratio(Row(instr=-5.0), Row(instr=150.0), "instr") is None

    def test_absent_either_side_is_uncomparable(self):
        assert metrics.ratio(Row(instr=None), Row(instr=1.0), "instr") is None
        assert metrics.ratio(Row(instr=1.0), Row(instr=None), "instr") is None

    def test_missing_key_is_uncomparable(self):
        assert metrics.ratio(Row(), Row(), "instr") is None


class TestHasData:
    def test_false_when_every_row_lacks_the_metric(self):
        """Gating a metric no row carries would silently gate on nothing — which
        is the state of `instr` on any host without a PMU."""
        rows = [Row(instr=None), Row(instr=None)]
        assert not metrics.has_data(rows, "instr")

    def test_true_when_any_row_has_it(self):
        assert metrics.has_data([Row(instr=None), Row(instr=1.0)], "instr")


class TestGeomean:
    def test_is_geometric_not_arithmetic(self):
        # 0.5 and 2.0 cancel: the right average of a ratio set is 1.0, not 1.25.
        assert metrics.geomean([0.5, 2.0]) == 1.0

    def test_empty_is_none(self):
        assert metrics.geomean([]) is None

    def test_non_positive_values_are_excluded(self):
        assert metrics.geomean([0.0, 1.0]) == 1.0


class TestNormaliseMs:
    def test_offset_comes_from_the_control_row(self):
        """The control query does almost no query work, so its ratio *is* the
        per-host speed difference."""
        pr = {metrics.CONTROL_QUERY: Row(ms=1.5)}
        base = {metrics.CONTROL_QUERY: Row(ms=1.0)}
        assert metrics.normalise_ms(pr, base) == 1.5

    def test_none_without_a_control_row_on_either_side(self):
        """Callers must then not report `ms` at all, rather than report an
        uncorrected cross-host ratio — which on identical engines read 1.464."""
        assert metrics.normalise_ms({"q": Row(ms=1.0)}, {"q": Row(ms=1.0)}) is None
        assert metrics.normalise_ms({metrics.CONTROL_QUERY: Row(ms=1.0)}, {}) is None

    def test_none_when_the_control_row_has_no_timing(self):
        pr = {metrics.CONTROL_QUERY: Row(ms=None)}
        base = {metrics.CONTROL_QUERY: Row(ms=1.0)}
        assert metrics.normalise_ms(pr, base) is None

    def test_none_on_a_non_positive_control(self):
        pr = {metrics.CONTROL_QUERY: Row(ms=1.0)}
        base = {metrics.CONTROL_QUERY: Row(ms=0.0)}
        assert metrics.normalise_ms(pr, base) is None
