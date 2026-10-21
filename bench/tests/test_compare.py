"""The local gate.

Its whole reason for sharing `metrics` with the reporter is that it used to
disagree with it on the same CSVs — in particular it gated raw wall-clock at
1.25x, which across two hosts flags noise. These tests pin the shared behaviour
so the two cannot drift apart again.
"""

from falkorbench import compare as compare_mod
from falkorbench.metrics import CONTROL_QUERY
from falkorbench.metrics import Row


def test_clean_comparison_has_no_regressions():
    rows = {"q": Row(instr=100.0)}
    result = compare_mod.compare(rows, {"q": Row(instr=100.0)})
    assert result.regressions == []
    assert "no regressions" in "\n".join(compare_mod.render(result))


def test_a_regression_is_flagged_with_its_threshold():
    result = compare_mod.compare({"q": Row(instr=200.0)}, {"q": Row(instr=100.0)})
    assert [r.metric for r in result.regressions] == ["instr"]
    assert result.regressions[0].ratio == 2.0
    assert "REGRESSION" in "\n".join(compare_mod.render(result))


def test_an_improvement_is_not_flagged():
    result = compare_mod.compare({"q": Row(instr=50.0)}, {"q": Row(instr=100.0)})
    assert result.regressions == []


def test_metric_absent_on_both_sides_is_skipped_not_gated():
    """Gating a metric no row carries would gate on nothing — the state of
    `instr` on any host without a PMU."""
    result = compare_mod.compare({"q": Row(instr=None)}, {"q": Row(instr=None)})
    assert "instr" in result.skipped
    assert "instr" not in result.metrics


def test_threshold_override_applies_to_every_metric():
    cur, base = {"q": Row(instr=120.0)}, {"q": Row(instr=100.0)}
    assert compare_mod.compare(cur, base).regressions  # 1.2x > default 1.10
    assert not compare_mod.compare(cur, base, threshold=1.5).regressions


def test_metric_subset_narrows_the_gate():
    cur = {"q": Row(instr=200.0, cycles=200.0)}
    base = {"q": Row(instr=100.0, cycles=100.0)}
    result = compare_mod.compare(cur, base, metrics=["cycles"])
    assert [r.metric for r in result.regressions] == ["cycles"]


def test_wall_clock_is_normalised_by_the_control_row():
    """Both rows are 3x slower, which is the host. The gate must see 1.0x, not
    3.0x — the un-normalised version of this is a noise detector."""
    cur = {CONTROL_QUERY: Row(ms=3.0), "q": Row(ms=9.0)}
    base = {CONTROL_QUERY: Row(ms=1.0), "q": Row(ms=3.0)}
    result = compare_mod.compare(cur, base, metrics=["ms"])
    assert result.ms_offset == 3.0
    assert result.regressions == []
    assert result.ratios["q"]["ms"] == 1.0


def test_wall_clock_is_not_reported_without_a_control_row():
    """An uncorrected cross-host ratio is worse than no ratio."""
    result = compare_mod.compare({"q": Row(ms=9.0)}, {"q": Row(ms=1.0)}, metrics=["ms"])
    assert result.ms_offset is None
    assert result.ratios["q"]["ms"] is None
    assert result.regressions == []


def test_genuinely_slower_wall_clock_still_trips():
    cur = {CONTROL_QUERY: Row(ms=1.0), "q": Row(ms=5.0)}
    base = {CONTROL_QUERY: Row(ms=1.0), "q": Row(ms=1.0)}
    result = compare_mod.compare(cur, base, metrics=["ms"])
    assert [r.metric for r in result.regressions] == ["ms"]


def test_non_positive_baseline_is_not_a_regression():
    """It used to make the ratio meaningless and sign-flipped."""
    result = compare_mod.compare({"q": Row(instr=100.0)}, {"q": Row(instr=0.0)})
    assert result.regressions == []


def test_missing_and_new_rows_are_reported_not_gated():
    result = compare_mod.compare({"new": Row(instr=1.0)}, {"gone": Row(instr=1.0)})
    assert result.missing == ["gone"]
    assert result.added == ["new"]
    rendered = "\n".join(compare_mod.render(result))
    assert "MISSING from current" in rendered
    assert "NEW (not in baseline)" in rendered


def test_render_survives_a_row_with_no_comparable_metrics():
    """Regression guard for a crash: `render` reads the precomputed ratio map, so
    it must not assume every metric is present for every row."""
    cur = {"q": Row(instr=None, cycles=100.0)}
    base = {"q": Row(instr=100.0, cycles=100.0)}
    result = compare_mod.compare(cur, base)
    assert compare_mod.render(result)  # does not raise
