"""Unit tests for soak_check.py's comparison and result logic (no server, no docker).

    python -I -m pytest tests/soak -v
"""
import argparse
import json
import re
from pathlib import Path

import pytest

import soak_check as sc

HERE = Path(__file__).resolve().parent
DIGEST_A = "ghcr.io/falkordb/falkordb-server@sha256:" + "a" * 64
DIGEST_B = "falkordb/falkordb-server@sha256:" + "b" * 64


def run(side, rep=1, completed=True, checks=None, ops=None, setup_s=None, scenario="s12_iot_retention",
        server_ran=True, cut=False, wall_s=400.0):
    return sc.Run(scenario=scenario, side=side, rep=rep, completed=completed,
                  reason="" if completed else "exit 1", checks=checks or {}, ops=ops or {},
                  setup_s=setup_s, server_ran=server_ran, cut=cut, wall_s=wall_s)


def op(p99, tput=10.0, p50=None, ok=1000):
    return {"p50_ms": p99 / 3 if p50 is None else p50, "p99_ms": p99, "throughput": tput, "ok": ok}


def by_name(rows):
    return {r.name: r for r in rows}


def args(**kw):
    base = dict(candidate=DIGEST_A, baseline=DIGEST_B, mode="nightly", check="soak-b",
                candidate_label=None, baseline_label=None, candidate_commit=None,
                baseline_commit=None)
    base.update(kw)
    return argparse.Namespace(**base)


# ------------------------------------------------------------------ 1-held / 0-broken pairs
@pytest.mark.parametrize("c,b,status", [
    (0, 1, "regression"), (0, None, "regression"), (0, 0, "info"),
    (1, 0, "improvement"), (1, 1, "ok"), (1, None, "ok"),
    (None, 1, "info"), (None, 0, "info"),
])
def test_zero_one_status(c, b, status):
    assert sc._zero_one("x", "invariant", c, b).status == status


def test_correctness_check_broken_on_candidate_is_a_regression_at_once():
    rows = by_name(sc.scenario_rows("s12_iot_retention",
                                    [run("candidate", checks={"ingest reading count": False})],
                                    [run("baseline", checks={"ingest reading count": True})], True))
    row = rows["s12 ingest reading count"]
    assert (row.status, row.candidate, row.baseline, row.crossed) == ("regression", 0, 1, False)


def test_correctness_check_broken_on_both_sides_is_info():
    rows = by_name(sc.scenario_rows("s01_social_feed",
                                    [run("candidate", checks={"undirected shortestPath supported": False})],
                                    [run("baseline", checks={"undirected shortestPath supported": False})],
                                    True))
    assert rows["s01 undirected shortestPath supported"].status == "info"


def test_known_flaky_check_is_never_gated_but_named(tmp_path):
    name = "Live (under writes) index == scan during build/after"
    assert f"s10 {name}" in sc.KNOWN_FLAKY
    c = run("candidate", checks={name: False, "y": True}, scenario="s10_schema_under_load")
    c.details = {name: "1 of 40 runs failed; first: [567,667] idx=1242 scan=1243"}
    sched = sc.Schedule(runs={"s10_schema_under_load": {
        "candidate": [c], "baseline": [run("baseline", checks={name: True, "y": True}, scenario="s10_schema_under_load")]}})
    res = sc.finish(args(), tmp_path, 1.7e9, ["s10_schema_under_load"], sched, None, [], sc.MODES["nightly"])
    assert res["verdict"] == "pass" and res["findings"] == []
    assert f"known flaky, not gated: s10 {name} (#3095) broke on candidate" in res["summary"]
    assert ",known," in (tmp_path / "raw" / "rows.csv").read_text()
    if sc.VALIDATOR.exists():
        assert sc.validate(res) == []


def test_every_known_flaky_entry_names_an_issue():
    for name, issue in sc.KNOWN_FLAKY.items():
        assert re.fullmatch(r"#[1-9][0-9]*", issue), (name, issue)
        assert re.fullmatch(r"s\d\d .+", name), name


def test_known_flaky_broken_every_time_on_every_run_is_still_a_regression():
    name = "Live (under writes) index == scan during build/after"
    def cand(rep, k, n):
        r = run("candidate", rep, checks={name: False}, scenario="s10_schema_under_load")
        r.details = {name: f"{k} of {n} runs failed; first: [1,2] idx=1 scan=2"}
        return r
    base = [run("baseline", checks={name: True}), run("baseline", 2, checks={name: True})]
    rows = {r.name: r for r in sc.scenario_rows("s10_schema_under_load", [cand(1, 26, 26), cand(2, 30, 30)], base, True)}
    assert rows[f"s10 {name}"].status == "regression"
    # the old failure-only recording ("3 of 3") is under the 10-check floor: the known flake
    rows = {r.name: r for r in sc.scenario_rows("s10_schema_under_load", [cand(1, 3, 3)], base[:1], True)}
    assert rows[f"s10 {name}"].status == "known"
    # broke every time in one run, only sometimes in the other: the known flake
    rows = {r.name: r for r in sc.scenario_rows("s10_schema_under_load", [cand(1, 26, 26), cand(2, 3, 30)], base, True)}
    assert rows[f"s10 {name}"].status == "known"
    # one run, a few misses of many: the known flake; every time: a regression
    assert {r.name: r for r in sc.scenario_rows("s10_schema_under_load", [cand(1, 1, 40)], base[:1], True)}[
        f"s10 {name}"].status == "known"
    assert {r.name: r for r in sc.scenario_rows("s10_schema_under_load", [cand(1, 40, 40)], base[:1], True)}[
        f"s10 {name}"].status == "regression"
    # broke on both sides: still known, and the summary says both
    b = run("baseline", checks={name: False})
    b.details = {name: "2 of 40 runs failed"}
    row = {r.name: r for r in sc.scenario_rows("s10_schema_under_load", [cand(1, 1, 40)], [b], True)}[f"s10 {name}"]
    assert row.status == "known" and row.note.endswith("broke on candidate and baseline")


def test_correctness_breaks_in_the_rerun_count_too():
    cand = [run("candidate", checks={"x": True}), run("candidate", 2, checks={"x": False})]
    base = [run("baseline", checks={"x": True}), run("baseline", 2, checks={"x": True})]
    assert by_name(sc.scenario_rows("s07_mvcc", cand, base, True))["s07 x"].status == "regression"


# ------------------------------------------------------------------ timing checks
def test_timing_check_broken_once_is_unconfirmed_info_and_asks_for_a_rerun():
    name = "p99 does not drift up: ingest_batch"
    assert sc.is_timing_check(name)
    rows = sc.scenario_rows("s12_iot_retention", [run("candidate", checks={name: False})],
                            [run("baseline", checks={name: True})], True)
    row = [r for r in rows if r.metric == "invariant"][0]
    assert row.status == "info" and row.crossed and row.candidate == 1
    assert row.name == f"s12 {name} (broke in 1 of 1 candidate runs)"


def test_timing_check_broken_on_every_run_is_a_regression():
    name = "RSS plateaus in second half (slope projected over run length <= 10% of mean RSS)"
    cand = [run("candidate", checks={name: False}), run("candidate", 2, checks={name: False})]
    base = [run("baseline", checks={name: True}), run("baseline", 2, checks={name: True})]
    row = by_name(sc.scenario_rows("s12_iot_retention", cand, base, True))[f"s12 {name}"]
    assert (row.status, row.candidate, row.baseline) == ("regression", 0, 1)


def test_timing_check_in_queue_mode_never_fails_and_stays_valid():
    name = "replication lag p99 < 1 s"
    cand = [run("candidate", checks={name: False}), run("candidate", 2, checks={name: False})]
    base = [run("baseline", checks={name: True}), run("baseline", 2, checks={name: True})]
    row = [r for r in sc.scenario_rows("s09_replication", cand, base, False) if r.metric == "invariant"][0]
    # candidate 0 with baseline 1 must be a regression (validator), so queue mode reports 1 + note
    assert (row.status, row.candidate, row.baseline) == ("info", 1, 1)


def test_timing_confirmation_pairs_the_runs():
    """Broken on the candidate twice, but the baseline also broke in the re-run: noise that hit
    both sides, not a confirmed regression."""
    name = "p99 does not drift up: ingest_batch"
    cand = [run("candidate", checks={name: False}), run("candidate", 2, checks={name: False})]
    base = [run("baseline", checks={name: True}), run("baseline", 2, checks={name: False})]
    row = [r for r in sc.scenario_rows("s12_iot_retention", cand, base, True) if r.metric == "invariant"][0]
    assert row.status == "info" and row.candidate == 1


def test_timing_check_broken_once_on_the_baseline_is_not_an_improvement():
    name = "victim open-loop schedule kept (few late/missed sends)"
    row = [r for r in sc.scenario_rows("s11_client_chaos", [run("candidate", checks={name: True})],
                                       [run("baseline", checks={name: False})], True) if r.metric == "invariant"][0]
    assert (row.status, row.candidate, row.baseline) == ("ok", 1, 1)


def test_timing_check_broken_on_both_sides_every_run_is_info():
    name = "steady state: RSS plateau (2nd half growth <10%+50MB)"
    cand = [run("candidate", checks={name: False}), run("candidate", 2, checks={name: False})]
    base = [run("baseline", checks={name: False}), run("baseline", 2, checks={name: False})]
    row = [r for r in sc.scenario_rows("s05_multitenant_churn", cand, base, True) if r.metric == "invariant"][0]
    assert (row.status, row.candidate, row.baseline) == ("info", 0, 0)


@pytest.mark.parametrize("name", [
    "ingest reading count", "deleted ids are reused (max(id) stays bounded)",
    "runaway query with TIMEOUT 500 ms times out", "server alive at end", "recall@10 late >= 0.90",
    "A1-strict-kill9: recovery (kill->first query) < 180s", "replica initial full sync completes (<=900s)",
    "rss_slope written", "slots slope slow progress address stress",
])
def test_correctness_names_are_not_timing(name):
    assert not sc.is_timing_check(name)


@pytest.mark.parametrize("name", [
    "no missed schedule slots", "SLO pay p99<=100ms", "steady state: used_memory plateau (2nd half growth <10%+20MB)",
    "RSS plateaus in second half (slope projected over run length <= 10% of mean RSS)",
    "no memory growth between chaos rounds (RSS)", "replication lag p99 < 1 s",
    "victim open-loop schedule kept (few late/missed sends)", "segments completed within time budget",
])
def test_timing_names(name):
    assert sc.is_timing_check(name)


# ------------------------------------------------------------------ completed
def test_crash_on_the_candidate_is_a_completed_regression():
    rows = by_name(sc.scenario_rows("s12_iot_retention", [run("candidate", completed=False)],
                                    [run("baseline")], True))
    assert rows["s12 completed"].status == "regression"
    assert not [r for r in rows.values() if r.metric in ("p99_ms", "throughput")]


def test_crash_on_both_sides_is_info():
    rows = by_name(sc.scenario_rows("s12_iot_retention", [run("candidate", completed=False)],
                                    [run("baseline", completed=False)], True))
    assert rows["s12 completed"].status == "info"


# ------------------------------------------------------------------ perf
def test_p50_crossing_needs_the_rerun_to_repeat_it():
    cand1 = run("candidate", ops={"ingest_batch": op(60.0, p50=30.0)})
    base1 = run("baseline", ops={"ingest_batch": op(9.0, p50=3.0)})
    # first pass only: info, asks for a re-run
    rows = {(r.name, r.metric): r for r in sc.scenario_rows("s12_iot_retention", [cand1], [base1], True)}
    row = rows[("s12 ingest_batch (not re-run)", "p50_ms")]
    assert (row.status, row.crossed) == ("info", True)
    # repeated (milder): regression with the milder ratio
    cand2 = run("candidate", 2, ops={"ingest_batch": op(40.0, p50=15.0)})
    base2 = run("baseline", 2, ops={"ingest_batch": op(9.0, p50=3.0)})
    rows = [r for r in sc.scenario_rows("s12_iot_retention", [cand1, cand2], [base1, base2], True)
            if r.metric == "p50_ms"]
    assert rows[0].status == "regression" and rows[0].name == "s12 ingest_batch"
    assert rows[0].ratio == pytest.approx(5.0)
    # repeated, but under the 2.0x ratio (the worst A/A p50 swing was 1.81x): not repeated
    cand2.ops["ingest_batch"] = op(40.0, p50=5.4)
    rows = [r for r in sc.scenario_rows("s12_iot_retention", [cand1, cand2], [base1, base2], True)
            if r.metric == "p50_ms"]
    assert rows[0].status == "info" and rows[0].name == "s12 ingest_batch (not repeated on re-run)"


def test_p99_is_reported_never_gated():
    """A/A: a stall episode moved p99 56x; p99 is info, never a crossing or a re-run."""
    cand = run("candidate", ops={"q": op(550.0, p50=1.5, ok=5000)})
    base = run("baseline", ops={"q": op(9.9, p50=1.48, ok=5000)})
    rows = {r.metric: r for r in sc.scenario_rows("s07_mvcc_consistency", [cand], [base], True)}
    assert rows["p99_ms"].status == "info" and not rows["p99_ms"].crossed
    assert rows["p99_ms"].name == "s07 q (p99 shift, not gated)" and rows["p50_ms"].status == "ok"
    assert "1 p99 shift(s) reported, not gated" in sc.summary_line(list(rows.values()), ["s07"], [], "nightly", 300, None)
    # few samples: no p99 row at all
    cand.ops["q"]["ok"] = base.ops["q"]["ok"] = 900
    rows = {r.metric for r in sc.scenario_rows("s07_mvcc_consistency", [cand], [base], True)}
    assert "p99_ms" not in rows


def test_p50_below_the_absolute_floor_is_ok():
    rows = sc.scenario_rows("s04_recommendation", [run("candidate", ops={"q": op(3.0, p50=0.9)})],
                            [run("baseline", ops={"q": op(3.0, p50=0.3)})], True)
    row = [r for r in rows if r.metric == "p50_ms"][0]
    assert row.ratio == pytest.approx(3.0) and row.status == "ok" and not row.crossed


def test_throughput_drop_and_setup_time():
    cand = run("candidate", ops={"w": op(1.0, tput=5.0)}, setup_s=100.0)
    base = run("baseline", ops={"w": op(1.0, tput=10.0)}, setup_s=20.0)
    rows = {(r.name, r.metric): r for r in sc.scenario_rows("s12_iot_retention", [cand], [base], True)}
    assert rows[("s12 w (not re-run)", "throughput")].crossed
    setup = rows[("s12 setup wall time (not re-run)", "p50_ms")]
    assert (setup.candidate, setup.baseline, setup.crossed) == (100000.0, 20000.0, True)


def test_queue_mode_reports_perf_as_info():
    cand, base = run("candidate", ops={"w": op(50.0)}), run("baseline", ops={"w": op(5.0)})
    rows = [r for r in sc.scenario_rows("s12_iot_retention", [cand], [base], False) if r.metric == "p50_ms"]
    assert rows[0].status == "info" and rows[0].crossed


def test_op_the_candidate_nearly_stopped_doing_still_compares():
    base = run("baseline", ops={"w": op(5.0, tput=10.0)})
    for cand_ops in ({}, {"w": {"p99_ms": 900.0, "throughput": 0.2, "ok": 12}}):
        rows = {(r.name, r.metric): r for r in sc.scenario_rows(
            "s12_iot_retention", [run("candidate", ops=cand_ops)], [base], True)}
        tput = rows[("s12 w (not re-run)", "throughput")]
        assert tput.crossed and tput.ratio < sc.TPUT_RATIO
        assert ("s12 w", "p99_ms") not in rows   # too few candidate samples for a p99


def test_improvement():
    rows = sc.scenario_rows("s12_iot_retention", [run("candidate", ops={"w": op(3.0)})],
                            [run("baseline", ops={"w": op(30.0)})], True)
    assert [r for r in rows if r.metric == "p50_ms"][0].status == "improvement"


# ------------------------------------------------------------------ per-run parsing
def test_read_run(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"ops": {
        "ingest_batch": {"count": 500, "errors": 0, "p99_ms": 4.0, "throughput_ops_s": 2.0},
        "_late_start": {"count": 900, "errors": 0, "p99_ms": 60.0, "throughput_ops_s": 3.0},
        "rare": {"count": 20, "errors": 0, "p99_ms": 9.0, "throughput_ops_s": 0.1}}}))
    (tmp_path / "checks.json").write_text(json.dumps([
        {"name": "a", "ok": True}, {"name": "a", "ok": False}, {"name": "b", "ok": True}]))
    (tmp_path / "phases.json").write_text(json.dumps([{"t": 0.0, "phase": "ingest"}, {"t": 42.5, "phase": "soak"}]))
    r = sc.read_run(tmp_path, "s12_iot_retention", "candidate", 1, 0, False, 400.0)
    assert r.completed and r.checks == {"a": False, "b": True}
    assert set(r.ops) == {"ingest_batch", "rare"} and r.ops["rare"]["ok"] == 20 and r.setup_s == 42.5
    assert not r.server_ran and not r.measured            # no monitor.csv: no server seen alive
    (tmp_path / "monitor.csv").write_text("t,cpu_pct,rss_mb,used_mem_mb,threads,clients,alive,phase\n"
                                          "0.0,nan,nan,nan,nan,nan,0,setup\n1.0,5,50,3,9,1,1,ingest\n")
    r = sc.read_run(tmp_path, "s12_iot_retention", "candidate", 1, 0, False, 400.0)
    assert r.server_ran and r.measured
    cut = sc.read_run(tmp_path, "s12_iot_retention", "candidate", 1, None, True, 400.0, cut=True)
    assert cut.cut and not cut.completed and cut.reason == "stopped at the deadline"


def test_compress_large_keeps_small_files(tmp_path):
    import gzip
    (tmp_path / "ledger-0.jsonl").write_text("x" * (1 << 20))
    (tmp_path / "checks.json").write_text("[]")
    sc.compress_large(tmp_path)
    assert sorted(f.name for f in tmp_path.iterdir()) == ["checks.json", "ledger-0.jsonl.gz"]
    with gzip.open(tmp_path / "ledger-0.jsonl.gz") as f:
        data = f.read()
    assert data == b"x" * (1 << 20)


def test_read_run_crash_and_exit(tmp_path):
    (tmp_path / "summary.json").write_text("{}")
    (tmp_path / "server-21000.log").write_text("1:M 24 Nov 2026 * Ready\n1:M ------ crashed by signal: 11 ------\n")
    r = sc.read_run(tmp_path, "s01_social_feed", "candidate", 1, 0, False, 1.0)
    assert not r.completed and "crashed by signal" in r.reason
    (tmp_path / "server-21000.log").write_text("fine\n")
    assert sc.read_run(tmp_path, "s01_social_feed", "candidate", 1, 1, False, 1.0).reason == "exit 1"
    assert sc.read_run(tmp_path, "s01_social_feed", "candidate", 1, None, True, 1.0).reason == "timed out"


# ------------------------------------------------------------------ findings and result
def test_pick_findings_caps_and_order():
    rows = [sc.Row(f"i{i}", "invariant", 0, 0, None, None, "lower_is_worse", "info") for i in range(15)]
    rows += [sc.Row(f"r{i}", "p99_ms", 2.0 + i, 1.0, 2.0 + i, 1.5, "higher_is_worse", "regression")
             for i in range(45)]
    out = sc.pick_findings(rows)
    assert len(out) == 50 and out[0]["name"] == "r44"
    assert sum(1 for f in out if f["status"] == "info") == 5


def test_scenario_list():
    assert sc.list_scenarios("s12,s01_social_feed") == ["s12_iot_retention", "s01_social_feed"]
    assert "s00_pilot" not in sc.list_scenarios(None) and len(sc.list_scenarios(None)) == 12
    with pytest.raises(ValueError):
        sc.list_scenarios("s99")


def test_default_label():
    assert sc.side_of(DIGEST_A, None, None)["label"] == "ghcr.io/falkordb/falkordb-server@aaaaaaaaaaaa"
    assert sc.side_of("x/y:6.2.0@sha256:" + "c" * 64, None, None)["label"] == "x/y@cccccccccccc"


def _validator():
    if not sc.VALIDATOR.exists():
        pytest.skip("tests/perf/validate.py is not on this branch yet")


class FakeSoak:
    """run_one stub on a fake clock. walls: scn or (scn, side) or (scn, side, rep) -> seconds a
    run would take (a run longer than its cap is stopped there); crash: keys whose run exits 1."""

    def __init__(self, clock, walls, crash=()):
        self.clock, self.walls, self.crash, self.calls = clock, walls, set(crash), []

    def _get(self, d, scn, side, rep):
        for k in ((scn, side, rep), (scn, side), scn):
            if k in d:
                return d[k]
        return None

    def run_one(self, scn, side, image, rep, hang_cap_s, stop_at):
        cap = max(60.0, min(hang_cap_s, stop_at - self.clock()))
        self.calls.append((scn, side, rep, round(cap)))
        wall = self._get(self.walls, scn, side, rep)
        timed_out = wall > cap
        crashed = not timed_out and any(k in self.crash for k in ((scn, side, rep), (scn, side)))
        self.clock.t += min(wall, cap)
        r = run(side, rep, completed=not (timed_out or crashed), checks={"x": True}, scenario=scn,
                cut=timed_out and cap < hang_cap_s, ops={"w": op(9.0 if side == "baseline" else 90.0)},
                wall_s=min(wall, cap))
        r.rc = None if timed_out else 1 if crashed else 0
        r.reason = ("stopped at the deadline" if r.cut else "timed out" if timed_out else
                    "exit 1" if crashed else "")
        return r


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


IMGS = {"candidate": "c", "baseline": "b"}


def test_a_side_the_deadline_would_cut_is_not_started():
    clock = Clock()
    p = dict(sc.MODES["nightly"])
    scns = ["s07_mvcc_consistency", "s08_durability_recovery", "s09_replication", "s10_schema_under_load"]
    soak = FakeSoak(clock, dict.fromkeys(scns, 950.0))
    sched = sc.run_all(soak, scns, IMGS, p, deadline=3500.0, clock=clock)
    # 2 x 950 s fit; then 1.2 x 950 no longer fits before the second side of the second scenario
    assert list(sched.runs) == ["s07_mvcc_consistency"]
    assert [n.split(" ")[0] for n in sched.not_run] == scns[1:]
    assert all(not r.cut for sides in sched.runs.values() for side in sides.values() for r in side)


def test_a_run_cut_by_the_deadline_drops_its_scenario():
    clock = Clock()
    p = dict(sc.MODES["nightly"], reruns=False)
    # the baseline side starts with 400 s left before stop_at: it is cut, not hung
    soak = FakeSoak(clock, {("s12_iot_retention", "candidate"): 400.0, ("s12_iot_retention", "baseline"): 5000.0})
    sched = sc.run_all(soak, ["s12_iot_retention"], IMGS, p, deadline=1200.0, clock=clock)
    assert soak.calls[1][3] == 1100 < 3 * 400        # stop_at (deadline + 300) bound it, not the hang cap
    assert sched.runs == {} and sched.not_run == ["s12_iot_retention (deadline)"]


def test_a_candidate_hang_is_a_regression_not_a_deadline(tmp_path):
    clock = Clock()
    p = dict(sc.MODES["nightly"])
    full = sc.run_cap(p)
    # the candidate (side A of s07) hangs to the full cap; the baseline still runs
    soak = FakeSoak(clock, {("s07_mvcc_consistency", "candidate"): 1e9, ("s07_mvcc_consistency", "baseline"): 380.0})
    sched = sc.run_all(soak, ["s07_mvcc_consistency"], IMGS, p, deadline=6600.0, clock=clock)
    assert soak.calls[0][3] == round(full) and len(soak.calls) == 2
    res = sc.finish(args(), tmp_path, 1.7e9, ["s07_mvcc_consistency"], sched, None, [], p)
    assert res["verdict"] == "regression" and res["findings"][0]["name"] == "s07 completed"


def test_a_candidate_hang_with_no_time_left_for_the_baseline_is_still_a_regression(tmp_path):
    clock = Clock()
    p = dict(sc.MODES["nightly"])
    soak = FakeSoak(clock, {("s07_mvcc_consistency", "candidate"): 1e9, ("s07_mvcc_consistency", "baseline"): 380.0})
    sched = sc.run_all(soak, ["s07_mvcc_consistency"], IMGS, p, deadline=2000.0, clock=clock)
    assert len(soak.calls) == 1 and sched.runs["s07_mvcc_consistency"]["baseline"] == []
    res = sc.finish(args(), tmp_path, 1.7e9, ["s07_mvcc_consistency"], sched, None, [], p)
    assert res["verdict"] == "regression" and res["findings"][0]["name"] == "s07 completed"
    if sc.VALIDATOR.exists():
        assert sc.validate(res) == []


def test_side_b_hang_is_capped_by_side_a_and_counts_as_a_hang(tmp_path):
    clock = Clock()
    p = dict(sc.MODES["nightly"])
    # s08 is index 1: baseline first (380 s), then the candidate hangs
    soak = FakeSoak(clock, {"s07_mvcc_consistency": 380.0,
                            ("s08_durability_recovery", "baseline"): 430.0,
                            ("s08_durability_recovery", "candidate"): 1e9})
    sched = sc.run_all(soak, ["s07_mvcc_consistency", "s08_durability_recovery"], IMGS,
                       dict(p, reruns=False), deadline=6600.0, clock=clock)
    hang = sched.runs["s08_durability_recovery"]["candidate"][0]
    assert soak.calls[-1][3] == 430 * 3 and not hang.cut and not hang.completed
    res = sc.finish(args(), tmp_path, 1.7e9, ["s07_mvcc_consistency", "s08_durability_recovery"],
                    sched, None, [], dict(p, reruns=False))
    assert res["verdict"] == "regression" and res["findings"][0]["name"] == "s08 completed"


def test_rerun_without_time_is_unconfirmed_and_an_error(tmp_path):
    clock = Clock()
    p = dict(sc.MODES["nightly"])
    soak = FakeSoak(clock, {"s12_iot_retention": 400.0})
    sched = sc.run_all(soak, ["s12_iot_retention"], IMGS, p, deadline=1200.0, clock=clock)
    assert sched.unconfirmed == ["s12_iot_retention (no time to re-run)"]
    res = sc.finish(args(), tmp_path, 1.7e9, ["s12_iot_retention"], sched, None, [], p)
    assert res["verdict"] == "error" and "no time to re-run" in res["error"]
    if sc.VALIDATOR.exists():
        assert sc.validate(res) == []


def test_rerun_that_repeats_confirms():
    clock = Clock()
    p = dict(sc.MODES["nightly"])
    soak = FakeSoak(clock, {"s12_iot_retention": 400.0})
    sched = sc.run_all(soak, ["s12_iot_retention"], IMGS, p, deadline=1e6, clock=clock)
    assert [c[1:3] for c in soak.calls] == [("candidate", 1), ("baseline", 1), ("baseline", 2), ("candidate", 2)]
    rows = sc.scenario_rows("s12_iot_retention", sched.runs["s12_iot_retention"]["candidate"],
                            sched.runs["s12_iot_retention"]["baseline"], True)
    assert sorted((r.name, r.metric) for r in rows if r.status == "regression") == [("s12 w", "p50_ms")]


def test_a_rerun_whose_baseline_crashed_is_unconfirmed(tmp_path):
    clock = Clock()
    p = dict(sc.MODES["nightly"])
    soak = FakeSoak(clock, {"s12_iot_retention": 400.0}, crash={("s12_iot_retention", "baseline", 2)})
    sched = sc.run_all(soak, ["s12_iot_retention"], IMGS, p, deadline=1e6, clock=clock)
    res = sc.finish(args(), tmp_path, 1.7e9, ["s12_iot_retention"], sched, None, [], p)
    assert res["verdict"] == "error" and "re-run incomplete" in res["error"]
    assert any(f["name"] == "s12 w (re-run incomplete)" for f in res["findings"])


def test_timing_pair_with_a_missing_baseline_value_is_not_confirmed():
    name = "victim open-loop schedule kept (few late/missed sends)"
    cand = [run("candidate", checks={name: False}), run("candidate", 2, checks={name: False})]
    base = [run("baseline", checks={name: True}), run("baseline", 2, checks={})]
    row = [r for r in sc.scenario_rows("s11_client_chaos", cand, base, True) if r.metric == "invariant"][0]
    assert row.status == "info" and row.candidate == 1


def test_both_sides_killed_outside_the_engine_is_not_measured(tmp_path):
    def killed(side):
        r = run(side, completed=False, checks={"ingest": True})
        r.rc, r.reason = -9, "exit -9"
        return r
    sched = sc.Schedule(runs={"s12_iot_retention": {"candidate": [killed("candidate")], "baseline": [killed("baseline")]}})
    res = sc.finish(args(), tmp_path, 1.7e9, ["s12_iot_retention"], sched, None, [], sc.MODES["nightly"])
    assert res["verdict"] == "error" and "ended outside the engine" in res["error"]


def test_summary_does_not_repeat_the_not_measured_list(tmp_path):
    res = sc.finish(args(), tmp_path, 1.7e9, ["s12_iot_retention"],
                    sc.Schedule(not_run=["s12_iot_retention (deadline)"]), None, [], sc.MODES["nightly"])
    assert res["summary"].count("s12_iot_retention (deadline)") == 1


def test_nothing_measured_is_an_error_not_a_pass(tmp_path):
    dead = sc.Schedule(runs={s: {"candidate": [run("candidate", completed=False, server_ran=False, scenario=s)],
                                 "baseline": [run("baseline", completed=False, server_ran=False, scenario=s)]}
                             for s in ("s07_mvcc_consistency", "s12_iot_retention")})
    res = sc.finish(args(), tmp_path, 1.7e9, list(dead.runs), dead, None, [], sc.MODES["nightly"])
    assert res["verdict"] == "error" and "no check recorded on either side" in res["error"]
    if sc.VALIDATOR.exists():
        assert sc.validate(res) == []


def test_candidate_server_that_never_starts_is_a_regression(tmp_path):
    sched = sc.Schedule(runs={"s12_iot_retention": {
        "candidate": [run("candidate", completed=False, server_ran=False)],
        "baseline": [run("baseline", checks={"x": True})]}})
    res = sc.finish(args(), tmp_path, 1.7e9, ["s12_iot_retention"], sched, None, [], sc.MODES["nightly"])
    assert res["verdict"] == "regression" and res["findings"][0]["name"] == "s12 completed"


def test_regression_summary_keeps_the_error(tmp_path):
    sched = sc.Schedule(runs={"s12_iot_retention": {
        "candidate": [run("candidate", checks={"x": False})], "baseline": [run("baseline", checks={"x": True})]}})
    res = sc.finish(args(), tmp_path, 1.7e9, ["s12_iot_retention"], sched, "terminated by signal 15", [],
                    sc.MODES["nightly"])
    assert res["verdict"] == "regression" and res["error"] is None
    assert "terminated by signal 15" in res["summary"]


@pytest.mark.parametrize("kw,msg", [
    ({"check": "Soak"}, "--check"), ({"check": "soak--a"}, "--check"),
    ({"candidate_label": "  "}, "label"), ({"baseline_commit": "abc"}, "40 hex"),
])
def test_inputs_refused_before_any_run(tmp_path, kw, msg):
    a = ["--candidate", DIGEST_A, "--baseline", DIGEST_B, "--out", str(tmp_path), "--mode", "queue"]
    flags = {"check": "--check", "candidate_label": "--candidate-label", "baseline_commit": "--baseline-commit"}
    for k, v in kw.items():
        a += [flags[k], v]
    assert sc.main(a) == 0
    res = json.loads((tmp_path / "result.json").read_text())
    assert res["verdict"] == "error" and msg in res["error"]


def test_uppercase_repo_is_refused_when_the_validator_is_here(tmp_path):
    _validator()
    sc.main(["--candidate", "FalkorDB/server@sha256:" + "a" * 64, "--baseline", DIGEST_B,
             "--out", str(tmp_path), "--mode", "queue"])
    res = json.loads((tmp_path / "result.json").read_text())
    assert "not an image pinned by digest" in res["error"]


def test_relative_out_is_resolved_before_use(tmp_path, monkeypatch):
    """_perf-vm.yml passes `--out out`: the result lands there, and Soak (whose run dirs are
    handed to the scenario subprocesses) sees the absolute path."""
    monkeypatch.chdir(tmp_path)
    seen = {}
    real = sc.Soak

    def spy(a):
        seen["out"] = a.out
        return real(a)
    monkeypatch.setattr(sc, "Soak", spy)
    assert sc.main(["--candidate", DIGEST_A, "--baseline", DIGEST_B, "--out", "out",
                    "--mode", "queue", "--check", "Bad"]) == 0
    assert Path(seen["out"]).is_absolute() and Path(seen["out"]) == tmp_path / "out"
    assert real(argparse.Namespace(**{**vars(args()), "out": seen["out"], "duration": None, "scale": None,
                                    "deadline_minutes": None, "no_quiet_gate": True})).raw == tmp_path / "out" / "raw"
    assert json.loads((tmp_path / "out" / "result.json").read_text())["verdict"] == "error"


def test_main_restores_the_sigterm_handler(tmp_path):
    import signal
    before = signal.getsignal(signal.SIGTERM)
    sc.main(["--candidate", "x:1", "--baseline", DIGEST_B, "--out", str(tmp_path), "--mode", "queue"])
    assert signal.getsignal(signal.SIGTERM) == before


def test_summary_is_one_line():
    line = sc.summary_line([], ["s12_iot_retention"], [], "queue", 60, "boom\u2028and\rmore\x85x")
    assert not sc.LINE_BREAK_RE.search(line)


def test_finish_writes_a_valid_regression(tmp_path):
    _validator()
    cand = [run("candidate", checks={"ingest reading count": True}, ops={"ingest_batch": op(90.0)}),
            run("candidate", 2, checks={"ingest reading count": True}, ops={"ingest_batch": op(75.0)})]
    base = [run("baseline", checks={"ingest reading count": True}, ops={"ingest_batch": op(9.0)}),
            run("baseline", 2, checks={"ingest reading count": True}, ops={"ingest_batch": op(9.0)})]
    res = sc.finish(args(), tmp_path, 1.7e9, ["s12_iot_retention"],
                    sc.Schedule(runs={"s12_iot_retention": {"candidate": cand, "baseline": base}}),
                    None, [], sc.MODES["nightly"])
    assert sc.validate(res) == []
    assert res["verdict"] == "regression" and res["confirmed"]
    assert res["findings"][0]["name"] == "s12 ingest_batch"
    assert json.loads((tmp_path / "result.json").read_text()) == res
    assert (tmp_path / "raw" / "rows.csv").exists()


def test_finish_pass_and_both_broken(tmp_path):
    _validator()
    cand = [run("candidate", checks={"x": False, "y": True}, ops={"w": op(3.0)})]
    base = [run("baseline", checks={"x": False, "y": True}, ops={"w": op(3.1)})]
    res = sc.finish(args(), tmp_path, 1.7e9, ["s12_iot_retention"],
                    sc.Schedule(runs={"s12_iot_retention": {"candidate": cand, "baseline": base}}),
                    None, [], sc.MODES["nightly"])
    assert sc.validate(res) == [] and res["verdict"] == "pass" and not res["confirmed"]
    assert [f["name"] for f in res["findings"]] == ["s12 x"]
    assert "1 broken on both sides" in res["summary"]


def test_finish_scenarios_not_run_is_an_error(tmp_path):
    _validator()
    res = sc.finish(args(), tmp_path, 1.7e9, ["s07_mvcc_consistency", "s12_iot_retention"],
                    sc.Schedule(not_run=["s07_mvcc_consistency (deadline)", "s12_iot_retention (deadline)"]),
                    None, [], sc.MODES["nightly"])
    assert sc.validate(res) == [] and res["verdict"] == "error"
    assert "s12_iot_retention (deadline)" in res["error"]


def test_main_rejects_a_tag_and_still_writes_result(tmp_path):
    rc = sc.main(["--candidate", "falkordb/falkordb-server:6.2.0", "--baseline", DIGEST_B,
                  "--out", str(tmp_path), "--mode", "nightly", "--scenarios", "s12"])
    res = json.loads((tmp_path / "result.json").read_text())
    assert rc == 0 and res["verdict"] == "error" and "pinned by digest" in res["error"]


def test_main_with_digests_off_linux_or_without_docker_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(sc.shutil, "which", lambda _: None)
    rc = sc.main(["--candidate", DIGEST_A, "--baseline", DIGEST_B, "--out", str(tmp_path),
                  "--mode", "queue", "--scenarios", "s12"])
    res = json.loads((tmp_path / "result.json").read_text())
    assert rc == 0 and res["verdict"] == "error"
    assert "Linux only" in res["error"] or "docker is not on PATH" in res["error"]
    if sc.VALIDATOR.exists():
        assert sc.validate(res) == []
