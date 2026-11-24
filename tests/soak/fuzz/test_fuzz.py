"""Tests for the fuzz check that need no server, no Docker and no redis-py.

    python -I -m pytest tests/soak -v      (perf-contract.yml runs this)

They cover the model (sizes, wiring, the build's queries), the checks on synthetic driver records,
the verdict logic, and every shape of result.json the check writes, through the contract's
validator (tests/perf/validate.py, which must be present: a missing validator fails, never skips).
"""

from __future__ import annotations

import ast
import collections
import importlib.util
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import types
from argparse import Namespace
from pathlib import Path

import pytest

from soakfuzz import check, checks, model, verdict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DIGEST_A = "ghcr.io/falkordb/falkordb-server@sha256:" + "a" * 64
DIGEST_B = "falkordb/falkordb-server@sha256:" + "b" * 64


# --- the model ----------------------------------------------------------------------------------


def test_sizes_come_from_the_formula():
    for i, label in enumerate(model.LABELS):
        assert model.COUNTS[label] == max(100, int(round(1_200_000 / 2**i, -2)))
    assert sum(model.COUNTS.values()) == 2_399_900


def test_wiring_gives_one_edge_per_driver_node_and_the_pairs_their_initial_edges():
    n = model.label_sizes(0.01)
    base, nxt = {}, 0
    for label, c in n.items():
        base[label], nxt = nxt, nxt + c
    plan = list(model.build_plan(n, base, batch=50))
    edges = collections.Counter()
    for phase, t, q in plan:
        if phase != "edges":
            continue
        lo, hi = map(int, re.search(r"range\((\d+), (\d+)\)", q).groups())
        mod = re.search(r"j % (\d+) = (\d+)", q)
        js = [j for j in range(lo, hi + 1) if not mod or j % int(mod[1]) == int(mod[2])]
        edges[t] += len(js)
    assert sum(edges.values()) == sum(w[4] for w in model.wiring(n))
    for src, t, dst in model.PAIRS:
        assert edges[t] == n[dst]
        assert all(model.initial_pair(n, src, dst, j % n[src], j) for j in range(n[dst]))
        assert not model.initial_pair(n, src, dst, 1, n[dst])
    assert set(edges) <= set(model.REL_TYPES)


def test_node_batches_cover_every_label_once():
    n = model.label_sizes(0.05)
    seen = collections.Counter()
    for phase, label, q in model.build_plan(n, None, batch=1000):
        assert phase == "nodes"
        lo, hi = map(int, re.search(r"range\((\d+), (\d+)\)", q).groups())
        seen[label] += hi - lo + 1
    assert seen == collections.Counter(n)


def test_queries_render_and_upsert_keys_match_their_params():
    n = model.label_sizes(1.0)
    rng = random.Random(7)
    for _ in range(200):
        for op in model.READ_OPS:
            cypher, params = model.make_read(op, n, rng)
            assert not re.search(r"\{[a-z_]+\}", cypher) and all(f"${p}" in cypher for p in params)
        cypher, params, keys = model.make_upsert(n, rng)
        assert all(f"${p}" in cypher for p in params)
        if len(keys) == 3:
            (la, ka), (lb, kb), (t, a, b) = keys
            assert (params["ka"], params["kb"]) == (ka, kb) == (a, b)
            assert params["ga"] == ka % model.GROUPS and model.pair_of(t)[0] == la


def test_the_full_scan_stays_a_label_less_scan_without_a_limit():
    """It holds the read lock because it visits every edge: no label, no LIMIT, a selective filter."""
    q = model.FULL_SCAN
    assert q.startswith("MATCH (a)-[r]->(b) WHERE") and "LIMIT" not in q


def test_the_control_epoch_is_fixed():
    hp, ep = model.control_params()
    assert hp["scales"] == [1.0, 0.1] and hp["thread_count"] == 32
    assert ep["busy"] == [0] and ep["long_streams"] == {"0": 1} and ep["write_pattern"] == "poisson"
    assert model.epoch_params("s", 3, 2, 3) == model.epoch_params("s", 3, 2, 3)


# --- anonymity guard ----------------------------------------------------------------------------


def _numbers(path: Path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            yield node.lineno, node.value


def test_the_workload_has_only_round_numbers():
    """Every number in the model is round (at most two significant digits, or a share with at most
    two decimals)."""
    bad = []
    for lineno, v in _numbers(HERE / "soakfuzz" / "model.py"):
        if isinstance(v, int) and abs(v) >= 100:
            digits = str(abs(v)).rstrip("0")
            if len(digits) > 2:
                bad.append((lineno, v))
        elif isinstance(v, float) and v != round(v, 2):
            bad.append((lineno, v))
    assert not bad


def test_no_internal_paths_or_addresses():
    pattern = re.compile(r"gs://|/opt/|\.handover|@[a-z0-9-]+\.(com|io|net)\b|\b\w+_\w+\.log\b")
    hits = []
    for path in sorted(HERE.rglob("*")):
        if path.suffix in (".py", ".md", ".toml") and ".venv" not in path.parts:
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if pattern.search(line) and path.name != "test_fuzz.py":
                    hits.append(f"{path.name}:{lineno}")
    assert not hits


# --- the checks -----------------------------------------------------------------------------------


def _epoch_data(reads=(), writes=(), longs=()):
    return {"reads": list(reads), "writes": list(writes), "longs": list(longs), "errors": {}, "died": [],
            "unfinished": 0, "pending_at_stop": 0, "drain_s": 0.0, "harness": {}}


def _analyse(data, pings=None, quiet=False):
    hp, ep = model.control_params((1.0, 0.1) if quiet else (1.0,))
    cal = [{"lookup": 1.0, "filter-sort": 3.0, "distinct": 3.0, "scan-lookup": 200.0}] * hp["graphs"]
    pings = pings or {"falkor": [(t / 100, 1.0) for t in range(1000)], "control": [(t / 100, 1.0) for t in range(1000)]}
    return checks.analyse(ep, hp, cal, data, pings, [], [], {}, [])["checks"]


def test_a_cheap_read_that_times_out_breaks_no_timeouts_while_waiting():
    ok = _analyse(_epoch_data(reads=[[0, "lookup", 1.0, 1.0, 1.01, 0], [0, "scan-lookup", 2.0, 2.0, 2.5, 1]]))
    assert ok["no_timeouts_while_waiting"]["pass"] and ok["no_timeouts_while_waiting"]["all_read_timeouts"] == 1
    bad = _analyse(_epoch_data(reads=[[0, "lookup", 1.0, 1.0, 2.0, 1]]))
    assert bad["no_timeouts_while_waiting"]["pass"] is False


def test_a_late_ping_is_a_freeze_only_while_the_plain_server_answered():
    late = {"falkor": [(5.0, 900.0)], "control": [(5.0, 2.0)]}
    assert _analyse(_epoch_data(), late)["no_freeze"]["pass"] is False
    host_stall = {"falkor": [(5.0, 900.0)], "control": [(5.0, 400.0)]}
    assert _analyse(_epoch_data(), host_stall)["no_freeze"]["pass"] is True


def test_a_write_finishing_long_after_the_long_read_it_waited_behind_starves():
    longs = [[0, "full-scan", 10.0, 18.0, 0, ""]]
    fine = [[0, "upsert", 11.0, 11.0, 18.5, 0, "", []]]
    slow = [[0, "upsert", 11.0, 11.0, 20.0, 0, "", []]]
    assert _analyse(_epoch_data(writes=fine, longs=longs))["writers_dont_starve"]["pass"]
    assert _analyse(_epoch_data(writes=slow, longs=longs))["writers_dont_starve"]["pass"] is False


def test_quiet_graph_reads_must_stay_near_their_cost():
    fast = [[1, "lookup", 1.0, 1.0, 1.05, 0]]
    stalled = [[1, "lookup", 1.0, 1.0, 1.5, 0]]
    assert _analyse(_epoch_data(reads=fast), quiet=True)["quiet_graphs_stay_fast"]["pass"]
    assert _analyse(_epoch_data(reads=stalled), quiet=True)["quiet_graphs_stay_fast"]["pass"] is False


class FakeGraph:
    """Answers the ledger's count queries from a table."""

    def __init__(self, counts):
        self.counts = counts

    def execute_command(self, cmd, graph, q):
        for key, v in self.counts.items():
            if key in q:
                return [[], [[v]]]
        return [[], [[0]]]


def test_ledger_tells_missing_upsert_edges_from_wrong_counts():
    n = model.label_sizes(0.01)
    led = checks.Ledger([n])
    src, t, dst = model.PAIRS[0]
    new = n[dst] + 3
    led.add(0, [(src, 1), (dst, new), (t, 1, new)], 0)
    exact = {f"(x:{lbl})" + " RETURN count(x)": n[lbl] for lbl in model.UPSERT_LABELS}
    exact[f"(x:{dst}) RETURN count(x)"] = n[dst] + 1
    for s_, tt, d in model.PAIRS:
        exact[f"[e:{tt}]"] = n[d]
    missing = led.check(FakeGraph(exact))
    assert [k for k, _ in missing] == ["missing"]
    exact[f"[e:{t}]"] = n[dst] + 1
    assert led.check(FakeGraph(exact)) == []
    exact[f"[e:{t}]"] = n[dst] + 2
    assert [k for k, _ in led.check(FakeGraph(exact))] == ["count"]


# --- the verdict ----------------------------------------------------------------------------------


def _run(side, i, held, extra=False, died=False, started=True):
    """A control run whose checks all hold except the ones named in `held` as False."""
    cks = {c: {"pass": held.get(c, True)} for c in verdict.CONTROL_CHECKS}
    cks["server_stays_up"] = {"pass": not died}
    return {"side": side, "index": i, "extra": extra, "started": started, "server_died": died,
            "epoch": {"checks": cks} if started else None}


def _first(cand=None, base=None):
    cand, base = cand or {}, base or {}
    return [_run(s, i, cand if s == "candidate" else base) for i, s in enumerate(check.FIRST_ORDER)]


def _by_name(findings):
    return {f["name"]: f for f in findings}


def test_same_image_passes():
    f, _ = verdict.control_findings(_first(), False)
    assert {x["status"] for x in f} == {"ok"} and verdict.verdict_of(f, []) == "pass"


def test_a_timing_break_needs_a_repeat():
    runs = _first(cand={"no_timeouts_while_waiting": False})
    assert verdict.needs_confirmation(runs) == ["no_timeouts_while_waiting"]
    repeated = runs + [_run("baseline", 4, {}, True), _run("candidate", 5, {"no_timeouts_while_waiting": False}, True)]
    f = _by_name(verdict.control_findings(repeated, True)[0])["control no_timeouts_while_waiting"]
    assert (f["candidate"], f["baseline"], f["status"]) == (0, 1, "regression")
    held = runs + [_run("baseline", 4, {}, True), _run("candidate", 5, {}, True)]
    f, notes = verdict.control_findings(held, True)
    f = _by_name(f)["control no_timeouts_while_waiting"]
    assert (f["candidate"], f["baseline"], f["status"]) == (1, 1, "info")
    assert any("broke in 2 of 3 candidate runs, held on the repeat" in n for n in notes)
    none_ran = _by_name(verdict.control_findings(runs, False)[0])["control no_timeouts_while_waiting"]
    assert none_ran["candidate"] is None and none_ran["status"] == "info"


def test_a_break_seen_only_in_the_extra_pair_is_not_a_regression():
    runs = _first(cand={"no_freeze": False})
    runs += [_run("baseline", 4, {}, True), _run("candidate", 5, {"no_freeze": False, "requests_drain": False}, True)]
    f = _by_name(verdict.control_findings(runs, True)[0])
    assert f["control no_freeze"]["status"] == "regression"
    assert (f["control requests_drain"]["candidate"], f["control requests_drain"]["status"]) == (1, "info")


def test_data_breaks_and_crashes_count_at_once_and_both_broken_is_info():
    runs = _first(cand={"writes_add_up": False})
    assert verdict.needs_confirmation(runs) == []
    assert _by_name(verdict.control_findings(runs, False)[0])["control writes_add_up"]["status"] == "regression"
    both = _first(cand={"writers_dont_starve": False}, base={"writers_dont_starve": False})
    f = _by_name(verdict.control_findings(both, False)[0])["control writers_dont_starve"]
    assert (f["candidate"], f["baseline"], f["status"]) == (0, 0, "info")
    crash = _first()
    crash[0] = _run("candidate", 0, {}, died=True, started=False)
    f = _by_name(verdict.control_findings(crash, False)[0])["control server_stays_up"]
    assert (f["metric"], f["candidate"], f["status"]) == ("completed", 0, "regression")


def _build(side, i, edges, extra=False):
    return {"side": side, "index": i, "extra": extra, "ok": True, "edges_ms": edges, "nodes_ms": 900.0,
            "probe_1_ms": 0.5, "probe_100_ms": 0.6}


def test_build_regression_needs_the_ratio_the_floor_and_a_repeat():
    slow = [_build(s, i, 7900.0 if s == "candidate" else 1300.0) for i, s in enumerate(check.FIRST_ORDER)]
    assert verdict.build_needs_confirmation(slow) == ["build edge phase"]
    rep = slow + [_build("baseline", 4, 1300.0, True), _build("candidate", 5, 8000.0, True)]
    f = _by_name(verdict.build_findings(rep)[0])["build edge phase"]
    assert f["status"] == "regression" and f["ratio"] > 1.5
    gone = slow + [_build("baseline", 4, 1300.0, True), _build("candidate", 5, 1350.0, True)]
    assert _by_name(verdict.build_findings(gone)[0])["build edge phase"]["status"] == "info"
    small = [_build(s, i, 300.0 if s == "candidate" else 100.0) for i, s in enumerate(check.FIRST_ORDER)]
    assert verdict.build_needs_confirmation(small) == []  # 3x but under the 500 ms floor


# --- result.json through the contract's validator ------------------------------------------------------


def _validator():
    path = ROOT / "tests" / "perf" / "validate.py"
    if not path.exists():
        pytest.fail(f"{path} is missing: the fuzz check's results are validated with the contract's "
                    "validator (perf-judge, #3986)")
    spec = importlib.util.spec_from_file_location("perf_validate", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["perf_validate"] = mod
    spec.loader.exec_module(mod)
    return mod


def _args(**kw):
    return Namespace(candidate=DIGEST_A, baseline=DIGEST_B, candidate_label=kw.get("cl"), baseline_label=None,
                     candidate_commit=kw.get("cc"), baseline_commit=None)


MACHINE = {"runner": "local", "type": "test", "cpu": "test", "cores": 1}


def _results():
    t0, t1 = "2026-11-24T00:00:00Z", "2026-11-24T00:40:00Z"
    aa = verdict.control_findings(_first(), False)[0]
    runs = _first(cand={"no_timeouts_while_waiting": False}, base={"no_freeze": False, "writers_dont_starve": False})
    runs += [_run("baseline", 4, {"no_freeze": False, "writers_dont_starve": False}, True),
             _run("candidate", 5, {"no_timeouts_while_waiting": False}, True)]
    reg = verdict.control_findings(runs, True)[0]
    slow = [_build(s, i, 7900.0 if s == "candidate" else 1300.0) for i, s in enumerate(check.FIRST_ORDER)]
    slow += [_build("baseline", 4, 1300.0, True), _build("candidate", 5, 8000.0, True)]
    bf = verdict.build_findings(slow)[0]
    unconfirmed = verdict.control_findings(_first(cand={"requests_drain": False}), False)[0]
    return {
        "pass": check.build_result(_args(cl="main@4c10889", cc="4" * 40), t0, t1, aa, [], [], None, MACHINE),
        "regression": check.build_result(_args(), t0, t1, reg + bf, ["note"], [], None, MACHINE),
        "unconfirmed": check.build_result(_args(), t0, t1, unconfirmed, [], ["control requests_drain: x"], None, MACHINE),
        "error": check.build_result(_args(), t0, t1, [], [], [], "RuntimeError: baseline image: pull failed", MACHINE),
    }


@pytest.mark.parametrize("kind", ["pass", "regression", "unconfirmed", "error"])
def test_every_result_shape_is_valid(kind, tmp_path):
    validate = _validator()
    doc = _results()[kind]
    path = tmp_path / "result.json"
    path.write_text(json.dumps(doc, allow_nan=False))
    _kind, problems = validate.validate_file(path, "result")
    assert problems == [], problems
    assert doc["verdict"] == {"pass": "pass", "regression": "regression", "unconfirmed": "error", "error": "error"}[kind]


def test_proof_patterns_match_the_regression_findings():
    doc = _results()["regression"]
    regs = [f["name"] for f in doc["findings"] if f["status"] == "regression"]
    assert any(re.search("control", n, re.I) for n in regs) and any(re.search("build", n, re.I) for n in regs)
    assert doc["confirmed"] is True


def test_the_command_writes_an_error_result_for_an_image_without_a_digest(tmp_path):
    """The one result that cannot be valid: the image it must record is not a digest (merge.py stubs it)."""
    out = tmp_path / "out"
    assert check.main(["--candidate", "falkordb/falkordb-server:edge", "--baseline", DIGEST_B,
                       "--out", str(out), "--mode", "nightly"]) == 0
    doc = json.loads((out / "result.json").read_text())
    assert doc["verdict"] == "error" and "digest" in doc["error"]
    _kind, problems = _validator().validate_file(out / "result.json", "result")
    assert [p.code for p in problems] == ["image-not-digest"], problems


def test_the_digest_gate_is_the_validators():
    assert check.DIGEST.fullmatch(DIGEST_A) and check.DIGEST.fullmatch("localhost:5000/a/b:t@sha256:" + "c" * 64)
    assert not check.DIGEST.fullmatch("ghcr.io/FalkorDB/falkordb-server@sha256:" + "a" * 64)
    assert not check.DIGEST.fullmatch("falkordb/falkordb-server@sha256:" + "A" * 64)


def test_more_result_shapes_are_valid(tmp_path):
    """A data break against an unmeasured baseline, and long-mode findings."""
    validate = _validator()
    t0, t1 = "2026-11-24T00:00:00Z", "2026-11-24T04:00:00Z"
    runs = _first(cand={"writes_add_up": False})
    runs[1] = _run("baseline", 1, {}, started=False)
    data_break = verdict.control_findings(runs, False)[0]
    hours = [_hour("candidate", {"x-h1-e1": {"server_stays_up": False}, "x-h1-e2": {"no_freeze": False}}),
             _hour("baseline", {"x-h1-e1": {}, "x-h1-e2": {}})]
    lf, _notes, unmeasured = check.long_findings(hours, [])
    for i, (fs, um) in enumerate(((data_break, ["control: x"]), (lf, unmeasured))):
        doc = check.build_result(_args(), t0, t1, fs, [], um, None, MACHINE)
        path = tmp_path / f"r{i}.json"
        path.write_text(json.dumps(doc, allow_nan=False))
        assert validate.validate_file(path, "result")[1] == []
        assert doc["verdict"] == "regression"


# --- long mode ------------------------------------------------------------------------------------


def _hour(side, seeds, replay=False):
    """A long-mode server-hour: {seed: {check: held}}; every check not named holds."""
    return {"side": side, "replay": replay,
            "epochs": {s_: {c: held.get(c, True) for c in check.LONG_CHECKS} for s_, held in seeds.items()}}


def test_long_mode_counts_a_seed_only_when_its_replay_repeats():
    first = [_hour("candidate", {"x-h1-e1": {"no_freeze": False}, "x-h1-e2": {}}),
             _hour("baseline", {"x-h1-e1": {}, "x-h1-e2": {}})]
    repeated = first + [_hour("baseline", {"x-h1-e1": {}}, True), _hour("candidate", {"x-h1-e1": {"no_freeze": False}}, True)]
    f = _by_name(check.long_findings(repeated, [])[0])
    assert f["long no_freeze"]["status"] == "regression" and f["long writes_add_up"]["status"] == "ok"
    gone = first + [_hour("baseline", {"x-h1-e1": {}}, True), _hour("candidate", {"x-h1-e1": {}}, True)]
    f = _by_name(check.long_findings(gone, [])[0])["long no_freeze"]
    assert (f["candidate"], f["status"]) == (1, "info")
    unreplayed = _by_name(check.long_findings(first, [])[0])["long no_freeze"]
    assert unreplayed["candidate"] is None and unreplayed["status"] == "info"
    both = [_hour("candidate", {"x-h1-e1": {"writers_dont_starve": False}}),
            _hour("baseline", {"x-h1-e1": {"writers_dont_starve": False}})]
    f = _by_name(check.long_findings(both, [])[0])["long writers_dont_starve"]
    assert (f["candidate"], f["baseline"], f["status"]) == (0, 0, "info")


def test_a_repro_break_needs_a_repeat():
    first = [{"name": "write-create", "side": "candidate", "replay": False, "pass": False},
             {"name": "write-create", "side": "baseline", "replay": False, "pass": True}]
    again = first + [{"name": "write-create", "side": "baseline", "replay": True, "pass": True},
                     {"name": "write-create", "side": "candidate", "replay": True, "pass": False}]
    assert _by_name(check.long_findings([], again)[0])["long repro write-create"]["status"] == "regression"
    held = first + [{"name": "write-create", "side": "baseline", "replay": True, "pass": True},
                    {"name": "write-create", "side": "candidate", "replay": True, "pass": True}]
    f = _by_name(check.long_findings([], held)[0])["long repro write-create"]
    assert (f["candidate"], f["status"]) == (1, "info")


def test_long_mode_crashes_and_data_breaks_count_at_once():
    hours = [_hour("candidate", {"x-h1-e1": {"server_stays_up": False}, "x-h1-e2": {"writes_add_up": False}}),
             _hour("baseline", {"x-h1-e1": {}, "x-h1-e2": {}})]
    f, _notes, unmeasured = check.long_findings(hours, [])
    f = _by_name(f)
    assert (f["long server_stays_up"]["metric"], f["long server_stays_up"]["status"]) == ("completed", "regression")
    assert f["long writes_add_up"]["status"] == "regression" and unmeasured == []
    crash = [{"name": "index-build", "side": "candidate", "replay": False, "pass": False, "crashed": True},
             {"name": "index-build", "side": "baseline", "replay": False, "pass": True, "crashed": False}]
    assert _by_name(check.long_findings([], crash)[0])["long repro index-build"]["status"] == "regression"


def test_long_mode_with_nothing_paired_is_unmeasured():
    _f, _n, unmeasured = check.long_findings([], [])
    assert unmeasured
    one_side = [_hour("candidate", {"x-h1-e1": {}})]
    assert check.long_findings(one_side, [])[2]


def test_a_timing_break_against_an_unmeasured_baseline_is_not_judged():
    runs = _first(cand={"no_freeze": False})
    runs[1] = _run("baseline", 1, {}, started=False)
    runs[2] = _run("baseline", 2, {}, started=False)
    f = _by_name(verdict.control_findings(runs, False)[0])["control no_freeze"]
    assert f["candidate"] is None and f["status"] == "info"
    assert verdict.status_of(0, None) == "regression"


def test_an_unmeasured_first_run_does_not_borrow_the_extra_pair():
    runs = _first()
    runs[0] = _run("candidate", 0, {}, started=False)
    runs[3] = _run("candidate", 3, {"no_freeze": False})
    runs += [_run("baseline", 4, {}, True), _run("candidate", 5, {"requests_drain": False}, True)]
    f = _by_name(verdict.control_findings(runs, True)[0])
    assert f["control requests_drain"]["status"] != "regression"


def test_a_baseline_break_only_on_the_repeat_is_not_an_improvement():
    runs = [_run("candidate", 0, {"no_freeze": False}), _run("baseline", 1, {}), _run("baseline", 2, {}),
            _run("candidate", 3, {}), _run("baseline", 4, {"no_freeze": False}, True), _run("candidate", 5, {}, True)]
    f = _by_name(verdict.control_findings(runs, True)[0])["control no_freeze"]
    assert (f["candidate"], f["baseline"], f["status"]) == (0, 0, "info")


def test_error_replies_break_no_error_replies():
    ok = _analyse(_epoch_data(reads=[[0, "lookup", 1.0, 1.0, 1.01, 0]]))
    assert ok["no_error_replies"]["pass"]
    bad = _analyse(_epoch_data(reads=[[0, "lookup", 1.0, 1.0, 1.01, 2]],
                               writes=[[0, "upsert", 1.0, 1.0, 1.1, 2, "ERR x", []]]))
    assert bad["no_error_replies"]["pass"] is False and bad["no_error_replies"]["by_op"] == {"lookup": 1, "upsert": 1}


def test_a_confirmed_long_break_is_not_masked_by_a_seed_broken_on_both_sides():
    first = [_hour("candidate", {"x-h1-e1": {"writers_dont_starve": False}, "x-h1-e2": {"writers_dont_starve": False}}),
             _hour("baseline", {"x-h1-e1": {}, "x-h1-e2": {"writers_dont_starve": False}})]
    rep = first + [_hour("baseline", {"x-h1-e1": {}}, True), _hour("candidate", {"x-h1-e1": {"writers_dont_starve": False}}, True)]
    f = _by_name(check.long_findings(rep, [])[0])["long writers_dont_starve"]
    assert (f["candidate"], f["baseline"], f["status"]) == (0, 1, "regression")


def test_a_crash_on_replay_counts_and_leaves_the_timing_check_unmeasured():
    first = [_hour("candidate", {"x-h1-e1": {"no_freeze": False}}), _hour("baseline", {"x-h1-e1": {}})]
    crashed = {c: None for c in check.LONG_CHECKS}
    crashed["server_stays_up"] = False
    rep = first + [_hour("baseline", {"x-h1-e1": {}}, True),
                   {"side": "candidate", "replay": True, "epochs": {"x-h1-e1": crashed}}]
    f, _n, unmeasured = check.long_findings(rep, [])
    f = _by_name(f)
    assert f["long server_stays_up"]["status"] == "regression"
    assert f["long no_freeze"]["candidate"] is None and unmeasured


def test_a_long_crash_on_a_seed_only_the_candidate_ran_still_counts():
    hours = [_hour("candidate", {"x-h1-e1": {}, "x-h2-e1": {"server_stays_up": False}}),
             _hour("baseline", {"x-h1-e1": {}})]
    f = _by_name(check.long_findings(hours, [])[0])["long server_stays_up"]
    assert (f["candidate"], f["baseline"], f["status"]) == (0, None, "regression")
    assert "long memory_steady" not in _by_name(check.long_findings(hours, [])[0])


def _stub_redis(monkeypatch):
    """Just enough of redis-py for soakfuzz.harness to import (no test talks to a server)."""
    mods = {name: types.ModuleType(name) for name in ("redis", "redis.backoff", "redis.exceptions", "redis.retry")}
    mods["redis"].Redis = object
    mods["redis.backoff"].NoBackoff = object
    mods["redis.exceptions"].ResponseError = type("ResponseError", (Exception,), {})
    mods["redis.retry"].Retry = object
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "soakfuzz.harness", raising=False)


def test_sigterm_is_not_swallowed_and_still_writes_a_result(tmp_path, monkeypatch):
    _stub_redis(monkeypatch)
    from soakfuzz import harness

    def probe(image):
        try:  # the signal arrives inside a harness `except Exception` block
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)
        except Exception:
            pytest.fail("SIGTERM was swallowed by an `except Exception`")

    monkeypatch.setattr(harness, "probe_image", probe)
    monkeypatch.setattr(check, "cleanup", lambda prefix: None)
    before = signal.getsignal(signal.SIGTERM)
    out = tmp_path / "out"
    assert check.main(["--candidate", DIGEST_A, "--baseline", DIGEST_B, "--out", str(out), "--mode", "queue"]) == 0
    doc = json.loads((out / "result.json").read_text())
    assert doc["verdict"] == "error" and "signal" in doc["error"]
    assert signal.getsignal(signal.SIGTERM) == before


def test_a_docker_timeout_is_a_docker_error_not_a_crash(monkeypatch):
    _stub_redis(monkeypatch)
    from soakfuzz import harness

    def hang(*a, **kw):
        raise subprocess.TimeoutExpired(a[0], kw.get("timeout"))

    monkeypatch.setattr(harness.subprocess, "run", hang)
    with pytest.raises(harness.DockerError):
        harness.docker("inspect", "x")


def test_a_long_data_break_against_an_unmeasured_baseline_counts():
    base = {c: True for c in check.LONG_CHECKS}
    base["writes_add_up"] = None
    hours = [_hour("candidate", {"x-h1-e1": {"writes_add_up": False}}),
             {"side": "baseline", "replay": False, "epochs": {"x-h1-e1": base}}]
    f = _by_name(check.long_findings(hours, [])[0])["long writes_add_up"]
    assert (f["candidate"], f["baseline"], f["status"]) == (0, None, "regression")
    crash_on_replay = [_hour("candidate", {"x-h1-e1": {"no_freeze": False}}), _hour("baseline", {"x-h1-e1": {}}),
                       _hour("candidate", {"x-h1-e1": {"server_stays_up": False}}, True)]
    assert _by_name(check.long_findings(crash_on_replay, [])[0])["long server_stays_up"]["status"] == "regression"


def test_a_relative_out_is_resolved_where_the_check_starts(tmp_path, monkeypatch):
    """Contract v1.8: `--out out` is relative; a later chdir must not move result.json."""
    _stub_redis(monkeypatch)
    from soakfuzz import harness
    start, elsewhere = tmp_path / "start", tmp_path / "elsewhere"
    start.mkdir()
    elsewhere.mkdir()

    def probe(image):
        os.chdir(elsewhere)
        raise harness.DockerError("no docker in this test")

    monkeypatch.setattr(harness, "probe_image", probe)
    monkeypatch.setattr(check, "cleanup", lambda prefix: None)
    monkeypatch.chdir(start)
    assert check.main(["--candidate", DIGEST_A, "--baseline", DIGEST_B, "--out", "out", "--mode", "queue"]) == 0
    doc = json.loads((start / "out" / "result.json").read_text())
    assert doc["verdict"] == "error" and "no docker" in doc["error"]
    assert (start / "out" / "raw" / "rows.csv").exists() and not (elsewhere / "out").exists()


def test_a_failed_clean_up_is_reported_and_the_result_still_written(tmp_path, monkeypatch, capsys):
    def boom(prefix):
        raise RuntimeError("docker is gone")

    monkeypatch.setattr(check, "cleanup", boom)
    out = tmp_path / "out"
    assert check.main(["--candidate", "x:latest", "--baseline", DIGEST_B, "--out", str(out), "--mode", "queue"]) == 0
    assert json.loads((out / "result.json").read_text())["verdict"] == "error"
    assert "docker is gone" in capsys.readouterr().err
