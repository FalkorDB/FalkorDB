"""Tests for the fuzz check that need no server and no Docker.

    uv run --project bench --frozen --group dev --group fuzz python -I -B -m pytest bench/tests/test_fuzz.py

They cover the model (sizes, wiring, the build's queries), the checks on synthetic driver records,
the verdict logic, every shape of result.json the check writes (through the contract's
validator), and the harness's mechanisms on fakes: the pool wait, the epoch's schema events, the
clean-up, the server's container, and clients that never re-send.
"""

from __future__ import annotations

import ast
import asyncio
import collections
import io
import itertools
import json
import os
import random
import re
import signal
import socket
import subprocess
import threading
import time
import types
from argparse import Namespace
from pathlib import Path

import pytest
import redis.asyncio as aredis
import redis.exceptions
import redis.utils

from falkorbench import contract
from falkorbench import kit
from falkorbench.fuzz import build
from falkorbench.fuzz import check
from falkorbench.fuzz import checks
from falkorbench.fuzz import harness
from falkorbench.fuzz import model
from falkorbench.fuzz import repros
from falkorbench.fuzz import verdict
from falkorbench.soak import soakkit

PACKAGE = Path(check.__file__).resolve().parent
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
            (la, ka), (_lb, kb), (t, a, b) = keys
            assert (params["ka"], params["kb"]) == (ka, kb) == (a, b)
            assert params["ga"] == ka % model.GROUPS and model.pair_of(t)[0] == la


def test_the_full_scan_stays_a_label_less_scan_without_a_limit():
    """It holds the read lock because it visits every edge: no label, no LIMIT, a selective filter."""
    q = model.FULL_SCAN
    assert q.startswith("MATCH (a)-[r]->(b) WHERE") and "LIMIT" not in q


def test_a_scale_range_the_cap_cannot_fit_is_refused():
    with pytest.raises(ValueError, match="exceed the cap"):
        model.hour_params("s", 1, (0.25, 2.0, 0.4))
    assert model.hour_params("s", 1, (0.25, 0.3, 0.6))["graphs"] == 2  # only two fit
    for h in range(1, 20):  # the default range draws the graph count as before
        assert model.hour_params("s", h)["graphs"] == random.Random(f"s-h{h}").choice([2, 3, 4])


def test_the_control_epoch_is_fixed():
    hp, ep = model.control_params()
    assert hp["scales"] == [1.0, 0.1] and hp["thread_count"] == 32
    assert ep["busy"] == [0] and ep["long_streams"] == {"0": 1} and ep["write_pattern"] == "poisson"
    assert model.epoch_params("s", 3, 2, 3) == model.epoch_params("s", 3, 2, 3)


# --- anonymity guard ----------------------------------------------------------------------------


def _numbers(path: Path):
    for node in ast.walk(ast.parse(path.read_text())):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            yield node.lineno, node.value


def test_the_workload_has_only_round_numbers():
    """Every number in the model is round (at most two significant digits, or a share with at most
    two decimals)."""
    bad = []
    for lineno, v in _numbers(PACKAGE / "model.py"):
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
    for path in sorted(PACKAGE.rglob("*")):
        if path.suffix in (".py", ".md"):
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if pattern.search(line):
                    hits.append(f"{path.name}:{lineno}")
    assert not hits


# --- the checks -----------------------------------------------------------------------------------


def _epoch_data(reads=(), writes=(), longs=()):
    return {
        "reads": list(reads),
        "writes": list(writes),
        "longs": list(longs),
        "errors": {},
        "died": [],
        "unfinished": 0,
        "pending_at_stop": 0,
        "drain_s": 0.0,
        "harness": {},
    }


LONG = [[0, "full-scan", 10.0, 18.0, 0, ""]]  # a long read from 10 s to 18 s
# 2 s on a quiet graph, all of it waiting for a connection
POOL_STALLED = [1, "lookup", 1.0, 1.0, 3.0, 0, 1.99]
QUIET_FAST = [1, "lookup", 2.0, 2.0, 2.01, 0, 0.0]
# 30 ms late to be sent, then 30 ms for a connection: 60 ms in the driver
QUIET_LATE_TWICE = [1, "lookup", 1.0, 1.03, 1.2, 0, 0.03]
# a write that waited 3 s for a connection, and one sent after it that reached the server first
WRITE_POOL_STALLED = [0, "upsert", 0.0, 0.0, 4.1, 0, "", [], 3.0]
WRITE_ARRIVED_FIRST = [0, "upsert", 1.0, 1.0, 2.5, 0, "", [], 0.0]


def _pings(n, late=None, control_gap=None):
    """n PINGs 10 ms apart on both servers; `late` = (index, ms) on FalkorDB; `control_gap` = (from, to)
    seconds with no plain-server replies."""
    f = [(i / 100, 1.0) for i in range(n)]
    c = [
        (i / 100, 1.0)
        for i in range(n)
        if not control_gap or not control_gap[0] <= i / 100 <= control_gap[1]
    ]
    if late:
        f[late[0]] = (late[0] / 100, late[1])
    return {"falkor": f, "control": c}


# fmt: off
ANALYSE_CASES = [  # (id, check, pass, records, pings, window_s, quiet, ledger problems, more fields to assert)
    ("a slow read may time out", "no_timeouts_while_waiting", True,
     {"reads": [[0, "lookup", 1.0, 1.0, 1.01, 0], [0, "scan-lookup", 2.0, 2.0, 2.5, 1]]}, None, None, False, (),
     {"all_read_timeouts": 1}),
    ("a cheap read may not", "no_timeouts_while_waiting", False, {"reads": [[0, "lookup", 1.0, 1.0, 2.0, 1]]},
     None, None, False, (), {}),
    ("no cheap read answered", "no_timeouts_while_waiting", None,
     {"reads": [[0, "lookup", 1.0, 1.0, 2.0, 2], [0, "scan-lookup", 2.0, 2.0, 2.5, 0]]}, None, None, False, (), {}),
    ("late PING, plain server fast", "no_freeze", False, {}, {"falkor": [(5.0, 900.0)], "control": [(5.0, 2.0)]},
     None, False, (), {}),
    ("late PING during a host stall", "no_freeze", True, {}, {"falkor": [(5.0, 900.0)], "control": [(5.0, 400.0)]},
     None, False, (), {}),
    ("PINGs cover the epoch", "no_freeze", True, {}, _pings(1000), 10.0, False, (), {}),
    ("late PING, full coverage", "no_freeze", False, {}, _pings(1000, late=(500, 900.0)), 10.0, False, (), {}),
    ("the prober never ran", "no_freeze", None, {}, {"falkor": [], "control": []}, 10.0, False, (), {}),
    ("under half the epoch covered", "no_freeze", None, {}, _pings(300), 10.0, False, (), {}),
    ("late PING, no plain reply near it", "no_freeze", None, {},
     _pings(1000, late=(500, 900.0), control_gap=(3.9, 6.1)), 10.0, False, (), {"unattributed_late_pings": 1}),
    ("a freeze with little coverage is still a freeze", "no_freeze", False, {}, _pings(300, late=(100, 900.0)),
     10.0, False, (), {"freeze_pings": 1}),
    ("a write 0.5 s after the long read", "writers_dont_starve", True,
     {"writes": [[0, "upsert", 11.0, 11.0, 18.5, 0, "", []]], "longs": LONG}, None, None, False, (), {}),
    ("a write 2 s after it", "writers_dont_starve", False,
     {"writes": [[0, "upsert", 11.0, 11.0, 20.0, 0, "", []]], "longs": LONG}, None, None, False, (), {}),
    ("a driver-limited write is not judged", "writers_dont_starve", True,
     {"writes": [[0, "upsert", 11.0, 11.0, 20.0, 0, "", [], 8.9], [0, "upsert", 11.0, 11.0, 18.5, 0, "", [], 0.0]],
      "longs": LONG}, None, None, False, (), {}),
    ("every write driver-limited", "writers_dont_starve", None,
     {"writes": [[0, "upsert", 11.0, 11.0, 20.0, 0, "", [], 8.9]], "longs": LONG}, None, None, False, (), {}),
    ("a quiet graph read near its cost", "quiet_graphs_stay_fast", True, {"reads": [[1, "lookup", 1.0, 1.0, 1.05, 0]]},
     None, None, True, (), {}),
    ("a quiet graph read stalled", "quiet_graphs_stay_fast", False, {"reads": [[1, "lookup", 1.0, 1.0, 1.5, 0]]},
     None, None, True, (), {}),
    ("a driver-limited read is not judged", "quiet_graphs_stay_fast", True, {"reads": [POOL_STALLED, QUIET_FAST]},
     None, None, True, (), {"reads": 1}),
    ("every read driver-limited", "quiet_graphs_stay_fast", None, {"reads": [POOL_STALLED]}, None, None, True, (), {}),
    ("late to send and late for a connection", "quiet_graphs_stay_fast", True,
     {"reads": [QUIET_LATE_TWICE, QUIET_FAST]}, None, None, True, (), {"reads": 1}),
    ("every quiet read timed out", "quiet_graphs_stay_fast", None, {"reads": [[1, "lookup", 1.0, 1.0, 2.0, 1]]},
     None, None, True, (), {}),
    ("a write still waiting for a connection queues nobody", "writers_dont_starve", False,
     {"writes": [WRITE_POOL_STALLED, WRITE_ARRIVED_FIRST]}, None, None, False, (), {}),
    ("every read driver-limited, drain", "requests_drain", None, {"reads": [POOL_STALLED]}, None, None, True, (),
     {"pool_limited_reads": 1}),
    ("no error reply", "no_error_replies", True, {"reads": [[0, "lookup", 1.0, 1.0, 1.01, 0]]}, None, None, False,
     (), {}),
    ("error replies", "no_error_replies", False,
     {"reads": [[0, "lookup", 1.0, 1.0, 1.01, 2]], "writes": [[0, "upsert", 1.0, 1.0, 1.1, 2, "ERR x", []]]},
     None, None, False, (), {"by_op": {"lookup": 1, "upsert": 1}}),
    ("a duplicated key, no constraint", "writes_add_up", False, {}, None, None, False,
     [("duplicate", "g0 L11: 1 duplicated keys")], {}),
    ("no count, no word on timed-out writes", "timed_out_writes_leave_nothing", None, {}, None, None, False,
     [("harness", "end-of-epoch checks failed: ConnectionError")], {}),
    ("a wrong count found before the queries failed", "writes_add_up", False, {}, None, None, False,
     [("count", "g0 L0: 0 nodes, expected 5"), ("harness", "end-of-epoch checks failed: ConnectionError")],
     {}),
    ("a lost edge found before the queries failed", "upsert_edges_complete", False, {}, None, None,
     False, [("missing", "g0 R0: 1 of 1 acknowledged new pairs have no edge"),
             ("harness", "end-of-epoch checks failed: ConnectionError")], {}),
]
# fmt: on


@pytest.mark.parametrize(
    "check_name, want, records, pings, window, quiet, ledger, more",
    [c[1:] for c in ANALYSE_CASES],
    ids=[c[0] for c in ANALYSE_CASES],
)
def test_analyse(check_name, want, records, pings, window, quiet, ledger, more):
    hp, ep = model.control_params((1.0, 0.1) if quiet else (1.0,))
    cal = [{"lookup": 1.0, "filter-sort": 3.0, "distinct": 3.0, "scan-lookup": 200.0}] * hp[
        "graphs"
    ]
    data = {**_epoch_data(**records), **({"window_s": window} if window else {})}
    got = checks.analyse(ep, hp, cal, data, pings or _pings(1000), [], [], {}, list(ledger))[
        "checks"
    ][check_name]
    assert got["pass"] is want and all(got[k] == v for k, v in more.items())


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
    for _s, tt, d in model.PAIRS:
        exact[f"[e:{tt}]"] = n[d]
    exact["WITH DISTINCT p"] = 0  # the acknowledged pair's edge, looked up by its keys
    exact["count(DISTINCT x.k)"] = 1  # the acknowledged new node, looked up by its key
    missing = led.check(FakeGraph(exact))
    assert [k for k, _ in missing] == ["missing", "missing"]
    exact[f"[e:{t}]"] = n[dst] + 1
    exact["WITH DISTINCT p"] = 1
    assert led.check(FakeGraph(exact)) == []
    exact[f"[e:{t}]"] = n[dst] + 2
    assert [k for k, _ in led.check(FakeGraph(exact))] == ["count"]
    # the count is right, but only because an unknown-outcome write added an edge the acknowledged
    # one lost: the pair lookup still finds the loss
    exact[f"[e:{t}]"] = n[dst] + 1
    exact["WITH DISTINCT p"] = 0
    assert [m for k, m in led.check(FakeGraph(exact)) if k == "missing"] == [
        f"g0 {t}: 1 of 1 acknowledged new pairs have no edge"
    ]
    # the same for a node: the label's count is right, but the acknowledged new key has no node
    exact["WITH DISTINCT p"] = 1
    exact["count(DISTINCT x.k)"] = 0
    assert led.check(FakeGraph(exact)) == [
        ("count", f"g0 {dst}: 1 of 1 acknowledged new keys have no node")
    ]


# --- the verdict ----------------------------------------------------------------------------------


def _run(side, i, held, extra=False, died=False, started=True):
    """A control run whose checks all hold except the ones named in `held` as False."""
    cks = {c: {"pass": held.get(c, True)} for c in verdict.CONTROL_CHECKS}
    cks["server_stays_up"] = {"pass": not died}
    return {
        "side": side,
        "index": i,
        "extra": extra,
        "started": started,
        "server_died": died,
        "epoch": {"checks": cks} if started else None,
    }


def _first(cand=None, base=None):
    cand, base = cand or {}, base or {}
    return [_run(s, i, cand if s == "candidate" else base) for i, s in enumerate(check.FIRST_ORDER)]


def _by_name(findings):
    return {f["name"]: f for f in findings}


def _build(side, i, edges, extra=False):
    return {
        "side": side,
        "index": i,
        "extra": extra,
        "ok": True,
        "edges_ms": edges,
        "nodes_ms": 900.0,
        "probe_1_ms": 0.5,
        "probe_100_ms": 0.6,
    }


# --- result.json through the contract's validator ------------------------------------------------------


def _args(**kw):
    return Namespace(
        candidate=DIGEST_A,
        baseline=DIGEST_B,
        candidate_label=kw.get("cl"),
        baseline_label=None,
        candidate_commit=kw.get("cc"),
        baseline_commit=None,
        mode="nightly",
    )


MACHINE = {"runner": "local", "type": "test", "cpu": "test", "cores": 1}


def _results():
    t0 = "2026-11-24T00:00:00Z"
    aa = verdict.control_findings(_first(), False)[0]
    runs = _first(
        cand={"no_timeouts_while_waiting": False},
        base={"no_freeze": False, "writers_dont_starve": False},
    )
    runs += [
        _run("baseline", 4, {"no_freeze": False, "writers_dont_starve": False}, True),
        _run("candidate", 5, {"no_timeouts_while_waiting": False}, True),
    ]
    reg = verdict.control_findings(runs, True)[0]
    slow = [
        _build(s, i, 7900.0 if s == "candidate" else 1300.0)
        for i, s in enumerate(check.FIRST_ORDER)
    ]
    slow += [_build("baseline", 4, 1300.0, True), _build("candidate", 5, 8000.0, True)]
    bf = verdict.build_findings(slow)[0]
    unconfirmed = verdict.control_findings(_first(cand={"requests_drain": False}), False)[0]
    return {
        "pass": check.build_result(
            _args(cl="main@4c10889", cc="4" * 40), t0, aa, [], [], None, MACHINE
        ),
        "regression": check.build_result(_args(), t0, reg + bf, ["note"], [], None, MACHINE),
        "unconfirmed": check.build_result(
            _args(), t0, unconfirmed, [], ["control requests_drain: x"], None, MACHINE
        ),
        "error": check.build_result(
            _args(), t0, [], [], [], "RuntimeError: baseline image: pull failed", MACHINE
        ),
    }


def test_acceptance_patterns_match_the_regression_findings():
    """tests/perf/acceptance.yaml matches fuzz findings by name: `control` and `build`."""
    doc = _results()["regression"]
    regs = [f["name"] for f in doc["findings"] if f["status"] == "regression"]
    assert any(re.search("control", n, re.I) for n in regs) and any(
        re.search("build", n, re.I) for n in regs
    )
    assert doc["confirmed"] is True


def test_the_command_writes_an_error_result_for_an_image_without_a_digest(tmp_path):
    """The one result that cannot be valid: the image it must record is not a digest (merge.py stubs it)."""
    out = tmp_path / "out"
    assert (
        check.main(
            [
                "--candidate",
                "falkordb/falkordb-server:edge",
                "--baseline",
                DIGEST_B,
                "--out",
                str(out),
                "--mode",
                "nightly",
            ]
        )
        == 0
    )
    doc = json.loads((out / "result.json").read_text())
    assert doc["verdict"] == "error" and "digest" in doc["error"]
    _kind, problems = contract.validate_file(out / "result.json", "result")
    assert [p.code for p in problems] == ["image-not-digest"], problems


# --- long mode ------------------------------------------------------------------------------------


def _hour(side, seeds, replay=False):
    """A long-mode server-hour: {seed: {check: held}}; every check not named holds."""
    return {
        "side": side,
        "replay": replay,
        "epochs": {
            s_: {c: held.get(c, True) for c in check.LONG_CHECKS} for s_, held in seeds.items()
        },
    }


def test_sigterm_is_not_swallowed_and_still_writes_a_result(tmp_path, monkeypatch):
    def probe(image):
        try:  # the signal arrives inside a harness `except Exception` block
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)
        except Exception:
            pytest.fail("SIGTERM was swallowed by an `except Exception`")

    monkeypatch.setattr(harness, "probe_image", probe)
    monkeypatch.setattr(kit, "remove_labelled", lambda label: None)
    before = signal.getsignal(signal.SIGTERM)
    out = tmp_path / "out"
    assert (
        check.main(
            ["--candidate", DIGEST_A, "--baseline", DIGEST_B, "--out", str(out), "--mode", "queue"]
        )
        == 0
    )
    doc = json.loads((out / "result.json").read_text())
    assert doc["verdict"] == "error" and "signal" in doc["error"]
    assert signal.getsignal(signal.SIGTERM) == before


def test_a_docker_timeout_is_a_docker_error_not_a_crash(monkeypatch):
    def hang(*a, **kw):
        raise subprocess.TimeoutExpired(a[0], kw.get("timeout"))

    monkeypatch.setattr(harness.subprocess, "run", hang)
    with pytest.raises(kit.DockerError):
        harness.docker("inspect", "x")


def test_a_relative_out_is_resolved_where_the_check_starts(tmp_path, monkeypatch):
    """Contract v1.8: `--out out` is relative; a later chdir must not move result.json. The run's
    containers are removed by its label, after the result is written."""
    start, elsewhere = tmp_path / "start", tmp_path / "elsewhere"
    start.mkdir()
    elsewhere.mkdir()

    def probe(image):
        os.chdir(elsewhere)
        raise kit.DockerError("no docker in this test")

    removed = []

    def remove_labelled(label):
        removed.append((label, (start / "out" / "result.json").exists()))

    monkeypatch.setattr(harness, "probe_image", probe)
    monkeypatch.setattr(kit, "remove_labelled", remove_labelled)
    monkeypatch.chdir(start)
    assert (
        check.main(
            ["--candidate", DIGEST_A, "--baseline", DIGEST_B, "--out", "out", "--mode", "queue"]
        )
        == 0
    )
    doc = json.loads((start / "out" / "result.json").read_text())
    assert doc["verdict"] == "error" and "no docker" in doc["error"]
    assert (start / "out" / "raw" / "rows.csv").exists() and not (elsewhere / "out").exists()
    assert removed == [(f"{check.LABEL}=perf-fuzz-{os.getpid()}", True)]


def test_a_failed_clean_up_is_reported_and_the_result_still_written(tmp_path, monkeypatch, capsys):
    def boom(label):
        raise RuntimeError("docker is gone")

    def probe(image):
        raise RuntimeError("no such image")

    monkeypatch.setattr(harness, "probe_image", probe)
    monkeypatch.setattr(kit, "remove_labelled", boom)
    out = tmp_path / "out"
    assert (
        check.main(
            ["--candidate", DIGEST_A, "--baseline", DIGEST_B, "--out", str(out), "--mode", "queue"]
        )
        == 0
    )
    assert json.loads((out / "result.json").read_text())["verdict"] == "error"
    assert "docker is gone" in capsys.readouterr().err


def _pings(n, late=None, control_gap=None):
    """n PINGs 10 ms apart on both servers; `late` = (index, ms) on FalkorDB; `control_gap` = (from, to)
    seconds with no plain-server replies."""
    f = [(i / 100, 1.0) for i in range(n)]
    c = [
        (i / 100, 1.0)
        for i in range(n)
        if not control_gap or not control_gap[0] <= i / 100 <= control_gap[1]
    ]
    if late:
        f[late[0]] = (late[0] / 100, late[1])
    return {"falkor": f, "control": c}


def test_long_mode_starts_nothing_past_the_time_limit(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(check, "long_hour", lambda *a, **kw: calls.append(("hour", a[3])) or {})
    monkeypatch.setattr(
        repros, "run_repro", lambda *a, **kw: calls.append(("repro", a[2])) or {"pass": True}
    )
    run = check.Run(tmp_path)
    opts = {
        "epoch_s": 300,
        "drain_s": 60,
        "long_scale_range": (0.25, 2.0, 4.0),
        "repros": ["write-create"],
        "deadline": time.time() - 1,
    }
    assert check.long_phase(opts, {}, run, "s", time.time() + 3600) == []
    assert calls == []
    assert any("long hour 1" in u for u in run.unmeasured) and any(
        "repro write-create" in u for u in run.unmeasured
    )


def test_replays_of_one_hour_get_their_own_directories(monkeypatch, tmp_path):
    dirs = []
    monkeypatch.setattr(
        harness,
        "run_server_hour",
        lambda opts, info, tag, hp, eps, hdir, deadline=None: (
            dirs.append(hdir) or {"epochs": [], "build": []}
        ),
    )
    run, opts = check.Run(tmp_path), {"long_scale_range": (0.25, 2.0, 4.0)}
    for e in (1, 2):
        ep = model.epoch_params("s", 1, e, model.hour_params("s", 1)["graphs"])
        check.long_hour(
            opts, {"candidate": {}}, run, "candidate", 1, "s", None, epochs=[ep], replay=True
        )
    assert len(set(dirs)) == 2


def test_a_server_that_dies_mid_hour_is_a_crash_on_the_epochs_it_did_not_finish(
    monkeypatch, tmp_path
):
    """One epoch done, then the server dies (an error with server_alive False): the epoch it did
    not finish is a crash, the one after it never ran, and long mode reports the server did not
    stay up."""
    hp = model.hour_params("s", 1)
    eps = [model.epoch_params("s", 1, e, hp["graphs"]) for e in (1, 2, 3)]
    done = {"params": eps[0], "checks": {c: {"pass": True} for c in check.LONG_CHECKS}}

    def died(opts, info, tag, hp, eps, hdir, deadline=None):
        return {
            "epochs": [done],
            "build": [],
            "error": "ConnectionError: closed",
            "server_alive": False,
        }

    monkeypatch.setattr(harness, "run_server_hour", died)
    run, opts = check.Run(tmp_path), {"long_scale_range": (0.25, 2.0, 4.0)}
    cand = check.long_hour(opts, {"candidate": {}}, run, "candidate", 1, "s", None)
    stayed_up = [cand["epochs"].get(e["seed"], {}).get("server_stays_up") for e in eps]
    assert stayed_up == [True, False, None]
    held = {c: True for c in check.LONG_CHECKS}
    base = {**cand, "side": "baseline", "epochs": {e["seed"]: held for e in eps}}
    f = _by_name(check.long_findings([cand, base], [])[0])["long server_stays_up"]
    assert (f["candidate"], f["baseline"], f["status"]) == (0, 1, "regression")
    assert run.unmeasured == []  # a crash, not a failure to measure


def test_a_stale_hang_marker_is_cleared(monkeypatch, tmp_path):
    class NoDocker:
        def __init__(self, info, name, *a, **kw):
            self.port, self.name = 1, name

        def start(self):
            raise kit.DockerError("no docker here")

        def stop(self):
            return ""

    monkeypatch.setattr(harness, "Server", NoDocker)
    hdir = tmp_path / "h"
    hdir.mkdir()
    (hdir / "HANG").write_text("left by an earlier run\n")
    opts = {"name_prefix": "t", "server_cpus": None, "client_cpus": None, "net": "host"}
    out = harness.run_server_hour(
        opts, {}, "t", {"thread_count": 4, "hour": 1, "scales": []}, [], str(hdir)
    )
    assert not (hdir / "HANG").exists() and out["server_alive"] is None


def test_every_container_carries_the_run_label(monkeypatch, tmp_path):
    """The clean-up removes what a cut run left by its label: every server must carry it."""
    made = []

    class NoDocker:
        def __init__(self, info, name, *a, labels=None, **kw):
            made.append(labels)
            self.port, self.name = 1, name

        def start(self):
            raise kit.DockerError("no docker here")

        def stop(self):
            return ""

    monkeypatch.setattr(harness, "Server", NoDocker)
    labels = {check.LABEL: "run"}
    opts = {
        "name_prefix": "t",
        "labels": labels,
        "server_cpus": None,
        "client_cpus": None,
        "net": "host",
    }
    hour = {"thread_count": 4, "hour": 1, "scales": []}
    # the server and its plain one
    harness.run_server_hour(opts, {}, "h", hour, [], str(tmp_path / "h"))
    build.build_run(opts, {}, "b", 0.01, str(tmp_path / "b"))
    repros.run_repro(opts, {}, "r", "write-create")
    assert made == [labels] * 4


def test_the_pool_wait_is_the_wait_for_a_slot_only(monkeypatch):
    clock = [100.0]  # a fake clock: the test does not depend on how fast the runner is
    monkeypatch.setattr(harness, "now", lambda: clock[0])

    class Base:  # a slot after 0.10 s, then 0.20 s to open the connection (a server round trip)
        async def get_connection(self, *a, **kw):
            clock[0] += 0.10
            await self.ensure_connection("c")
            return "c"

        async def ensure_connection(self, connection):
            clock[0] += 0.20

    waits = {}

    async def main():
        await harness.timed_pool_class(Base, waits)().get_connection("PING")

    asyncio.run(main())
    assert list(waits.values()) == [pytest.approx(0.10)]

    class Refused(Base):  # opening the connection fails: no wait recorded, nothing left behind
        async def ensure_connection(self, connection):
            raise ConnectionError("refused")

    pool = harness.timed_pool_class(Refused, waits)()

    async def refused():
        with pytest.raises(ConnectionError):
            await pool.get_connection("PING")

    asyncio.run(refused())
    assert pool._opening == {}


def test_the_servers_are_stopped_even_when_their_log_cannot_be_written(monkeypatch, tmp_path):
    stopped = []

    class NoDocker:
        def __init__(self, info, name, *a, **kw):
            self.port, self.name = 1, name

        def start(self):
            raise kit.DockerError("no docker here")

        def stop(self):
            stopped.append(self.name)
            return "log"

    monkeypatch.setattr(harness, "Server", NoDocker)
    opts = {"name_prefix": "t", "server_cpus": None, "client_cpus": None, "net": "host"}
    for d in ("b", "h"):  # a directory where the log file should go: writing it fails
        (tmp_path / d / "server-tail.log").mkdir(parents=True)
    with pytest.raises(IsADirectoryError):
        build.build_run(opts, {}, "b", 0.01, str(tmp_path / "b"))
    hour = {"thread_count": 4, "hour": 1, "scales": []}
    with pytest.raises(IsADirectoryError):
        harness.run_server_hour(opts, {}, "h", hour, [], str(tmp_path / "h"))
    assert stopped == ["t-b", "t-h", "t-h-plain"]


def test_a_reply_that_is_not_pong_is_no_ping_sample():
    async def main():
        reader = asyncio.StreamReader()
        reader.feed_data(b"-LOADING the dataset is loading\r\n+PONG\r\n")
        reader.feed_eof()
        f, alive = io.StringIO(), [True]
        await harness._replies(f, "control", reader, collections.deque([1.0, 2.0]), alive)
        return f.getvalue().splitlines(), alive[0]

    lines, alive = asyncio.run(main())
    assert [x.split(",")[:2] for x in lines] == [["control", "2.0000"]] and alive is False


def test_a_driver_result_left_by_an_earlier_attempt_is_removed(tmp_path):
    (tmp_path / "long.json").write_text('{"longs": [], "writes": [[0, "stale"]]}')
    (tmp_path / "async.json").write_text("{}")
    lp, ap = harness.driver_results(str(tmp_path))
    assert (lp, ap) == (str(tmp_path / "long.json"), str(tmp_path / "async.json"))
    assert not os.path.exists(lp) and not os.path.exists(ap)


def test_an_aligned_write_still_waiting_at_the_end_is_unfinished():
    gate, release = harness.SnapshotGate(), threading.Event()
    done = threading.Thread(target=lambda: None)
    stuck = threading.Thread(target=release.wait, daemon=True)
    for t in (done, stuck):
        t.start()
    try:
        assert harness.finish_aligned(gate, [done, stuck], harness.now() + 0.05) == 1
        assert gate.admit(["after the drain"]) is False  # no aligned write starts any more
    finally:
        release.set()


def test_index_waits_after_the_load_share_one_budget(monkeypatch):
    """Three graphs' indexes still building at the load's end: 300 s for all of them, not each."""
    clock, given = [0.0], []

    def wait_indexes(r, graph, timeout_s=300, stop=None):
        given.append(timeout_s)
        clock[0] += min(timeout_s, 200.0)
        return False, min(timeout_s, 200.0)

    monkeypatch.setattr(harness, "now", lambda: clock[0])
    monkeypatch.setattr(harness, "wait_indexes", wait_indexes)
    events = [
        {"event": "indexes_pending_at_epoch_end", "graph": g, "seconds": 1.0} for g in range(3)
    ]
    harness.wait_pending_indexes(None, events, 0.0)
    assert given == [300.0, 100.0, 0.0]
    assert [e["event"] for e in events[3:]] == ["indexes_stuck"] * 3


def test_a_failed_index_comparison_is_unmeasured_and_keeps_the_count(monkeypatch):
    """The ledger finds a wrong count, then the index comparison fails: the count stays a failure,
    and index_matches_scan is not measured (rather than held)."""
    n = model.label_sizes(0.01)

    class Conn:
        def execute_command(self, *a):
            if a[0] == "GRAPH.CONFIG":
                return "OK"
            q = a[2]
            if "+ 0 =" in q:  # the comparison's scan
                raise ConnectionError("the comparison failed")
            if "CALL db.indexes()" in q:
                return [[], [[b"L0", [b"k"], {}, {}, {}, {}, b"NODE", b"OPERATIONAL"]]]
            if "CALL db." in q or "DROP" in q:
                return [[], []]
            return [[], [[0]]]  # every count 0: the graph lost its nodes

        def info(self, section):
            return {"used_memory": 1, "used_memory_rss": 2}

    ep = {**model.control_params((0.01,))[1], "indexes": True}
    out = harness.end_of_epoch(Conn(), ep, 1, [n], checks.Ledger([n]), [], time.monotonic())
    assert out["ivs"] is None and "the comparison failed" in " ".join(out["notes"])
    kinds = {kind for kind, _ in out["ledger_problems"]}
    assert "count" in kinds and "harness" not in kinds
    assert harness.index_check(ep, out["ivs"])["pass"] is None
    assert (
        harness.index_check(ep, [])["pass"] is True
        and harness.index_check(ep, ["x"])["pass"] is False
    )


def test_constraints_cut_short_by_the_epoch_end_are_judged_on_what_was_created(monkeypatch):
    stop, sent = threading.Event(), []

    class Conn:
        def execute_command(self, *a):
            sent.append(a)
            if a[:2] == ("GRAPH.CONSTRAINT", "CREATE"):
                stop.set()  # the epoch ends while the first constraint is being created
            if a[0] == "GRAPH.RO_QUERY" and "db.constraints" in a[2]:
                return [[], [["UNDER CONSTRUCTION"]]]
            return [[], []]

        def close(self):
            pass

    monkeypatch.setattr(harness, "connect", lambda *a, **kw: Conn())
    ep = {**model.control_params((1.0,))[1], "constraints": True}
    info = {}
    harness.epoch_events(0, 0.0, ep, 3, stop, [], time.monotonic(), info)
    assert list(info["states"]) == ["0"] and info["pending_at_epoch_end"]
    # no violation probes
    assert not any(a[0] == "GRAPH.QUERY" and "CREATE (:" in a[2] for a in sent)
    hp = model.control_params((1.0,))[0]
    cal = [{"lookup": 1.0, "filter-sort": 3.0, "distinct": 3.0, "scan-lookup": 200.0}]
    held = checks.analyse(ep, hp, cal, _epoch_data(), _pings(100), [], [], info, [])
    assert held["checks"]["constraints_hold"]["pass"] is False
    none = checks.analyse(ep, hp, cal, _epoch_data(), _pings(100), [], [], {}, [])
    assert none["checks"]["constraints_hold"]["pass"] is None  # created on no graph: not measured


def test_a_failed_docker_inspect_is_not_a_crash(monkeypatch):
    server = harness.Server({"image": "x", "server": "s", "module": "m"}, "n", None, "host")
    reply = {}

    def state(name):
        if isinstance(reply["state"], Exception):
            raise reply["state"]
        return reply["state"]

    monkeypatch.setattr(kit, "state", state)
    reply["state"] = kit.DockerError(
        "Cannot connect to the Docker daemon at unix:///var/run/docker.sock"
    )
    with pytest.raises(kit.DockerError):
        server.state()
    for said, want in (
        ("running", (True, 0)),
        ("gone", (False, -1)),
        ("exited 139", (False, 139)),
        ("exited 137 (OOM-killed)", (False, 137)),
        ("dead 1", (False, 1)),
        ("created 0", (False, 0)),
    ):
        reply["state"] = said
        assert server.state() == want, said


def test_an_aligned_write_is_recorded_or_never_sent(monkeypatch):
    gate = harness.SnapshotGate()
    assert gate.admit(["before"])
    seen = []
    gate.close(lambda records: seen.append(list(records)))
    assert seen == [[["before"]]] and gate.admit(["after"]) is False


def test_schema_commands_stop_when_the_epoch_ends(monkeypatch):
    stop, sent = threading.Event(), []

    class Conn:
        def execute_command(self, *a):
            sent.append(a)
            stop.set()  # the epoch ends while the first index is being created
            return [[], []]

        def close(self):
            pass

    monkeypatch.setattr(harness, "connect", lambda *a, **kw: Conn())
    ep = {**model.control_params((1.0,))[1], "indexes": True}
    events = []
    harness.epoch_events(0, 0.0, ep, 3, stop, events, time.monotonic(), {})
    assert {a[1] for a in sent} == {"soak_g0"}  # nothing for graphs 1 and 2
    # and every graph is left for the wait after the load, graph 0's half-built indexes included
    assert sorted(e["graph"] for e in events if e["event"] == "indexes_pending_at_epoch_end") == [
        0,
        1,
        2,
    ]


def test_clean_up_runs_even_when_the_counting_fails(monkeypatch):
    dropped, schema = [], {"left": 0}

    class Conn:
        def execute_command(self, *a):
            if a[0] == "GRAPH.CONFIG":
                return "OK"
            q = a[2]
            if "DROP INDEX" in q:
                dropped.append(q)
                return [[], []]
            if "CALL db.indexes()" in q:
                return [[], [["x"]] * schema["left"]]
            if "CALL db.constraints()" in q:
                return [[], []]
            raise ConnectionError("the count query failed")

        def info(self, section):
            return {"used_memory": 1, "used_memory_rss": 2}

    n = model.label_sizes(0.01)
    ep = {**model.control_params((0.01,))[1], "indexes": True}
    events = []
    out = harness.end_of_epoch(Conn(), ep, 1, [n], checks.Ledger([n]), events, time.monotonic())
    assert (
        out["ledger_problems"][0][0] == "harness"
        and "the count query failed" in out["ledger_problems"][0][1]
    )
    assert dropped and out["reusable"]
    schema["left"] = 1  # an index survived the clean-up: the server must not run another epoch
    out = harness.end_of_epoch(Conn(), ep, 1, [n], checks.Ledger([n]), events, time.monotonic())
    assert out["reusable"] is False and events[-1]["event"] == "cleanup_incomplete"


def test_a_failed_long_read_is_no_freeze_evidence(monkeypatch):
    class Pinger:
        error = None

        def during(self, a, b):
            return [1.0, 1.0]

    verdict_of = repros._freeze_verdict
    with pytest.raises(repros.Incomplete):
        verdict_of(Pinger(), [(0.0, 1.0, "lost: ConnectionError: closed")], {})
    with pytest.raises(repros.Incomplete):
        verdict_of(Pinger(), [], {})
    assert verdict_of(Pinger(), [(0.0, 30.0, None)], {})["pass"] is True
    # an error reply is an answer: the engine refused the long read, or the command sent during it
    assert verdict_of(Pinger(), [(0.0, 1.0, "reply: ResponseError: x")], {})["pass"] is False
    refused = {"error": "reply: ResponseError: y"}
    assert verdict_of(Pinger(), [(0.0, 30.0, None)], refused)["pass"] is False


def test_a_freeze_repro_needs_its_scenario_to_form():
    """The command must reach the engine during the read, the index must still be building when
    the read starts, and the PING probe must keep running; otherwise the run is incomplete."""

    class Pinger:
        error = None

        def during(self, a, b):
            return [1.0, 1.0]

    read = [(10.0, 40.0, None)]
    got = repros._freeze_verdict(Pinger(), read, {}, sent=11.0)
    assert got["pass"] is True and got["detail"]["sent_into_read_s"] == 1.0
    with pytest.raises(repros.Incomplete, match="after the long read ended"):
        repros._freeze_verdict(Pinger(), read, {}, sent=41.0)
    with pytest.raises(repros.Incomplete, match="index was ready"):
        repros._freeze_verdict(Pinger(), read, {}, building=False)
    stopped = Pinger()
    stopped.error = "lost: ConnectionError: closed"
    with pytest.raises(repros.Incomplete, match="PING probe stopped"):
        repros._freeze_verdict(stopped, read, {})
    # an error reply still fails it: the engine refused the command, whatever the probe saw
    assert repros._freeze_verdict(stopped, read, {"error": "reply: x"}, sent=41.0)["pass"] is False
    probe = repros._Pinger(harness.free_port())  # nothing listens there: its first PING fails
    probe._run()
    assert probe.error.startswith("lost: ConnectionError") and probe.pings == []
    blocked = repros._Pinger(1)  # a PING still waiting for its reply after the read
    blocked.t = threading.Thread(target=threading.Event().wait, daemon=True)
    blocked.t.start()
    blocked.t.join = lambda timeout=None: None
    blocked.__exit__(None, None, None)
    assert blocked.error == "a PING still had no reply after the long read ended"


def test_reads_behind_writer_needs_every_reply(monkeypatch):
    reads = [("before-writer", 5.0, "ok")] * 2 + [("behind-writer", 5.0, "ok")] * 2
    wr, lr = [(1.0, 0.5, None)], [(0.0, 30.0, None)]
    assert repros.judge_reads(reads, wr, lr, 2)["pass"] is True
    assert (
        repros.judge_reads([*reads[:3], ("behind-writer", 1000.0, "timeout")], wr, lr, 2)["pass"]
        is False
    )
    for bad in (
        (reads[:3], wr, lr),
        ([*reads[:3], ("behind-writer", 1.0, "failed: ConnectionError")], wr, lr),
        (reads, [], lr),
        (reads, [(1.0, 0.5, "lost: ConnectionError")], lr),
        (reads, wr, [(0.0, 1.0, "lost: ConnectionError")]),
    ):
        with pytest.raises(repros.Incomplete):
            repros.judge_reads(*bad, 2)
    assert (
        repros.judge_reads(reads, [(1.0, 0.5, "reply: ResponseError: no")], lr, 2)["pass"] is False
    )


def test_writer_starvation_needs_every_client_to_get_replies(monkeypatch):
    writes = [(w, i, 0.1) for w in range(2) for i in range(3)]
    assert repros.judge_writers(writes, [], [], False, 5.0, 6, 6, 6)["pass"] is True
    assert repros.judge_writers(writes, ["write: ERR"], [], False, 5.0, 6, 6, 6)["pass"] is False
    # starved
    assert repros.judge_writers(writes[:5], [], [], True, 120.0, 5, 5, 6)["pass"] is False
    with pytest.raises(repros.Incomplete):
        repros.judge_writers(writes[:5], [], ["write: lost: ConnectionError"], False, 5.0, 5, 5, 6)


# --- the verdict: one repeat rule, the control epoch, build timing, what result.json lists ----------


# fmt: off
JUDGE_REPEAT_CASES = [
    (0, 1, 0, 1, (0, 1, "regression")),    # broke again, the baseline held again
    (0, 1, 1, 1, (1, 1, "info")),          # held on the repeat
    (0, 1, 0, 0, (0, 0, "info")),          # the baseline broke on the repeat: broken on both sides
    (0, 1, 1, 0, (0, 0, "info")),
    (0, 1, None, None, (None, 1, "info")),  # no repeat ran
    (0, 1, 0, None, (None, 1, "info")),    # the baseline's repeat did not measure it
    (0, 1, None, 1, (None, 1, "info")),
    (0, None, 0, 1, (None, None, "info")),  # a break against an unmeasured baseline is not judged
    (1, 1, None, None, (1, 1, "ok")),
    (1, 0, None, None, (1, 0, "improvement")),
    (0, 0, None, None, (0, 0, "info")),
    (1, None, None, None, (1, None, "info")),  # baseline unmeasured: info, never ok
]
# fmt: on


@pytest.mark.parametrize("first_c, first_b, rep_c, rep_b, want", JUDGE_REPEAT_CASES)
def test_judge_repeat(first_c, first_b, rep_c, rep_b, want):
    assert verdict.judge_repeat(first_c, first_b, rep_c, rep_b) == want


def _repeat(cand_first, cand_rep, base_rep=None):
    """Four first runs breaking `cand_first` on the candidate, plus a repeat pair."""
    return [
        *_first(cand=cand_first),
        _run("baseline", 4, base_rep or {}, True),
        _run("candidate", 5, cand_rep, True),
    ]


# fmt: off
CONTROL_CASES = [  # (id, runs, repeat pair ran, check, (candidate, baseline, status))
    ("same image", _first(), False, "no_freeze", (1, 1, "ok")),
    ("repeated", _repeat({"no_freeze": False}, {"no_freeze": False}), True, "no_freeze", (0, 1, "regression")),
    ("held on the repeat", _repeat({"no_freeze": False}, {}), True, "no_freeze", (1, 1, "info")),
    ("baseline broke on the repeat", _repeat({"no_freeze": False}, {"no_freeze": False}, {"no_freeze": False}), True,
     "no_freeze", (0, 0, "info")),
    ("no repeat ran", _first(cand={"no_freeze": False}), False, "no_freeze", (None, 1, "info")),
    ("baseline repeat unmeasured", _repeat({"no_freeze": False}, {"no_freeze": False}, {"no_freeze": None}), True,
     "no_freeze", (None, 1, "info")),
    ("a break seen only in the repeat", _repeat({"no_freeze": False}, {"no_freeze": False, "requests_drain": False}),
     True, "requests_drain", (1, 1, "info")),
    ("data break counts at once", _first(cand={"writes_add_up": False}), False, "writes_add_up", (0, 1, "regression")),
    ("broken on both sides", _first(cand={"writers_dont_starve": False}, base={"writers_dont_starve": False}), False,
     "writers_dont_starve", (0, 0, "info")),
    ("baseline never started", [_run("candidate", 0, {"no_freeze": False}), _run("baseline", 1, {}, started=False),
                                _run("baseline", 2, {}, started=False), _run("candidate", 3, {})], False,
     "no_freeze", (None, None, "info")),
    ("candidate crash", [_run("candidate", 0, {}, died=True, started=False), *_first()[1:]], False,
     "server_stays_up", (0, 1, "regression")),
    ("an unmeasured first run does not borrow the repeat",
     [_run("candidate", 0, {}, started=False), *_first()[1:3], _run("candidate", 3, {"no_freeze": False}),
      _run("baseline", 4, {}, True), _run("candidate", 5, {"requests_drain": False}, True)], True,
     "requests_drain", (None, 1, "info")),
]
# fmt: on


@pytest.mark.parametrize(
    "runs, repeat, check_name, want",
    [c[1:] for c in CONTROL_CASES],
    ids=[c[0] for c in CONTROL_CASES],
)
def test_control_findings(runs, repeat, check_name, want):
    f = _by_name(verdict.control_findings(runs, repeat)[0])[f"control {check_name}"]
    assert (f["candidate"], f["baseline"], f["status"]) == want


# fmt: off
BUILD_THRESHOLD_CASES = [
    (151.0, 100.0, 50.0, 1), (150.0, 100.0, 0.0, 0),    # the ratio must exceed 1.5
    (151.0, 100.0, 51.0, 0),                            # and the difference the floor
    (100.0, 151.0, 50.0, -1), (100.0, 150.0, 0.0, 0),   # improvements mirror it
    (None, 100.0, 0.0, 0),
    (100.0, 0.0, 0.0, 1), (100.0, 0.0, 100.0, 0),     # a zero baseline: the floor alone decides (kit)
]
# fmt: on


@pytest.mark.parametrize("cand, base, floor, want", BUILD_THRESHOLD_CASES)
def test_the_build_threshold_is_strict(cand, base, floor, want):
    assert verdict.crosses(cand, base, floor) == want


# fmt: off
BUILD_REPEAT_CASES = [
    (7900.0, 8000.0, "regression"),   # crossed, and again in the extra pair
    (7900.0, 1350.0, "info"),         # crossed once, not repeated
    (None, None, "ok"),               # under the threshold: no repeat, ok
]
# fmt: on


def test_a_build_row_shows_the_values_that_decided_it():
    """First runs 100 and 8,000 ms against 2,000 and 3,000 (medians 4,050 vs 2,500: crossed), then
    a repeat of 1,000 vs 100 (crossed again): a regression. Pooled, the medians would be 1,000 vs
    2,000 and show the candidate faster; the row shows the first runs' medians instead."""
    first = [
        ("candidate", 100.0),
        ("baseline", 2000.0),
        ("baseline", 3000.0),
        ("candidate", 8000.0),
    ]
    runs = [_build(s, i, ms) for i, (s, ms) in enumerate(first)]
    runs += [_build("baseline", 4, 100.0, True), _build("candidate", 5, 1000.0, True)]
    findings, notes = verdict.build_findings(runs)
    row = _by_name(findings)["build edge phase"]
    assert (row["status"], row["candidate"], row["baseline"]) == ("regression", 4050.0, 2500.0)
    assert row["ratio"] == pytest.approx(1.62) and "repeat 1000 vs 100 ms" in notes[0]


@pytest.mark.parametrize("first_edges, repeat_edges, want", BUILD_REPEAT_CASES)
def test_build_regression_needs_a_repeat(first_edges, repeat_edges, want):
    edges = first_edges or 1400.0
    runs = [
        _build(s, i, edges if s == "candidate" else 1300.0) for i, s in enumerate(check.FIRST_ORDER)
    ]
    assert verdict.build_needs_confirmation(runs) == (["build edge phase"] if first_edges else [])
    if repeat_edges:
        runs += [_build("baseline", 4, 1300.0, True), _build("candidate", 5, repeat_edges, True)]
    assert _by_name(verdict.build_findings(runs)[0])["build edge phase"]["status"] == want


# fmt: off
BUILD_CRASH_CASES = [
    ([{"side": "candidate", "server_died": True}, {"side": "baseline", "ok": False}], (0, None, "regression")),
    ([{"side": "candidate", "server_died": True}, {"side": "baseline", "ok": True}], (0, 1, "regression")),
    ([{"side": "candidate", "ok": True}, {"side": "baseline", "server_died": True}], (1, 0, "improvement")),
    ([{"side": "candidate", "ok": True}, {"side": "baseline", "ok": True}], None),
]
# fmt: on


@pytest.mark.parametrize("runs, want", BUILD_CRASH_CASES)
def test_a_build_crash_is_judged_on_what_each_side_showed(runs, want):
    f = check.build_crash_finding(runs)
    assert (f if f is None else (f["candidate"], f["baseline"], f["status"])) == want


def test_the_worst_findings_come_first_within_a_status():
    """kit.select keeps at most 10 info rows in the order given: a break against a held baseline
    before one held on both, before one broken on both or unmeasured; timings by ratio."""
    rows = [
        kit.finding("both broke", "invariant", 0, 0, "info", direction="lower_is_worse"),
        kit.finding("not measured", "invariant", None, 1, "info", direction="lower_is_worse"),
        kit.finding("held on the repeat", "invariant", 1, 1, "info", direction="lower_is_worse"),
        kit.finding("broke, not repeated", "invariant", 0, 1, "info", direction="lower_is_worse"),
        kit.finding("build 2x", "p50_ms", 2000.0, 1000.0, "regression", threshold=1.5),
        kit.finding("build 5x", "p50_ms", 5000.0, 1000.0, "regression", threshold=1.5),
    ]
    assert [f["name"] for f in verdict.worst_first(rows)] == [
        "build 5x",
        "build 2x",
        "broke, not repeated",
        "held on the repeat",
        "both broke",
        "not measured",
    ]


def test_result_json_lists_no_ok_rows_and_caps_info():
    rows = (
        [
            kit.finding(f"ok {i}", "invariant", 1, 1, "ok", direction="lower_is_worse")
            for i in range(5)
        ]
        + [
            kit.finding(f"info {i}", "invariant", None, 1, "info", direction="lower_is_worse")
            for i in range(12)
        ]
        + [kit.finding("r", "invariant", 0, 1, "regression", direction="lower_is_worse")]
    )
    shown = kit.select(verdict.worst_first(rows))
    assert [f["status"] for f in shown] == ["regression"] + ["info"] * 10


def _shapes():
    """Every kind of result.json the check writes, with the verdict it must have."""
    t0 = "2026-11-24T00:00:00Z"
    data_break = _first(cand={"writes_add_up": False})
    data_break[1] = _run("baseline", 1, {}, started=False)
    long_crash = check.long_findings(
        [
            _hour("candidate", {"x-h1-e1": {"server_stays_up": False}}),
            _hour("baseline", {"x-h1-e1": {}}),
        ],
        [],
    )
    out = {
        kind: (doc, kind if kind != "unconfirmed" else "error") for kind, doc in _results().items()
    }
    out["data break, baseline unmeasured"] = (
        check.build_result(
            _args(), t0, verdict.control_findings(data_break, False)[0], [], ["x"], None, MACHINE
        ),
        "regression",
    )
    out["long mode"] = (
        check.build_result(_args(), t0, long_crash[0], [], long_crash[2], None, MACHINE),
        "regression",
    )
    return out


@pytest.mark.parametrize(
    "kind",
    ["pass", "regression", "unconfirmed", "error", "data break, baseline unmeasured", "long mode"],
)
def test_every_result_shape_is_valid(kind, tmp_path):
    doc, want = _shapes()[kind]
    path = tmp_path / "result.json"
    path.write_text(json.dumps(doc, allow_nan=False))
    assert contract.validate_file(path, "result")[1] == []
    assert doc["verdict"] == want and not any(f["status"] == "ok" for f in doc["findings"])


# --- long mode: seeds paired, replays, repros ----------------------------------------------------------


def _rep(side, seeds):
    return _hour(side, seeds, True)


# fmt: off
LONG_CASES = [  # (id, hours, check, (candidate, baseline, status), something unmeasured)
    ("replay repeats", [_hour("candidate", {"x-h1-e1": {"no_freeze": False}}), _hour("baseline", {"x-h1-e1": {}}),
                        _rep("baseline", {"x-h1-e1": {}}), _rep("candidate", {"x-h1-e1": {"no_freeze": False}})],
     "no_freeze", (0, 1, "regression"), False),
    ("held on replay", [_hour("candidate", {"x-h1-e1": {"no_freeze": False}}), _hour("baseline", {"x-h1-e1": {}}),
                        _rep("baseline", {"x-h1-e1": {}}), _rep("candidate", {"x-h1-e1": {}})],
     "no_freeze", (1, 1, "info"), False),
    ("baseline broke on the replay", [_hour("candidate", {"x-h1-e1": {"no_freeze": False}}),
                                      _hour("baseline", {"x-h1-e1": {}}),
                                      _rep("baseline", {"x-h1-e1": {"no_freeze": False}}),
                                      _rep("candidate", {"x-h1-e1": {"no_freeze": False}})],
     "no_freeze", (0, 0, "info"), False),
    ("not replayed", [_hour("candidate", {"x-h1-e1": {"no_freeze": False}}), _hour("baseline", {"x-h1-e1": {}})],
     "no_freeze", (None, 1, "info"), True),
    ("a confirmed seed wins over one broken on both sides",
     [_hour("candidate", {"x-h1-e1": {"writers_dont_starve": False}, "x-h1-e2": {"writers_dont_starve": False}}),
      _hour("baseline", {"x-h1-e1": {}, "x-h1-e2": {"writers_dont_starve": False}}),
      _rep("baseline", {"x-h1-e1": {}}), _rep("candidate", {"x-h1-e1": {"writers_dont_starve": False}})],
     "writers_dont_starve", (0, 1, "regression"), False),
    ("broken on both sides", [_hour("candidate", {"x-h1-e1": {"writers_dont_starve": False}}),
                              _hour("baseline", {"x-h1-e1": {"writers_dont_starve": False}})],
     "writers_dont_starve", (0, 0, "info"), False),
    ("a crash counts at once", [_hour("candidate", {"x-h1-e1": {"server_stays_up": False}}),
                                _hour("baseline", {"x-h1-e1": {}})], "server_stays_up", (0, 1, "regression"), False),
    ("a data break counts at once", [_hour("candidate", {"x-h1-e1": {"writes_add_up": False}}),
                                     _hour("baseline", {"x-h1-e1": {}})], "writes_add_up", (0, 1, "regression"), False),
    ("a crash on replay counts", [_hour("candidate", {"x-h1-e1": {"no_freeze": False}}),
                                  _hour("baseline", {"x-h1-e1": {}}), _rep("baseline", {"x-h1-e1": {}}),
                                  {"side": "candidate", "replay": True, "epochs": {"x-h1-e1": {
                                      **{c: None for c in check.LONG_CHECKS}, "server_stays_up": False}}}],
     "server_stays_up", (0, 1, "regression"), True),
    ("data break, baseline unmeasured", [_hour("candidate", {"x-h1-e1": {"writes_add_up": False}}),
                                         {"side": "baseline", "replay": False, "epochs": {"x-h1-e1": {
                                             **{c: True for c in check.LONG_CHECKS}, "writes_add_up": None}}}],
     "writes_add_up", (0, None, "regression"), True),
    ("lone crash, nothing paired", [_hour("candidate", {"x-h1-e1": {"server_stays_up": False}})],
     "server_stays_up", (0, None, "regression"), True),
    ("lone crash beside a paired seed", [_hour("candidate", {"x-h1-e1": {}, "x-h2-e1": {"server_stays_up": False}}),
                                         _hour("baseline", {"x-h1-e1": {}})],
     "server_stays_up", (0, None, "regression"), True),
    ("lone seeds that held", [_hour("candidate", {"x-h1-e1": {}})], "writes_add_up", (1, None, "info"), True),
    ("data check unmeasured on the candidate, held on the baseline",
     [_hour("candidate", {"x-h1-e1": {"writes_add_up": None}}), _hour("baseline", {"x-h1-e1": {}})],
     "writes_add_up", (None, 1, "info"), True),
    ("a seed the candidate did not measure", [_hour("candidate", {"x-h1-e1": {"no_freeze": None}}),
                                              _hour("baseline", {"x-h1-e1": {}})],
     "no_freeze", (None, 1, "info"), True),
]
# fmt: on


@pytest.mark.parametrize(
    "hours, check_name, want, unmeasured",
    [c[1:] for c in LONG_CASES],
    ids=[c[0] for c in LONG_CASES],
)
def test_long_findings(hours, check_name, want, unmeasured):
    findings, _notes, missing = check.long_findings(hours, [])
    f = _by_name(findings)[f"long {check_name}"]
    assert (f["candidate"], f["baseline"], f["status"]) == want and bool(missing) == unmeasured
    assert "long memory_steady" not in _by_name(findings)


def test_long_mode_with_nothing_paired_reports_nothing_but_unmeasured():
    findings, _notes, missing = check.long_findings([], [])
    assert findings == [] and missing


def _repro(side, passed, replay=False, crashed=False):
    return {
        "name": "write-create",
        "side": side,
        "replay": replay,
        "pass": passed,
        "crashed": crashed,
    }


# fmt: off
REPRO_CASES = [
    ([_repro("candidate", False), _repro("baseline", True), _repro("baseline", True, True),
      _repro("candidate", False, True)], (0, 1, "regression")),
    ([_repro("candidate", False), _repro("baseline", True), _repro("baseline", True, True),
      _repro("candidate", True, True)], (1, 1, "info")),
    ([_repro("candidate", False), _repro("baseline", True), _repro("baseline", False, True),
      _repro("candidate", False, True)], (0, 0, "info")),
    ([_repro("candidate", False, crashed=True), _repro("baseline", True)], (0, 1, "regression")),
    ([_repro("candidate", False), _repro("baseline", True)], (None, 1, "info")),
    # a crash counts at once against the baseline's crashes, in the repeat too
    ([_repro("candidate", False), _repro("baseline", True), _repro("baseline", None, True),
      _repro("candidate", False, True, crashed=True)], (0, 1, "regression")),
    ([_repro("candidate", False, crashed=True), _repro("baseline", False)], (0, 1, "regression")),
    ([_repro("candidate", False, crashed=True), _repro("baseline", False, crashed=True)], (0, 0, "info")),
]
# fmt: on


@pytest.mark.parametrize("runs, want", REPRO_CASES)
def test_long_repros_use_the_same_repeat_rule(runs, want):
    f = _by_name(check.long_findings([], runs)[0])["long repro write-create"]
    assert (f["candidate"], f["baseline"], f["status"]) == want


# --- images and failures -------------------------------------------------------------------------------


# fmt: off
SERVER_BINARY_CASES = [
    (["PATH=/bin", "FALKORDB_SERVER=valkey"], ["/usr/local/bin/valkey-server", "/usr/local/bin/redis-server"],
     "/usr/local/bin/valkey-server"),
    (["PATH=/bin"], ["/usr/local/bin/valkey-server", "/usr/local/bin/redis-server"], "/usr/local/bin/redis-server"),
    (["FALKORDB_SERVER="], ["/usr/local/bin/redis-server"], "/usr/local/bin/redis-server"),  # empty is unset
    (["FALKORDB_SERVER=valkey"], ["/usr/local/bin/redis-server"], None),  # no fallback to the other binary
]
# fmt: on


@pytest.mark.parametrize("env, has, want", SERVER_BINARY_CASES)
def test_the_server_is_the_images_own_default(monkeypatch, env, has, want):
    def fake_docker(*a, **kw):
        script = a[-1]
        binary = script.split("command -v ")[1].split(";")[0]
        found = [p for p in has if p.endswith("/" + binary)]
        return subprocess.CompletedProcess(
            a, 0, "\n".join([*found, soakkit.IMAGE_MODULE]) + "\n", ""
        )

    soakkit.image_server.cache_clear()  # it caches per image, and every case uses the same one
    monkeypatch.setattr(kit, "pull", lambda image: None)
    monkeypatch.setattr(kit, "image_env", lambda image: dict(e.partition("=")[::2] for e in env))
    monkeypatch.setattr(harness, "docker", fake_docker)
    if want is None:
        with pytest.raises(RuntimeError, match="has no valkey-server"):
            harness.probe_image("x@sha256:" + "a" * 64)
    else:
        assert harness.probe_image("x@sha256:" + "a" * 64)["server"] == want


def test_after_failure_tells_a_crash_from_docker(monkeypatch):
    class S:
        port = 1

        def __init__(self, state):
            self._state = state

        def state(self):
            if isinstance(self._state, Exception):
                raise self._state
            return self._state

    monkeypatch.setattr(harness, "server_hung", lambda port: False)
    assert harness.after_failure(S((False, 139)), RuntimeError("x"))["server_alive"] is False
    assert harness.after_failure(S((True, 0)), RuntimeError("x"))["server_alive"] is True
    assert harness.after_failure(S((True, 0)), kit.DockerError("x"))["server_alive"] is None
    assert (
        harness.after_failure(S(kit.DockerError("no daemon")), RuntimeError("x"))["server_alive"]
        is None
    )
    monkeypatch.setattr(harness, "server_hung", lambda port: True)
    assert harness.after_failure(S((True, 0)), RuntimeError("x"))["hung"] is True


def test_the_image_must_ship_the_module(monkeypatch):
    soakkit.image_server.cache_clear()
    monkeypatch.setattr(kit, "pull", lambda image: None)
    monkeypatch.setattr(kit, "image_env", lambda image: {})
    monkeypatch.setattr(
        harness,
        "docker",
        lambda *a, **kw: subprocess.CompletedProcess(a, 0, "/bin/redis-server\n", ""),
    )
    with pytest.raises(RuntimeError, match="has no " + re.escape(soakkit.IMAGE_MODULE)):
        harness.probe_image("y@sha256:" + "a" * 64)


@pytest.mark.parametrize("net", ["host", "publish"])
def test_the_server_listens_where_its_client_connects(monkeypatch, net):
    """Host network: a free port of 127.0.0.1. Published: CONTAINER_PORT inside, behind the host
    port docker picks. Either way the client connects to the port the server listens behind."""
    started = {}

    def start(image, name, **kw):
        started.update(kw, image=image)
        return kit.Container(name, kw["port"] if kw["host_network"] else 49999)

    monkeypatch.setattr(kit, "start", start)
    monkeypatch.setattr(kit, "wait_ping", lambda port, timeout_s, container: True)
    info = {
        "image": DIGEST_A,
        "server": "/usr/local/bin/valkey-server",
        "module": soakkit.IMAGE_MODULE,
    }
    labels = {check.LABEL: "run"}
    r = (server := harness.Server(info, "n", "0-3", net, 4, labels=labels)).start()
    cmd = started["command"]
    listen, bind = int(cmd[cmd.index("--port") + 1]), cmd[cmd.index("--bind") + 1]
    assert (started["image"], started["entrypoint"], started["labels"], started["cpus"]) == (
        DIGEST_A,
        info["server"],
        labels,
        "0-3",
    )
    assert started["port"] == listen and cmd[cmd.index("--loadmodule") + 1 :][:2] == [
        soakkit.IMAGE_MODULE,
        "THREAD_COUNT",
    ]
    if net == "host":
        assert started["host_network"] and server.port == listen != 6379 and bind == "127.0.0.1"
    else:
        assert (
            not started["host_network"]
            and listen == harness.CONTAINER_PORT
            and server.port == 49999
            and bind == "0.0.0.0"
        )
    assert r.connection_pool.connection_kwargs["port"] == server.port
    plain = harness.Server(info, "p", None, net, module=False)
    plain.start()
    assert "--loadmodule" not in started["command"]


def test_a_server_that_does_not_answer_fails_its_start_with_its_log(monkeypatch):
    """A server that died at start (no host port) fails at once, with no wait for its PONG."""
    pinged = []
    monkeypatch.setattr(kit, "logs", lambda name, tail=None: "# Module could not be loaded")
    monkeypatch.setattr(
        kit, "wait_ping", lambda port, timeout_s, container: pinged.append(port) or False
    )
    info = {"image": DIGEST_A, "server": "s", "module": "m"}
    monkeypatch.setattr(kit, "start", lambda image, name, **kw: kit.Container(name, 0))
    with pytest.raises(RuntimeError, match="did not start: # Module could not be loaded"):
        harness.Server(info, "n", None, "publish").start()
    assert pinged == []
    monkeypatch.setattr(kit, "start", lambda image, name, **kw: kit.Container(name, 5))
    with pytest.raises(RuntimeError, match="did not start"):
        harness.Server(info, "n", None, "publish").start()
    assert pinged == [5]


# --- the clients never re-send ---------------------------------------------------------------------


def _command(f):
    """One RESP command from a client, or None when it closed."""
    line = f.readline()
    if not line:
        return None
    args = []
    for _ in range(int(line[1:])):
        size = int(f.readline()[1:])
        args.append(f.read(size + 2)[:-2])
    return args


class DroppingServer:
    """Answers a connection's set-up (CLIENT SETINFO …), then closes it on the first other command
    without a reply: a client that re-sends shows up as that command more than once. HELLO (a
    RESP3 client) is refused and recorded."""

    def __init__(self):
        self.sock = socket.create_server(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        self.commands: list[bytes] = []
        self.hello = 0
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return  # closed
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        with conn, conn.makefile("rb") as f:
            while (args := _command(f)) is not None:
                if args[0].upper() == b"HELLO":
                    self.hello += 1
                    conn.sendall(b"-ERR unknown command 'HELLO'\r\n")
                    continue
                if args[0].upper() in (b"CLIENT", b"SELECT"):
                    conn.sendall(b"+OK\r\n")
                    continue
                self.commands.append(args[0])
                return

    def close(self):
        self.sock.close()


def test_a_dropped_connection_fails_the_request_and_nothing_is_sent_again():
    """redis-py 8 retries a failed command by default: a slow write that lost its connection would
    be applied again, and a latency would span several attempts. It also asks for RESP3 (HELLO 3)
    by default: both clients speak RESP2, as the acceptance runs did."""
    server = DroppingServer()
    try:
        with pytest.raises(redis.exceptions.ConnectionError):
            harness.connect(server.port).execute_command("GRAPH.QUERY", "g", "CREATE ()")
        assert server.commands == [b"GRAPH.QUERY"]

        async def driver():  # the async driver's own pool and client
            pool = harness.async_pool(
                harness.timed_pool_class(aredis.BlockingConnectionPool, {}), server.port, 10
            )
            try:
                with pytest.raises(redis.exceptions.ConnectionError):
                    await aredis.Redis(connection_pool=pool).execute_command(
                        "GRAPH.RO_QUERY", "g", "RETURN 1"
                    )
            finally:
                await pool.disconnect()

        asyncio.run(driver())
        assert server.commands == [b"GRAPH.QUERY", b"GRAPH.RO_QUERY"] and server.hello == 0
    finally:
        server.close()


def test_replies_are_parsed_by_hiredis():
    """The full scan returns ~100k rows; the pure-Python parser would add seconds to each. The
    fuzz group installs hiredis, and redis-py 8.1 uses only 3.2 or later."""
    assert redis.utils.HIREDIS_AVAILABLE


def test_a_crossing_whose_repeat_did_not_run_is_unmeasured_and_a_zero_baseline_does_not_divide():
    runs = [_build(s, i, 100.0 if s == "baseline" else 4000.0) for i, s in enumerate(check.FIRST_ORDER)]
    runs += [{"side": "baseline", "index": 4, "extra": True, "ok": False}]
    notes = verdict.build_findings(runs)[1]
    assert check.unconfirmed_builds(notes) == ["build edge phase: the confirming repeat was not measured"]
    zero = [_build(s, i, 0.0 if s == "baseline" else 4000.0) for i, s in enumerate(check.FIRST_ORDER)]
    assert verdict.build_findings(zero)[1]  # no ZeroDivisionError


def test_index_probes_follow_the_indexes_the_epoch_installed():
    """A constraint epoch indexes L9-L13: the comparison probes those, not fixed labels, and a graph
    with no operational index to probe is unmeasured (it raises), never a scan agreeing with a scan."""
    asked, rows = [], []

    class Conn:
        def execute_command(self, *a):
            q = a[2]
            if "CALL db.indexes()" in q:
                return [[], rows]
            asked.append(q)
            return [[], [[1]]]

    n = model.label_sizes(0.01)
    rows[:] = [[b"L9", [b"k", b"g"], {}, {}, {}, {}, b"NODE", b"OPERATIONAL"],
               [b"L11", [b"k"], {}, {}, {}, {}, b"NODE", b"UNDER CONSTRUCTION"]]
    assert harness.index_vs_scan(Conn(), 1, [n], "s") == []
    assert asked and all("(n:L9" in q or "n:L9)" in q for q in asked)
    rows.clear()
    with pytest.raises(RuntimeError, match="no operational index"):
        harness.index_vs_scan(Conn(), 1, [n], "s")


def test_a_failed_background_save_is_recorded_as_failed(monkeypatch):
    class Conn:
        def execute_command(self, *a):
            return "Background saving started"

        def info(self, section):
            return {"rdb_bgsave_in_progress": 0, "rdb_last_bgsave_status": status["v"]}

        def close(self):
            pass

    status = {"v": "err"}
    for want in ("bgsave_error", "bgsave_done"):
        events = []
        ep = {**model.control_params((0.01,))[1], "indexes": False, "constraints": False, "bgsave": True}
        monkeypatch.setattr(harness, "connect", lambda *a, **k: Conn())
        harness.epoch_events(1, 0.01, ep, 1, threading.Event(), events, time.monotonic(), {})
        assert [e["event"] for e in events] == [want]
        status["v"] = "ok"


def test_a_repeat_that_measured_zero_is_not_unmeasured():
    runs = [_build(s, i, 100.0 if s == "baseline" else 4000.0) for i, s in enumerate(check.FIRST_ORDER)]
    runs += [_build("baseline", 4, 1.0, True), _build("candidate", 5, 0.0, True)]
    notes = verdict.build_findings(runs)[1]
    assert not any("no repeat ran" in n for n in notes) and check.unconfirmed_builds(notes) == []


# --- fuzz harness: what an epoch found stays found ------------------------------------------------


class _EndOfEpochConn:
    """A server at the end of an epoch: g0 holds an index whose lookups disagree with scans,
    g1 holds none to probe; the count queries answer `count` until `fail_after` of them ran."""

    def __init__(self, count=0, fail_after=None):
        self.count, self.fail_after, self.counted = count, fail_after, 0

    def execute_command(self, *a):
        if a[0] == "GRAPH.CONFIG":
            return "OK"
        graph, q = a[1], a[2]
        if "CALL db.indexes()" in q:
            index = [b"L9", [b"k"], {}, {}, {}, {}, b"NODE", b"OPERATIONAL"]
            return [[], [index] if graph == harness.gname(0) else []]
        if "CALL db." in q or "DROP" in q:
            return [[], []]
        if "+ 0 =" in q:  # the comparison's scan
            return [[], [[2]]]
        if "RETURN count(n)" in q:  # the comparison's index lookup
            return [[], [[1]]]
        self.counted += 1
        if self.fail_after is not None and self.counted > self.fail_after:
            raise ConnectionError("the server went away")
        return [[], [[self.count]]]

    def info(self, section):
        return {"used_memory": 1, "used_memory_rss": 2}


def test_index_mismatches_found_before_the_comparison_failed_still_fail_it():
    n = model.label_sizes(0.01)
    ep = {**model.control_params((0.01,))[1], "indexes": True}
    out = harness.end_of_epoch(
        _EndOfEpochConn(), ep, 2, [n, n], checks.Ledger([n, n]), [], time.monotonic()
    )
    assert out["ivs"] and all(p.endswith(": index 1, scan 2") for p in out["ivs"])
    assert "g1 has no operational index to probe" in " ".join(out["notes"])
    assert harness.index_check(ep, out["ivs"])["pass"] is False


def test_ledger_problems_found_before_a_failed_query_stay():
    n = model.label_sizes(0.01)
    ep = {**model.control_params((0.01,))[1], "indexes": False, "constraints": False}
    conn = _EndOfEpochConn(count=0, fail_after=1)  # the first label's count is wrong, then it fails
    out = harness.end_of_epoch(conn, ep, 1, [n], checks.Ledger([n]), [], time.monotonic())
    assert [kind for kind, _ in out["ledger_problems"]] == ["count", "harness"]
    assert "the server went away" in out["ledger_problems"][1][1]


def test_a_failure_found_after_a_lost_driver_file_stands():
    found = {
        "writes_add_up": {"pass": False, "problems": ["g0 L0: 1 duplicated keys"]},
        "upsert_edges_complete": {"pass": True, "problems": []},
        "timed_out_writes_leave_nothing": {"pass": None, "problems": []},
    }
    harness.counts_unknown(found)
    assert found["writes_add_up"] == {"pass": False, "problems": ["g0 L0: 1 duplicated keys"]}
    for k in ("upsert_edges_complete", "timed_out_writes_leave_nothing"):
        assert found[k]["pass"] is None and "result file was lost" in found[k]["skipped"]


def _bgsave_events(info, stop, monkeypatch, clock=None):
    class Conn:
        def execute_command(self, *a):
            return "Background saving started"

        def info(self, section):
            return info

    if clock is not None:
        monkeypatch.setattr(harness, "now", lambda: next(clock))
    events = []
    harness.bgsave(Conn(), stop, lambda kind, **kw: events.append((kind, kw.get("why"))))
    return events


def test_a_background_save_still_running_when_the_wait_ends_is_unfinished(monkeypatch):
    saving = {"rdb_bgsave_in_progress": 1, "rdb_last_bgsave_status": "ok"}  # the previous save's
    stopped = threading.Event()
    stopped.set()
    assert _bgsave_events(saving, stopped, monkeypatch) == [
        ("bgsave_unfinished", "the epoch ended")
    ]
    clock = itertools.chain([0.0, 0.0], itertools.repeat(301.0))  # one poll, then past 300 s
    assert _bgsave_events(saving, _NeverSet(), monkeypatch, clock) == [
        ("bgsave_stuck", "still saving after 300 s")
    ]
    # the epoch ends just after the 300 s limit did: the limit ended the wait, so stuck
    late_stop = types.SimpleNamespace(wait=lambda timeout: False, is_set=lambda: True)
    clock = itertools.chain([0.0, 0.0], itertools.repeat(301.0))
    assert _bgsave_events(saving, late_stop, monkeypatch, clock) == [
        ("bgsave_stuck", "still saving after 300 s")
    ]
    done = {"rdb_bgsave_in_progress": 0, "rdb_last_bgsave_status": "ok"}
    assert _bgsave_events(done, threading.Event(), monkeypatch) == [("bgsave_done", None)]


class _NeverSet:
    """A stop event the epoch never sets, whose wait returns at once."""

    def wait(self, timeout):
        return False

    def is_set(self):
        return False


@pytest.mark.parametrize("event, fails", [("bgsave_stuck", True), ("bgsave_unfinished", False)])
def test_a_stuck_background_save_is_an_unexpected_error(event, fails):
    """Still saving after 300 s fails no_unexpected_errors; a save the epoch's end cut short
    is the harness's doing, not the server's."""
    hp, ep = model.control_params((1.0,))
    cal = [{"lookup": 1.0, "filter-sort": 3.0, "distinct": 3.0, "scan-lookup": 200.0}] * hp[
        "graphs"
    ]
    events = [{"t": 1.0, "event": event, "seconds": 300.0}]
    got = checks.analyse(ep, hp, cal, _epoch_data(), _pings(1000), [], events, {}, [])
    assert got["checks"]["no_unexpected_errors"]["pass"] is not fails
