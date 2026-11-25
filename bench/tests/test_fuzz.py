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


LONG = [[0, "full-scan", 10.0, 18.0, 0, ""]]  # a long read from 10 s to 18 s
POOL_STALLED = [1, "lookup", 1.0, 1.0, 3.0, 0, 1.99]  # 2 s on a quiet graph, all of it waiting for a connection
QUIET_FAST = [1, "lookup", 2.0, 2.0, 2.01, 0, 0.0]


def _pings(n, late=None, control_gap=None):
    """n PINGs 10 ms apart on both servers; `late` = (index, ms) on FalkorDB; `control_gap` = (from, to)
    seconds with no plain-server replies."""
    f = [(i / 100, 1.0) for i in range(n)]
    c = [(i / 100, 1.0) for i in range(n) if not control_gap or not control_gap[0] <= i / 100 <= control_gap[1]]
    if late:
        f[late[0]] = (late[0] / 100, late[1])
    return {"falkor": f, "control": c}


ANALYSE_CASES = [  # (id, check, pass, records, pings, window_s, quiet, ledger problems, more fields to assert)
    ("a slow read may time out", "no_timeouts_while_waiting", True,
     {"reads": [[0, "lookup", 1.0, 1.0, 1.01, 0], [0, "scan-lookup", 2.0, 2.0, 2.5, 1]]}, None, None, False, (),
     {"all_read_timeouts": 1}),
    ("a cheap read may not", "no_timeouts_while_waiting", False, {"reads": [[0, "lookup", 1.0, 1.0, 2.0, 1]]},
     None, None, False, (), {}),
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
    ("every read driver-limited, drain", "requests_drain", None, {"reads": [POOL_STALLED]}, None, None, True, (),
     {"pool_limited_reads": 1}),
    ("no error reply", "no_error_replies", True, {"reads": [[0, "lookup", 1.0, 1.0, 1.01, 0]]}, None, None, False,
     (), {}),
    ("error replies", "no_error_replies", False,
     {"reads": [[0, "lookup", 1.0, 1.0, 1.01, 2]], "writes": [[0, "upsert", 1.0, 1.0, 1.1, 2, "ERR x", []]]},
     None, None, False, (), {"by_op": {"lookup": 1, "upsert": 1}}),
    ("a duplicated key, no constraint", "writes_add_up", False, {}, None, None, False,
     [("duplicate", "g0 L11: 1 duplicated keys")], {}),
]


@pytest.mark.parametrize("check_name, want, records, pings, window, quiet, ledger, more",
                         [c[1:] for c in ANALYSE_CASES], ids=[c[0] for c in ANALYSE_CASES])
def test_analyse(check_name, want, records, pings, window, quiet, ledger, more):
    hp, ep = model.control_params((1.0, 0.1) if quiet else (1.0,))
    cal = [{"lookup": 1.0, "filter-sort": 3.0, "distinct": 3.0, "scan-lookup": 200.0}] * hp["graphs"]
    data = {**_epoch_data(**records), **({"window_s": window} if window else {})}
    got = checks.analyse(ep, hp, cal, data, pings or _pings(1000), [], [], {}, list(ledger))["checks"][check_name]
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
    for s_, tt, d in model.PAIRS:
        exact[f"[e:{tt}]"] = n[d]
    exact["WITH DISTINCT p"] = 0  # the acknowledged pair's edge, looked up by its keys
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
        f"g0 {t}: 1 of 1 acknowledged new pairs have no edge"]


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


def _build(side, i, edges, extra=False):
    return {"side": side, "index": i, "extra": extra, "ok": True, "edges_ms": edges, "nodes_ms": 900.0,
            "probe_1_ms": 0.5, "probe_100_ms": 0.6}


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


def test_acceptance_patterns_match_the_regression_findings():
    """tests/perf/acceptance.yaml matches fuzz findings by name: `control` and `build`."""
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


# --- long mode ------------------------------------------------------------------------------------


def _hour(side, seeds, replay=False):
    """A long-mode server-hour: {seed: {check: held}}; every check not named holds."""
    return {"side": side, "replay": replay,
            "epochs": {s_: {c: held.get(c, True) for c in check.LONG_CHECKS} for s_, held in seeds.items()}}


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


def _pings(n, late=None, control_gap=None):
    """n PINGs 10 ms apart on both servers; `late` = (index, ms) on FalkorDB; `control_gap` = (from, to)
    seconds with no plain-server replies."""
    f = [(i / 100, 1.0) for i in range(n)]
    c = [(i / 100, 1.0) for i in range(n) if not control_gap or not control_gap[0] <= i / 100 <= control_gap[1]]
    if late:
        f[late[0]] = (late[0] / 100, late[1])
    return {"falkor": f, "control": c}


def test_long_mode_starts_nothing_past_the_time_limit(monkeypatch, tmp_path):
    _stub_redis(monkeypatch)
    from soakfuzz import repros
    calls = []
    monkeypatch.setattr(check, "long_hour", lambda *a, **kw: calls.append(("hour", a[3])) or {})
    monkeypatch.setattr(repros, "run_repro", lambda *a, **kw: calls.append(("repro", a[2])) or {"pass": True})
    run = check.Run(tmp_path)
    opts = {"epoch_s": 300, "drain_s": 60, "long_scale_range": (0.25, 2.0, 4.0), "repros": ["write-create"],
            "deadline": time.time() - 1}
    assert check.long_phase(opts, {}, run, "s", time.time() + 3600) == []
    assert calls == []
    assert any("long hour 1" in u for u in run.unmeasured) and any("repro write-create" in u for u in run.unmeasured)


def test_replays_of_one_hour_get_their_own_directories(monkeypatch, tmp_path):
    _stub_redis(monkeypatch)
    from soakfuzz import harness
    dirs = []
    monkeypatch.setattr(harness, "run_server_hour", lambda opts, info, tag, hp, eps, hdir, deadline=None:
                        dirs.append(hdir) or {"epochs": [], "build": []})
    run, opts = check.Run(tmp_path), {"long_scale_range": (0.25, 2.0, 4.0)}
    for e in (1, 2):
        ep = model.epoch_params("s", 1, e, model.hour_params("s", 1)["graphs"])
        check.long_hour(opts, {"candidate": {}}, run, "candidate", 1, "s", None, epochs=[ep], replay=True)
    assert len(set(dirs)) == 2


def test_a_stale_hang_marker_is_cleared(monkeypatch, tmp_path):
    _stub_redis(monkeypatch)
    from soakfuzz import harness

    class NoDocker:
        def __init__(self, info, name, *a, **kw):
            self.port, self.name = 1, name

        def start(self):
            raise harness.DockerError("no docker here")

        def stop(self):
            return ""

    monkeypatch.setattr(harness, "Server", NoDocker)
    hdir = tmp_path / "h"
    hdir.mkdir()
    (hdir / "HANG").write_text("left by an earlier run\n")
    opts = {"name_prefix": "t", "server_cpus": None, "client_cpus": None, "net": "host"}
    out = harness.run_server_hour(opts, {}, "t", {"thread_count": 4, "hour": 1, "scales": []}, [], str(hdir))
    assert not (hdir / "HANG").exists() and out["server_alive"] is None


def test_the_pool_wait_is_the_wait_for_a_slot_only(monkeypatch):
    _stub_redis(monkeypatch)
    import asyncio
    from soakfuzz import harness

    class Base:  # a slot after 0.10 s, then 0.20 s to open the connection (a server round trip)
        async def get_connection(self, *a, **kw):
            await asyncio.sleep(0.10)
            await self.ensure_connection("c")
            return "c"

        async def ensure_connection(self, connection):
            await asyncio.sleep(0.20)

    waits = {}

    async def main():
        await harness.timed_pool_class(Base, waits)().get_connection("PING")

    asyncio.run(main())
    (wait,) = waits.values()
    assert 0.08 < wait < 0.18

    class Refused(Base):  # opening the connection fails: no wait recorded, nothing left behind
        async def ensure_connection(self, connection):
            raise ConnectionError("refused")

    pool = harness.timed_pool_class(Refused, waits)()

    async def refused():
        with pytest.raises(ConnectionError):
            await pool.get_connection("PING")

    asyncio.run(refused())
    assert pool._opening == {}


def test_constraints_cut_short_by_the_epoch_end_are_judged_on_what_was_created(monkeypatch):
    _stub_redis(monkeypatch)
    import threading
    from soakfuzz import harness
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
    assert not any(a[0] == "GRAPH.QUERY" and "CREATE (:" in a[2] for a in sent)  # no violation probes
    hp = model.control_params((1.0,))[0]
    cal = [{"lookup": 1.0, "filter-sort": 3.0, "distinct": 3.0, "scan-lookup": 200.0}]
    held = checks.analyse(ep, hp, cal, _epoch_data(), _pings(100), [], [], info, [])
    assert held["checks"]["constraints_hold"]["pass"] is False
    none = checks.analyse(ep, hp, cal, _epoch_data(), _pings(100), [], [], {}, [])
    assert none["checks"]["constraints_hold"]["pass"] is None  # created on no graph: not measured


def test_a_failed_docker_inspect_is_not_a_crash(monkeypatch):
    _stub_redis(monkeypatch)
    from soakfuzz import harness
    server = harness.Server({"image": "x", "server": "s", "module": "m"}, "n", None, "host")
    reply = {}
    monkeypatch.setattr(harness, "docker", lambda *a, **kw: subprocess.CompletedProcess(a, reply["rc"], reply["out"], reply["err"]))
    reply.update(rc=1, out="", err="Cannot connect to the Docker daemon at unix:///var/run/docker.sock")
    with pytest.raises(harness.DockerError):
        server.state()
    reply.update(rc=1, out="", err="Error: No such object: n")
    assert server.state() == (False, -1)
    reply.update(rc=1, out="", err="error during connect: dial tcp: lookup docker-host: no such host")
    with pytest.raises(harness.DockerError):
        server.state()
    reply.update(rc=0, out="false 139\n", err="")
    assert server.state() == (False, 139)


def test_an_aligned_write_is_recorded_or_never_sent(monkeypatch):
    _stub_redis(monkeypatch)
    from soakfuzz import harness
    gate = harness.SnapshotGate()
    assert gate.admit(["before"])
    seen = []
    gate.close(lambda records: seen.append(list(records)))
    assert seen == [[["before"]]] and gate.admit(["after"]) is False


def test_schema_commands_stop_when_the_epoch_ends(monkeypatch):
    _stub_redis(monkeypatch)
    import threading
    from soakfuzz import harness
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
    assert sorted(e["graph"] for e in events if e["event"] == "indexes_pending_at_epoch_end") == [0, 1, 2]


def test_clean_up_runs_even_when_the_counting_fails(monkeypatch):
    _stub_redis(monkeypatch)
    from soakfuzz import harness
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
    assert out["ledger_problems"][0][0] == "harness" and "the count query failed" in out["ledger_problems"][0][1]
    assert dropped and out["reusable"]
    schema["left"] = 1  # an index survived the clean-up: the server must not run another epoch
    out = harness.end_of_epoch(Conn(), ep, 1, [n], checks.Ledger([n]), events, time.monotonic())
    assert out["reusable"] is False and events[-1]["event"] == "cleanup_incomplete"


def test_a_failed_long_read_is_no_freeze_evidence(monkeypatch):
    _stub_redis(monkeypatch)
    from soakfuzz import repros

    class Pinger:
        def during(self, a, b):
            return [1.0, 1.0]

    with pytest.raises(repros.Incomplete):
        repros._freeze_verdict(Pinger(), [(0.0, 1.0, "lost: ConnectionError: closed")], {})
    with pytest.raises(repros.Incomplete):
        repros._freeze_verdict(Pinger(), [], {})
    assert repros._freeze_verdict(Pinger(), [(0.0, 30.0, None)], {})["pass"] is True
    # an error reply is an answer: the engine refused the long read, or the command sent during it
    assert repros._freeze_verdict(Pinger(), [(0.0, 1.0, "reply: ResponseError: x")], {})["pass"] is False
    assert repros._freeze_verdict(Pinger(), [(0.0, 30.0, None)], {"error": "reply: ResponseError: y"})["pass"] is False


def test_reads_behind_writer_needs_every_reply(monkeypatch):
    _stub_redis(monkeypatch)
    from soakfuzz import repros
    reads = [("before-writer", 5.0, "ok")] * 2 + [("behind-writer", 5.0, "ok")] * 2
    wr, lr = [(1.0, 0.5, None)], [(0.0, 30.0, None)]
    assert repros.judge_reads(reads, wr, lr, 2)["pass"] is True
    assert repros.judge_reads(reads[:3] + [("behind-writer", 1000.0, "timeout")], wr, lr, 2)["pass"] is False
    for bad in ((reads[:3], wr, lr), (reads[:3] + [("behind-writer", 1.0, "failed: ConnectionError")], wr, lr),
                (reads, [], lr), (reads, [(1.0, 0.5, "lost: ConnectionError")], lr),
                (reads, wr, [(0.0, 1.0, "lost: ConnectionError")])):
        with pytest.raises(repros.Incomplete):
            repros.judge_reads(*bad, 2)
    assert repros.judge_reads(reads, [(1.0, 0.5, "reply: ResponseError: no")], lr, 2)["pass"] is False


def test_writer_starvation_needs_every_client_to_get_replies(monkeypatch):
    _stub_redis(monkeypatch)
    from soakfuzz import repros
    writes = [(w, i, 0.1) for w in range(2) for i in range(3)]
    assert repros.judge_writers(writes, [], [], False, 5.0, 6, 6, 6)["pass"] is True
    assert repros.judge_writers(writes, ["write: ERR"], [], False, 5.0, 6, 6, 6)["pass"] is False
    assert repros.judge_writers(writes[:5], [], [], True, 120.0, 5, 5, 6)["pass"] is False  # starved
    with pytest.raises(repros.Incomplete):
        repros.judge_writers(writes[:5], [], ["write: lost: ConnectionError"], False, 5.0, 5, 5, 6)


# --- the verdict: one repeat rule, the control epoch, build timing, what result.json lists ----------


@pytest.mark.parametrize("first_c, first_b, rep_c, rep_b, want", [
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
])
def test_judge_repeat(first_c, first_b, rep_c, rep_b, want):
    assert verdict.judge_repeat(first_c, first_b, rep_c, rep_b) == want


def _repeat(cand_first, cand_rep, base_rep=None):
    """Four first runs breaking `cand_first` on the candidate, plus a repeat pair."""
    return (_first(cand=cand_first)
            + [_run("baseline", 4, base_rep or {}, True), _run("candidate", 5, cand_rep, True)])


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
    ("candidate crash", [_run("candidate", 0, {}, died=True, started=False)] + _first()[1:], False,
     "server_stays_up", (0, 1, "regression")),    ("an unmeasured first run does not borrow the repeat",
     [_run("candidate", 0, {}, started=False), *_first()[1:3], _run("candidate", 3, {"no_freeze": False}),
      _run("baseline", 4, {}, True), _run("candidate", 5, {"requests_drain": False}, True)], True,
     "requests_drain", (None, 1, "info")),
]


@pytest.mark.parametrize("runs, repeat, check_name, want", [c[1:] for c in CONTROL_CASES],
                         ids=[c[0] for c in CONTROL_CASES])
def test_control_findings(runs, repeat, check_name, want):
    f = _by_name(verdict.control_findings(runs, repeat)[0])[f"control {check_name}"]
    assert (f["candidate"], f["baseline"], f["status"]) == want


@pytest.mark.parametrize("cand, base, floor, want", [
    (151.0, 100.0, 50.0, 1), (150.0, 100.0, 0.0, 0),    # the ratio must exceed 1.5
    (151.0, 100.0, 51.0, 0),                            # and the difference the floor
    (100.0, 151.0, 50.0, -1), (100.0, 150.0, 0.0, 0),   # improvements mirror it
    (None, 100.0, 0.0, 0), (100.0, 0.0, 0.0, 0),
])
def test_the_build_threshold_is_strict(cand, base, floor, want):
    assert verdict.crosses(cand, base, floor) == want


@pytest.mark.parametrize("first_edges, repeat_edges, want", [
    (7900.0, 8000.0, "regression"),   # crossed, and again in the extra pair
    (7900.0, 1350.0, "info"),         # crossed once, not repeated
    (None, None, "ok"),               # under the threshold: no repeat, ok
])
def test_build_regression_needs_a_repeat(first_edges, repeat_edges, want):
    edges = first_edges or 1400.0
    runs = [_build(s, i, edges if s == "candidate" else 1300.0) for i, s in enumerate(check.FIRST_ORDER)]
    assert verdict.build_needs_confirmation(runs) == (["build edge phase"] if first_edges else [])
    if repeat_edges:
        runs += [_build("baseline", 4, 1300.0, True), _build("candidate", 5, repeat_edges, True)]
    assert _by_name(verdict.build_findings(runs)[0])["build edge phase"]["status"] == want


@pytest.mark.parametrize("runs, want", [
    ([{"side": "candidate", "server_died": True}, {"side": "baseline", "ok": False}], (0, None, "regression")),
    ([{"side": "candidate", "server_died": True}, {"side": "baseline", "ok": True}], (0, 1, "regression")),
    ([{"side": "candidate", "ok": True}, {"side": "baseline", "server_died": True}], (1, 0, "improvement")),
    ([{"side": "candidate", "ok": True}, {"side": "baseline", "ok": True}], None),
])
def test_a_build_crash_is_judged_on_what_each_side_showed(runs, want):
    f = check.build_crash_finding(runs)
    assert (f if f is None else (f["candidate"], f["baseline"], f["status"])) == want


def test_result_json_lists_no_ok_rows_and_caps_info():
    rows = ([{"name": f"ok {i}", "status": "ok", "ratio": 1.0, "direction": "lower_is_worse"} for i in range(5)]
            + [{"name": f"info {i}", "status": "info", "ratio": None, "direction": "lower_is_worse"} for i in range(12)]
            + [{"name": "r", "status": "regression", "ratio": 0.0, "direction": "lower_is_worse"}])
    shown = verdict.select(rows)
    assert [f["status"] for f in shown] == ["regression"] + ["info"] * 10


def _shapes():
    """Every kind of result.json the check writes, with the verdict it must have."""
    t0, t1 = "2026-11-24T00:00:00Z", "2026-11-24T04:00:00Z"
    data_break = _first(cand={"writes_add_up": False})
    data_break[1] = _run("baseline", 1, {}, started=False)
    long_crash = check.long_findings([_hour("candidate", {"x-h1-e1": {"server_stays_up": False}}),
                                      _hour("baseline", {"x-h1-e1": {}})], [])
    out = {kind: (doc, kind if kind != "unconfirmed" else "error") for kind, doc in _results().items()}
    out["data break, baseline unmeasured"] = (
        check.build_result(_args(), t0, t1, verdict.control_findings(data_break, False)[0], [], ["x"], None, MACHINE),
        "regression")
    out["long mode"] = (check.build_result(_args(), t0, t1, long_crash[0], [], long_crash[2], None, MACHINE),
                        "regression")
    return out


@pytest.mark.parametrize("kind", ["pass", "regression", "unconfirmed", "error", "data break, baseline unmeasured",
                                  "long mode"])
def test_every_result_shape_is_valid(kind, tmp_path):
    validate = _validator()
    doc, want = _shapes()[kind]
    path = tmp_path / "result.json"
    path.write_text(json.dumps(doc, allow_nan=False))
    assert validate.validate_file(path, "result")[1] == []
    assert doc["verdict"] == want and not any(f["status"] == "ok" for f in doc["findings"])


# --- long mode: seeds paired, replays, repros ----------------------------------------------------------


def _rep(side, seeds):
    return _hour(side, seeds, True)


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
]


@pytest.mark.parametrize("hours, check_name, want, unmeasured", [c[1:] for c in LONG_CASES],
                         ids=[c[0] for c in LONG_CASES])
def test_long_findings(hours, check_name, want, unmeasured):
    findings, _notes, missing = check.long_findings(hours, [])
    f = _by_name(findings)[f"long {check_name}"]
    assert (f["candidate"], f["baseline"], f["status"]) == want and bool(missing) == unmeasured
    assert "long memory_steady" not in _by_name(findings)


def test_long_mode_with_nothing_paired_reports_nothing_but_unmeasured():
    findings, _notes, missing = check.long_findings([], [])
    assert findings == [] and missing


def _repro(side, passed, replay=False, crashed=False):
    return {"name": "write-create", "side": side, "replay": replay, "pass": passed, "crashed": crashed}


@pytest.mark.parametrize("runs, want", [
    ([_repro("candidate", False), _repro("baseline", True), _repro("baseline", True, True),
      _repro("candidate", False, True)], (0, 1, "regression")),
    ([_repro("candidate", False), _repro("baseline", True), _repro("baseline", True, True),
      _repro("candidate", True, True)], (1, 1, "info")),
    ([_repro("candidate", False), _repro("baseline", True), _repro("baseline", False, True),
      _repro("candidate", False, True)], (0, 0, "info")),
    ([_repro("candidate", False, crashed=True), _repro("baseline", True)], (0, 1, "regression")),
    ([_repro("candidate", False), _repro("baseline", True)], (None, 1, "info")),
])
def test_long_repros_use_the_same_repeat_rule(runs, want):
    f = _by_name(check.long_findings([], runs)[0])["long repro write-create"]
    assert (f["candidate"], f["baseline"], f["status"]) == want


# --- images and failures -------------------------------------------------------------------------------


@pytest.mark.parametrize("env, has, want", [
    (["PATH=/bin", "FALKORDB_SERVER=valkey"], ["/usr/local/bin/valkey-server", "/usr/local/bin/redis-server"],
     "/usr/local/bin/valkey-server"),
    (["PATH=/bin"], ["/usr/local/bin/valkey-server", "/usr/local/bin/redis-server"], "/usr/local/bin/redis-server"),
    (["FALKORDB_SERVER="], ["/usr/local/bin/redis-server"], "/usr/local/bin/redis-server"),  # empty is unset
    (["FALKORDB_SERVER=valkey"], ["/usr/local/bin/redis-server"], None),  # no fallback to the other binary
])
def test_the_server_is_the_images_own_default(monkeypatch, env, has, want):
    _stub_redis(monkeypatch)
    from soakfuzz import harness

    def fake_docker(*a, **kw):
        if a[0] == "pull":
            return subprocess.CompletedProcess(a, 0, "", "")
        if a[:2] == ("image", "inspect"):
            return subprocess.CompletedProcess(a, 0, json.dumps(env), "")
        script = a[-1]
        binary = script.split("command -v ")[1].split(";")[0]
        found = [p for p in has if p.endswith("/" + binary)]
        return subprocess.CompletedProcess(a, 0, "\n".join(found + [harness.MODULE_PATH]) + "\n", "")

    monkeypatch.setattr(harness, "docker", fake_docker)
    if want is None:
        with pytest.raises(RuntimeError, match="has no valkey-server"):
            harness.probe_image("x@sha256:" + "a" * 64)
    else:
        assert harness.probe_image("x@sha256:" + "a" * 64)["server"] == want


def test_after_failure_tells_a_crash_from_docker(monkeypatch):
    _stub_redis(monkeypatch)
    from soakfuzz import harness

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
    assert harness.after_failure(S((True, 0)), harness.DockerError("x"))["server_alive"] is None
    assert harness.after_failure(S(harness.DockerError("no daemon")), RuntimeError("x"))["server_alive"] is None
    monkeypatch.setattr(harness, "server_hung", lambda port: True)
    assert harness.after_failure(S((True, 0)), RuntimeError("x"))["hung"] is True
