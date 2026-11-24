"""`bench probe`'s rules, without a server: RESP parsing, the growth ratio, the
re-run rule, correctness findings and the contract's limits.

When the perf contract's validator is present (tests/perf/validate.py, from
#3986), every result built here is also checked against it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from falkorbench import probe
from falkorbench.probe import SHAPES
from falkorbench.probe import Cell
from falkorbench.probe import Reply
from falkorbench.probe import RespReader
from falkorbench.probe import Setup

SIZES = (10_000, 100_000, 500_000)
IMAGE_A = "ghcr.io/falkordb/falkordb-server@sha256:" + "a" * 64
IMAGE_B = "falkordb/falkordb-server@sha256:" + "b" * 64
VALIDATOR = Path(__file__).resolve().parents[2] / "tests" / "perf" / "validate.py"
UNWIND_ID_1 = "UNWIND id MATCH edge 1 row"


def _validator():
    if not VALIDATOR.exists():
        return None
    spec = importlib.util.spec_from_file_location("perf_validate", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- building cells ----------------------------------------------------------


def cell(shape, size, cand_ms, base_ms, phase="screen", n=20, counts=None):
    """A unit whose candidate and baseline answered `n` times at these latencies."""
    c = Cell(shape, size, phase, "candidate")
    for side, ms in (("candidate", cand_ms), ("baseline", base_ms)):
        run = c.runs[side]
        run.wall = [ms] * n
        run.server = [ms * 0.9] * n
        run.counts = dict(counts[side]) if counts else {k: 1 for k in range(n)}
    return c


def grid(latency, phase="screen", shapes=SHAPES, sizes=SIZES):
    """Every (shape, size) cell; latency(shape, size) -> (candidate ms, baseline ms)."""
    return {
        (s.name, z): cell(s.name, z, *latency(s.name, z), phase=phase)
        for s in shapes
        for z in sizes
    }


def flat(shape, size):
    return 1.0, 1.0


def grows_on_candidate(names):
    """The candidate's cost grows with node count on `names`; the baseline's does not."""

    def latency(shape, size):
        return (size / 10_000, 1.0) if shape in names else (1.0, 1.0)

    return latency


def result(screen, confirm=None, setup=None, errors=()):
    res = probe.build_result(
        screen=screen,
        confirm=confirm or {},
        sizes=SIZES,
        candidate=probe.side_ref(IMAGE_A, "main@4c10889", None),
        baseline=probe.side_ref(IMAGE_B, "6.2.0", None),
        machine={"runner": "local", "type": "test", "cpu": "test", "cores": 1},
        started="2026-11-24T00:00:00Z",
        artifacts=["raw/rows.csv"],
        setup=setup,
        errors=errors,
    )
    validator = _validator()
    if validator is not None:
        _, problems = validator.validate_text(json.dumps(res, allow_nan=False), "result")
        assert not problems, problems
    return res


def by_status(res, status):
    return [f for f in res["findings"] if f["status"] == status]


# --- RESP --------------------------------------------------------------------


def reader(*chunks):
    pieces = list(chunks)
    return RespReader(lambda: pieces.pop(0) if pieces else b"")


def test_resp_nested_reply_split_across_chunks():
    r = reader(b"*3\r\n*1\r\n$4\r\nn.id\r\n*1\r\n*1\r", b"\n:42\r\n*1\r\n+Cached execution: 0\r\n")
    assert r.read() == [["n.id"], [[42]], ["Cached execution: 0"]]


def test_resp_error_and_null():
    r = reader(b"-ERR boom\r\n$-1\r\n")
    err = r.read()
    assert isinstance(err, probe.ErrorReply) and err == "ERR boom"
    assert r.read() is None


def test_resp_closed_connection_raises():
    with pytest.raises(ConnectionError):
        reader(b"*2\r\n:1\r\n").read()


def test_reply_parse_stats_only_write():
    reply = Reply.parse(
        [["Relationships created: 100", "Query internal execution time: 1.250000 milliseconds"]]
    )
    assert reply.rows == [] and reply.stat_int("Relationships created") == 100
    assert reply.stat_int("Nodes created") == 0
    assert reply.server_ms() == 1.25


def test_reply_parse_nested_error_raises():
    with pytest.raises(probe.QueryError):
        Reply.parse([["x"], [], probe.ErrorReply("Query timed out")])


def test_shape_counts():
    edges, lookup, create = SHAPES[1], SHAPES[4], SHAPES[5]
    assert edges.count(Reply([], [], {"Relationships created": "100"})) == 100
    assert lookup.count(Reply(["n.id"], [[7]], {})) == 1
    assert create.count(Reply([], [], {"Nodes created": "1"})) == 1


# --- shapes ------------------------------------------------------------------


def test_unwind_shapes_are_micros_3963_rows():
    by_name = {s.name: s for s in SHAPES}
    rng = probe.random.Random(1)
    q = by_name[UNWIND_ID_1].query(by_name[UNWIND_ID_1].params(rng, 10_000))
    assert "UNWIND $rows AS e MATCH (a) WHERE id(a) = e.s MATCH (b) WHERE id(b) = e.d" in q
    assert q.endswith("CREATE (a)-[r:UWE]->(b) WITH r DELETE r")
    p = by_name["UNWIND index MATCH edge 100 rows"].params(rng, 500_000)
    assert len(p) == 100 and all(0 <= s < 500_000 and 0 <= d < 500_000 for s, d in p)
    assert all("unwind" in s.name.lower() for s in SHAPES[:4])
    assert not any("unwind" in s.name.lower() for s in SHAPES[4:])


# --- the growth ratio --------------------------------------------------------


def test_growth_is_ratio_at_largest_over_ratio_at_smallest():
    small = cell(UNWIND_ID_1, 10_000, 2.0, 0.5)  # ratio 4
    large = cell(UNWIND_ID_1, 500_000, 40.0, 0.5)  # ratio 80
    cand_g, base_g, g = probe.growth(small, large)
    assert (cand_g, base_g) == (20.0, 1.0)
    assert g == pytest.approx(20.0)


def test_flat_a_a_passes_with_no_loud_finding():
    res = result(grid(flat))
    assert res["verdict"] == "pass" and res["error"] is None and not res["confirmed"]
    assert all(f["status"] == "ok" for f in res["findings"])
    assert {f["metric"] for f in res["findings"]} == {"growth_ratio"}
    assert "no shape's candidate/baseline ratio grew" in res["summary"]


def test_constant_factor_slowdown_does_not_cross():
    """2x slower at every size: the growth ratio is 1. The micro tier owns that."""
    res = result(grid(lambda s, z: (2.0, 1.0)))
    assert res["verdict"] == "pass"


def test_confirmed_growth_is_a_regression_named_unwind():
    names = {UNWIND_ID_1}
    res = result(grid(grows_on_candidate(names)), grid(grows_on_candidate(names), "confirm"))
    assert res["verdict"] == "regression" and res["confirmed"] is True
    (reg,) = by_status(res, "regression")
    assert reg["name"] == UNWIND_ID_1 and "unwind" in reg["name"].lower()
    assert reg["metric"] == "growth_ratio" and reg["ratio"] == pytest.approx(50.0)
    assert (reg["candidate"], reg["baseline"], reg["threshold"]) == (50.0, 1.0, 1.5)
    # The per-size ratios, as info, growing with size.
    sizes = [f for f in res["findings"] if f["metric"] == "p50_ms"]
    assert [f["name"] for f in sizes] == [f"{UNWIND_ID_1} @{s}" for s in ("10k", "100k", "500k")]
    assert [f["ratio"] for f in sizes] == [1.0, 10.0, 50.0]
    assert all(f["status"] == "info" for f in sizes)


def test_crossing_not_repeated_by_the_rerun_is_info():
    res = result(grid(grows_on_candidate({UNWIND_ID_1})), grid(flat, "confirm"))
    assert res["verdict"] == "pass"
    growth_rows = [f for f in res["findings"] if f["name"] == UNWIND_ID_1]
    assert growth_rows[0]["status"] == "info"
    assert f"{UNWIND_ID_1} crossed once (50.00x) but not on the re-run (1.00x)" in res["summary"]
    # "highest" reads the shapes that settled, not the one that crossed once.
    assert (
        "highest" in res["summary"] and "highest UNWIND id MATCH edge 1 row" not in res["summary"]
    )


def test_crossing_the_rerun_could_not_measure_is_an_error():
    res = result(grid(grows_on_candidate({UNWIND_ID_1})))
    assert res["verdict"] == "error"
    assert "re-run could not measure it" in res["error"]
    assert not by_status(res, "regression")


def test_improvement_needs_no_rerun():
    """main vs 6.2.0: the baseline grows, the candidate does not."""

    def latency(shape, size):
        return (1.0, size / 10_000) if shape.startswith("UNWIND") else (1.0, 1.0)

    res = result(grid(latency))
    assert res["verdict"] == "pass"
    improved = by_status(res, "improvement")
    assert {f["name"] for f in improved} == {s.name for s in SHAPES[:4]}
    assert all(f["ratio"] == pytest.approx(0.02) for f in improved)


# --- correctness -------------------------------------------------------------


def test_candidate_error_is_an_immediate_completed_regression():
    screen = grid(flat)
    c = screen[(UNWIND_ID_1, 500_000)].runs["candidate"]
    c.wall = []
    c.fail("error", "error reply: boom")
    res = result(screen)
    assert res["verdict"] == "regression" and res["confirmed"]
    (reg,) = by_status(res, "regression")
    assert reg["name"] == f"{UNWIND_ID_1} @500k" and reg["metric"] == "completed"
    assert (reg["candidate"], reg["baseline"], reg["direction"]) == (0, 1, "lower_is_worse")


def fail(screen, shape, sizes, side, kind, why="boom"):
    for size in sizes:
        run = screen[(shape, size)].runs[side]
        run.wall = []
        run.fail(kind, why)
    return screen


def test_shape_the_baseline_cannot_run_is_left_out_as_info():
    screen = fail(grid(flat), UNWIND_ID_1, SIZES, "baseline", "error", "unknown function")
    res = result(screen)
    assert res["verdict"] == "pass"
    (left,) = [f for f in res["findings"] if f["metric"] == "completed"]
    assert left["name"] == UNWIND_ID_1 and left["status"] == "info"
    assert (left["candidate"], left["baseline"], left["ratio"]) == (1, 0, None)
    assert f"the baseline cannot run them: {UNWIND_ID_1}" in res["summary"]


@pytest.mark.parametrize(
    ("kind", "sizes"),
    [
        ("timeout", SIZES[-1:]),  # did not finish where the candidate could regress
        ("timeout", SIZES),
        ("error", SIZES[-1:]),  # an error at some sizes only is not "cannot run"
    ],
)
def test_other_baseline_failures_are_never_a_pass(kind, sizes):
    screen = grid(flat)
    for shape in SHAPES[:4]:
        fail(screen, shape.name, sizes, "baseline", kind)
    res = result(screen)
    assert res["verdict"] == "error" and "baseline boom" in res["error"]


def test_broken_on_both_sides_is_info_and_said():
    screen = grid(flat)
    for side in ("candidate", "baseline"):
        fail(screen, UNWIND_ID_1, SIZES[-1:], side, "timeout")
    res = result(screen)
    assert res["verdict"] == "pass"
    (both,) = [f for f in res["findings"] if f["metric"] == "completed"]
    assert both["status"] == "info" and (both["candidate"], both["baseline"]) == (0, 0)
    assert f"broken on both sides (not new): {UNWIND_ID_1} @500k" in res["summary"]


def test_baseline_down_is_an_error_even_when_the_candidate_failed_too():
    screen = grid(flat)
    fail(screen, "create node", SIZES[-1:], "candidate", "down")
    fail(screen, "create node", SIZES[-1:], "baseline", "down", "server stopped answering")
    res = result(screen)
    assert res["verdict"] == "error" and "server stopped answering" in res["error"]


def test_baseline_server_down_is_an_error():
    screen = grid(flat)
    b = screen[("create node", 500_000)].runs["baseline"]
    b.wall = []
    b.fail("down", "server stopped answering (container exited exit 139)")
    res = result(screen)
    assert res["verdict"] == "error" and "server stopped answering" in res["error"]


def test_count_mismatch_is_a_rows_regression():
    screen = grid(flat)
    screen[("UNWIND index MATCH edge 100 rows", 100_000)] = cell(
        "UNWIND index MATCH edge 100 rows",
        100_000,
        1.0,
        1.0,
        counts={"candidate": {0: 100, 1: 99}, "baseline": {0: 100, 1: 100}},
    )
    res = result(screen)
    (reg,) = by_status(res, "regression")
    assert reg["metric"] == "rows" and reg["direction"] == "mismatch"
    assert (reg["candidate"], reg["baseline"]) == (99, 100)


def test_candidate_fails_to_start_is_a_regression_baseline_is_an_error():
    setup = Setup()
    setup.side_failed("server starts and answers PING", {"candidate": "no PONG"})
    res = result({}, setup=setup)
    assert res["verdict"] == "regression" and res["error"] is None
    # Docker refusing the container says nothing about the engine.
    setup = Setup()
    setup.side_failed("server starts", {"candidate": "name in use"}, infra=["candidate"])
    res = result({}, setup=setup)
    assert res["verdict"] == "error" and "name in use" in res["error"]
    setup = Setup()
    setup.side_failed("build the 500k graph", {"baseline": "boom", "candidate": "boom"})
    res = result({}, setup=setup)
    assert res["verdict"] == "error" and "baseline" in res["error"]
    assert by_status(res, "info")[0]["metric"] == "completed"


def test_missing_units_are_never_a_pass():
    screen = grid(flat)
    del screen[("create node", 500_000)]
    res = result(screen)
    assert res["verdict"] == "error" and "17 of 18 units" in res["error"]


def test_budget_error_is_kept():
    res = result(grid(flat), errors=["the nightly budget (35 min) ran out"])
    assert res["verdict"] == "error" and "budget" in res["error"]


def test_regression_still_says_what_it_could_not_measure():
    screen = fail(grid(flat), UNWIND_ID_1, SIZES[-1:], "candidate", "timeout")
    fail(screen, "create node", SIZES[-1:], "baseline", "down", "exit 139")
    res = result(screen)
    assert res["verdict"] == "regression" and res["error"] is None
    assert "also could not measure: baseline exit 139" in res["summary"]


def test_summary_is_one_line_whatever_the_errors_hold():
    res = result(grid(flat), errors=["a\x0bb\u2028c\nd"])
    assert res["summary"] == " ".join(res["summary"].splitlines())


def test_correctness_findings_come_first():
    names = {UNWIND_ID_1}
    screen = grid(grows_on_candidate(names))
    fail(screen, "create node", SIZES[-1:], "candidate", "error")
    res = result(screen, grid(grows_on_candidate(names), "confirm"))
    assert [f["metric"] for f in by_status(res, "regression")] == ["completed", "growth_ratio"]


# --- contract limits ---------------------------------------------------------


def test_quiet_findings_are_capped_with_whole_triples():
    """Four regressing shapes: four triples of per-size rows do not fit ten
    quiet slots, so three whole triples go in, worst growth first."""
    # Growth exponents: the larger, the steeper the shape's ratio grows.
    power = {s.name: 0.5 + i * 0.25 for i, s in enumerate(SHAPES[:4])}

    def latency(shape, size):
        if shape not in power:
            return 1.0, 1.0
        return (size / 10_000) ** power[shape], 1.0

    res = result(grid(latency), grid(latency, "confirm"))
    regressions = by_status(res, "regression")
    assert [f["name"] for f in regressions] == [s.name for s in reversed(SHAPES[:4])]
    quiet = [f for f in res["findings"] if f["status"] in ("ok", "info")]
    assert len(quiet) == 10
    per_size = [f for f in quiet if f["metric"] == "p50_ms"]
    assert len(per_size) == 9  # three whole triples, never a split one
    triples = [f["name"].rsplit(" @", 1)[0] for f in per_size[::3]]
    assert triples == [s.name for s in reversed(SHAPES[1:4])]  # worst growth first
    # The tenth slot goes to a control's ok growth row.
    assert [f["metric"] for f in quiet if f["metric"] != "p50_ms"] == ["growth_ratio"]


def test_select_findings_keeps_every_loud_row_first():
    loud = [probe.finding(f"r{i}", "completed", 0, 1, "regression") for i in range(60)]
    quiet = [[probe.finding("q", "p50_ms", 1.0, 1.0, "info")]]
    out = probe.select_findings(probe.Assessment(loud, quiet, [], []))
    assert len(out) == 50 and all(f["status"] == "regression" for f in out)


def test_no_nan_or_infinity_reaches_the_result():
    assert probe._num(float("inf")) is None and probe._num(float("nan")) is None
    assert probe.ratio(1.0, 0.0) is None


def test_default_label_and_digest_rule():
    assert probe.default_label(IMAGE_A) == "ghcr.io/falkordb/falkordb-server@aaaaaaaaaaaa"
    assert probe.default_label("localhost:5000/x:tag@sha256:" + "c" * 64) == (
        "localhost:5000/x@cccccccccccc"
    )
    assert probe.DIGEST_RE.match(IMAGE_B) and not probe.DIGEST_RE.match("falkordb/x:6.2.0")
    assert probe.DIGEST_RE.match("localhost:5000/x:tag@sha256:" + "c" * 64)
    assert not probe.DIGEST_RE.match("FalkorDB/x@sha256:" + "c" * 64)  # docker refuses it


def test_run_with_a_tag_writes_an_error_result(tmp_path):
    """No Docker needed: bad arguments stop before any server starts."""
    res = probe.run(
        candidate="falkordb/falkordb-server:edge",
        baseline=IMAGE_B,
        out=tmp_path,
        mode="nightly",
        echo=lambda _: None,
    )
    assert res["verdict"] == "error" and "pinned by digest" in res["error"]
    written = json.loads((tmp_path / "result.json").read_text())
    assert written == res
    assert (tmp_path / "raw" / "rows.csv").exists()


def test_run_with_an_unknown_mode_writes_an_error_result(tmp_path):
    res = probe.run(
        candidate=IMAGE_A, baseline=IMAGE_B, out=tmp_path, mode="weekly", echo=lambda _: None
    )
    assert res["verdict"] == "error" and "unknown --mode" in res["error"]


@pytest.mark.parametrize("mode", ["queue", "nightly", "long"])
def test_every_mode_fits_its_budget_when_every_block_hits_its_cap(mode):
    """30 units (18 + 12 re-run), each a warm-up per side and four blocks."""
    spec = probe.MODES[mode]
    worst = (len(SHAPES) * 3 + len(SHAPES) * 2) * 6 * spec.block_seconds
    assert worst <= 0.9 * spec.budget_minutes * 60
    assert spec.reps >= probe.MIN_REPS


# --- run_unit, against fake servers ----------------------------------------


class FakeServer:
    """Answers every query at once; records what it was sent."""

    def __init__(self, side, log, fail_with=None):
        self.side, self.log, self.fail_with, self.settled = side, log, fail_with, 0

    def query(self, graph, cypher):
        self.log.append((self.side, cypher))
        if self.fail_with is not None:
            raise self.fail_with
        return Reply([], [], {"Relationships created": "1"}), 1.0

    def settle(self, graph):
        self.settled += 1


def unit(spec, deadline=None, **fails):
    log = []
    servers = {s: FakeServer(s, log, fails.get(s)) for s in probe.SIDES}
    shape = SHAPES[2]  # UNWIND id MATCH edge 1 row
    cell = probe.run_unit(
        servers, shape, 10_000, "screen", "baseline", spec, lambda r: None, deadline
    )
    return cell, log, servers


def test_run_unit_runs_a_b_b_a_with_identical_requests_on_both_sides():
    spec = probe.ModeSpec(warmup=2, reps=3, block_seconds=60.0, budget_minutes=1.0)
    cell, log, _ = unit(spec)
    sides = "".join(side[0] for side, _ in log)
    assert sides == "bb" + "cc" + "bbb" + "ccc" + "ccc" + "bbb"  # warm-ups, then A B B A
    sent = {s: [q for side, q in log if side == s] for s in probe.SIDES}
    assert sent["candidate"] == sent["baseline"] and len(set(sent["baseline"])) == 8
    assert [len(cell.runs[s].wall) for s in probe.SIDES] == [6, 6]
    assert probe.first_mismatch(cell) is None


def test_run_unit_caps_every_block_and_the_warm_up():
    spec = probe.ModeSpec(warmup=4, reps=20, block_seconds=0.0, budget_minutes=1.0)
    cell, log, _ = unit(spec)
    per_side = 1 + probe.MIN_REPS * 2  # warm-up stops after one, blocks after MIN_REPS
    assert len(log) == 2 * per_side
    assert len(cell.runs["candidate"].wall) == 2 * probe.MIN_REPS


def test_run_unit_candidate_timeout_settles_and_the_baseline_still_runs():
    spec = probe.ModeSpec(warmup=1, reps=3, block_seconds=60.0, budget_minutes=1.0)
    cell, log, servers = unit(spec, candidate=probe.Timeout("no reply within 120 s"))
    c, b = cell.runs["candidate"], cell.runs["baseline"]
    assert c.failure_kind == "timeout" and servers["candidate"].settled == 1
    assert len([1 for side, _ in log if side == "candidate"]) == 1  # never re-sent
    assert b.failure is None and len(b.wall) == 6


def test_run_unit_server_side_timeout_reply_is_a_timeout():
    spec = probe.ModeSpec(warmup=1, reps=3, block_seconds=60.0, budget_minutes=1.0)
    cell, _, servers = unit(spec, baseline=probe.QueryError("Query timed out"))
    assert cell.runs["baseline"].failure_kind == "timeout"
    assert servers["baseline"].settled == 0  # the server already ended it


def test_run_unit_past_the_deadline_raises_with_the_partial_cell():
    spec = probe.ModeSpec(warmup=1, reps=3, block_seconds=60.0, budget_minutes=1.0)
    with pytest.raises(probe.BudgetOut) as raised:
        unit(spec, deadline=0.0)
    assert raised.value.cell is not None and raised.value.cell.shape == SHAPES[2].name


def test_run_without_docker_still_writes_an_error_result(tmp_path, monkeypatch):
    """A failure inside the measurement (here: no `docker` binary at all) is an
    error result, never a crash without result.json."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    res = probe.run(
        candidate=IMAGE_A,
        baseline=IMAGE_B,
        out=tmp_path / "out",
        mode="nightly",
        echo=lambda _: None,
    )
    assert res["verdict"] == "error" and "FileNotFoundError" in res["error"]
    assert json.loads((tmp_path / "out" / "result.json").read_text())["verdict"] == "error"
