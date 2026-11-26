"""Unit tests for the pokec check: parsing, the comparison rules and result.json.

No Docker, no network: runs are built in memory. Every result the tests build is
checked with falkorbench.contract, the contract's checker.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
from pathlib import Path

import pytest

from falkorbench import contract
from falkorbench import kit
from falkorbench.pokec import check as pokec

CAND = "ghcr.io/falkordb/falkordb-server@sha256:" + "a" * 64
BASE = "falkordb/falkordb-server@sha256:" + "b" * 64
THREE_HOP = "aggregate_expansion_3_with_filter"


def validate(result: dict) -> list:
    return [str(p) for p in contract.validate_result(json.loads(json.dumps(result)))]


# --- parsing -----------------------------------------------------------------


def test_convert_dataset_matches_the_nov_23_csvs(tmp_path):
    source = tmp_path / "dump.cypher"
    source.write_text(
        'CREATE (:User {id: 10279, completion_percentage: 14, gender: "man", age: 26});\n'
        'CREATE (:User {id: 7, completion_percentage: 0, gender: "woman"});\n'
        ";\n"
        "MATCH (n:User {id: 10279}), (m:User {id: 7}) CREATE (n)-[e: Friend]->(m);\n"
    )
    users, friends = tmp_path / "User.csv", tmp_path / "FRIEND.csv"
    assert pokec.convert_dataset(source, users, friends) == (2, 1)
    assert users.read_text() == "id,completion_percentage,gender,age\n10279,14,man,26\n7,0,woman,\n"
    # 1 + (10279 * 31 + 7 * 17) % 20 = 1 + 318768 % 20 = 9
    assert friends.read_text() == "src_id,dst_id,bench_capacity\n10279,7,9\n"


METRICS = """\
# HELP falkordb_latency_p99_us p99
# TYPE falkordb_latency_p99_us gauge
falkordb_latency_p50_us 2100
falkordb_latency_p95_us 51000
falkordb_latency_p99_us 210000
falkordb_query_latency_pct_us{pct="50",query="single_vertex_read"} 400
falkordb_query_latency_pct_us{query="single_vertex_read",pct="99"} 6500
falkordb_query_latency_pct_us{pct="10",query="single_vertex_read"} 100
benchmark_query_timeout_rate_pct{query="aggregate_expansion_3_with_filter",vendor="falkor"} 50
benchmark_query_failure_rate_pct{query="single_edge_write",vendor="falkor"} 25
benchmark_global_timeout_rate_pct{vendor="falkor"} 1.5
"""


def test_parse_metrics():
    metrics = pokec.parse_metrics(METRICS)
    assert metrics["overall"] == {"p50_ms": 2.1, "p95_ms": 51.0, "p99_ms": 210.0}
    assert metrics["per_query"]["single_vertex_read"] == {"p50_ms": 0.4, "p99_ms": 6.5}
    assert metrics["rates"][THREE_HOP] == {"timeout": 50.0}
    assert metrics["rates"]["single_edge_write"] == {"failure": 25.0}
    assert metrics["global_rates"] == {"timeout": 1.5}


Q3 = "CYPHER id = 28105 MATCH (s:User {id: $id})-->()-->()-->(n:User)  WHERE n.age >= 18  RETURN DISTINCT n.id"
QW = "CYPHER id = 5 MATCH (n:User {id: $id}) SET n.age = 3"
LOG = (
    "\x1b[2m2026-11-24T09:31:41.170739Z\x1b[0m \x1b[31mERROR\x1b[0m benchmark::falkor::falkor_driver: "
    f'Error executing query: {Q3}, the error is: RedisError("Query timed out")\n'
    "    at src/falkor/falkor_driver.rs:1244\n"
    f"2026-11-24T09:31:42Z ERROR benchmark::falkor::falkor_driver: Error executing query: {Q3}, the error is: Elapsed(())\n"
    f'2026-11-24T09:31:43Z ERROR benchmark::falkor::falkor_driver: Error executing query: {QW}, the error is: ParsingError("x")\n'
    "2026-11-24T09:31:44Z ERROR benchmark::falkor::falkor_driver: Error executing query: CYPHER id = 1 RETURN 9, "
    'the error is: ParsingError("x")\n'
    # The worker's own line repeats the error inside OtherError; it is not a second failure.
    f"2026-11-24T09:31:45Z  INFO benchmark: worker 3 failed to process query, not sleeping for 3 seconds "
    f'OtherError("Error (type x) executing query: {Q3}, the error is: y")\n'
)


def test_parse_errors_counts_each_failed_request_once():
    errors, unmatched = pokec.parse_errors(LOG, {Q3: THREE_HOP, QW: "single_vertex_update"})
    assert errors == {
        THREE_HOP: {"timeouts": 2, "failures": 0},
        "single_vertex_update": {"timeouts": 0, "failures": 1},
    }
    assert unmatched == ["CYPHER id = 1 RETURN 9"]


def test_read_query_file(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(
        json.dumps({"size": 2, "catalog": [{"id": 10, "name": THREE_HOP, "q_type": "Read"}]})
        + "\n"
        + json.dumps({"q_name": THREE_HOP, "q_type": "Read", "cypher": Q3})
        + "\n"
        + json.dumps({"q_name": "single_vertex_update", "q_type": "Write", "cypher": QW})
        + "\n"
    )
    qfile = pokec.read_query_file(path)
    assert qfile.counts == {THREE_HOP: 1, "single_vertex_update": 1}
    assert qfile.by_text == {Q3: THREE_HOP, QW: "single_vertex_update"}


def test_hop_label_names_the_3_hop_query():
    assert (
        pokec.hop_label(THREE_HOP)
        == "3-hop expansion, age >= 18 (aggregate_expansion_3_with_filter)"
    )
    assert pokec.hop_label("aggregate_expansion_3") == "3-hop expansion (aggregate_expansion_3)"
    assert pokec.hop_label("single_vertex_read") == "single_vertex_read"


# --- the comparison ----------------------------------------------------------

QUERIES = {"single_vertex_read": 200, THREE_HOP: 200, "single_edge_write": 50}


def qfiles(workloads=("read", "write5")):
    files = {}
    for key in workloads:
        files[("medium", key)] = pokec.QueryFile(Path("q.jsonl"), dict(QUERIES), {}, {})
    return files


def make_run(
    side,
    pass_no,
    rep,
    workload="write5",
    scale=1.0,
    finished=True,
    ping=5.0,
    errors=None,
    overrides=None,
    stop_reason=None,
):
    per_query = {
        q: {"p50_ms": 10.0 * scale, "p95_ms": 50.0 * scale, "p99_ms": 100.0 * scale}
        for q in QUERIES
    }
    for query, values in (overrides or {}).items():
        per_query[query] = {k: v for k, v in values.items()}
    run = pokec.Run("medium", workload, side, pass_no, rep)
    run.finished = finished
    run.stop_reason = None if finished else (stop_reason or "stopped at the deadline (180 s)")
    run.wall_s = 60.0 * scale
    run.elapsed_s = 55.0 * scale if finished else None
    run.ping = {"ping_max_ms": ping}
    if finished:
        run.per_query = per_query
        run.overall = {"p50_ms": 10.0 * scale, "p95_ms": 50.0 * scale, "p99_ms": 100.0 * scale}
    run.errors = {q: dict(v) for q, v in (errors or {}).items()}  # each run its own
    return run


def pass_runs(pass_no, reps, **kw_by_side):
    runs = []
    for rep in range(1, reps + 1):
        for side in ("baseline", "candidate"):
            runs.append(make_run(side, pass_no, rep, **kw_by_side.get(side, {})))
    return runs


def analyse(runs, mode="nightly", rerun_done=None):
    return pokec.analyse(
        runs, pokec.MODES[mode], qfiles(), [], set() if rerun_done is None else rerun_done
    )


def statuses(analysis, status):
    return [(f["name"], f["metric"]) for f in analysis.findings if f["status"] == status]


def test_a_a_is_quiet():
    analysis = analyse(pass_runs(1, 3), rerun_done={("medium", "write5")})
    assert not analysis.rerun
    assert not statuses(analysis, "regression")
    assert not analysis.unmeasured
    rows = {(r["name"], r["metric"]) for r in analysis.rows}
    assert (f"{pokec.hop_label(THREE_HOP)} [5% writes]", "p99_ms") in rows
    assert ("PING max [5% writes]", "ping_max_ms") in rows


def test_latency_regression_needs_the_re_run():
    slow = {
        "workload": "read",
        "overrides": {"single_vertex_read": {"p50_ms": 10.0, "p95_ms": 50.0, "p99_ms": 200.0}},
    }
    read = {"workload": "read"}
    first = pass_runs(1, 3, candidate=slow, baseline=read)
    analysis = analyse(first)
    assert analysis.rerun == {("medium", "read")}
    assert not statuses(analysis, "regression")

    repeated = first + pass_runs(2, 2, candidate=slow, baseline=read)
    analysis = analyse(repeated, rerun_done={("medium", "read")})
    assert statuses(analysis, "regression") == [("single_vertex_read [read-only]", "p99_ms")]

    not_repeated = first + pass_runs(2, 2, candidate=read, baseline=read)
    analysis = analyse(not_repeated, rerun_done={("medium", "read")})
    assert not statuses(analysis, "regression")
    assert any("not in the re-run" in note for note in analysis.notes)


def test_under_writes_only_p50_gates_per_query():
    # A read's p95/p99 under writes is the writer gate, not the query: rows.csv only.
    tail = {"overrides": {"single_vertex_read": {"p50_ms": 10.0, "p95_ms": 500.0, "p99_ms": 900.0}}}
    analysis = analyse(
        pass_runs(1, 3, candidate=tail) + pass_runs(2, 2, candidate=tail),
        rerun_done={("medium", "write5")},
    )
    assert not analysis.rerun and not statuses(analysis, "regression")
    row = next(
        r
        for r in analysis.rows
        if r["name"] == "single_vertex_read [5% writes]" and r["metric"] == "p99_ms"
    )
    assert row["use"] == "row" and row["first_pass"] == 1
    median = {
        "overrides": {"single_vertex_read": {"p50_ms": 20.0, "p95_ms": 50.0, "p99_ms": 100.0}}
    }
    analysis = analyse(
        pass_runs(1, 3, candidate=median) + pass_runs(2, 2, candidate=median),
        rerun_done={("medium", "write5")},
    )
    assert statuses(analysis, "regression") == [("single_vertex_read [5% writes]", "p50_ms")]


def test_under_writes_write_queries_never_gate():
    files = qfiles()
    files[("medium", "write5")].types["single_edge_write"] = "Write"
    slow = {
        "overrides": {"single_edge_write": {"p50_ms": 900.0, "p95_ms": 2000.0, "p99_ms": 3000.0}}
    }
    runs = pass_runs(1, 3, candidate=slow) + pass_runs(2, 2, candidate=slow)
    analysis = pokec.analyse(runs, pokec.MODES["nightly"], files, [], {("medium", "write5")})
    assert not statuses(analysis, "regression")
    assert {
        r["use"]
        for r in analysis.rows
        if r["name"] == "single_edge_write [5% writes]" and r["metric"] != "completed"
    } == {"row"}


def test_latency_floor_ignores_tiny_moves():
    # 0.4 ms -> 0.8 ms is 2x but under the 1 ms p50 floor.
    tiny = {"overrides": {"single_vertex_read": {"p50_ms": 0.8, "p95_ms": 50.0, "p99_ms": 100.0}}}
    base = {"overrides": {"single_vertex_read": {"p50_ms": 0.4, "p95_ms": 50.0, "p99_ms": 100.0}}}
    analysis = analyse(pass_runs(1, 3, candidate=tiny, baseline=base))
    assert not analysis.rerun


def test_queue_mode_reports_latency_but_never_fails_on_it():
    slow = {"scale": 2.0}
    analysis = analyse(pass_runs(1, 1, candidate=slow), mode="queue")
    assert not analysis.rerun
    assert not statuses(analysis, "regression")
    assert ("all queries [5% writes]", "p99_ms") in statuses(analysis, "info")


def test_timeouts_in_two_runs_are_a_regression_named_3_hop():
    hung = {"errors": {THREE_HOP: {"timeouts": 40, "failures": 0}}}
    analysis = analyse(pass_runs(1, 3, candidate=hung))
    regressions = statuses(analysis, "regression")
    assert (f"{pokec.hop_label(THREE_HOP)} [5% writes]", "completed") in regressions
    assert any(re.search("3-hop", name, re.I) for name, _ in regressions)


def test_timeouts_in_one_run_ask_for_a_re_run_then_stay_info():
    runs = pass_runs(1, 3)
    runs[1].errors = {THREE_HOP: {"timeouts": 1, "failures": 0}}  # candidate, rep 1
    analysis = analyse(runs)
    assert analysis.rerun == {("medium", "write5")}
    analysis = analyse(runs + pass_runs(2, 2), rerun_done={("medium", "write5")})
    assert not statuses(analysis, "regression")
    assert (f"{pokec.hop_label(THREE_HOP)} [5% writes]", "completed") in statuses(analysis, "info")


def test_one_candidate_timeout_run_asks_for_a_re_run_even_when_the_baseline_is_broken():
    """#4065: the repeat decides red or green whatever the baseline did; until it has, a
    baseline that errs does not make the candidate's row a fix."""
    broken = {"errors": {THREE_HOP: {"timeouts": 0, "failures": 5}}}
    runs = pass_runs(1, 3, baseline=broken)
    cand = [r for r in runs if r.side == "candidate"]
    cand[0].errors = {THREE_HOP: {"timeouts": 1, "failures": 0}}
    label = f"{pokec.hop_label(THREE_HOP)} [5% writes]"
    analysis = analyse(runs)
    assert analysis.rerun == {("medium", "write5")}
    assert (label, "completed") not in statuses(analysis, "improvement")
    again = [r for r in pass_runs(2, 2, baseline=broken)]
    for r in again:
        if r.side == "candidate":
            r.errors = {THREE_HOP: {"timeouts": 1, "failures": 0}}
    analysis = analyse(runs + again, rerun_done={("medium", "write5")})
    assert (label, "completed") in statuses(analysis, "regression")  # it repeated


def test_an_error_reply_is_a_regression_at_once():
    runs = pass_runs(1, 3)
    runs[1].errors = {"single_edge_write": {"timeouts": 0, "failures": 1}}
    analysis = analyse(runs)
    assert ("single_edge_write [5% writes]", "completed") in statuses(analysis, "regression")


def test_errors_on_both_sides_are_a_regression():
    """#4065: an error reply on the candidate is a regression whether or not the baseline
    errs too; the result's `broken` tells the release gate the baseline shares it."""
    broken = {"errors": {"single_edge_write": {"timeouts": 0, "failures": 50}}}
    analysis = analyse(pass_runs(1, 3, candidate=broken, baseline=broken))
    assert statuses(analysis, "regression") == [("single_edge_write [5% writes]", "completed")]


def test_a_run_that_does_not_finish_is_a_regression_at_once():
    analysis = analyse(pass_runs(1, 3, candidate={"finished": False, "ping": 30000.0}))
    regressions = statuses(analysis, "regression")
    assert ("benchmark run finished [5% writes]", "completed") in regressions
    assert ("PING max [5% writes]", "ping_max_ms") in regressions
    assert analysis.unmeasured  # no latency comparison without a finished candidate run


def test_a_run_that_finished_on_neither_side_is_a_regression_and_unmeasured():
    analysis = analyse(pass_runs(1, 3, candidate={"finished": False}, baseline={"finished": False}))
    assert ("benchmark run finished [5% writes]", "completed") in statuses(analysis, "regression")
    assert analysis.unmeasured


def test_one_ping_freeze_asks_for_a_re_run():
    runs = pass_runs(1, 1, candidate={"ping": 2500.0})
    analysis = analyse(runs, mode="queue")
    assert analysis.rerun == {("medium", "write5")}
    analysis = analyse(
        runs + pass_runs(2, 1, candidate={"ping": 2400.0}),
        mode="queue",
        rerun_done={("medium", "write5")},
    )
    assert ("PING max [5% writes]", "ping_max_ms") in statuses(analysis, "regression")


def test_freezes_on_both_sides_are_not_new():
    analysis = analyse(pass_runs(1, 3, candidate={"ping": 2200.0}, baseline={"ping": 2100.0}))
    assert ("PING max [5% writes]", "ping_max_ms") in statuses(analysis, "info")


def test_throughput_is_lower_is_worse():
    # scale 2 -> every latency 2x and the run half as fast.
    runs = pass_runs(1, 3, candidate={"scale": 2.0}) + pass_runs(2, 2, candidate={"scale": 2.0})
    analysis = analyse(runs, rerun_done={("medium", "write5")})
    regressions = {(n, m) for n, m in statuses(analysis, "regression")}
    assert ("throughput [5% writes]", "throughput") in regressions
    finding = next(f for f in analysis.findings if f["metric"] == "throughput")
    assert finding["direction"] == "lower_is_worse" and finding["ratio"] == 0.5


# --- result.json -------------------------------------------------------------


def make_check(tmp_path, mode="nightly", **extra):
    args = argparse.Namespace(
        candidate=CAND,
        baseline=BASE,
        out=str(tmp_path / "out"),
        mode=mode,
        candidate_label="6.0.4",
        baseline_label=None,
        candidate_commit="6ca3adc9d0418f76c5a15f820552e0603a65e553",
        baseline_commit=None,
        tools_dir=str(tmp_path / "tools"),
        reps=None,
        workloads=None,
        **extra,
    )
    check = pokec.Check(args)
    check.machine = {"runner": "local", "type": "test", "cpu": "test", "cores": 1}
    check.query_files = qfiles()
    return check


def test_result_pass_is_valid(tmp_path):
    check = make_check(tmp_path)
    check.runs = pass_runs(1, 3) + pass_runs(
        1, 3, baseline={"workload": "read"}, candidate={"workload": "read"}
    )
    result = check.build()
    assert result["verdict"] == "pass", result["summary"]
    assert result["confirmed"] is False and result["error"] is None
    assert result["baseline"]["label"] == "falkordb/falkordb-server@" + "b" * 12
    assert result["candidate"]["commit"] == "6ca3adc9d0418f76c5a15f820552e0603a65e553"
    assert "raw/rows.csv" in result["artifacts"] and (Path(check.out) / "raw" / "rows.csv").exists()
    assert validate(result) == []


def test_result_regression_is_valid_and_matches_the_acceptance_pattern(tmp_path):
    check = make_check(tmp_path)
    hung = {
        "finished": False,
        "ping": 5100.0,
        "errors": {THREE_HOP: {"timeouts": 30, "failures": 0}},
    }
    check.runs = pass_runs(1, 3, candidate=hung)
    result = check.build()
    assert result["verdict"] == "regression" and result["confirmed"] is True
    assert any(
        re.search("3-hop", f["name"], re.I) and f["status"] == "regression"
        for f in result["findings"]
    )
    assert "\n" not in result["summary"]
    assert validate(result) == []


def test_result_error_when_the_baseline_does_not_finish(tmp_path):
    check = make_check(tmp_path)
    check.runs = pass_runs(1, 3, baseline={"finished": False})
    result = check.build()
    assert result["verdict"] == "error" and "not measured" in result["error"]
    assert validate(result) == []


def test_result_error_when_setup_failed(tmp_path):
    check = make_check(tmp_path)
    check.error = "could not pull the baseline image"
    result = check.build()
    assert result["verdict"] == "error" and result["findings"] == []
    assert result["summary"].startswith("Could not measure: could not pull")
    assert validate(result) == []


def test_candidate_that_does_not_start_is_a_regression(tmp_path):
    check = make_check(tmp_path)
    check.info.append(
        kit.finding(
            "server started [medium]", "completed", 0, 1, "regression", direction="lower_is_worse"
        )
    )
    result = check.build()
    assert result["verdict"] == "regression"
    assert validate(result) == []


def test_cross_check_catches_a_log_parser_that_misses_failures(tmp_path):
    check = make_check(tmp_path)
    run = make_run("candidate", 1, 1)
    run.errors = {}  # the parser saw nothing ...
    metrics = {"rates": {THREE_HOP: {"timeout": 10.0}}}  # ... but the tool counted 20 of 200
    with pytest.raises(pokec.SetupError, match="log parser"):
        check._cross_check(run, metrics, qfiles()[("medium", "write5")], [])
    run.errors = {THREE_HOP: {"timeouts": 20, "failures": 0}}
    check._cross_check(run, metrics, qfiles()[("medium", "write5")], [])


# --- the RESP client and the PING probe, against a fake server ----------------


class FakeServer:
    """Replies to PING with +PONG; `stall` seconds before the n-th reply; `mute` after it."""

    def __init__(
        self,
        stall_at=None,
        stall=0.0,
        mute_at=None,
        reply=b"+PONG\r\n",
        signal_at=None,
        hold_at=None,
    ):
        import socket
        import threading

        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen()
        self.port = self.sock.getsockname()[1]
        self.stall_at, self.stall, self.mute_at, self.reply = stall_at, stall, mute_at, reply
        self.count = 0
        self.signal_at = signal_at
        self.got = threading.Event()  # set once the signal_at-th PING has arrived
        self.hold_at = hold_at
        self.release = threading.Event()  # the hold_at-th reply waits for this
        threading.Thread(target=self.serve, daemon=True).start()

    def serve(self):
        # the probe closed its end (stop() shuts the socket): nothing left to serve
        with contextlib.suppress(OSError):
            self._serve()

    def _serve(self):
        import time

        conn, _ = self.sock.accept()
        while True:
            data = conn.recv(1024)
            if not data:
                return
            for _ in range(data.count(b"PING")):
                self.count += 1
                if self.count == self.signal_at:
                    self.got.set()
                if self.mute_at is not None and self.count >= self.mute_at:
                    continue
                if self.count == self.stall_at:
                    time.sleep(self.stall)
                if self.count == self.hold_at:
                    self.release.wait(10)
                conn.sendall(self.reply)


def test_resp_parses_nested_replies():
    reply = b"*3\r\n*1\r\n$8\r\ncount(n)\r\n*1\r\n*2\r\n:100000\r\n$-1\r\n*1\r\n+OK\r\n"
    server = FakeServer(reply=reply)
    with pokec.Resp(server.port, 5) as conn:
        assert conn.call("PING") == [["count(n)"], [[100000, None]], ["OK"]]
    server = FakeServer(reply=b"-ERR Query timed out\r\n")
    with pokec.Resp(server.port, 5) as conn:
        error = conn.call("PING")
    assert isinstance(error, pokec.RespError) and "timed out" in str(error)


def test_ping_probe_sees_a_freeze(tmp_path):
    import time

    server = FakeServer(stall_at=3, stall=1.2)
    probe = pokec.PingProbe(server.port, tmp_path / "ping.csv")
    probe.start()
    time.sleep(2.2)
    probe.stop()
    summary = probe.summary()
    assert summary["ping_over_1s"] == 1 and 1150 <= summary["ping_max_ms"] < 2000
    assert summary["ping_samples"] >= 5
    assert (tmp_path / "ping.csv").read_text().startswith("epoch_ms,latency_ms,kind\n")


def test_ping_probe_counts_a_freeze_that_outlasts_the_run(tmp_path):
    import time

    server = FakeServer(mute_at=3, signal_at=3)
    probe = pokec.PingProbe(server.port, tmp_path / "ping.csv")
    probe.start()
    # Time the freeze from the muted PING's arrival, not from the probe's start.
    assert server.got.wait(5)
    time.sleep(1.2)
    probe.stop()
    summary = probe.summary()
    assert summary["ping_timeouts"] == 1  # the PING still waiting when the run ended
    assert summary["ping_max_ms"] >= 1000


# --- review findings, kept as tests ---------------------------------------------


def test_a_crossed_row_whose_re_run_did_not_finish_is_not_a_pass(tmp_path):
    slow = {
        "workload": "read",
        "overrides": {"single_vertex_read": {"p50_ms": 10.0, "p95_ms": 50.0, "p99_ms": 300.0}},
    }
    read = {"workload": "read"}
    runs = pass_runs(1, 3, candidate=slow, baseline=read)
    runs += pass_runs(2, 2, candidate=slow, baseline={"workload": "read", "finished": False})
    analysis = analyse(runs, rerun_done={("medium", "read")})
    assert not statuses(analysis, "regression")
    assert any("never re-measured" in u for u in analysis.unmeasured)
    check = make_check(tmp_path)
    check.runs = runs
    result = check.build()
    assert result["verdict"] == "error"
    assert validate(result) == []


def test_one_freeze_in_three_runs_asks_for_a_re_run():
    runs = pass_runs(1, 3)
    runs[1].ping = {"ping_max_ms": 2500.0}  # candidate rep 1
    analysis = analyse(runs)
    assert analysis.rerun == {("medium", "write5")}
    # two of five candidate runs froze, the baseline never did: a regression at the repeated level
    more = pass_runs(2, 2)
    more[0].ping = {"ping_max_ms": 1800.0} if more[0].side == "candidate" else more[0].ping
    more[1].ping = {"ping_max_ms": 1800.0} if more[1].side == "candidate" else more[1].ping
    analysis = analyse(runs + more, rerun_done={("medium", "write5")})
    finding = next(f for f in analysis.findings if f["metric"] == "ping_max_ms")
    assert finding["status"] == "regression" and finding["candidate"] == 1800.0


def test_a_baseline_run_that_did_not_finish_is_not_a_pass(tmp_path):
    runs = pass_runs(1, 3)
    runs[0].finished, runs[0].stop_reason = False, "bulk load exited 1"  # baseline rep 1
    check = make_check(tmp_path)
    check.runs = runs
    result = check.build()
    assert result["verdict"] == "error" and "baseline did not finish" in result["error"]
    assert validate(result) == []


def test_the_summary_names_queries_broken_on_both_sides(tmp_path):
    broken = {"errors": {"single_edge_write": {"timeouts": 0, "failures": 50}}}
    check = make_check(tmp_path)
    check.runs = pass_runs(1, 3, candidate=broken, baseline=broken) + pass_runs(
        1, 3, candidate={"workload": "read"}, baseline={"workload": "read"}
    )
    result = check.build()
    assert result["verdict"] == "regression"
    assert (
        "single_edge_write [5% writes] did not complete on the candidate (baseline too)"
        in result["summary"]
    )
    assert result["broken"] == [
        {
            "name": "single_edge_write [5% writes]",
            "baseline_too": True,
            "issue": None,
            "introduced": False,
        }
    ]
    assert validate(result) == []


def test_an_error_with_a_carriage_return_stays_one_line(tmp_path):
    check = make_check(tmp_path)
    check.error = "docker said:\r\nno space left on device"
    result = check.build()
    assert "\r" not in result["summary"] and "\n" not in result["summary"]
    assert validate(result) == []


def test_main_writes_an_error_result_for_a_tag(tmp_path, monkeypatch):
    monkeypatch.delenv("PERF_MACHINE_JSON", raising=False)
    monkeypatch.setattr(kit, "remove_labelled", lambda label: None)  # no Docker in unit tests
    out = tmp_path / "out"
    code = pokec.main(
        [
            "--candidate",
            "falkordb/falkordb-server:6.2.0",
            "--baseline",
            BASE,
            "--out",
            str(out),
            "--mode",
            "nightly",
            "--tools-dir",
            str(tmp_path / "tools"),
        ]
    )
    assert code == 0
    result = json.loads((out / "result.json").read_text())
    assert (
        result["verdict"] == "error"
        and "is not name[:tag]@sha256:<64 hex>, a digest" in result["error"]
    )


def test_no_re_run_once_the_candidate_did_not_finish():
    runs = pass_runs(
        1, 3, candidate={"finished": False, "errors": {THREE_HOP: {"timeouts": 30, "failures": 0}}}
    )
    runs[1].errors["single_edge_write"] = {
        "timeouts": 1,
        "failures": 0,
    }  # one run only: would ask for a re-run
    analysis = analyse(runs)
    assert not analysis.rerun
    assert any("no re-run" in note for note in analysis.notes)
    assert (f"{pokec.hop_label(THREE_HOP)} [5% writes]", "completed") in statuses(
        analysis, "regression"
    )


def test_a_freeze_with_no_baseline_to_compare_is_not_a_regression():
    runs = pass_runs(1, 3, candidate={"ping": 5000.0})
    for run in runs:
        if run.side == "baseline":
            run.ping = {}  # the probe never ran (the load failed first)
            run.finished, run.stop_reason = False, "bulk load exited 1"
    analysis = analyse(runs)
    assert ("PING max [5% writes]", "ping_max_ms") in statuses(analysis, "info")
    assert any("too few runs" in u for u in analysis.unmeasured)


def test_sigterm_during_a_checkpoint_keeps_the_findings(tmp_path):
    check = make_check(tmp_path)
    hung = {
        "finished": False,
        "ping": 5100.0,
        "errors": {THREE_HOP: {"timeouts": 30, "failures": 0}},
    }
    check.runs = pass_runs(1, 3, candidate=hung)
    real = check.build

    def interrupted(error=None):
        raise kit.Stopped("signal 15")

    check.build = interrupted
    with pytest.raises(kit.Stopped):  # the checkpoint does not swallow the stop
        check.checkpoint("after a run")
    check.build = real
    result = check.build("stopped by a signal")
    assert result["verdict"] == "regression" and result["findings"]


# --- review round 2 (PR #4000 bot threads), kept as tests --------------------------

V1 = "CYPHER id = 7 MATCH (n:User {id: $id}) RETURN n"


def test_a_shared_query_text_keeps_both_types_and_is_split_from_metrics(tmp_path):
    path = tmp_path / "q.jsonl"
    lines = [
        {"size": 3, "catalog": []},
        {"q_name": "vertex_on_label_property", "q_type": "Read", "cypher": V1},
        {"q_name": "vertex_on_label_property_index", "q_type": "Read", "cypher": V1},
        {
            "q_name": "single_vertex_read",
            "q_type": "Read",
            "cypher": "CYPHER id = 8 MATCH (n) RETURN n",
        },
    ]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    qfile = pokec.read_query_file(path)
    shared = "vertex_on_label_property|vertex_on_label_property_index"
    assert qfile.by_text[V1] == shared
    log = f"2026-11-24T09:31:42Z ERROR benchmark::falkor::falkor_driver: Error executing query: {V1}, the error is: Elapsed(())\n"
    errors, unmatched = pokec.parse_errors(log, qfile.by_text)
    assert errors == {shared: {"timeouts": 1, "failures": 0}} and not unmatched
    # metrics.prom: the _index type had its one request fail (100 % of 1)
    rates = {"vertex_on_label_property_index": {"failure": 100.0}}
    resolved, problem = pokec.attribute_ambiguous(errors, qfile.counts, rates)
    assert problem is None
    assert resolved == {"vertex_on_label_property_index": {"timeouts": 1, "failures": 0}}
    check = make_check(tmp_path)
    run = make_run("candidate", 1, 1)
    run.errors = resolved
    check._cross_check(run, {"rates": rates}, qfile, [])  # agrees with metrics.prom
    # metrics.prom says nothing failed: the shared failure cannot be placed
    assert pokec.attribute_ambiguous(errors, qfile.counts, {})[1]


def test_a_re_run_without_a_value_for_the_row_is_not_a_pass(tmp_path):
    slow = {
        "workload": "read",
        "overrides": {"single_vertex_read": {"p50_ms": 10.0, "p95_ms": 50.0, "p99_ms": 300.0}},
    }
    read = {"workload": "read"}
    runs = pass_runs(1, 3, candidate=slow, baseline=read) + pass_runs(
        2, 2, candidate=slow, baseline=read
    )
    for run in runs:
        if run.pass_no == 2 and run.side == "baseline":
            run.per_query.pop("single_vertex_read")  # every request of it failed in the re-run
    analysis = analyse(runs, rerun_done={("medium", "read")})
    row = next(
        r
        for r in analysis.rows
        if r["name"] == "single_vertex_read [read-only]" and r["metric"] == "p99_ms"
    )
    assert row["first_pass"] == 1 and row["re_run"] is None
    assert any("never re-measured" in u for u in analysis.unmeasured)
    check = make_check(tmp_path)
    check.runs = runs
    assert check.build()["verdict"] == "error"


def test_requests_the_benchmark_never_ended_are_caught():
    metrics = pokec.parse_metrics(
        'operations_total{dataset="",dataset_size="",name="single_vertex_read",spawn_id="0",type="",vendor="falkor"} 120\n'
        'operations_total{dataset="",dataset_size="",name="single_vertex_read",spawn_id="1",type="",vendor="falkor"} 79\n'
        'operations_total{dataset="",dataset_size="",name="single_edge_write",spawn_id="0",type="",vendor="falkor"} 50\n'
    )
    assert metrics["executed"] == {"single_vertex_read": 199, "single_edge_write": 50}
    assert pokec.missing_requests(metrics["executed"], {"single_edge_write": 50}) is None
    missing = pokec.missing_requests(
        metrics["executed"], {"single_vertex_read": 200, "single_edge_write": 50}
    )
    assert missing and "single_vertex_read 199 of 200" in missing


def test_a_ping_answered_while_the_probe_stops_is_recorded_once(tmp_path):
    server = FakeServer(hold_at=2, signal_at=2)
    probe = pokec.PingProbe(server.port, tmp_path / "ping.csv")
    probe.start()
    assert server.got.wait(5)  # PING 2 is in flight, and its reply is held
    probe.stop()
    server.release.set()  # the reply now arrives after stop() recorded the PING
    assert len(probe.samples) == server.count == 2
    assert [kind for _, _, kind in probe.samples] == ["ok", "inflight"]
    assert not probe._thread.is_alive()  # stop() unblocked the probe's recv


def test_one_timeout_on_each_side_still_asks_for_a_re_run():
    runs = pass_runs(1, 3)
    runs[0].errors = {THREE_HOP: {"timeouts": 1, "failures": 0}}  # baseline rep 1
    runs[1].errors = {THREE_HOP: {"timeouts": 1, "failures": 0}}  # candidate rep 1
    assert analyse(runs).rerun == {("medium", "write5")}


def test_a_finished_run_with_no_ping_answered_is_unmeasured():
    runs = pass_runs(1, 3)
    runs[1].ping = {
        "ping_max_ms": None,
        "ping_answered": 0,
        "ping_errors": 140,
    }  # candidate rep 1: every PING failed
    analysis = analyse(runs, rerun_done={("medium", "write5")})
    assert any("no PING answered during a finished candidate run" in u for u in analysis.unmeasured)


def test_development_options_must_schedule_runs(tmp_path):
    base = ["--candidate", CAND, "--baseline", BASE, "--out", str(tmp_path), "--mode", "nightly"]
    for bad in (["--reps", "0"], ["--reps", "-1"], ["--workloads", "typo"], ["--workloads", ","]):
        with pytest.raises(SystemExit):
            pokec.parse_args([*base, *bad])
    assert pokec.parse_args([*base, "--workloads", "read,write5", "--reps", "1"]).reps == 1
    check = make_check(tmp_path)  # and a result with no run at all is never a pass
    result = check.build()
    assert result["verdict"] == "error" and "no benchmark run was made" in result["error"]


def test_the_timeout_test_needs_the_requested_timeout_to_apply():
    ok = {"timed_out": True, "elapsed_s": 2.503, "reply": "Query timed out"}
    assert pokec.timeout_problem(ok, 1000) is None
    assert pokec.timeout_problem(ok, 0) is None
    # a 5 s default that ignores `timeout 2500` must not pass
    assert "outside" in pokec.timeout_problem({**ok, "elapsed_s": 5.01}, 5000)
    assert "outside" in pokec.timeout_problem({**ok, "elapsed_s": 1.002}, 1000)
    assert "inside" in pokec.timeout_problem(ok, 2500)
    assert "ended" in pokec.timeout_problem(
        {"timed_out": False, "elapsed_s": 0.012, "reply": "[[1]]"}, 1000
    )


def test_a_failed_benchmark_start_still_stops_the_probe(tmp_path, monkeypatch):
    server = FakeServer()
    check = make_check(tmp_path)
    run = make_run("candidate", 1, 1)
    run.deadline_s = 30
    started = []
    real = pokec.PingProbe.start

    def start(self):
        started.append(self)
        real(self)

    monkeypatch.setattr(pokec.PingProbe, "start", start)
    with pytest.raises(OSError):
        check._drive(run, server.port, [str(tmp_path / "no-such-benchmark")], {}, tmp_path)
    assert started and not started[0]._thread.is_alive()
    assert (tmp_path / "ping.csv").exists() and run.ping is not None


def test_only_this_run_s_files_are_artifacts(tmp_path):
    out = tmp_path / "out"
    (out / "raw").mkdir(parents=True)
    (out / "raw" / "queries-large-read.jsonl").write_text("an earlier run's")
    check = make_check(tmp_path)
    artifacts = check.build()["artifacts"]
    assert "raw/queries-large-read.jsonl" not in artifacts
    assert {"raw/check.log", "raw/rows.csv", "raw/runs.csv"} <= set(artifacts)


# --- what _run_on reads after the benchmark (no Docker: the files are written here) ---

V2 = "CYPHER id = 9 MATCH (n:User) RETURN n LIMIT 1"


def bench_output(tmp_path, *, executed, rates, log_lines):
    """A query file and the run folder the benchmark would leave behind."""
    qpath = tmp_path / "q.jsonl"
    rows = [
        {"size": 3, "catalog": []},
        {"q_name": "vertex_on_label_property", "q_type": "Read", "cypher": V1},
        {"q_name": "vertex_on_label_property_index", "q_type": "Read", "cypher": V1},
        {"q_name": "single_vertex_read", "q_type": "Read", "cypher": V2},
    ]
    qpath.write_text("".join(json.dumps(r) + "\n" for r in rows))
    folder = tmp_path / "run"
    (folder / "bench" / "falkor").mkdir(parents=True)
    (folder / "run.log").write_text(
        "".join(
            f"2026-11-24T09:31:42Z ERROR benchmark::falkor::falkor_driver: Error executing query: {q}, the error is: {e}\n"
            for q, e in log_lines
        )
    )
    prom = [
        f'operations_total{{name="{q}",spawn_id="0",vendor="falkor"}} {n}'
        for q, n in executed.items()
    ]
    prom += [
        f'benchmark_query_failure_rate_pct{{query="{q}",vendor="falkor"}} {r}'
        for q, r in rates.items()
    ]
    prom += [
        f'falkordb_query_latency_pct_us{{pct="{p}",query="{q}"}} 500'
        for q in executed
        for p in ("50", "95", "99")
    ]
    prom += [f"falkordb_latency_{p}_us 500" for p in ("p50", "p95", "p99")]
    (folder / "bench" / "falkor" / "metrics.prom").write_text("\n".join(prom) + "\n")
    (folder / "bench" / "falkor" / "meta.json").write_text(json.dumps({"elapsed_ms": 25000}))
    return pokec.read_query_file(qpath), folder


ALL_ENDED = {
    "vertex_on_label_property": 1,
    "vertex_on_label_property_index": 1,
    "single_vertex_read": 1,
}


def test_a_finished_run_splits_a_shared_text_failure(tmp_path):
    qfile, folder = bench_output(
        tmp_path,
        executed=ALL_ENDED,
        rates={"vertex_on_label_property_index": 100.0},
        log_lines=[(V1, "Elapsed(())")],
    )
    check = make_check(tmp_path)
    run = pokec.Run("medium", "write5", "candidate", 1, 1)  # as _run_on hands it over
    check._read_output(run, qfile, folder, 0)
    assert run.finished and run.stop_reason is None
    assert run.errors == {"vertex_on_label_property_index": {"timeouts": 1, "failures": 0}}


def test_a_run_that_ended_too_few_requests_did_not_finish(tmp_path):
    # single_vertex_read: 1 generated, 0 ended; the request that never ended was not logged either
    qfile, folder = bench_output(
        tmp_path, executed={**ALL_ENDED, "single_vertex_read": 0}, rates={}, log_lines=[]
    )
    check = make_check(tmp_path)
    run = pokec.Run("medium", "write5", "candidate", 1, 1)  # as _run_on hands it over
    check._read_output(run, qfile, folder, 0)
    assert not run.finished and "single_vertex_read 0 of 1" in run.stop_reason


def test_a_short_run_is_not_finished_before_the_log_cross_check(tmp_path):
    # Rates are over the requests that ended (1 of 2 failed = 50 %); against the 2 generated
    # that reads 1, the log has 1: fine. With 1 of 1 ended and failed (100 %) the rate times
    # the generated 2 reads 2, the log 1: the cross-check would call it a parser error.
    rows_q = {
        "vertex_on_label_property": 1,
        "vertex_on_label_property_index": 1,
        "single_vertex_read": 1,
    }
    qfile, folder = bench_output(
        tmp_path,
        executed={**rows_q, "single_vertex_read": 0},
        rates={"vertex_on_label_property": 100.0},
        log_lines=[],
    )
    qfile.counts["vertex_on_label_property"] = 2  # 2 generated, 1 ended (and failed), 1 lost
    check = make_check(tmp_path)
    run = pokec.Run("medium", "write5", "candidate", 1, 1)  # as _run_on hands it over
    check._read_output(run, qfile, folder, 0)  # no SetupError
    assert not run.finished and "ended" in run.stop_reason


def test_the_probe_stops_even_when_its_csv_cannot_be_written(tmp_path):
    server = FakeServer()
    probe = pokec.PingProbe(server.port, tmp_path / "no-such-dir" / "ping.csv")
    probe.start()
    probe.stop()  # an OSError here would replace a stop signal unwinding through _drive
    assert probe.summary()["ping_samples"] >= 0


# --- cut round: alignment rules ------------------------------------------------------


def test_the_threshold_is_strict_on_ratio_and_floor():
    # exactly 1.3x, or exactly the floor's delta, does not cross
    assert pokec.crossed("p50_ms", 13.0, 10.0, "higher_is_worse") == 0  # ratio 1.3, delta 3 > 1
    assert (
        pokec.crossed("p99_ms", 65.0, 50.0, "higher_is_worse") == 0
    )  # ratio 1.3, delta 15 = floor
    assert pokec.crossed("p99_ms", 66.0, 50.0, "higher_is_worse") == 1  # 1.32x, +16 ms
    assert pokec.crossed("p99_ms", 40.0, 25.0, "higher_is_worse") == 0  # 1.6x, +15 ms = floor
    assert pokec.crossed("throughput", 100.0, 130.0, "lower_is_worse") == 0  # exactly 1/1.3
    assert pokec.crossed("throughput", 100.0, 131.0, "lower_is_worse") == 1


def test_findings_carry_no_ok_rows(tmp_path):
    check = make_check(tmp_path)
    check.runs = pass_runs(1, 3) + pass_runs(
        1, 3, baseline={"workload": "read"}, candidate={"workload": "read"}
    )
    result = check.build()
    assert result["verdict"] == "pass" and result["findings"] == []
    rows = (Path(check.out) / "raw" / "rows.csv").read_text()
    assert "benchmark run finished [5% writes],completed,1,1" in rows  # the ok rows live here


# --- review round 3 (bot threads on 46d931e4b) ---------------------------------------


def test_mixed_kinds_on_a_shared_text_are_not_guessed():
    shared = "vertex_on_label_property|vertex_on_label_property_index"
    errors = {shared: {"timeouts": 1, "failures": 1}}
    counts = {"vertex_on_label_property": 1, "vertex_on_label_property_index": 1}
    rates = {
        "vertex_on_label_property": {"failure": 100.0},
        "vertex_on_label_property_index": {"failure": 100.0},
    }
    assert "cannot be told apart" in pokec.attribute_ambiguous(errors, counts, rates)[1]
    # one kind, or one type to give it to: still placed
    assert (
        pokec.attribute_ambiguous({shared: {"timeouts": 2, "failures": 0}}, counts, rates)[1]
        is None
    )


def test_a_finished_run_needs_the_percentiles_of_what_succeeded():
    counts = {"single_vertex_read": 2, "single_edge_write": 1}
    full = {"p50_ms": 1.0, "p95_ms": 2.0, "p99_ms": 3.0}
    metrics = {"overall": dict(full), "per_query": {"single_vertex_read": dict(full)}}
    errors = {"single_edge_write": {"timeouts": 0, "failures": 1}}  # its one request failed: exempt
    assert pokec.missing_percentiles(metrics, counts, errors) is None
    assert "single_edge_write" in pokec.missing_percentiles(metrics, counts, {})
    assert "overall" in pokec.missing_percentiles({**metrics, "overall": {}}, counts, errors)
    all_failed = {"single_vertex_read": {"failures": 2}, "single_edge_write": {"failures": 1}}
    assert pokec.missing_percentiles({"overall": {}, "per_query": {}}, counts, all_failed) is None


def test_an_empty_overall_histogram_is_missing_not_zero():
    metrics = pokec.parse_metrics(
        "falkordb_latency_p50_us 0\nfalkordb_latency_p95_us 0\nfalkordb_latency_p99_us 0\n"
    )
    assert metrics["overall"] == {}
    slow = {"workload": "read", "scale": 2.0}
    read = {"workload": "read"}
    runs = pass_runs(1, 3, candidate=slow, baseline=read) + pass_runs(
        2, 2, candidate=slow, baseline=read
    )
    for run in runs:
        if run.pass_no == 2 and run.side == "baseline":
            run.overall = {}  # every baseline request of the re-run failed
    analysis = analyse(runs, rerun_done={("medium", "read")})
    row = next(
        r
        for r in analysis.rows
        if r["name"] == "all queries [read-only]" and r["metric"] == "p99_ms"
    )
    assert row["first_pass"] == 1 and row["re_run"] is None
    assert any("all queries [read-only] p99_ms" in u for u in analysis.unmeasured)


def test_an_interrupted_re_run_does_not_confirm():
    slow = {
        "workload": "read",
        "overrides": {"single_vertex_read": {"p50_ms": 10.0, "p95_ms": 50.0, "p99_ms": 300.0}},
    }
    read = {"workload": "read"}
    runs = pass_runs(1, 3, candidate=slow, baseline=read) + pass_runs(
        2, 1, candidate=slow, baseline=read
    )  # 1 of 2
    analysis = analyse(runs, rerun_done={("medium", "read")})
    assert not statuses(analysis, "regression")
    assert any("never re-measured" in u for u in analysis.unmeasured)


def test_a_pending_ping_is_not_an_answer():
    runs = pass_runs(1, 3)
    runs[1].ping = {
        "ping_max_ms": 50.0,
        "ping_answered": 0,
    }  # every PING failed; the last was 50 ms in flight
    analysis = analyse(runs, rerun_done={("medium", "write5")})
    assert any("no PING answered" in u for u in analysis.unmeasured)


def test_the_log_never_raises(tmp_path, capsys):
    log = pokec.Log(tmp_path / "check.log")

    class Full:
        def write(self, _):
            raise OSError(28, "No space left on device")

        def flush(self):
            raise OSError(28, "No space left on device")

    log.file = Full()
    log("ERROR: something")  # must not raise: main() still writes the result and cleans up
    assert "ERROR: something" in capsys.readouterr().out


def test_a_finished_run_without_a_succeeded_query_s_percentiles_is_rejected(tmp_path):
    qfile, folder = bench_output(tmp_path, executed=ALL_ENDED, rates={}, log_lines=[])
    prom = folder / "bench" / "falkor" / "metrics.prom"
    prom.write_text(
        "".join(
            line + "\n"
            for line in prom.read_text().splitlines()
            if 'query="single_vertex_read"' not in line or "operations_total" in line
        )
    )
    check = make_check(tmp_path)
    run = pokec.Run("medium", "write5", "candidate", 1, 1)
    with pytest.raises(pokec.SetupError, match="single_vertex_read"):
        check._read_output(run, qfile, folder, 0)


# --- the move into bench (falkorbench.kit), kept as tests ---------------------------


def test_the_log_survives_a_closed_stdout(tmp_path, monkeypatch):
    class Closed:
        def write(self, _):
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self):
            raise BrokenPipeError(32, "Broken pipe")

    log = pokec.Log(tmp_path / "check.log")
    monkeypatch.setattr("sys.stdout", Closed())
    log("verdict regression")  # `| tee` closed: the result write must still happen
    assert "verdict regression" in (tmp_path / "check.log").read_text()


def test_a_relative_out_is_made_absolute(tmp_path, monkeypatch):
    # _perf-vm.yml passes `--out out`; the benchmark runs from other directories (contract v1.8).
    monkeypatch.chdir(tmp_path)
    args = make_check(tmp_path).args
    args.out = "out"
    check = pokec.Check(args)
    assert check.out == tmp_path / "out" and check.raw.is_absolute()


def test_every_container_carries_the_label_cleanup_removes(tmp_path, monkeypatch):
    seen = {}

    def start(image, name, **kwargs):
        seen.update(kwargs)
        return kit.Container(name, 12345)

    monkeypatch.setattr(kit, "start", start)
    monkeypatch.setattr(kit, "image_env", lambda image: {"FALKORDB_ARGS": "TIMEOUT 1000"})
    check = make_check(tmp_path)
    check.start("candidate", "probe")
    assert [f"{k}={v}" for k, v in seen["labels"].items()] == [check.label]
    assert seen["env"] == {"FALKORDB_ARGS": "TIMEOUT 1000 THREAD_COUNT 10"}


def test_a_docker_failure_is_an_error_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.delenv("PERF_MACHINE_JSON", raising=False)
    monkeypatch.setattr(kit, "remove_labelled", lambda label: None)

    def main(self):
        raise kit.DockerError("docker run failed (exit 125): no space left on device")

    monkeypatch.setattr(pokec.Check, "main", main)
    out = tmp_path / "out"
    pokec.main(
        [
            "--candidate",
            CAND,
            "--baseline",
            BASE,
            "--out",
            str(out),
            "--mode",
            "nightly",
            "--tools-dir",
            str(tmp_path / "tools"),
        ]
    )
    result = json.loads((out / "result.json").read_text())
    assert result["verdict"] == "error" and "no space left on device" in result["error"]
    assert "crashed" not in result["error"]


def test_rank_puts_the_worst_first_within_a_status():
    rows = [
        {**kit.finding(f"q{i}", "p99_ms", 2.0, 1.0, "regression"), "_sort": s}
        for i, s in ((1, 1.5), (2, 9.0), (3, 3.0))
    ]
    assert [f["name"] for f in pokec.rank(rows)] == ["q2", "q3", "q1"]
    assert all(not k.startswith("_") for f in pokec.rank(rows) for k in f)


def test_a_re_run_with_one_unmeasured_run_does_not_confirm():
    slow = {
        "workload": "read",
        "overrides": {"single_vertex_read": {"p50_ms": 10.0, "p95_ms": 50.0, "p99_ms": 300.0}},
    }
    read = {"workload": "read"}
    runs = pass_runs(1, 3, candidate=slow, baseline=read) + pass_runs(
        2, 2, candidate=slow, baseline=read
    )
    rerun_base = [r for r in runs if r.pass_no == 2 and r.side == "baseline"]
    rerun_base[0].per_query.pop("single_vertex_read")  # all its requests timed out in that run
    analysis = analyse(runs, rerun_done={("medium", "read")})
    assert not statuses(analysis, "regression")  # one measured re-run run is not the re-run
    assert any("single_vertex_read [read-only] p99_ms" in u for u in analysis.unmeasured)


# --- bot review of f0cbf018a ---------------------------------------------------------


def test_a_raw_table_that_cannot_be_written_keeps_the_regression(tmp_path):
    check = make_check(tmp_path)
    hung = {
        "finished": False,
        "ping": 5100.0,
        "errors": {THREE_HOP: {"timeouts": 30, "failures": 0}},
    }
    check.runs = pass_runs(1, 3, candidate=hung)
    (check.raw / "rows.csv").mkdir()  # cannot be opened as a file
    result = check.build()
    assert result["verdict"] == "regression" and result["findings"]
    assert "raw/rows.csv" not in result["artifacts"]


def test_the_ping_finding_s_threshold_is_the_ratio_it_must_pass():
    analysis = analyse(pass_runs(1, 3, candidate={"finished": False, "ping": 30000.0}))
    finding = next(f for f in analysis.findings if f["metric"] == "ping_max_ms")
    assert finding["status"] == "regression" and finding["threshold"] == pokec.RATIO_THRESHOLD


def test_non_finite_exporter_values_are_not_measurements():
    metrics = pokec.parse_metrics(
        'falkordb_query_latency_pct_us{pct="99",query="q"} +Inf\n'
        'falkordb_query_latency_pct_us{pct="50",query="q"} 400\n'
        "falkordb_latency_p99_us NaN\n"
    )
    assert metrics["per_query"] == {"q": {"p50_ms": 0.4}} and "p99_ms" not in metrics["overall"]
