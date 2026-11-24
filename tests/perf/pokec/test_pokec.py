"""Unit tests for the pokec check: parsing, the comparison rules and result.json.

No Docker, no network: runs are built in memory. `pytest tests/perf` (perf-contract.yml)
runs them. Every result the tests build is checked with tests/perf/validate.py, the
contract's validator, which lands on main with #3986 (as does perf-contract.yml, the CI
job that runs these tests). Until then those checks report as skipped, by name; the
comparison assertions in the same tests still run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pokec = load("pokec_check", HERE / "pokec.py")
VALIDATE = HERE.parent / "validate.py"

CAND = "ghcr.io/falkordb/falkordb-server@sha256:" + "a" * 64
BASE = "falkordb/falkordb-server@sha256:" + "b" * 64
THREE_HOP = "aggregate_expansion_3_with_filter"


def validate(result: dict) -> list:
    if not VALIDATE.exists():
        # Deliberate: the validator and the CI job that runs this file arrive together (#3986).
        pytest.skip("result.json not validated: tests/perf/validate.py (#3986) is not on this branch yet")
    validator = load("perf_validate", VALIDATE)
    return [str(p) for p in validator.validate_result(json.loads(json.dumps(result)))]


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
    f"Error executing query: {Q3}, the error is: RedisError(\"Query timed out\")\n"
    "    at src/falkor/falkor_driver.rs:1244\n"
    f"2026-11-24T09:31:42Z ERROR benchmark::falkor::falkor_driver: Error executing query: {Q3}, the error is: Elapsed(())\n"
    f"2026-11-24T09:31:43Z ERROR benchmark::falkor::falkor_driver: Error executing query: {QW}, the error is: ParsingError(\"x\")\n"
    "2026-11-24T09:31:44Z ERROR benchmark::falkor::falkor_driver: Error executing query: CYPHER id = 1 RETURN 9, "
    "the error is: ParsingError(\"x\")\n"
    # The worker's own line repeats the error inside OtherError; it is not a second failure.
    f"2026-11-24T09:31:45Z  INFO benchmark: worker 3 failed to process query, not sleeping for 3 seconds "
    f"OtherError(\"Error (type x) executing query: {Q3}, the error is: y\")\n"
)


def test_parse_errors_counts_each_failed_request_once():
    errors, unmatched = pokec.parse_errors(LOG, {Q3: THREE_HOP, QW: "single_vertex_update"})
    assert errors == {THREE_HOP: {"timeouts": 2, "failures": 0}, "single_vertex_update": {"timeouts": 0, "failures": 1}}
    assert unmatched == ["CYPHER id = 1 RETURN 9"]


def test_read_query_file(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(
        json.dumps({"size": 2, "catalog": [{"id": 10, "name": THREE_HOP, "q_type": "Read"}]}) + "\n"
        + json.dumps({"q_name": THREE_HOP, "q_type": "Read", "cypher": Q3}) + "\n"
        + json.dumps({"q_name": "single_vertex_update", "q_type": "Write", "cypher": QW}) + "\n"
    )
    qfile = pokec.read_query_file(path)
    assert qfile.counts == {THREE_HOP: 1, "single_vertex_update": 1}
    assert qfile.by_text == {Q3: THREE_HOP, QW: "single_vertex_update"}


def test_hop_label_names_the_3_hop_query():
    assert pokec.hop_label(THREE_HOP) == "3-hop expansion, age >= 18 (aggregate_expansion_3_with_filter)"
    assert pokec.hop_label("aggregate_expansion_3") == "3-hop expansion (aggregate_expansion_3)"
    assert pokec.hop_label("single_vertex_read") == "single_vertex_read"


# --- the comparison ----------------------------------------------------------

QUERIES = {"single_vertex_read": 200, THREE_HOP: 200, "single_edge_write": 50}


def qfiles(workloads=("read", "write5")):
    files = {}
    for key in workloads:
        files[("medium", key)] = pokec.QueryFile(Path("q.jsonl"), dict(QUERIES), {}, {})
    return files


def make_run(side, pass_no, rep, workload="write5", scale=1.0, finished=True, ping=5.0, errors=None,
             overrides=None, stop_reason=None):
    per_query = {q: {"p50_ms": 10.0 * scale, "p95_ms": 50.0 * scale, "p99_ms": 100.0 * scale} for q in QUERIES}
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
    return pokec.analyse(runs, pokec.MODES[mode], qfiles(), [], set() if rerun_done is None else rerun_done)


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
    slow = {"workload": "read", "overrides": {"single_vertex_read": {"p50_ms": 10.0, "p95_ms": 50.0, "p99_ms": 200.0}}}
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
    analysis = analyse(pass_runs(1, 3, candidate=tail) + pass_runs(2, 2, candidate=tail),
                       rerun_done={("medium", "write5")})
    assert not analysis.rerun and not statuses(analysis, "regression")
    row = next(r for r in analysis.rows if r["name"] == "single_vertex_read [5% writes]" and r["metric"] == "p99_ms")
    assert row["use"] == "row" and row["first_pass"] == 1
    median = {"overrides": {"single_vertex_read": {"p50_ms": 20.0, "p95_ms": 50.0, "p99_ms": 100.0}}}
    analysis = analyse(pass_runs(1, 3, candidate=median) + pass_runs(2, 2, candidate=median),
                       rerun_done={("medium", "write5")})
    assert statuses(analysis, "regression") == [("single_vertex_read [5% writes]", "p50_ms")]


def test_under_writes_write_queries_never_gate():
    files = qfiles()
    files[("medium", "write5")].types["single_edge_write"] = "Write"
    slow = {"overrides": {"single_edge_write": {"p50_ms": 900.0, "p95_ms": 2000.0, "p99_ms": 3000.0}}}
    runs = pass_runs(1, 3, candidate=slow) + pass_runs(2, 2, candidate=slow)
    analysis = pokec.analyse(runs, pokec.MODES["nightly"], files, [], {("medium", "write5")})
    assert not statuses(analysis, "regression")
    assert {r["use"] for r in analysis.rows if r["name"] == "single_edge_write [5% writes]" and r["metric"] != "completed"} == {"row"}


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


def test_an_error_reply_is_a_regression_at_once():
    runs = pass_runs(1, 3)
    runs[1].errors = {"single_edge_write": {"timeouts": 0, "failures": 1}}
    analysis = analyse(runs)
    assert ("single_edge_write [5% writes]", "completed") in statuses(analysis, "regression")


def test_errors_on_both_sides_are_info():
    broken = {"errors": {"single_edge_write": {"timeouts": 0, "failures": 50}}}
    analysis = analyse(pass_runs(1, 3, candidate=broken, baseline=broken))
    assert ("single_edge_write [5% writes]", "completed") in statuses(analysis, "info")
    assert not statuses(analysis, "regression")


def test_a_run_that_does_not_finish_is_a_regression_at_once():
    analysis = analyse(pass_runs(1, 3, candidate={"finished": False, "ping": 30000.0}))
    regressions = statuses(analysis, "regression")
    assert ("benchmark run finished [5% writes]", "completed") in regressions
    assert ("PING max [5% writes]", "ping_max_ms") in regressions
    assert analysis.unmeasured  # no latency comparison without a finished candidate run


def test_broken_on_both_sides_is_info_and_unmeasured():
    analysis = analyse(pass_runs(1, 3, candidate={"finished": False}, baseline={"finished": False}))
    assert ("benchmark run finished [5% writes]", "completed") in statuses(analysis, "info")
    assert not statuses(analysis, "regression")
    assert analysis.unmeasured


def test_one_ping_freeze_asks_for_a_re_run():
    runs = pass_runs(1, 1, candidate={"ping": 2500.0})
    analysis = analyse(runs, mode="queue")
    assert analysis.rerun == {("medium", "write5")}
    analysis = analyse(runs + pass_runs(2, 1, candidate={"ping": 2400.0}), mode="queue",
                       rerun_done={("medium", "write5")})
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


def test_select_findings_caps():
    findings = [pokec.make_finding(f"r{i}", "p99_ms", 2.0, 1.0, 1.3, "higher_is_worse", "regression")
                for i in range(60)]
    findings += [pokec.make_finding(f"i{i}", "p99_ms", 1.0, 1.0, 1.3, "higher_is_worse", "info") for i in range(20)]
    selected = pokec.select_findings(findings)
    assert len(selected) == 50 and all(f["status"] == "regression" for f in selected)
    selected = pokec.select_findings(findings[60:])
    assert len(selected) == 10


# --- result.json -------------------------------------------------------------


def make_check(tmp_path, mode="nightly", **extra):
    args = argparse.Namespace(
        candidate=CAND, baseline=BASE, out=str(tmp_path / "out"), mode=mode, candidate_label="6.0.4",
        baseline_label=None, candidate_commit="6ca3adc9d0418f76c5a15f820552e0603a65e553", baseline_commit=None,
        tools_dir=str(tmp_path / "tools"), reps=None, workloads=None, **extra,
    )
    check = pokec.Check(args)
    check.machine = {"runner": "local", "type": "test", "cpu": "test", "cores": 1}
    check.query_files = qfiles()
    return check


def test_result_pass_is_valid(tmp_path):
    check = make_check(tmp_path)
    check.runs = pass_runs(1, 3) + pass_runs(1, 3, baseline={"workload": "read"}, candidate={"workload": "read"})
    result = check.result()
    assert result["verdict"] == "pass", result["summary"]
    assert result["confirmed"] is False and result["error"] is None
    assert result["baseline"]["label"] == "falkordb/falkordb-server@" + "b" * 12
    assert result["candidate"]["commit"] == "6ca3adc9d0418f76c5a15f820552e0603a65e553"
    assert "raw/rows.csv" in result["artifacts"] and (Path(check.out) / "raw" / "rows.csv").exists()
    assert validate(result) == []


def test_result_regression_is_valid_and_matches_the_acceptance_pattern(tmp_path):
    check = make_check(tmp_path)
    hung = {"finished": False, "ping": 5100.0, "errors": {THREE_HOP: {"timeouts": 30, "failures": 0}}}
    check.runs = pass_runs(1, 3, candidate=hung)
    result = check.result()
    assert result["verdict"] == "regression" and result["confirmed"] is True
    assert any(re.search("3-hop", f["name"], re.I) and f["status"] == "regression" for f in result["findings"])
    assert "\n" not in result["summary"]
    assert validate(result) == []


def test_result_error_when_the_baseline_does_not_finish(tmp_path):
    check = make_check(tmp_path)
    check.runs = pass_runs(1, 3, baseline={"finished": False})
    result = check.result()
    assert result["verdict"] == "error" and "not measured" in result["error"]
    assert validate(result) == []


def test_result_error_when_setup_failed(tmp_path):
    check = make_check(tmp_path)
    check.error = "could not pull the baseline image"
    result = check.result()
    assert result["verdict"] == "error" and result["findings"] == []
    assert result["summary"].startswith("Could not measure: could not pull")
    assert validate(result) == []


def test_candidate_that_does_not_start_is_a_regression(tmp_path):
    check = make_check(tmp_path)
    check.info.append(pokec.make_finding("server started [medium]", "completed", 0, 1, None, "lower_is_worse",
                                         "regression"))
    result = check.result()
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


def test_digest_required(tmp_path):
    assert pokec.DIGEST_RE.match(CAND) and not pokec.DIGEST_RE.match("falkordb/falkordb-server:6.2.0")


# --- the RESP client and the PING probe, against a fake server ----------------


class FakeServer:
    """Replies to PING with +PONG; `stall` seconds before the n-th reply; `mute` after it."""

    def __init__(self, stall_at=None, stall=0.0, mute_at=None, reply=b"+PONG\r\n", signal_at=None):
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
        threading.Thread(target=self.serve, daemon=True).start()

    def serve(self):
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
    slow = {"workload": "read", "overrides": {"single_vertex_read": {"p50_ms": 10.0, "p95_ms": 50.0, "p99_ms": 300.0}}}
    read = {"workload": "read"}
    runs = pass_runs(1, 3, candidate=slow, baseline=read)
    runs += pass_runs(2, 2, candidate=slow, baseline={"workload": "read", "finished": False})
    analysis = analyse(runs, rerun_done={("medium", "read")})
    assert not statuses(analysis, "regression")
    assert any("never re-measured" in u for u in analysis.unmeasured)
    check = make_check(tmp_path)
    check.runs = runs
    result = check.result()
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
    result = check.result()
    assert result["verdict"] == "error" and "baseline did not finish" in result["error"]
    assert validate(result) == []


def test_the_summary_names_queries_broken_on_both_sides(tmp_path):
    broken = {"errors": {"single_edge_write": {"timeouts": 0, "failures": 50}}}
    check = make_check(tmp_path)
    check.runs = pass_runs(1, 3, candidate=broken, baseline=broken) + pass_runs(
        1, 3, candidate={"workload": "read"}, baseline={"workload": "read"})
    result = check.result()
    assert result["verdict"] == "pass"
    assert "failed on both sides (not new): single_edge_write [5% writes]" in result["summary"]
    assert validate(result) == []


def test_an_error_with_a_carriage_return_stays_one_line(tmp_path):
    check = make_check(tmp_path)
    check.error = "docker said:\r\nno space left on device"
    result = check.result()
    assert "\r" not in result["summary"] and "\n" not in result["summary"]
    assert validate(result) == []


def test_main_writes_an_error_result_for_a_tag(tmp_path, monkeypatch):
    monkeypatch.delenv("PERF_MACHINE_JSON", raising=False)
    monkeypatch.setattr(pokec.Docker, "remove_all", lambda self: None)  # no Docker in unit tests
    out = tmp_path / "out"
    code = pokec.main(["--candidate", "falkordb/falkordb-server:6.2.0", "--baseline", BASE, "--out", str(out),
                       "--mode", "nightly", "--tools-dir", str(tmp_path / "tools")])
    assert code == 0
    result = json.loads((out / "result.json").read_text())
    assert result["verdict"] == "error" and "not pinned by digest" in result["error"]


def test_no_re_run_once_the_candidate_did_not_finish():
    runs = pass_runs(1, 3, candidate={"finished": False, "errors": {THREE_HOP: {"timeouts": 30, "failures": 0}}})
    runs[1].errors["single_edge_write"] = {"timeouts": 1, "failures": 0}  # one run only: would ask for a re-run
    analysis = analyse(runs)
    assert not analysis.rerun
    assert any("no re-run" in note for note in analysis.notes)
    assert (f"{pokec.hop_label(THREE_HOP)} [5% writes]", "completed") in statuses(analysis, "regression")


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
    hung = {"finished": False, "ping": 5100.0, "errors": {THREE_HOP: {"timeouts": 30, "failures": 0}}}
    check.runs = pass_runs(1, 3, candidate=hung)
    real = check.result

    def interrupted(unfinished=None):
        raise pokec.Stopped("signal 15")

    check.result = interrupted
    with pytest.raises(pokec.Stopped):  # not swallowed into a bare error result
        check.checkpoint("after a run")
    check.result = real
    check.error = "stopped by a signal"
    result = check.write_result()
    assert result["verdict"] == "regression" and result["findings"]


def test_a_relative_out_is_made_absolute(tmp_path, monkeypatch):
    # _perf-vm.yml passes `--out out`; the benchmark runs from other directories, and a
    # relative query-file path made `generate-queries` fail with NotFound on the first VM leg.
    monkeypatch.chdir(tmp_path)
    check = make_check(tmp_path)
    check.args.out = "out"
    check = pokec.Check(check.args)
    assert check.out == tmp_path / "out" and check.raw.is_absolute()


# --- review round 2 (PR #4000 bot threads), kept as tests --------------------------

V1 = "CYPHER id = 7 MATCH (n:User {id: $id}) RETURN n"


def test_a_shared_query_text_keeps_both_types_and_is_split_from_metrics(tmp_path):
    path = tmp_path / "q.jsonl"
    lines = [{"size": 3, "catalog": []},
             {"q_name": "vertex_on_label_property", "q_type": "Read", "cypher": V1},
             {"q_name": "vertex_on_label_property_index", "q_type": "Read", "cypher": V1},
             {"q_name": "single_vertex_read", "q_type": "Read", "cypher": "CYPHER id = 8 MATCH (n) RETURN n"}]
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
    slow = {"workload": "read", "overrides": {"single_vertex_read": {"p50_ms": 10.0, "p95_ms": 50.0, "p99_ms": 300.0}}}
    read = {"workload": "read"}
    runs = pass_runs(1, 3, candidate=slow, baseline=read) + pass_runs(2, 2, candidate=slow, baseline=read)
    for run in runs:
        if run.pass_no == 2 and run.side == "baseline":
            run.per_query.pop("single_vertex_read")  # every request of it failed in the re-run
    analysis = analyse(runs, rerun_done={("medium", "read")})
    row = next(r for r in analysis.rows if r["name"] == "single_vertex_read [read-only]" and r["metric"] == "p99_ms")
    assert row["first_pass"] == 1 and row["re_run"] is None
    assert any("never re-measured" in u for u in analysis.unmeasured)
    check = make_check(tmp_path)
    check.runs = runs
    assert check.result()["verdict"] == "error"


def test_requests_the_benchmark_never_ended_are_caught():
    metrics = pokec.parse_metrics(
        'operations_total{dataset="",dataset_size="",name="single_vertex_read",spawn_id="0",type="",vendor="falkor"} 120\n'
        'operations_total{dataset="",dataset_size="",name="single_vertex_read",spawn_id="1",type="",vendor="falkor"} 79\n'
        'operations_total{dataset="",dataset_size="",name="single_edge_write",spawn_id="0",type="",vendor="falkor"} 50\n'
    )
    assert metrics["executed"] == {"single_vertex_read": 199, "single_edge_write": 50}
    assert pokec.missing_requests(metrics["executed"], {"single_edge_write": 50}) is None
    missing = pokec.missing_requests(metrics["executed"], {"single_vertex_read": 200, "single_edge_write": 50})
    assert missing and "single_vertex_read 199 of 200" in missing


def test_a_ping_answered_while_the_probe_stops_is_recorded_once(tmp_path):
    server = FakeServer(stall_at=2, stall=0.5, signal_at=2)
    probe = pokec.PingProbe(server.port, tmp_path / "ping.csv")
    probe.start()
    assert server.got.wait(5)  # PING 2 is in flight; its reply comes in 0.5 s
    probe.stop()
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
    runs[1].ping = {"ping_max_ms": None, "ping_errors": 140}  # candidate rep 1: every PING failed
    analysis = analyse(runs, rerun_done={("medium", "write5")})
    assert any("no PING answered during a finished candidate run" in u for u in analysis.unmeasured)


def test_development_options_must_schedule_runs(tmp_path):
    base = ["--candidate", CAND, "--baseline", BASE, "--out", str(tmp_path), "--mode", "nightly"]
    for bad in (["--reps", "0"], ["--reps", "-1"], ["--workloads", "typo"], ["--workloads", ","]):
        with pytest.raises(SystemExit):
            pokec.parse_args(base + bad)
    assert pokec.parse_args(base + ["--workloads", "read,write5", "--reps", "1"]).reps == 1
    check = make_check(tmp_path)  # and a result with no run at all is never a pass
    result = check.result()
    assert result["verdict"] == "error" and "no benchmark run was made" in result["error"]


def test_the_timeout_test_needs_the_requested_timeout_to_apply():
    ok = {"timed_out": True, "elapsed_s": 2.503, "reply": "Query timed out"}
    assert pokec.timeout_problem(ok, 1000) is None
    assert pokec.timeout_problem(ok, 0) is None
    # a 5 s default that ignores `timeout 2500` must not pass
    assert "outside" in pokec.timeout_problem({**ok, "elapsed_s": 5.01}, 5000)
    assert "outside" in pokec.timeout_problem({**ok, "elapsed_s": 1.002}, 1000)
    assert "inside" in pokec.timeout_problem(ok, 2500)
    assert "ended" in pokec.timeout_problem({"timed_out": False, "elapsed_s": 0.012, "reply": "[[1]]"}, 1000)


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


def test_an_earlier_run_s_files_are_not_this_run_s_artifacts(tmp_path):
    out = tmp_path / "out"
    (out / "raw").mkdir(parents=True)
    (out / "raw" / "queries-medium-read.jsonl").write_text("old")
    (out / "result.json").write_text("{}")
    check = make_check(tmp_path)
    assert not (out / "raw" / "queries-medium-read.jsonl").exists() and not (out / "result.json").exists()
    assert "raw/queries-medium-read.jsonl" not in check.result()["artifacts"]


def test_a_signal_during_the_first_checkpoint_still_writes_the_result(tmp_path, monkeypatch):
    monkeypatch.delenv("PERF_MACHINE_JSON", raising=False)
    removed = []
    monkeypatch.setattr(pokec.Docker, "remove_all", lambda self: removed.append(True))
    real = pokec.Check.checkpoint

    def checkpoint(self, where):
        if where == "at start":
            raise pokec.Stopped("signal 15")
        real(self, where)

    monkeypatch.setattr(pokec.Check, "checkpoint", checkpoint)
    out = tmp_path / "out"
    assert pokec.main(["--candidate", CAND, "--baseline", BASE, "--out", str(out), "--mode", "nightly",
                       "--tools-dir", str(tmp_path / "tools")]) == 0
    result = json.loads((out / "result.json").read_text())
    assert result["verdict"] == "error" and "stopped by a signal" in result["error"] and removed


def test_containers_are_removed_even_when_the_final_write_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("PERF_MACHINE_JSON", raising=False)
    removed = []
    monkeypatch.setattr(pokec.Docker, "remove_all", lambda self: removed.append(True))
    monkeypatch.setattr(pokec.Check, "main", lambda self: None)

    def write_result(self, unfinished=None):
        if unfinished is None:
            raise OSError(28, "No space left on device")
        return {}

    monkeypatch.setattr(pokec.Check, "write_result", write_result)
    with pytest.raises(OSError):
        pokec.main(["--candidate", CAND, "--baseline", BASE, "--out", str(tmp_path / "out"), "--mode", "nightly",
                    "--tools-dir", str(tmp_path / "tools")])
    assert removed
