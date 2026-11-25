"""kit.py: the contract plumbing, the threshold test and the docker calls the checks share.

The docker tests run against tests/fake_docker.py on PATH (no Docker needed):
each "container" is a small RESP server, and every docker call is logged so a
test can assert the exact argv kit sends.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from falkorbench import contract
from falkorbench import kit

FAKE_DOCKER = Path(__file__).with_name("fake_docker.py")
# The acceptance images: main@4c1088949 (with its rc tag) and 6.2.1.
MAIN = (
    "ghcr.io/falkordb/falkordb-server:rc-mg-4c1088949a5852e3949c05bfb3d92a603c4cd4ca-valkey"
    "@sha256:95b610a600682533d854b6ece6bddbea80b49d0b89e2cd0b2cf19bd0a8142aed"
)
V621 = "falkordb/falkordb-server@sha256:a8fb264e6b90db5c1e6e3a42af5128efcc13590ce2ca8e65d870b456806f24de"
COMMIT = "4c1088949a5852e3949c05bfb3d92a603c4cd4ca"


@pytest.fixture(autouse=True)
def _no_vm_env(monkeypatch):
    for var in ("PERF_MACHINE_JSON", "PERF_SERVER_CPUS"):
        monkeypatch.delenv(var, raising=False)


def make_args(tmp_path, **changes):
    values = {
        "candidate": MAIN,
        "baseline": V621,
        "out": str(tmp_path / "out"),
        "mode": "nightly",
        "candidate_label": None,
        "baseline_label": None,
        "candidate_commit": None,
        "baseline_commit": None,
    }
    return SimpleNamespace(**{**values, **changes})


def assert_valid(result):
    assert contract.validate_result(json.loads(json.dumps(result))) == []


def test_kit_imports_only_the_standard_library_and_the_contract():
    tree = ast.parse(Path(kit.__file__).read_text(encoding="utf-8"))
    roots = {
        a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    }
    froms = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert all(n.level == 0 for n in ast.walk(tree) if isinstance(n, ast.ImportFrom))
    roots |= {m.split(".")[0] for m in froms if m != "falkorbench.contract"}
    assert roots <= sys.stdlib_module_names, sorted(roots - sys.stdlib_module_names)


# --- the contract's flags -------------------------------------------------------


def test_add_args_parses_the_contract_flags():
    parser = argparse.ArgumentParser()
    kit.add_args(parser)
    required = ["--candidate", MAIN, "--baseline", V621, "--out", "out", "--mode", "long"]
    args = parser.parse_args(
        [*required, "--candidate-label", "main@4c10889", "--baseline-commit", COMMIT]
    )
    assert (args.candidate, args.baseline, args.out, args.mode) == (MAIN, V621, "out", "long")
    assert (args.candidate_label, args.baseline_label) == ("main@4c10889", None)
    assert (args.candidate_commit, args.baseline_commit) == (None, COMMIT)


@pytest.mark.parametrize(
    "argv",
    [
        ["--baseline", V621, "--out", "o", "--mode", "nightly"],  # no candidate
        ["--candidate", MAIN, "--baseline", V621, "--out", "o"],  # no mode
        ["--candidate", MAIN, "--baseline", V621, "--out", "o", "--mode", "weekly"],
    ],
)
def test_add_args_refuses_missing_flags_and_unknown_modes(argv):
    parser = argparse.ArgumentParser()
    kit.add_args(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


@pytest.mark.parametrize(
    ("image", "label"),
    [
        (MAIN, "ghcr.io/falkordb/falkordb-server@95b610a60068"),
        (V621, "falkordb/falkordb-server@a8fb264e6b90"),
        ("localhost:5000/x/y:t@sha256:" + "ab" * 32, "localhost:5000/x/y@abababababab"),
        ("falkordb/falkordb-server:edge", "falkordb/falkordb-server"),
    ],
)
def test_default_label_drops_the_tag_and_keeps_a_registry_port(image, label):
    assert kit.default_label(image) == label


def test_side_defaults_a_blank_label_and_nulls_a_bad_commit(tmp_path):
    args = make_args(tmp_path, candidate_label="  ", candidate_commit=COMMIT.upper())
    assert kit.side(args, "candidate") == {
        "image": MAIN,
        "commit": None,
        "label": "ghcr.io/falkordb/falkordb-server@95b610a60068",
    }
    args = make_args(tmp_path, baseline_label=" 6.2.1 ", baseline_commit=COMMIT)
    assert kit.side(args, "baseline") == {"image": V621, "commit": COMMIT, "label": "6.2.1"}
    # A click command passes only what it has: absent optional attributes are fine.
    assert kit.side(SimpleNamespace(baseline=V621), "baseline")["label"].endswith("@a8fb264e6b90")


def test_side_nulls_a_commit_with_a_trailing_newline(tmp_path):
    args = make_args(tmp_path, baseline_commit=COMMIT + "\n")
    assert kit.side(args, "baseline")["commit"] is None  # fullmatch, not match


def test_input_errors_accepts_the_acceptance_images(tmp_path):
    args = make_args(tmp_path, candidate_commit=COMMIT, baseline_label="6.2.1")
    assert kit.input_errors(args) == []


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        ({"candidate": "falkordb/falkordb-server:edge"}, "--candidate"),
        ({"baseline": V621.replace("falkordb/", "FalkorDB/", 1)}, "--baseline"),  # uppercase repo
        ({"baseline": V621 + "\n"}, "--baseline"),  # fullmatch, not match
        ({"candidate": MAIN[:-1]}, "--candidate"),  # 63 hex
        ({"candidate_commit": COMMIT.upper()}, "--candidate-commit"),
        ({"baseline_commit": COMMIT[:12]}, "--baseline-commit"),
        ({"candidate_commit": COMMIT + "\n"}, "--candidate-commit"),  # fullmatch, not match
        ({"candidate_label": "main\n@4c10889"}, "--candidate-label"),
        ({"baseline_label": " "}, "--baseline-label"),
        ({"mode": "weekly"}, "--mode"),
    ],
)
def test_input_errors_names_the_flag(tmp_path, changes, fragment):
    errors = kit.input_errors(make_args(tmp_path, **changes))
    assert len(errors) == 1
    assert errors[0].startswith(fragment + " ")


def test_out_dir_resolves_a_relative_out_against_the_start_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = kit.out_dir("out")  # what _perf-vm.yml passes (contract v1.8)
    monkeypatch.chdir("/")
    assert out == tmp_path.resolve() / "out"
    assert (out / "raw").is_dir()


def test_result_finished_is_never_before_started(tmp_path):
    later = "2099-01-01T00:00:00Z"  # `started` from another clock (micro's preflight runner)
    res = kit.result("micro-hw", make_args(tmp_path), started=later, findings=[], summary="s")
    assert res["finished"] == later
    assert_valid(res)


def test_utc_now_is_a_contract_timestamp():
    assert contract.TIMESTAMP_RE.fullmatch(kit.utc_now())


EPYC = "processor\t: 0\nmodel name\t: AMD EPYC 9B14\n"
ARM = "processor\t: 0\nFeatures\t: fp asimd\n"  # arm64 /proc/cpuinfo has no model name


@pytest.mark.parametrize(
    ("system", "cpuinfo", "sysctl", "processor", "arch", "cpu"),
    [
        ("linux", EPYC, None, "", "x86_64", "AMD EPYC 9B14"),
        ("linux", ARM, None, "", "aarch64", "aarch64"),
        ("linux", OSError, None, "", "aarch64", "aarch64"),  # no /proc
        ("darwin", None, "Apple M5\n", "arm", "arm64", "Apple M5"),
        ("darwin", None, OSError, "arm", "arm64", "arm"),  # no sysctl
        ("linux", OSError, None, "", "", "unknown"),
    ],
)
def test_machine_cpu_from_the_host_with_its_fallbacks(
    monkeypatch, system, cpuinfo, sysctl, processor, arch, cpu
):
    def read_text(path):
        assert str(path) == "/proc/cpuinfo"
        if cpuinfo is OSError:
            raise OSError("no /proc")
        return cpuinfo

    def run(argv, **_kw):
        assert argv[:2] == ["sysctl", "-n"]
        if sysctl is OSError:
            raise OSError("no sysctl")
        return subprocess.CompletedProcess(argv, 0, sysctl, "")

    monkeypatch.setattr(kit.sys, "platform", system)
    monkeypatch.setattr(kit.Path, "read_text", read_text)
    monkeypatch.setattr(kit.subprocess, "run", run)
    monkeypatch.setattr(kit.platform, "processor", lambda: processor)
    monkeypatch.setattr(kit.platform, "machine", lambda: arch)
    assert kit.machine()["cpu"] == cpu


def test_machine_copies_perf_machine_json(tmp_path, monkeypatch):
    vm = {"runner": "gce", "type": "n4-standard-16", "cpu": "INTEL(R) XEON(R)", "cores": 16}
    (tmp_path / "machine.json").write_text(json.dumps({**vm, "zone": math.nan}))
    monkeypatch.setenv("PERF_MACHINE_JSON", str(tmp_path / "machine.json"))
    assert kit.machine() == vm  # only the contract's four fields: a stray NaN stays out


VM = '"runner": "gce", "type": "n4-standard-16", "cpu": "x"'


@pytest.mark.parametrize(
    "content",
    [
        None,
        "{not json",
        '["a list"]',
        '{"type": "no runner"}',
        '{"runner": "gce"}',
        "{" + VM + ', "cores": "16"}',
        "{" + VM + ', "cores": true}',
        "{" + VM + ', "cores": NaN}',
        "{" + VM + ', "cores": 0}',
        '{"runner": "gce", "type": " ", "cpu": "x", "cores": 16}',
    ],
)
def test_machine_describes_this_host_when_perf_machine_json_is_unusable(
    tmp_path, monkeypatch, capsys, content
):
    path = tmp_path / "machine.json"
    if content is not None:
        path.write_text(content)
    monkeypatch.setenv("PERF_MACHINE_JSON", str(path))
    found = kit.machine()
    assert found["runner"] == "local"
    assert "PERF_MACHINE_JSON" in capsys.readouterr().err
    machine = {k: found[k] for k in ("runner", "type", "cpu", "cores")}
    assert found == machine and all(machine.values()) and isinstance(machine["cores"], int)


# --- findings, verdict, result.json ------------------------------------------------


def test_finding_computes_and_rounds_the_ratio():
    f = kit.finding("UNWIND id seek @200k", "p50_ms", 13.3, 0.4, "regression", threshold=1.5)
    assert f == {
        "name": "UNWIND id seek @200k",
        "metric": "p50_ms",
        "candidate": 13.3,
        "baseline": 0.4,
        "ratio": 33.25,
        "threshold": 1.5,
        "direction": "higher_is_worse",
        "status": "regression",
    }
    assert kit.finding("q", "p50_ms", 1 / 3, 1.0, "info")["candidate"] == 0.333333
    assert kit.finding("q", "instructions", 2_500_000.25, 1, "info")["candidate"] == 2_500_000.2


def test_finding_nulls_what_json_cannot_hold_and_ratios_that_mean_nothing():
    assert kit.finding("q", "p50_ms", 5.0, 0, "info")["ratio"] is None
    nan = kit.finding("q", "p50_ms", math.nan, math.inf, "info", threshold=math.nan)
    assert (nan["candidate"], nan["baseline"], nan["ratio"], nan["threshold"]) == (None,) * 4
    done = kit.finding("q completed", "completed", True, 1, "ok", direction="lower_is_worse")
    assert (done["candidate"], done["baseline"], done["ratio"]) == (1, 1, None)
    assert kit.finding("q", "p50_ms", 3.0, 2.0, "info", ratio=1.42)["ratio"] == 1.42
    json.dumps([nan, done], allow_nan=False)


@pytest.mark.parametrize(
    ("candidate", "baseline", "status"),
    [
        (0, 1, "regression"),
        (0, None, "regression"),  # the validator: 0 against anything but 0
        (0, 0, "info"),  # broken on both sides: not new
        (1, 0, "improvement"),
        (1, 1, "ok"),
        (1, None, "info"),  # baseline not measured: info, never ok
        (None, 1, "info"),
        (None, None, "info"),
    ],
)
def test_correctness_status(tmp_path, candidate, baseline, status):
    assert kit.correctness_status(candidate, baseline) == status
    f = kit.finding(
        "s01 completed", "completed", candidate, baseline, status, direction="lower_is_worse"
    )
    assert_valid(
        kit.result("soak", make_args(tmp_path), started=kit.utc_now(), findings=[f], summary="x")
    )


@pytest.mark.parametrize(
    ("candidate", "baseline", "threshold", "floor", "direction", "expected"),
    [
        (3.0, 1.0, 2.0, 0.0, "higher_is_worse", 1),
        (2.0, 1.0, 2.0, 0.0, "higher_is_worse", 0),  # ratio == threshold: not past it
        (2.1, 1.0, 2.0, 1.1, "higher_is_worse", 0),  # difference == floor: not past it
        (2.2, 1.0, 2.0, 1.1, "higher_is_worse", 1),
        (0.9, 0.3, 3.0, 0.1, "higher_is_worse", 0),  # 0.9 / 0.3 is 3.0; 3.0 * 0.3 < 0.9
        (0.9, 0.6, 1.5, 0.1, "higher_is_worse", 0),  # the same at 1.5x
        (0.3, 0.9, 3.0, 0.1, "higher_is_worse", 0),  # and on the improvement side
        (3668, 1310, 2.8, 2000, "higher_is_worse", 0),  # micro-hw's 2.8x tier: exactly 2.8
        (30.0, 10.0, 2.0, 25.0, "higher_is_worse", 0),  # 3x, but only 20 ms
        (1.0, 3.0, 2.0, 0.0, "higher_is_worse", -1),
        (1.0, 3.0, 2.0, 0.0, "lower_is_worse", 1),  # throughput fell to a third
        (3.0, 1.0, 2.0, 0.0, "lower_is_worse", -1),
        (0.5, 0.0, 2.0, 0.0, "higher_is_worse", 1),  # zero baseline: an infinite ratio
        (0.0, 0.0, 2.0, 0.0, "higher_is_worse", 0),
        (None, 1.0, 2.0, 0.0, "higher_is_worse", 0),
        (5.0, -1.0, 2.0, 0.0, "higher_is_worse", 0),
        (math.nan, 1.0, 2.0, 0.0, "higher_is_worse", 0),
        (math.inf, 1.0, 2.0, 0.0, "higher_is_worse", 0),  # finding() writes it as null
        (1.0, math.inf, 2.0, 0.0, "lower_is_worse", 0),
    ],
)
def test_crossed_is_strict_on_ratio_and_floor(
    candidate, baseline, threshold, floor, direction, expected
):
    assert kit.crossed(candidate, baseline, threshold, floor, direction) == expected


def test_crossed_refuses_a_mismatch_direction():
    with pytest.raises(ValueError):
        kit.crossed(3, 4, 1.5, 0.0, direction="mismatch")


def rows(status, count):
    return [kit.finding(f"{status} {i}", "p50_ms", 2.0, 1.0, status) for i in range(count)]


def test_select_orders_by_status_keeps_the_given_order_and_drops_ok():
    findings = rows("info", 15) + rows("ok", 5) + rows("improvement", 2) + rows("regression", 3)
    kept = kit.select(findings)
    assert [f["name"] for f in kept] == (
        [f"regression {i}" for i in range(3)]
        + [f"improvement {i}" for i in range(2)]
        + [f"info {i}" for i in range(10)]
    )


def test_select_refuses_a_status_the_contract_does_not_know():
    with pytest.raises(ValueError, match="regresion"):
        kit.select(rows("regresion", 1))


def test_select_caps_at_50_crossings_first():
    kept = kit.select(rows("info", 5) + rows("improvement", 10) + rows("regression", 45))
    assert len(kept) == kit.MAX_FINDINGS == 50
    assert [f["status"] for f in kept] == ["regression"] * 45 + ["improvement"] * 5


def test_verdict_regression_outranks_error_outranks_pass():
    assert (
        kit.verdict(rows("regression", 1) + rows("info", 1), ["the baseline crashed"])
        == "regression"
    )
    assert kit.verdict(rows("improvement", 1), ["the baseline crashed"]) == "error"
    assert kit.verdict(rows("improvement", 1), ["", ""]) == "pass"
    assert kit.verdict([], []) == "pass"


def test_result_pass(tmp_path):
    started = kit.utc_now()
    res = kit.result(
        "pokec",
        make_args(tmp_path, baseline_label="6.2.1", candidate_commit=COMMIT),
        started=started,
        findings=rows("ok", 3) + rows("improvement", 1) + rows("info", 2),
        summary="Pass: 30 runs\ncompared, no regression",
        artifacts=["raw/rows.csv", "raw/check.log", "raw/rows.csv"],
    )
    assert_valid(res)
    assert res["verdict"] == "pass" and res["confirmed"] is False and res["error"] is None
    assert res["summary"] == "Pass: 30 runs compared, no regression"
    assert [f["status"] for f in res["findings"]] == ["improvement", "info", "info"]
    assert res["artifacts"] == ["raw/check.log", "raw/rows.csv"]
    assert res["candidate"]["commit"] == COMMIT and res["baseline"]["label"] == "6.2.1"
    assert res["machine"]["runner"] == "local" and res["started"] == started
    assert res["finished"] >= started


def test_result_regression_outranks_its_errors(tmp_path):
    res = kit.result(
        "ldbc",
        make_args(tmp_path),
        started=kit.utc_now(),
        findings=rows("regression", 1),
        summary="IC1 p50 140x",
        errors=["IC12 not re-timed"],
    )
    assert_valid(res)
    assert (res["verdict"], res["confirmed"], res["error"]) == ("regression", True, None)


def test_result_error_carries_every_reason_once_and_on_one_line(tmp_path):
    vm = {"runner": "gce", "type": "c4-standard-16", "cpu": "x", "cores": 16}
    res = kit.result(
        "fuzz",
        make_args(tmp_path, candidate_commit="abc"),
        started=kit.utc_now(),
        findings=rows("info", 1),
        summary="",
        errors=["no C4\ncapacity", "", "--candidate-commit 'abc' is not 40 lowercase hex"],
        machine_info=vm,
    )
    assert_valid(res)
    assert res["verdict"] == "error" and res["summary"] == "error" and res["machine"] == vm
    assert res["error"] == "--candidate-commit 'abc' is not 40 lowercase hex; no C4 capacity"
    assert res["candidate"]["commit"] is None


def test_blank_errors_are_no_errors_and_a_blank_error_result_still_says_error(tmp_path):
    args, started = make_args(tmp_path), kit.utc_now()
    res = kit.result("pokec", args, started=started, findings=[], summary="s", errors=[" \n", ""])
    assert res["verdict"] == "pass" and kit.verdict([], [" \n"]) == "pass"
    res = kit.error_result("pokec", args, started, str(TimeoutError()))
    assert_valid(res)
    assert (res["verdict"], res["error"]) == ("error", "unknown error")


def test_result_caps_the_summary(tmp_path):
    res = kit.result(
        "soak", make_args(tmp_path), started=kit.utc_now(), findings=[], summary="x " * 400
    )
    assert len(res["summary"]) == kit.SUMMARY_LIMIT and res["summary"].endswith("…")


def test_error_result(tmp_path):
    res = kit.error_result("micro-hw", make_args(tmp_path), kit.utc_now(), "no C4 capacity")
    assert_valid(res)
    assert (res["verdict"], res["error"], res["findings"]) == ("error", "no C4 capacity", [])
    assert res["summary"] == "could not measure: no C4 capacity"


def test_write_result_is_atomic_and_keeps_the_old_file_on_nan(tmp_path):
    good = kit.error_result("pokec", make_args(tmp_path), kit.utc_now(), "first")
    kit.write_result(tmp_path, good)
    assert json.loads((tmp_path / "result.json").read_text()) == good
    with pytest.raises(ValueError):
        kit.write_result(tmp_path, {**good, "findings": [{"ratio": math.nan}]})
    assert json.loads((tmp_path / "result.json").read_text()) == good
    assert sorted(p.name for p in tmp_path.iterdir()) == ["result.json"]  # no temp file left


# --- always a result.json ----------------------------------------------------------


def read_result(tmp_path):
    return json.loads((tmp_path / "out" / "result.json").read_text())


def builder(tmp_path, args, findings, seen):
    def build(error):
        seen["error"], seen["stopping"] = error, kit.stopping()
        return kit.result(
            "pokec", args, started=kit.utc_now(), findings=findings, summary="s", errors=[error]
        )

    return build


@pytest.fixture
def sentinel():
    """Our own handlers on SIGTERM and SIGINT, so a test can tell they came back."""

    def handler(*_):
        raise AssertionError("a stop signal reached the test's own handler")

    before = {s: signal.signal(s, handler) for s in kit.STOP_SIGNALS}
    yield handler
    for signum, previous in before.items():
        signal.signal(signum, previous)


def test_run_check_writes_an_error_result_before_work_starts(tmp_path):
    args, seen = make_args(tmp_path), {}

    def work():
        seen["early"] = read_result(tmp_path)

    res = kit.run_check("pokec", args, work, builder(tmp_path, args, [], seen))
    assert seen["early"]["verdict"] == "error"
    assert seen["early"]["error"] == "the check did not finish"
    assert_valid(seen["early"])
    assert res == read_result(tmp_path) and res["verdict"] == "pass" and seen["error"] is None


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_a_stop_signal_becomes_an_error_and_later_ones_cannot_cut_the_write(
    tmp_path, sentinel, signum
):
    args, seen, cleaned = make_args(tmp_path), {}, []
    build = builder(tmp_path, args, rows("info", 1), seen)

    def work():
        os.kill(os.getpid(), signum)
        time.sleep(5)  # Stopped interrupts this
        raise AssertionError("work went on after the signal")

    def build_under_fire(error):
        os.kill(os.getpid(), signal.SIGTERM)  # ignored: the result must still be written
        os.kill(os.getpid(), signal.SIGINT)
        return build(error)

    started = time.monotonic()
    res = kit.run_check("pokec", args, work, build_under_fire, cleanup=lambda: cleaned.append(1))
    assert time.monotonic() - started < 4
    name = signal.Signals(signum).name
    assert seen["error"] == f"stopped by signal {int(signum)} ({name}); partial data in raw/"
    assert seen["stopping"] is True and cleaned == [1]
    assert res == read_result(tmp_path) and res["verdict"] == "error"
    assert_valid(res)
    assert all(signal.getsignal(s) is sentinel for s in kit.STOP_SIGNALS)


def test_a_signal_after_work_cannot_cut_the_write(tmp_path, sentinel):
    args, seen = make_args(tmp_path), {}
    build = builder(tmp_path, args, [], seen)

    def build_under_fire(error):
        os.kill(os.getpid(), signal.SIGTERM)  # no signal before this one: still ignored
        return build(error)

    res = kit.run_check("pokec", args, lambda: None, build_under_fire)
    assert res == read_result(tmp_path) and res["verdict"] == "pass" and seen["error"] is None
    assert all(signal.getsignal(s) is sentinel for s in kit.STOP_SIGNALS)


def test_a_second_signal_cannot_cut_the_check_s_own_teardown(tmp_path, sentinel):
    args, seen, torn_down = make_args(tmp_path), {}, []

    def work():
        try:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)
        finally:  # the check removing its containers while Stopped unwinds
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.1)
            torn_down.append(True)

    kit.run_check("pokec", args, work, builder(tmp_path, args, [], seen))
    assert torn_down == [True] and seen["error"].startswith("stopped by signal 15")


def test_a_check_s_broad_except_cannot_swallow_the_stop(tmp_path, sentinel):
    args, seen, swallowed = make_args(tmp_path), {}, []

    def work():
        try:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)
        except Exception:  # a best-effort handler in the check
            swallowed.append(True)

    kit.run_check("pokec", args, work, builder(tmp_path, args, [], seen))
    assert swallowed == [] and seen["error"].startswith("stopped by signal 15")


def test_a_regression_measured_before_the_stop_outranks_it(tmp_path):
    args, seen, findings = make_args(tmp_path), {}, []

    def work():
        findings.append(kit.finding("s12 completed", "completed", 0, 1, "regression"))
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)

    res = kit.run_check("soak", args, work, builder(tmp_path, args, findings, seen))
    assert res["verdict"] == "regression" and seen["error"].startswith("stopped by signal")


def test_a_sys_exit_in_work_is_a_crash_and_keeps_what_was_measured(tmp_path):
    args, seen, findings = make_args(tmp_path), {}, []

    def work():
        findings.append(kit.finding("s01 completed", "completed", 0, 1, "regression"))
        sys.exit(2)  # argparse's parser.error, deep in a check

    res = kit.run_check("soak", args, work, builder(tmp_path, args, findings, seen))
    assert seen["error"] == "the check crashed: SystemExit: 2"
    assert res["verdict"] == "regression" and read_result(tmp_path) == res


def test_a_stop_landing_right_after_work_still_ends_in_a_result(tmp_path, sentinel, monkeypatch):
    args, seen, findings = make_args(tmp_path), {}, []
    ignore, fired = kit._ignore_stop_signals, []

    def ignore_late():  # the signal arrives just before run_check ignores the rest
        if not fired:
            fired.append(True)
            os.kill(os.getpid(), signal.SIGTERM)
        ignore()

    monkeypatch.setattr(kit, "_ignore_stop_signals", ignore_late)

    def work():
        findings.append(kit.finding("s01 completed", "completed", 0, 1, "regression"))

    res = kit.run_check("soak", args, work, builder(tmp_path, args, findings, seen))
    assert fired and seen["error"].startswith("stopped by signal 15")
    assert res["verdict"] == "regression" and read_result(tmp_path) == res
    assert all(signal.getsignal(s) is sentinel for s in kit.STOP_SIGNALS)


def test_a_stop_landing_in_the_crash_handler_still_ends_in_a_result(
    tmp_path, sentinel, monkeypatch
):
    args, seen = make_args(tmp_path), {}

    def format_exc():  # the signal arrives while the crash is being recorded
        os.kill(os.getpid(), signal.SIGTERM)
        return "Traceback\n"

    monkeypatch.setattr(kit, "traceback", SimpleNamespace(format_exc=format_exc))

    def work():
        raise KeyError("rows.csv")

    res = kit.run_check("ldbc", args, work, builder(tmp_path, args, [], seen))
    assert seen["error"].startswith("stopped by signal 15") and res["verdict"] == "error"
    assert read_result(tmp_path) == res


def test_a_stop_the_check_swallows_is_still_a_stop(tmp_path, sentinel):
    args, seen = make_args(tmp_path), {}

    def work():
        try:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)
        except kit.Stopped:  # as a bare except in the check (or Python's in a __del__) would
            pass

    res = kit.run_check("pokec", args, work, builder(tmp_path, args, [], seen))
    assert seen["error"].startswith("stopped by signal 15") and res["verdict"] == "error"


class BrokenPipe:
    def write(self, _text):
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self):
        raise BrokenPipeError(32, "Broken pipe")


def closed_stream():
    stream = io.StringIO()
    stream.close()  # writing raises ValueError, not OSError
    return stream


@pytest.mark.parametrize("stderr", [BrokenPipe, closed_stream], ids=["broken-pipe", "closed"])
def test_a_broken_or_closed_stderr_cannot_stop_the_write(tmp_path, monkeypatch, stderr):
    monkeypatch.setattr(sys, "stderr", stderr())
    monkeypatch.setenv("PERF_MACHINE_JSON", str(tmp_path / "missing.json"))  # warns, too
    args = make_args(tmp_path)

    def work():
        raise KeyError("rows.csv")

    def build(error):
        raise ValueError("no rows")

    def cleanup():
        raise kit.DockerError("docker rm failed")

    res = kit.run_check("pokec", args, work, build, cleanup)
    assert res == read_result(tmp_path) and res["verdict"] == "error"


def test_a_cleanup_that_exits_still_restores_the_handlers(tmp_path, sentinel):
    args = make_args(tmp_path)

    def cleanup():
        sys.exit(3)

    res = kit.run_check("pokec", args, lambda: None, builder(tmp_path, args, [], {}), cleanup)
    assert res["verdict"] == "pass"
    assert all(signal.getsignal(s) is sentinel for s in kit.STOP_SIGNALS)


def test_a_sys_exit_in_build_still_ends_in_a_result(tmp_path):
    def build(error):
        sys.exit(4)

    res = kit.run_check("pokec", make_args(tmp_path), lambda: None, build)
    assert res == read_result(tmp_path)
    assert res["error"] == "building the result failed: SystemExit: 4"


def test_a_build_that_returns_nothing_is_an_error(tmp_path):
    res = kit.run_check("pokec", make_args(tmp_path), lambda: None, lambda error: None)
    assert res == read_result(tmp_path)
    assert (
        res["error"]
        == "building the result failed: TypeError: build() returned NoneType, not a result"
    )


def test_a_build_that_drops_the_error_is_an_error(tmp_path):
    args = make_args(tmp_path)

    def work():
        raise OSError("no space left on device")

    def build(error):  # forgets to pass error on
        return kit.result("fuzz", args, started=kit.utc_now(), findings=[], summary="fine")

    res = kit.run_check("fuzz", args, work, build)
    assert res["verdict"] == "error" and read_result(tmp_path) == res
    assert res["error"] == "the check crashed: OSError: no space left on device"


def test_a_result_that_cannot_be_written_becomes_an_error_result(tmp_path):
    args = make_args(tmp_path)

    def build(error):
        good = kit.result("fuzz", args, started=kit.utc_now(), findings=[], summary="s")
        return {**good, "machine": {**good["machine"], "cores": math.nan}}

    res = kit.run_check("fuzz", args, lambda: None, build)
    assert_valid(res)
    assert res == read_result(tmp_path)
    assert res["error"].startswith("writing the result failed: ValueError")


def test_run_check_makes_a_relative_out_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = make_args(tmp_path, out="out")
    kit.run_check("pokec", args, lambda: None, builder(tmp_path, args, [], {}))
    assert args.out == str(tmp_path.resolve() / "out")
    assert (tmp_path / "out" / "result.json").is_file()


def test_a_crash_in_work_is_an_error_result(tmp_path, capsys):
    args, seen = make_args(tmp_path), {}

    def work():
        raise KeyError("rows.csv")

    res = kit.run_check("ldbc", args, work, builder(tmp_path, args, [], seen))
    assert seen["error"] == "the check crashed: KeyError: 'rows.csv'"
    assert res["verdict"] == "error" and "Traceback" in capsys.readouterr().err
    assert seen["stopping"] is False


def test_a_build_that_fails_still_leaves_a_valid_error_result(tmp_path):
    def work():
        raise OSError("disk full")

    def build(error):
        raise ZeroDivisionError("division by zero")

    res = kit.run_check("fuzz", make_args(tmp_path), work, build)
    assert res == read_result(tmp_path)
    assert_valid(res)
    assert res["error"] == (
        "the check crashed: OSError: disk full; "
        "building the result failed: ZeroDivisionError: division by zero"
    )


def test_a_failing_cleanup_is_reported_not_raised(tmp_path, capsys):
    args = make_args(tmp_path)

    def cleanup():
        raise kit.DockerError("docker rm failed")

    res = kit.run_check("pokec", args, lambda: None, builder(tmp_path, args, [], {}), cleanup)
    assert res["verdict"] == "pass" and read_result(tmp_path) == res
    assert "cleanup failed: DockerError: docker rm failed" in capsys.readouterr().err


def test_run_check_off_the_main_thread_installs_no_handlers(tmp_path):
    args, results = make_args(tmp_path), []
    before = signal.getsignal(signal.SIGTERM)

    def call():
        results.append(kit.run_check("pokec", args, lambda: None, builder(tmp_path, args, [], {})))

    thread = threading.Thread(target=call)
    thread.start()
    thread.join()
    assert results[0]["verdict"] == "pass" and signal.getsignal(signal.SIGTERM) is before


# --- docker (fake) -----------------------------------------------------------------


@pytest.fixture
def docker(tmp_path, monkeypatch):
    """`docker` on PATH is fake_docker.py. Yields its state directory."""
    bin_dir, state = tmp_path / "bin", tmp_path / "state"
    bin_dir.mkdir()
    state.mkdir()
    shim = bin_dir / "docker"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I "{FAKE_DOCKER}" "$@"\n')
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DOCKER_STATE", str(state))
    yield state
    for record in state.glob("*.json"):  # a failed test must not leave servers behind
        with contextlib.suppress(ProcessLookupError, TypeError, ValueError):
            os.kill(json.loads(record.read_text())["pid"], signal.SIGKILL)


def calls(state, command):
    lines = (state / "calls.jsonl").read_text().splitlines()
    return [argv for argv in map(json.loads, lines) if argv[0] == command]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_pull_retries_then_succeeds(docker, monkeypatch):
    monkeypatch.setenv("FAKE_DOCKER_PULL_FAILS", "2")
    kit.pull(MAIN, attempts=3, delay_s=0)
    assert calls(docker, "pull") == [["pull", "--quiet", MAIN]] * 3


def test_pull_gives_up_with_docker_s_reason(docker, monkeypatch):
    monkeypatch.setenv("FAKE_DOCKER_PULL_FAILS", "9")
    with pytest.raises(
        kit.DockerError, match=r"\(2 attempts\) failed \(exit 1\): .*toomanyrequests"
    ):
        kit.pull(MAIN, attempts=2, delay_s=0)
    assert len(calls(docker, "pull")) == 2


def test_image_env(docker, monkeypatch):
    assert kit.image_env(V621)["FALKORDB_SERVER"] == "valkey"
    monkeypatch.setenv("FAKE_DOCKER_ENV", '["FALKORDB_ARGS=TIMEOUT 1000 THREAD_COUNT 8", "EMPTY="]')
    assert kit.image_env(V621) == {"FALKORDB_ARGS": "TIMEOUT 1000 THREAD_COUNT 8", "EMPTY": ""}
    monkeypatch.setenv("FAKE_DOCKER_ENV", "null")
    assert kit.image_env(V621) == {}


def test_a_detached_server_publishes_on_loopback_and_reads_the_port_back(docker, tmp_path):
    server = kit.start(
        V621,
        "kit-a",
        env={"FALKORDB_ARGS": "IMPORT_FOLDER /import/"},
        volumes={tmp_path: "/import:ro"},
        labels={"perf-kit": "run-1"},
        cpus="0-3",
    )
    volume = f"{tmp_path.resolve()}:/import:ro"
    assert calls(docker, "run") == [
        [
            *("run", "--name", "kit-a", "--detach", "--label", "perf-kit=run-1"),
            *("--env", "FALKORDB_ARGS=IMPORT_FOLDER /import/", "--volume", volume),
            *("--cpuset-cpus", "0-3", "--publish", "127.0.0.1::6379", V621),
        ]
    ]
    assert calls(docker, "rm")[0][-1] == "kit-a"  # a same-name leftover goes first
    assert server.proc is None and server.port not in (0, 6379)
    assert kit.wait_ping(server.port, 10, server)
    assert kit.ping(server.port)
    assert kit.state("kit-a") == "running" and kit.host_pid("kit-a") > 0
    # stdout and stderr in the order written, as one stream
    assert kit.logs("kit-a") == "fake server log\nfake warning\nReady to accept connections\n"
    assert kit.logs("kit-a", tail=1) == "Ready to accept connections\n"
    assert kit.logs("kit-a", tail=0) == ""
    kit.remove("kit-a")
    assert (kit.state("kit-a"), kit.host_pid("kit-a")) == ("gone", 0)


def test_a_foreground_server_on_the_host_network(docker, tmp_path):
    port = free_port()
    server = kit.start(
        MAIN,
        "kit-fg",
        entrypoint="valkey-server",
        command=["--port", str(port), "--bind", "127.0.0.1"],
        extra=["--rm", "--user", "1000:1000"],
        port=port,
        host_network=True,
        foreground=True,
        output=tmp_path / "server.out",
    )
    assert server.port == port and server.proc.poll() is None
    assert kit.wait_ping(port, 10, server)  # also: the child has logged its argv by now
    argv = calls(docker, "run")[0]
    assert argv[:3] == ["run", "--name", "kit-fg"] and "--detach" not in argv
    assert argv[3:] == [
        *("--network", "host", "--entrypoint", "valkey-server", "--rm", "--user", "1000:1000"),
        *(MAIN, "--port", str(port), "--bind", "127.0.0.1"),
    ]
    assert kit.host_pid("kit-fg") > 0
    kit.remove("kit-fg")
    assert server.proc.wait(10) != 0  # the `docker run` child ends with its container


def test_a_foreground_run_docker_refused_is_a_docker_error_not_a_dead_server(docker, monkeypatch):
    monkeypatch.setenv("FAKE_DOCKER_REFUSE", "kit-")
    port = free_port()
    server = kit.start(MAIN, "kit-fg", port=port, host_network=True, foreground=True)
    with pytest.raises(kit.DockerError, match=r"exit 125"):
        kit.wait_ping(port, 10, server)


def test_a_foreground_server_needs_the_host_network_and_never_6379(docker):
    with pytest.raises(ValueError, match="needs host_network"):
        kit.start(MAIN, "kit-fg", foreground=True)
    with pytest.raises(ValueError, match="never 6379"):
        kit.start(MAIN, "kit-a", host_network=True)
    assert not (docker / "calls.jsonl").exists()


def test_a_refused_run_raises_with_docker_s_reason(docker, monkeypatch):
    monkeypatch.setenv("FAKE_DOCKER_REFUSE", "kit-")
    with pytest.raises(kit.DockerError, match=r"docker run .* \(exit 125\): .*Conflict"):
        kit.start(MAIN, "kit-a")
    assert not list(docker.glob("*.json"))


def test_a_stop_while_start_handles_a_failure_still_removes_the_container(docker, monkeypatch):
    monkeypatch.setenv("FAKE_DOCKER_NO_PORT", "1")

    def stopped_here(*_args, **_kwargs):
        raise kit.Stopped("stopped by signal 15 (SIGTERM)")

    monkeypatch.setattr(kit, "_ended", stopped_here)
    with pytest.raises(kit.Stopped):
        kit.start(MAIN, "kit-a")
    assert not list(docker.glob("*.json"))


def test_a_stop_during_start_removes_the_container(docker, monkeypatch):
    real = kit._docker

    def stopped_at_port(*argv, **kwargs):
        if argv[0] == "port":
            raise kit.Stopped("stopped by signal 15 (SIGTERM)")
        return real(*argv, **kwargs)

    monkeypatch.setattr(kit, "_docker", stopped_at_port)
    with pytest.raises(kit.Stopped):
        kit.start(MAIN, "kit-a")
    assert not list(docker.glob("*.json"))


def test_a_container_whose_port_cannot_be_read_is_removed(docker, monkeypatch):
    monkeypatch.setenv("FAKE_DOCKER_NO_PORT", "1")
    with pytest.raises(kit.DockerError, match="no host port for kit-a:6379"):
        kit.start(MAIN, "kit-a")
    assert not list(docker.glob("*.json"))
    assert calls(docker, "rm")[-1][-1] == "kit-a"


def test_a_server_that_dies_at_start_keeps_its_container_for_the_reason(docker, monkeypatch):
    # Real docker clears the port map of a stopped container: `docker port` fails.
    monkeypatch.setenv("FAKE_DOCKER_CRASH", "falkordb-server@")
    server = kit.start(V621, "kit-a")
    assert server.port == 0 and kit.state("kit-a") == "exited 139"
    assert "Ready to accept connections" in kit.logs("kit-a")
    started = time.monotonic()
    assert not kit.wait_ping(server.port, 30, server)
    assert time.monotonic() - started < 5 and kit.host_pid("kit-a") == 0


@pytest.mark.parametrize(
    ("extra", "foreground"),
    [((), False), (("--rm",), False), (("--rm",), True)],
    ids=["exited", "removed-by-rm", "foreground"],
)
def test_wait_ping_ends_at_once_when_the_server_stops(docker, monkeypatch, extra, foreground):
    monkeypatch.setenv("FAKE_DOCKER_CRASH", "falkordb-server@")
    port = free_port()
    options = {"command": ["--port", str(port)], "extra": extra, "port": port}
    server = kit.start(V621, "kit-a", **options, host_network=True, foreground=foreground)
    started = time.monotonic()
    assert not kit.wait_ping(port, 30, server)
    assert time.monotonic() - started < 5


def test_a_host_network_port_held_by_a_silent_listener_is_refused(docker):
    with socket.socket() as listener:  # a leftover server still loading, or wanting AUTH
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(kit.DockerError, match=f"port {port} already has a server"):
            kit.start(V621, "kit-a", command=["--port", str(port)], port=port, host_network=True)
    assert not (docker / "calls.jsonl").exists()


def test_a_host_network_port_that_already_answers_is_refused(docker):
    port = free_port()
    kit.start(V621, "kit-a", command=["--port", str(port)], port=port, host_network=True)
    assert kit.wait_ping(port, 10)
    with pytest.raises(kit.DockerError, match=f"port {port} already has a server"):
        kit.start(V621, "kit-b", command=["--port", str(port)], port=port, host_network=True)
    assert [argv[2] for argv in calls(docker, "run")] == ["kit-a"]


def test_state_reports_an_oom_kill(docker, monkeypatch):
    monkeypatch.setenv("FAKE_DOCKER_OOM", "falkordb-server@")
    port = free_port()
    kit.start(V621, "kit-a", command=["--port", str(port)], port=port, host_network=True)
    assert kit.state("kit-a") == "exited 137 (OOM-killed)"


def test_a_docker_that_does_not_answer_is_not_a_dead_server(docker, monkeypatch):
    server = kit.start(V621, "kit-a")
    monkeypatch.setenv("FAKE_DOCKER_INSPECT_DOWN", "1")
    with pytest.raises(kit.DockerError, match="Cannot connect to the Docker daemon"):
        kit.state("kit-a")
    with pytest.raises(kit.DockerError):
        kit.host_pid("kit-a")
    started = time.monotonic()
    assert not kit.wait_ping(free_port(), 1.5, server)
    assert time.monotonic() - started >= 1.4  # inspect failing did not end the wait early


def test_wait_ping_keeps_its_budget_while_docker_inspect_hangs(docker, monkeypatch):
    server = kit.start(V621, "kit-a")
    monkeypatch.setenv("FAKE_DOCKER_INSPECT_HANG", "10")
    started = time.monotonic()
    assert not kit.wait_ping(free_port(), 1.0, server)
    assert time.monotonic() - started < 3


def test_wait_ping_waits_out_loading(docker, monkeypatch):
    monkeypatch.setenv("FAKE_DOCKER_LOADING_S", "3")
    server = kit.start(V621, "kit-a")
    started = time.monotonic()
    while True:  # listening, and still loading
        with contextlib.suppress(OSError):
            socket.create_connection(("127.0.0.1", server.port), timeout=1).close()
            break
        assert time.monotonic() - started < 10, f"never listened: {kit.state('kit-a')}"
        time.sleep(0.05)
    assert not kit.ping(server.port)  # -LOADING is not an answer
    assert kit.wait_ping(server.port, 10, server)
    assert time.monotonic() - started >= 2


def test_wait_ping_gives_up_at_its_timeout():
    port = free_port()  # nothing listens
    started = time.monotonic()
    assert not kit.wait_ping(port, 1.0)
    assert time.monotonic() - started < 3


def test_ping_is_one_request_and_false_on_a_closed_connection():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(5)  # a ping that never connects fails the test, not hangs it
        port, received = listener.getsockname()[1], []

        def close_one():
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(5)
                received.append(connection.recv(64))

        thread = threading.Thread(target=close_one, daemon=True)
        thread.start()
        assert not kit.ping(port, timeout=2)
        thread.join(10)
        assert received == [b"*1\r\n$4\r\nPING\r\n"]  # one PING, then the close


def test_ping_keeps_one_deadline_against_a_reply_that_never_ends():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(5)
        port, done = listener.getsockname()[1], threading.Event()

        def trickle():
            connection, _ = listener.accept()
            with connection, contextlib.suppress(OSError):
                connection.recv(64)
                t0 = time.monotonic()  # one byte at a time, never a line end (for 5 s)
                while not done.is_set() and time.monotonic() - t0 < 5:
                    connection.sendall(b"+")
                    time.sleep(0.2)

        thread = threading.Thread(target=trickle, daemon=True)
        thread.start()
        started = time.monotonic()
        assert not kit.ping(port, timeout=1.0)
        assert time.monotonic() - started < 2
        done.set()
        thread.join(10)


def test_remove_tolerates_a_container_that_is_gone(docker):
    kit.remove("never-started")
    kit.remove()
    assert calls(docker, "rm") == [["rm", "--force", "--volumes", "never-started"]]


def test_remove_labelled_removes_only_its_run(docker):
    kit.start(V621, "kit-mine-1", labels={"perf-kit": "run-1"})
    kit.start(V621, "kit-mine-2", labels={"perf-kit": "run-1", "other": "x"})
    kit.start(V621, "kit-other", labels={"perf-kit": "run-2"})
    kit.remove_labelled("perf-kit=run-1")
    assert sorted(p.stem for p in docker.glob("*.json")) == ["kit-other"]
    kit.remove_labelled("perf-kit=run-1")  # nothing left: no docker rm at all
    assert calls(docker, "rm")[-1] == ["rm", "--force", "--volumes", "kit-mine-1", "kit-mine-2"]


def test_a_silent_docker_rm_failure_is_a_failure(monkeypatch):
    monkeypatch.setattr(
        kit, "_docker", lambda *argv, **kw: subprocess.CompletedProcess(argv, 137, "", "")
    )
    with pytest.raises(kit.DockerError, match=r"docker rm kit-a failed \(exit 137\)"):
        kit.remove("kit-a")


def test_docker_that_cannot_be_started_is_a_docker_error(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(subprocess, "run", refuse)
    with pytest.raises(kit.DockerError, match="docker could not be started"):
        kit.remove("x")


def test_only_docker_s_own_no_such_container_means_gone(monkeypatch):
    def reply(stderr):
        return lambda *argv, **kw: subprocess.CompletedProcess(argv, 1, "", stderr)

    monkeypatch.setattr(kit, "_docker", reply("Error: No such object: kit-a"))
    assert (kit.state("kit-a"), kit.host_pid("kit-a")) == ("gone", 0)
    monkeypatch.setattr(kit, "_docker", reply("error during connect: lookup dockerd: no such host"))
    with pytest.raises(kit.DockerError, match="no such host"):
        kit.state("kit-a")
    with pytest.raises(kit.DockerError, match="no such host"):
        kit.host_pid("kit-a")


def test_docker_missing_from_path_is_a_docker_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    for call in (
        lambda: kit.pull(MAIN, delay_s=0),
        lambda: kit.state("x"),
        lambda: kit.remove("x"),
    ):
        with pytest.raises(kit.DockerError, match="docker is not on PATH"):
            call()


def test_a_docker_that_hangs_is_a_docker_error(tmp_path, monkeypatch):
    (tmp_path / "docker").write_text("#!/bin/sh\nexec /bin/sleep 5\n")
    (tmp_path / "docker").chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(subprocess, "run", _with_timeout(subprocess.run, 0.5))
    with pytest.raises(kit.DockerError, match="did not return within"):
        kit.state("x")


def _with_timeout(run, seconds):
    def bounded(*args, **kwargs):
        kwargs["timeout"] = seconds
        return run(*args, **kwargs)

    return bounded
