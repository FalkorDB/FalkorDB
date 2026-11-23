"""tests/perf/merge.py: the plan, skip and merge rules of the nightly perf run.

Every nightly.json these tests build is also run through perf-judge's
tests/perf/validate.py, so "merge wrote it" and "the contract accepts it" are
checked together.
"""

import importlib.util
import json
import math
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(f"perf_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


merge = load("merge")
validate = load("validate")

MAIN = "4c1088949a5852e3949c05bfb3d92a603c4cd4ca"
OLDER = "4ac2af3103ae5a293baef097b749fb8338bc0bfd"
RELEASE = "d8c329b777c177fc7343d959157d6acdf7398fa5"
CAND = "ghcr.io/falkordb/falkordb-server@sha256:" + "95b610a6" + "0" * 56
BASE = "falkordb/falkordb-server@sha256:" + "44eb4a6d" + "0" * 56
OTHER_BASE = "falkordb/falkordb-server@sha256:" + "698a6e74" + "0" * 56

CANDIDATE = {"image": CAND, "commit": MAIN, "label": "main@4c10889"}
BASELINE = {"image": BASE, "commit": RELEASE, "label": "6.2.0"}


def result(check, verdict="pass", **extra):
    findings = []
    if verdict == "regression":
        findings = [
            {
                "name": "UNWIND id seek",
                "metric": "instructions",
                "candidate": 3.0,
                "baseline": 1.0,
                "ratio": 3.0,
                "threshold": 1.05,
                "direction": "higher_is_worse",
                "status": "regression",
            }
        ]
    doc = {
        "schema": 1,
        "check": check,
        "verdict": verdict,
        "confirmed": verdict == "regression",
        "summary": f"{check} says {verdict}",
        "candidate": CANDIDATE,
        "baseline": BASELINE,
        "machine": {"runner": "hosted", "type": "ubuntu-latest", "cpu": "x86", "cores": 4},
        "started": "2026-11-24T00:05:00Z",
        "finished": "2026-11-24T00:30:00Z",
        "findings": findings,
        "error": "could not start the server" if verdict == "error" else None,
        "artifacts": [],
    }
    doc.update(extra)
    return doc


def context(skip_reason=None, previous_green=None):
    return {
        "date": "2026-11-24",
        "run_url": "https://github.com/FalkorDB/FalkorDB/actions/runs/1",
        "candidate": CANDIDATE,
        "baseline": BASELINE,
        "previous_green_commit": previous_green,
        "skip_reason": skip_reason,
    }


def put(results, check, doc=None, name="result.json", text=None):
    artifact = results / f"perf-result-{check}"
    artifact.mkdir(parents=True, exist_ok=True)
    if text is None:
        text = json.dumps(doc)
    (artifact / name).write_text(text)
    return artifact


def run_merge(tmp_path, plan, ctx=None):
    """merge.py's CLI end to end; returns (exit code, nightly)."""
    (tmp_path / "context.json").write_text(json.dumps(ctx or context()))
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    out = tmp_path / "nightly.json"
    code = merge.main(
        [
            "merge",
            "--context", str(tmp_path / "context.json"),
            "--plan", str(tmp_path / "plan.json"),
            "--results", str(tmp_path / "results"),
            "--validator", str(HERE / "validate.py"),
            "--out", str(out),
        ]
    )
    nightly = json.loads(out.read_text())
    # Independent of merge's own check: the contract accepts what it wrote.
    assert validate.validate_nightly(nightly) == []
    return code, nightly


MICRO_ONLY = [{"check": "micro", "runner": "micro"}]
VM_ENTRY = {
    "name": "size-probe",
    "runner": "vm",
    "command": "uv run --project bench bench probe",
    "timeout_minutes": 45,
    "machine_type": "n4-standard-16",
}
HOSTED_ENTRY = {"name": "ldbc", "runner": "hosted", "command": "true", "timeout_minutes": 30}


# --- plan ------------------------------------------------------------------------


def test_empty_registry_plans_micro_alone_with_placeholder_matrices():
    plan = merge.build_plan([], "")
    assert plan == MICRO_ONLY
    out = merge.plan_outputs(plan)
    assert out["run_micro"] == "true"
    assert out["has_vm"] == out["has_hosted"] == "false"
    # Never an empty matrix: that fails the job before its `if:` is read.
    assert json.loads(out["vm_matrix"]) == merge.PLACEHOLDER
    assert json.loads(out["hosted_matrix"]) == merge.PLACEHOLDER


def test_registry_entries_become_matrix_rows():
    plan = merge.build_plan([VM_ENTRY, HOSTED_ENTRY], "")
    assert [row["check"] for row in plan] == ["micro", "size-probe", "ldbc"]
    out = merge.plan_outputs(plan)
    assert json.loads(out["vm_matrix"]) == [
        {
            "check": "size-probe",
            "command": "uv run --project bench bench probe",
            "timeout_minutes": 45,
            "machine_type": "n4-standard-16",
        }
    ]
    assert json.loads(out["hosted_matrix"])[0]["machine_type"] == ""
    assert out["has_vm"] == out["has_hosted"] == "true"


def test_selection_keeps_order_and_plans_unknown_names_as_stubs():
    plan = merge.build_plan([VM_ENTRY, HOSTED_ENTRY], " ldbc, typo ,micro,ldbc")
    assert [row["check"] for row in plan] == ["ldbc", "typo", "micro"]
    assert plan[1] == {"check": "typo", "runner": None, "error": "not in tests/perf/checks.yaml"}
    assert merge.plan_outputs(plan)["has_vm"] == "false"


def test_selection_without_micro_skips_micro():
    plan = merge.build_plan([VM_ENTRY], "size-probe")
    assert merge.plan_outputs(plan)["run_micro"] == "false"


@pytest.mark.parametrize(
    "entry, why",
    [
        ({**HOSTED_ENTRY, "runner": "gce"}, "runner 'gce'"),
        ({**HOSTED_ENTRY, "command": " "}, "command is missing"),
        ({**HOSTED_ENTRY, "timeout_minutes": "30"}, "timeout_minutes '30'"),
        ({**HOSTED_ENTRY, "timeout_minutes": True}, "timeout_minutes True"),
        ({**HOSTED_ENTRY, "machine_type": "n4-standard-16"}, "only for runner: vm"),
        ({**HOSTED_ENTRY, "env": {}}, "unknown key(s) env"),
    ],
)
def test_malformed_entry_is_planned_as_a_stub(entry, why):
    (row,) = [r for r in merge.build_plan([entry], "") if r["check"] == "ldbc"]
    assert row["runner"] is None
    assert why in row["error"]


@pytest.mark.parametrize(
    "registry",
    [
        {"name": "x"},
        [{**HOSTED_ENTRY, "name": "Bad_Name"}],
        [{**HOSTED_ENTRY, "name": "micro"}],
        [HOSTED_ENTRY, HOSTED_ENTRY],
    ],
)
def test_registry_without_usable_names_fails_the_plan(registry):
    with pytest.raises(merge.PlanError):
        merge.build_plan(registry, "")


# --- merge -----------------------------------------------------------------------


def test_missing_micro_result_is_a_stub_and_the_night_is_not_green(tmp_path):
    (tmp_path / "results").mkdir()
    code, nightly = run_merge(tmp_path, MICRO_ONLY)
    assert code == 0
    assert nightly["checks"] == [{"check": "micro", "verdict": "error", "error": "no result.json"}]
    assert nightly["status"] == "error"


def test_every_planned_check_appears_once(tmp_path):
    results = tmp_path / "results"
    put(results, "micro", result("micro"))
    plan = merge.build_plan([VM_ENTRY, HOSTED_ENTRY], "")
    code, nightly = run_merge(tmp_path, plan)
    assert code == 0
    assert [c["check"] for c in nightly["checks"]] == ["micro", "size-probe", "ldbc"]
    assert nightly["checks"][0] == result("micro")
    assert nightly["checks"][1]["error"] == nightly["checks"][2]["error"] == "no result.json"


def test_all_pass_is_green_and_records_both_sides(tmp_path):
    results = tmp_path / "results"
    put(results, "micro", result("micro"))
    code, nightly = run_merge(tmp_path, MICRO_ONLY, context(previous_green=OLDER))
    assert code == 0
    assert nightly["status"] == "green"
    assert nightly["candidate"] == CANDIDATE and nightly["baseline"] == BASELINE
    assert nightly["previous_green_commit"] == OLDER
    assert nightly["skip_reason"] is None


def test_regression_outranks_error(tmp_path):
    results = tmp_path / "results"
    put(results, "micro", result("micro", "regression"))
    plan = merge.build_plan([HOSTED_ENTRY], "")
    _, nightly = run_merge(tmp_path, plan)
    assert [c["verdict"] for c in nightly["checks"]] == ["regression", "error"]
    assert nightly["status"] == "red"


def test_an_error_result_is_kept_whole(tmp_path):
    results = tmp_path / "results"
    put(results, "micro", result("micro", "error"))
    _, nightly = run_merge(tmp_path, MICRO_ONLY)
    assert nightly["checks"] == [result("micro", "error")]
    assert nightly["status"] == "error"


def test_invalid_result_becomes_a_stub_naming_the_problem(tmp_path):
    results = tmp_path / "results"
    bad = result("micro", candidate={**CANDIDATE, "image": "falkordb/falkordb-server:6.2.0"})
    put(results, "micro", bad)
    _, nightly = run_merge(tmp_path, MICRO_ONLY)
    (entry,) = nightly["checks"]
    assert entry["verdict"] == "error"
    assert entry["error"].startswith("result.json failed validation: image-not-digest")


def test_nan_in_a_result_becomes_a_stub(tmp_path):
    results = tmp_path / "results"
    text = json.dumps(result("micro", "regression")).replace("3.0", "NaN", 1)
    put(results, "micro", text=text)
    _, nightly = run_merge(tmp_path, MICRO_ONLY)
    assert "failed validation" in nightly["checks"][0]["error"]


def test_unparsable_result_becomes_a_stub(tmp_path):
    put(tmp_path / "results", "micro", text="{not json")
    _, nightly = run_merge(tmp_path, MICRO_ONLY)
    assert nightly["checks"][0]["error"].startswith("result.json failed validation: invalid-json")


def test_result_naming_another_check_becomes_a_stub(tmp_path):
    put(tmp_path / "results", "micro", result("size-probe"))
    _, nightly = run_merge(tmp_path, MICRO_ONLY)
    assert nightly["checks"][0]["error"] == "result.json names check 'size-probe'"


def test_result_below_the_artifact_root_is_reported_as_misplaced(tmp_path):
    artifact = put(tmp_path / "results", "micro", text="{}", name="ignored.txt")
    (artifact / "out").mkdir()
    (artifact / "out" / "result.json").write_text(json.dumps(result("micro")))
    _, nightly = run_merge(tmp_path, MICRO_ONLY)
    assert nightly["checks"][0]["error"] == "no result.json at the artifact root (found out/result.json)"


@pytest.mark.parametrize(
    "vm, reason",
    [
        (
            {"create_error": "all zones out of stock for n4-standard-16 and n2-standard-16"},
            "no result.json: no VM came up: all zones out of stock for n4-standard-16 and n2-standard-16",
        ),
        ({"create_error": None, "timed_out": True}, "no result.json: the check timed out on the VM"),
        ({"create_error": "", "command_exit": 137}, "no result.json: the check exited 137 on the VM"),
        ({"create_error": None, "command_exit": 0}, "no result.json"),
    ],
)
def test_perf_vm_json_explains_a_missing_result(tmp_path, vm, reason):
    plan = merge.build_plan([VM_ENTRY], "size-probe")
    put(tmp_path / "results", "size-probe", vm, name="perf-vm.json")
    _, nightly = run_merge(tmp_path, plan)
    assert nightly["checks"] == [{"check": "size-probe", "verdict": "error", "error": reason}]


def test_planned_stub_keeps_its_reason(tmp_path):
    (tmp_path / "results").mkdir()
    plan = merge.build_plan([], "micro,typo")
    _, nightly = run_merge(tmp_path, plan)
    assert nightly["checks"][1] == {
        "check": "typo",
        "verdict": "error",
        "error": "not in tests/perf/checks.yaml",
    }


def test_unplanned_artifact_is_reported_not_dropped(tmp_path):
    results = tmp_path / "results"
    put(results, "micro", result("micro"))
    put(results, "extra", result("extra"))
    _, nightly = run_merge(tmp_path, MICRO_ONLY)
    assert [c["check"] for c in nightly["checks"]] == ["micro", "extra"]


def test_skipped_night_lists_no_checks(tmp_path):
    results = tmp_path / "results"
    put(results, "micro", result("micro"))
    code, nightly = run_merge(tmp_path, MICRO_ONLY, context(skip_reason="main has not moved"))
    assert code == 0
    assert nightly["status"] == "skipped"
    assert nightly["checks"] == []
    assert nightly["skip_reason"] == "main has not moved"


def test_json_out_refuses_nan(tmp_path):
    with pytest.raises(ValueError):
        merge.write_json(tmp_path / "x.json", {"ratio": math.nan})


# --- context: skip and previous green --------------------------------------------


def night(status, commit=MAIN, base=BASE, date="2026-11-23"):
    return {
        "schema": 1,
        "date": date,
        "run_url": f"https://github.com/FalkorDB/FalkorDB/actions/runs/{date}",
        "candidate": {**CANDIDATE, "commit": commit},
        "baseline": {**BASELINE, "image": base},
        "status": status,
    }


def history(*nights):
    return [(str(100 - i), n) for i, n in enumerate(nights)]


@pytest.mark.parametrize("status", ["green", "red", "skipped"])
def test_unchanged_pair_after_a_settled_night_skips(status):
    reason = merge.skip_reason(history(night(status), night("green")), CANDIDATE, BASELINE)
    assert reason is not None and "main has not moved" in reason


def test_skip_reason_names_the_night_that_measured(tmp_path):
    nights = history(
        night("skipped", date="2026-11-23"),
        night("skipped", date="2026-11-22"),
        night("red", date="2026-11-21"),
    )
    reason = merge.skip_reason(nights, CANDIDATE, BASELINE)
    assert "the 2026-11-21 nightly (red, https://" in reason
    assert "same candidate main@4c10889 and baseline 6.2.0" in reason


@pytest.mark.parametrize(
    "nights",
    [
        [],  # the first nightly ever
        history(None, night("green")),  # last run left no artifact
        history(night("error"), night("green")),  # retry a failed night
        history(night("green", commit=OLDER)),  # main moved
        history(night("green", base=OTHER_BASE)),  # a new release
    ],
)
def test_runs_when_anything_is_new_or_unsettled(nights):
    assert merge.skip_reason(nights, CANDIDATE, BASELINE) is None


def test_candidate_without_commit_never_skips():
    nights = history(night("green", commit=None))
    assert merge.skip_reason(nights, {**CANDIDATE, "commit": None}, BASELINE) is None


def test_previous_green_is_the_newest_green_night():
    nights = history(night("red"), None, night("skipped"), night("green", commit=OLDER), night("green"))
    assert merge.previous_green_commit(nights) == OLDER
    assert merge.previous_green_commit(history(night("red"))) is None


def test_context_cli_reads_downloaded_history(tmp_path):
    prev = tmp_path / "prev"
    for run_id, doc in (("30", night("green")), ("20", night("green", commit=OLDER))):
        (prev / run_id).mkdir(parents=True)
        (prev / run_id / "nightly.json").write_text(json.dumps(doc))
    (prev / "25").mkdir()  # a run whose artifact did not download
    runs = tmp_path / "runs.txt"
    runs.write_text("30\n25\n20\n")
    output = tmp_path / "github_output"
    args = [
        "context",
        "--candidate-image", CAND, "--candidate-commit", MAIN, "--candidate-label", "main@4c10889",
        "--baseline-image", BASE, "--baseline-commit", RELEASE, "--baseline-label", "6.2.0",
        "--history-runs", str(runs), "--history-dir", str(prev),
        "--date", "2026-11-24", "--run-url", "https://github.com/x/y/actions/runs/2",
        "--out", str(tmp_path / "context.json"), "--github-output", str(output),
    ]
    assert merge.main(args + ["--may-skip"]) == 0
    ctx = json.loads((tmp_path / "context.json").read_text())
    assert ctx["skip_reason"].startswith("main has not moved since the 2026-11-23 nightly")
    assert ctx["previous_green_commit"] == MAIN
    assert output.read_text() == "skip=true\n"

    output.write_text("")
    assert merge.main(args) == 0  # a dispatch never skips
    assert json.loads((tmp_path / "context.json").read_text())["skip_reason"] is None
    assert output.read_text() == "skip=false\n"


@pytest.mark.parametrize(
    "flag, value",
    [
        ("--candidate-image", "ghcr.io/falkordb/falkordb-server:rc-mg-x-valkey"),
        ("--baseline-commit", "d8c329b"),
        ("--candidate-label", ""),
    ],
)
def test_context_refuses_unpinned_or_malformed_sides(tmp_path, flag, value):
    args = {
        "--candidate-image": CAND, "--candidate-commit": MAIN, "--candidate-label": "main@4c10889",
        "--baseline-image": BASE, "--baseline-commit": RELEASE, "--baseline-label": "6.2.0",
    }
    args[flag] = value
    argv = ["context"] + [x for pair in args.items() for x in pair]
    argv += ["--date", "2026-11-24", "--run-url", "https://x/y", "--out", str(tmp_path / "c.json")]
    assert merge.main(argv) == 1
