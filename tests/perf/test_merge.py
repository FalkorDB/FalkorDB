"""tests/perf/merge.py: the plan, skip and merge rules of the nightly perf run.

Every nightly.json these tests build is also run through perf-judge's
tests/perf/validate.py, so "merge wrote it" and "the contract accepts it" are
checked together.
"""

import importlib.util
import json
import math
import shutil
import subprocess
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
MICRO_TIER = MICRO_ONLY + [{"check": "micro-hw", "runner": "micro-hw"}]
VM_ENTRY = {
    "name": "size-probe",
    "runner": "vm",
    "command": "uv run --project bench bench probe",
    "timeout_minutes": 45,
    "machine_type": "n4-standard-16",
}
HOSTED_ENTRY = {"name": "ldbc", "runner": "hosted", "command": "true", "timeout_minutes": 30}


# --- plan ------------------------------------------------------------------------


def test_empty_registry_plans_the_micro_tier_with_placeholder_matrices():
    plan = merge.build_plan([], "")
    assert plan == MICRO_TIER
    out = merge.plan_outputs(plan)
    assert out["run_micro"] == out["micro"] == out["micro_hw"] == "true"
    assert out["has_vm"] == out["has_hosted"] == "false"
    # Never an empty matrix: that fails the job before its `if:` is read.
    assert json.loads(out["vm_matrix"]) == merge.PLACEHOLDER
    assert json.loads(out["hosted_matrix"]) == merge.PLACEHOLDER


def test_registry_entries_become_matrix_rows():
    plan = merge.build_plan([VM_ENTRY, HOSTED_ENTRY], "")
    assert [row["check"] for row in plan] == ["micro", "micro-hw", "size-probe", "ldbc"]
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
    "select, run_micro, micro, micro_hw",
    [
        ("micro", "true", "true", "false"),  # no C4 VM for a callgrind-only dispatch
        ("micro-hw", "true", "false", "true"),
        ("micro-hw,micro", "true", "true", "true"),
        ("size-probe", "false", "false", "false"),
    ],
)
def test_one_micro_call_runs_the_selected_halves(select, run_micro, micro, micro_hw):
    """_perf-micro.yml makes both micro checks; its `micro` and `micro_hw`
    inputs say which, so a selection never starts a half nobody planned."""
    out = merge.plan_outputs(merge.build_plan([VM_ENTRY], select))
    assert (out["run_micro"], out["micro"], out["micro_hw"]) == (run_micro, micro, micro_hw)


@pytest.mark.parametrize(
    "entry, why",
    [
        ({**HOSTED_ENTRY, "runner": "gce"}, "runner 'gce'"),
        ({**HOSTED_ENTRY, "command": " "}, "command is missing"),
        ({**HOSTED_ENTRY, "timeout_minutes": "30"}, "timeout_minutes '30'"),
        ({**HOSTED_ENTRY, "timeout_minutes": 331}, "from 1 to 330"),
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
    "registry, why",
    [
        ({"name": "x"}, "it is a dict, not a list"),
        ([{**HOSTED_ENTRY, "name": "Bad_Name"}, VM_ENTRY], "entry 0: name 'Bad_Name' is not a check name"),
        ([{**HOSTED_ENTRY, "name": "ldbc--x"}, VM_ENTRY], "entry 0: name 'ldbc--x' is not a check name"),
        ([{**HOSTED_ENTRY, "name": "micro"}, VM_ENTRY], "entry 0: name 'micro' is reserved"),
        ([{**HOSTED_ENTRY, "name": "micro-hw"}, VM_ENTRY], "entry 0: name 'micro-hw' is reserved"),
        ([{**HOSTED_ENTRY, "name": "checks-yaml"}, VM_ENTRY], "entry 0: name 'checks-yaml' is reserved"),
        ([VM_ENTRY, VM_ENTRY], "entry 1: name 'size-probe' is reserved or already used"),
        (["size-probe", VM_ENTRY], "entry 0 is not a mapping"),
    ],
)
def test_registry_problems_are_reported_not_fatal(registry, why):
    """A bad registry must not stop the night: micro (and every usable entry)
    still runs, and the problem is a `checks-yaml` error in the report."""
    plan = merge.build_plan(registry, "")
    assert plan[0] == {"check": "micro", "runner": "micro"}
    assert plan[-1]["check"] == "checks-yaml" and plan[-1]["runner"] is None
    assert why in plan[-1]["error"]
    if isinstance(registry, list) and VM_ENTRY in registry:
        assert merge.plan_outputs(plan)["has_vm"] == "true"


def test_registry_that_does_not_parse_still_plans_micro(tmp_path):
    out = tmp_path / "github_output"
    argv = ["plan", "--registry-error", "yq: bad indentation", "--out", str(tmp_path / "plan.json")]
    assert merge.main(argv + ["--github-output", str(out)]) == 0
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan == MICRO_TIER + [
        {"check": "checks-yaml", "runner": None, "error": "tests/perf/checks.yaml: yq: bad indentation"},
    ]
    assert "run_micro=true\n" in out.read_text()


@pytest.mark.parametrize("converted", ["", "null\n", "[]\n"])
def test_registry_with_no_checks_plans_micro_alone(tmp_path, converted):
    (tmp_path / "checks.json").write_text(converted)
    argv = ["plan", "--registry", str(tmp_path / "checks.json"), "--out", str(tmp_path / "plan.json")]
    assert merge.main(argv) == 0
    assert json.loads((tmp_path / "plan.json").read_text()) == MICRO_TIER


def parse_yaml(path):
    """The way the workflow reads checks.yaml (yq), else PyYAML."""
    if shutil.which("yq"):
        text = subprocess.run(
            ["yq", "-o=json", ".", str(path)], check=True, capture_output=True, text=True
        ).stdout
        return json.loads(text) if text.strip() else None
    yaml = pytest.importorskip("yaml", reason="needs yq or PyYAML to parse checks.yaml")
    return yaml.safe_load(path.read_text())


def test_the_real_registry_is_usable():
    """Every check PR appends to tests/perf/checks.yaml; this is where a bad
    append fails, before it can turn a night into an error."""
    plan = merge.build_plan(parse_yaml(HERE / "checks.yaml"), "")
    assert plan[0]["check"] == "micro"
    assert [row for row in plan if row["runner"] is None] == []


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
    assert [c["check"] for c in nightly["checks"]] == ["micro", "micro-hw", "size-probe", "ldbc"]
    assert nightly["checks"][0] == result("micro")
    assert [c["error"] for c in nightly["checks"][1:]] == ["no result.json"] * 3


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
    assert [c["verdict"] for c in nightly["checks"]] == ["regression", "error", "error"]
    assert nightly["status"] == "red"


def test_an_error_result_is_kept_whole(tmp_path):
    results = tmp_path / "results"
    put(results, "micro", result("micro", "error"))
    _, nightly = run_merge(tmp_path, MICRO_ONLY)
    assert nightly["checks"] == [result("micro", "error")]
    assert nightly["status"] == "error"


def test_invalid_result_becomes_a_stub_naming_the_codes(tmp_path):
    results = tmp_path / "results"
    tag = "falkordb/falkordb-server:6.2.0"
    bad = result("micro", candidate={**CANDIDATE, "image": tag}, baseline={**BASELINE, "commit": "d8c"})
    put(results, "micro", bad)
    _, nightly = run_merge(tmp_path, MICRO_ONLY)
    assert nightly["checks"] == [
        {"check": "micro", "verdict": "error", "error": "invalid result.json (image-not-digest, bad-commit)"}
    ]


def test_invalid_error_result_keeps_what_the_check_said(tmp_path):
    """A check handed a tag cannot write a valid result (no digest to record),
    but its own message must reach the report."""
    said = "candidate image falkordb/falkordb-server:6.2.0 is not pinned by digest"
    bad = result("micro", "error", error=said, candidate={**CANDIDATE, "image": "falkordb/falkordb-server:6.2.0"})
    put(tmp_path / "results", "micro", bad)
    _, nightly = run_merge(tmp_path, MICRO_ONLY)
    assert nightly["checks"][0]["error"] == f"invalid result.json (image-not-digest); check said: {said}"


def test_nan_in_a_result_becomes_a_stub(tmp_path):
    results = tmp_path / "results"
    text = json.dumps(result("micro", "regression")).replace("3.0", "NaN", 1)
    put(results, "micro", text=text)
    _, nightly = run_merge(tmp_path, MICRO_ONLY)
    assert nightly["checks"][0]["error"].startswith("invalid result.json (")
    assert nightly["status"] == "error"


def test_unparsable_result_becomes_a_stub(tmp_path):
    put(tmp_path / "results", "micro", text="{not json")
    _, nightly = run_merge(tmp_path, MICRO_ONLY)
    assert nightly["checks"][0]["error"] == "invalid result.json (invalid-json)"


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
        ({"create_error": None, "timed_out": True}, "no result.json: the check timed out"),
        ({"create_error": "", "command_exit": 137}, "no result.json: the check exited 137"),
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


def test_unplanned_and_internal_artifacts_are_ignored(tmp_path):
    """_perf-micro.yml uploads perf-result-micro--cg-1 etc. beside the real one;
    they are not checks, and must not turn a passing night into an error."""
    results = tmp_path / "results"
    put(results, "micro", result("micro"))
    put(results, "micro--cg-1", text="cg", name="rows.csv")
    put(results, "extra", result("extra"))
    _, nightly = run_merge(tmp_path, MICRO_ONLY)
    assert [c["check"] for c in nightly["checks"]] == ["micro"]
    assert nightly["status"] == "green"


@pytest.mark.parametrize("facts", ["perf-hosted.json", "perf-vm.json"])
@pytest.mark.parametrize(
    "ran, verdict, kept",
    [
        ({"command_exit": 0, "timed_out": False}, "pass", True),
        ({"command_exit": 1, "timed_out": False}, "pass", False),
        ({"command_exit": 124, "timed_out": True}, "pass", False),
        ({"command_exit": 1, "timed_out": False}, "regression", True),
        ({"command_exit": 1, "timed_out": False}, "error", True),
    ],
)
def test_a_pass_from_a_check_that_crashed_is_not_green(tmp_path, facts, ran, verdict, kept):
    """Exit 0 whenever result.json was written (contract §1): a non-zero exit
    or a timeout after writing it means the check crashed, so its `pass` is
    not a measurement. A regression or error it wrote still stands."""
    results = tmp_path / "results"
    put(results, "ldbc", result("ldbc", verdict))
    put(results, "ldbc", ran, name=facts)
    _, nightly = run_merge(tmp_path, [{"check": "ldbc", "runner": "hosted"}])
    (entry,) = nightly["checks"]
    if kept:
        assert entry == result("ldbc", verdict)
    else:
        assert entry["verdict"] == "error"
        assert entry["error"].startswith("result.json says pass, but the check ")
        assert nightly["status"] == "error"


@pytest.mark.parametrize("runner", ["vm", "hosted"])
@pytest.mark.parametrize(
    "facts",
    [
        None,  # the job was cancelled before writing perf-hosted.json
        {"create_error": None, "command_exit": None, "timed_out": False},  # _perf-vm.yml, run job cancelled
        {"command_exit": "0"},
        {"command_exit": False},
    ],
)
def test_a_pass_needs_a_recorded_clean_exit(tmp_path, runner, facts):
    """Where the runner records the exit, silence is not success: a job
    cancelled mid-command uploads the result.json it wrote before the kill."""
    results = tmp_path / "results"
    put(results, "ldbc", result("ldbc"))
    if facts is not None:
        put(results, "ldbc", facts, name="perf-vm.json" if runner == "vm" else "perf-hosted.json")
    _, nightly = run_merge(tmp_path, [{"check": "ldbc", "runner": runner}])
    assert nightly["checks"][0]["error"] == (
        "result.json says pass, but nothing records that the check's command finished (no exit status)"
    )
    assert nightly["status"] == "error"


def test_micro_pass_needs_no_exit_record(tmp_path):
    """_perf-micro.yml writes result.json from its own job; it has no run-facts file."""
    put(tmp_path / "results", "micro", result("micro"))
    _, nightly = run_merge(tmp_path, MICRO_ONLY)
    assert nightly["status"] == "green"


FIXTURES = HERE / "fixtures" / "valid"


@pytest.mark.parametrize(
    "fixture", sorted(p.name for p in FIXTURES.glob("result-*.json")) or ["(none)"]
)
def test_perf_judges_valid_results_pass_through_whole(tmp_path, fixture):
    if fixture == "(none)":
        pytest.skip("tests/perf/fixtures/valid is perf-judge's and not here yet")
    doc = json.loads((FIXTURES / fixture).read_text())
    put(tmp_path / "results", doc["check"], doc)
    put(tmp_path / "results", doc["check"], {"command_exit": 0, "timed_out": False}, name="perf-hosted.json")
    plan = [{"check": doc["check"], "runner": "hosted"}]
    _, nightly = run_merge(tmp_path, plan)
    assert nightly["checks"] == [doc]


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


def night(status, commit=MAIN, base=BASE, date="2026-11-23", checks=None):
    if checks is None:
        verdict = {"green": "pass", "red": "regression", "error": "error"}.get(status)
        checks = [{"check": "micro", "verdict": verdict}] if verdict else []
    return {
        "schema": 1,
        "date": date,
        "run_url": f"https://github.com/FalkorDB/FalkorDB/actions/runs/{date}",
        "candidate": {**CANDIDATE, "commit": commit},
        "baseline": {**BASELINE, "image": base},
        "status": status,
        "checks": checks,
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
        # A red night whose other check could not measure (run 45332918572:
        # micro regressed, writes-nothing left no result.json): retry it too.
        history(
            night("red", checks=[
                {"check": "micro", "verdict": "regression"},
                {"check": "writes-nothing", "verdict": "error", "error": "no result.json"},
            ]),
            night("green"),
        ),
        history(night("green", commit=OLDER)),  # main moved
        history(night("green", base=OTHER_BASE)),  # a new release
    ],
)
def test_runs_when_anything_is_new_or_unsettled(nights):
    assert merge.skip_reason(nights, CANDIDATE, BASELINE) is None


def test_candidate_without_commit_never_skips():
    nights = history(night("green", commit=None))
    assert merge.skip_reason(nights, {**CANDIDATE, "commit": None}, BASELINE) is None


@pytest.mark.parametrize(
    "broken",
    [
        [1, 2],
        {**night("green"), "candidate": "main"},
        {**night("green"), "baseline": None},
        {**night("green"), "status": None},
        {**night("green"), "checks": None},
    ],
)
def test_an_unusable_earlier_nightly_counts_as_missing(tmp_path, broken):
    """Never crash on history (no nightly at all would follow), and never skip
    on the word of a file that cannot be read."""
    (tmp_path / "prev" / "30").mkdir(parents=True)
    (tmp_path / "prev" / "30" / "nightly.json").write_text(json.dumps(broken))
    (tmp_path / "runs.txt").write_text("30\n")
    history = merge.read_history(tmp_path / "runs.txt", tmp_path / "prev")
    assert history == [("30", None)]
    assert merge.skip_reason(history, CANDIDATE, BASELINE) is None


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
    "flag, value, said",
    [
        ("--candidate-image", "ghcr.io/falkordb/falkordb-server:rc-mg-x-valkey", "candidate image"),
        ("--baseline-image", "falkordb/falkordb-server@sha256:44eb4a6d", "baseline image"),
        ("--baseline-commit", "d8c329b", "baseline commit"),
        ("--candidate-label", "", "candidate label"),
    ],
)
def test_context_refuses_unpinned_or_malformed_sides(tmp_path, capsys, flag, value, said):
    args = {
        "--candidate-image": CAND, "--candidate-commit": MAIN, "--candidate-label": "main@4c10889",
        "--baseline-image": BASE, "--baseline-commit": RELEASE, "--baseline-label": "6.2.0",
    }
    args[flag] = value
    argv = ["context"] + [x for pair in args.items() for x in pair]
    argv += ["--date", "2026-11-24", "--run-url", "https://x/y", "--out", str(tmp_path / "c.json")]
    assert merge.main(argv) == 1
    assert f"::error::{said} " in capsys.readouterr().err
    assert not (tmp_path / "c.json").exists()


# --- conclude: the night's status as the run's ----------------------------------


@pytest.mark.parametrize(
    "status, code",
    [("green", 0), ("skipped", 0), ("red", 1), ("error", 1), ("unknown", 1), (None, 1)],
)
def test_only_green_and_skipped_nights_succeed(tmp_path, capsys, status, code):
    """A red night fails the run (Dvir, 2026-11-24), and so does an error night:
    one that could not measure is not green. Anything unrecognised fails too."""
    path = tmp_path / "nightly.json"
    path.write_text(json.dumps({"status": status}))
    assert merge.main(["conclude", "--nightly", str(path)]) == code
    out = capsys.readouterr().out
    assert out.startswith("::error::the night is ") if code else out.startswith("the night is ")


def test_conclude_on_a_real_nightly(tmp_path):
    """End to end: the nightly.json merge writes for a regression fails the run."""
    put(tmp_path / "results", "micro", result("micro", "regression"))
    _, nightly = run_merge(tmp_path, MICRO_ONLY)
    assert nightly["status"] == "red"
    assert merge.main(["conclude", "--nightly", str(tmp_path / "nightly.json")]) == 1


@pytest.mark.parametrize("text", ["{not json", "[]", '{"status": NaN}'])
def test_conclude_fails_on_an_unreadable_nightly(tmp_path, text):
    path = tmp_path / "nightly.json"
    path.write_text(text)
    assert merge.main(["conclude", "--nightly", str(path)]) == 1


# --- micro-hw: the micro tier's hardware-counter half, on a C4 VM -----------------

HW = [{"check": "micro-hw", "runner": "micro-hw"}]


def test_missing_micro_hw_is_a_stub_with_the_vm_reason(tmp_path):
    put(tmp_path / "results", "micro-hw", {"create_error": "no C4 in any zone"}, name="perf-vm.json")
    _, nightly = run_merge(tmp_path, HW)
    assert nightly["checks"] == [
        {"check": "micro-hw", "verdict": "error", "error": "no result.json: no VM came up: no C4 in any zone"}
    ]


@pytest.mark.parametrize("facts, kept", [({"command_exit": 0, "timed_out": False}, True), (None, False)])
def test_micro_hw_pass_needs_a_recorded_clean_exit(tmp_path, facts, kept):
    """It runs through _perf-vm.yml, so its pass is held to the vm rule."""
    put(tmp_path / "results", "micro-hw", result("micro-hw"))
    if facts is not None:
        put(tmp_path / "results", "micro-hw", facts, name="perf-vm.json")
    _, nightly = run_merge(tmp_path, HW)
    assert (nightly["checks"][0] == result("micro-hw")) is kept
