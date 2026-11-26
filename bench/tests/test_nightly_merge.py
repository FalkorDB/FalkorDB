"""`bench nightly merge` (falkorbench.nightly.merge): the plan, skip and merge
rules of the nightly perf run.

Every nightly.json these tests build is also run through the result contract
(falkorbench.contract), so "merge wrote it" and "the contract accepts it" are
checked together.
"""

import io
import json
import math
import re
import subprocess
from pathlib import Path

import pytest

from falkorbench import contract
from falkorbench.nightly import merge

HERE = Path(__file__).resolve().parent

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


CLEAN_EXIT = {"create_error": None, "command_exit": 0, "timed_out": False}


def measured(results, check, doc=None, text=None):
    """An artifact as _perf-vm.yml leaves it after a clean run: result.json and
    a perf-vm.json recording exit 0, which every pass needs."""
    put(results, check, doc, text=text)
    return put(results, check, CLEAN_EXIT, name="perf-vm.json")


def run_merge(tmp_path, plan, ctx=None):
    """merge.py's CLI end to end; returns (exit code, nightly)."""
    (tmp_path / "context.json").write_text(json.dumps(ctx or context()))
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    out = tmp_path / "nightly.json"
    code = merge.main(
        [
            "results",
            "--context",
            str(tmp_path / "context.json"),
            "--plan",
            str(tmp_path / "plan.json"),
            "--results",
            str(tmp_path / "results"),
            "--out",
            str(out),
        ]
    )
    nightly = json.loads(out.read_text())
    # Independent of merge's own check: the contract accepts what it wrote.
    assert contract.validate_nightly(nightly) == []
    return code, nightly


MICRO_HW = [{"check": "micro-hw", "runner": "micro-hw"}]
VM_ENTRY = {
    "name": "pokec",
    "runner": "vm",
    "command": "uv run --project bench bench pokec",
    "timeout_minutes": 75,
    "machine_type": "n4-standard-16",
}
HOSTED_ENTRY = {"name": "ldbc", "runner": "hosted", "command": "true", "timeout_minutes": 30}


# --- plan ------------------------------------------------------------------------


def test_empty_registry_plans_micro_hw_with_placeholder_matrices():
    plan = merge.build_plan([], "")
    assert plan == MICRO_HW
    out = merge.plan_outputs(plan)
    # _perf-micro.yml takes no switch for which checks to run (contract v1.11):
    # one output says whether to call it at all.
    assert set(out) == {"run_micro", "has_vm", "vm_matrix", "has_hosted", "hosted_matrix"}
    assert out["run_micro"] == "true"
    assert out["has_vm"] == out["has_hosted"] == "false"
    # Never an empty matrix: that fails the job before its `if:` is read.
    assert json.loads(out["vm_matrix"]) == merge.PLACEHOLDER
    assert json.loads(out["hosted_matrix"]) == merge.PLACEHOLDER


def test_registry_entries_become_matrix_rows():
    plan = merge.build_plan([VM_ENTRY, HOSTED_ENTRY], "")
    assert [row["check"] for row in plan] == ["micro-hw", "pokec", "ldbc"]
    out = merge.plan_outputs(plan)
    assert json.loads(out["vm_matrix"]) == [
        {
            "check": "pokec",
            "command": "uv run --project bench bench pokec",
            "timeout_minutes": 75,
            "machine_type": "n4-standard-16",
        }
    ]
    assert json.loads(out["hosted_matrix"])[0]["machine_type"] == ""
    assert out["has_vm"] == out["has_hosted"] == "true"


def test_selection_keeps_order_and_plans_unknown_names_as_stubs():
    plan = merge.build_plan([VM_ENTRY, HOSTED_ENTRY], " ldbc, typo ,micro-hw,ldbc")
    assert [row["check"] for row in plan] == ["ldbc", "typo", "micro-hw"]
    assert plan[1] == {"check": "typo", "runner": None, "error": "not in bench/checks.yaml"}
    assert merge.plan_outputs(plan)["has_vm"] == "false"


def test_the_removed_micro_check_is_an_unknown_name():
    """`micro` (callgrind on hosted runners) is gone (contract v1.11): a
    dispatch that still asks for it gets an error saying so, not a silent gap."""
    plan = merge.build_plan([], "micro")
    assert plan == [{"check": "micro", "runner": None, "error": "not in bench/checks.yaml"}]
    assert merge.plan_outputs(plan)["run_micro"] == "false"


@pytest.mark.parametrize(
    "select, run_micro",
    [("micro-hw", "true"), ("micro-hw,pokec", "true"), ("pokec", "false")],
)
def test_the_micro_call_runs_only_when_micro_hw_is_planned(select, run_micro):
    """_perf-micro.yml always runs micro-hw, so a selection without it must not
    call the workflow at all: that is what keeps it from starting a C4 VM."""
    assert merge.plan_outputs(merge.build_plan([VM_ENTRY], select))["run_micro"] == run_micro


@pytest.mark.parametrize(
    "entry, why",
    [
        ({**HOSTED_ENTRY, "runner": "gce"}, "runner 'gce'"),
        ({**HOSTED_ENTRY, "command": " "}, "command is missing"),
        ({**HOSTED_ENTRY, "timeout_minutes": "30"}, "timeout_minutes '30'"),
        ({**HOSTED_ENTRY, "timeout_minutes": 331}, "from 1 to 330 (hosted)"),
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
        (
            [{**HOSTED_ENTRY, "name": "Bad_Name"}, VM_ENTRY],
            "entry 0: name 'Bad_Name' is not a check name",
        ),
        (
            [{**HOSTED_ENTRY, "name": "ldbc--x"}, VM_ENTRY],
            "entry 0: name 'ldbc--x' is not a check name",
        ),
        ([{**HOSTED_ENTRY, "name": "micro-hw"}, VM_ENTRY], "entry 0: name 'micro-hw' is reserved"),
        (
            [{**HOSTED_ENTRY, "name": "checks-yaml"}, VM_ENTRY],
            "entry 0: name 'checks-yaml' is reserved",
        ),
        ([VM_ENTRY, VM_ENTRY], "entry 1: name 'pokec' is reserved or already used"),
        (["pokec", VM_ENTRY], "entry 0 is not a mapping"),
    ],
)
def test_registry_problems_are_reported_not_fatal(registry, why):
    """A bad registry must not stop the night: micro-hw (and every usable
    entry) still runs, and the problem is a `checks-yaml` error in the report."""
    plan = merge.build_plan(registry, "")
    assert plan[0] == {"check": "micro-hw", "runner": "micro-hw"}
    assert plan[-1]["check"] == "checks-yaml" and plan[-1]["runner"] is None
    assert why in plan[-1]["error"]
    if isinstance(registry, list) and VM_ENTRY in registry:
        assert merge.plan_outputs(plan)["has_vm"] == "true"


def test_registry_that_does_not_parse_still_plans_micro_hw(tmp_path):
    out = tmp_path / "github_output"
    argv = ["plan", "--registry-error", "yq: bad indentation", "--out", str(tmp_path / "plan.json")]
    assert merge.main([*argv, "--github-output", str(out)]) == 0
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan == [
        *MICRO_HW,
        {"check": "checks-yaml", "runner": None, "error": "bench/checks.yaml: yq: bad indentation"},
    ]
    assert "run_micro=true\n" in out.read_text()


@pytest.mark.parametrize("converted", ["", "null\n", "[]\n"])
def test_registry_with_no_checks_plans_micro_hw_alone(tmp_path, converted):
    (tmp_path / "checks.json").write_text(converted)
    argv = [
        "plan",
        "--registry",
        str(tmp_path / "checks.json"),
        "--out",
        str(tmp_path / "plan.json"),
    ]
    assert merge.main(argv) == 0
    assert json.loads((tmp_path / "plan.json").read_text()) == MICRO_HW


def parse_yaml(path):
    """A YAML file the strict way (PyYAML, from bench's dev group). A key given
    twice is an append gone wrong, and PyYAML's own loader would keep the last
    value, so this one refuses it."""
    import yaml

    class Strict(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark
                )
            seen.add(key)
        return yaml.SafeLoader.construct_mapping(loader, node, deep)

    Strict.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    return yaml.load(path.read_text(), Loader=Strict)


def test_the_registry_parser_refuses_a_duplicate_key(tmp_path):
    import yaml

    path = tmp_path / "checks.yaml"
    path.write_text("- name: ldbc\n  command: a\n  command: b\n")
    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key 'command'"):
        parse_yaml(path)


def test_the_real_registry_is_usable():
    """Every check PR appends to bench/checks.yaml; this is where a bad
    append fails, before it can turn a night into an error."""
    plan = merge.build_plan(parse_yaml(HERE.parent / "checks.yaml"), "")
    assert plan[0]["check"] == "micro-hw"
    assert [row for row in plan if row["runner"] is None] == []


# --- merge -----------------------------------------------------------------------


def test_missing_micro_hw_result_is_a_stub_and_the_night_is_not_green(tmp_path):
    (tmp_path / "results").mkdir()
    code, nightly = run_merge(tmp_path, MICRO_HW)
    assert code == 0
    assert nightly["checks"] == [
        {"check": "micro-hw", "verdict": "error", "error": "no result.json"}
    ]
    assert nightly["status"] == "error"


def test_every_planned_check_appears_once(tmp_path):
    results = tmp_path / "results"
    measured(results, "micro-hw", result("micro-hw"))
    plan = merge.build_plan([VM_ENTRY, HOSTED_ENTRY], "")
    code, nightly = run_merge(tmp_path, plan)
    assert code == 0
    assert [c["check"] for c in nightly["checks"]] == ["micro-hw", "pokec", "ldbc"]
    assert nightly["checks"][0] == result("micro-hw")
    assert [c["error"] for c in nightly["checks"][1:]] == ["no result.json"] * 2


def test_all_pass_is_green_and_records_both_sides(tmp_path):
    results = tmp_path / "results"
    measured(results, "micro-hw", result("micro-hw"))
    code, nightly = run_merge(tmp_path, MICRO_HW, context(previous_green=OLDER))
    assert code == 0
    assert nightly["status"] == "green"
    assert nightly["candidate"] == CANDIDATE and nightly["baseline"] == BASELINE
    assert nightly["previous_green_commit"] == OLDER
    assert nightly["skip_reason"] is None


def test_regression_outranks_error(tmp_path):
    results = tmp_path / "results"
    measured(results, "micro-hw", result("micro-hw", "regression"))
    plan = merge.build_plan([HOSTED_ENTRY], "")
    _, nightly = run_merge(tmp_path, plan)
    assert [c["verdict"] for c in nightly["checks"]] == ["regression", "error"]
    assert nightly["status"] == "red"


def test_an_error_result_is_kept_whole(tmp_path):
    results = tmp_path / "results"
    measured(results, "micro-hw", result("micro-hw", "error"))
    _, nightly = run_merge(tmp_path, MICRO_HW)
    assert nightly["checks"] == [result("micro-hw", "error")]
    assert nightly["status"] == "error"


def test_invalid_result_becomes_a_stub_naming_the_codes(tmp_path):
    results = tmp_path / "results"
    tag = "falkordb/falkordb-server:6.2.0"
    bad = result(
        "micro-hw", candidate={**CANDIDATE, "image": tag}, baseline={**BASELINE, "commit": "d8c"}
    )
    measured(results, "micro-hw", bad)
    _, nightly = run_merge(tmp_path, MICRO_HW)
    assert nightly["checks"] == [
        {
            "check": "micro-hw",
            "verdict": "error",
            "error": "invalid result.json (image-not-digest, bad-commit)",
        }
    ]


def test_invalid_error_result_keeps_what_the_check_said(tmp_path):
    """A check handed a tag cannot write a valid result (no digest to record),
    but its own message must reach the report."""
    said = "candidate image falkordb/falkordb-server:6.2.0 is not pinned by digest"
    bad = result(
        "micro-hw",
        "error",
        error=said,
        candidate={**CANDIDATE, "image": "falkordb/falkordb-server:6.2.0"},
    )
    measured(tmp_path / "results", "micro-hw", bad)
    _, nightly = run_merge(tmp_path, MICRO_HW)
    assert (
        nightly["checks"][0]["error"]
        == f"invalid result.json (image-not-digest); check said: {said}"
    )


def test_nan_in_a_result_becomes_a_stub(tmp_path):
    results = tmp_path / "results"
    text = json.dumps(result("micro-hw", "regression")).replace("3.0", "NaN", 1)
    measured(results, "micro-hw", text=text)
    _, nightly = run_merge(tmp_path, MICRO_HW)
    assert nightly["checks"][0]["error"].startswith("invalid result.json (")
    assert nightly["status"] == "error"


def test_unparsable_result_becomes_a_stub(tmp_path):
    measured(tmp_path / "results", "micro-hw", text="{not json")
    _, nightly = run_merge(tmp_path, MICRO_HW)
    assert nightly["checks"][0]["error"] == "invalid result.json (invalid-json)"


def test_result_naming_another_check_becomes_a_stub(tmp_path):
    measured(tmp_path / "results", "micro-hw", result("pokec"))
    _, nightly = run_merge(tmp_path, MICRO_HW)
    assert nightly["checks"][0]["error"] == "result.json names check 'pokec'"


def test_result_below_the_artifact_root_is_reported_as_misplaced(tmp_path):
    artifact = put(tmp_path / "results", "micro-hw", CLEAN_EXIT, name="perf-vm.json")
    (artifact / "out").mkdir()
    (artifact / "out" / "result.json").write_text(json.dumps(result("micro-hw")))
    _, nightly = run_merge(tmp_path, MICRO_HW)
    assert (
        nightly["checks"][0]["error"]
        == "no result.json at the artifact root (found out/result.json)"
    )


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
    plan = merge.build_plan([VM_ENTRY], "pokec")
    put(tmp_path / "results", "pokec", vm, name="perf-vm.json")
    _, nightly = run_merge(tmp_path, plan)
    assert nightly["checks"] == [{"check": "pokec", "verdict": "error", "error": reason}]


def test_planned_stub_keeps_its_reason(tmp_path):
    (tmp_path / "results").mkdir()
    plan = merge.build_plan([], "micro-hw,typo")
    _, nightly = run_merge(tmp_path, plan)
    assert nightly["checks"][1] == {
        "check": "typo",
        "verdict": "error",
        "error": "not in bench/checks.yaml",
    }


def test_unplanned_and_internal_artifacts_are_ignored(tmp_path):
    """A reusable workflow's internal artifacts (<artifact>--*) sit beside the
    real one; they are not checks, and must not turn a passing night into an
    error."""
    results = tmp_path / "results"
    measured(results, "micro-hw", result("micro-hw"))
    put(results, "micro-hw--rows", text="rows", name="rows.csv")
    measured(results, "extra", result("extra"))
    _, nightly = run_merge(tmp_path, MICRO_HW)
    assert [c["check"] for c in nightly["checks"]] == ["micro-hw"]
    assert nightly["status"] == "green"


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
def test_a_pass_from_a_check_that_crashed_is_not_green(tmp_path, ran, verdict, kept):
    """Exit 0 whenever result.json was written (contract §1): a non-zero exit
    or a timeout after writing it means the check crashed, so its `pass` is
    not a measurement. A regression or error it wrote still stands."""
    results = tmp_path / "results"
    put(results, "ldbc", result("ldbc", verdict))
    put(results, "ldbc", ran, name="perf-vm.json")
    _, nightly = run_merge(tmp_path, [{"check": "ldbc", "runner": "hosted"}])
    (entry,) = nightly["checks"]
    if kept:
        assert entry == result("ldbc", verdict)
    else:
        assert entry["verdict"] == "error"
        assert entry["error"].startswith("result.json says pass, but the check ")
        assert nightly["status"] == "error"


@pytest.mark.parametrize("runner", ["vm", "hosted", "micro-hw"])
@pytest.mark.parametrize(
    "facts",
    [
        None,  # the job was cancelled before writing perf-vm.json
        {
            "create_error": None,
            "command_exit": None,
            "timed_out": False,
        },  # _perf-vm.yml, run job cancelled
        {"command_exit": "0"},
        {"command_exit": False},
    ],
)
def test_a_pass_needs_a_recorded_clean_exit(tmp_path, runner, facts):
    """Every runner records the exit, and silence is not success: a job
    cancelled mid-command uploads the result.json it wrote before the kill."""
    results = tmp_path / "results"
    put(results, "ldbc", result("ldbc"))
    if facts is not None:
        put(results, "ldbc", facts, name="perf-vm.json")
    _, nightly = run_merge(tmp_path, [{"check": "ldbc", "runner": runner}])
    assert nightly["checks"][0]["error"] == (
        "result.json says pass, but nothing records that the check's command finished (no exit status)"
    )
    assert nightly["status"] == "error"


FIXTURES = HERE / "fixtures" / "valid"


def test_perf_judges_valid_results_are_here():
    assert sorted(FIXTURES.glob("result-*.json")), f"no result fixtures in {FIXTURES}"


@pytest.mark.parametrize("fixture", sorted(p.name for p in FIXTURES.glob("result-*.json")))
def test_perf_judges_valid_results_pass_through_whole(tmp_path, fixture):
    doc = json.loads((FIXTURES / fixture).read_text())
    measured(tmp_path / "results", doc["check"], doc)
    plan = [{"check": doc["check"], "runner": "vm"}]
    ctx = {**context(), "candidate": doc["candidate"], "baseline": doc["baseline"]}
    _, nightly = run_merge(tmp_path, plan, ctx)
    assert nightly["checks"] == [doc]


def test_skipped_night_lists_no_checks(tmp_path):
    results = tmp_path / "results"
    measured(results, "micro-hw", result("micro-hw"))
    code, nightly = run_merge(tmp_path, MICRO_HW, context(skip_reason="main has not moved"))
    assert code == 0
    assert nightly["status"] == "skipped"
    assert nightly["checks"] == []
    assert nightly["skip_reason"] == "main has not moved"


def test_json_out_refuses_nan(tmp_path):
    with pytest.raises(ValueError):
        merge.write_json(tmp_path / "x.json", {"ratio": math.nan})


# --- the workflow file -----------------------------------------------------------

WORKFLOWS = HERE.parent.parent / ".github" / "workflows"


def workflow(name):
    doc = parse_yaml(WORKFLOWS / name)
    if True in doc:  # PyYAML reads the key `on` as True
        doc["on"] = doc.pop(True)
    return doc


def test_each_reusable_workflow_gets_only_its_inputs():
    """GitHub rejects, at startup, a call that passes an input the callee does
    not declare, and actionlint cannot see across an owner/repo@ref call: one
    stale `micro_hw:` would fail every night before any job runs."""
    jobs = workflow("perf-nightly.yml")["jobs"]
    called = {name: job for name, job in jobs.items() if "uses" in job}
    files = {name: job["uses"].split("@")[0].rsplit("/", 1)[-1] for name, job in called.items()}
    assert files == {"micro": "_perf-micro.yml", "vm": "_perf-vm.yml"}
    for name, job in called.items():
        declared = workflow(files[name])["on"]["workflow_call"]["inputs"]
        passed = set(job.get("with", {}))
        assert passed <= set(declared), (name, sorted(passed - set(declared)))
        assert {key for key, spec in declared.items() if spec.get("required")} <= passed, name
        assert job.get("secrets") == "inherit", name


def test_every_resolve_output_a_job_reads_is_declared():
    """An undeclared needs.resolve.outputs.<name> reads as '' without an error."""
    read = set(
        re.findall(
            r"needs\.resolve\.outputs\.([a-z_]+)", (WORKFLOWS / "perf-nightly.yml").read_text()
        )
    )
    declared = set(workflow("perf-nightly.yml")["jobs"]["resolve"]["outputs"])
    assert read and read <= declared, sorted(read - declared)


def test_micro_is_called_only_when_planned_and_conclude_runs_last():
    jobs = workflow("perf-nightly.yml")["jobs"]
    assert "needs.resolve.outputs.run_micro == 'true'" in jobs["micro"]["if"]
    steps = jobs["report"]["steps"]
    # The run fails for a red or error night even when the report step failed.
    assert "bench nightly merge conclude" in steps[-1]["run"] and "!cancelled()" in steps[-1]["if"]
    post = next(step for step in steps if "bench nightly report" in step.get("run", ""))
    assert "GH_TOKEN" not in post.get("env", {})


# --- context: skip and previous green --------------------------------------------


def night(status, commit=MAIN, base=BASE, date="2026-11-23", checks=None):
    if checks is None:
        verdict = {"green": "pass", "red": "regression", "error": "error"}.get(status)
        checks = [{"check": "micro-hw", "verdict": verdict}] if verdict else []
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


PLANNED = ["micro-hw"]


@pytest.mark.parametrize("status", ["green", "red", "skipped"])
def test_unchanged_pair_after_a_settled_night_skips(status):
    reason = merge.skip_reason(history(night(status), night("green")), CANDIDATE, BASELINE, PLANNED)
    assert reason is not None and "main has not moved" in reason


def test_skip_reason_names_the_night_that_measured(tmp_path):
    nights = history(
        night("skipped", date="2026-11-23"),
        night("skipped", date="2026-11-22"),
        night("red", date="2026-11-21"),
    )
    reason = merge.skip_reason(nights, CANDIDATE, BASELINE, PLANNED)
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
            night(
                "red",
                checks=[
                    {"check": "micro-hw", "verdict": "regression"},
                    {"check": "writes-nothing", "verdict": "error", "error": "no result.json"},
                ],
            ),
            night("green"),
        ),
        history(night("green", commit=OLDER)),  # main moved
        history(night("green", base=OTHER_BASE)),  # a new release
    ],
)
def test_runs_when_anything_is_new_or_unsettled(nights):
    assert merge.skip_reason(nights, CANDIDATE, BASELINE, PLANNED) is None


def test_candidate_without_commit_never_skips():
    nights = history(night("green", commit=None))
    assert merge.skip_reason(nights, {**CANDIDATE, "commit": None}, BASELINE, PLANNED) is None


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
    assert merge.skip_reason(history, CANDIDATE, BASELINE, PLANNED) is None


def test_previous_green_is_the_newest_green_night():
    nights = history(
        night("red"), None, night("skipped"), night("green", commit=OLDER), night("green")
    )
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
    (tmp_path / "plan.json").write_text(json.dumps(MICRO_HW))
    args = [
        "context",
        "--plan",
        str(tmp_path / "plan.json"),
        "--candidate",
        CAND,
        "--candidate-commit",
        MAIN,
        "--candidate-label",
        "main@4c10889",
        "--baseline",
        BASE,
        "--baseline-commit",
        RELEASE,
        "--baseline-label",
        "6.2.0",
        "--history-runs",
        str(runs),
        "--history-dir",
        str(prev),
        "--date",
        "2026-11-24",
        "--run-url",
        "https://github.com/x/y/actions/runs/2",
        "--out",
        str(tmp_path / "context.json"),
        "--github-output",
        str(output),
    ]
    assert merge.main([*args, "--may-skip"]) == 0
    ctx = json.loads((tmp_path / "context.json").read_text())
    assert ctx["skip_reason"].startswith("main has not moved since the 2026-11-23 nightly")
    assert ctx["previous_green_commit"] == MAIN
    assert output.read_text() == "skip=true\n"

    # A registry error keeps the check's name but must be reported, not skipped.
    broken = [
        {**row, "runner": None, "error": "invalid bench/checks.yaml entry"} for row in MICRO_HW
    ]
    (tmp_path / "plan.json").write_text(json.dumps(broken))
    assert merge.main([*args, "--may-skip"]) == 0
    assert json.loads((tmp_path / "context.json").read_text())["skip_reason"] is None
    (tmp_path / "plan.json").write_text(json.dumps(MICRO_HW))

    output.write_text("")
    assert merge.main(args) == 0  # a dispatch never skips
    assert json.loads((tmp_path / "context.json").read_text())["skip_reason"] is None
    assert output.read_text() == "skip=false\n"


@pytest.mark.parametrize(
    "flag, value, said",
    [
        ("--candidate", "ghcr.io/falkordb/falkordb-server:rc-mg-x-valkey", "--candidate "),
        ("--baseline", "falkordb/falkordb-server@sha256:44eb4a6d", "--baseline "),
        # Pinned, but not a reference the contract accepts (upper case): taking
        # it would make every result of the night invalid.
        ("--candidate", "GHCR.io/FalkorDB/Server@sha256:" + "a" * 64, "--candidate "),
        ("--baseline-commit", "d8c329b", "--baseline-commit "),
        ("--candidate-label", "", "--candidate-label "),
    ],
)
def test_context_refuses_unpinned_or_malformed_sides(tmp_path, capsys, flag, value, said):
    args = {
        "--candidate": CAND,
        "--candidate-commit": MAIN,
        "--candidate-label": "main@4c10889",
        "--baseline": BASE,
        "--baseline-commit": RELEASE,
        "--baseline-label": "6.2.0",
    }
    args[flag] = value
    argv = ["context"] + [x for pair in args.items() for x in pair]
    (tmp_path / "plan.json").write_text(json.dumps(MICRO_HW))
    argv += ["--plan", str(tmp_path / "plan.json")]
    argv += ["--date", "2026-11-24", "--run-url", "https://x/y", "--out", str(tmp_path / "c.json")]
    assert merge.main(argv) == 1
    # kit.input_errors' wording: the same check every perf check runs on its flags.
    assert f"::error::{said}" in capsys.readouterr().err
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


def test_a_lone_surrogate_in_a_summary_does_not_fail_the_merge(tmp_path, monkeypatch):
    doc = result("micro-hw")
    doc["summary"] = "bad SURROGATE text"
    text = json.dumps(doc).replace("SURROGATE", "\\ud800")  # the JSON escape, not the character
    measured(tmp_path / "results", "micro-hw", text=text)
    strict = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="strict")
    monkeypatch.setattr("sys.stdout", strict)
    code, nightly = run_merge(tmp_path, MICRO_HW)
    assert code == 0
    assert nightly["status"] == "green"


def test_conclude_on_a_real_nightly(tmp_path):
    """End to end: the nightly.json merge writes for a regression fails the run."""
    measured(tmp_path / "results", "micro-hw", result("micro-hw", "regression"))
    _, nightly = run_merge(tmp_path, MICRO_HW)
    assert nightly["status"] == "red"
    assert merge.main(["conclude", "--nightly", str(tmp_path / "nightly.json")]) == 1


@pytest.mark.parametrize("text", ["{not json", "[]", '{"status": NaN}'])
def test_conclude_fails_on_an_unreadable_nightly(tmp_path, text):
    path = tmp_path / "nightly.json"
    path.write_text(text)
    assert merge.main(["conclude", "--nightly", str(path)]) == 1


# --- conclude as the release gate: waivers (#4004) -------------------------------


def finding(name, status="regression", metric="instructions"):
    return {
        "name": name,
        "metric": metric,
        "candidate": 2.0,
        "baseline": 1.0,
        "ratio": 2.0,
        "threshold": 1.1,
        "direction": "higher_is_worse",
        "status": status,
    }


def red_night(*checks):
    return {"status": "red", "checks": list(checks)}


def regressed(check, *names):
    return {"check": check, "verdict": "regression", "findings": [finding(n) for n in names]}


def waiver(check="micro-hw", pattern=r"cross product filter \[reads\]", issue=3996):
    return {"check": check, "finding": pattern, "issue": issue, "reason": "accepted"}


def decide(nightly, *waivers, gate=True):
    return merge.conclude(nightly, merge.read_waivers(list(waivers)), gate=gate)


def test_a_waived_regression_passes_and_is_still_reported():
    code, lines = decide(red_night(regressed("micro-hw", "cross product filter [reads]")), waiver())
    assert code == 0
    assert lines == [
        "the night is red: the run succeeds",
        "waived (reported, not gated): micro-hw: cross product filter [reads]"
        " (instructions 2.0x) waived by #3996",
    ]


def test_an_unwaived_regression_fails():
    code, lines = decide(red_night(regressed("micro-hw", "cross product filter [reads]")))
    assert code == 1
    assert lines[-1] == "not waived: micro-hw: cross product filter [reads] (instructions 2.0x)"


def test_one_unwaived_finding_fails_the_whole_night():
    night = red_night(regressed("micro-hw", "cross product filter [reads]", "id eq join [reads]"))
    code, lines = decide(night, waiver())
    assert code == 1
    assert any(line.startswith("waived") for line in lines)
    assert lines[-1].startswith("not waived: micro-hw: id eq join [reads]")


@pytest.mark.parametrize(
    "pattern, name, waived",
    [
        # The pattern must match the whole name: a bare row does not waive the
        # same row in another class, nor a longer name that contains it.
        ("cross product filter", "cross product filter [reads]", False),
        (r"cross product filter \[reads\]", "cross product filter [writes]", False),
        (r"cross product filter \[reads\]", "big cross product filter [reads]", False),
        (r"cross product filter \[reads\]", "Cross product filter [reads]", False),
        (r"cross product filter \[(reads|writes)\]", "cross product filter [writes]", True),
        (r"cross product .*", "cross product filter [reads]", True),
    ],
)
def test_a_waiver_matches_the_whole_finding_name(pattern, name, waived):
    code, _ = decide(red_night(regressed("micro-hw", name)), waiver(pattern=pattern))
    assert code == (0 if waived else 1)


def test_a_waiver_names_its_check():
    code, _ = decide(red_night(regressed("soak-b", "cross product filter [reads]")), waiver())
    assert code == 1


def test_errors_are_never_waived():
    """A check that could not measure fails the gate even when every regression
    beside it is waived; a re-run is what clears an infrastructure failure."""
    night = red_night(
        regressed("micro-hw", "cross product filter [reads]"),
        {"check": "soak-a", "verdict": "error", "error": "no VM came up"},
    )
    code, lines = decide(night, waiver(), waiver(check="soak-a", pattern=".*"))
    assert code == 1
    assert lines[-1] == "not waived: soak-a: error: no VM came up"
    code, _ = decide({"status": "error", "checks": [night["checks"][1]]}, waiver(pattern=".*"))
    assert code == 1


def test_a_regression_with_no_finding_cannot_be_waived():
    night = red_night({"check": "micro-hw", "verdict": "regression", "findings": []})
    code, lines = decide(night, waiver(pattern=".*"))
    assert code == 1
    assert lines[-1] == "not waived: micro-hw: regression with no finding to waive"


@pytest.mark.parametrize(
    "extra, metric, ratio, waived",
    [
        ({"metric": "instructions"}, "instructions", 2.0, True),
        ({"metric": "instructions"}, "alloc_bytes", 2.0, False),  # another metric on the row
        ({"max_ratio": 1.35}, "instructions", 1.30, True),
        ({"max_ratio": 1.35}, "instructions", 1.36, False),  # the row got worse again
        ({"max_ratio": 1.35}, "instructions", None, False),  # no ratio to bound
        ({"metric": "instructions", "max_ratio": 1.35}, "instructions", 1.35, True),
        ({"max_ratio": 1.5}, "instructions", 0.7, True),  # higher-is-worse: a faster row
    ],
)
def test_a_waiver_can_name_its_metric_and_bound_its_ratio(extra, metric, ratio, waived):
    row = {**finding("cross product filter [reads]", metric=metric), "ratio": ratio}
    night = red_night({"check": "micro-hw", "verdict": "regression", "findings": [row]})
    assert decide(night, {**waiver(), **extra})[0] == (0 if waived else 1)


def test_a_regression_from_a_run_that_did_not_finish_is_not_waived():
    """A check killed after it found a waived regression may have measured
    nothing else: its regression stands, unwaived."""
    night = red_night(regressed("micro-hw", "cross product filter [reads]"))
    assert decide(night, waiver())[0] == 0
    code, lines = merge.conclude(
        night,
        merge.read_waivers([waiver()]),
        gate=True,
        unfinished={"micro-hw": "the check timed out"},
    )
    assert code == 1
    assert lines[-1] == (
        "not waived: micro-hw: regression from a run that did not finish (the check timed out)"
    )


def incomplete_night(check, reason):
    entry = regressed(check, "cross product filter [reads]")
    return red_night({**entry, "incomplete": reason})


def test_a_waived_regression_from_an_incomplete_result_is_not_waived():
    """A result that says it left parts unmeasured or cut its findings is not
    trusted with a waiver: the night fails, naming the check and the reason."""
    night = incomplete_night("micro-hw", "3 rows not measured")
    code, lines = decide(night, waiver())
    assert code == 1
    assert lines[-1] == (
        "not waived: micro-hw: regression from a result that did not measure"
        " everything (3 rows not measured)"
    )
    assert not any(line.startswith("waived") for line in lines)


def test_a_waived_regression_with_a_null_or_absent_incomplete_still_passes():
    plain = regressed("micro-hw", "cross product filter [reads]")
    for entry in (plain, {**plain, "incomplete": None}):
        code, lines = decide(red_night(entry), waiver())
        assert code == 0
        assert lines[1].startswith("waived (reported, not gated)")


def test_an_incomplete_cov_gate_result_is_not_waived(tmp_path):
    entry = {**regressed("benchmark-cov", "UNWIND id seek"), "incomplete": "findings cut at 50"}
    night = red_night(entry)
    assert decide(night, waiver(check="benchmark-cov", pattern="UNWIND id seek"))[0] == 1
    entry["incomplete"] = None
    assert decide(night, waiver(check="benchmark-cov", pattern="UNWIND id seek"))[0] == 0


@pytest.mark.parametrize(
    "facts, unfinished",
    [
        (CLEAN_EXIT, {}),
        ({**CLEAN_EXIT, "timed_out": True}, {"micro-hw": "the check timed out"}),
        ({**CLEAN_EXIT, "command_exit": 143}, {"micro-hw": "the check exited 143"}),
        (None, {"micro-hw": "nothing records that the check's command finished (no exit status)"}),
    ],
)
def test_merge_lists_the_regressions_whose_check_did_not_finish(tmp_path, facts, unfinished):
    results = tmp_path / "results"
    put(results, "micro-hw", result("micro-hw", "regression"))
    if facts is not None:
        put(results, "micro-hw", facts, name="perf-vm.json")
    (tmp_path / "context.json").write_text(json.dumps(context()))
    (tmp_path / "plan.json").write_text(json.dumps(MICRO_HW))
    argv = ["results", "--context", str(tmp_path / "context.json")]
    argv += ["--plan", str(tmp_path / "plan.json"), "--results", str(results)]
    argv += ["--out", str(tmp_path / "nightly.json")]
    argv += ["--unfinished-out", str(tmp_path / "unfinished.json")]
    assert merge.main(argv) == 0
    assert json.loads((tmp_path / "unfinished.json").read_text()) == unfinished
    # And conclude reads it back: the waived regression is refused when unfinished.
    waivers = write(tmp_path, "w.json", [waiver(pattern="UNWIND id seek")])
    argv = ["conclude", "--gate", "--nightly", str(tmp_path / "nightly.json")]
    argv += ["--waivers", waivers, "--unfinished", str(tmp_path / "unfinished.json")]
    assert merge.main(argv) == (1 if unfinished else 0)


@pytest.mark.parametrize(
    "verdict, waivers, code",
    [
        ("pass", [], 0),
        ("regression", [], 1),
        ("regression", [waiver(check="benchmark-cov", pattern="UNWIND id seek")], 0),
        ("error", [waiver(check="benchmark-cov", pattern=".*")], 1),
    ],
)
def test_conclude_judges_one_result(tmp_path, verdict, waivers, code):
    """benchmark-cov's gate has one result.json, not a night: judged the same way."""
    path = write(tmp_path, "result.json", result("benchmark-cov", verdict))
    argv = ["conclude", "--gate", "--result", path, "--waivers", write(tmp_path, "w.json", waivers)]
    assert merge.main(argv) == code


def test_a_waived_single_result_must_be_valid(tmp_path, capsys):
    doc = result("benchmark-cov", "regression")
    doc["candidate"] = {**doc["candidate"], "image": "not-pinned:latest"}
    path = write(tmp_path, "result.json", doc)
    waivers = write(tmp_path, "w.json", [waiver(check="benchmark-cov", pattern=".*")])
    assert merge.main(["conclude", "--result", path, "--waivers", waivers]) == 1
    assert "the result is not valid" in capsys.readouterr().out


@pytest.mark.parametrize(
    "below, expected",
    [
        ("6.3.0", "v6.2.1"),  # a pull request into 6.2: the newest 6.2.x
        ("6.2.2", "v6.2.1"),  # tag v6.2.2: the release before it
        ("6.2.1", "v6.2.0"),  # re-releasing v6.2.1: never itself
        ("6.10.0", "v6.2.1"),  # versions compare as numbers, not text
        ("6.4.0-rc.1", "v6.2.1"),  # a prerelease tag of a new line
        ("6.2.1-rc.1", "v6.2.0"),  # 6.2.1 is above its own prerelease
        ("6.0.5", "v6.0.4"),  # the 6.0 line ignores the later 6.2
        ("6.0.0", None),  # nothing below (v4.x is the C engine, never a baseline)
    ],
)
def test_the_gate_baseline_is_the_highest_release_below_a_version(below, expected):
    tags = ["v6.2.0", "v4.22.4", "v6.0.4", "v6.2.1", "v6.0.1", "v6.2.2-rc.1", "6.1.0", "nightly"]
    assert merge.release_below(tags, below) == expected


@pytest.mark.parametrize("below", ["6.2", "v6.2", "6.2.x", ""])
def test_the_gate_baseline_needs_a_version(below):
    with pytest.raises(merge.PlanError):
        merge.release_below(["v6.2.1"], below)


@pytest.mark.parametrize("status", ["red", "error"])
@pytest.mark.parametrize("checks", [[], [{"check": "micro-hw", "verdict": "pass"}]])
def test_a_bad_night_that_names_no_reason_does_not_pass(status, checks):
    assert decide({"status": status, "checks": checks}, waiver(pattern=".*"))[0] == 1


def test_only_regression_findings_are_matched():
    """An improvement or info row a waiver happens to match explains nothing."""
    night = red_night(
        {
            "check": "micro-hw",
            "verdict": "regression",
            "findings": [finding("a", status="improvement"), finding("b")],
        }
    )
    assert decide(night, waiver(pattern="a"))[0] == 1
    assert decide(night, waiver(pattern="b"))[0] == 0


@pytest.mark.parametrize("gate, code", [(True, 1), (False, 0)])
def test_a_gate_that_measured_nothing_fails(gate, code):
    assert decide({"status": "skipped", "checks": []}, gate=gate)[0] == code


def test_green_passes_the_gate():
    assert decide({"status": "green", "checks": []}) == (
        0,
        ["the night is green: the run succeeds"],
    )


@pytest.mark.parametrize(
    "entry, why",
    [
        ({"check": "micro-hw", "finding": "x", "issue": 1}, "needs the keys"),
        ({**waiver(), "until": "6.2.2"}, "needs the keys"),
        ({**waiver(), "metric": "instr"}, "not a contract metric"),
        ({**waiver(), "max_ratio": 0}, "not a positive number"),
        ({**waiver(), "max_ratio": True}, "not a positive number"),
        ({**waiver(), "max_ratio": "1.5"}, "not a positive number"),
        (waiver(check="Micro"), "not a check name"),
        (waiver(check="checks-yaml"), "not a check name"),
        (waiver(issue=0), "not an issue number"),
        (waiver(issue=True), "not an issue number"),
        (waiver(issue="3996"), "not an issue number"),
        (waiver(pattern=""), "non-empty regex"),
        (waiver(pattern=["x"]), "non-empty regex"),
        (waiver(pattern="cross ["), "not a regex"),
        ({**waiver(), "reason": "two\nlines"}, "one non-empty line"),
        ({**waiver(), "reason": " "}, "one non-empty line"),
    ],
)
def test_a_waiver_conclude_cannot_use_is_refused(entry, why):
    with pytest.raises(merge.PlanError, match=why):
        merge.read_waivers([entry])


@pytest.mark.parametrize("doc", [{"check": "micro-hw"}, "micro-hw", ["x"]])
def test_waivers_must_be_a_list_of_mappings(doc):
    with pytest.raises(merge.PlanError):
        merge.read_waivers(doc)


def test_the_real_waivers_file_is_usable():
    """The file the gate reads, as yq hands it to conclude (null when it holds
    only comments)."""
    merge.read_waivers(parse_yaml(HERE.parent / "waivers.yaml"))


def write(tmp_path, name, doc):
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return str(path)


def test_conclude_end_to_end_with_a_waiver(tmp_path, capsys):
    """The nightly.json merge writes for a regression, through the CLI: refused
    without a waiver, passed with one, and the waived row is a warning on the
    run page."""
    measured(tmp_path / "results", "micro-hw", result("micro-hw", "regression"))
    run_merge(tmp_path, MICRO_HW)
    night = str(tmp_path / "nightly.json")
    none = write(tmp_path, "none.json", None)
    assert merge.main(["conclude", "--nightly", night, "--waivers", none, "--gate"]) == 1
    waivers = write(tmp_path, "w.json", [waiver(pattern="UNWIND id seek")])
    capsys.readouterr()
    assert merge.main(["conclude", "--nightly", night, "--waivers", waivers, "--gate"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "the night is red: the run succeeds"
    assert out[1].startswith("::warning::waived (reported, not gated): micro-hw: UNWIND id seek")


def test_a_waived_pass_needs_a_valid_nightly(tmp_path, capsys):
    night = write(
        tmp_path, "n.json", red_night(regressed("micro-hw", "cross product filter [reads]"))
    )
    waivers = write(tmp_path, "w.json", [waiver()])
    assert merge.main(["conclude", "--nightly", night, "--waivers", waivers]) == 1
    assert "not waived: the result is not valid" in capsys.readouterr().out


@pytest.mark.parametrize("text", ["", "\n", "null\n"])
def test_a_comment_only_waivers_file_waives_nothing(tmp_path, text):
    """yq prints nothing or `null` for bench/waivers.yaml as it starts."""
    (tmp_path / "w.json").write_text(text)
    for status, code in (("green", 0), ("red", 1)):
        night = write(
            tmp_path, "n.json", red_night(regressed("micro-hw", "x")) | {"status": status}
        )
        argv = ["conclude", "--nightly", night, "--waivers", str(tmp_path / "w.json")]
        assert merge.main(argv) == code


@pytest.mark.parametrize("text", ["{not json", '[{"check": "micro-hw"}]'])
def test_unreadable_waivers_fail_the_run(tmp_path, text):
    night = write(tmp_path, "n.json", {"status": "green", "checks": []})
    (tmp_path / "w.json").write_text(text)
    assert merge.main(["conclude", "--nightly", night, "--waivers", str(tmp_path / "w.json")]) == 1


# --- micro-hw: the micro tier's one check, on a C4 VM ----------------------------


def test_no_c4_capacity_is_a_micro_hw_stub_with_the_vm_reason(tmp_path):
    """No C4 in any zone: no result, so the night says error with the reason
    (accepted cost of contract v1.11), never green."""
    put(
        tmp_path / "results", "micro-hw", {"create_error": "no C4 in any zone"}, name="perf-vm.json"
    )
    _, nightly = run_merge(tmp_path, MICRO_HW)
    assert nightly["checks"] == [
        {
            "check": "micro-hw",
            "verdict": "error",
            "error": "no result.json: no VM came up: no C4 in any zone",
        }
    ]
    assert nightly["status"] == "error"


# --- the command line ------------------------------------------------------------


def test_bench_nightly_merge_runs_the_subcommands(tmp_path):
    """The workflow calls `uv run --project bench bench nightly merge <sub> …`:
    the click command passes every argument through to merge.main unchanged."""
    from click.testing import CliRunner

    from falkorbench.cli import cli

    (tmp_path / "checks.json").write_text("null\n")
    plan = tmp_path / "plan.json"
    run = CliRunner().invoke(
        cli,
        [
            "nightly",
            "merge",
            "plan",
            "--registry",
            str(tmp_path / "checks.json"),
            "--out",
            str(plan),
        ],
    )
    assert run.exit_code == 0, run.output
    assert json.loads(plan.read_text()) == MICRO_HW
    night = tmp_path / "nightly.json"
    night.write_text(json.dumps({"status": "red"}))
    run = CliRunner().invoke(cli, ["nightly", "merge", "conclude", "--nightly", str(night)])
    assert run.exit_code == 1


# --- review round: registry edges, skip edges, the pinned pair -------------------


@pytest.mark.parametrize("runner, limit", [("vm", 300), ("hosted", 330)])
def test_timeout_limit_is_the_runners(runner, limit):
    """tests/perf/vm/plan.sh refuses more than 300 minutes (the VM deletes itself
    6 h after boot), so a VM entry over that must fail here, not every night."""
    entry = {**(VM_ENTRY if runner == "vm" else HOSTED_ENTRY), "timeout_minutes": limit}
    (row,) = [r for r in merge.build_plan([entry], "") if r["check"] == entry["name"]]
    assert row["runner"] == runner
    (row,) = [
        r
        for r in merge.build_plan([{**entry, "timeout_minutes": limit + 1}], "")
        if r["check"] == entry["name"]
    ]
    assert row["runner"] is None and f"from 1 to {limit} ({runner})" in row["error"]


def test_a_name_with_a_trailing_newline_is_not_a_check_name():
    """`$` also matches before a final newline; a YAML block scalar can carry one."""
    plan = merge.build_plan([{**HOSTED_ENTRY, "name": "pokec\n"}], "")
    assert plan[-1]["check"] == "checks-yaml" and "is not a check name" in plan[-1]["error"]


def test_a_command_over_several_lines_runs_as_one_line():
    """run-check.sh reads one line: a folded command must not lose its tail."""
    entry = {**HOSTED_ENTRY, "command": "uv run --project bench\n  --frozen bench ldbc\n"}
    (row,) = [r for r in merge.build_plan([entry], "") if r["check"] == "ldbc"]
    assert row["command"] == "uv run --project bench --frozen bench ldbc"


def test_previous_green_survives_a_stretch_with_no_green_night():
    """The workflow reads a bounded number of earlier runs: with none of them
    green, the newest recorded previous green is handed on."""
    nights = history(
        night("red"), night("skipped"), {**night("red"), "previous_green_commit": OLDER}
    )
    assert merge.previous_green_commit(nights) == OLDER
    assert merge.previous_green_commit(history(night("red"), night("skipped"))) is None


@pytest.mark.parametrize(
    "nights, planned",
    [
        # Same commit, rebuilt image: new bytes to measure.
        (history({**night("green"), "candidate": {**CANDIDATE, "image": OTHER_BASE}}), PLANNED),
        # A check added to the registry while the pair stayed put.
        (history(night("green")), [*PLANNED, "pokec"]),
        # A registry that stopped parsing.
        (history(night("green")), [*PLANNED, "checks-yaml"]),
        # The night that measured the pair left no artifact: nothing says what it ran.
        (history(night("skipped"), None), PLANNED),
    ],
)
def test_no_skip_unless_the_measured_night_matches_tonight(nights, planned):
    assert merge.skip_reason(nights, CANDIDATE, BASELINE, planned) is None


def test_a_result_for_another_pair_is_not_tonights(tmp_path):
    """An earlier attempt's artifact, say: a pass for another candidate must not
    make tonight green."""
    other = {**CANDIDATE, "image": "ghcr.io/falkordb/falkordb-server@sha256:" + "c" * 64}
    measured(tmp_path / "results", "micro-hw", result("micro-hw", candidate=other))
    _, nightly = run_merge(tmp_path, MICRO_HW)
    (entry,) = nightly["checks"]
    assert entry["verdict"] == "error"
    assert entry["error"] == (
        f"result.json measured candidate sha256:{'c' * 64}, but the night pinned {CAND.rpartition('@')[2]}"
    )
    assert nightly["status"] == "error"


# --- the workflow's own shell steps, run for real with fake gh / docker -----------


def step_script(job, name):
    """The `run:` text of one step of perf-nightly.yml."""
    for step in workflow("perf-nightly.yml")["jobs"][job]["steps"]:
        if step.get("name", "").startswith(name) or step.get("id") == name:
            return step["run"]
    raise AssertionError(f"no step {name!r} in job {job!r}")


def fake_tool(bin_dir, name, script):
    path = bin_dir / name
    path.write_text("#!/bin/bash\n" + script)
    path.chmod(0o755)


def run_step(script, cwd, env):
    import os
    import subprocess

    return subprocess.run(
        ["bash", "-e", "-c", script],
        cwd=cwd,
        env={**os.environ, **env, "PATH": f"{env['BIN']}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )


def test_plan_runs_before_the_skip_decision_that_reads_it():
    steps = workflow("perf-nightly.yml")["jobs"]["resolve"]["steps"]
    order = [s.get("id") for s in steps]
    assert order.index("plan") < order.index("context")
    assert '--plan "$RUNNER_TEMP/plan/plan.json"' in step_script("resolve", "context")


def test_the_results_download_turns_failures_into_missing_results(tmp_path):
    """One artifact lookup failing, or one archive broken, must not stop the
    step: those checks are left without a result.json (merge stubs them) and
    the others still download."""
    import zipfile

    bin_dir, temp = tmp_path / "bin", tmp_path / "temp"
    bin_dir.mkdir()
    (temp / "plan").mkdir(parents=True)
    (temp / "plan" / "plan.json").write_text(
        json.dumps([{"check": c, "runner": "vm"} for c in ("lost", "broken", "fine")])
    )
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as archive:
        archive.writestr("result.json", "{}")
        archive.writestr("perf-vm.json", "{}")
    fake_tool(
        bin_dir,
        "gh",
        f"""case "$*" in
  *artifacts?name=perf-result-lost*) echo "HTTP 502" >&2; exit 1 ;;
  *artifacts?name=perf-result-broken*) echo 11 ;;
  *artifacts?name=perf-result-fine*) echo 12 ;;
  *artifacts/11/zip*) printf 'PK\\003\\004 not a zip' ;;
  *artifacts/12/zip*) cat {good} ;;
  *) echo "unexpected gh $*" >&2; exit 9 ;;
esac
""",
    )
    run = run_step(
        step_script("merge", "Download the results"),
        tmp_path,
        {
            "BIN": str(bin_dir),
            "RUNNER_TEMP": str(temp),
            "GITHUB_REPOSITORY": "o/r",
            "GITHUB_RUN_ID": "1",
        },
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "perf-result-lost: could not list its artifacts" in run.stdout
    assert "perf-result-broken: artifact 11 did not download" in run.stdout
    results = temp / "results"
    assert not (results / "perf-result-lost").exists()
    assert not (results / "perf-result-broken").exists()
    assert (results / "perf-result-fine" / "result.json").is_file()


def test_the_candidate_is_mains_newest_image_whatever_ref_runs(tmp_path):
    """A dispatch from another branch (whose checkout may sit many commits above
    main) still measures main's newest image, and falls back along main's
    first-parent history when the tip has none."""
    import os
    import subprocess

    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }

    def git(*args, cwd):
        return subprocess.run(
            ["git", *args], cwd=cwd, env=git_env, capture_output=True, text=True, check=True
        ).stdout.strip()

    origin, work = tmp_path / "origin.git", tmp_path / "work"
    git("init", "--quiet", "--bare", "--initial-branch=main", str(origin), cwd=tmp_path)
    git("clone", "--quiet", str(origin), str(work), cwd=tmp_path)
    main = []
    for n in range(3):
        git("commit", "--quiet", "--allow-empty", "-m", f"main {n}", cwd=work)
        main.append(git("rev-parse", "HEAD", cwd=work))
    git("push", "--quiet", "origin", "HEAD:main", cwd=work)
    git("checkout", "--quiet", "-b", "elsewhere", cwd=work)
    git("commit", "--quiet", "--allow-empty", "-m", "not main", cwd=work)
    elsewhere = git("rev-parse", "HEAD", cwd=work)

    bin_dir, temp = tmp_path / "bin", tmp_path / "temp"
    bin_dir.mkdir()
    temp.mkdir()
    # Only main's first commit has an image; so does the release tag.
    fake_tool(
        bin_dir,
        "docker",
        f"""ref=$4
case "$ref" in
  *rc-mg-{main[0]}-valkey|*:6.2.1) echo '{{"digest": "sha256:{"a" * 64}"}}' ;;
  *{elsewhere}*) echo '{{"digest": "sha256:{"e" * 64}"}}' ;;
  *) echo "ERROR: $ref: not found" >&2; exit 1 ;;
esac
""",
    )
    fake_tool(
        bin_dir,
        "gh",
        f"""case "$*" in
  *releases/latest*) echo v6.2.1 ;;
  *releases?per_page*) printf 'v6.2.1\\nv6.2.0\\n' ;;
  *commits/v6.2.1*) echo {"d" * 40} ;;
  *) echo "unexpected gh $*" >&2; exit 9 ;;
esac
""",
    )
    out = temp / "output"
    run = run_step(
        step_script("resolve", "images"),
        work,
        {
            "BIN": str(bin_dir),
            "RUNNER_TEMP": str(temp),
            "GITHUB_OUTPUT": str(out),
            "GITHUB_STEP_SUMMARY": str(temp / "summary"),
            "GITHUB_REPOSITORY": "o/r",
            "WALK": "20",
            "CANDIDATE_REPO": "ghcr.io/falkordb/falkordb-server",
            "BASELINE_REPO": "falkordb/falkordb-server",
            "CANDIDATE_INPUT": "",
            "BASELINE_INPUT": "",
            "BASELINE_BELOW": "",
        },
    )
    assert run.returncode == 0, run.stdout + run.stderr
    outputs = dict(line.split("=", 1) for line in out.read_text().splitlines())
    assert outputs["candidate_commit"] == main[0]
    assert outputs["candidate_label"] == f"main@{main[0][:7]} (tip {main[2][:7]} has no image)"
    assert outputs["baseline_label"] == "6.2.1"


def test_every_upload_replaces_an_earlier_attempts_artifact():
    """upload-artifact refuses a name an earlier attempt of the same run already
    uploaded, so without `overwrite` a re-run's merge (or plan, or hosted check)
    fails at its upload and the night is never reported."""
    uploads = [
        (job, step)
        for job, spec in workflow("perf-nightly.yml")["jobs"].items()
        for step in spec.get("steps", [])
        if step.get("uses", "").startswith("actions/upload-artifact@")
    ]
    assert len(uploads) == 3, [step["with"]["name"] for _, step in uploads]
    for job, step in uploads:
        assert step["with"].get("overwrite") is True, (job, step["with"]["name"])


RELEASES = [
    {"tag_name": tag, "draft": draft, "prerelease": pre}
    for tag, draft, pre in [
        ("v6.2.0", False, False),
        ("v4.22.4", False, False),  # the C engine shares the repository
        ("v6.0.4", False, False),
        ("v6.2.1", False, False),
        ("v6.3.0", True, False),  # a draft
        ("v6.2.2", False, True),  # a prerelease
        ("v6.0.1", False, False),
    ]
]


def fake_uv(bin_dir):
    """`uv run --project bench --frozen bench <args>` as this interpreter running
    falkorbench.cli, so the step runs this checkout's code wherever the test is."""
    import sys

    src = HERE.parent / "src"
    fake_tool(
        bin_dir,
        "uv",
        f"""[[ $1 == run && $5 == bench ]] || {{ echo "unexpected uv $*" >&2; exit 9; }}
shift 5
PYTHONPATH={src} exec {sys.executable} -I -m falkorbench.cli "$@"
""",
    )


@pytest.mark.parametrize(
    "below, baseline",
    [
        ("6.3.0", "6.2.1"),  # a pull request into 6.2: the newest 6.2.x
        ("6.2.2", "6.2.1"),  # tag v6.2.2: the release before it, not its prerelease
        ("6.2.1", "6.2.0"),  # re-releasing v6.2.1: never itself
        ("6.4.0", "6.2.1"),  # the draft v6.3.0 is no release
        ("6.0.5", "6.0.4"),  # the 6.0 line ignores the later 6.2
        ("6.0.0", None),  # nothing below: v4.22.4 is filtered out by the step's jq
        ("6.2", None),  # not a version
    ],
)
def test_a_gate_baseline_step_picks_the_highest_release_below(tmp_path, below, baseline):
    """The resolve step end to end: GitHub's release list through the step's own
    jq filter (stable, v5 and up), then `bench nightly merge release`."""
    import shutil

    if shutil.which("jq") is None:
        pytest.skip("jq is not installed")
    bin_dir, temp = tmp_path / "bin", tmp_path / "temp"
    bin_dir.mkdir()
    temp.mkdir()
    fake_uv(bin_dir)
    fake_tool(
        bin_dir,
        "docker",
        f"""case "$4" in
  *:[46].*|*rc-pr-7-valkey) echo '{{"digest": "sha256:{"a" * 64}"}}' ;;
  *) echo "ERROR: $4: not found" >&2; exit 1 ;;
esac
""",
    )
    (temp / "releases.json").write_text(json.dumps(RELEASES))
    fake_tool(
        bin_dir,
        "gh",
        f"""jqexpr=""
while [[ $# -gt 0 ]]; do [[ $1 == --jq ]] && jqexpr=$2; shift; done
case "$jqexpr" in
  *tag_name*) jq -r "$jqexpr" {temp / "releases.json"} ;;
  .sha) echo {"d" * 40} ;;
  *) echo "unexpected gh --jq $jqexpr" >&2; exit 9 ;;
esac
""",
    )
    out = temp / "output"
    run = run_step(
        step_script("resolve", "images"),
        tmp_path,
        {
            "BIN": str(bin_dir),
            "RUNNER_TEMP": str(temp),
            "GITHUB_OUTPUT": str(out),
            "GITHUB_STEP_SUMMARY": str(temp / "summary"),
            "GITHUB_REPOSITORY": "o/r",
            "WALK": "20",
            "CANDIDATE_REPO": "ghcr.io/falkordb/falkordb-server",
            "BASELINE_REPO": "falkordb/falkordb-server",
            "CANDIDATE_INPUT": "ghcr.io/falkordb/falkordb-server:rc-pr-7-valkey",
            "BASELINE_INPUT": "",
            "BASELINE_BELOW": below,
        },
    )
    if baseline is None:
        assert run.returncode != 0
        assert "::error::" in run.stdout + run.stderr
        return
    assert run.returncode == 0, run.stdout + run.stderr
    outputs = dict(line.split("=", 1) for line in out.read_text().splitlines())
    assert outputs["baseline_label"] == baseline
    assert outputs["candidate_label"] == "rc-pr-7-valkey"


# --- the release gate's callers (#4004) ------------------------------------------


GATE_CALLERS = {("rust-pr.yml", "perf"): "nightly"}


@pytest.mark.parametrize("caller, mode", GATE_CALLERS.items())
def test_a_gate_caller_passes_only_declared_inputs(caller, mode):
    """GitHub rejects a call with an undeclared input before any job runs, and
    actionlint cannot see across an owner/repo@ref call."""
    file, job_name = caller
    job = workflow(file)["jobs"][job_name]
    # Main's copy names `./` (its own perf code); a version branch's names @main.
    assert job["uses"] == "./.github/workflows/perf-nightly.yml"
    declared = workflow("perf-nightly.yml")["on"]["workflow_call"]["inputs"]
    passed = set(job["with"])
    assert passed <= set(declared), sorted(passed - set(declared))
    assert {key for key, spec in declared.items() if spec.get("required")} <= passed
    assert job["with"]["mode"] == mode
    assert job["secrets"] == "inherit"
    # A caller grants at least what the called jobs ask for.
    assert job["permissions"] == {"contents": "read", "actions": "read", "packages": "write"}


def test_every_job_runs_the_gate_workflows_own_commit():
    """Called from a version branch, the caller's commit has no perf code: every
    checkout names this workflow's commit, and the checks get it as code_ref."""
    jobs = workflow("perf-nightly.yml")["jobs"]
    checkouts = [
        step
        for job in jobs.values()
        for step in job.get("steps", [])
        if step.get("uses", "").startswith("actions/checkout@")
    ]
    assert len(checkouts) == 4
    for step in checkouts:
        assert step["with"]["repository"] == "${{ job.workflow_repository }}"
        assert step["with"]["ref"] == "${{ job.workflow_sha }}"
    for name in ("micro", "vm"):
        assert jobs[name]["uses"].startswith("./.github/workflows/")
        assert jobs[name]["with"]["code_ref"] == "${{ needs.resolve.outputs.code_sha }}"
        assert jobs[name]["with"]["mode"] == "${{ inputs.mode || 'nightly' }}"
    hardware = workflow("_perf-micro.yml")["jobs"]["hardware"]
    assert hardware["uses"] == "./.github/workflows/_perf-vm.yml"
    assert hardware["with"]["code_ref"] == "${{ inputs.code_ref }}"


def test_a_call_concludes_as_a_gate_with_the_waivers():
    conclude_step = workflow("perf-nightly.yml")["jobs"]["report"]["steps"][-1]
    assert conclude_step["env"]["GATE_MODE"] == "${{ inputs.mode }}"
    assert "--waivers" in conclude_step["run"] and "--gate" in conclude_step["run"]
    assert "yq -o=json '.' bench/waivers.yaml" in conclude_step["run"]


@pytest.mark.parametrize(
    "event, pull_base, queue_ref, below",
    [
        # Only the queue of a version branch X.Y gets the gate, whatever the ref's spelling.
        ("merge_group", "", "refs/heads/6.2", "6.3.0"),
        ("merge_group", "", "refs/heads/6.0", "6.1.0"),
        ("merge_group", "", "refs/heads/6.9", "6.10.0"),
        ("merge_group", "", "refs/heads/main", None),
        ("merge_group", "", "refs/heads/perf/fuzz-scenario", None),  # a stacked branch
        ("merge_group", "", "refs/heads/6.2.1", None),
        ("merge_group", "", "refs/heads/x6.2", None),
        # A pull request never does, even into a version branch (its CI is the cherry-pick's).
        ("pull_request", "6.2", "", None),
        ("pull_request", "6.0", "", None),
        ("pull_request", "main", "", None),
    ],
)
def test_only_the_queue_of_a_version_branch_gets_the_gate(
    tmp_path, event, pull_base, queue_ref, below
):
    run_job = workflow("rust-pr.yml")["jobs"]["check-files"]
    step = next(s for s in run_job["steps"] if s.get("id") == "key")
    assert step["env"]["QUEUE_BASE_REF"] == "${{ github.event.merge_group.base_ref }}"
    (tmp_path / "bin").mkdir()
    out = tmp_path / "output"
    env = {"BIN": str(tmp_path / "bin"), "GITHUB_OUTPUT": str(out), "GITHUB_SHA": "a" * 40}
    env = {**env, "EVENT": event, "PR_NUM": "7", "BASE_REF": pull_base, "QUEUE_BASE_REF": queue_ref}
    run = run_step(step["run"], tmp_path, env)
    assert run.returncode == 0, run.stderr
    outputs = dict(line.split("=", 1) for line in out.read_text().splitlines())
    assert outputs.get("perf_below") == below
    assert outputs["ci_key"] == ("mg-" + "a" * 40 if event == "merge_group" else "pr-7")


@pytest.mark.parametrize("job_name", ["perf", "perf-cov"])
@pytest.mark.parametrize(
    "perf_required, perf, passes",
    [
        ("false", "skipped", True),  # a pull request, or main's queue
        ("true", "success", True),
        (
            "true",
            "skipped",
            False,
        ),  # the CI suites failed, or the gate never ran in the version queue
        ("true", "failure", False),
        ("true", "cancelled", False),
        ("false", "failure", False),
    ],
)
def test_ci_gate_requires_the_perf_gate_only_where_it_runs(
    tmp_path, job_name, perf_required, perf, passes
):
    import shutil

    if shutil.which("jq") is None:
        pytest.skip("jq is not installed")
    job = workflow("rust-pr.yml")["jobs"]["ci-gate"]
    assert {"perf", "perf-cov"} <= set(job["needs"])
    step = job["steps"][0]
    names = [*job["needs"]]
    # The other gate job did what it does where it is required, or skipped elsewhere.
    results = {name: {"result": "success", "outputs": {}} for name in names}
    other = "perf-cov" if job_name == "perf" else "perf"
    results[other]["result"] = "success" if perf_required == "true" else "skipped"
    results[job_name]["result"] = perf
    (tmp_path / "bin").mkdir()
    env = {
        "BIN": str(tmp_path / "bin"),
        "RESULTS": json.dumps(results),
        "EVENT": "merge_group" if perf_required == "true" else "pull_request",
        "PERF_REQUIRED": perf_required,
    }
    run = run_step(step["run"], tmp_path, env)
    assert (run.returncode == 0) == passes, run.stdout + run.stderr


def test_release_tag_has_no_perf_gate():
    """The gate runs in the version branch's merge queue, never at tagging."""
    jobs = workflow("release-tag.yml")["jobs"]
    assert "perf" not in jobs
    for name, job in jobs.items():
        assert "perf" not in job.get("needs", []), name
        assert "needs.perf." not in str(job.get("if", "")), name
    assert not any("perf-nightly" in str(job.get("uses", "")) for job in jobs.values())


def test_the_perf_gate_runs_the_nightly_flow_on_the_queues_image():
    jobs = workflow("rust-pr.yml")["jobs"]
    assert jobs["perf"]["with"]["mode"] == "nightly"
    assert "checks" not in jobs["perf"]["with"]  # every registered check
    for name in ("perf", "perf-cov"):
        assert jobs[name]["needs"] == ["check-files", "ci-pipeline"]
        assert "needs.check-files.outputs.perf_below != ''" in jobs[name]["if"]
        assert jobs[name]["with"]["candidate_image"] == (
            "ghcr.io/falkordb/falkordb-server:rc-${{ needs.check-files.outputs.ci_key }}-valkey"
        )
        assert jobs[name]["with"]["baseline_below"] == "${{ needs.check-files.outputs.perf_below }}"
    # Nothing narrows the checks inside the called workflow either.
    assert "GATE_CHECKS" not in (WORKFLOWS / "perf-nightly.yml").read_text()


GITHUB_EXPRESSION_TOKEN = re.compile(
    r"'(?:[^']|'')*'|&&|\|\||==|!=|!|\(|\)|[A-Za-z_][\w.-]*\(\)|[A-Za-z_][\w.-]*"
)


def github_expression(expr, context):
    """A GitHub Actions expression of the shapes these workflows use (&&, ||, ==, !=, !,
    string literals, context paths, cancelled()), evaluated against `context` (nested
    dicts; a missing path is ''). Python's `and`/`or` return an operand, as `&&`/`||` do."""
    expr = expr.strip()
    if expr.startswith("${{") and expr.endswith("}}"):
        expr = expr[3:-2]
    assert not GITHUB_EXPRESSION_TOKEN.sub("", expr).strip(), f"unsupported syntax: {expr}"
    python = []
    for token in GITHUB_EXPRESSION_TOKEN.findall(expr):
        if token.startswith("'"):
            python.append(repr(token[1:-1].replace("''", "'")))
        elif token in ("==", "!=", "(", ")"):
            python.append(token)
        else:
            python.append(
                {"&&": " and ", "||": " or ", "!": " not ", "cancelled()": "False"}.get(
                    token, f"_get({token!r})"
                )
            )

    def _get(path):
        value = context
        for part in path.split("."):
            value = value.get(part, "") if isinstance(value, dict) else ""
        return value

    return eval("".join(python), {"_get": _get, "__builtins__": {}})


def in_env(value):
    """An expression's value as `env` receives it: a boolean is 'true' or 'false'."""
    return ("true" if value else "false") if isinstance(value, bool) else value


@pytest.mark.parametrize(
    "perf_below, same_repo, gate_runs",
    [
        ("6.3.0", "true", True),  # a version branch's queue, a pull request from this repository
        ("6.3.0", "false", False),  # ... from a fork: no gate, and ci-gate does not require it
        ("", "true", False),  # a pull request, or main's queue
    ],
)
def test_the_release_gate_runs_only_for_a_pull_request_from_this_repository(
    tmp_path, perf_below, same_repo, gate_runs
):
    import shutil

    if shutil.which("jq") is None:
        pytest.skip("jq is not installed")
    jobs = workflow("rust-pr.yml")["jobs"]
    assert jobs["check-files"]["outputs"]["same_repo"] == "${{ steps.fork.outputs.same_repo }}"
    outputs = {"perf_below": perf_below, "same_repo": same_repo}
    context = {"needs": {"check-files": {"outputs": outputs}, "ci-pipeline": {"result": "success"}}}
    for name in ("perf", "perf-cov"):
        assert bool(github_expression(jobs[name]["if"], context)) is gate_runs, name
    gate = jobs["ci-gate"]
    step = gate["steps"][0]
    required = in_env(github_expression(step["env"]["PERF_REQUIRED"], context))
    assert required == ("true" if gate_runs else "false")
    (tmp_path / "bin").mkdir()

    def ci_gate(perf_result):
        results = {name: {"result": "success", "outputs": {}} for name in gate["needs"]}
        results["perf"]["result"] = results["perf-cov"]["result"] = perf_result
        env = {
            "BIN": str(tmp_path / "bin"),
            "RESULTS": json.dumps(results),
            "EVENT": "merge_group",
            "PERF_REQUIRED": required,
        }
        return run_step(step["run"], tmp_path, env).returncode == 0

    if gate_runs:
        assert ci_gate("success") and not ci_gate("skipped")
    else:
        assert ci_gate("skipped")  # a fork's queued pull request merges without the gate


def _fork_step(tmp_path, cwd, event, head_repos, perf_below="", head_repo=""):
    """check-files' fork step, run in `cwd` with a fake gh that knows the head repository
    of each pull request in `head_repos` (any other lookup fails); its same_repo and stdout."""
    steps = workflow("rust-pr.yml")["jobs"]["check-files"]["steps"]
    step = next(s for s in steps if s.get("id") == "fork")
    changed = tmp_path / "changed-files.txt"
    changed.write_text("graph/src/lib.rs\n")  # no infrastructure path
    script = step["run"].replace("/tmp/changed-files.txt", str(changed))
    if subprocess.run(["sed", "--version"], capture_output=True, check=False).returncode:
        script = script.replace("\\+", "\\{1,\\}")  # BSD sed (macOS): the POSIX spelling of \+
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    known = "".join(
        f"  /repos/FalkorDB/FalkorDB/pulls/{n}) echo {repo} ;;\n" for n, repo in head_repos.items()
    )
    fake_tool(bin_dir, "gh", f'case "$2" in\n{known}  *) echo "HTTP 404" >&2; exit 1 ;;\nesac\n')
    out = tmp_path / "output"
    env = {
        "BIN": str(bin_dir),
        "GITHUB_OUTPUT": str(out),
        "GITHUB_REPOSITORY": "FalkorDB/FalkorDB",
        "EVENT": event,
        "PR_HEAD_REPO": head_repo,
        "REF": "refs/heads/gh-readonly-queue/6.2/pr-7-" + "a" * 40,
        "QUEUE_BASE_REF": "refs/heads/6.2",
        "PATTERN": "^build/",
        "PERF_BELOW": perf_below,
    }
    run = run_step(script, cwd, env)
    assert run.returncode == 0, run.stdout + run.stderr
    outputs = dict(line.split("=", 1) for line in out.read_text().splitlines())
    return outputs["same_repo"], run.stdout


@pytest.mark.parametrize(
    "event, head_repo, same_repo",
    [
        ("merge_group", "FalkorDB/FalkorDB", "true"),  # main's queue: no gate, the PR alone
        ("merge_group", "someone/FalkorDB", "false"),
        ("pull_request", "FalkorDB/FalkorDB", "true"),
        ("pull_request", "someone/FalkorDB", "false"),
    ],
)
def test_check_files_says_whether_the_pull_request_is_from_this_repository(
    tmp_path, event, head_repo, same_repo
):
    queued = {7: head_repo} if event == "merge_group" else {}
    pull_head = head_repo if event == "pull_request" else ""
    got, stdout = _fork_step(tmp_path, tmp_path, event, queued, head_repo=pull_head)
    assert got == same_repo
    assert "::warning::" not in stdout  # no gate where perf_below is empty, so none skipped


def _git(work, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c",
         "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args],
        cwd=work, check=True, capture_output=True, text=True,
    ).stdout.strip()  # fmt: skip


def _queue(tmp_path, subjects):
    """A checkout of a merge group in version branch 6.2's queue: origin has 6.2, and HEAD
    is 6.2 plus one squash commit per subject, the entries queued ahead first."""
    work, origin = tmp_path / "work", tmp_path / "origin.git"
    work.mkdir()
    _git(work, "init", "-q", "-b", "6.2")
    _git(work, "commit", "-q", "--allow-empty", "-m", "Bump version to 6.2.1")
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    _git(work, "remote", "add", "origin", str(origin))
    _git(work, "push", "-q", "origin", "6.2")
    _git(work, "checkout", "-q", "--detach")
    for subject in subjects:
        _git(work, "commit", "-q", "--allow-empty", "-m", subject)
    return work


def _queue_merge(tmp_path, subject, own):
    """_queue with one merge-commit entry (how 6.2's queue merges): `own` are the merged
    branch's own commits, `subject` the merge commit's."""
    work = _queue(tmp_path, [])
    base = _git(work, "rev-parse", "HEAD")
    for message in own:
        _git(work, "commit", "-q", "--allow-empty", "-m", message)
    branch = _git(work, "rev-parse", "HEAD")
    _git(work, "checkout", "-q", "--detach", base)
    _git(work, "merge", "-q", "--no-ff", "-m", subject, branch)
    return work


GATE_SKIPPED = "::warning::The release gate (perf, perf-cov) does not run"


@pytest.mark.parametrize(
    "subjects, head_repos, same_repo, named",
    [
        # a fork's pull request queued ahead rides in this group: no gate
        (["fix: a (#5)", "fix: b (#7)"], {5: "someone/FalkorDB", 7: "FalkorDB/FalkorDB"}, "false", "#5"),
        # every pull request from this repository: the gate runs
        (["fix: a (#6)", "fix: b (#7)"], {6: "FalkorDB/FalkorDB", 7: "FalkorDB/FalkorDB"}, "true", None),
        # #5 cannot be looked up: counted as a fork, never run on a guess
        (["fix: a (#5)", "fix: b (#7)"], {7: "FalkorDB/FalkorDB"}, "false", "#5"),
        # a commit that names no pull request
        (["wip", "fix: b (#7)"], {7: "FalkorDB/FalkorDB"}, "false", "a commit naming no pull request: wip"),
        # the queued fork itself
        (["fix: b (#7)"], {7: "someone/FalkorDB"}, "false", "#7"),
        # a fork's squashed title spoofs the merge form, naming this repository's #6
        (["Merge pull request #6 from FalkorDB/x (#5)", "fix: b (#7)"], {5: "someone/FalkorDB", 6: "FalkorDB/FalkorDB", 7: "FalkorDB/FalkorDB"}, "false", "#5"),
        # a fork's merged branch, named x(#6), spoofs the squash form
        (["Merge pull request #5 from someone/x(#6)", "fix: b (#7)"], {5: "someone/FalkorDB", 6: "FalkorDB/FalkorDB", 7: "FalkorDB/FalkorDB"}, "false", "#5"),
    ],
)  # fmt: skip
def test_the_release_gate_skips_a_merge_group_holding_any_fork(
    tmp_path, subjects, head_repos, same_repo, named
):
    work = _queue(tmp_path, subjects)
    got, stdout = _fork_step(tmp_path, work, "merge_group", head_repos, perf_below="6.3.0")
    assert got == same_repo
    if named is None:
        assert GATE_SKIPPED not in stdout
    else:
        assert GATE_SKIPPED in stdout and named in stdout


def test_the_release_gate_reads_only_the_groups_own_merges(tmp_path):
    """6.2's queue merges: a merged branch's own commits are cherry-picks that name other
    pull requests (here a fork's, #99, merged into main long ago); only first parents count."""
    work = _queue_merge(
        tmp_path, "Merge pull request #7 from FalkorDB/backport/6.2", ["fix: x (#99)", "Bump"]
    )
    head_repos = {7: "FalkorDB/FalkorDB", 99: "someone/FalkorDB"}
    got, stdout = _fork_step(tmp_path, work, "merge_group", head_repos, perf_below="6.3.0")
    assert (got, GATE_SKIPPED in stdout) == ("true", False)


def test_the_release_gate_skips_when_the_groups_commits_cannot_be_listed(tmp_path):
    work = _queue(tmp_path, ["fix: b (#7)"])
    _git(work, "remote", "set-url", "origin", str(tmp_path / "gone.git"))
    got, stdout = _fork_step(
        tmp_path, work, "merge_group", {7: "FalkorDB/FalkorDB"}, perf_below="6.3.0"
    )
    assert got == "false" and "the group's commits could not be listed" in stdout


def test_every_perf_result_and_gate_upload_replaces_an_earlier_attempts_artifact():
    """As in perf-nightly.yml: a re-run uploads each name again, which upload-artifact
    refuses without `overwrite`, and the merge would keep the first attempt's result."""
    for name in ("_perf-vm.yml", "benchmark-cov.yml"):
        uploads = [
            step
            for spec in workflow(name)["jobs"].values()
            for step in spec.get("steps", [])
            if step.get("uses", "").startswith("actions/upload-artifact@")
        ]
        assert uploads, name
        for step in uploads:
            assert step["with"].get("overwrite") is True, (name, step["with"]["name"])


@pytest.mark.parametrize(
    "event, base_sha, base_ref",
    [
        (
            {"merge_group": {"base_sha": "b" * 40, "base_ref": "refs/heads/6.2"}},
            "b" * 40,
            "refs/heads/6.2",
        ),
        ({"pull_request": {"base": {"sha": "c" * 40, "ref": "6.2"}}}, "c" * 40, "6.2"),
    ],
)
def test_benchmark_cov_reads_the_base_of_a_merge_group_too(event, base_sha, base_ref):
    """The gate's caller runs only in the merge queue, where there is no pull_request."""
    steps = workflow("benchmark-cov.yml")["jobs"]["preflight"]["steps"]
    env = next(step for step in steps if step.get("id") == "gate")["env"]
    context = {"github": {"event": event}}
    assert github_expression(env["BASE_SHA"], context) == base_sha
    assert github_expression(env["BASE_REF"], context) == base_ref


# --- benchmark-cov as the release gate (#4004) -----------------------------------

from falkorbench import cov_gate  # noqa: E402
from falkorbench.metrics import Row  # noqa: E402


def rows(column, **values):
    return {query: Row({column: value}) for query, value in values.items()}


def cov_sources(instr_pr, instr_base, alloc_pr=None, alloc_base=None):
    alloc_pr = alloc_pr if alloc_pr is not None else {"RETURN 1": 1000.0}
    alloc_base = alloc_base if alloc_base is not None else {"RETURN 1": 1000.0}
    return {
        "callgrind": {"pr": rows("instr", **instr_pr), "base": rows("instr", **instr_base)},
        "measure": {
            "pr": rows("alloc_bytes", **alloc_pr),
            "base": rows("alloc_bytes", **alloc_base),
        },
    }


def by_name(findings):
    return {(f["name"], f["metric"]): f["status"] for f in findings}


def test_cov_gate_flags_3996_at_1_30x():
    """#3996's row as callgrind counted it in run 45332918572 (main@4c1088949,
    which 6.2.1 matches, against 6.2.0): a regression with no floor."""
    xs = {"cross product filter": 313_644.0}
    findings, errors = cov_gate.judge(cov_sources(xs, {"cross product filter": 240_361.0}))
    assert errors == []
    assert by_name(findings) == {("cross product filter [reads]", "instructions"): "regression"}
    assert round(findings[0]["ratio"], 3) == 1.305


@pytest.mark.parametrize(
    "pr, base, status",
    [
        (1100.0, 1000.0, None),  # exactly 1.10x: the bound is strict
        (1111.0, 1000.0, "regression"),  # 1.111x on a tiny row: no floor on instructions
        (910.0, 1000.0, None),  # 0.91x: base over PR is 1.099x, within the bound
        (900.0, 1000.0, "improvement"),  # base over PR is 1.11x
    ],
)
def test_cov_gate_instructions_cross_at_10_percent_with_no_floor(pr, base, status):
    findings, _ = cov_gate.judge(cov_sources({"arithmetic": pr}, {"arithmetic": base}))
    assert by_name(findings).get(("arithmetic [reads]", "instructions")) == status


@pytest.mark.parametrize(
    "pr, base, status",
    [
        (170_000.0, 100_000.0, "regression"),  # 1.7x and +70,000 B (> 64 KiB)
        (118_000.0, 100_000.0, None),  # 1.18x but +18,000 B: under the floor
        (9_500.0, 1_000.0, "regression"),  # 9.5x and +8,500 B: the steep tier
        (3_800.0, 400.0, None),  # 9.5x but +3,400 B: under the steep floor
        (40_000.0, 400_000.0, "improvement"),
    ],
)
def test_cov_gate_allocations_use_micro_hws_rule(pr, base, status):
    sources = cov_sources(
        {"RETURN 1": 1.0}, {"RETURN 1": 1.0}, {"SET property": pr}, {"SET property": base}
    )
    findings, _ = cov_gate.judge(sources)
    assert by_name(findings).get(("SET property [writes]", "alloc_bytes")) == status


def test_cov_gate_says_error_when_a_side_or_a_row_is_missing():
    _, errors = cov_gate.judge(cov_sources({}, {"RETURN 1": 1.0}))
    assert errors == ["no callgrind rows for pr"]
    _, errors = cov_gate.judge(
        cov_sources({"RETURN 1": 1.0}, {"RETURN 1": 1.0, "arithmetic": 5.0, "reduce": 5.0})
    )
    assert errors == ["2 callgrind row(s) measured on base, not on pr: arithmetic, reduce"]


def write_csv(path, column, values):
    path.write_text("query," + column + "\n" + "".join(f"{q},{v}\n" for q, v in values.items()))


def test_cov_gate_end_to_end_is_refused_then_waived(tmp_path, capsys):
    """The workflow's two steps on CSVs in benchmark-cov's format: cov-gate writes a
    result the contract accepts, and conclude refuses it until a waiver names it."""
    art = tmp_path / "art"
    art.mkdir()
    write_csv(art / "cg-pr-1.csv", "instr", {"cross product filter": 313644, "RETURN 1": 63797})
    write_csv(art / "cg-base-1.csv", "instr", {"cross product filter": 240361, "RETURN 1": 63797})
    write_csv(art / "full-pr.csv", "alloc_bytes", {"RETURN 1": 1258})
    write_csv(art / "full-base.csv", "alloc_bytes", {"RETURN 1": 1258})
    argv = ["--mode", "queue", "--out", str(tmp_path / "out")]
    argv += ["--candidate", CAND, "--candidate-label", "rc-pr-3995-valkey"]
    argv += ["--baseline", BASE, "--baseline-label", "6.2@d8c329b"]
    argv += ["--measure-pr", str(art / "full-pr.csv"), "--measure-base", str(art / "full-base.csv")]
    argv += [
        "--callgrind-pr",
        str(art / "cg-pr-*.csv"),
        "--callgrind-base",
        str(art / "cg-base-*.csv"),
    ]
    assert cov_gate.main(argv) == 0
    path = tmp_path / "out" / "result.json"
    doc = json.loads(path.read_text())
    assert contract.validate_result(doc) == []
    assert (doc["check"], doc["verdict"], doc["machine"]["runner"]) == (
        "benchmark-cov",
        "regression",
        "hosted",
    )
    none = write(tmp_path, "none.json", None)
    assert merge.main(["conclude", "--gate", "--result", str(path), "--waivers", none]) == 1
    waived = write(
        tmp_path,
        "w.json",
        [waiver(check="benchmark-cov", pattern=r"cross product filter \[reads\]")],
    )
    assert merge.main(["conclude", "--gate", "--result", str(path), "--waivers", waived]) == 0


def test_benchmark_cov_gate_caller_passes_only_declared_inputs():
    job = workflow("rust-pr.yml")["jobs"]["perf-cov"]
    assert job["uses"] == "./.github/workflows/benchmark-cov.yml"
    declared = workflow("benchmark-cov.yml")["on"]["workflow_call"]["inputs"]
    assert set(job["with"]) <= set(declared)
    assert {key for key, spec in declared.items() if spec.get("required")} <= set(job["with"])
    assert job["permissions"] == {
        "contents": "read",
        "pull-requests": "write",
        "packages": "write",
    }
    assert job["if"] == workflow("rust-pr.yml")["jobs"]["perf"]["if"]


def test_benchmark_cov_runs_its_own_commit_and_keeps_the_label_run():
    """A call measures with main's bench code; the label's run is what it was:
    main-only, three sides, coverage, no gate step."""
    jobs = workflow("benchmark-cov.yml")["jobs"]
    checkouts = [
        step
        for job in jobs.values()
        for step in job.get("steps", [])
        if step.get("uses", "").startswith("actions/checkout@")
    ]
    assert len(checkouts) == 5
    for step in checkouts:
        assert step["with"]["repository"] == "${{ job.workflow_repository }}"
        assert step["with"]["ref"] == "${{ job.workflow_sha }}"
    label = "github.event.label.name == 'benchmark-cov'"
    main_only = "contains(fromJSON('[\"main\"]'), github.event.pull_request.base.ref)"
    assert label in jobs["preflight"]["if"] and main_only in jobs["preflight"]["if"]
    side = jobs["measure"]["strategy"]["matrix"]["side"]
    assert side == (
        "${{ fromJSON(inputs.candidate_image != '' && "
        '\'["pr","base"]\' || \'["pr","base","c"]\') }}'
    )
    assert jobs["coverage"]["if"] == "inputs.candidate_image == ''"
    gate_steps = [
        s for s in jobs["report"]["steps"] if "inputs.candidate_image != ''" in str(s.get("if"))
    ]
    assert [s.get("name") for s in gate_steps] == [
        "Judge the reading (release gate)",
        None,
        "Fail the gate unless every regression is waived",
    ]
    assert "--gate --result out/result.json --waivers waivers.json" in gate_steps[-1]["run"]


def test_a_benchmark_cov_call_from_the_queue_has_no_pull_request_to_read():
    """Under merge_group the PR number is empty: nothing a call runs may need it."""
    jobs = workflow("benchmark-cov.yml")["jobs"]
    pre = {s.get("name"): s for s in jobs["preflight"]["steps"]}
    assert pre["Acknowledge the request"]["if"] == "inputs.candidate_image == ''"
    assert (
        pre["Wait for the PR image to be built from this commit"]["if"]
        == "inputs.candidate_image == ''"
    )
    images = pre["Resolve the three images"]["env"]
    assert images["PR_IMAGE"] == "${{ steps.gate.outputs.pr_image || env.PR_IMAGE }}"
    assert images["HEAD_SHA"].startswith("${{ inputs.candidate_image == '' &&")
    post = next(
        s for s in jobs["report"]["steps"] if s.get("name") == "Post or update the PR comment"
    )
    assert post["if"] == "github.event.pull_request.number != ''"


def test_a_benchmark_cov_call_has_a_concurrency_group_of_its_own_run():
    """A label added to a version-branch PR starts that branch's own copy of
    benchmark-cov.yml in group benchmark-cov-<N>, which would cancel the gate."""
    group = workflow("benchmark-cov.yml")["concurrency"]["group"]
    assert group == (
        "${{ inputs.candidate_image && format('benchmark-cov-gate-{0}', github.run_id)"
        " || format('benchmark-cov-{0}', github.event.pull_request.number) }}"
    )


def test_a_partial_reading_is_never_waived(tmp_path):
    """The contract drops `error` under a regression, so a PR that lost a row
    (a callgrind query that timed out) beside a waived one would pass: cov-gate's
    unfinished.json refuses it."""
    art = tmp_path / "art"
    art.mkdir()
    write_csv(art / "cg-pr-1.csv", "instr", {"cross product filter": 313644, "RETURN 1": 63797})
    base = {"cross product filter": 240361, "RETURN 1": 63797, "arithmetic": 900000}
    write_csv(art / "cg-base-1.csv", "instr", base)
    write_csv(art / "full-pr.csv", "alloc_bytes", {"RETURN 1": 1258})
    write_csv(art / "full-base.csv", "alloc_bytes", {"RETURN 1": 1258})
    argv = ["--mode", "queue", "--out", str(tmp_path / "out"), "--candidate", CAND]
    argv += ["--baseline", BASE, "--unfinished-out", str(tmp_path / "unfinished.json")]
    argv += ["--measure-pr", str(art / "full-pr.csv"), "--measure-base", str(art / "full-base.csv")]
    argv += ["--callgrind-pr", str(art / "cg-pr-*.csv")]
    argv += ["--callgrind-base", str(art / "cg-base-*.csv")]
    assert cov_gate.main(argv) == 0
    doc = json.loads((tmp_path / "out" / "result.json").read_text())
    assert (doc["verdict"], doc["error"]) == ("regression", None)
    unfinished = json.loads((tmp_path / "unfinished.json").read_text())
    assert unfinished == {
        "benchmark-cov": "1 callgrind row(s) measured on base, not on pr: arithmetic"
    }
    waived = write(tmp_path, "w.json", [waiver(check="benchmark-cov", pattern="cross .*")])
    argv = ["conclude", "--gate", "--result", str(tmp_path / "out" / "result.json")]
    argv += ["--waivers", waived]
    assert doc["incomplete"]  # the result says it itself, so a waiver cannot cover it
    assert merge.main(argv) == 1
    assert merge.main([*argv, "--unfinished", str(tmp_path / "unfinished.json")]) == 1


def test_more_regressions_than_a_result_lists_are_never_waived():
    many = [kit_finding(f"q{i}") for i in range(contract.MAX_FINDINGS + 1)]
    assert cov_gate.incomplete(many, []) == "51 regressions, more than the 50 result.json lists"
    assert cov_gate.incomplete(many[:50], []) == ""


def kit_finding(name):
    return {**finding(name), "status": "regression"}


def test_one_result_is_always_validated(tmp_path, capsys):
    """`--result` takes its status from a verdict nothing else checked: a file
    with no verdict must not read as green."""
    doc = result("benchmark-cov", "pass")
    del doc["verdict"]
    path = write(tmp_path, "result.json", doc)
    assert merge.main(["conclude", "--gate", "--result", path]) == 1
    assert "the result is not valid" in capsys.readouterr().out


BENCH_COV_RELEASES = [
    {"tag_name": t, "draft": False, "prerelease": False} for t in ("v6.2.0", "v6.2.1", "v6.0.4")
]


@pytest.mark.parametrize(
    "baseline, base_sha, inspect, code, base_label, base_ref",
    [
        ("", "b" * 40, "found", 0, "6.2@bbbbbbb", "6.2"),  # the base tip's queue image
        ("", "b" * 40, "found", 0, "6.2@bbbbbbb", "refs/heads/6.2"),  # the merge group's spelling
        ("", "b" * 40, "missing", 0, "6.2.1", "6.2"),  # swept or never queued: the newest release
        ("", "", "missing", 0, "6.2.1", "6.2"),  # no base sha (a push caller): the release
        ("", "b" * 40, "broken", 1, None, "6.2"),  # a registry error is not "no image"
        (
            "falkordb/falkordb-server:6.2.0@sha256:" + "c" * 64,
            "b" * 40,
            "found",
            0,
            "6.2.0",
            "6.2",
        ),
    ],
)
def test_benchmark_cov_pins_the_gates_base(
    tmp_path, baseline, base_sha, inspect, code, base_label, base_ref
):
    """benchmark-cov's "Pin the gate's images" step, with fake docker, gh and uv."""
    import shutil

    if shutil.which("jq") is None:
        pytest.skip("jq is not installed")
    steps = workflow("benchmark-cov.yml")["jobs"]["preflight"]["steps"]
    script = next(step for step in steps if step.get("id") == "gate")["run"]
    bin_dir, temp = tmp_path / "bin", tmp_path / "temp"
    bin_dir.mkdir()
    temp.mkdir()
    fake_uv(bin_dir)
    rc_mg = {
        "found": f'echo \'{{"digest": "sha256:{"e" * 64}"}}\'',
        "missing": 'echo "ERROR: $4: not found" >&2; exit 1',
        "broken": 'echo "ERROR: 500 Internal Server Error" >&2; exit 1',
    }[inspect]
    fake_tool(
        bin_dir,
        "docker",
        f"""case "$4" in
  *rc-mg-*) {rc_mg} ;;
  *) echo \'{{"digest": "sha256:{"a" * 64}"}}\' ;;
esac
""",
    )
    (temp / "releases.json").write_text(json.dumps(BENCH_COV_RELEASES))
    fake_tool(
        bin_dir,
        "gh",
        f"""jqexpr=""
while [[ $# -gt 0 ]]; do [[ $1 == --jq ]] && jqexpr=$2; shift; done
jq -r "$jqexpr" {temp / "releases.json"}
""",
    )
    out = temp / "output"
    run = run_step(
        script,
        tmp_path,
        {
            "BIN": str(bin_dir),
            "RUNNER_TEMP": str(temp),
            "GITHUB_OUTPUT": str(out),
            "GITHUB_REPOSITORY": "o/r",
            "CANDIDATE": "ghcr.io/falkordb/falkordb-server:rc-pr-7-valkey",
            "BASELINE": baseline,
            "BASELINE_BELOW": "6.3.0",
            "BASE_SHA": base_sha,
            "BASE_REF": base_ref,
        },
    )
    assert run.returncode == code, run.stdout + run.stderr
    if code:
        return
    outputs = dict(line.split("=", 1) for line in out.read_text().splitlines())
    assert outputs["pr_image"] == "ghcr.io/falkordb/falkordb-server@sha256:" + "a" * 64
    assert outputs["pr_label"] == "rc-pr-7-valkey"
    assert outputs["base_label"] == base_label
    want = {
        "6.2@bbbbbbb": "ghcr.io/falkordb/falkordb-server@sha256:" + "e" * 64,
        "6.2.1": "docker.io/falkordb/falkordb-server@sha256:" + "a" * 64,
        "6.2.0": "falkordb/falkordb-server@sha256:" + "a" * 64,
    }[base_label]
    assert outputs["base_image"] == want


# --- benchmark-cov: queries a side does not answer (Pokec's rule) ------------------


def failed_sources():
    """`geo` is new on main and neither 6.2 side answers it; `MATCH all` broke on
    the PR; `reduce` failed on the base only."""
    instr = {"RETURN 1": 60000.0, "reduce": 5000.0}
    measure_pr = {"RETURN 1": 1000.0, "reduce": 1000.0}
    measure_base = {"RETURN 1": 1000.0, "MATCH all": 1000.0}
    sources = cov_sources(
        instr, {"RETURN 1": 60000.0, "MATCH all": 7000.0}, measure_pr, measure_base
    )
    failed = {
        "pr": {"geo": "unknown function", "MATCH all": "crashed"},
        "base": {"geo": "unknown function", "reduce": "timed out"},
    }
    return sources, failed


def test_a_query_no_side_answers_is_info_and_one_the_pr_broke_is_a_regression():
    sources, failed = failed_sources()
    findings, errors = cov_gate.judge(sources, failed)
    got = {(f["name"], f["metric"]): (f["candidate"], f["baseline"], f["status"]) for f in findings}
    assert got == {
        ("geo", "completed"): (0, 0, "info"),  # not supported on this branch
        ("MATCH all", "completed"): (0, 1, "regression"),  # answers on the base, not on the PR
        ("reduce [reads]", "completed"): (1, None, "info"),  # not measured on the base
    }
    # The rows those three explain are not "unmeasured" errors in either source.
    assert errors == []


def test_without_the_failed_records_the_same_rows_are_unmeasured():
    sources, _ = failed_sources()
    _, errors = cov_gate.judge(sources)
    assert errors == [
        "1 callgrind row(s) measured on base, not on pr: MATCH all",
        "1 measure row(s) measured on base, not on pr: MATCH all",
    ]


def test_a_query_failed_on_one_side_and_unmeasured_on_the_other_stays_unmeasured():
    findings, explained = cov_gate.answered({"pr": {}, "base": {}}, {"pr": {"x": "e"}})
    assert (findings, explained) == ([], {})


def test_cov_gate_end_to_end_with_failed_queries(tmp_path):
    art = tmp_path / "art"
    art.mkdir()
    write_csv(art / "cg-pr-1.csv", "instr", {"RETURN 1": 63797})
    write_csv(art / "cg-base-1.csv", "instr", {"RETURN 1": 63797})
    write_csv(art / "full-pr.csv", "alloc_bytes", {"RETURN 1": 1258})
    write_csv(art / "full-base.csv", "alloc_bytes", {"RETURN 1": 1258, "MATCH all": 900})
    (art / "failed-pr.json").write_text(json.dumps({"geo": "e", "MATCH all": "crashed"}))
    (art / "failed-base.json").write_text(json.dumps({"geo": "e"}))
    argv = ["--mode", "queue", "--out", str(tmp_path / "out"), "--candidate", CAND]
    argv += ["--baseline", BASE, "--unfinished-out", str(tmp_path / "unfinished.json")]
    argv += ["--measure-pr", str(art / "full-pr.csv"), "--measure-base", str(art / "full-base.csv")]
    argv += ["--callgrind-pr", str(art / "cg-pr-*.csv")]
    argv += ["--callgrind-base", str(art / "cg-base-*.csv")]
    argv += [
        "--failed-pr",
        str(art / "failed-pr.json"),
        "--failed-base",
        str(art / "failed-base.json"),
    ]
    assert cov_gate.main(argv) == 0
    doc = json.loads((tmp_path / "out" / "result.json").read_text())
    assert contract.validate_result(doc) == []
    assert doc["verdict"] == "regression"
    assert doc["summary"].startswith("1 regression(s), worst MATCH all completed;")
    assert json.loads((tmp_path / "unfinished.json").read_text()) == {}
    # Only `geo` failing on both sides: a pass, with the info row.
    (art / "failed-pr.json").write_text(json.dumps({"geo": "e"}))
    write_csv(art / "full-base.csv", "alloc_bytes", {"RETURN 1": 1258})
    assert cov_gate.main(argv) == 0
    doc = json.loads((tmp_path / "out" / "result.json").read_text())
    assert (doc["verdict"], [f["status"] for f in doc["findings"]]) == ("pass", ["info"])


def test_bench_measure_records_the_queries_that_did_not_answer(tmp_path):
    from falkorbench import measure as measure_mod

    path = tmp_path / "failed.json"
    measure_mod.write_failures(path, [("geo", "unknown function"), ("MATCH all", "crashed")])
    assert json.loads(path.read_text()) == {"MATCH all": "crashed", "geo": "unknown function"}


def test_only_a_call_records_failures_instead_of_failing():
    """The label's run still fails on a query that does not answer."""
    measure = workflow("benchmark-cov.yml")["jobs"]["measure"]
    step = next(s for s in measure["steps"] if s.get("name") == "Measure")
    assert step["env"]["GATE"] == "${{ inputs.candidate_image != '' }}"
    assert (
        'if [ "$GATE" = true ]; then flags+=(--failed-out "/w/failed-$SIDE.json"); fi'
        in step["run"]
    )
    judge_step = next(
        s
        for s in workflow("benchmark-cov.yml")["jobs"]["report"]["steps"]
        if s.get("name") == "Judge the reading (release gate)"
    )
    assert "--failed-pr art/failed-pr.json --failed-base art/failed-base.json" in judge_step["run"]


def test_cov_gate_compares_a_pair_both_sides_read_even_when_the_query_is_explained():
    sources = cov_sources(
        {"arithmetic": 2000.0}, {"arithmetic": 1000.0}, {"arithmetic": 1000.0}, {"RETURN 1": 1000.0}
    )
    findings, _ = cov_gate.judge(sources, {"base": {"arithmetic": "timed out"}})
    assert by_name(findings)[("arithmetic [reads]", "instructions")] == "regression"


def test_cov_gate_errors_when_a_baseline_counter_is_missing():
    sources = cov_sources({}, {}, {"RETURN 1": 1000.0}, {"RETURN 1": None})
    _, errors = cov_gate.judge(sources)
    assert any("no baseline alloc_bytes: RETURN 1" in e for e in errors)


def test_cov_gate_does_not_call_a_failed_measured_run_unsupported():
    sources = cov_sources({}, {})
    failed = {
        "pr": {"reduce": "measured run: timeout"},
        "base": {"reduce": "measured run: timeout"},
    }
    findings, errors = cov_gate.judge(sources, failed)
    assert findings == []
    assert any("failed their measured run: reduce" in e for e in errors)


@pytest.mark.parametrize(
    "ratio, waived",
    [(0.7, True), (0.5, False), (0.1, False), (0.0, False), (0.67, True)],
)
def test_a_max_ratio_bounds_a_lower_is_worse_row_in_the_worse_direction(ratio, waived):
    row = {**finding("cross product filter [reads]", metric="throughput"), "ratio": ratio}
    row["direction"] = "lower_is_worse"
    night = red_night({"check": "micro-hw", "verdict": "regression", "findings": [row]})
    assert decide(night, {**waiver(), "max_ratio": 1.5})[0] == (0 if waived else 1)


def test_a_failure_on_one_side_does_not_excuse_a_missing_reading_on_the_other():
    # `x` failed on the base only; the PR's callgrind run dropped it. The base
    # has an instruction row, so the PR's missing one must not pass quietly.
    sources = cov_sources({"y": 1.0}, {"x": 5000.0, "y": 1.0}, {"x": 1.0}, {})
    _, errors = cov_gate.judge(sources, {"base": {"x": "timed out"}})
    assert any("measured on base, not on pr" in e and "x" in e for e in errors)
