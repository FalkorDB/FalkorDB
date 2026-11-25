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
    """A dispatch from another branch (or a pull_request run, whose merge commit
    may sit many commits above main) still measures main's newest image, and
    falls back along main's first-parent history when the tip has none."""
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
