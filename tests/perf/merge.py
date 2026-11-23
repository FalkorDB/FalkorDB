#!/usr/bin/env python3
"""The bookkeeping steps of the nightly perf run (.github/workflows/perf-nightly.yml).

Three subcommands, one per job that needs them:

  plan     tests/perf/checks.yaml (as JSON) + the `checks` input
           -> plan.json, and the job matrices on $GITHUB_OUTPUT
  context  the resolved candidate and baseline + the earlier nightlies
           -> context.json: skip_reason and previous_green_commit
  merge    context.json + plan.json + every perf-result-<check> artifact
           -> nightly.json

The shapes are perf-judge's: tests/perf/schema/ and tests/perf/validate.py, which
`merge` runs on every result.json it embeds and on the nightly.json it writes.

What `merge` guarantees, whatever the check jobs did:

  * every planned check appears exactly once in nightly.json, micro included;
  * a check that left no result.json, an unreadable one, one the validator
    rejects, or one that names another check, appears as the stub
    {"check": name, "verdict": "error", "error": reason}, so a missing
    measurement can never make a night green;
  * the night's status is derived from the checks (red > error > green), or
    `skipped` with an empty checks list when `context` decided to skip.

Stdlib only: it runs under `python -I` on a bare ubuntu-latest runner.
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

SCHEMA = 1

# The micro tier is not in the registry; the workflow calls _perf-micro.yml for it.
MICRO = "micro"

# A check's name becomes a GCE label, a runner label and part of an artifact name
# (_perf-vm.yml), so it is held to the narrowest of the three.
CHECK_NAME = re.compile(r"^[a-z][a-z0-9-]{0,42}$")

RUNNERS = ("vm", "hosted")

DIGEST_REF = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")

NO_RESULT = "no result.json"

# A matrix with no entries fails the job before its `if:` is read, so an empty
# matrix is replaced by this one entry and the job is switched off by has_<runner>.
PLACEHOLDER = [{"check": "none"}]


class PlanError(Exception):
    pass


# --- plan ------------------------------------------------------------------------


def registry_entry(index, entry):
    """A checks.yaml entry as a plan row; a malformed one becomes a stub row.

    Raises PlanError only when the entry has no usable name, because then there is
    nothing to report the problem under.
    """
    if not isinstance(entry, dict):
        raise PlanError(f"checks.yaml entry {index} is not a mapping")
    name = entry.get("name")
    if not isinstance(name, str) or not CHECK_NAME.match(name):
        raise PlanError(
            f"checks.yaml entry {index}: name {name!r} does not match {CHECK_NAME.pattern}"
        )
    if name == MICRO:
        raise PlanError(f"checks.yaml entry {index}: '{MICRO}' is reserved for the micro tier")

    def invalid(why):
        return {"check": name, "runner": None, "error": f"invalid tests/perf/checks.yaml entry: {why}"}

    known = {"name", "runner", "command", "timeout_minutes", "machine_type"}
    unknown = sorted(set(entry) - known)
    if unknown:
        return invalid(f"unknown key(s) {', '.join(unknown)}")
    runner = entry.get("runner")
    if runner not in RUNNERS:
        return invalid(f"runner {runner!r} is not one of {', '.join(RUNNERS)}")
    command = entry.get("command")
    if not isinstance(command, str) or not command.strip():
        return invalid("command is missing or empty")
    timeout = entry.get("timeout_minutes")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 360:
        return invalid(f"timeout_minutes {timeout!r} is not an integer from 1 to 360")
    machine_type = entry.get("machine_type", "")
    if machine_type is None:
        machine_type = ""
    if not isinstance(machine_type, str):
        return invalid(f"machine_type {machine_type!r} is not a string")
    if machine_type and runner != "vm":
        return invalid("machine_type is only for runner: vm")
    return {
        "check": name,
        "runner": runner,
        "command": command.strip(),
        "timeout_minutes": timeout,
        "machine_type": machine_type,
    }


def build_plan(registry, select):
    """The checks tonight's run measures, in registry order (micro first).

    `select` is the workflow's `checks` input: a comma list, empty for all. A name
    it asks for that is not in the registry is planned as a stub, so a typo shows
    up in the report instead of silently measuring less.
    """
    if not isinstance(registry, list):
        raise PlanError("tests/perf/checks.yaml is not a list")
    rows = {MICRO: {"check": MICRO, "runner": "micro"}}
    for index, entry in enumerate(registry):
        row = registry_entry(index, entry)
        if row["check"] in rows:
            raise PlanError(f"checks.yaml entry {index}: duplicate name '{row['check']}'")
        rows[row["check"]] = row

    wanted = [name.strip() for name in (select or "").split(",") if name.strip()]
    if not wanted:
        return list(rows.values())
    plan = []
    for name in dict.fromkeys(wanted):
        if name in rows:
            plan.append(rows[name])
        elif CHECK_NAME.match(name):
            plan.append({"check": name, "runner": None, "error": "not in tests/perf/checks.yaml"})
        else:
            raise PlanError(f"checks input: {name!r} is not a check name")
    return plan


def matrix(plan, runner):
    rows = [
        {key: row[key] for key in ("check", "command", "timeout_minutes", "machine_type")}
        for row in plan
        if row["runner"] == runner
    ]
    return rows


def plan_outputs(plan):
    """The $GITHUB_OUTPUT lines the workflow's jobs switch on."""
    out = {"run_micro": json.dumps(any(row["runner"] == "micro" for row in plan))}
    for runner in RUNNERS:
        rows = matrix(plan, runner)
        out[f"has_{runner}"] = json.dumps(bool(rows))
        out[f"{runner}_matrix"] = json.dumps(rows or PLACEHOLDER, separators=(",", ":"))
    return out


# --- context ---------------------------------------------------------------------


def strict_load(path):
    """json.load that refuses NaN/Infinity, which json.load accepts by default."""

    def refuse(token):
        raise ValueError(f"{token} is not JSON")

    with open(path, encoding="utf-8") as handle:
        return json.load(handle, parse_constant=refuse)


def side(image, commit, label, what):
    if not DIGEST_REF.match(image or ""):
        raise PlanError(f"{what} image {image!r} is not pinned by a full sha256 digest")
    if commit and not COMMIT.match(commit):
        raise PlanError(f"{what} commit {commit!r} is not 40 lowercase hex")
    if not label or "\n" in label:
        raise PlanError(f"{what} label {label!r} is empty or not one line")
    return {"image": image, "commit": commit or None, "label": label}


def read_history(runs_file, history_dir):
    """Earlier nightlies, newest first: [(run_id, nightly or None)].

    `runs_file` lists the run ids of completed earlier nightlies, newest first; the
    workflow downloads each one's perf-nightly artifact into <history_dir>/<id>/
    and stops once it holds a green one. A run whose artifact is missing or
    unreadable is kept with None: the newest run leaving nothing is itself a fact
    the skip rule reads.
    """
    if not runs_file:
        return []
    history = []
    for line in Path(runs_file).read_text(encoding="utf-8").split():
        path = Path(history_dir) / line / "nightly.json"
        try:
            nightly = strict_load(path)
            if not isinstance(nightly, dict):
                nightly = None
        except (OSError, ValueError):
            nightly = None
        history.append((line, nightly))
    return history


def previous_green_commit(history):
    for _, nightly in history:
        if nightly and nightly.get("status") == "green":
            return (nightly.get("candidate") or {}).get("commit")
    return None


def skip_reason(history, candidate, baseline):
    """Why tonight need not run, or None.

    Skipped only when there is nothing new to measure AND the last answer stands:
    the newest earlier nightly left a nightly.json, was not `error` (a transient
    failure gets retried), and measured the same candidate commit against the same
    baseline digest. A skipped night counts too, since it carries the pair it
    skipped, so a quiet stretch keeps skipping.
    """
    if not history or candidate["commit"] is None:
        return None
    _, last = history[0]
    if last is None or last.get("status") == "error":
        return None
    if (last.get("candidate") or {}).get("commit") != candidate["commit"]:
        return None
    if (last.get("baseline") or {}).get("image") != baseline["image"]:
        return None
    # Name the night that actually measured this pair, not the last skip. Every
    # night between it and now skipped, so they all carry the same pair.
    measured = last
    for _, nightly in history:
        if nightly is None or nightly.get("status") != "skipped":
            if nightly is not None:
                measured = nightly
            break
    return one_line(
        f"main has not moved since the {measured.get('date')} nightly "
        f"({measured.get('status')}, {measured.get('run_url')}): "
        f"same candidate {candidate['label']} and baseline {baseline['label']}"
    )


# --- merge -----------------------------------------------------------------------


def load_validator(path):
    """perf-judge's validate.py, loaded by path (tests/perf is not a package)."""
    spec = importlib.util.spec_from_file_location("perf_validate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def vm_reason(artifact):
    """Why _perf-vm.yml produced no result.json, from its perf-vm.json, or ''."""
    try:
        vm = strict_load(artifact / "perf-vm.json")
    except (OSError, ValueError):
        return ""
    if not isinstance(vm, dict):
        return ""
    if vm.get("create_error"):
        return f"no VM came up: {vm['create_error']}"
    if vm.get("timed_out"):
        return "the check timed out on the VM"
    if vm.get("command_exit") not in (None, 0):
        return f"the check exited {vm['command_exit']} on the VM"
    return ""


def one_line(text):
    return " ".join(str(text).split())


def stub(check, reason):
    return {"check": check, "verdict": "error", "error": one_line(reason) or NO_RESULT}


def collect(check, artifact, validator):
    """The result.json in one artifact directory, or a stub saying why not."""
    path = artifact / "result.json"
    if not path.is_file():
        nested = sorted(str(p.relative_to(artifact)) for p in artifact.rglob("result.json"))
        if nested:
            return stub(check, f"{NO_RESULT} at the artifact root (found {', '.join(nested)})")
        why = vm_reason(artifact)
        return stub(check, f"{NO_RESULT}: {why}" if why else NO_RESULT)
    if validator is not None:
        _, problems = validator.validate_file(str(path), kind="result")
        if problems:
            more = f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""
            return stub(check, f"result.json failed validation: {problems[0]}{more}")
    try:
        result = strict_load(path)
    except (OSError, ValueError) as err:
        return stub(check, f"result.json is unreadable: {err}")
    if not isinstance(result, dict):
        return stub(check, "result.json is not an object")
    if result.get("check") != check:
        return stub(check, f"result.json names check {result.get('check')!r}")
    return result


def status_of(checks):
    verdicts = {entry.get("verdict") for entry in checks}
    if "regression" in verdicts:
        return "red"
    if "error" in verdicts or not checks:
        return "error"
    return "green"


def build_nightly(context, plan, results_dir, validator):
    nightly = {
        "schema": SCHEMA,
        "date": context["date"],
        "run_url": context["run_url"],
        "candidate": context["candidate"],
        "baseline": context["baseline"],
        "previous_green_commit": context["previous_green_commit"],
        "status": "skipped",
        "skip_reason": context["skip_reason"],
        "checks": [],
    }
    if context["skip_reason"]:
        return nightly

    results_dir = Path(results_dir)
    checks = []
    for row in plan:
        if row.get("error"):
            checks.append(stub(row["check"], row["error"]))
            continue
        artifact = results_dir / f"perf-result-{row['check']}"
        checks.append(collect(row["check"], artifact, validator))

    # An artifact nobody planned (should not happen) is still reported rather than
    # dropped, under its own name.
    planned = {row["check"] for row in plan}
    for artifact in sorted(results_dir.glob("perf-result-*")):
        name = artifact.name[len("perf-result-"):]
        if artifact.is_dir() and name not in planned and CHECK_NAME.match(name):
            checks.append(collect(name, artifact, validator))

    nightly["checks"] = checks
    nightly["status"] = status_of(checks)
    return nightly


def write_json(path, obj):
    text = json.dumps(obj, indent=2, allow_nan=False) + "\n"
    Path(path).write_text(text, encoding="utf-8")


def write_outputs(path, outputs):
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        for key, value in outputs.items():
            if "\n" in value:
                raise PlanError(f"output {key} spans lines")
            handle.write(f"{key}={value}\n")


# --- CLI -------------------------------------------------------------------------


def cmd_plan(args):
    registry = strict_load(args.registry)
    plan = build_plan(registry, args.select)
    write_json(args.out, plan)
    outputs = plan_outputs(plan)
    write_outputs(args.github_output, outputs)
    for row in plan:
        print(f"planned {row['check']}: {row.get('error') or row['runner']}")
    return 0


def cmd_context(args):
    candidate = side(args.candidate_image, args.candidate_commit, args.candidate_label, "candidate")
    baseline = side(args.baseline_image, args.baseline_commit, args.baseline_label, "baseline")
    history = read_history(args.history_runs, args.history_dir)
    for run_id, nightly in history:
        if nightly is None:
            print(f"history: run {run_id} left no readable perf-nightly artifact")
        else:
            print(
                f"history: run {run_id} {nightly.get('date')} {nightly.get('status')} "
                f"candidate {(nightly.get('candidate') or {}).get('commit')}"
            )
    reason = skip_reason(history, candidate, baseline) if args.may_skip else None
    context = {
        "date": args.date,
        "run_url": args.run_url,
        "candidate": candidate,
        "baseline": baseline,
        "previous_green_commit": previous_green_commit(history),
        "skip_reason": reason,
    }
    write_json(args.out, context)
    write_outputs(args.github_output, {"skip": json.dumps(bool(reason))})
    print(f"skip: {reason}" if reason else "skip: no")
    print(f"previous_green_commit: {context['previous_green_commit']}")
    return 0


def cmd_merge(args):
    context = strict_load(args.context)
    plan = strict_load(args.plan) if not context["skip_reason"] else []
    validator = load_validator(args.validator) if args.validator else None
    nightly = build_nightly(context, plan, args.results, validator)
    write_json(args.out, nightly)
    if validator is not None:
        _, problems = validator.validate_file(args.out, kind="nightly")
        if problems:
            for problem in problems:
                print(f"{args.out}: {problem}", file=sys.stderr)
            return 1
    print(f"status: {nightly['status']}")
    for entry in nightly["checks"]:
        print(f"  {entry['check']}: {entry['verdict']} {entry.get('error') or entry.get('summary', '')}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="checks.json + selection -> plan.json and job matrices")
    p.add_argument("--registry", required=True, help="tests/perf/checks.yaml converted to JSON")
    p.add_argument("--select", default="", help="comma list of checks; empty for all")
    p.add_argument("--out", required=True)
    p.add_argument("--github-output", default="")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("context", help="resolved images + earlier nightlies -> context.json")
    for which in ("candidate", "baseline"):
        p.add_argument(f"--{which}-image", required=True)
        p.add_argument(f"--{which}-commit", default="")
        p.add_argument(f"--{which}-label", required=True)
    p.add_argument("--history-runs", default="", help="earlier run ids, newest first")
    p.add_argument("--history-dir", default="", help="holds <run id>/nightly.json")
    p.add_argument("--may-skip", action="store_true", help="tonight may be skipped")
    p.add_argument("--date", required=True)
    p.add_argument("--run-url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--github-output", default="")
    p.set_defaults(func=cmd_context)

    p = sub.add_parser("merge", help="context + plan + perf-result-* -> nightly.json")
    p.add_argument("--context", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--results", required=True, help="holds perf-result-<check>/ directories")
    p.add_argument("--validator", default="", help="path to tests/perf/validate.py")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_merge)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except PlanError as err:
        print(f"::error::{err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
