"""The nightly perf run's bookkeeping. What a night does and promises is in the
header of .github/workflows/perf-nightly.yml; each step calls one subcommand:

  plan      checks.yaml (as JSON) + the `checks` input -> plan.json, job matrices
  context   pinned images + earlier nightlies -> context.json (skip, previous green)
  results   context + plan + each planned perf-result-<check> -> nightly.json
  conclude  nightly.json -> the run's exit status (0 green or skipped, 1 otherwise)

Standard library only (with contract and kit, which are too): `bench nightly merge`.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

from falkorbench import contract
from falkorbench import kit

SCHEMA = 1

# The micro tier is not in the registry: the workflow calls _perf-micro.yml,
# which runs its one check, `micro-hw` (hardware counters and allocations on a
# C4 VM, through a nested _perf-vm.yml).
MICRO_HW = "micro-hw"
# A problem with bench/checks.yaml itself is reported under this name, so a
# bad registry shows up in the night's report instead of stopping the night.
REGISTRY = "checks-yaml"
RESERVED = (MICRO_HW, REGISTRY)
MICRO_TIER = {MICRO_HW: {"check": MICRO_HW, "runner": "micro-hw"}}

# A check's name becomes a GCE label, a runner label and part of an artifact name
# (_perf-vm.yml), so it is held to the narrowest of the three. `--` is reserved
# besides: the reusable workflows name their internal artifacts <artifact>--*.
CHECK_NAME = re.compile(r"^[a-z][a-z0-9-]{0,42}$")

RUNNERS = ("vm", "hosted")
MAX_TIMEOUT = 330


NO_RESULT = "no result.json"

# A matrix with no entries fails the job before its `if:` is read, so an empty
# matrix is replaced by this one entry and the job is switched off by has_<runner>.
PLACEHOLDER = [{"check": "none"}]


class PlanError(Exception):
    pass


# --- plan ------------------------------------------------------------------------


def is_check_name(name):
    return isinstance(name, str) and bool(CHECK_NAME.match(name)) and "--" not in name


def registry_entry(name, entry):
    """One checks.yaml entry as a plan row; a malformed one becomes a stub row."""

    def invalid(why):
        return {"check": name, "runner": None, "error": f"invalid bench/checks.yaml entry: {why}"}

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
    # Below the hosted job's 360-minute limit, so the step's own `timeout` fires
    # first and the exit is still recorded.
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= MAX_TIMEOUT:
        return invalid(f"timeout_minutes {timeout!r} is not an integer from 1 to {MAX_TIMEOUT}")
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


def read_registry(registry):
    """(plan rows, problems) for the parsed checks.yaml.

    An entry with a usable name is planned even if the rest of it is wrong (it
    then reports as an error under its own name); one without a usable name, or
    repeating one, can only be reported as a problem with the registry.
    """
    if registry is None:  # a file holding only comments
        return [], []
    if not isinstance(registry, list):
        return [], [f"it is a {type(registry).__name__}, not a list"]
    rows, problems, seen = [], [], set(RESERVED)
    for index, entry in enumerate(registry):
        if not isinstance(entry, dict):
            problems.append(f"entry {index} is not a mapping")
            continue
        name = entry.get("name")
        if not is_check_name(name):
            problems.append(f"entry {index}: name {name!r} is not a check name")
        elif name in seen:
            problems.append(f"entry {index}: name '{name}' is reserved or already used")
        else:
            seen.add(name)
            rows.append(registry_entry(name, entry))
    return rows, problems


def build_plan(registry, select, registry_error=None):
    """The checks tonight's run measures, in registry order (micro first).

    `select` is the workflow's `checks` input: a comma list, empty for all. A name
    it asks for that is not in the registry is planned as a stub, so a typo shows
    up in the report instead of silently measuring less. `registry_error` says the
    file could not be parsed at all.
    """
    rows, problems = ([], [registry_error]) if registry_error else read_registry(registry)
    by_name = {name: dict(row) for name, row in MICRO_TIER.items()}
    by_name.update((row["check"], row) for row in rows)
    registry_stub = []
    if problems:
        why = "bench/checks.yaml: " + "; ".join(problems)
        registry_stub = [{"check": REGISTRY, "runner": None, "error": why}]

    wanted = [name.strip() for name in (select or "").split(",") if name.strip()]
    if not wanted:
        return list(by_name.values()) + registry_stub
    plan = []
    for name in dict.fromkeys(wanted):
        if name in by_name:
            plan.append(by_name[name])
        elif is_check_name(name) and name != REGISTRY:
            plan.append({"check": name, "runner": None, "error": "not in bench/checks.yaml"})
        else:
            raise PlanError(f"checks input: {name!r} is not a check name")
    return plan + registry_stub


def matrix(plan, runner):
    rows = [
        {key: row[key] for key in ("check", "command", "timeout_minutes", "machine_type")}
        for row in plan
        if row["runner"] == runner
    ]
    return rows


def plan_outputs(plan):
    """The $GITHUB_OUTPUT lines the workflow's jobs switch on."""
    # _perf-micro.yml always runs micro-hw, so it is called only when micro-hw
    # is planned: a selection without it starts no C4 VM.
    out = {"run_micro": json.dumps(any(row["runner"] == "micro-hw" for row in plan))}
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


SIDE_FLAGS = tuple(
    f"{which}{suffix}" for which in kit.SIDES for suffix in ("", "_commit", "_label")
)


def sides(args):
    """The pinned candidate and baseline as result.json's sides, checked by kit's
    rules (the same every check applies to its own flags)."""
    # The context step takes kit's flag names, so kit's messages name its flags.
    flags = SimpleNamespace(mode="nightly", **{key: getattr(args, key) for key in SIDE_FLAGS})
    errors = kit.input_errors(flags)
    if errors:
        raise PlanError("; ".join(errors))
    return kit.side(flags, "candidate"), kit.side(flags, "baseline")


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
            nightly = usable(strict_load(path))
        except (OSError, ValueError):
            nightly = None
        history.append((line, nightly))
    return history


def usable(nightly):
    """The fields the skip rule reads, type-checked; None if any is unusable."""
    if not isinstance(nightly, dict):
        return None
    candidate, baseline = nightly.get("candidate"), nightly.get("baseline")
    if not (isinstance(candidate, dict) and isinstance(baseline, dict)):
        return None
    if not isinstance(nightly.get("status"), str) or not isinstance(nightly.get("checks"), list):
        return None
    return nightly


def left_unmeasured(nightly):
    """Whether any check of that night could not measure. A red night can hold
    one too (a regression outranks an error in the status), and skipping after
    it would leave that check unmeasured until main moves."""
    if nightly["status"] == "error":
        return True
    return any(not isinstance(c, dict) or c.get("verdict") == "error" for c in nightly["checks"])


def previous_green_commit(history):
    for _, nightly in history:
        if nightly and nightly["status"] == "green":
            commit = nightly["candidate"].get("commit")
            return (
                commit if isinstance(commit, str) and contract.COMMIT_RE.fullmatch(commit) else None
            )
    return None


def skip_reason(history, candidate, baseline):
    """Why tonight need not run, or None.

    Skipped only when there is nothing new to measure AND the last answer stands:
    the newest earlier nightly left a nightly.json, none of its checks said
    `error` (a transient failure gets retried, even beside a regression), and it
    measured the same candidate commit against the same baseline digest. A
    skipped night counts too, since it carries the pair it skipped, so a quiet
    stretch keeps skipping.
    """
    if not history or candidate["commit"] is None:
        return None
    _, last = history[0]
    if last is None or left_unmeasured(last):
        return None
    if last["candidate"].get("commit") != candidate["commit"]:
        return None
    if last["baseline"].get("image") != baseline["image"]:
        return None
    # Name the night that actually measured this pair, not the last skip. Every
    # night between it and now skipped, so they all carry the same pair.
    measured = last
    for _, nightly in history:
        if nightly is None or nightly["status"] != "skipped":
            if nightly is not None:
                measured = nightly
            break
    return one_line(
        f"main has not moved since the {measured.get('date')} nightly "
        f"({measured.get('status')}, {measured.get('run_url')}): "
        f"same candidate {candidate['label']} and baseline {baseline['label']}"
    )


# --- merge -----------------------------------------------------------------------


# How the check's command ended, beside result.json: tests/perf/vm/vm.py's
# `record` writes it on every runner (_perf-vm.yml for vm and micro-hw, this
# workflow's hosted job for hosted).
RUN_FACTS = "perf-vm.json"


def how_it_ended(artifact):
    """(clean, why): clean is True when perf-vm.json says the command exited 0
    in time, False when it says it did not, None when nothing says either (no
    file, or no exit status: the job was cancelled or killed first)."""
    unknown = None, "nothing records that the check's command finished (no exit status)"
    try:
        facts = strict_load(artifact / RUN_FACTS)
    except (OSError, ValueError):
        return unknown
    if not isinstance(facts, dict):
        return unknown
    exit_status = facts.get("command_exit")
    if facts.get("create_error"):
        return False, f"no VM came up: {facts['create_error']}"
    if facts.get("timed_out"):
        return False, "the check timed out"
    if isinstance(exit_status, int) and not isinstance(exit_status, bool):
        return (True, "") if exit_status == 0 else (False, f"the check exited {exit_status}")
    return unknown


def one_line(text):
    return " ".join(str(text).split())


def stub(check, reason):
    return {"check": check, "verdict": "error", "error": one_line(reason) or NO_RESULT}


def own_error(doc):
    """The `error` text an invalid result.json carries, if it parsed at all."""
    error = doc.get("error") if isinstance(doc, dict) else None
    return error.strip() if isinstance(error, str) else ""


def read_result(path):
    """(document or None, validator problem codes): parsed once, checked once."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None, ["unreadable"]
    try:
        doc = contract.load_json(text)
    except (ValueError, RecursionError):
        return None, ["invalid-json"]
    return doc, list(dict.fromkeys(problem.code for problem in contract.validate_result(doc)))


def collect(check, artifact):
    """The result.json in one artifact directory, or a stub saying why not."""
    path = artifact / "result.json"
    clean, why = how_it_ended(artifact)
    if not path.is_file():
        nested = sorted(str(p.relative_to(artifact)) for p in artifact.rglob("result.json"))
        if nested:
            return stub(check, f"{NO_RESULT} at the artifact root (found {', '.join(nested)})")
        return stub(check, f"{NO_RESULT}: {why}" if clean is False else NO_RESULT)
    result, codes = read_result(path)
    if codes:
        said = own_error(result)
        return stub(
            check,
            f"invalid result.json ({', '.join(codes)})" + (f"; check said: {said}" if said else ""),
        )
    if result["check"] != check:
        return stub(check, f"result.json names check {result.get('check')!r}")
    # The contract exits 0 whenever result.json was written, so a check that did
    # not end cleanly crashed after writing it: its `pass` is not a measurement.
    # Every runner records the exit in perf-vm.json, and a `pass` needs that
    # record to say 0: a job cancelled mid-command uploads result.json with no
    # exit status at all.
    # A regression or error the check wrote still stands.
    if result.get("verdict") == "pass" and clean is not True:
        return stub(check, f"result.json says pass, but {why}")
    return result


def status_of(checks):
    # The rule contract.py checks a nightly against (nightly-status-mismatch);
    # it keeps no function of its own to call.
    verdicts = {entry.get("verdict") for entry in checks}
    if "regression" in verdicts:
        return "red"
    if "error" in verdicts or not checks:
        return "error"
    return "green"


def build_nightly(context, plan, results_dir):
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
        # Only planned artifacts are read: the workflow downloads each one by
        # name, and the reusable workflows' internal <artifact>--* ones are not
        # checks.
        artifact = results_dir / f"perf-result-{row['check']}"
        checks.append(collect(row["check"], artifact))

    nightly["checks"] = checks
    nightly["status"] = status_of(checks)
    return nightly


# How a night ends its workflow run: only a green or skipped night succeeds. A
# red night, and an error night (one that could not measure is not green),
# fail the run, so the list of runs reads as a timeline and a failed scheduled
# run sends GitHub's email.
RUN_SUCCEEDS = ("green", "skipped")


def run_exit_status(status):
    return 0 if status in RUN_SUCCEEDS else 1


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
    registry, error = None, args.registry_error
    if not error:
        try:
            text = Path(args.registry).read_text(encoding="utf-8")
            # yq prints nothing or `null` for a file holding only comments.
            registry = json.loads(text) if text.strip() else None
        except (OSError, ValueError) as err:
            error = f"could not read the converted file: {err}"
    plan = build_plan(registry, args.select, registry_error=error)
    write_json(args.out, plan)
    outputs = plan_outputs(plan)
    write_outputs(args.github_output, outputs)
    for row in plan:
        print(f"planned {row['check']}: {row.get('error') or row['runner']}")
    return 0


def cmd_context(args):
    candidate, baseline = sides(args)
    history = read_history(args.history_runs, args.history_dir)
    for run_id, nightly in history:
        if nightly is None:
            print(f"history: run {run_id} left no readable perf-nightly artifact")
        else:
            print(
                f"history: run {run_id} {nightly.get('date')} {nightly['status']} "
                f"candidate {nightly['candidate'].get('commit')}"
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
    nightly = build_nightly(context, plan, args.results)
    write_json(args.out, nightly)
    # The file as written, read back the way the reporter will read it.
    _, problems = contract.validate_file(args.out, kind="nightly")
    if problems:
        for problem in problems:
            print(f"{args.out}: {problem}", file=sys.stderr)
        return 1
    print(f"status: {nightly['status']}")
    for entry in nightly["checks"]:
        print(
            f"  {entry['check']}: {entry['verdict']} {entry.get('error') or entry.get('summary', '')}"
        )
    return 0


def cmd_conclude(args):
    try:
        nightly = strict_load(args.nightly)
    except (OSError, ValueError) as err:
        print(f"::error::cannot read {args.nightly}: {err}", file=sys.stderr)
        return 1
    status = nightly.get("status") if isinstance(nightly, dict) else None
    code = run_exit_status(status)
    if code:
        print(
            f"::error::the night is {status}: failing the run (the report and nightly.json are already out)"
        )
    else:
        print(f"the night is {status}: the run succeeds")
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bench nightly merge", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="checks.json + selection -> plan.json and job matrices")
    registry = p.add_mutually_exclusive_group(required=True)
    registry.add_argument("--registry", help="bench/checks.yaml converted to JSON")
    registry.add_argument("--registry-error", help="why bench/checks.yaml could not be parsed")
    p.add_argument("--select", default="", help="comma list of checks; empty for all")
    p.add_argument("--out", required=True)
    p.add_argument("--github-output", default="")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("context", help="resolved images + earlier nightlies -> context.json")
    for which in ("candidate", "baseline"):
        p.add_argument(f"--{which}", required=True, help="image, name[:tag]@sha256:<64 hex>")
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

    p = sub.add_parser("results", help="context + plan + perf-result-* -> nightly.json")
    p.add_argument("--context", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--results", required=True, help="holds perf-result-<check>/ directories")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("conclude", help="nightly.json -> this run's exit status")
    p.add_argument("--nightly", required=True)
    p.set_defaults(func=cmd_conclude)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except PlanError as err:
        print(f"::error::{err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
