"""The nightly perf run's bookkeeping. What a night does and promises is in the
header of .github/workflows/perf-nightly.yml; each step calls one subcommand:

  plan      checks.yaml (as JSON) + the `checks` input -> plan.json, job matrices
  context   pinned images + earlier nightlies -> context.json (skip, previous green)
  results   context + plan + each planned perf-result-<check> -> nightly.json
  conclude  nightly.json + bench/waivers.yaml -> the run's exit status (see conclude())
  release   release tags -> the release gate's baseline (see release_below())

Standard library only (with contract and kit, which are too): `bench nightly merge`.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

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
CHECK_NAME = re.compile(r"[a-z][a-z0-9-]{0,42}")

RUNNERS = ("vm", "hosted")
# Minutes a check's command may run. A VM deletes itself 6 h after boot, so
# tests/perf/vm/plan.sh refuses more than 300; the hosted job's limit is 360,
# and 330 keeps run-check.sh's own `timeout` (and so the exit record) first.
MAX_TIMEOUT = {"vm": 300, "hosted": 330}


NO_RESULT = "no result.json"

# A matrix with no entries fails the job before its `if:` is read, so an empty
# matrix is replaced by this one entry and the job is switched off by has_<runner>.
PLACEHOLDER = [{"check": "none"}]


class PlanError(Exception):
    pass


# --- plan ------------------------------------------------------------------------


def is_check_name(name):
    return isinstance(name, str) and bool(CHECK_NAME.fullmatch(name)) and "--" not in name


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
    limit = MAX_TIMEOUT[runner]
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= limit:
        return invalid(
            f"timeout_minutes {timeout!r} is not an integer from 1 to {limit} ({runner})"
        )
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
        # run-check.sh reads one line and splits it on blanks: a command written
        # over several YAML lines is the same words on one line.
        "command": " ".join(command.split()),
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
    """One runner's job matrix. perf-nightly.yml caps how many VM legs run at
    once, and GitHub queues the legs in matrix order, so the VM legs go
    longest budget first (ties in plan order): a long leg left to the end
    would stretch the run by its whole length."""
    rows = [
        {key: row[key] for key in ("check", "command", "timeout_minutes", "machine_type")}
        for row in plan
        if row["runner"] == runner
    ]
    if runner == "vm":
        rows.sort(key=lambda row: -row["timeout_minutes"])
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


def _commit(value):
    return value if isinstance(value, str) and contract.COMMIT_RE.fullmatch(value) else None


def previous_green_commit(history):
    """The newest green night's candidate commit. The workflow reads a bounded
    number of earlier runs, so when none of them was green, the newest night that
    recorded a previous green commit hands it on: a long red or skipped stretch
    keeps the commit range of the next report."""
    for _, nightly in history:
        if nightly and nightly["status"] == "green":
            return _commit(nightly["candidate"].get("commit"))
    for _, nightly in history:
        if nightly and _commit(nightly.get("previous_green_commit")):
            return nightly["previous_green_commit"]
    return None


def skip_reason(history, candidate, baseline, planned):
    """Why tonight need not run, or None.

    Skipped only when there is nothing new to measure AND the last answer stands:
    the newest earlier nightly left a nightly.json, none of its checks said
    `error` (a transient failure gets retried, even beside a regression), and it
    had the same candidate (commit and digest) and baseline digest. A skipped
    night counts too, since it carries the pair it skipped, so a quiet stretch
    keeps skipping. The night that measured that pair must be at hand and must
    have run exactly tonight's planned checks (`planned`, their names): a check
    added to the registry, or a registry that no longer parses, is measured even
    when the image pair has not moved.
    """
    if not history or candidate["commit"] is None:
        return None
    _, last = history[0]
    if last is None or left_unmeasured(last):
        return None
    if last["candidate"].get("commit") != candidate["commit"]:
        return None
    if last["candidate"].get("image") != candidate["image"]:
        return None
    if last["baseline"].get("image") != baseline["image"]:
        return None
    # The night that actually measured this pair: every night after it skipped
    # and carries the same pair. If its artifact is gone, nothing says what it
    # measured, so tonight runs.
    measured = next((n for _, n in history if n is None or n["status"] != "skipped"), None)
    if measured is None:
        return None
    ran = {c.get("check") for c in measured["checks"] if isinstance(c, dict)}
    if ran != set(planned):
        return None
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


def run_facts(artifact):
    """perf-vm.json as a dict, or None when it is missing or not an object."""
    try:
        facts = strict_load(artifact / RUN_FACTS)
    except (OSError, ValueError):
        return None
    return facts if isinstance(facts, dict) else None


def how_it_ended(artifact):
    """(clean, why): clean is True when perf-vm.json says the command exited 0
    in time, False when it says it did not, None when nothing says either (no
    file, or no exit status: the job was cancelled or killed first)."""
    unknown = None, "nothing records that the check's command finished (no exit status)"
    facts = run_facts(artifact)
    if facts is None:
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


def fallback_note(artifact):
    """What a result's summary says first when its VM was the fallback machine
    type (perf-vm.json `fallback_used`), whose numbers do not line up with
    other nights'; "" otherwise."""
    facts = run_facts(artifact)
    if facts is None or facts.get("fallback_used") is not True:
        return ""
    machine = facts.get("machine_type")
    machine = one_line(machine) if isinstance(machine, str) and machine.strip() else "unknown"
    return f"ran on {machine}, the fallback machine type: not comparable with other nights"


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


def _digest(image):
    return image.rpartition("@")[2] if isinstance(image, str) else None


def collect(check, artifact, pinned):
    """The result.json in one artifact directory, or a stub saying why not.
    `pinned` is the night's candidate and baseline: a result that measured
    another pair (an earlier attempt's artifact, say) is not tonight's."""
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
    for which, side in pinned.items():
        got, want = _digest(result[which]["image"]), _digest(side["image"])
        if got != want:
            return stub(check, f"result.json measured {which} {got}, but the night pinned {want}")
    # The contract exits 0 whenever result.json was written, so a check that did
    # not end cleanly crashed after writing it: its `pass` is not a measurement.
    # Every runner records the exit in perf-vm.json, and a `pass` needs that
    # record to say 0: a job cancelled mid-command uploads result.json with no
    # exit status at all.
    # A regression or error the check wrote still stands.
    if result.get("verdict") == "pass" and clean is not True:
        return stub(check, f"result.json says pass, but {why}")
    note = fallback_note(artifact)
    if note:
        result["summary"] = f"{note}; {result['summary']}"
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
        pinned = {"candidate": context["candidate"], "baseline": context["baseline"]}
        checks.append(collect(row["check"], artifact, pinned))

    nightly["checks"] = checks
    nightly["status"] = status_of(checks)
    return nightly


def unfinished(plan, results_dir, nightly):
    """{check: why} for each regression whose check did not end cleanly (killed,
    timed out, cancelled, crashed after writing result.json). The regression
    stands in nightly.json, but the gate does not waive it: rows after the one
    that regressed may never have been measured."""
    out = {}
    by_check = {c.get("check"): c for c in nightly["checks"] if isinstance(c, dict)}
    for row in plan:
        entry = by_check.get(row["check"], {})
        if row.get("error") or entry.get("verdict") != "regression":
            continue
        clean, why = how_it_ended(Path(results_dir) / f"perf-result-{row['check']}")
        if clean is not True:
            out[row["check"]] = why
    return out


# How a night ends its workflow run: a green or skipped night succeeds, and a
# red one whose every regression is waived. Any other red night, and an error
# night (one that could not measure is not green), fails the run, so the list of
# runs reads as a timeline, a failed scheduled run sends GitHub's email, and a
# release gate (a call of the workflow) says no. A gate also fails a skipped
# night: it measured nothing.
WAIVER_KEYS = ("check", "finding", "issue", "reason")
WAIVER_OPTIONAL = ("metric", "max_ratio")


class Waiver(NamedTuple):
    check: str
    pattern: re.Pattern
    issue: int
    metric: str | None
    max_ratio: float | None

    def covers(self, check, finding):
        name, ratio = finding.get("name"), finding.get("ratio")
        if check != self.check or not isinstance(name, str) or not self.pattern.fullmatch(name):
            return False
        if self.metric is not None and finding.get("metric") != self.metric:
            return False
        if self.max_ratio is None:
            return True
        # A bound needs a ratio under it: a finding with none is not covered. It
        # bounds how much worse the row is, so a lower-is-worse ratio (throughput,
        # candidate / baseline) is inverted: 0.7x is 1.43x worse, and 0x is unbounded.
        if not _is_number(ratio):
            return False
        if finding.get("direction") == "lower_is_worse":
            return ratio > 0 and 1 / ratio <= self.max_ratio
        return ratio <= self.max_ratio


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def read_waivers(doc):
    """bench/waivers.yaml (as JSON) -> [Waiver], or PlanError. Every entry is
    checked: a waiver that cannot be read must not quietly stop matching, nor
    match more than it says."""
    if doc is None:  # a file holding only comments
        return []
    if not isinstance(doc, list):
        raise PlanError(f"bench/waivers.yaml: a {type(doc).__name__}, not a list")
    waivers = []
    for index, entry in enumerate(doc):
        where = f"bench/waivers.yaml entry {index}"
        if not isinstance(entry, dict) or not (
            set(WAIVER_KEYS) <= set(entry) <= set(WAIVER_KEYS + WAIVER_OPTIONAL)
        ):
            raise PlanError(
                f"{where}: needs the keys {', '.join(WAIVER_KEYS)}"
                f" and may have {', '.join(WAIVER_OPTIONAL)}"
            )
        check, finding, issue, reason = (entry[key] for key in WAIVER_KEYS)
        metric, max_ratio = entry.get("metric"), entry.get("max_ratio")
        if not (is_check_name(check) and check != REGISTRY):
            raise PlanError(f"{where}: check {check!r} is not a check name")
        if isinstance(issue, bool) or not isinstance(issue, int) or issue < 1:
            raise PlanError(f"{where}: issue {issue!r} is not an issue number")
        if not isinstance(reason, str) or not reason.strip() or "\n" in reason.strip():
            raise PlanError(f"{where}: reason must be one non-empty line")
        if "metric" in entry and metric not in contract.METRICS:
            raise PlanError(f"{where}: metric {metric!r} is not a contract metric")
        if "max_ratio" in entry and not (_is_number(max_ratio) and max_ratio > 0):
            raise PlanError(f"{where}: max_ratio {max_ratio!r} is not a positive number")
        try:
            pattern = re.compile(finding) if isinstance(finding, str) and finding else None
        except re.error as err:
            raise PlanError(f"{where}: finding {finding!r} is not a regex: {err}") from err
        if pattern is None:
            raise PlanError(f"{where}: finding must be a non-empty regex")
        waivers.append(Waiver(check, pattern, issue, metric, max_ratio))
    return waivers


def conclude(nightly, waivers, gate=False, unfinished=None):
    """(exit status, lines saying why). A regression finding is waived when a
    waiver names its check, its pattern matches the WHOLE finding name, and its
    metric and max_ratio (when given) hold; it is still reported. An error is
    never waived, nor a regression verdict with no regression finding to name,
    nor one from a run that did not finish (`unfinished`: {check: why}), nor one
    from a result whose `incomplete` is not null (it left parts unmeasured or
    cut its findings)."""
    unfinished = unfinished or {}
    status = nightly.get("status") if isinstance(nightly, dict) else None
    if status == "green" or (status == "skipped" and not gate):
        return 0, [f"the night is {status}: the run succeeds"]
    if status == "skipped":
        return 1, ["the night is skipped: a gate that measured nothing does not pass"]
    checks = nightly.get("checks") if status in ("red", "error") else None
    if not isinstance(checks, list):
        return 1, [f"the night is {status}: the run fails"]
    blocking, waived = [], []
    for entry in checks:
        entry = entry if isinstance(entry, dict) else {}
        check, verdict = entry.get("check"), entry.get("verdict")
        if verdict == "pass":
            continue
        if verdict != "regression":
            blocking.append(f"{check}: {verdict}: {one_line(entry.get('error') or '')}")
            continue
        if check in unfinished:
            blocking.append(
                f"{check}: regression from a run that did not finish ({unfinished[check]})"
            )
            continue
        if entry.get("incomplete") is not None:
            blocking.append(
                f"{check}: regression from a result that did not measure everything"
                f" ({one_line(str(entry['incomplete']))})"
            )
            continue
        found = [
            f
            for f in entry.get("findings") or []
            if isinstance(f, dict) and f.get("status") == "regression"
        ]
        if not found:
            blocking.append(f"{check}: regression with no finding to waive")
        for finding in found:
            said = (
                f"{check}: {finding.get('name')} ({finding.get('metric')} {finding.get('ratio')}x)"
            )
            issue = next((w.issue for w in waivers if w.covers(check, finding)), None)
            if issue is None:
                blocking.append(said)
            else:
                waived.append(f"{said} waived by #{issue}")
    # A red or error night with nothing to block on and nothing waived names no
    # reason to be what it is: it does not pass either.
    code = 1 if blocking or not waived else 0
    verb = "fails" if code else "succeeds"
    lines = [f"the night is {status}: the run {verb}"]
    lines += [f"waived (reported, not gated): {line}" for line in waived]
    lines += [f"not waived: {line}" for line in blocking]
    return code, lines


# The release gate's baseline: the highest stable release strictly below a
# version. A pull request into X.Y names X.Y+1.0 (the newest X.Y.z, or the line
# before when X.Y has none), a tag vX.Y.Z names X.Y.Z (never itself).
VERSION = re.compile(r"v?([0-9]+)\.([0-9]+)\.([0-9]+)(-[0-9A-Za-z.-]+)?")


def release_below(tags, below):
    """The tag in `tags` (stable vX.Y.Z, v5 and up) highest below `below`, or None."""
    bound = VERSION.fullmatch(below or "")
    if not bound:
        raise PlanError(f"{below!r} is not X.Y.Z[-prerelease]")
    # A prerelease bound X.Y.Z-pre sits below X.Y.Z, so X.Y.Z itself is excluded too.
    limit = tuple(int(part) for part in bound.groups()[:3])
    found = []
    for tag in tags:
        match = VERSION.fullmatch(tag)
        if match and tag.startswith("v") and not match.group(4) and int(match.group(1)) >= 5:
            version = tuple(int(part) for part in match.groups()[:3])
            if version < limit:
                found.append((version, tag))
    return max(found)[1] if found else None


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
    rows = strict_load(args.plan)
    planned = [row["check"] for row in rows]
    # A registry entry that cannot run keeps its name, so the names alone would
    # match the last measured night and the error would never be reported.
    broken = any(row.get("error") for row in rows)
    may_skip = args.may_skip and not broken
    reason = skip_reason(history, candidate, baseline, planned) if may_skip else None
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
    if args.unfinished_out:
        write_json(args.unfinished_out, unfinished(plan, args.results, nightly))
    # The file as written, read back the way the reporter will read it.
    _, problems = contract.validate_file(args.out, kind="nightly")
    if problems:
        for problem in problems:
            print(f"{args.out}: {problem}", file=sys.stderr)
        return 1
    print(f"status: {nightly['status']}")
    for entry in nightly["checks"]:
        line = f"  {entry['check']}: {entry['verdict']} {entry.get('error') or entry.get('summary', '')}"
        # JSON allows a lone surrogate a strict UTF-8 console cannot encode; the
        # night is already written, so the log must not be what fails it.
        print(line.encode("utf-8", "replace").decode("utf-8"))
    return 0


def read_optional_json(path):
    """A JSON file yq or merge wrote, or None for no path; nothing in the file
    (yq on a file holding only comments) is None too."""
    text = Path(path).read_text(encoding="utf-8") if path else ""
    return contract.load_json(text) if text.strip() else None


def cmd_conclude(args):
    try:
        if args.result:
            # One check's result.json (benchmark-cov's gate): judged as a night of one.
            result = strict_load(args.result)
            nightly = {"status": status_of([result]), "checks": [result]}
            problems = contract.validate_result(result)
        else:
            nightly = strict_load(args.nightly)
            problems = None
    except (OSError, ValueError) as err:
        print(f"::error::cannot read {args.result or args.nightly}: {err}", file=sys.stderr)
        return 1
    try:
        waivers = read_waivers(read_optional_json(args.waivers))
        unfinished = read_optional_json(args.unfinished) or {}
    except (OSError, ValueError, RecursionError) as err:
        print(f"::error::cannot read the waivers or unfinished file: {err}", file=sys.stderr)
        return 1
    if not isinstance(unfinished, dict):
        print(f"::error::{args.unfinished}: not a mapping of check to reason", file=sys.stderr)
        return 1
    code, lines = conclude(nightly, waivers, gate=args.gate, unfinished=unfinished)
    # A red night that passes rests on waived findings: only from a document the
    # contract accepts. One result is always checked: its status comes from a
    # verdict nothing else has validated.
    if not code and (args.result or nightly.get("status") not in ("green", "skipped")):
        if problems is None:
            problems = contract.validate_nightly(nightly)
        if problems:
            code = 1
            lines.append(f"not waived: the result is not valid ({problems[0]})")
    first, *rest = lines
    if code:
        print(f"::error::{first} (the report and nightly.json are already out)")
    else:
        print(first)
    for line in rest:
        level = "warning" if line.startswith("waived") else "error"
        print(f"::{level}::{line}")
    return code


def cmd_release(args):
    tag = release_below(sys.stdin.read().split(), args.below)
    if tag is None:
        print(f"::error::no stable release below {args.below}", file=sys.stderr)
        return 1
    print(tag)
    return 0


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
    p.add_argument("--plan", required=True, help="tonight's plan.json (`plan` writes it)")
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
    p.add_argument(
        "--unfinished-out", default="", help="{check: why} for regressions that did not finish"
    )
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("conclude", help="nightly.json (+ waivers) -> this run's exit status")
    judged = p.add_mutually_exclusive_group(required=True)
    judged.add_argument("--nightly")
    judged.add_argument("--result", help="one check's result.json, judged as a night of one")
    p.add_argument("--waivers", default="", help="bench/waivers.yaml converted to JSON")
    p.add_argument("--unfinished", default="", help="what `results --unfinished-out` wrote")
    p.add_argument("--gate", action="store_true", help="a release gate: a skipped night fails")
    p.set_defaults(func=cmd_conclude)

    p = sub.add_parser("release", help="release tags on stdin -> the highest below --below")
    p.add_argument("--below", required=True, help="X.Y.Z[-prerelease]")
    p.set_defaults(func=cmd_release)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except PlanError as err:
        print(f"::error::{err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
