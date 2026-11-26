"""Every failure of one stored perf run, and how it compares with the nights before it.

`bench nightly failures --data <dir> --run <date>-<run id> [--json]`. <dir> is laid out by
`bench nightly store` (or is the gh-pages archive's data/). A failure is:

  regression  a row the check judged a regression (result.json, and raw/rows.csv beyond its cap)
  broken      a correctness check the CANDIDATE broke, whatever else is true of it: also broken
              on the baseline, a known flaky check with an open issue, a timing check, or broken in
              some runs only. Each says which. Read from the raw files as well as result.json (soak
              and pokec raw/rows.csv, fuzz's per-run rows, ldbc's raw/summary.csv, micro-hw's row
              notes), since a row missing from the findings was not unmeasured: the check left it
              out (an ok row, an info row past the cap, a known flaky row).
  error       a check that could not measure (its verdict is error), or whose files are not stored.

Each failure is compared with the scheduled nights stored before it (run.json trigger
"nightly"; a skipped night is looked through): new, still failing since a date, and the earlier
night's failures that are gone (fixed), for the checks this run measured. Standard library only
(with the soak and fuzz checks' own name lists, which are too).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

from falkorbench.fuzz import checks as fuzz_checks
from falkorbench.nightly import dashboard
from falkorbench.nightly import report
from falkorbench.soak import check as soak_check

CORRECTNESS = ("invariant", "completed")
KINDS = ("error", "regression", "broken")  # the order a report lists them in
FUZZ_INVARIANTS = frozenset(
    (
        *fuzz_checks.TIMING_CHECKS,
        *fuzz_checks.DATA_CHECKS,
        *fuzz_checks.LONG_TIMING_CHECKS,
        *fuzz_checks.LONG_DATA_CHECKS,
        "server_stays_up",
    )
)


@dataclass(frozen=True)
class Failure:
    check: str
    kind: str  # regression | broken | error
    name: str
    metric: str
    candidate: object
    baseline: object
    both: bool  # broken on the baseline too
    detail: str  # what else is true of it, in plain words

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.check, self.kind, self.name, self.metric)


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _is_zero(value: object) -> bool:
    return _number(value) == 0.0


def _rows(path: Path) -> list[dict]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error):
        return []


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _row_name(check: str, row: dict) -> str:
    """The name a check gives a row in its findings, rebuilt from its raw/rows.csv columns."""
    if "query" in row and "class" in row:  # micro-hw
        return f"{row['query']} [{row['class']}]"
    if "query" in row and "phase" in row:  # ldbc
        return f"{row['query']} {str(row.get('metric', '')).removesuffix('_ms')}"
    return str(row.get("name", ""))


def _known(check: str, name: str, status: str) -> str:
    if status != "known":
        return ""
    issue = soak_check.KNOWN_FLAKY.get(name) if check.startswith("soak") else None
    return f"known flaky ({issue})" if issue else "known flaky"


def _broken_detail(both: bool, *extra: str) -> str:
    parts = ["broken on both sides" if both else "broken on the candidate only"]
    return "; ".join([*parts, *(p for p in extra if p)])


def _ratio_text(ratio: object, threshold: object, direction: object) -> str:
    """`1.42x past 1.10x`; for a metric where lower is worse (throughput) `0.62x, below 1/1.30x`."""
    shown, gate = report.ratio(_number(ratio)), report.ratio(_number(threshold))
    if not gate:
        return shown or "no ratio"
    if direction == "lower_is_worse":
        return f"{shown or 'no ratio'}, below 1/{gate}"
    return f"{shown or 'no ratio'} past {gate}"


def _median(reps: object) -> float | None:
    values = [v for v in (_number(x) for x in str(reps or "").split(";")) if v is not None]
    return statistics.median(values) if values else None


POKEC_DETAIL = re.compile(r"candidate: (\d+) errors, (\d+) timeouts in (\d+) of (\d+) runs")
POKEC_BASELINE = re.compile(r"baseline: (\d+) errors, (\d+) timeouts in (\d+) of (\d+) runs")


def _fuzz_broken(name: str, rows: list[dict]) -> list[Failure]:
    """fuzz's raw/rows.csv holds one value per invariant, side and run (1 held, 0 broke), and
    every finding as `finding (<status>)` rows, also those past the findings cap."""
    per_run: dict[tuple[str, str], dict[str, list[float]]] = {}
    judged: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in rows:
        phase, invariant = str(row.get("phase", "")), str(row.get("name", ""))
        value = _number(row.get("value"))
        if value is None:
            continue
        if phase.startswith("finding"):
            # "control writes_add_up", "long index_matches_scan": a finding on an invariant.
            phase, _, invariant = invariant.partition(" ")
            into = judged
        else:
            into = per_run
        if invariant in FUZZ_INVARIANTS:
            sides = into.setdefault((phase, invariant), {"candidate": [], "baseline": []})
            sides.setdefault(str(row.get("side")), []).append(value)
    out = []
    for key in sorted(per_run.keys() | judged.keys()):
        sides = per_run.get(key) or judged[key]
        cand, base = sides.get("candidate", []), sides.get("baseline", [])
        c_broke, b_broke = sum(v == 0 for v in cand), sum(v == 0 for v in base)
        if not c_broke:
            continue
        runs = (
            f"broke in {c_broke} of {len(cand)} candidate runs, {b_broke} of {len(base)} baseline runs"
            if key in per_run
            else "from its finding (no per-run rows)"
        )
        phase, invariant = key
        timing = "a timing check" if invariant in fuzz_checks.LONG_TIMING_CHECKS else ""
        both = bool(b_broke)
        out.append(
            Failure(
                name,
                "broken",
                f"{phase} {invariant}",
                "invariant",
                0,
                0 if both else 1,
                both,
                _broken_detail(both, timing, runs),
            )
        )
    return out


def _rows_broken(name: str, rows: list[dict]) -> list[Failure]:
    """The other checks' raw/rows.csv: a 0/1 correctness row (also one the candidate broke in some
    runs only), a row-count mismatch, and (micro-hw) a query that errors, named as the check
    names it in its findings."""
    out = []
    for row in rows:
        rname, metric = _row_name(name, row), str(row.get("metric", ""))
        status, note = str(row.get("status", "")), str(row.get("note") or "")
        if "class" in row:  # micro-hw: a query that errors, once per query
            if note.startswith("errors on candidate"):
                both = "baseline" in note
                out.append(
                    Failure(
                        name,
                        "broken",
                        rname,
                        "errors",
                        0,
                        0 if both else 1,
                        both,
                        _broken_detail(both, "the query errors"),
                    )
                )
            continue
        if metric in CORRECTNESS and _is_zero(row.get("candidate")):
            both = _is_zero(row.get("baseline"))
            judged = f"judged {status}" if status not in ("", "known") else ""
            out.append(
                Failure(
                    name,
                    "broken",
                    rname,
                    metric,
                    _number(row.get("candidate")),
                    _number(row.get("baseline")),
                    both,
                    _broken_detail(both, _known(name, rname, status), judged),
                )
            )
        elif metric in CORRECTNESS and row.get("kind") == "unconfirmed":
            # soak: a timing check the candidate broke in some runs, not in every pair.
            both = _is_zero(row.get("baseline"))
            out.append(
                Failure(
                    name,
                    "broken",
                    rname,
                    metric,
                    0,
                    _number(row.get("baseline")),
                    both,
                    _broken_detail(both, "a timing check", note, f"judged {status}"),
                )
            )
        elif metric in CORRECTNESS and (m := POKEC_DETAIL.search(str(row.get("detail") or ""))):
            errors, _, runs, _ = (int(x) for x in m.groups())
            if errors or runs:
                b = POKEC_BASELINE.search(str(row.get("detail") or ""))
                both = bool(b and (int(b.group(1)) or int(b.group(3))))
                out.append(
                    Failure(
                        name,
                        "broken",
                        rname,
                        metric,
                        0,
                        0 if both else 1,
                        both,
                        _broken_detail(both, str(row.get("detail")), f"judged {status}"),
                    )
                )
        elif (
            metric == "rows"
            and row.get("direction") == "mismatch"
            and row.get("candidate") != row.get("baseline")
        ):
            out.append(
                Failure(
                    name,
                    "broken",
                    rname,
                    metric,
                    row.get("candidate"),
                    row.get("baseline"),
                    False,
                    "rows differ from the baseline's",
                )
            )
    return out


def _ldbc_broken(name: str, folder: Path) -> list[Failure]:
    """ldbc keeps its correctness results in result.json only; raw/summary.csv has every query's
    requests per phase and side, whatever the findings cap kept."""
    tally: dict[str, dict[str, list[int]]] = {}
    for row in _rows(folder / "raw" / "summary.csv"):
        attempted, ok = _number(row.get("attempted")), _number(row.get("ok"))
        if attempted is None or ok is None:
            continue
        sides = tally.setdefault(str(row.get("query")), {})
        total = sides.setdefault(str(row.get("side")), [0, 0])
        total[0] += int(attempted)
        total[1] += int(attempted - ok)
    out = []
    for query, sides in sorted(tally.items()):
        c_total, c_failed = sides.get("candidate", [0, 0])
        b_total, b_failed = sides.get("baseline", [0, 0])
        if not c_failed:
            continue
        both = bool(b_failed)
        failed = (
            f"{c_failed} of {c_total} candidate requests failed, {b_failed} of {b_total} baseline"
        )
        out.append(
            Failure(
                name,
                "broken",
                f"{query} completed",
                "completed",
                0,
                0 if both else 1,
                both,
                _broken_detail(both, failed + " (raw/summary.csv)"),
            )
        )
    return out


SOME_PASSES = "queries error on the candidate in some passes only"


def _findings(name: str, findings: list[dict]) -> list[Failure]:
    out = []
    for f in findings:
        fname, metric, status = str(f.get("name", "")), str(f.get("metric", "")), f.get("status")
        if metric in CORRECTNESS and _is_zero(f.get("candidate")):
            both = _is_zero(f.get("baseline"))
            out.append(
                Failure(
                    name,
                    "broken",
                    fname,
                    metric,
                    f.get("candidate"),
                    f.get("baseline"),
                    both,
                    _broken_detail(both, f"judged {status}"),
                )
            )
        elif metric == "completed" and SOME_PASSES in fname:
            # micro-hw: queries that errored in some passes only; their rows carry no note.
            out.append(
                Failure(
                    name,
                    "broken",
                    fname,
                    metric,
                    None,
                    f.get("baseline"),
                    False,
                    _broken_detail(False, "errored in some passes only", f"judged {status}"),
                )
            )
        elif metric == "rows" and f.get("candidate") != f.get("baseline"):
            out.append(
                Failure(
                    name,
                    "broken",
                    fname,
                    metric,
                    f.get("candidate"),
                    f.get("baseline"),
                    False,
                    "rows differ from the baseline's",
                )
            )
        elif status == "regression":
            out.append(
                Failure(
                    name,
                    "regression",
                    fname,
                    metric,
                    f.get("candidate"),
                    f.get("baseline"),
                    False,
                    _ratio_text(f.get("ratio"), f.get("threshold"), f.get("direction")),
                )
            )
    return out


def _rows_regressions(name: str, rows: list[dict]) -> list[Failure]:
    """Regression rows of raw/rows.csv, for those a capped result.json left out."""
    out = []
    for row in rows:
        metric = str(row.get("metric", ""))
        if row.get("status") == "regression" and metric not in CORRECTNESS and metric != "rows":
            cand = (
                _number(row.get("candidate"))
                if "candidate" in row
                else _median(row.get("candidate_reps"))
            )
            base = (
                _number(row.get("baseline"))
                if "baseline" in row
                else _median(row.get("baseline_reps"))
            )
            ratio = _ratio_text(row.get("ratio"), row.get("threshold"), row.get("direction"))
            out.append(
                Failure(
                    name,
                    "regression",
                    _row_name(name, row),
                    metric,
                    cand,
                    base,
                    False,
                    f"{ratio} (raw/rows.csv only: past the findings cap)",
                )
            )
        elif row.get("phase") == "finding (regression)" and row.get("side") == "candidate":  # fuzz
            out.append(
                Failure(
                    name,
                    "regression",
                    str(row.get("name", "")),
                    "",
                    _number(row.get("value")),
                    None,
                    False,
                    "raw/rows.csv only",
                )
            )
    return out


def check_failures(name: str, entry: dict, folder: Path) -> list[Failure]:
    """One check's failures, from its stored result.json and raw files (the night's entry when the
    check stored no result). The raw rows come first: they carry what the findings drop."""
    stored = folder / name / "result.json"
    result = _json(stored) or entry
    rows = _rows(folder / name / "raw" / "rows.csv")
    fuzz = bool(rows) and "side" in rows[0] and "value" in rows[0]
    out: dict[tuple, Failure] = {}
    if entry.get("verdict") == "error":
        why = report.one_line(entry.get("error") or result.get("error"), "no reason given")
        out[(name, "error", name, "")] = Failure(
            name, "error", name, "", None, None, False, f"could not measure: {why}"
        )
    elif not stored.is_file():
        out[(name, "error", name, "")] = Failure(
            name,
            "error",
            name,
            "",
            None,
            None,
            False,
            "its files are not stored (perf-result-"
            + name
            + " did not download): its raw rows cannot be "
            "read, so a pass here is not a pass",
        )
    findings = [f for f in result.get("findings") or [] if isinstance(f, dict)]
    if rows and "class" in rows[0]:
        # micro-hw names each erroring query in its rows; its other `completed` findings are the
        # same queries again, summed up ("10 queries error on both sides (e.g. ...)").
        findings = [
            f
            for f in findings
            if f.get("metric") != "completed" or SOME_PASSES in str(f.get("name"))
        ]
    if fuzz:
        raw = _fuzz_broken(name, rows)
    elif (folder / name / "raw" / "summary.csv").is_file():  # ldbc
        raw = _ldbc_broken(name, folder / name)
    else:
        raw = _rows_broken(name, rows)
    names = set()
    for failure in (*raw, *_findings(name, findings)):
        # fuzz names one invariant the same in its rows and findings, whatever the metric.
        if fuzz and failure.name in names:
            continue
        out.setdefault(failure.key, failure)
        names.add(failure.name)
    for failure in _rows_regressions(name, rows):
        if failure.key not in out and not (fuzz and failure.name in names):
            out[failure.key] = failure
    return sorted(out.values(), key=lambda f: (KINDS.index(f.kind), f.name, f.metric))


@dataclass
class Run:
    folder: str
    night: dict
    meta: dict
    failures: list[Failure]

    @property
    def checks(self) -> list[dict]:
        return report.checks_of(self.night)

    @property
    def measured(self) -> set[str]:
        return {
            str(c.get("check")) for c in self.checks if c.get("verdict") in ("pass", "regression")
        }


def load_run(data: Path, folder: str) -> Run:
    path = data / folder
    night, _, _ = report.read_night(path / "nightly.json")
    failures = []
    for entry in report.checks_of(night):
        name = entry.get("check")
        if isinstance(name, str) and dashboard.CHECK_RE.fullmatch(name):
            failures += check_failures(name, entry, path)
    return Run(folder, night, _json(path / "run.json"), failures)


def _order(folder: str) -> tuple[str, int]:
    return folder[:10], int(folder[11:])


def earlier_nights(data: Path, folder: str) -> list[str]:
    """The scheduled nights stored before `folder`, newest first."""
    found = []
    for path in data.iterdir() if data.is_dir() else []:
        if not (path.is_dir() and dashboard.FOLDER_RE.fullmatch(path.name)) or path.name == folder:
            continue
        if _json(path / "run.json").get("trigger") == "nightly" and _order(path.name) < _order(
            folder
        ):
            found.append(path.name)
    return sorted(found, key=_order, reverse=True)


@dataclass
class Compared:
    data: Path
    run: Run
    history: list[Run]  # newest first
    since: dict[tuple, str]  # failure key -> "new" | "since <date>" | "no earlier night"
    fixed: list[Failure]  # in the newest earlier night, gone now, in a check this run measured


def compare(data: Path, folder: str, nights: int = 7) -> Compared:
    run = load_run(data, folder)
    # A skipped night measured nothing (main had not moved): it is neither a pass nor a fix, so
    # the comparison looks through it to the nights that measured.
    history = [
        night
        for night in (load_run(data, f) for f in earlier_nights(data, folder))
        if night.night.get("status") != "skipped"
    ][:nights]
    since: dict[tuple, str] = {}
    for failure in run.failures:
        if not history:
            since[failure.key] = "no earlier night"
            continue
        streak = 0
        for night in history:
            if failure.check not in night.measured and failure.kind != "error":
                break
            if failure.key not in {f.key for f in night.failures}:
                break
            streak += 1
        since[failure.key] = "new" if streak == 0 else f"since {history[streak - 1].folder[:10]}"
    fixed = []
    if history:
        now = {f.key for f in run.failures}
        fixed = [f for f in history[0].failures if f.key not in now and f.check in run.measured]
    return Compared(data, run, history, since, fixed)


def _value(value: object) -> str:
    return "n/a" if value is None else report.number(value)


def text(result: Compared) -> str:
    run, night = result.run, result.run.night
    lines = [
        f"Run {run.folder} · {run.meta.get('scope', 'main')} · {run.meta.get('trigger', 'unknown')} · "
        f"{str(night.get('status', 'unknown')).upper()}",
        f"candidate {report.label(night.get('candidate'))}  {night.get('candidate', {}).get('image', '')}",
        f"baseline  {report.label(night.get('baseline'))}  {night.get('baseline', {}).get('image', '')}",
        "",
        "check | verdict | failures (regression / broken on the candidate / error)",
    ]
    for entry in run.checks:
        name = str(entry.get("check"))
        mine = [f for f in run.failures if f.check == name]
        counts = "/".join(
            str(sum(f.kind == k for f in mine)) for k in ("regression", "broken", "error")
        )
        stored = _json(Path(result.data) / run.folder / name / "result.json") or entry
        partial = stored.get("incomplete")
        partial = f" | incomplete: {report.one_line(partial, '')}" if partial else ""
        lines.append(f"{name} | {entry.get('verdict')} | {counts}{partial}")
    lines += ["", f"Failures ({len(run.failures)}):"]
    if not run.failures:
        lines.append("none")
    for f in run.failures:
        what = f"{f.name}" + (f" ({f.metric})" if f.metric else "")
        lines.append(
            f"- {f.check} | {f.kind} | {what} | candidate {_value(f.candidate)} vs baseline "
            f"{_value(f.baseline)} | {f.detail} | {result.since[f.key]}"
        )
    if result.history:
        previous = result.history[0]
        lines += [
            "",
            f"Compared with {len(result.history)} earlier scheduled night(s), newest {previous.folder}.",
        ]
        base_then = report.label(previous.night.get("baseline"))
        if base_then != report.label(night.get("baseline")):
            lines.append(
                f"Note: that night's baseline was {base_then}, this run's is {report.label(night.get('baseline'))}."
            )
        lines.append(f"Fixed since {previous.folder} ({len(result.fixed)}):")
        lines += [
            f"- {f.check} | {f.kind} | {f.name}" + (f" ({f.metric})" if f.metric else "")
            for f in result.fixed
        ] or ["none"]
    else:
        lines += [
            "",
            "No earlier scheduled night is stored in this data folder: nothing to compare with.",
        ]
    return "\n".join(lines) + "\n"


def as_json(result: Compared) -> dict:
    return {
        "run": result.run.folder,
        "status": result.run.night.get("status"),
        "trigger": result.run.meta.get("trigger"),
        "scope": result.run.meta.get("scope"),
        "checks": {str(c.get("check")): c.get("verdict") for c in result.run.checks},
        "failures": [{**asdict(f), "since": result.since[f.key]} for f in result.run.failures],
        "compared_with": [r.folder for r in result.history],
        "fixed": [asdict(f) for f in result.fixed],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench nightly failures", description=__doc__.split("\n")[0]
    )
    parser.add_argument(
        "--data", required=True, type=Path, help="holds one <date>-<run id>/ per run"
    )
    parser.add_argument("--run", required=True, help="the run's folder, <date>-<run id>")
    parser.add_argument(
        "--nights", type=int, default=7, help="earlier scheduled nights to compare with"
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args(argv)
    if (
        not dashboard.FOLDER_RE.fullmatch(args.run)
        or not (args.data / args.run / "nightly.json").is_file()
    ):
        print(
            f"::error::{args.run!r} is not a stored run (<date>-<run id> with a nightly.json) in {args.data}",
            file=sys.stderr,
        )
        return 1
    result = compare(args.data, args.run, args.nights)
    print(json.dumps(as_json(result), indent=2, default=str) if args.json else text(result), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
