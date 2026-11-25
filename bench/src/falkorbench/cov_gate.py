"""benchmark-cov's reading as the release gate's result.json (#4004).

On a pull request into a version branch, benchmark-cov.yml measures the PR's
image and its base branch's image on hosted runners: callgrind instructions over
the `cg` subset (sharded) and jemalloc's allocated bytes over every query
(`measure`). This turns those CSVs into one contract result.json, check
`benchmark-cov`, which `bench nightly merge conclude --result` judges against
bench/waivers.yaml.

Gates, PR against base, both bounds strict:

  instructions  > 1.10x, no floor. Callgrind counts repeat to about 0.2%, so a
                10% move on any row is the code. #3996 (1.30x on a query of
                about 240k instructions) is far below micro-hw's hardware floor
                and well above this one.
  alloc_bytes   > 1.10x and > 64 KiB per query, or > 8x and > 4 KiB: micro-hw's
                Rust allocation gate (verdict.HW_GATES), on the same counter.

Both counters are deterministic for an image, so a crossing is a regression
without a re-run. A side with no rows, or a row measured on the base and not on
the PR, is an error: the gate fails, and a re-run clears an infrastructure
failure. Which queries did not answer comes from `bench measure --failed-out`
on each side, by Pokec's rule: a query neither side answers is info (not
supported on this branch: a query main added that the line's engine lacks), one
the base answers and the PR does not is a regression, and one only the base
fails is info (not measured on the base). The contract drops `error` when a regression outranks it, so the
reading's completeness goes to `--unfinished-out` too: a regression in a
partial reading, or one with more regressions than result.json lists, is never
waived (`bench nightly merge conclude --unfinished`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from falkorbench import kit
from falkorbench import metrics
from falkorbench.contract import MAX_FINDINGS
from falkorbench.queries import QUERIES
from falkorbench.verdict import HW_GATES

CHECK = "benchmark-cov"
INSTR_THRESHOLD = metrics.THRESHOLDS["instr"]
ALLOC_THRESHOLD = metrics.THRESHOLDS["alloc_bytes"]
ALLOC_FLOOR, ALLOC_STEEP, ALLOC_STEEP_FLOOR = HW_GATES["rust"]["alloc_bytes"]
CLASS = {q.name: "writes" if q.write else "reads" for q in QUERIES}

# (CSV column, contract metric, tiers [(threshold, floor)], where the rows come from)
GATES = (
    ("instr", "instructions", ((INSTR_THRESHOLD, 0.0),), "callgrind"),
    (
        "alloc_bytes",
        "alloc_bytes",
        ((ALLOC_THRESHOLD, ALLOC_FLOOR), (ALLOC_STEEP, ALLOC_STEEP_FLOOR)),
        "measure",
    ),
)


def name_of(query: str) -> str:
    """micro-hw's finding name, `<row> [<reads|writes>]`, so one waiver pattern
    style serves both checks."""
    klass = CLASS.get(query)
    return f"{query} [{klass}]" if klass else query


MEASURED_RUN = "measured run"  # measure.py's failure prefix: warmup answered, run did not finish


def answered(
    rows: dict[str, dict[str, metrics.Row]], failed: dict[str, dict[str, str]]
) -> tuple[list[dict], dict[str, set[str]]]:
    """(findings, {query: the sides whose missing reading they explain}) from which queries did not answer.

    `rows` is the measure source's {"pr"|"base": rows}; `failed` is each side's
    `bench measure --failed-out`. A query failed on one side and not measured
    on the other is left to the comparison, which calls it unmeasured.
    """
    pr_failed, base_failed = failed.get("pr", {}), failed.get("base", {})
    findings, explained = [], {}
    for query in sorted(set(pr_failed) | set(base_failed)):
        if any(
            why.startswith(MEASURED_RUN)
            for why in (pr_failed.get(query), base_failed.get(query))
            if why
        ):
            continue  # the warmup answered: a broken measurement, not an unsupported query
        if query in pr_failed and query in base_failed:
            candidate, baseline = 0, 0  # not supported on this branch: info
        elif query in pr_failed and query in rows.get("base", {}):
            candidate, baseline = 0, 1  # the PR broke it: a regression
        elif query in base_failed and query in rows.get("pr", {}):
            candidate, baseline = 1, None  # not measured on the base: info
        else:
            continue
        status = kit.correctness_status(candidate, baseline)
        findings.append(
            kit.finding(
                name_of(query), "completed", candidate, baseline, status, direction="lower_is_worse"
            )
        )
        explained[query] = {side for side, side_failed in failed.items() if query in side_failed}
    return findings, explained


def judge(
    sources: dict[str, dict[str, dict[str, metrics.Row]]],
    failed: dict[str, dict[str, str]] | None = None,
) -> tuple[list[dict], list[str]]:
    """(findings, errors) from {"callgrind"|"measure": {"pr"|"base": rows}} and
    each side's failed queries ({"pr"|"base": {query: why}})."""
    findings, explained = answered(sources["measure"], failed or {})
    errors = []
    unfinished = sorted(
        {
            q
            for side in (failed or {}).values()
            for q, why in side.items()
            if why.startswith(MEASURED_RUN)
        }
    )
    if unfinished:
        errors.append(
            f"{len(unfinished)} query(ies) failed their measured run: {', '.join(unfinished[:5])}"
        )
    for column, metric, tiers, source in GATES:
        pr, base = sources[source].get("pr", {}), sources[source].get("base", {})
        if not pr or not base:
            missing = " and ".join(side for side, rows in (("pr", pr), ("base", base)) if not rows)
            errors.append(f"no {source} rows for {missing}")
            continue
        unmeasured, no_baseline = [], []
        for query, base_row in base.items():
            b = base_row.get(column)
            if b is None:
                if "base" not in explained.get(query, ()):
                    no_baseline.append(query)
                continue
            c = pr.get(query, metrics.Row()).get(column)
            if c is None:
                # A failed side has no reading to excuse; a missing reading on the
                # side that did not fail is still an error, and a pair both read is compared.
                if "pr" not in explained.get(query, ()):
                    unmeasured.append(query)
                continue
            moved = {kit.crossed(c, b, threshold, floor) for threshold, floor in tiers}
            status = "regression" if 1 in moved else "improvement" if -1 in moved else None
            if status:
                findings.append(
                    kit.finding(name_of(query), metric, c, b, status, threshold=tiers[0][0])
                )
        if no_baseline:
            shown = ", ".join(no_baseline[:5]) + (" ..." if len(no_baseline) > 5 else "")
            errors.append(f"{len(no_baseline)} {source} row(s) with no baseline {column}: {shown}")
        if unmeasured:
            shown = ", ".join(unmeasured[:5]) + (" ..." if len(unmeasured) > 5 else "")
            errors.append(f"{len(unmeasured)} {source} row(s) measured on base, not on pr: {shown}")

    def worst(f: dict) -> tuple[int, float]:
        rank = {"regression": 0, "improvement": 1}.get(f["status"], 2)
        ratio = f["ratio"] or 1.0
        return rank, -ratio if f["status"] == "regression" else ratio

    return sorted(findings, key=worst), errors


def incomplete(findings: list[dict], errors: list[str]) -> str:
    """Why a waiver cannot be trusted with this reading, or "" when it can."""
    regressions = sum(f["status"] == "regression" for f in findings)
    reasons = list(errors)
    if regressions > MAX_FINDINGS:
        reasons.append(f"{regressions} regressions, more than the {MAX_FINDINGS} result.json lists")
    return "; ".join(reasons)


def summary(findings: list[dict], errors: list[str]) -> str:
    regressed = [f for f in findings if f["status"] == "regression"]
    if regressed:
        f = regressed[0]
        ratio = f" {f['ratio']:.2f}x" if f["ratio"] is not None else ""
        head = f"{len(regressed)} regression(s), worst {f['name']} {f['metric']}{ratio}"
    else:
        head = "no regression"
    improved = sum(f["status"] == "improvement" for f in findings)
    unsupported = sum(f["status"] == "info" for f in findings)
    info = f", {unsupported} quer(ies) not answered on the base" if unsupported else ""
    tail = f"; could not measure: {'; '.join(errors)}" if errors else ""
    return f"{head}; {improved} improvement(s){info} (PR vs base, callgrind and allocations){tail}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench cov-gate", description=__doc__.split("\n")[0])
    kit.add_args(parser)
    for side in ("pr", "base"):
        parser.add_argument(f"--measure-{side}", required=True, help=f"full-{side}.csv")
        parser.add_argument(f"--callgrind-{side}", required=True, help=f"glob of cg-{side}-*.csv")
        parser.add_argument(f"--failed-{side}", default="", help="bench measure --failed-out")
    parser.add_argument(
        "--unfinished-out", default="", help="{benchmark-cov: why} when the reading is partial"
    )
    args = parser.parse_args(argv)
    started = kit.utc_now()
    out = kit.out_dir(args.out)
    sources = {
        "callgrind": {
            s: metrics.read_rows(getattr(args, f"callgrind_{s}")) for s in ("pr", "base")
        },
        "measure": {s: metrics.read_rows(getattr(args, f"measure_{s}")) for s in ("pr", "base")},
    }
    failed = {}
    for side in ("pr", "base"):
        path = Path(getattr(args, f"failed_{side}") or "/nonexistent")
        # No file: that side's measure job did not finish, and its rows are missing too.
        failed[side] = json.loads(path.read_text()) if path.is_file() else {}
    findings, errors = judge(sources, failed)
    # The measuring ran on separate hosted runners, one per side and shard; both
    # counters are the same on any of them. This describes the runner that judged.
    host = kit.machine()
    machine = {**host, "runner": "hosted", "type": "ubuntu-latest"}
    result = kit.result(
        CHECK,
        args,
        started=started,
        findings=findings,
        summary=summary(findings, errors),
        errors=errors,
        machine_info=machine,
    )
    kit.write_result(Path(out), result)
    if args.unfinished_out:
        why = incomplete(findings, errors)
        Path(args.unfinished_out).write_text(json.dumps({CHECK: why} if why else {}) + "\n")
    print(f"{result['verdict']}: {result['summary']}")
    for f in result["findings"]:
        print(f"  [{f['status']}] {f['name']} {f['metric']} {f['ratio']}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
