"""micro-hw's verdict: two images' measure passes in, one contract `result.json` out.

Apart from reading the pass CSVs and writing rows.csv, it is pure (kit.result
builds the result; kit.run_check writes it), so every rule is covered by
tests/test_verdict.py without a server. What the check measures, its rules and
the basis of each gate: bench/README.md, "Nightly micro check".
"""

from __future__ import annotations

import csv
import math
import statistics
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from falkorbench import kit
from falkorbench.contract import MAX_FINDINGS
from falkorbench.metrics import THRESHOLDS
from falkorbench.metrics import Row
from falkorbench.metrics import read_rows
from falkorbench.model import Query

CHECK = "micro-hw"
SIDES = kit.SIDES
REPS = 2


@dataclass(frozen=True)
class MetricSpec:
    """One reading the check gates on: its contract name, CSV column and gate."""

    name: str
    column: str
    floor: float
    steep: float | None = None  # a ratio that crosses on `steep_floor` instead
    steep_floor: float = 0.0
    threshold: float = 1.10

    def crossing(self, cand: Sequence[float], base: Sequence[float]) -> str | None:
        return crossing(
            cand,
            base,
            threshold=self.threshold,
            floor=self.floor,
            steep=self.steep,
            steep_floor=self.steep_floor,
        )


# --- the gates ---------------------------------------------------------------
#
# metric column -> (floor, steep ratio or None, steep floor). Each comment is
# the same-image (A/A) basis; bench/README.md has the table, TestHwGates pins it.
HW_GATES: dict[str, dict[str, tuple[float, float | None, float]]] = {
    "rust": {
        # worst A/A past 1.10x: +103,619
        "instr": (250_000.0, None, 0.0),
        # worst A/A past 1.10x: +9,781,665; steep basis 1.113x
        "cycles": (20_000_000.0, 1.8, 250_000.0),
        # worst A/A past 1.25x: +5,598; steep basis 1.716x
        "br_miss": (12_000.0, 2.8, 2_000.0),
        # worst A/A past 1.25x: +111,503; steep basis 1.816x
        "l2_miss": (225_000.0, 3.0, 15_000.0),
        # worst A/A past 1.10x: +17,655; steep basis 2.620x
        "alloc_bytes": (65_536.0, 8.0, 4_096.0),
    },
    "c": {
        # worst A/A past 1.10x: +629,577 (the worker's busy-wait is counted)
        "instr": (1_300_000.0, None, 0.0),
        # worst A/A past 1.10x: +134,843
        "cycles": (270_000.0, None, 0.0),
        # worst A/A past 1.25x: +20,159; no steep tier (a 3.58x split at +10,998)
        "br_miss": (40_000.0, None, 0.0),
        # worst A/A past 1.25x: +332,270; steep basis 1.534x
        "l2_miss": (665_000.0, 3.1, 15_000.0),
        # none past 1.10x (1.097x at +1,537,039); steep basis 1.097x
        "alloc_bytes": (3_100_000.0, 8.0, 4_096.0),
    },
}

_NAMES = {
    "instr": "instructions",
    "cycles": "cycles",
    "br_miss": "branch_misses",
    "l2_miss": "l2_misses",
    "alloc_bytes": "alloc_bytes",
}


def hw_metrics(engine: str = "rust") -> tuple[MetricSpec, ...]:
    """The gates for one engine ("rust" or "c"; C when either side is)."""
    return tuple(
        MetricSpec(
            name,
            col,
            floor,
            steep=steep,
            steep_floor=steep_floor,
            threshold=THRESHOLDS[col],
        )
        for col, name in _NAMES.items()
        for floor, steep, steep_floor in [HW_GATES[engine][col]]
    )


HW_METRICS = hw_metrics("rust")


# --- file layout -------------------------------------------------------------


def screen_path(root: Path, side: str, rep: int) -> Path:
    """Where a pass's CSV lives. One convention, shared by `bench micro-hw` (which
    writes these names) and `load` (which checks them)."""
    return root / f"measure-{side}-r{rep}.csv"


def confirm_path(root: Path, side: str, rep: int) -> Path:
    """Where a second-run pass's CSV lives."""
    return root / f"confirm-{screen_path(Path(), side, rep).name}"


@dataclass
class Passes:
    """One side's reps: a table per pass."""

    reps: list[dict[str, Row]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def load(root: Path, side: str) -> Passes:
    """Read every pass of one side. A missing file is recorded, never treated
    as an empty pass."""
    out = Passes()
    for rep in range(1, REPS + 1):
        p = screen_path(root, side, rep)
        if p.exists():
            out.reps.append(read_rows(str(p)))
        else:
            out.missing.append(p.name)
            out.reps.append({})
    return out


def load_confirm(root: Path, side: str) -> Passes:
    """Read the second run's passes of one side.

    No missing-file check here: a crossed row with no second-run reading is an
    error in `_confirm`, which is the check that matters.
    """
    return Passes(
        reps=[
            read_rows(str(p)) if (p := confirm_path(root, side, rep)).exists() else {}
            for rep in range(1, REPS + 1)
        ]
    )


# --- the crossing rule -------------------------------------------------------


def values(passes: Passes, query: str, column: str) -> list[float]:
    return [
        v
        for t in passes.reps
        if (r := t.get(query)) is not None and (v := r.get(column)) is not None
    ]


def ratio_of(cand: Sequence[float], base: Sequence[float]) -> float | None:
    """Median candidate over median baseline; None when it cannot be formed."""
    if not cand or not base:
        return None
    b = statistics.median(base)
    if b <= 0:
        return None
    return statistics.median(cand) / b


def crossing(
    cand: Sequence[float],
    base: Sequence[float],
    *,
    threshold: float,
    floor: float,
    steep: float | None = None,
    steep_floor: float = 0.0,
) -> str | None:
    """'regression', 'improvement' or None for one row's reps.

    The medians must cross `threshold` by more than `floor`, or `steep` (when
    given) by more than `steep_floor` (kit.crossed: both bounds strict, and with
    a zero baseline the floor alone decides); and the reps must not overlap. Over
    a zero baseline a tier with no floor never crosses, or any reading would.
    """
    if not cand or not base:
        return None
    mc, mb = statistics.median(cand), statistics.median(base)
    tiers = [(threshold, floor), *([(steep, steep_floor)] if steep is not None else [])]
    tiers = [(t, f) for t, f in tiers if f > 0 or mb > 0]
    moved = {kit.crossed(mc, mb, t, f) for t, f in tiers}
    if 1 in moved and min(cand) > max(base):
        return "regression"
    if -1 in moved and max(cand) < min(base):
        return "improvement"
    return None


# --- assessment --------------------------------------------------------------


@dataclass
class RowResult:
    query: str
    klass: str  # "reads" | "writes"
    metric: str
    cand: list[float]
    base: list[float]
    confirm_cand: list[float] = field(default_factory=list)
    confirm_base: list[float] = field(default_factory=list)
    crossed: str | None = None
    status: str = "ok"
    note: str = ""
    threshold: float = 1.10

    @property
    def ratio(self) -> float | None:
        return ratio_of(self.cand, self.base)


@dataclass
class Inputs:
    """Everything the verdict reads, already loaded."""

    queries: Sequence[Query]
    screen: dict[str, Passes]  # side -> passes
    confirm: dict[str, Passes] = field(default_factory=dict)
    metrics: tuple[MetricSpec, ...] = HW_METRICS


def failed_in_measure(passes: Passes, names: Iterable[str]) -> set[str]:
    """Queries absent from every rep of a measure pass that exists.

    `bench measure` leaves a row out only when its warmup reply was an error.
    """
    present = [t for t in passes.reps if t]
    if not present:
        return set()
    return {n for n in names if all(n not in t for t in present)}


def failed_sometimes(passes: Passes, names: Iterable[str]) -> set[str]:
    """Queries absent from some, but not all, of a side's measure passes."""
    present = [t for t in passes.reps if t]
    return {n for n in names if any(n not in t for t in present) and any(n in t for t in present)}


def assess(inp: Inputs) -> tuple[list[RowResult], list[str], dict[str, set[str]]]:
    """Per-row results for every metric, plus the reasons the run is an error,
    plus which queries errored on each side in `measure`."""
    errors: list[str] = []
    for side, p in inp.screen.items():
        if p.missing:
            errors.append(f"{side}: missing {', '.join(sorted(p.missing)[:6])}")

    all_names = [q.name for q in inp.queries]
    failed = {side: failed_in_measure(inp.screen[side], all_names) for side in SIDES}

    results: list[RowResult] = []
    expected = list(inp.queries)
    cand_p, base_p = inp.screen["candidate"], inp.screen["baseline"]
    for spec in inp.metrics:
        unmeasured: list[str] = []
        for q in expected:
            cand = values(cand_p, q.name, spec.column)
            base = values(base_p, q.name, spec.column)
            row = RowResult(
                q.name,
                "writes" if q.write else "reads",
                spec.name,
                cand,
                base,
                threshold=spec.threshold,
            )
            results.append(row)
            errored = [s for s in SIDES if q.name in failed[s]]
            if errored:
                # Reported once, as a correctness finding, not as a gap per metric.
                row.status = "info"
                row.note = f"errors on {' and '.join(errored)}"
                continue
            if not cand or not base:
                # No reading is never a pass: a row that goes missing on one side is
                # exactly where something may have broken.
                unmeasured.append(q.name)
                row.status = "info"
                row.note = "unmeasured on " + " and ".join(
                    s for s, v in (("candidate", cand), ("baseline", base)) if not v
                )
                continue
            row.crossed = spec.crossing(cand, base)
            if row.crossed == "improvement":
                row.status = "improvement"
        # A side whose every reading is 0 measured nothing: no query costs zero
        # instructions or allocates nothing on every row, and a counter that a
        # PMU filter blocks reads 0 rather than failing.
        if not any(v > 0 for q in expected for v in values(cand_p, q.name, spec.column)):
            errors.append(f"no {spec.name} reading for the candidate")
        if not any(v > 0 for q in expected for v in values(base_p, q.name, spec.column)):
            errors.append(f"no {spec.name} reading for the baseline")
        if unmeasured:
            errors.append(
                f"{len(unmeasured)} {spec.name} row(s) have no reading on a side "
                f"(e.g. {unmeasured[0]})"
            )

    _confirm(inp, results, errors)
    return results, errors, failed


def to_confirm(results: Iterable[RowResult]) -> list[RowResult]:
    return [r for r in results if r.crossed == "regression"]


def _confirm(inp: Inputs, results: list[RowResult], errors: list[str]) -> None:
    """Apply the re-run to every crossed regression.

    A crossed row the re-run has no reading for cannot be called either way, so
    it makes the run an error rather than a pass.
    """
    spec_of = {s.name: s for s in inp.metrics}
    empty = Passes()
    for r in to_confirm(results):
        spec = spec_of[r.metric]
        r.confirm_cand = values(inp.confirm.get("candidate", empty), r.query, spec.column)
        r.confirm_base = values(inp.confirm.get("baseline", empty), r.query, spec.column)
        again = spec.crossing(r.confirm_cand, r.confirm_base)
        if again == "regression":
            r.status = "regression"
            r.note = "confirmed by re-run"
        elif not r.confirm_cand or not r.confirm_base:
            r.status = "info"
            r.note = "crossed; re-run has no reading"
            errors.append(f"{r.query} ({r.metric}) crossed but the re-run has no reading")
        else:
            r.status = "info"
            r.note = "crossed once; re-run did not repeat it"


# --- findings ----------------------------------------------------------------


def _badness(f: dict) -> float:
    """How bad a finding is, for 'worst first': a query that errors on the
    candidate only first, then the ratio, inverted for lower-is-worse metrics.
    A finding with no ratio sorts last."""
    if f["metric"] == "completed" and f["status"] == "regression":
        return math.inf
    r = f["ratio"]
    if r is None:
        return -math.inf
    if f["direction"] == "lower_is_worse":
        return math.inf if r == 0 else 1 / r
    return r


def row_finding(r: RowResult) -> dict:
    """The finding for one row, reporting readings that crossed the gate.

    The screen's readings, except that a confirmed regression reports whichever
    of its two runs crossed more weakly. Pooling the two runs' readings into one
    median could show a ratio under the threshold for a row that crossed it
    twice (e.g. 2.0x, then 1.5x, pooled to 1.004x).
    """
    cand, base = r.cand, r.base
    if r.status == "regression" and r.confirm_cand and r.confirm_base:
        second = ratio_of(r.confirm_cand, r.confirm_base)
        first = ratio_of(r.cand, r.base)
        if second is not None and (first is None or second < first):
            cand, base = r.confirm_cand, r.confirm_base
    mc = statistics.median(cand) if cand else None
    mb = statistics.median(base) if base else None
    ratio = mc / mb if mc is not None and mb else None
    name = f"{r.query} [{r.klass}]"
    return kit.finding(name, r.metric, mc, mb, r.status, threshold=r.threshold, ratio=ratio)


# The metrics that get per-class rows: where a small fixed per-query cost shows.
# The miss counters are too noisy for a median-per-row signal, and the contract
# allows 10 info findings; every class number is in raw/rows.csv either way.
CLASS_METRICS = ("instructions", "cycles", "alloc_bytes")


def class_findings(results: Sequence[RowResult], metrics: Sequence[MetricSpec]) -> list[dict]:
    """Median per-row ratio and sign count per (class, metric). Always `info`."""
    out = []
    for spec in metrics:
        if spec.name not in CLASS_METRICS:
            continue
        for klass in ("reads", "writes"):
            ratios = [
                rt
                for r in results
                if r.metric == spec.name and r.klass == klass and not r.note.startswith("errors")
                if (rt := r.ratio) is not None
            ]
            if not ratios:
                continue
            up = sum(1 for x in ratios if x > 1)
            down = sum(1 for x in ratios if x < 1)
            same = len(ratios) - up - down
            name = (
                f"{klass}: median per-row ratio over {len(ratios)} rows "
                f"({up} up, {down} down, {same} unchanged)"
            )
            out.append(
                kit.finding(name, spec.name, None, None, "info", ratio=statistics.median(ratios))
            )
    return out


def correctness_findings(failed: dict[str, set[str]], flaky: set[str] = frozenset()) -> list[dict]:
    """Queries that errored, as `completed` findings with kit.correctness_status:
    each candidate-only error is a regression at once; errors on the baseline
    only, on both sides, or on the candidate in some passes only are one
    finding each.

    `flaky` are queries that errored on the candidate in some passes only, while
    the baseline answered in every pass. Reported, not failed on: an error that
    does not repeat is what an A/A pair can also produce.
    """

    def completed(name: str, candidate: int | None, baseline: int) -> dict:
        status = kit.correctness_status(candidate, baseline)
        return kit.finding(
            name, "completed", candidate, baseline, status, direction="lower_is_worse"
        )

    out = [
        completed(f"{name} [errors on candidate only]", 0, 1)
        for name in sorted(failed["candidate"] - failed["baseline"])
    ]
    base_only = failed["baseline"] - failed["candidate"]
    both = failed["candidate"] & failed["baseline"]
    if flaky:
        name = (
            f"{len(flaky)} queries error on the candidate in some passes only "
            f"(e.g. {sorted(flaky)[0]})"
        )
        out.append(completed(name, None, 1))
    if base_only:
        name = f"{len(base_only)} queries error on the baseline only (e.g. {sorted(base_only)[0]})"
        out.append(completed(name, 1, 0))
    if both:
        out.append(
            completed(f"{len(both)} queries error on both sides (e.g. {sorted(both)[0]})", 0, 0)
        )
    return out


def ordered(
    regressions: list[dict],
    improvements: list[dict],
    classes: list[dict],
    notices: list[dict],
) -> list[dict]:
    """The findings in the order kit.select keeps them: regressions worst first
    (a query that errors on the candidate only first of all); the improvements
    that fit, largest gain kept, least gain listed first; then the 10 info
    slots, filled by `notices` (queries that error, rows with no reading, rows
    that crossed once and not on the second run) before the per-class rows, and
    listed by ratio, a row with none last."""
    room = max(MAX_FINDINGS - len(regressions), 0)
    gains = sorted(improvements, key=lambda f: f["ratio"] if f["ratio"] is not None else 1)[:room]
    quiet = (notices + classes)[: kit.MAX_INFORMATIVE]
    return [
        *sorted(regressions, key=_badness, reverse=True),
        *sorted(gains, key=_badness, reverse=True),
        *sorted(quiet, key=_badness, reverse=True),
    ]


# --- result.json -------------------------------------------------------------


def write_rows_csv(path: Path, results: Sequence[RowResult]) -> None:
    def fmt(vs: Sequence[float]) -> str:
        return ";".join(f"{v:.0f}" for v in vs)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "query",
                "class",
                "metric",
                "status",
                "ratio",
                "candidate_reps",
                "baseline_reps",
                "confirm_candidate_reps",
                "confirm_baseline_reps",
                "note",
            ]
        )
        for r in results:
            ratio = r.ratio
            w.writerow(
                [
                    r.query,
                    r.klass,
                    r.metric,
                    r.status,
                    "" if ratio is None else f"{ratio:.4f}",
                    fmt(r.cand),
                    fmt(r.base),
                    fmt(r.confirm_cand),
                    fmt(r.confirm_base),
                    r.note,
                ]
            )


def summary_line(
    verdict: str,
    findings: Sequence[dict],
    results: Sequence[RowResult],
    errors: Sequence[str],
    broken_on_both: int = 0,
    metrics: Sequence[MetricSpec] = HW_METRICS,
) -> str:
    counts = {
        s.name: sum(1 for r in results if r.metric == s.name and r.ratio is not None)
        for s in metrics
    }
    if len(set(counts.values())) == 1:
        n = next(iter(counts.values()))
        measured = f"{n} rows on {', '.join(s.name for s in metrics)}"
    else:
        measured = ", ".join(f"{counts[s.name]} {s.name} rows" for s in metrics)
    thresholds = {s.threshold for s in metrics}
    past = f"past {thresholds.pop():.2f}x" if len(thresholds) == 1 else "past its threshold"
    gaps = ""
    if errors:
        more = f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""
        gaps = f"could not measure: {errors[0]}{more}"
    both = f"; {broken_on_both} queries error on both images" if broken_on_both else ""
    if verdict == "error":
        return gaps + both
    regs = [f for f in findings if f["status"] == "regression"]
    imps = [f for f in findings if f["status"] == "improvement"]
    if regs:
        worst = max(regs, key=_badness)
        ratio = f"{worst['ratio']:.2f}x" if worst["ratio"] is not None else "n/a"
        # A regression outranks an error (contract v1.4), so a gap that came with
        # it is said here instead of in `error`.
        also = f"; also {gaps}" if gaps else ""
        return (
            f"{len(regs)} confirmed regression(s), worst {worst['name']} {worst['metric']} "
            f"{ratio}; {len(imps)} improvement(s); {measured}{both}{also}"
        )
    return f"no confirmed regression {past}; {len(imps)} improvement(s); {measured}{both}"


def build_result(
    inp: Inputs | None,
    args: Any,
    *,
    started: str,
    machine: dict | None = None,
    extra_errors: Sequence[str] = (),
    artifacts: Sequence[str] = (),
    metrics: tuple[MetricSpec, ...] = HW_METRICS,
) -> tuple[dict, list[RowResult]]:
    """The contract result (kit.result), and the per-row table behind it.

    `args` carries the contract flags (candidate, baseline, labels, commits,
    mode). `metrics` must match `inp.metrics`; it is separate so a run with no
    inputs at all still names the check's metrics in its summary.
    """
    errors = list(extra_errors)
    results: list[RowResult] = []
    findings: list[dict] = []
    broken_on_both = 0
    if inp is not None:
        results, assess_errors, failed = assess(inp)
        errors += assess_errors
        flaky = failed_sometimes(inp.screen["candidate"], [q.name for q in inp.queries])
        flaky -= failed_sometimes(inp.screen["baseline"], flaky) | failed["baseline"]
        correctness = correctness_findings(failed, flaky)
        broken_on_both = len(failed["candidate"] & failed["baseline"])
        by_status = {
            s: [f for f in correctness if f["status"] == s]
            for s in ("regression", "improvement", "info")
        }
        regressions = [row_finding(r) for r in results if r.status == "regression"]
        improvements = [row_finding(r) for r in results if r.status == "improvement"]
        unconfirmed = [row_finding(r) for r in results if r.crossed and r.status == "info"]
        unmeasured = [r for r in results if r.note.startswith("unmeasured")]
        notices = by_status["info"]
        if unmeasured:
            first = unmeasured[0]
            name = f"{len(unmeasured)} rows unmeasured (e.g. {first.query} [{first.metric}])"
            notices.append(kit.finding(name, first.metric, None, None, "info"))
        notices += sorted(unconfirmed, key=_badness, reverse=True)
        findings = ordered(
            regressions + by_status["regression"],
            improvements + by_status["improvement"],
            class_findings(results, inp.metrics),
            notices,
        )
    kept = kit.select(findings)
    all_errors = [*kit.input_errors(args), *errors]
    summary = summary_line(
        kit.verdict(kept, all_errors), kept, results, all_errors, broken_on_both, metrics
    )
    result = kit.result(
        CHECK,
        args,
        started=started,
        findings=findings,
        summary=summary,
        errors=errors,
        artifacts=artifacts,
        machine_info=machine,
    )
    return result, results
