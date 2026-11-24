"""micro-hw's verdict: two images' measure passes in, one contract `result.json` out.

Apart from reading the pass CSVs, the machine file and the clock, and writing
the result, it is pure, so every rule is covered by tests/test_verdict.py
without a server. What the check measures, its
rules and the basis of each gate: bench/README.md, "Nightly micro check".
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from pathlib import Path

from falkorbench.metrics import THRESHOLDS
from falkorbench.metrics import Row
from falkorbench.metrics import read_rows
from falkorbench.model import Query

CHECK = "micro-hw"
SIDES = ("candidate", "baseline")
REPS = 2

# The contract's limits on findings; tests/perf/validate.py checks the 50.
MAX_FINDINGS = 50
MAX_QUIET_FINDINGS = 10

# tests/perf/validate.py's rules, copied because bench runs without tests/perf;
# test_verdict checks the copy is exact. Used with fullmatch.
_PATH_PART = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
_HOST_PART = r"(?:[A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9-]*[A-Za-z0-9])"
IMAGE_RE = re.compile(
    rf"(?:(?:localhost|{_HOST_PART}(?:\.{_HOST_PART})+)(?::[0-9]+)?/|{_HOST_PART}:[0-9]+/)?"
    rf"{_PATH_PART}(?:/{_PATH_PART})*(?::[A-Za-z0-9_][A-Za-z0-9_.-]{{0,127}})?@sha256:[0-9a-f]{{64}}"
)
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


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
    """Where a pass's CSV lives. One convention, shared by `bench hw` (which
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

    Both directions need the reps not to overlap. Then the ratio must pass
    `threshold` with the medians more than `floor` apart, or pass `steep` (when
    given) with them more than `steep_floor` apart; every bound is strict. A
    zero baseline cannot give a ratio, so a row going from 0 to something is
    judged on `floor` or `steep_floor` alone.
    """
    if not cand or not base:
        return None
    mc, mb = statistics.median(cand), statistics.median(base)
    r = ratio_of(cand, base)
    up, down = mc - mb, mb - mc
    if r is None:
        # The baseline median is 0, so any growth is steeper than `steep`; it
        # still needs a positive absolute floor, or any non-zero reading flags.
        worse = (floor > 0 and up > floor) or (
            steep is not None and steep_floor > 0 and up > steep_floor
        )
        better = False
    else:
        worse = (r > threshold and up > floor) or (
            steep is not None and r > steep and up > steep_floor
        )
        better = (r < 1 / threshold and down > floor) or (
            steep is not None and r < 1 / steep and down > steep_floor
        )
    if worse and min(cand) > max(base):
        return "regression"
    if better and max(cand) < min(base):
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


def _num(v: float | None) -> float | None:
    if v is None or not math.isfinite(v):
        return None
    return round(v, 6) if abs(v) < 1e6 else round(v, 1)


def _finding(
    name: str,
    metric: str,
    cand: float | None,
    base: float | None,
    ratio: float | None,
    threshold: float | None,
    status: str,
    direction: str = "higher_is_worse",
) -> dict:
    return {
        "name": name,
        "metric": metric,
        "candidate": _num(cand),
        "baseline": _num(base),
        "ratio": _num(ratio),
        "threshold": _num(threshold),
        "direction": direction,
        "status": status,
    }


def _badness(f: dict) -> float:
    """How bad a finding is, for 'worst first': the ratio, inverted for
    lower-is-worse metrics. A finding with no ratio sorts last."""
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
    return _finding(f"{r.query} [{r.klass}]", r.metric, mc, mb, ratio, r.threshold, r.status)


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
            out.append(
                _finding(
                    f"{klass}: median per-row ratio over {len(ratios)} rows "
                    f"({up} up, {down} down, {same} unchanged)",
                    spec.name,
                    None,
                    None,
                    statistics.median(ratios),
                    None,
                    "info",
                )
            )
    return out


def correctness_findings(
    failed: dict[str, set[str]], flaky: set[str] = frozenset()
) -> tuple[list[dict], list[dict]]:
    """(regressions, info): candidate-only errors fail the run at once.

    `flaky` are queries that errored on the candidate in some passes only, while
    the baseline answered in every pass. Reported, not failed on: an error that
    does not repeat is what an A/A pair can also produce.
    """
    regress = [
        _finding(
            f"{name} [errors on candidate only]",
            "completed",
            0,
            1,
            0,
            None,
            "regression",
            "lower_is_worse",
        )
        for name in sorted(failed["candidate"] - failed["baseline"])
    ]
    info = []
    base_only = failed["baseline"] - failed["candidate"]
    both = failed["candidate"] & failed["baseline"]
    if flaky:
        info.append(
            _finding(
                f"{len(flaky)} queries error on the candidate in some passes only "
                f"(e.g. {sorted(flaky)[0]})",
                "completed",
                None,
                1,
                None,
                None,
                "info",
                "lower_is_worse",
            )
        )
    if base_only:
        info.append(
            _finding(
                f"{len(base_only)} queries error on the baseline only "
                f"(e.g. {sorted(base_only)[0]})",
                "completed",
                1,
                0,
                None,
                None,
                "info",
                "lower_is_worse",
            )
        )
    if both:
        info.append(
            _finding(
                f"{len(both)} queries error on both sides (e.g. {sorted(both)[0]})",
                "completed",
                0,
                0,
                None,
                None,
                "info",
                "lower_is_worse",
            )
        )
    return regress, info


def select_findings(
    regressions: list[dict],
    improvements: list[dict],
    classes: list[dict],
    notices: list[dict],
) -> list[dict]:
    """At most 50, at most 10 ok/info, regressions kept before improvements,
    then ordered worst first.

    Of the 10 ok/info slots, `notices` (queries that error, rows with no
    reading, rows that crossed once and not on the second run) come first, in
    the order given; the per-class rows fill what is left.
    """
    quiet = (notices + classes)[:MAX_QUIET_FINDINGS]
    room = MAX_FINDINGS - len(quiet)
    loud = sorted(regressions, key=_badness, reverse=True)[:room]
    room -= len(loud)
    # Improvements: the largest gain first, so the cut drops the smallest.
    loud += sorted(improvements, key=lambda f: f["ratio"] if f["ratio"] is not None else 1)[:room]
    return sorted([*loud, *quiet], key=lambda f: (f["status"] in ("ok", "info"), -_badness(f)))


# --- result.json -------------------------------------------------------------


@dataclass
class ImageRef:
    image: str
    label: str | None = None
    commit: str | None = None

    def as_json(self) -> dict:
        # A malformed commit is reported in `error` (input_errors) and written as
        # null, so the result stays valid; a blank label falls back to the default.
        label = (self.label or "").strip() or default_label(self.image)
        commit = self.commit if self.commit and COMMIT_RE.fullmatch(self.commit) else None
        return {"image": self.image, "commit": commit, "label": label}


def default_label(image: str) -> str:
    """`<repo>@<first 12 hex of the digest>` (contract v1.2)."""
    name, _, digest = image.partition("@")
    repo = name.rsplit(":", 1)[0] if ":" in name.rsplit("/", 1)[-1] else name
    hexpart = digest.removeprefix("sha256:")[:12]
    return f"{repo}@{hexpart}" if hexpart else repo


def input_errors(candidate: ImageRef, baseline: ImageRef, mode: str) -> list[str]:
    errs = []
    for side, ref in (("candidate", candidate), ("baseline", baseline)):
        if not IMAGE_RE.fullmatch(ref.image):
            errs.append(f"{side} image {ref.image!r} is not name[:tag]@sha256:<64 hex>, a digest")
        if ref.commit and not COMMIT_RE.fullmatch(ref.commit):
            errs.append(f"{side} commit {ref.commit!r} is not 40 lowercase hex")
    if mode not in ("queue", "nightly", "long"):
        errs.append(f"mode {mode!r} is not queue, nightly or long")
    return errs


def now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def vm_machine() -> dict | None:
    """The `machine` object `_perf-vm.yml` writes to $PERF_MACHINE_JSON, or None
    when that is unset, unreadable or not a machine object."""
    env = os.environ.get("PERF_MACHINE_JSON")
    if not env:
        return None
    try:
        m = json.loads(Path(env).read_text())
    except (OSError, ValueError):
        # A file that cannot be read or parsed describes no machine. The caller
        # then describes the host it can see, rather than failing a result over
        # its machine field.
        return None
    return m if isinstance(m, dict) and m.get("runner") else None


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
    *,
    candidate: ImageRef,
    baseline: ImageRef,
    mode: str,
    started: str,
    machine: dict,
    finished: str | None = None,
    extra_errors: Sequence[str] = (),
    artifacts: Sequence[str] = (),
    check: str = CHECK,
    metrics: tuple[MetricSpec, ...] = HW_METRICS,
) -> tuple[dict, list[RowResult]]:
    """The contract-v1 result, and the per-row table behind it.

    `metrics` must match `inp.metrics`; it is separate so a run with no inputs
    at all still names the check's metrics in its summary.
    """
    errors = [*input_errors(candidate, baseline, mode), *extra_errors]
    results: list[RowResult] = []
    findings: list[dict] = []
    broken_on_both = 0
    if inp is not None:
        results, assess_errors, failed = assess(inp)
        errors += assess_errors
        flaky = failed_sometimes(inp.screen["candidate"], [q.name for q in inp.queries])
        flaky -= failed_sometimes(inp.screen["baseline"], flaky) | failed["baseline"]
        regress_c, info_c = correctness_findings(failed, flaky)
        broken_on_both = len(failed["candidate"] & failed["baseline"])
        regressions = [row_finding(r) for r in results if r.status == "regression"] + regress_c
        improvements = [row_finding(r) for r in results if r.status == "improvement"]
        unconfirmed = [row_finding(r) for r in results if r.crossed and r.status == "info"]
        unmeasured = [r for r in results if r.note.startswith("unmeasured")]
        notices = list(info_c)
        if unmeasured:
            notices.append(
                _finding(
                    f"{len(unmeasured)} rows unmeasured (e.g. {unmeasured[0].query} "
                    f"[{unmeasured[0].metric}])",
                    unmeasured[0].metric,
                    None,
                    None,
                    None,
                    None,
                    "info",
                )
            )
        notices += sorted(unconfirmed, key=_badness, reverse=True)
        findings = select_findings(
            regressions, improvements, class_findings(results, inp.metrics), notices
        )

    # A confirmed regression outranks an error (contract v1.4): the night is red
    # either way, and `error` may not carry a regression finding.
    if any(f["status"] == "regression" for f in findings):
        verdict = "regression"
    elif errors:
        verdict = "error"
    else:
        verdict = "pass"

    result = {
        "schema": 1,
        "check": check,
        "verdict": verdict,
        "confirmed": verdict == "regression",
        "summary": summary_line(
            verdict, findings, results, errors, broken_on_both, metrics
        ).replace("\n", " "),
        "candidate": candidate.as_json(),
        "baseline": baseline.as_json(),
        "machine": machine,
        "started": started,
        # Never before `started`: that comes from preflight's runner clock.
        "finished": max(finished or now_utc(), started),
        "findings": findings,
        "error": "; ".join(errors).replace("\n", " ") if verdict == "error" else None,
        "artifacts": list(artifacts),
    }
    return result, results


def dump(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
