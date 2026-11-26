"""`bench soak`: the release soak as a nightly perf check (results schema 1). Each scenario
runs once per side on one machine, compared into <out>/result.json. Rules, gates and their
evidence: README.md ("Nightly check")."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from falkorbench import kit

HERE = Path(__file__).resolve().parent
SCENARIOS_DIR = HERE / "scenarios"

# --mode sets only the length. deadline_min stops starting runs when the check runs by hand; in
# the workflows the deadline follows the check's own budget instead (budget_deadline_min).
MODES = {
    "queue": {
        "duration": 60,
        "scale": 0.05,
        "perf_gate": False,
        "reruns": False,
        "deadline_min": 50,
    },
    "nightly": {
        "duration": 300,
        "scale": 0.25,
        "perf_gate": True,
        "reruns": True,
        "deadline_min": 110,
    },
    "long": {
        "duration": 1200,
        "scale": 1.0,
        "perf_gate": True,
        "reruns": True,
        "deadline_min": 320,
    },
}

# Gates sit above the worst main-vs-main swing measured (README); crossing is strict.
P50_RATIO, P50_FLOOR_MS = 2.0, 2.0
P99_INFO_RATIO, P99_INFO_FLOOR_MS, MIN_P99_SAMPLES = 2.0, 10.0, 1000  # p99: reported, not gated
TPUT_RATIO = 0.7
SETUP_RATIO, SETUP_FLOOR_S = 1.5, 5.0
MIN_SAMPLES = 200  # baseline ops before an op is compared at all
MIN_CAND_SAMPLES = 20  # candidate ops before its latency is (throughput always is)

# Latency/resource checks: broken only when every run pair repeats it.
TIMING_RE = re.compile(
    r"\bp99\b|\bSLO\b|\bschedule\b|\bplateaus?\b|\bRSS\b|\bused_memory\b|memory growth|\bdrift\b|"
    r"within time budget|replication lag|fires promptly|stays fast|CPU idle|churn volume|"
    r"concurrent connections were reached|no marker timeouts",
    re.I,
)

# Contract v1.9: check name -> open issue. Entry rule and evidence: README "Known flaky checks".
# A listed check broken on the candidate is still a regression (#4065); only the release gate
# leaves it out, unless the candidate broke it every time.
KNOWN_FLAKY = {
    "s10 Live (under writes) index == scan during build/after": "#3095",
    "s10 Live: index-path count == scan-path count (same query)": "#3095",
    "s10 during build person_age: result == ground truth": "#3512",
    "s10 during build person_city: result == ground truth": "#3512",
}

# Known flaky checks the scenario records only when they fail: every record is a failure, so a
# "k of n" with k == n says nothing about how often it held, and a total breakage cannot be told
# from a random one. The release gate never counts these (README "Known flaky checks").
FAILURE_ONLY = frozenset(
    {
        "s10 during build person_age: result == ground truth",
        "s10 during build person_city: result == ground truth",
    }
)

# A known flaky check the candidate broke every time it was checked, at least this many times
# in every run, is a total breakage: the release gate counts it (README "Known flaky checks").
ALWAYS_BROKEN_MIN_CHECKS = 10

# The open issue of a pass/fail check that breaks today, for the report only: it never changes a
# verdict. README "Known breaks" is the same map with its evidence (a test keeps the two equal);
# a check not listed has no issue known.
ISSUES = {
    **{name: int(issue.lstrip("#")) for name, issue in KNOWN_FLAKY.items()},
    "s03 range index finds chunks by cid (0, 1, mid, last)": 3512,
    "s03 range index finds every initial chunk": 3512,
    "s03 in-load: updated_ann_miss_top10 == 0": 3886,
    "s05 used_memory returns to baseline after deleting all (residual <50MB)": 4050,
    "s10 after build person_age: indexed result == ground truth": 3512,
    "s10 after build person_city: indexed result == ground truth": 3512,
    "s10 during build person_age: result == ground truth": 3512,
    "s10 during build person_city: result == ground truth": 3512,
    "s10 before restart: Person.age indexed results == ground truth": 3512,
    "s11 CLIENT KILLed query is cancelled (CPU idle within 5s of the kill)": 4051,
    "s11 deeply nested expressions are answered within 60 s": 2400,
}

# Server-log lines that mean the server process crashed (not just a failed query).
CRASH_RE = re.compile(
    r"crashed by signal|panicked at|=== (?:REDIS|VALKEY) BUG REPORT|"
    r"Segmentation fault|ASSERTION FAILED|Guru Meditation",
    re.I,
)


def is_timing_check(name: str) -> bool:
    return bool(TIMING_RE.search(name))


# ------------------------------------------------------------------ one run's record
@dataclass
class Run:
    """What one scenario run on one side left behind (raw/<scn>/<side>-<rep>/)."""

    scenario: str
    side: str  # candidate | baseline
    rep: int
    completed: bool
    reason: str = ""  # why not completed
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)  # check -> its first failing detail
    ops: dict[str, dict] = field(default_factory=dict)  # op -> {p50_ms, p99_ms, throughput, ok}
    setup_s: float | None = None
    wall_s: float = 0.0
    server_ran: bool = True  # monitor.csv saw a server alive
    rc: int | None = 0  # None: stopped by a cap
    cut: bool = False  # stopped by the deadline, not a hang: does not count

    @property
    def measured(self) -> bool:
        return self.server_ran and bool(self.checks)

    @property
    def harness_end(self) -> bool:
        """Killed by a signal (OOM killer, runner shutdown) or exited without its summary."""
        return (self.rc is not None and self.rc < 0) or self.reason == "no summary.json"


def checks_by_name(items: list[dict]) -> dict[str, bool]:
    """A name recorded several times held only if every one held."""
    out: dict[str, bool] = {}
    for c in items:
        out[c["name"]] = out.get(c["name"], True) and bool(c["ok"])
    return out


def ops_of(summary: dict) -> dict[str, dict]:
    """Worker ops (`_`-ops are the harness's bookkeeping) with their successful count."""

    def finite(x):
        return x if x is not None and math.isfinite(x) else None

    out = {}
    for op, d in (summary.get("ops") or {}).items():
        if not op.startswith("_"):
            out[op] = {
                "p50_ms": finite(d.get("p50_ms")),
                "p99_ms": finite(d.get("p99_ms")),
                "throughput": d.get("throughput_ops_s") or 0.0,
                "ok": (d.get("count") or 0) - (d.get("errors") or 0),
            }
    return out


def server_ran(run_dir: Path) -> bool:
    for mon in run_dir.rglob("monitor.csv"):
        for line in mon.read_text(errors="replace").splitlines()[1:]:
            parts = line.split(",")
            if len(parts) > 6 and parts[6] == "1":
                return True
    return False


def setup_seconds(phases: list[dict]) -> float | None:
    """Wall time of the scenario's first "ingest" phase."""
    for i, ev in enumerate(phases):
        if ev.get("phase") == "ingest" and i + 1 < len(phases):
            return round(float(phases[i + 1]["t"]) - float(ev["t"]), 1)
    return None


def crash_lines(run_dir: Path) -> list[str]:
    bad = []
    for log in sorted(run_dir.rglob("*.log")):
        if log.name != "stdout.log":
            bad += [
                f"{log.relative_to(run_dir)}: {line[:200]}"
                for line in log.read_text(errors="replace").splitlines()
                if CRASH_RE.search(line)
            ]
    return bad[:20]


def read_run(
    run_dir: Path,
    scenario: str,
    side: str,
    rep: int,
    rc: int | None,
    timed_out: bool,
    wall_s: float,
    cut: bool = False,
) -> Run:
    summary_p, checks_p, phases_p = (
        run_dir / n for n in ("summary.json", "checks.json", "phases.json")
    )
    summary = json.loads(summary_p.read_text()) if summary_p.exists() else None
    checks = json.loads(checks_p.read_text()) if checks_p.exists() else []
    phases = json.loads(phases_p.read_text()) if phases_p.exists() else []
    crashes = crash_lines(run_dir)
    reason = (
        "stopped at the deadline"
        if cut
        else "timed out"
        if timed_out
        else f"exit {rc}"
        if rc
        else "no summary.json"
        if summary is None
        else f"server crash: {crashes[0]}"
        if crashes
        else ""
    )
    return Run(
        scenario=scenario,
        side=side,
        rep=rep,
        completed=not reason,
        reason=reason,
        checks=checks_by_name(checks),
        ops=ops_of(summary or {}),
        details={c["name"]: c.get("detail", "") for c in checks if not c["ok"]},
        setup_s=setup_seconds(phases),
        wall_s=round(wall_s, 1),
        server_ran=server_ran(run_dir),
        cut=cut,
        rc=rc,
    )


# ------------------------------------------------------------------ comparison
@dataclass
class Row:
    """One comparison. `kind` says why a row is info: unconfirmed (a perf/timing signal not
    confirmed by its re-run), incomplete (its re-run did not finish), tail (p99 shift); known
    marks a KNOWN_FLAKY check. `introduced`, on a pass/fail row, is the release gate's answer
    when it differs from kit.introduced(). ok rows stay in raw/rows.csv only."""

    name: str
    metric: str
    candidate: float | None
    baseline: float | None
    ratio: float | None
    threshold: float | None
    direction: str
    status: str
    crossed: bool = False  # crossed in the first pass (asks for a re-run)
    kind: str = ""
    note: str = ""
    introduced: bool | None = None

    def finding(self) -> dict:
        f = kit.finding(
            self.name,
            self.metric,
            self.candidate,
            self.baseline,
            self.status,
            direction=self.direction,
            threshold=self.threshold,
            ratio=self.ratio,
        )
        if self.introduced is not None and self.status == "regression":
            f[kit.INTRODUCED] = self.introduced
        return f


def _ratio(c, b):
    return None if c is None or b is None or not b else c / b


def _zero_one(name, metric, c, b) -> Row:
    """A 1-held / 0-broken pair, judged by kit.correctness_status (contract v1.4)."""
    return Row(name, metric, c, b, None, None, "lower_is_worse", kit.correctness_status(c, b))


def _correctness_value(runs: list[Run], name: str) -> int | None:
    vals = [r.checks[name] for r in runs if name in r.checks]
    return None if not vals else int(all(vals))


_K_OF_N = re.compile(r"^(\d+) of (\d+) runs failed")


def _always_broken(runs: list[Run], name: str) -> bool:
    """Every run broke it every time it was checked (k == n >= ALWAYS_BROKEN_MIN_CHECKS); no "k of n" counts as
    always broken, so a total breakage is gated rather than hidden."""
    if not runs or not all(name in r.checks for r in runs):
        return False
    for r in runs:
        if r.checks[name]:
            return False
        m = _K_OF_N.match(r.details.get(name, ""))
        if m and not (int(m.group(1)) == int(m.group(2)) >= ALWAYS_BROKEN_MIN_CHECKS):
            return False
    return True


def _timing_row(tag: str, name: str, cand: list[Run], base: list[Run], perf_gate: bool) -> Row:
    """Broken on the candidate only when every pair (at least two: the first run asks for a
    re-run) has it broken there, whatever the baseline did; the baseline broke it too when it
    did in any pair. The release gate counts it only when every pair has it held on the
    baseline. Otherwise a side counts as broken only if every run (at least two) broke it."""
    pairs = [
        (c.checks.get(name), b.checks.get(name)) for c, b in zip(cand, base, strict=False)
    ]  # re-runs come in pairs
    c_seen = [cv for cv, _ in pairs if cv is not None]
    b_seen = [bv for _, bv in pairs if bv is not None]
    c_broke = sum(1 for v in c_seen if v is False)
    confirmed = perf_gate and len(pairs) >= 2 and all(cv is False for cv, _ in pairs)
    if confirmed:
        cv = 0
        bv = 0 if any(bv is False for _, bv in pairs) else (1 if b_seen else None)
    else:
        cv = 1 if c_seen else None
        bv = None if not b_seen else int(not (len(b_seen) >= 2 and all(v is False for v in b_seen)))
    row = _zero_one(f"{tag} {name}", "invariant", cv, bv)
    row.crossed = bool(pairs) and pairs[0][0] is False
    if confirmed:
        row.introduced = all(bv is True for _, bv in pairs)
    if cv == 1 and c_broke:
        row.status, row.kind = "info", "unconfirmed"
        row.note = f"broke in {c_broke} of {len(c_seen)} candidate runs"
        row.name += f" ({row.note})"
    return row


def scenario_rows(scn: str, cand: list[Run], base: list[Run], perf_gate: bool) -> list[Row]:
    """Every comparison for one scenario; rep 1 first, then any re-run."""
    tag = scn.split("_")[0]
    rows = [
        _zero_one(
            f"{tag} completed",
            "completed",
            int(all(r.completed for r in cand)) if cand else None,
            int(all(r.completed for r in base)) if base else None,
        )
    ]
    for n in sorted({n for r in cand + base for n in r.checks}):
        if is_timing_check(n):
            rows.append(_timing_row(tag, n, cand, base, perf_gate))
            continue
        row = _zero_one(
            f"{tag} {n}", "invariant", _correctness_value(cand, n), _correctness_value(base, n)
        )
        if row.name in KNOWN_FLAKY:
            row.kind = "known"
            row.note = "broke on " + (
                " and ".join(
                    side
                    for side, v in (("candidate", row.candidate), ("baseline", row.baseline))
                    if v == 0
                )
                or "neither side"
            )
            if row.status == "regression":  # the gate counts it only when it always broke
                row.introduced = (
                    row.baseline != 0 and row.name not in FAILURE_ONLY and _always_broken(cand, n)
                )
            elif row.status == "improvement":  # a flake the baseline hit: not a fix
                row.status = "info"
        rows.append(row)
    return rows + _perf_rows(tag, cand, base, perf_gate)


def _perf_crossings(c_run: Run, b_run: Run) -> dict[tuple[str, str], tuple]:
    """(op, metric) -> (candidate, baseline, ratio, threshold, direction, move, tail); move is
    kit.crossed's +1 worse / -1 better / 0, always 0 for p99 (tail says it shifted)."""
    out = {}
    none = {"p50_ms": None, "p99_ms": None, "throughput": 0.0, "ok": 0}
    # gated on the baseline's count: an op the candidate nearly stopped doing still compares
    for op in sorted(o for o, b in b_run.ops.items() if b["ok"] >= MIN_SAMPLES):
        c, b = c_run.ops.get(op, none), b_run.ops[op]
        if c["ok"] >= MIN_CAND_SAMPLES and _ratio(c["p50_ms"], b["p50_ms"]) is not None:
            move = kit.crossed(c["p50_ms"], b["p50_ms"], P50_RATIO, P50_FLOOR_MS)
            out[(op, "p50_ms")] = (
                c["p50_ms"],
                b["p50_ms"],
                c["p50_ms"] / b["p50_ms"],
                P50_RATIO,
                "higher_is_worse",
                move,
                False,
            )
        if (
            c["ok"] >= MIN_CAND_SAMPLES
            and b["ok"] >= MIN_P99_SAMPLES
            and _ratio(c["p99_ms"], b["p99_ms"])
        ):
            shift = kit.crossed(c["p99_ms"], b["p99_ms"], P99_INFO_RATIO, P99_INFO_FLOOR_MS)
            out[(op, "p99_ms")] = (
                c["p99_ms"],
                b["p99_ms"],
                c["p99_ms"] / b["p99_ms"],
                None,
                "higher_is_worse",
                0,
                shift != 0,
            )
        r = _ratio(c["throughput"], b["throughput"])
        if r is not None:  # lower is worse: crossing 1/0.7 the other way up is under 0.7x
            move = kit.crossed(
                c["throughput"], b["throughput"], 1 / TPUT_RATIO, 0.0, "lower_is_worse"
            )
            out[(op, "throughput")] = (
                c["throughput"],
                b["throughput"],
                r,
                TPUT_RATIO,
                "lower_is_worse",
                move,
                False,
            )
    if c_run.setup_s is not None and b_run.setup_s:
        move = kit.crossed(c_run.setup_s, b_run.setup_s, SETUP_RATIO, SETUP_FLOOR_S)
        out[("setup wall time", "p50_ms")] = (
            c_run.setup_s * 1000,
            b_run.setup_s * 1000,
            c_run.setup_s / b_run.setup_s,
            SETUP_RATIO,
            "higher_is_worse",
            move,
            False,
        )
    return out


def _perf_rows(tag: str, cand: list[Run], base: list[Run], perf_gate: bool) -> list[Row]:
    if not cand or not base or not cand[0].completed or not base[0].completed:
        return []
    first = _perf_crossings(cand[0], base[0])
    rerun = len(cand) > 1 and len(base) > 1
    again = (
        _perf_crossings(cand[1], base[1])
        if rerun and cand[1].completed and base[1].completed
        else None
    )
    rows = []
    for (op, metric), (c, b, r, thr, direction, move, tail) in first.items():
        reg = move > 0
        row = Row(f"{tag} {op}", metric, c, b, r, thr, direction, "ok", crossed=reg)
        if tail:
            row.status, row.kind = "info", "tail"
            row.name += " (p99 shift, not gated)"
        if reg:
            rep = again.get((op, metric)) if again is not None else None
            if not perf_gate:
                row.status, row.note = "info", "perf is informational in this mode"
            elif rep is not None and rep[5] > 0:
                if rep[2] > 0 and (r <= 0 or abs(math.log(rep[2])) < abs(math.log(r))):
                    row.candidate, row.baseline, row.ratio = (
                        rep[0],
                        rep[1],
                        rep[2],
                    )  # the milder one
                row.status = "regression"
            else:
                row.status = "info"
                row.kind = "incomplete" if rerun and rep is None else "unconfirmed"
                row.note = (
                    "not re-run"
                    if not rerun
                    else "re-run incomplete"
                    if rep is None
                    else "not repeated on re-run"
                )
                row.name += f" ({row.note})"
        elif move < 0:
            row.status = "improvement"
        rows.append(row)
    return rows


def move_size(r: Row) -> float:
    """How far a row moved, for ordering findings and re-runs: |log ratio|. A 0/1 row held
    on both sides (a timing check that broke in one run only) did not move: 0, after every
    perf move. Any other 0/1 row, and a perf value that fell to zero, moved all the way."""
    if r.metric in ("invariant", "completed"):
        return 0.0 if r.candidate == r.baseline == 1 else math.inf
    return abs(math.log(r.ratio)) if r.ratio and r.ratio > 0 else math.inf


def worst_first(rows: list[Row]) -> list[dict]:
    """The findings result.json can carry, in the order kit.select() keeps within a status:
    regressions and improvements by the size of the move, info rows unconfirmed first, then
    the rest, then p99 shifts. ok rows stay in raw/rows.csv."""
    size = move_size
    rank = {"unconfirmed": 0, "incomplete": 0, "tail": 2}

    def info_rank(r: Row):
        return (rank.get(r.kind, 1), -size(r))

    reg = sorted((r for r in rows if r.status == "regression"), key=size, reverse=True)
    imp = sorted((r for r in rows if r.status == "improvement"), key=size, reverse=True)
    info = sorted((r for r in rows if r.status == "info"), key=info_rank)
    return [r.finding() for r in reg + imp + info]


def _short(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


PASS_FAIL = kit.PASS_FAIL_METRICS


def broken_note(r: Row) -> str:
    """How the report names a broken pass/fail check: baseline too, its issue, known flaky."""
    issue = ISSUES.get(r.name)
    return ", ".join(
        part
        for part in (
            f"baseline too: {'yes' if r.baseline == 0 else 'no'}",
            f"#{issue}" if issue else "",
            "known flaky" if r.kind == "known" else "",
        )
        if part
    )


def summary_line(
    rows: list[Row],
    scenarios: list[str],
    not_run: list[str],
    mode: str,
    duration: float,
    error: str | None,
    unconfirmed: list[str] = (),
) -> str:
    reg = [r for r in rows if r.status == "regression" and r.metric not in PASS_FAIL]
    broken = [r for r in rows if r.status == "regression" and r.metric in PASS_FAIL]
    fixed = [r for r in rows if r.status == "improvement" and r.metric in PASS_FAIL]
    unconf = [
        r for r in rows if r.kind in ("unconfirmed", "incomplete") and r.metric not in PASS_FAIL
    ]
    timing = [r for r in rows if r.kind == "unconfirmed" and r.metric in PASS_FAIL]
    tails = [r for r in rows if r.kind == "tail"]
    known = [r for r in rows if r.kind == "known" and r.status == "info" and r.baseline == 0]
    missing = {n.split(" ")[0] for n in not_run}
    parts = [
        f"{len([s for s in scenarios if s not in missing])} of {len(scenarios)} scenarios "
        f"measured, {mode} ({duration:.0f} s per side)"
    ]
    # What went wrong first: the summary is cut at 500 characters, and with a regression
    # verdict it is the only place that says it (result.json's error is null then).
    if error:
        parts.append(error)
    if broken:
        names = [f"{_short(r.name, 80)} ({broken_note(r)})" for r in broken[:4]]
        parts.append(
            f"{len(broken)} pass/fail check(s) broken on the candidate: "
            + ", ".join(names)
            + (" …" if len(broken) > 4 else "")
        )
    if reg:
        names = [f"{_short(r.name, 80)}" + (f" {r.ratio:.2f}x" if r.ratio else "") for r in reg[:4]]
        parts.append(
            f"{len(reg)} perf regression(s): " + ", ".join(names) + (" …" if len(reg) > 4 else "")
        )
    if not broken and not reg:
        parts.append("no regression")
    if not_run:
        parts.append("not measured: " + ", ".join(not_run))
    if unconfirmed:
        parts.append("unconfirmed: " + ", ".join(unconfirmed))
    if timing:
        parts.append(
            "broken on the candidate, not confirmed by a re-run: "
            + ", ".join(_short(r.name, 100) for r in timing)
        )
    if fixed:
        parts.append(
            "fixed on the candidate (the baseline breaks it): "
            + ", ".join(_short(r.name, 80) for r in fixed)
        )
    if unconf:
        parts.append(f"{len(unconf)} perf signal(s) not confirmed")
    if known:
        parts.append(
            "known flaky, broke on the baseline only: "
            + ", ".join(f"{r.name} ({KNOWN_FLAKY[r.name]})" for r in known)
        )
    if tails:
        top = max(tails, key=lambda r: abs(math.log(r.ratio)) if r.ratio and r.ratio > 0 else 0)
        parts.append(
            f"{len(tails)} p99 shift(s) reported, not gated "
            f"(largest {top.name.split(' (')[0]} {top.ratio:.2f}x)"
        )
    return "; ".join(parts)  # kit.result() makes it one line


# ------------------------------------------------------------------ raw/
def write_rows(out: Path, rows: list[Row]) -> None:
    with open(out / "raw" / "rows.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "name",
                "metric",
                "candidate",
                "baseline",
                "ratio",
                "threshold",
                "direction",
                "status",
                "crossed_first_pass",
                "kind",
                "note",
            ]
        )
        for r in rows:
            d = r.finding()
            w.writerow(
                [
                    d["name"],
                    d["metric"],
                    d["candidate"],
                    d["baseline"],
                    d["ratio"],
                    d["threshold"],
                    d["direction"],
                    d["status"],
                    int(r.crossed),
                    r.kind,
                    r.note,
                ]
            )


# ------------------------------------------------------------------ running
def free_port_block(start: int, width: int = 10) -> int:
    """First port p >= start with p..p+width-1 all free on 127.0.0.1 (never 6379)."""
    p = start
    while p < 65000:
        if not (p <= 6379 < p + width) and all(_free(q) for q in range(p, p + width)):
            return p
        p += width
    raise RuntimeError("no free port block")


def _free(port: int) -> bool:
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def compress_large(run_dir: Path, min_bytes: int = 1 << 20) -> None:
    """gzip a run's big files (ledgers, latency histograms) once it has been read."""
    for f in run_dir.rglob("*"):
        if f.is_file() and f.suffix != ".gz" and f.stat().st_size >= min_bytes:
            with open(f, "rb") as src, gzip.open(f.with_name(f.name + ".gz"), "wb") as dst:
                shutil.copyfileobj(src, dst)
            f.unlink()


# The check stops starting runs this long before its command's timeout, and run_all cuts a run 5
# min before it, which leaves the rest to compare and write result.json.
DEADLINE_MARGIN_MIN = 10


def budget_deadline_min(environ=os.environ) -> float | None:
    """The deadline inside the check's own budget: PERF_TIMEOUT_SECONDS, which
    tests/perf/vm/run-check.sh sets from its checks.yaml entry's timeout_minutes, less the margin.
    None outside the workflows (unset, or not a positive number of seconds)."""
    try:
        seconds = float(environ.get("PERF_TIMEOUT_SECONDS", ""))
    except ValueError:
        return None
    if not (math.isfinite(seconds) and seconds > 0):
        return None
    return max(seconds / 60 - DEADLINE_MARGIN_MIN, 0.0)


def mode_params(a: argparse.Namespace, environ=os.environ) -> dict:
    """The mode's settings with the development overrides (--duration, --scale, --deadline-minutes).
    The deadline is the budget's when the workflow gives one, so a one-scenario entry's re-run fits
    its own timeout; --deadline-minutes still wins."""
    params = dict(MODES.get(a.mode, MODES["queue"]))
    budget = budget_deadline_min(environ)
    if budget is not None:
        params["deadline_min"] = budget
    for key, value in (
        ("duration", a.duration),
        ("scale", a.scale),
        ("deadline_min", a.deadline_minutes),
    ):
        if value:
            params[key] = value
    return params


def remove_run_containers(label: str, attempts: int = 3, delay_s: float = 5.0) -> None:
    """kit.remove_labelled, tried again on a docker error: a container that is exiting under
    --rm answers "removal already in progress" for a moment. Still failing, it raises: the
    next run must not share the machine with a server nobody removed."""
    for attempt in range(1, attempts + 1):
        try:
            kit.remove_labelled(label)
            return
        except kit.DockerError as e:
            if attempt == attempts:
                raise
            print(f"removing {label} failed ({e}); trying again in {delay_s:.0f} s", flush=True)
            time.sleep(delay_s)


class Soak:
    def __init__(self, a: argparse.Namespace):
        self.raw = Path(a.out).resolve() / "raw"
        self.params = mode_params(a)
        self.run_id = f"{os.getpid()}-{int(time.time())}"
        self.proc: subprocess.Popen | None = None
        self.runs_csv: list[dict] = []
        self.port = 21000
        self.launched = False  # a scenario was started: its containers may need removing

    def run_one(
        self, scn: str, side: str, image: str, rep: int, hang_cap_s: float, stop_at: float
    ) -> Run:
        """One run, capped at `hang_cap_s` (past it: did not finish) and at the absolute time
        `stop_at` (stopped earlier: `cut`, does not count). A side whose server never came up
        is tried once more."""
        run_dir = self.raw / scn / f"{side}-{rep}"
        for attempt in (1, 2):
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True)
            label = f"{self.run_id}-{scn}-{side}-{rep}"
            cap_s = max(60.0, min(hang_cap_s, stop_at - time.time()))
            self.port = free_port_block(self.port)
            cmd = [
                sys.executable,
                "-I",
                str(SCENARIOS_DIR / f"{scn}.py"),
                "--image",
                image,
                "--port",
                str(self.port),
                "--out",
                str(run_dir),
                "--duration",
                str(self.params["duration"]),
                "--scale",
                str(self.params["scale"]),
            ]
            env = dict(os.environ, SOAK_RUN_ID=label, SOAK_CHECK_ID=self.run_id)
            t0 = time.time()
            rc, timed_out = None, False
            print(f"{kit.utc_now()} {scn} {side} rep {rep}: {' '.join(cmd[2:])}", flush=True)
            try:
                self.launched = True
                with open(run_dir / "stdout.log", "w") as log:
                    self.proc = subprocess.Popen(
                        cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
                    )
                    try:
                        rc = self.proc.wait(cap_s)
                    except subprocess.TimeoutExpired:
                        timed_out = True
            finally:  # also on kit.Stopped: the scenario and its servers go first
                self.kill_current()
                remove_run_containers(f"falkordb-soak={label}")
            wall = time.time() - t0
            self.port += 10
            for f in list(run_dir.rglob("*.rdb")) + list(run_dir.rglob("*.aof")):
                f.unlink(missing_ok=True)
            for d in run_dir.rglob("aof"):
                if d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)
            run = read_run(
                run_dir, scn, side, rep, rc, timed_out, wall, cut=timed_out and cap_s < hang_cap_s
            )
            compress_large(run_dir)
            self.runs_csv.append(
                {
                    "scenario": scn,
                    "side": side,
                    "rep": rep,
                    "attempt": attempt,
                    "image": image,
                    "rc": rc,
                    "timed_out": timed_out,
                    "wall_s": run.wall_s,
                    "setup_s": run.setup_s,
                    "completed": run.completed,
                    "reason": run.reason,
                }
            )
            print(
                f"{kit.utc_now()} {scn} {side} rep {rep}: "
                f"{'completed' if run.completed else 'NOT completed: ' + run.reason} in {wall:.0f} s",
                flush=True,
            )
            if run.server_ran or attempt == 2 or run.cut:
                return run
            print(f"{scn} {side} rep {rep}: no server came up; run again", flush=True)
            run_dir.rename(run_dir.with_name(f"{run_dir.name}-retried-{attempt}"))
        raise AssertionError("unreachable")

    def kill_current(self):
        p, self.proc = self.proc, None
        if p is None:
            return
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(p.pid, signal.SIGKILL)  # the scenario and its worker processes
        with contextlib.suppress(subprocess.TimeoutExpired):
            p.wait(30)


def server_info(image: str) -> dict:
    """The image's server binary (${FALKORDB_SERVER:-redis}-server, from its config env) and
    version; an image without that binary is an error, never a fallback to the other one."""
    from falkorbench.soak import soakkit  # here: it needs the soak group, `bench` itself does not

    binary = soakkit.image_server(image)
    try:
        version = soakkit.image_server_version(image)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"image {image} has no runnable {binary} (picked from its FALKORDB_SERVER, "
            f"default redis): {(e.stderr or e.stdout or '').strip()[:200]}"
        ) from e
    return {"image": image, "server": binary, "version": version, "module": soakkit.IMAGE_MODULE}


def list_scenarios(arg: str | None) -> list[str]:
    allp = sorted(p.stem for p in SCENARIOS_DIR.glob("s[0-9][0-9]_*.py") if p.stem != "s00_pilot")
    if not arg:
        return allp
    out = []
    for w in (w.strip() for w in arg.split(",") if w.strip()):
        hit = [s for s in allp if s == w or s.split("_")[0] == w]
        if len(hit) != 1:
            raise ValueError(
                f"--scenarios: {w!r} matches {hit or 'nothing'} (have {', '.join(allp)})"
            )
        out.append(hit[0])
    return out


@dataclass
class Schedule:
    runs: dict[str, dict[str, list[Run]]] = field(default_factory=dict)
    not_run: list[str] = field(default_factory=list)  # "sNN_x (why)"
    unconfirmed: list[str] = field(default_factory=list)  # "sNN_x (why)": signal not confirmed


def run_cap(params: dict) -> float:
    """Hang cap of a side before any run of its scenario is known (the longest scenario took
    1.7x its load time at nightly, 2.4x in queue mode)."""
    return max(6 * params["duration"], params["duration"] + 900)


def run_all(
    soak,
    scenarios: list[str],
    imgs: dict,
    p: dict,
    deadline: float,
    clock=time.time,
    sched: Schedule | None = None,
) -> Schedule:
    """First pass over every scenario (order flipped every other one), then confirmation
    re-runs worst first. A side starts only when its expected time fits before the deadline;
    a run cut by the deadline takes its scenario (or re-run) out instead of reading as a crash;
    a run past its hang cap did not finish."""
    sched = sched or Schedule()
    full = run_cap(p)
    stop_at = deadline + 300  # inside the checks.yaml timeout (deadline + 10 min)

    def fits(seconds: float) -> bool:
        return clock() + seconds < deadline

    def hang_cap(normal: Run | None) -> float:
        # once a normal run of this scenario is known, a hang costs about three of it
        if normal is not None and normal.completed:
            return min(full, max(3 * normal.wall_s, normal.wall_s + 600))
        return full

    first_side = 1.5 * p["duration"] + 120
    for i, scn in enumerate(scenarios):
        order = ("candidate", "baseline") if i % 2 == 0 else ("baseline", "candidate")
        if not fits(2 * first_side):
            sched.not_run += [f"{s} (deadline)" for s in scenarios[i:]]
            break
        a = soak.run_one(scn, order[0], imgs[order[0]], 1, hang_cap(None), stop_at)
        b = None
        if not a.cut and fits(1.2 * a.wall_s if a.completed else first_side):
            b = soak.run_one(scn, order[1], imgs[order[1]], 1, hang_cap(a), stop_at)
        if a.cut or b is None or b.cut:
            if not a.cut and a.side == "candidate" and not a.completed:
                sched.runs[scn] = {"candidate": [a], "baseline": []}  # a candidate hang counts
            else:
                sched.not_run.append(f"{scn} (deadline)")
            continue
        sched.runs[scn] = {a.side: [a], b.side: [b]}
    if not p["reruns"]:
        return sched
    want = []
    for scn, sides in sched.runs.items():
        c, b = sides["candidate"], sides["baseline"]
        if not c or not b or not c[0].completed or not b[0].completed or not b[0].measured:
            continue  # a crash is a regression already; an unmeasured baseline cannot confirm
        crossed = [r for r in scenario_rows(scn, c, b, True) if r.crossed]
        if crossed:
            want.append((max(move_size(r) for r in crossed), scn))
    for _, scn in sorted(want, reverse=True):
        sides = sched.runs[scn]
        need = 1.2 * sum(r.wall_s for side in sides.values() for r in side)
        if not fits(need):
            print(f"no time to re-run {scn} (needs about {need:.0f} s)", flush=True)
            sched.unconfirmed.append(f"{scn} (no time to re-run)")
            continue
        order = (
            ("baseline", "candidate")
            if scenarios.index(scn) % 2 == 0
            else ("candidate", "baseline")
        )
        again = []
        for side in order:
            r = soak.run_one(scn, side, imgs[side], 2, hang_cap(sides[side][0]), stop_at)
            if r.cut:
                break
            again.append(r)
        if len(again) == 2:
            for r in again:
                sides[r.side].append(r)
        else:
            sched.unconfirmed.append(f"{scn} (re-run stopped at the deadline)")
    return sched


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="bench soak",
        description="Compare a candidate image with a baseline image on the release soak.",
    )
    kit.add_args(ap)
    ap.add_argument("--check", default="soak", help="result.json check name (checks.yaml entry)")
    ap.add_argument("--scenarios", help="comma list (s01,s12 or full names); default all but s00")
    ap.add_argument("--duration", type=float, help="override the mode's seconds per scenario")
    ap.add_argument("--scale", type=float, help="override the mode's dataset scale")
    ap.add_argument("--deadline-minutes", type=float, help="override the mode's deadline")
    return ap.parse_args(argv)


class Check:
    """One run of the check; kit.run_check() calls work(), then build(error), then cleanup()."""

    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.started = kit.utc_now()
        self.t0 = time.time()
        self.soak: Soak | None = None
        self.scenarios: list[str] = []
        self.sched = Schedule()
        self.problems: list[str] = []

    def work(self) -> None:
        a = self.a
        self.soak = Soak(a)  # a.out is absolute by now: the scenarios get absolute run dirs
        if kit.input_errors(a):
            return  # build() reports them
        try:
            self.scenarios = list_scenarios(a.scenarios)
        except ValueError as e:
            self.problems.append(str(e))
            return
        if not sys.platform.startswith("linux"):
            self.problems.append(
                "the soak runs images with host networking and host PIDs: Linux "
                "only (on macOS run a scenario with --module, see README.md)"
            )
        elif shutil.which("docker") is None:
            self.problems.append("docker is not on PATH")
        if self.problems:
            return
        for image in dict.fromkeys((a.candidate, a.baseline)):
            kit.pull(image)
        servers = {side: server_info(getattr(a, side)) for side in kit.SIDES}
        (Path(a.out) / "raw" / "servers.json").write_text(json.dumps(servers, indent=1))
        print(json.dumps(servers), flush=True)
        p = self.soak.params
        run_all(
            self.soak,
            self.scenarios,
            {"candidate": a.candidate, "baseline": a.baseline},
            p,
            self.t0 + p["deadline_min"] * 60,
            sched=self.sched,
        )

    def build(self, error: str | None) -> dict:
        a = self.a
        refused = kit.input_errors(a)
        if refused:
            return kit.error_result(a.check, a, self.started, "; ".join(refused))
        done = set(self.sched.runs) | {n.split(" ")[0] for n in self.sched.not_run}
        self.sched.not_run += [f"{s} (not reached)" for s in self.scenarios if s not in done]
        params = self.soak.params if self.soak else mode_params(a)
        runs_csv = self.soak.runs_csv if self.soak else []
        error = "; ".join(e for e in (*self.problems, error) if e) or None
        return finish(
            a, Path(a.out), self.started, self.scenarios, self.sched, error, runs_csv, params
        )

    def cleanup(self) -> None:
        """What a cut run left: the current scenario and any container of this check."""
        if self.soak is None or not self.soak.launched:
            return
        self.soak.kill_current()
        kit.remove_labelled(f"falkordb-soak-check={self.soak.run_id}")


def main(argv: list[str] | None = None) -> int:
    """`bench soak`: exits 0 whenever result.json was written, whatever its verdict."""
    a = parse_args(argv)
    check = Check(a)
    result = kit.run_check(a.check, a, check.work, check.build, check.cleanup)
    print(f"result: {result['verdict']} — {result['summary']}", flush=True)
    return 0


def finish(
    a: argparse.Namespace,
    out: Path,
    started: str,
    scenarios: list[str],
    sched: Schedule,
    error: str | None,
    runs_csv: list[dict],
    p: dict,
) -> dict:
    """Compare, write raw/rows.csv and raw/runs.csv; return the result (kit.run_check writes it)."""
    not_run = list(sched.not_run)
    unconfirmed = list(sched.unconfirmed) if p["reruns"] else []
    rows: list[Row] = []
    for scn, sides in sched.runs.items():
        cand, base = sides["candidate"], sides["baseline"]
        if not cand:
            continue
        if not base:  # the candidate did not finish and the deadline left no time for the baseline
            rows.append(_zero_one(f"{scn.split('_')[0]} completed", "completed", 0, None))
            continue
        if not base[0].measured and not cand[0].measured:
            not_run.append(
                f"{scn} (no check recorded on either side: {base[0].reason or 'no checks'})"
            )
            continue
        if not base[0].measured:
            not_run.append(f"{scn} (baseline not measured: {base[0].reason or 'no checks'})")
            continue
        if cand[0].harness_end and base[0].harness_end:
            not_run.append(
                f"{scn} (both sides ended outside the engine: {cand[0].reason}; {base[0].reason})"
            )
            continue
        scn_rows = scenario_rows(scn, cand, base, p["perf_gate"])
        if p["perf_gate"] and any(r.kind == "incomplete" for r in scn_rows):
            unconfirmed.append(f"{scn} (re-run incomplete)")
        rows += scn_rows
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if rows:
        write_rows(out, rows)
    if runs_csv:
        with open(raw / "runs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, list(runs_csv[0]))
            w.writeheader()
            w.writerows(runs_csv)
    errors = [
        error,
        "not measured: " + ", ".join(not_run) if not_run else "",
        "unconfirmed: " + ", ".join(unconfirmed) if unconfirmed else "",
    ]
    artifacts = [f"raw/{n}" for n in ("rows.csv", "runs.csv", "servers.json") if (raw / n).exists()]
    artifacts += sorted(f"raw/{d.name}" for d in raw.glob("s[0-9][0-9]_*") if d.is_dir())
    # the summary always says what went wrong, even when a regression outranks it
    summary = summary_line(rows, scenarios, not_run, a.mode, p["duration"], error, unconfirmed)
    return kit.result(
        a.check,
        a,
        started=started,
        findings=worst_first(rows),
        summary=summary,
        errors=errors,
        artifacts=artifacts,
        issues=ISSUES,
    )


if __name__ == "__main__":
    sys.exit(main())
