#!/usr/bin/env python3
"""The release soak as a nightly perf check (contract schema 1, tests/perf/README.md).

    python -I tests/soak/soak_check.py --candidate <image@sha256:...> --baseline <image@sha256:...>
        --out <dir> --mode <queue|nightly|long> [--check soak-a --scenarios s01,s02,...]
        [--candidate-label L] [--baseline-label L] [--candidate-commit C] [--baseline-commit C]

Runs each scenario (scenarios/sNN_*.py) once against each image, on this machine, one
after the other, with the order flipped every other scenario, and writes <out>/result.json:

* `sNN completed` (completed): 0 when the scenario did not finish (it raised, hung past its
  cap, or a server log shows a crash).
* `sNN <check>` (invariant): each check the scenario records, 1 held / 0 broken per side.
  A correctness check broken on the candidate only is a regression at once; broken on both
  sides it is `info` (not new). A latency or resource check (TIMING_RE) counts as broken
  only when every pair of runs, first pass and re-run, repeats it.
* `sNN <op>` (p50_ms, throughput) and `sNN setup wall time` (p50_ms): candidate/baseline
  ratios, gated above the worst same-image swing measured; a crossing is a regression only
  when a re-run of that scenario (both sides, order swapped) crosses again. p99 shifts are
  reported as info, never gated.
* Anything not measured to the end (a scenario left out by the deadline, a side whose
  harness failed, a signal whose re-run did not happen or did not complete) makes the
  verdict `error`, never `pass`.

Scenarios run from the images (lib/soakkit.py Server, image mode): the image's own server
binary and module, with the scenario's own arguments, never run.sh's defaults. Linux only.
Everything per run lands in <out>/raw/; result.json is written in every case, and the exit
status is 0 whenever it was.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCENARIOS_DIR = HERE / "scenarios"
VALIDATOR = HERE.parent / "perf" / "validate.py"

# ------------------------------------------------------------------ parameters
# --mode sets only the length (contract §1). duration: seconds of load per scenario and
# side; scale: dataset size (1.0 = Avi's full runs); perf_gate: whether perf can fail the
# check (queue reports it as info); reruns: confirmation re-runs; deadline_min: when to
# stop starting runs, a margin inside the checks.yaml timeout_minutes.
MODES = {
    "queue": {"duration": 60, "scale": 0.05, "perf_gate": False, "reruns": False, "deadline_min": 50},
    "nightly": {"duration": 300, "scale": 0.25, "perf_gate": True, "reruns": True, "deadline_min": 110},
    "long": {"duration": 1200, "scale": 1.0, "perf_gate": True, "reruns": True, "deadline_min": 320},
}

# Thresholds for a perf crossing, each above the worst same-image swing measured with
# main@4c1088949 against itself on an n4-standard-16 (queue mode for all 12 scenarios,
# nightly soak-b with re-runs; README): p50 1.81x (s10 live_delete 3.67 vs 6.65 ms),
# throughput 1.225x (s07 transfer_peak), setup wall time 1.03x. p99 is reported, never
# gated: the same runs swung it up to 56x (+541 ms) when one run caught a stall episode.
P50_RATIO = 2.0          # candidate p50 / baseline p50 at or above this ...
P50_FLOOR_MS = 2.0       # ... and at least this many ms slower
P99_INFO_RATIO = 2.0     # p99 shift reported as info (not gated) at or above this ...
P99_INFO_FLOOR_MS = 10.0  # ... and at least this many ms ...
MIN_P99_SAMPLES = 1000   # ... for ops with this many baseline samples
TPUT_RATIO = 0.7         # candidate throughput / baseline at or below this
SETUP_RATIO = 1.5        # setup wall time ratio at or above this ...
SETUP_FLOOR_S = 5.0      # ... and at least this many seconds slower
MIN_SAMPLES = 200        # successful baseline ops before an op is compared at all ...
MIN_CAND_SAMPLES = 20    # ... and candidate ops before its latency is (throughput always is)

# Checks whose outcome is a latency or resource measurement, not a correctness property:
# they count as broken only when a re-run repeats it (the fuzz check's rule, DECISIONS).
# Generous hang bounds (recovery < 180 s, replica sync <= 900 s, an index reaching
# OPERATIONAL) are not here: missing one is a run that did not finish, a regression at once.
TIMING_RE = re.compile(
    r"\bp99\b|\bSLO\b|\bschedule\b|\bplateaus?\b|\bRSS\b|\bused_memory\b|memory growth|\bdrift\b|"
    r"within time budget|replication lag|fires promptly|stays fast|CPU idle|churn volume|"
    r"concurrent connections were reached|no marker timeouts", re.I)

# Correctness checks that an open bug breaks intermittently (contract v1.9). An entry needs
# an open issue and same-image evidence of the intermittent break; remove it when the issue
# closes. A listed check is not gated for that break: kept in raw/rows.csv with status
# "known" and named in the summary. It is still a regression when the candidate breaks it
# every time it is checked (the scenario records "k of n runs failed": k == n >= 10) in
# every run of the pass, so a known flake cannot hide a total breakage.
KNOWN_FLAKY = {
    # one entry missing from the index during/after a build under writes, e.g.
    # "[567,667] idx=1242 scan=1243"; broke on both sides, neither, or one side in runs of
    # the same pair (main vs main; 6.2.0 vs 6.2.1, run 45401220445)
    "s10 Live (under writes) index == scan during build/after": "#3095",
}

# Server-log lines that mean the server process crashed (not just a failed query).
CRASH_RE = re.compile(r"crashed by signal|panicked at|=== (?:REDIS|VALKEY) BUG REPORT|"
                      r"Segmentation fault|ASSERTION FAILED|Guru Meditation", re.I)

IMAGE_RE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")   # the validator's stricter one wins
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
CHECK_RE = re.compile(r"[a-z][a-z0-9-]{0,42}")
LINE_BREAK_RE = re.compile("[\r\n\x0b\x0c\x1c-\x1e\x85\u2028\u2029]")


def is_timing_check(name: str) -> bool:
    return bool(TIMING_RE.search(name))


# ------------------------------------------------------------------ one run's record
@dataclass
class Run:
    """What one scenario run on one side left behind (raw/<scn>/<side>-<rep>/)."""
    scenario: str
    side: str                      # candidate | baseline
    rep: int
    completed: bool
    reason: str = ""               # why not completed
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)   # check -> its first failing detail
    ops: dict[str, dict] = field(default_factory=dict)  # op -> {p99_ms, throughput, ok}
    setup_s: float | None = None
    wall_s: float = 0.0
    contended: bool = False
    server_ran: bool = True        # a server answered at least once (monitor.csv saw it alive)
    rc: int | None = 0             # the scenario's exit status (None: stopped by a cap)
    cut: bool = False              # stopped by the deadline, not a hang: the run does not count

    @property
    def measured(self) -> bool:
        """The run got as far as recording a check against a live server."""
        return self.server_ran and bool(self.checks)

    @property
    def harness_end(self) -> bool:
        """Ended by something outside the engine: killed by a signal (the OOM killer, a
        runner shutting down) or exited cleanly without its summary."""
        return (self.rc is not None and self.rc < 0) or self.reason == "no summary.json"


def checks_by_name(items: list[dict]) -> dict[str, bool]:
    """A scenario may record one name several times; it held only if every one held."""
    out: dict[str, bool] = {}
    for c in items:
        out[c["name"]] = out.get(c["name"], True) and bool(c["ok"])
    return out


def ops_of(summary: dict) -> dict[str, dict]:
    """Worker ops (`_`-ops are the harness's bookkeeping) with their successful count."""
    out = {}
    for op, d in (summary.get("ops") or {}).items():
        if op.startswith("_"):
            continue
        ok = (d.get("count") or 0) - (d.get("errors") or 0)
        p99 = d.get("p99_ms")
        p50 = d.get("p50_ms")
        out[op] = {"p50_ms": p50 if p50 is not None and math.isfinite(p50) else None,
                   "p99_ms": p99 if p99 is not None and math.isfinite(p99) else None,
                   "throughput": d.get("throughput_ops_s") or 0.0, "ok": ok}
    return out


def server_ran(run_dir: Path) -> bool:
    """Did any server of this run come up? (soakkit's Monitor writes a row per second with
    alive=1 once a server answers; a run whose server never started has none.)"""
    for mon in run_dir.rglob("monitor.csv"):
        for line in mon.read_text(errors="replace").splitlines()[1:]:
            parts = line.split(",")
            if len(parts) > 6 and parts[6] == "1":
                return True
    return False


def setup_seconds(phases: list[dict]) -> float | None:
    """Wall time of the scenario's first "ingest" phase (to the next phase mark)."""
    for i, ev in enumerate(phases):
        if ev.get("phase") == "ingest" and i + 1 < len(phases):
            return round(float(phases[i + 1]["t"]) - float(ev["t"]), 1)
    return None


def crash_lines(run_dir: Path) -> list[str]:
    bad = []
    for log in sorted(run_dir.rglob("*.log")):
        if log.name in ("stdout.log", "load.log"):
            continue
        for line in log.read_text(errors="replace").splitlines():
            if CRASH_RE.search(line):
                bad.append(f"{log.relative_to(run_dir)}: {line[:200]}")
    return bad[:20]


def read_run(run_dir: Path, scenario: str, side: str, rep: int, rc: int | None,
             timed_out: bool, wall_s: float, cut: bool = False) -> Run:
    summary_p, checks_p, phases_p = (run_dir / n for n in ("summary.json", "checks.json", "phases.json"))
    summary = json.loads(summary_p.read_text()) if summary_p.exists() else None
    checks = json.loads(checks_p.read_text()) if checks_p.exists() else []
    phases = json.loads(phases_p.read_text()) if phases_p.exists() else []
    crashes = crash_lines(run_dir)
    reason = ("stopped at the deadline" if cut else "timed out" if timed_out else
              f"exit {rc}" if rc else "no summary.json" if summary is None else
              f"server crash: {crashes[0]}" if crashes else "")
    return Run(scenario=scenario, side=side, rep=rep, completed=not reason, reason=reason,
               checks=checks_by_name(checks), ops=ops_of(summary or {}),
               details={c["name"]: c.get("detail", "") for c in checks if not c["ok"]},
               setup_s=setup_seconds(phases), wall_s=round(wall_s, 1),
               server_ran=server_ran(run_dir), cut=cut, rc=rc)


# ------------------------------------------------------------------ findings
@dataclass
class Row:
    """One comparison; becomes a finding when it is not `ok` (or as an informative row)."""
    name: str
    metric: str
    candidate: float | None
    baseline: float | None
    ratio: float | None
    threshold: float | None
    direction: str
    status: str
    crossed: bool = False          # crossed in the first pass (asks for a re-run)
    note: str = ""

    def finding(self) -> dict:
        def num(x):
            if x is None or (isinstance(x, float) and not math.isfinite(x)):
                return None
            return round(x, 4) if isinstance(x, float) else x
        return {"name": self.name, "metric": self.metric, "candidate": num(self.candidate),
                "baseline": num(self.baseline), "ratio": num(self.ratio),
                "threshold": self.threshold, "direction": self.direction, "status": self.status}


def _ratio(c, b):
    if c is None or b is None or not b:
        return None
    return c / b


def _correctness_value(runs: list[Run], name: str) -> int | None:
    """1 held on every run that recorded it, 0 broken on any; None if no run recorded it."""
    vals = [r.checks[name] for r in runs if name in r.checks]
    return None if not vals else int(all(vals))


_K_OF_N = re.compile(r"^(\d+) of (\d+) runs failed")


def _always_broken(runs: list[Run], name: str) -> bool:
    """Every run recorded the check and broke it every time it was checked (k == n >= 10).
    A detail without "k of n" counts as always broken: gate rather than hide."""
    if not runs or not all(name in r.checks for r in runs):
        return False
    for r in runs:
        if r.checks[name]:
            return False
        m = _K_OF_N.match(r.details.get(name, ""))
        if m and not (int(m.group(1)) == int(m.group(2)) >= 10):
            return False
    return True


def _timing_row(tag: str, name: str, cand: list[Run], base: list[Run], perf_gate: bool) -> Row:
    """A latency/resource check counts as broken on the candidate only when every pair of runs
    (first pass and re-run, at least two) has it broken on the candidate and held on the
    baseline. Broken on both sides in every pair is info. Anything else is reported with the
    candidate as 1 and a note when it broke at all."""
    pairs = [(c.checks.get(name), b.checks.get(name)) for c, b in zip(cand, base)]
    c_seen = [cv for cv, _ in pairs if cv is not None]
    b_seen = [bv for _, bv in pairs if bv is not None]
    c_broke = sum(1 for v in c_seen if v is False)
    regress = [cv is False and bv is True for cv, bv in pairs]
    if perf_gate and len(pairs) >= 2 and all(regress):
        cv, bv = 0, 1
    elif pairs and all(cv is False and bv is False for cv, bv in pairs):
        cv, bv = 0, 0
    else:
        cv = 1 if c_seen else None
        bv = None if not b_seen else int(not all(v is False for v in b_seen))
    row = _zero_one(f"{tag} {name}", "invariant", cv, bv)
    row.crossed = bool(pairs) and regress[0]
    if cv == 1 and c_broke:
        row.status = "info"
        row.note = f"broke in {c_broke} of {len(c_seen)} candidate runs"
        row.name += f" ({row.note})"
    return row


def scenario_rows(scn: str, cand: list[Run], base: list[Run], perf_gate: bool) -> list[Row]:
    """Every comparison for one scenario. `cand`/`base` hold rep 1 first, then any re-run
    (runs stopped at the deadline are never passed in)."""
    tag = scn.split("_")[0]
    rows: list[Row] = []
    # completed: a crash or a hang on any run is immediate
    c_done = int(all(r.completed for r in cand)) if cand else None
    b_done = int(all(r.completed for r in base)) if base else None
    rows.append(_zero_one(f"{tag} completed", "completed", c_done, b_done))

    # invariants
    names = sorted({n for r in cand + base for n in r.checks})
    for n in names:
        if is_timing_check(n):
            rows.append(_timing_row(tag, n, cand, base, perf_gate))
            continue
        row = _zero_one(f"{tag} {n}", "invariant", _correctness_value(cand, n), _correctness_value(base, n))
        if row.name in KNOWN_FLAKY and not _always_broken(cand, n):
            row.status = "known"
            row.note = (f"known flaky ({KNOWN_FLAKY[row.name]}), not gated; broke on "
                        + (" and ".join(side for side, v in (("candidate", row.candidate),
                                                             ("baseline", row.baseline)) if v == 0)
                           or "neither side"))
        rows.append(row)

    # perf: per-op p99 and throughput, setup wall time
    rows += _perf_rows(tag, cand, base, perf_gate)
    return rows


def _zero_one(name, metric, c, b) -> Row:
    """Status of a 1-held / 0-broken pair (contract v1.4): broken on the candidate and not
    on the baseline is a regression; broken on both is info (not new)."""
    if c == 0:
        status = "info" if b == 0 else "regression"
    elif c == 1:
        status = "improvement" if b == 0 else "ok"
    else:  # the candidate never reached this check (a different path, or it stopped early)
        status = "info" if b is not None else "ok"
    ratio = c / b if c is not None and b else None
    return Row(name, metric, c, b, ratio, None, "lower_is_worse", status)


def _perf_crossings(c_run: Run, b_run: Run) -> dict[tuple[str, str], tuple]:
    """(op, metric) -> (candidate, baseline, ratio, threshold, direction, regressed, improved)."""
    out = {}
    none = {"p50_ms": None, "p99_ms": None, "throughput": 0.0, "ok": 0}
    # gated on the baseline: an op the candidate slowed down or broke enough to (nearly) stop
    # doing must not drop out of the comparison
    for op in sorted(o for o, b in b_run.ops.items() if b["ok"] >= MIN_SAMPLES):
        c, b = c_run.ops.get(op, none), b_run.ops[op]
        if c["ok"] >= MIN_CAND_SAMPLES:
            r = _ratio(c["p50_ms"], b["p50_ms"])
            if r is not None:
                out[(op, "p50_ms")] = (c["p50_ms"], b["p50_ms"], r, P50_RATIO, "higher_is_worse",
                                       r >= P50_RATIO and c["p50_ms"] - b["p50_ms"] >= P50_FLOOR_MS,
                                       r <= 1 / P50_RATIO and b["p50_ms"] - c["p50_ms"] >= P50_FLOOR_MS)
        if c["ok"] >= MIN_CAND_SAMPLES and b["ok"] >= MIN_P99_SAMPLES:
            r = _ratio(c["p99_ms"], b["p99_ms"])
            if r is not None:
                # reported, never gated (no crossing, no re-run): "tail" marks a notable shift
                tail = abs(c["p99_ms"] - b["p99_ms"]) >= P99_INFO_FLOOR_MS and (
                    r >= P99_INFO_RATIO or r <= 1 / P99_INFO_RATIO)
                out[(op, "p99_ms")] = (c["p99_ms"], b["p99_ms"], r, None, "higher_is_worse", False,
                                       False, tail)
        r = _ratio(c["throughput"], b["throughput"])
        if r is not None:
            out[(op, "throughput")] = (c["throughput"], b["throughput"], r, TPUT_RATIO,
                                       "lower_is_worse", r <= TPUT_RATIO, r >= 1 / TPUT_RATIO)
    if c_run.setup_s is not None and b_run.setup_s:
        c, b = c_run.setup_s, b_run.setup_s
        r = c / b
        out[("setup wall time", "p50_ms")] = (c * 1000, b * 1000, r, SETUP_RATIO, "higher_is_worse",
                                              r >= SETUP_RATIO and c - b >= SETUP_FLOOR_S,
                                              r <= 1 / SETUP_RATIO and b - c >= SETUP_FLOOR_S)
    return out


def _perf_rows(tag: str, cand: list[Run], base: list[Run], perf_gate: bool) -> list[Row]:
    if not cand or not base or not cand[0].completed or not base[0].completed:
        return []
    first = _perf_crossings(cand[0], base[0])
    rerun = len(cand) > 1 and len(base) > 1
    again = _perf_crossings(cand[1], base[1]) if rerun and cand[1].completed and base[1].completed else None
    rows = []
    for (op, metric), (c, b, r, thr, direction, reg, imp, *tail) in first.items():
        name = f"{tag} {op}"
        row = Row(name, metric, c, b, r, thr, direction, "ok", crossed=reg)
        if tail and tail[0]:
            row.status, row.note = "info", "tail shift, reported not gated"
            row.name += " (p99 shift, not gated)"
        if reg:
            rep = again.get((op, metric)) if again is not None else None
            if not perf_gate:
                row.status, row.note = "info", "perf is informational in this mode"
            elif rep is not None and rep[5]:
                # repeated: report the milder of the two crossings
                if rep[2] > 0 and (r <= 0 or abs(math.log(rep[2])) < abs(math.log(r))):
                    row.candidate, row.baseline, row.ratio = rep[0], rep[1], rep[2]
                row.status, row.note = "regression", "repeated on re-run"
            else:
                row.status = "info"
                row.note = ("not re-run" if not rerun else "re-run incomplete" if rep is None
                            else "not repeated on re-run")
                row.name += f" ({row.note})"
        elif imp:
            row.status = "improvement"
        rows.append(row)
    return rows


def pick_findings(rows: list[Row], cap: int = 50, informative_cap: int = 10) -> list[dict]:
    """Regressions worst first, then improvements, then up to 10 info rows; at most 50."""
    def worst(r: Row):
        if r.ratio is None or r.ratio <= 0:
            return float("inf")
        return abs(math.log(r.ratio))
    def info_rank(r: Row):
        # unconfirmed perf/timing signals first, then rows broken on both sides, then p99
        # shifts (reported, not gated), then the rest
        kind = (0 if r.note.startswith(("not re", "re-run", "broke in")) else
                1 if (r.candidate, r.baseline) == (0, 0) else
                2 if r.note.startswith("tail shift") else 3)
        return (kind, -worst(r))
    reg = sorted((r for r in rows if r.status == "regression"), key=worst, reverse=True)
    imp = sorted((r for r in rows if r.status == "improvement"), key=worst, reverse=True)
    info = sorted((r for r in rows if r.status == "info"), key=info_rank)
    out = reg[:cap]
    out += imp[:cap - len(out)]
    out += info[:min(informative_cap, cap - len(out))]
    return [r.finding() for r in out]


def verdict_of(rows: list[Row], not_run: list[str], error: str | None) -> tuple[str, bool]:
    """regression outranks error (contract v1.4); a scenario not measured, or a perf signal
    left unconfirmed for lack of time, is an error, never a pass."""
    if any(r.status == "regression" for r in rows):
        return "regression", True
    if error or not_run:
        return "error", False
    return "pass", False


def summary_line(rows: list[Row], scenarios: list[str], not_run: list[str], mode: str,
                 duration: float, error: str | None, unconfirmed: list[str] = ()) -> str:
    reg = [r for r in rows if r.status == "regression"]
    both = [r for r in rows if r.status == "info" and r.candidate == 0 and r.baseline == 0]
    unconf = [r for r in rows if r.status == "info" and r.note.startswith(("not re", "re-run", "broke in"))]
    tails = [r for r in rows if r.note.startswith("tail shift")]
    known = [r for r in rows if r.status == "known" and 0 in (r.candidate, r.baseline)]
    missing = {n.split(" ")[0] for n in not_run}
    parts = [f"{len([s for s in scenarios if s not in missing])} of {len(scenarios)} scenarios "
             f"measured, {mode} ({duration:.0f} s per side)"]
    if reg:
        parts.append(f"{len(reg)} regression(s): " + ", ".join(
            f"{r.name}" + (f" {r.ratio:.2f}x" if r.ratio and r.metric not in ("invariant", "completed") else "")
            for r in reg[:4]) + (" …" if len(reg) > 4 else ""))
    else:
        parts.append("no regression")
    if both:
        parts.append(f"{len(both)} broken on both sides (not new)")
    if unconf:
        parts.append(f"{len(unconf)} perf signal(s) not confirmed")
    if known:
        parts.append("known flaky, not gated: " + ", ".join(
            f"{r.name} ({KNOWN_FLAKY[r.name]}) {r.note.split('; ', 1)[1]}" for r in known))
    if tails:
        worst = max(tails, key=lambda r: abs(math.log(r.ratio)) if r.ratio and r.ratio > 0 else 0)
        parts.append(f"{len(tails)} p99 shift(s) reported, not gated (largest {worst.name.split(' (')[0]} "
                     f"{worst.ratio:.2f}x)")
    if not_run:
        parts.append("not measured: " + ", ".join(not_run))
    if unconfirmed:
        parts.append("unconfirmed: " + ", ".join(unconfirmed))
    if error:
        parts.append(error)
    return LINE_BREAK_RE.sub(" ", "; ".join(parts))


# ------------------------------------------------------------------ result.json
def utc(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def side_of(image: str, label: str | None, commit: str | None) -> dict:
    if not label:
        repo, _, digest = image.partition("@")
        repo = repo.rsplit(":", 1)[0] if ":" in repo.rsplit("/", 1)[-1] else repo
        label = f"{repo}@{digest.removeprefix('sha256:')[:12]}"
    return {"image": image, "commit": commit or None, "label": label}


def machine() -> dict:
    p = os.environ.get("PERF_MACHINE_JSON")
    if p and Path(p).exists():
        return json.loads(Path(p).read_text())
    cpu = platform.processor() or platform.machine() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"runner": "local", "type": platform.node() or "local", "cpu": cpu, "cores": os.cpu_count() or 1}


def _validator_module():
    """tests/perf/validate.py (the contract's reference), when this tree has it."""
    if not VALIDATOR.exists():
        return None
    spec = importlib.util.spec_from_file_location("perf_validate", VALIDATOR)
    mod = importlib.util.module_from_spec(spec)
    keep = sys.dont_write_bytecode
    sys.dont_write_bytecode = True      # leave no __pycache__ in tests/perf
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = keep
    return mod


def validate(result: dict) -> list[str]:
    """Problems per tests/perf/validate.py, when it is there."""
    mod = _validator_module()
    if mod is None:
        return []
    return [str(p) for p in mod.validate_result(json.loads(json.dumps(result)))]


def write_result(out: Path, result: dict) -> dict:
    """Write result.json. One that fails the validator becomes an error result that keeps the
    check's own error first (merge.py quotes it when it stubs the file); the original goes
    to raw/invalid-result.json."""
    problems = validate(result)
    if problems:
        (out / "raw").mkdir(parents=True, exist_ok=True)
        (out / "raw" / "invalid-result.json").write_text(json.dumps(result, indent=1, allow_nan=False))
        held = any(f.get("status") == "regression" for f in result["findings"])
        why = (f"result.json is not valid ({'; '.join(problems)[:500]}); the original is "
               f"raw/invalid-result.json" + (", which held a regression" if held else ""))
        result = dict(result, verdict="error", confirmed=False, findings=[],
                      error=f"{result['error']}; {why}" if result.get("error") else why,
                      summary=result["summary"][:300])
    (out / "result.json").write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    return result


def write_rows(out: Path, rows: list[Row]) -> None:
    with open(out / "raw" / "rows.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "metric", "candidate", "baseline", "ratio", "threshold", "direction",
                    "status", "crossed_first_pass", "note"])
        for r in rows:
            d = r.finding()
            w.writerow([d["name"], d["metric"], d["candidate"], d["baseline"], d["ratio"],
                        d["threshold"], d["direction"], d["status"], int(r.crossed), r.note])


# ------------------------------------------------------------------ running
class Terminated(Exception):
    pass


def _on_term(signum, _frame):
    raise Terminated(f"terminated by signal {signum} (the step's timeout?)")


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
    """gzip a run's big files (ledgers, latency histograms; about 10x smaller) once it has
    been read: the artifact keeps the evidence without uploading tens of MB a scenario."""
    import gzip
    for f in run_dir.rglob("*"):
        if f.is_file() and f.suffix != ".gz" and f.stat().st_size >= min_bytes:
            with open(f, "rb") as src, gzip.open(f.with_name(f.name + ".gz"), "wb") as dst:
                shutil.copyfileobj(src, dst)
            f.unlink()


def docker_cleanup(label: str, key: str = "falkordb-soak") -> None:
    ids = subprocess.run(["docker", "ps", "-aq", "--filter", f"label={key}={label}"],
                         capture_output=True, text=True).stdout.split()
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)


def pull(image: str, attempts: int = 4) -> None:
    for i in range(attempts):
        p = subprocess.run(["docker", "pull", "--quiet", image], capture_output=True, text=True)
        if p.returncode == 0:
            return
        if i + 1 == attempts:
            raise RuntimeError(f"docker pull {image} failed: {(p.stderr or p.stdout).strip()[-300:]}")
        time.sleep(15 * (i + 1))


class LoadSampler(threading.Thread):
    """1-minute load average every 10 s into <run>/load.log; max kept for the gate."""

    def __init__(self, path: Path):
        super().__init__(daemon=True)
        self.path, self.max = path, 0.0
        self._halt = threading.Event()

    def run(self):
        with open(self.path, "w") as f:
            while not self._halt.is_set():
                load = os.getloadavg()[0]
                self.max = max(self.max, load)
                f.write(f"{time.time():.0f} {load:.2f}\n")
                f.flush()
                self._halt.wait(10)

    def stop(self):
        self._halt.set()
        self.join(5)


class Soak:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.out = Path(a.out).resolve()
        self.raw = self.out / "raw"
        self.params = dict(MODES[a.mode])
        if a.duration:
            self.params["duration"] = a.duration
        if a.scale:
            self.params["scale"] = a.scale
        if a.deadline_minutes:
            self.params["deadline_min"] = a.deadline_minutes
        self.on_vm = bool(os.environ.get("PERF_MACHINE_JSON"))
        self.quiet_gate = not self.on_vm and not a.no_quiet_gate
        cores = os.cpu_count() or 1
        self.quiet_load, self.contended_load = 0.5 * cores, 1.25 * cores
        self.run_id = f"{os.getpid()}-{int(time.time())}"
        self.proc: subprocess.Popen | None = None
        self.runs_csv: list[dict] = []
        self.port = 21000

    # one scenario on one side
    def run_one(self, scn: str, side: str, image: str, rep: int, hang_cap_s: float,
                stop_at: float) -> Run:
        """One run, capped at `hang_cap_s` and at the absolute time `stop_at`. A run that
        outlives its hang cap timed out (did not finish); one stopped earlier by `stop_at` is
        `cut` and does not count. A side whose server never came up is tried once more."""
        run_dir = self.raw / scn / f"{side}-{rep}"
        for attempt in (1, 2):
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True)
            if self.quiet_gate:
                self.wait_quiet()
            label = f"{self.run_id}-{scn}-{side}-{rep}"
            cap_s = max(60.0, min(hang_cap_s, stop_at - time.time()))   # after the quiet wait
            self.port = free_port_block(self.port)
            cmd = [sys.executable, "-I", str(SCENARIOS_DIR / f"{scn}.py"), "--image", image,
                   "--port", str(self.port), "--out", str(run_dir),
                   "--duration", str(self.params["duration"]), "--scale", str(self.params["scale"])]
            env = dict(os.environ, SOAK_RUN_ID=label, SOAK_CHECK_ID=self.run_id)
            sampler = LoadSampler(run_dir / "load.log")
            sampler.start()
            t0 = time.time()
            rc, timed_out = None, False
            print(f"{utc(t0)} {scn} {side} rep {rep}: {' '.join(cmd[2:])}", flush=True)
            try:
                with open(run_dir / "stdout.log", "w") as log:
                    self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                                 start_new_session=True)
                    try:
                        rc = self.proc.wait(cap_s)
                    except subprocess.TimeoutExpired:
                        timed_out = True
            finally:
                self.kill_current()
                sampler.stop()
                docker_cleanup(label)
            wall = time.time() - t0
            self.port += 10
            for f in list(run_dir.rglob("*.rdb")) + list(run_dir.rglob("*.aof")):
                f.unlink(missing_ok=True)
            for d in run_dir.rglob("aof"):
                if d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)
            run = read_run(run_dir, scn, side, rep, rc, timed_out, wall,
                           cut=timed_out and cap_s < hang_cap_s)
            run.contended = self.quiet_gate and sampler.max > self.contended_load
            compress_large(run_dir)
            self.runs_csv.append({"scenario": scn, "side": side, "rep": rep, "attempt": attempt,
                                  "image": image, "rc": rc, "timed_out": timed_out,
                                  "wall_s": run.wall_s, "setup_s": run.setup_s,
                                  "completed": run.completed, "reason": run.reason,
                                  "max_load": round(sampler.max, 2), "contended": run.contended})
            print(f"{utc(time.time())} {scn} {side} rep {rep}: "
                  f"{'completed' if run.completed else 'NOT completed: ' + run.reason} in {wall:.0f} s",
                  flush=True)
            if (not run.contended and run.server_ran) or attempt == 2 or run.cut:
                return run
            why = (f"machine load reached {sampler.max:.1f}" if run.contended else
                   "no server came up (a harness flake is retried once)")
            print(f"{scn} {side} rep {rep}: {why}; run again", flush=True)
            run_dir.rename(run_dir.with_name(f"{run_dir.name}-retried-{attempt}"))
        raise AssertionError("unreachable")

    def kill_current(self):
        p, self.proc = self.proc, None
        if p is None:
            return
        try:
            os.killpg(p.pid, signal.SIGKILL)   # the scenario and its worker processes
        except (ProcessLookupError, PermissionError):
            pass
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            pass

    def wait_quiet(self, need_s: int = 30, max_s: int = 900):
        t0 = q0 = time.time()
        while time.time() - t0 < max_s:
            if os.getloadavg()[0] >= self.quiet_load:
                q0 = time.time()
            elif time.time() - q0 >= need_s:
                return
            time.sleep(5)


def server_info(image: str) -> dict:
    sys.path.insert(0, str(HERE / "lib"))
    import soakkit  # noqa: E402  (lazy: needs redis/psutil, which the unit tests do not)
    binary = soakkit.image_server(image)
    try:
        version = soakkit.image_server_version(image)
    except subprocess.CalledProcessError as e:
        version = f"unknown ({(e.stderr or '').strip()[:200]})"
    return {"image": image, "server": binary, "version": version, "module": soakkit.IMAGE_MODULE}


def list_scenarios(arg: str | None) -> list[str]:
    allp = sorted(p.stem for p in SCENARIOS_DIR.glob("s[0-9][0-9]_*.py") if p.stem != "s00_pilot")
    if not arg:
        return allp
    want = [w.strip() for w in arg.split(",") if w.strip()]
    out = []
    for w in want:
        hit = [s for s in allp if s == w or s.split("_")[0] == w]
        if len(hit) != 1:
            raise ValueError(f"--scenarios: {w!r} matches {hit or 'nothing'} (have {', '.join(allp)})")
        out.append(hit[0])
    return out


@dataclass
class Schedule:
    runs: dict[str, dict[str, list[Run]]] = field(default_factory=dict)
    not_run: list[str] = field(default_factory=list)      # "sNN_x (why)"
    unconfirmed: list[str] = field(default_factory=list)  # perf signal not re-run: no time


def run_all(soak, scenarios: list[str], imgs: dict, p: dict, deadline: float,
            clock=time.time, sched: Schedule | None = None) -> Schedule:
    """First pass over every scenario (order flipped every other one), then confirmation
    re-runs worst first. A side starts only when its expected time fits before the deadline.
    A run the deadline cut short takes its scenario (first pass) or its re-run out of the
    comparison; a run that outlived its hang cap is a run that did not finish."""
    sched = sched or Schedule()
    full = run_cap(p)
    stop_at = deadline + 300    # inside the checks.yaml timeout (deadline + 10 min)

    def fits(seconds: float) -> bool:
        return clock() + seconds < deadline

    def hang_cap(normal: Run | None) -> float:
        # once a normal run of this scenario is known, a hang costs about three of it
        if normal is not None and normal.completed:
            return min(full, max(3 * normal.wall_s, normal.wall_s + 600))
        return full

    def candidate_hang_only(a: Run) -> bool:
        return a.side == "candidate" and not a.completed

    first_side = 1.5 * p["duration"] + 120     # one side before anything is measured
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
            if not a.cut and candidate_hang_only(a):
                # the candidate did not finish: a regression whether or not the baseline ran
                sched.runs[scn] = {"candidate": [a], "baseline": []}
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
            continue   # a crash is a regression already; an unmeasured baseline cannot confirm
        crossed = [r for r in scenario_rows(scn, c, b, True) if r.crossed]
        if crossed:
            want.append((max(abs(math.log(r.ratio)) if r.ratio and r.ratio > 0 else 9.9
                             for r in crossed), scn))
    for _, scn in sorted(want, reverse=True):
        sides = sched.runs[scn]
        need = 1.2 * sum(r.wall_s for side in sides.values() for r in side)
        if not fits(need):
            print(f"no time to re-run {scn} (needs about {need:.0f} s)", flush=True)
            sched.unconfirmed.append(f"{scn} (no time to re-run)")
            continue
        i = scenarios.index(scn)
        order = ("baseline", "candidate") if i % 2 == 0 else ("candidate", "baseline")
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


def run_cap(params: dict) -> float:
    """Hang cap of a side before any run of its scenario is known: the longest scenario took
    1.7x its load time at nightly (s11, 499 s for 300 s) and 2.4x in queue mode (143 s)."""
    return max(6 * params["duration"], params["duration"] + 900)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", required=True, choices=sorted(MODES))
    ap.add_argument("--candidate-label")
    ap.add_argument("--baseline-label")
    ap.add_argument("--candidate-commit")
    ap.add_argument("--baseline-commit")
    ap.add_argument("--check", default="soak", help="result.json check name (checks.yaml entry)")
    ap.add_argument("--scenarios", help="comma list (s01,s12 or full names); default all but s00")
    ap.add_argument("--duration", type=float, help="override the mode's seconds per scenario")
    ap.add_argument("--scale", type=float, help="override the mode's dataset scale")
    ap.add_argument("--deadline-minutes", type=float, help="override the mode's deadline")
    ap.add_argument("--no-quiet-gate", action="store_true", help="local: don't wait for a quiet machine")
    a = ap.parse_args(argv)

    started = time.time()
    # _perf-vm.yml passes `--out out`: make it absolute before anything uses it (the scenario
    # subprocesses get paths under it)
    out = Path(a.out).resolve()
    a.out = str(out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    old_term = signal.signal(signal.SIGTERM, _on_term)
    soak = Soak(a)
    p = soak.params
    deadline = started + p["deadline_min"] * 60
    scenarios: list[str] = []
    sched = Schedule()
    error = None
    try:
        scenarios = list_scenarios(a.scenarios)
        check_inputs(a)
        if not CHECK_RE.fullmatch(a.check) or "--" in a.check:
            raise RuntimeError(f"--check {a.check!r} must match [a-z][a-z0-9-]{{0,42}} without '--'")
        if not sys.platform.startswith("linux"):
            raise RuntimeError("the soak runs images with host networking and host PIDs: Linux only "
                               "(on macOS run a scenario with --module, see tests/soak/README.md)")
        if shutil.which("docker") is None:
            raise RuntimeError("docker is not on PATH")
        for img in {a.candidate, a.baseline}:
            pull(img)
        servers = {side: server_info(img) for side, img in (("candidate", a.candidate),
                                                             ("baseline", a.baseline))}
        (out / "raw" / "servers.json").write_text(json.dumps(servers, indent=1))
        print(json.dumps(servers), flush=True)
        imgs = {"candidate": a.candidate, "baseline": a.baseline}

        run_all(soak, scenarios, imgs, p, deadline, sched=sched)
    except (Terminated, KeyboardInterrupt) as e:
        error = str(e) or type(e).__name__
    except Exception as e:  # noqa: BLE001  (always write result.json)
        error = f"{type(e).__name__}: {e}"
    # from here on a second TERM/INT must not stop result.json from being written
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    old_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        soak.kill_current()
        if shutil.which("docker"):
            docker_cleanup(soak.run_id, key="falkordb-soak-check")   # whatever a cut run left
        done = set(sched.runs) | {n.split(" ")[0] for n in sched.not_run}
        sched.not_run += [f"{s} (not reached)" for s in scenarios if s not in done]
        result = finish(a, out, started, scenarios, sched, error, soak.runs_csv, p)
    except Exception as e:  # noqa: BLE001
        result = fallback_result(a, out, started, f"{error + '; ' if error else ''}"
                                 f"writing the result failed: {type(e).__name__}: {e}")
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
    print(f"result: {result['verdict']} — {result['summary']}", flush=True)
    return 0


def check_inputs(a: argparse.Namespace) -> None:
    """Refuse inputs that could never give a valid result.json, before any run."""
    image_re = IMAGE_RE
    mod = _validator_module()
    if mod is not None:
        image_re = mod.IMAGE_RE
    for side in ("candidate", "baseline"):
        img = getattr(a, side)
        if not image_re.fullmatch(img):
            raise RuntimeError(f"--{side} {img!r} is not an image pinned by digest (name[:tag]@sha256:<64 hex>)")
        label = getattr(a, f"{side}_label")
        if label is not None and (not label.strip() or LINE_BREAK_RE.search(label)):
            raise RuntimeError(f"--{side}-label must be one non-blank line")
        commit = getattr(a, f"{side}_commit")
        if commit and not COMMIT_RE.fullmatch(commit):
            raise RuntimeError(f"--{side}-commit {commit!r} is not 40 hex")


def finish(a: argparse.Namespace, out: Path, started: float, scenarios: list[str],
           sched: Schedule, error: str | None, runs_csv: list[dict], p: dict) -> dict:
    """Compare, write raw/rows.csv, raw/runs.csv and result.json; return the result."""
    not_run = list(sched.not_run)
    unconfirmed = list(sched.unconfirmed) if p["reruns"] else []
    rows: list[Row] = []
    for scn, sides in sched.runs.items():
        cand, base = sides["candidate"], sides["baseline"]
        if not cand:
            continue
        if not base:
            # the candidate did not finish and the deadline left no time for the baseline
            rows.append(_zero_one(f"{scn.split('_')[0]} completed", "completed", 0, None))
            continue
        if not base[0].measured and not cand[0].measured:
            # neither side got a live server as far as one check: the harness, not the engine
            not_run.append(f"{scn} (no check recorded on either side: {base[0].reason or 'no checks'})")
            continue
        if not base[0].measured:
            not_run.append(f"{scn} (baseline not measured: {base[0].reason or 'no checks'})")
            continue
        if cand[0].harness_end and base[0].harness_end:
            not_run.append(f"{scn} (both sides ended outside the engine: {cand[0].reason}; {base[0].reason})")
            continue
        scn_rows = scenario_rows(scn, cand, base, p["perf_gate"])
        if p["perf_gate"] and any(r.note == "re-run incomplete" for r in scn_rows):
            unconfirmed.append(f"{scn} (re-run incomplete)")
        rows += scn_rows
    (out / "raw").mkdir(parents=True, exist_ok=True)
    if rows:
        write_rows(out, rows)
    if runs_csv:
        with open(out / "raw" / "runs.csv", "w", newline="") as f:
            cols = ["scenario", "side", "rep", "attempt", "image", "rc", "timed_out", "wall_s",
                    "setup_s", "completed", "reason", "max_load", "contended"]
            w = csv.DictWriter(f, cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(runs_csv)
    verdict, confirmed = verdict_of(rows, not_run + unconfirmed, error)
    reason = error
    if verdict == "error" and not reason:
        reason = "; ".join(x for x in ("not measured: " + ", ".join(not_run) if not_run else "",
                                       "unconfirmed: " + ", ".join(unconfirmed) if unconfirmed else "") if x)
    artifacts = [str(x.relative_to(out)) for x in (out / "raw" / n for n in
                 ("rows.csv", "runs.csv", "servers.json")) if x.exists()]
    artifacts += sorted(str(d.relative_to(out)) for d in (out / "raw").glob("s[0-9][0-9]_*") if d.is_dir())
    result = {
        "schema": 1, "check": a.check, "verdict": verdict, "confirmed": confirmed,
        # the summary always says what went wrong, even when a regression outranks it
        "summary": summary_line(rows, scenarios, not_run, a.mode, p["duration"], error, unconfirmed),
        "candidate": side_of(a.candidate, a.candidate_label, a.candidate_commit),
        "baseline": side_of(a.baseline, a.baseline_label, a.baseline_commit),
        "machine": machine(), "started": utc(started), "finished": utc(time.time()),
        "findings": pick_findings(rows), "error": reason if verdict == "error" else None,
        "artifacts": artifacts,
    }
    return write_result(out, result)


def fallback_result(a: argparse.Namespace, out: Path, started: float, error: str) -> dict:
    """The smallest result.json that says error, when building the real one failed."""
    try:
        mach = machine()
    except Exception:  # noqa: BLE001
        mach = {"runner": "local", "type": "unknown", "cpu": "unknown", "cores": 1}
    result = {"schema": 1, "check": a.check, "verdict": "error", "confirmed": False,
              "summary": LINE_BREAK_RE.sub(" ", error)[:300],
              "candidate": side_of(a.candidate, a.candidate_label, a.candidate_commit),
              "baseline": side_of(a.baseline, a.baseline_label, a.baseline_commit),
              "machine": mach, "started": utc(started), "finished": utc(time.time()),
              "findings": [], "error": error, "artifacts": []}
    (out / "result.json").write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    sys.exit(main())
