"""Per-flow-test-file measurement: server instructions/cycles/peak memory.

Runs each flow test file through ./flow.sh (RLTest, --parallelism 1) against a
given module. RLTest spawns transient redis-servers, and macOS `wait4` rusage
does not fold grandchildren, so a background thread polls every redis-server pid
that appears during the run and keeps its last-seen counters; each row is the
SUM over the servers that file spawned. Teardown work after the final poll
(~25 ms) is lost, which is fine for comparisons.

macOS-only, because `proc_pid_rusage` has no Linux equivalent — `measure` does
have a perf-based Linux backend, this does not.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from falkorbench.counters import read_rusage

SUMMARY_RE = re.compile(r"Total Tests Run:\s*(\d+), Total Tests Failed:\s*(\d+)")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

FIELDS = (
    "file",
    "wall_s",
    "instr",
    "cycles",
    "peak_mem_mb",
    "servers",
    "tests_run",
    "tests_failed",
)


@dataclass
class FlowRow:
    file: str
    wall_s: float
    instr: int
    cycles: int
    peak_mem_mb: float
    servers: int
    tests_run: int
    tests_failed: int

    def as_dict(self) -> dict[str, object]:
        return {
            "file": self.file,
            "wall_s": round(self.wall_s, 2),
            "instr": self.instr,
            "cycles": self.cycles,
            "peak_mem_mb": round(self.peak_mem_mb, 1),
            "servers": self.servers,
            "tests_run": self.tests_run,
            "tests_failed": self.tests_failed,
        }


def require_macos() -> None:
    if sys.platform != "darwin":
        raise RuntimeError(
            "flow measurement needs macOS: per-process instruction counters come from "
            "proc_pid_rusage, which Linux does not provide. Use `bench measure`, which "
            "has a perf-based Linux backend."
        )


def redis_pids() -> set[int]:
    # redis rewrites its proc title to "redis-server *:PORT", so -x won't match.
    out = subprocess.run(["pgrep", "-f", "redis-server"], capture_output=True, text=True).stdout
    return {int(p) for p in out.split()}


def flow_files(root: Path) -> list[str]:
    listing = root / "flow_tests_done.txt"
    return [
        line.strip().removesuffix(".py")
        for line in listing.read_text().splitlines()
        if line.strip()
    ]


def run_one(root: Path, test: str, env: dict[str, str]) -> FlowRow:
    """Run one flow file, summing counters over every server it spawns."""
    pre = redis_pids()
    seen: dict[int, tuple[int, int, int]] = {}
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            for pid in redis_pids() - pre:
                r = read_rusage(pid)
                if r is not None:
                    seen[pid] = (r.instructions, r.cycles, r.peak_footprint)
            stop.wait(0.025)

    thread = threading.Thread(target=poll, daemon=True)
    started = time.time()
    thread.start()
    out = subprocess.run(
        ["./flow.sh"],
        cwd=root,
        capture_output=True,
        text=True,
        env={**env, "TEST": test},
    )
    stop.set()
    thread.join()
    wall = time.time() - started

    text = ANSI_RE.sub("", out.stdout + out.stderr)
    m = SUMMARY_RE.search(text)
    run, failed = (int(m.group(1)), int(m.group(2))) if m else (0, -1)

    return FlowRow(
        file=os.path.basename(test),
        wall_s=wall,
        instr=sum(v[0] for v in seen.values()),
        cycles=sum(v[1] for v in seen.values()),
        peak_mem_mb=sum(v[2] for v in seen.values()) / 1e6,
        servers=len(seen),
        tests_run=run,
        tests_failed=failed,
    )


def build_env(
    root: Path, module: Path
) -> tuple[dict[str, str], tempfile.TemporaryDirectory | None]:
    """flow.sh hardcodes the module filename, so a foreign module is symlinked
    under the expected name (this is how the C module gets measured)."""
    env = {**os.environ, "VERBOSE": "0", "PARALLELISM": "--parallelism 1"}
    target = "libfalkordb.dylib" if sys.platform == "darwin" else "libfalkordb.so"
    tmp: tempfile.TemporaryDirectory | None = None
    if module.name == target:
        env["TARGET_DIR"] = str(module.parent)
    else:
        tmp = tempfile.TemporaryDirectory()
        os.symlink(module, Path(tmp.name) / target)
        env["TARGET_DIR"] = tmp.name
    return env, tmp


def merge_csv(out: Path, rows: Sequence[FlowRow]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, dict[str, object]] = {}
    if out.exists():
        with open(out, newline="") as f:
            merged = {r["file"]: dict(r) for r in csv.DictReader(f)}
    for row in rows:
        merged[row.file] = row.as_dict()
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(merged.values())


def compare_csvs(current: Path, baseline: Path) -> list[str]:
    """C-vs-Rust table. Tests may legitimately fail on C (Rust-specific
    behaviours), so both failure counts are shown rather than gated."""
    with open(current, newline="") as f:
        cur = {r["file"]: r for r in csv.DictReader(f)}
    with open(baseline, newline="") as f:
        base = {r["file"]: r for r in csv.DictReader(f)}

    def ratio(b: dict[str, str], c: dict[str, str], key: str) -> str:
        try:
            bv, cv = float(b.get(key) or 0), float(c.get(key) or 0)
        except ValueError:
            return f"{'-':>6}"
        return f"{cv / bv:>6.2f}" if bv else f"{'-':>6}"

    lines = [
        f"{'file':<32} {'base instr':>13} {'cur instr':>13} {'ratio':>6}  "
        f"{'cyc':>6} {'mem':>6} {'wall':>6}  {'run':>4} {'fail b/c':>8}"
    ]
    for name, b in base.items():
        c = cur.get(name)
        if c is None:
            lines.append(f"{name:<32} MISSING from current")
            continue
        lines.append(
            f"{name:<32} {float(b['instr']):>13,.0f} {float(c['instr']):>13,.0f} "
            f"{ratio(b, c, 'instr')}  {ratio(b, c, 'cycles')} "
            f"{ratio(b, c, 'peak_mem_mb')} {ratio(b, c, 'wall_s')}  "
            f"{c['tests_run']:>4} {b['tests_failed']:>3}/{c['tests_failed']}"
        )
    for name in cur:
        if name not in base:
            lines.append(f"{name:<32} NEW (not in baseline)")
    return lines
