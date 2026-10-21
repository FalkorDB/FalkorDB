"""Per-process instruction/cycle counters, by platform.

Three backends, chosen by what the host actually provides:

  rusage  macOS. `proc_pid_rusage` gives a running total for any pid with no
          privileges, so a window is read-before / read-after.
  perf    Linux. There is no rusage equivalent; the PMU is reached through
          `perf stat -p <pid> -- <cmd>`, which measures the process for exactly
          as long as `cmd` runs. That is a window measurement rather than a
          running total, which is why the two cannot share one code path.
  null    Neither available. instr/cycles are then reported as *absent*, never
          substituted with wall-clock: a time-based stand-in would turn the
          regression gate into a noise detector while still looking like a
          measurement.

`cmd` is always an external process. That is not incidental — the perf backend
defines its counting window by that process's lifetime, and `pmc_tool`'s
counters are system-wide, so a Python-side loop would put this interpreter's own
work into the numbers. Whatever drives the measured queries stays a C binary.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from typing import NamedTuple

from falkorbench.model import Metric


class Reading(NamedTuple):
    """One measurement window."""

    instr: Metric
    cycles: Metric
    elapsed: float
    events: dict[str, float]


class Rusage(NamedTuple):
    """The three `proc_pid_rusage` fields this harness uses."""

    instructions: int
    cycles: int
    peak_footprint: int


# --- macOS: proc_pid_rusage ---------------------------------------------------

_RUSAGE_INFO_V4 = 4


def read_rusage(pid: int) -> Rusage | None:
    """Running instruction/cycle/peak-footprint totals for `pid`, or None.

    Shared by the measure loop and the flow-test harness; it used to be copied
    into both. None (rather than raising) when the pid is gone, because the flow
    harness polls pids that come and go.
    """
    if sys.platform != "darwin":
        raise RuntimeError("proc_pid_rusage is macOS-only")
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    buf = ctypes.create_string_buffer(1024)
    if libproc.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(_RUSAGE_INFO_V4), buf) != 0:
        return None
    u64 = (ctypes.c_uint64 * 40).from_buffer_copy(buf.raw[16:336])
    # ri_instructions, ri_cycles, ri_lifetime_max_phys_footprint
    return Rusage(u64[29], u64[30], u64[28])


class PmcTool:
    """Optional Apple-silicon PMU counters (branches / branch-misses / L1D).

    `pmc_tool` deliberately does not run the measured command itself: it is
    installed setuid-root, and a setuid binary that execs a caller-supplied
    command is a local privilege escalation (put your own `redis-benchmark`
    earlier in `$PATH` and you have root). It opens a counter window, prints
    READY and waits on stdin; the caller runs the command unprivileged in that
    gap and then closes the window. The counters are system-wide, so bracketing
    in time was always sufficient.
    """

    def __init__(self, path: str) -> None:
        self.path = path

    def works(self) -> bool:
        return self.window(["true"])[0] is not None

    def window(self, cmd: Sequence[str]) -> tuple[dict[str, float] | None, float]:
        """Run `cmd` inside a counter window; return (events, elapsed)."""
        proc = subprocess.Popen(
            [self.path, "window"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            if (proc.stdout.readline() or "").strip() != "READY":  # type: ignore[union-attr]
                proc.kill()
                return None, 0.0
            subprocess.run(cmd, capture_output=True)
            out, _ = proc.communicate("\n", timeout=60)
        except (OSError, subprocess.SubprocessError):
            proc.kill()
            return None, 0.0
        if "EVENT" not in out:
            return None, 0.0
        events: dict[str, float] = {}
        elapsed = 0.0
        for line in out.splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "ELAPSED":
                elapsed = float(parts[1])
            elif parts[0] == "EVENT":
                events[parts[1]] = float(parts[2])
        return events, elapsed


class RusageBackend:
    name = "rusage"

    def __init__(self, pmc: PmcTool | None = None) -> None:
        self.pmc = pmc

    def run_and_count(self, pid: int, cmd: Sequence[str]) -> Reading:
        before = read_rusage(pid)
        if self.pmc is not None:
            events, elapsed = self.pmc.window(cmd)
            events = events or {}
        else:
            events = {}
            t0 = time.time()
            subprocess.run(cmd, capture_output=True)
            elapsed = time.time() - t0
        after = read_rusage(pid)
        if before is None or after is None:
            raise OSError(f"proc_pid_rusage failed for pid {pid} (process gone?)")
        return Reading(
            after.instructions - before.instructions,
            after.cycles - before.cycles,
            elapsed,
            events,
        )


# --- Linux: perf -------------------------------------------------------------


class PerfBackend:
    name = "perf"

    def __init__(self, perf: str) -> None:
        self.perf = perf

    def run_and_count(self, pid: int, cmd: Sequence[str]) -> Reading:
        """instructions/cycles for `pid` while `cmd` runs, plus elapsed seconds.

        `perf stat -p PID -- CMD` attaches to PID, runs CMD, and stops counting
        when CMD exits, so the counters cover exactly the benchmark window.
        `-x,` gives machine-readable `value,unit,event,...` lines on stderr.
        """
        t0 = time.time()
        out = subprocess.run(
            [self.perf, "stat", "-x,", "-e", "instructions,cycles", "-p", str(pid), "--", *cmd],
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - t0
        vals = _parse_perf(out.stderr)
        if "instructions" not in vals or "cycles" not in vals:
            raise OSError(
                "perf stat returned no instructions/cycles. Needs PMU access: "
                "kernel.perf_event_paranoid <= 0 (or CAP_PERFMON), and a host that "
                "exposes the PMU (bare metal or a VM with vPMU enabled). "
                f"stderr: {out.stderr.strip()[:300]}"
            )
        return Reading(vals["instructions"], vals["cycles"], elapsed, {})


def _parse_perf(stderr: str) -> dict[str, float]:
    """`value,unit,event,...` lines -> {event: value}, skipping unavailable ones.

    "<not supported>" / "<not counted>" arrive in the value column and are
    dropped, which is what lets `perf_counters_work` below tell a real PMU from
    a perf binary that runs fine and measures nothing.
    """
    vals: dict[str, float] = {}
    for line in stderr.splitlines():
        parts = line.split(",")
        if len(parts) >= 3:
            try:
                vals[parts[2].strip()] = float(parts[0].strip())
            except ValueError:
                continue
    return vals


def perf_counters_work(perf: str | None) -> bool:
    """True only if perf actually returns counter values.

    The binary being on PATH is not enough: without PMU access (a VM without
    vPMU, or a strict `kernel.perf_event_paranoid`) perf runs fine and reports
    `<not supported>` for every event. Selecting the backend on `which perf`
    alone made the availability check lie, so the graceful-degradation path
    never engaged and a run died on its first measurement instead of reporting
    instr/cycles as absent.
    """
    if not perf:
        return False
    try:
        out = subprocess.run(
            [perf, "stat", "-x,", "-e", "instructions,cycles", "--", "true"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(_parse_perf(out.stderr))


# --- no counters -------------------------------------------------------------


class NullBackend:
    name = "none"

    def run_and_count(self, pid: int, cmd: Sequence[str]) -> Reading:
        t0 = time.time()
        subprocess.run(cmd, capture_output=True)
        return Reading(None, None, time.time() - t0, {})


Backend = RusageBackend | PerfBackend | NullBackend


def select_backend(pmc_path: str | None = None) -> Backend:
    """Pick the counter backend this host can actually support."""
    if sys.platform == "darwin":
        pmc = PmcTool(pmc_path) if pmc_path else None
        if pmc is not None and not pmc.works():
            pmc = None
        return RusageBackend(pmc)
    perf = shutil.which("perf")
    if perf_counters_work(perf):
        return PerfBackend(perf)  # type: ignore[arg-type]
    return NullBackend()
