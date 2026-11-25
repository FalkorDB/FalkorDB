"""`bench micro-hw`: the micro-hw check's command, on a VM that exposes its PMU.

Runs the full query set for both images through `perf stat`, runs it a second
time when a row crosses, and always writes `<out>/result.json`; verdict.py
decides what the readings say. How it runs and why: bench/README.md, "Nightly
micro check".
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from falkorbench import kit
from falkorbench import verdict
from falkorbench.counters import L2_EVENT_ENV
from falkorbench.counters import L2_MISS_EVENT
from falkorbench.counters import PERF_EVENTS
from falkorbench.counters import _parse_perf
from falkorbench.model import Query

MODULE_PATH = "/var/lib/falkordb/bin/falkordb.so"

# Engine (THREAD_COUNT) and OpenMP (OMP_NUM_THREADS) threads for every pass.
# Allocations and counters depend on them, and an unpinned server would size
# both to whatever VM came up. 4 is a hosted runner's vCPU count, and with it the
# C4's allocations match a hosted runner's (median 1.000 over 373 queries).
MEASURE_THREADS = 4
SCREEN_ORDER = (("candidate", 1), ("baseline", 1), ("baseline", 2), ("candidate", 2))
CONFIRM_ORDER = (("baseline", 1), ("candidate", 1), ("candidate", 2), ("baseline", 2))

# What a Rust cdylib always carries and the C module never does: the standard
# library's source paths, kept in its panic locations even when stripped.
RUST_MARKER = b"/rustc/"

Runner = Callable[[Sequence[str]], subprocess.CompletedProcess]


def default_runner(argv: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(argv), capture_output=True, text=True)


class HwError(RuntimeError):
    """The run cannot produce a measurement; the message goes into `error`."""


def is_c_module(path: Path) -> bool:
    """True for the C engine's module: it has no Rust standard library in it."""
    return RUST_MARKER not in path.read_bytes()


@dataclass
class HwRun:
    bench_image: str
    # The contract flags: candidate, baseline, out, mode, labels and commits.
    args: Any
    repo_root: Path
    server_cpus: str | None = None
    queries: Sequence[Query] = ()
    # Runs one argv; default_runner when None. Not a class-level function
    # default, which would read as a bound method.
    runner: Runner | None = None
    echo: Callable[[str], None] = print
    pull_attempts: int = 3
    c_compat: bool = False
    notes: dict = field(default_factory=dict)
    # The screen's readings, kept as soon as they exist: a re-run that fails
    # must not throw away a regression the screen already found (a correctness
    # failure needs no re-run, and a regression outranks an error).
    screened: verdict.Inputs | None = None
    # The L2-miss event the probe found counting; passed to every pass.
    l2_event: str | None = None
    # Second-run passes that completed and passed their checks in this run.
    second_run_done: set[tuple[str, int]] = field(default_factory=set)

    def __post_init__(self) -> None:
        # `_perf-vm.yml` passes `--out out`, relative to where the check starts
        # (contract v1.8). Resolved now, before a docker mount or anything that
        # changes directory can read it against some other directory.
        self.args.out = str(Path(self.args.out).resolve())
        self.repo_root = Path(self.repo_root).resolve()

    def _call(self, argv: Sequence[str]) -> subprocess.CompletedProcess:
        return (self.runner or default_runner)(argv)

    @property
    def out(self) -> Path:
        return Path(self.args.out)

    @property
    def raw(self) -> Path:
        return self.out / "raw"

    # --- docker ------------------------------------------------------------

    def _ok(self, argv: Sequence[str], what: str) -> subprocess.CompletedProcess:
        proc = self._call(argv)
        if proc.returncode != 0:
            raise HwError(
                f"{what} failed (exit {proc.returncode}): {(proc.stderr or '').strip()[:300]}"
            )
        return proc

    def pull(self, image: str) -> None:
        for attempt in range(1, self.pull_attempts + 1):
            if self._call(["docker", "pull", "-q", image]).returncode == 0:
                return
            if attempt < self.pull_attempts:
                self.echo(f"pull of {image} failed (attempt {attempt}); retrying")
                time.sleep(15 * attempt)
        raise HwError(f"could not pull {image} after {self.pull_attempts} attempts")

    def extract(self, image: str, dest: Path) -> None:
        cid = self._ok(["docker", "create", image], f"docker create {image}").stdout.strip()
        try:
            self._ok(
                ["docker", "cp", f"{cid}:{MODULE_PATH}", str(dest)], f"copy module out of {image}"
            )
        finally:
            self._call(["docker", "rm", "-f", cid])

    def _container(self) -> list[str]:
        argv = ["docker", "run", "--rm", "--privileged", "-e", f"OMP_NUM_THREADS={MEASURE_THREADS}"]
        if self.l2_event:
            argv += ["-e", f"{L2_EVENT_ENV}={self.l2_event}"]
        if self.server_cpus:
            argv += ["--cpuset-cpus", self.server_cpus]
        return [
            *argv,
            "-v",
            f"{self.repo_root}:/w",
            "-v",
            f"{self.raw}:/data",
            "-w",
            "/w",
            self.bench_image,
        ]

    def probe(self) -> dict[str, float]:
        """The counters the measurement container can open, or HwError."""
        events = ",".join(["instructions", "cycles", *PERF_EVENTS])
        proc = self._call([*self._container(), "perf", "stat", "-x,", "-e", events, "--", "true"])
        (self.raw / "perf-probe.txt").write_text((proc.stdout or "") + (proc.stderr or ""))
        vals = _parse_perf(proc.stderr or "")
        # A counter must read above 0, not merely be present: under KVM's PMU
        # event filter a filtered event is still opened and reads 0 in the
        # guest. Even `true` misses L2 several thousand times (6,501-7,477 on
        # c4-standard-8, 2026-11-24).
        vals = {ev: n for ev, n in vals.items() if n > 0}
        if "instructions" not in vals or "cycles" not in vals:
            raise HwError(
                "no hardware counters in the measurement container (perf stat counted "
                f"neither instructions nor cycles; see raw/perf-probe.txt): "
                f"{(proc.stderr or '').strip()[:200]}"
            )
        # Every gated counter, before the first pass rather than after it.
        missing = [ev for ev in PERF_EVENTS if ev not in vals]
        if missing:
            raise HwError(
                f"this host's PMU does not count {', '.join(missing)} (see raw/perf-probe.txt)"
            )
        # The L2-miss event, in a call of its own: perf rejects an event name it
        # does not know outright, which would sink the others with it.
        p = self._call(
            [*self._container(), "perf", "stat", "-x,", "-e", L2_MISS_EVENT, "--", "true"]
        )
        (self.raw / "perf-probe.txt").write_text(
            (self.raw / "perf-probe.txt").read_text() + (p.stdout or "") + (p.stderr or "")
        )
        got = _parse_perf(p.stderr or "")
        if got.get(L2_MISS_EVENT, 0) <= 0:
            listing = self._call([*self._container(), "perf", "list", "--no-desc"])
            (self.raw / "perf-list.txt").write_text(listing.stdout or listing.stderr or "")
            raise HwError(
                f"this host's PMU does not count {L2_MISS_EVENT} (see raw/perf-probe.txt "
                "and raw/perf-list.txt); micro-hw never substitutes another L2 event"
            )
        self.l2_event = L2_MISS_EVENT
        vals[L2_MISS_EVENT] = got[L2_MISS_EVENT]
        self.notes["l2_event"] = L2_MISS_EVENT
        return vals

    def measure(self, side: str, rep: int, *, confirm: bool) -> None:
        path_of = verdict.confirm_path if confirm else verdict.screen_path
        csv = path_of(self.raw, side, rep)
        log = csv.with_suffix(".log")
        argv = [
            *self._container(),
            "bench",
            "measure",
            "--module",
            f"/data/{side}.so",
            "--out",
            f"/data/{csv.name}",
            "--module-args",
            "THREAD_COUNT",
            "--module-args",
            str(MEASURE_THREADS),
        ]
        if self.c_compat:
            argv.append("--c-compat")
        started = time.time()
        proc = self._call(argv)
        log.write_text((proc.stdout or "") + (proc.stderr or ""))
        # Exit 1 with a CSV is a query that errored on this engine: its row is
        # left out and the verdict reads that as a correctness finding.
        if proc.returncode not in (0, 1) or not csv.is_file():
            raise HwError(
                f"bench measure {csv.name} failed (exit {proc.returncode}); see raw/{log.name}"
            )
        # Each pass must have counted. `bench measure` falls back to no counters
        # on its own when perf fails, and one such pass among four would leave
        # the verdict reading the others as if nothing were missing.
        rows = verdict.read_pass(csv)
        silent = [
            spec.column
            for spec in self.metrics
            if rows and not any((r.get(spec.column) or 0) > 0 for r in rows.values())
        ]
        if silent:
            raise HwError(f"{csv.name} counted no {', '.join(silent)}; see raw/{log.name}")
        self.echo(f"{csv.name}: {time.time() - started:.0f}s")

    # --- the check ---------------------------------------------------------

    def inputs(self) -> verdict.Inputs:
        return verdict.Inputs(
            queries=self.queries,
            screen={side: verdict.load(self.raw, side) for side in verdict.SIDES},
            metrics=self.metrics,
        )

    @property
    def metrics(self) -> tuple[verdict.MetricSpec, ...]:
        """The C engine's gates when either side is C: its noise is its own."""
        return verdict.hw_metrics("c" if self.c_compat else "rust")

    def second_run(self) -> dict[str, verdict.Passes]:
        """The second run's readings, from the passes that completed in this run.

        Read on the failure path too: a regression that the completed passes
        repeat still outranks the error of the pass that failed.
        """
        return {
            side: verdict.Passes(
                reps=[
                    verdict.read_pass(verdict.confirm_path(self.raw, side, rep))
                    if (side, rep) in self.second_run_done
                    else None
                    for rep in range(1, verdict.REPS + 1)
                ]
            )
            for side in verdict.SIDES
        }

    def execute(self) -> verdict.Inputs:
        self.raw.mkdir(parents=True, exist_ok=True)
        # `bench measure` merges into a CSV that already exists, so an `--out`
        # used before would keep that run's rows: a query that errors now would
        # read as its old self. Every pass starts from no file.
        for old in [*self.raw.glob("measure-*.csv"), *self.raw.glob("confirm-measure-*.csv")]:
            old.unlink()
        for image in (self.bench_image, self.args.candidate, self.args.baseline):
            self.pull(image)
        for side in verdict.SIDES:
            self.extract(getattr(self.args, side), self.raw / f"{side}.so")
        c_sides = [s for s in verdict.SIDES if is_c_module(self.raw / f"{s}.so")]
        # Both sides, or the two would measure different graphs.
        self.c_compat = bool(c_sides)
        self.notes.update(c_engine=c_sides, c_compat=self.c_compat, server_cpus=self.server_cpus)
        self.notes["probe"] = self.probe()
        for side, rep in SCREEN_ORDER:
            self.measure(side, rep, confirm=False)
        inp = self.screened = self.inputs()
        results, _, _ = verdict.assess(inp)
        crossed = verdict.to_confirm(results)
        self.notes["crossed"] = sorted({f"{r.query} [{r.metric}]" for r in crossed})
        if crossed:
            self.echo(f"{len(crossed)} row(s) crossed; re-running the whole set")
            for side, rep in CONFIRM_ORDER:
                self.measure(side, rep, confirm=True)
                self.second_run_done.add((side, rep))
            inp.confirm = self.second_run()
        return inp


def run_check(hw: HwRun, *, started: str | None = None) -> dict:
    """Run the check and write result.json, whatever happens (kit.run_check)."""
    started = started or kit.utc_now()
    measured: list[verdict.Inputs] = []
    errors: list[str] = []

    def work() -> None:
        if kit.input_errors(hw.args):
            return  # the result says why; nothing is pulled for an unpinned image
        try:
            if shutil.which("docker") is None and hw.runner is None:
                raise HwError("docker is not on PATH")
            measured.append(hw.execute())
        except HwError as e:
            errors.append(str(e))

    def build(error: str | None) -> dict:
        inp = measured[0] if measured else hw.screened
        if not measured and inp is not None:
            # A pass failed after the screen finished: judge the screen with the
            # second-run passes that did complete. Crossed rows with no second-run
            # reading are errors there; a correctness failure, or a regression the
            # completed passes repeat, still outranks the error.
            inp.confirm = hw.second_run()
        hw.raw.mkdir(parents=True, exist_ok=True)
        (hw.raw / "hw.json").write_text(json.dumps(hw.notes, indent=2, default=str) + "\n")
        # The modules are inputs, not results; keep the artifact to what was measured.
        for side in verdict.SIDES:
            (hw.raw / f"{side}.so").unlink(missing_ok=True)
        files = sorted(f"raw/{p.name}" for p in hw.raw.iterdir() if p.is_file())
        res, rows = verdict.build_result(
            inp,
            hw.args,
            started=started,
            extra_errors=[*errors, *([error] if error else [])],
            artifacts=["raw/rows.csv", *files],
            metrics=hw.metrics,
        )
        verdict.write_rows_csv(hw.raw / "rows.csv", rows)
        return res

    return kit.run_check(verdict.CHECK, hw.args, work, build)
