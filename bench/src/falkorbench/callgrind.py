"""Deterministic per-query instruction counts via callgrind.

Why this exists: no hosted CI runner exposes a PMU. Measured, not assumed — GCE
rejects `--performance-monitoring-unit` on the v1 and beta APIs alike, `perf`
there reports `<not supported>`, macOS runners return 0 for `proc_pid_rusage`'s
ri_instructions, and kperf inside a macOS runner fails with
`kpep_db_create failed: 7`. Hardware counters are simply unavailable.

Callgrind counts instructions in *software*, so it needs no PMU and no
privileges, and its counts are near-deterministic.

## How a query is isolated: differencing, not windowing

Callgrind reports one total when the process exits. The obvious approach is to
window with `callgrind_control --instr=on/off --dump` around each query, and
that is what the first version did. **It does not work in a container.**
`callgrind_control` reaches the process through vgdb FIFOs in /tmp, and
reproduced locally in `debian:trixie-slim` (valgrind 3.24.0, the CI version):

    ==236== open fifo /tmp/vgdb-pipe-from-vgdb-to-236-by-???-on-???
    ==236== valgrind: fatal error: vgdb FIFO cannot be opened.

The server dies on the first control command, so every dump silently never
arrives and every query reports nothing. Setting USER/LOGNAME does not help;
`--vgdb-prefix` makes `callgrind_control` hang instead.

So each query is measured by **differencing two complete runs** of the same
query at different repeat counts:

    T(n2) = startup + setup + compile + n2 * exec
    T(n1) = startup + setup + compile + n1 * exec
    exec  = (T(n2) - T(n1)) / (n2 - n1)

Startup, graph setup and one-time plan compilation appear identically in both
runs and cancel *exactly* — not approximately — because the counts are
deterministic. The price is two valgrind runs per query, each paying setup,
which is why CG_SETUP is deliberately small.

## Precision, and why the span is chosen per query

With the module loaded, CI measured per-run drift of ~300-600k instructions on a
~236M baseline. At a fixed span of 100 that is 3-6k instr/exec of error —
nothing for a 7M-instruction query, but **6.7% for `RETURN 1`**, which is how
the control row once read 1.0673x on two builds whose Rust was byte-identical.
The error is *absolute*, so "treat sub-1% as noise" is wrong in both directions.

The span is therefore chosen per query as `drift / (TARGET_REL * cost)`, holding
the *differenced work* constant instead of the span. Cheap queries get a wide
span (still cheap: `RETURN 1` at span ~3300 is ~300M instructions), expensive
ones keep the default.

## Not comparable to the full measurement set

CG_SETUP builds a 1,000-node graph, not the 10,000-node one, and skips the
vector/fulltext indexes, constraints, UDFs and DEBUG RELOAD. Absolute numbers
here are *not* comparable to `measure` rows — only PR-vs-base ratios are, where
both sides run this identical setup.
"""

from __future__ import annotations

import glob
import math
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from falkorbench.model import Query

# Per-run drift in the whole-process total, measured in CI with the module
# loaded. Divided by the span this becomes the per-execution error, so it is an
# *absolute* budget, not a relative one.
DRIFT_INSTR = 600_000

# Per-execution precision to aim for. span = drift / (TARGET_REL * cost), so the
# differenced work is drift/TARGET_REL = 300M instructions regardless of how
# cheap the query is — a few seconds under valgrind.
TARGET_REL = 0.002
MAX_SPAN = 4000

# Widest per-execution error bar still worth reporting. A row that cannot be
# resolved better than this on the host it ran on is dropped rather than
# printed: a number nobody can reproduce is worse than a gap, because it still
# lands in the table looking like a measurement.
MAX_REL_ERR = 0.02

# A small graph that supports the cg-flagged subset. Deliberately not the full
# SETUP: that builds 10k nodes, 10k edges, vector and fulltext indexes,
# constraints, UDFs and a DEBUG RELOAD, and every one of those instructions
# would be paid twice per measured query under instrumentation.
#
# The Person index is created before the ring so the ring build is index-driven
# rather than a 1000x1000 nested scan.
CG_SETUP: tuple[str, ...] = (
    (
        "UNWIND range(0, 999) AS i "
        "CREATE (:Person {id: i, name: 'p' + toString(i), age: i % 80, score: i * 1.5})"
    ),
    "CREATE INDEX FOR (p:Person) ON (p.id)",
    (
        "UNWIND range(0, 999) AS i "
        "MATCH (a:Person {id: i}) MATCH (b:Person {id: (i + 1) % 1000}) "
        "CREATE (a)-[:KNOWS]->(b)"
    ),
    # `delete node` deletes one :Tmp per execution, so there must be more of them
    # than the highest repeat count — MAX_SPAN plus n1, since a cheap delete
    # query gets its span widened. Running dry would not fail loudly: the
    # remaining executions would measure a no-op delete and quietly halve the
    # reported cost.
    "UNWIND range(0, 4999) AS i CREATE (:Tmp {x: i})",
)


@dataclass
class Measurement:
    query: str
    instr: float
    span: int
    rel_err: float
    drift: float
    seconds: float
    widened_from: int | None = None


class Skipped(Exception):
    """This query produced no usable number, with the reason as the message."""


def parse_total(path: str) -> int | None:
    """Instruction count from a callgrind output file.

    Callgrind writes `totals:` (and `summary:`) with the first field being
    instruction reads. None when neither is present, which happens for a file
    still being written.
    """
    try:
        with open(path, errors="replace") as f:
            for line in f:
                if line.startswith(("totals:", "summary:")):
                    parts = line.split(":", 1)[1].split()
                    if parts:
                        return int(parts[0])
    except OSError:
        return None
    return None


@dataclass
class Runner:
    """Runs one instrumented server lifecycle and reads its instruction total.

    `bare` runs a plain redis-server with no module and no graph setup. That is
    not a toy mode: valgrind on arm64 cannot execute this module at all
    (`unhandled instruction 0xB8BFC108` — an ARMv8.1 LSE atomic in RediSearch's
    slots_tracker, valgrind's limitation rather than a module bug), so the
    differencing arithmetic above can only be validated against bare redis on
    that architecture. It used to require editing two module-level constants;
    making it a flag is what lets the arm64 validation run in CI or locally
    without a patched checkout.
    """

    module: Path | None
    port: int
    outdir: Path
    module_args: Sequence[str] = ()
    bare: bool = False

    @property
    def setup(self) -> tuple[str, ...]:
        return () if self.bare else CG_SETUP

    def total(self, cypher: str, reps: int, also_run: Sequence[str] = ()) -> int:
        """One instrumented lifecycle; returns its whole-process instruction total."""
        shutil.rmtree(self.outdir, ignore_errors=True)
        self.outdir.mkdir(parents=True, exist_ok=True)

        argv = [
            "valgrind",
            "--tool=callgrind",
            f"--callgrind-out-file={self.outdir}/callgrind.out.%p",
            "redis-server",
            "--port",
            str(self.port),
            "--save",
            "",
            # serverCron does work proportional to how long the process lives, and
            # the two runs being differenced live for different durations — so
            # cron lands in the subtraction as drift. Measured at the default
            # hz=10 it is ~240k instr per second of life, which swamped a PING
            # (~20k) and made two (n1,n2) pairs disagree by 44%. hz=1 is the
            # lowest redis accepts and cuts it 10x.
            "--hz",
            "1",
        ]
        if self.module is not None and not self.bare:
            argv += ["--loadmodule", str(self.module), *self.module_args]

        server = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            self._wait_ready(server)
            for stmt in self.setup:
                self._cli(*self._graph_cmd(stmt))
            for extra in also_run:
                self._cli(*self._graph_cmd(extra), check=False)
            if reps:
                # One redis-cli with -r, not `reps` of them: a fresh connection
                # per execution would put accept/handshake/teardown into the
                # measurement, and the extra wall time feeds the cron drift.
                self._cli("-r", str(reps), *self._graph_cmd(cypher))
            self._cli("shutdown", "nosave", check=False)
            server.wait(timeout=600)
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    server.kill()

        # Redis forks and valgrind profiles the child too; the child's total is
        # tiny, so the server's own run is the maximum.
        totals = [
            t
            for t in (parse_total(p) for p in glob.glob(str(self.outdir / "callgrind.out.*")))
            if t is not None
        ]
        if not totals:
            raise Skipped(f"no parseable callgrind output in {self.outdir}")
        return max(totals)

    def _graph_cmd(self, cypher: str) -> tuple[str, ...]:
        # In bare mode there is no module, so the payload is a plain PING-ish
        # command the differencing can still be validated against.
        return (cypher,) if self.bare else ("GRAPH.QUERY", "bench", cypher)

    def _wait_ready(self, server: subprocess.Popen) -> None:
        for _ in range(1200):  # instrumented startup is far slower than native
            if server.poll() is not None:
                raise Skipped("server exited during startup under callgrind")
            if self._cli("ping", check=False).strip() == "PONG":
                return
            time.sleep(0.5)
        raise Skipped("server did not answer PING under callgrind")

    def _cli(self, *args: str, check: bool = True, timeout: int = 1800) -> str:
        """redis-cli, kept deliberately: under callgrind this *is* the measured
        workload, and `-r N` drives N executions over one connection from C.

        Under callgrind a query can take minutes, but never forever: without a
        timeout a wedged redis-cli would hang the whole job instead of failing
        the one query.
        """
        try:
            out = subprocess.run(
                ["redis-cli", "-p", str(self.port), *args],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise Skipped(f"redis-cli {' '.join(args)} timed out after {timeout}s") from e
        if check and out.returncode != 0:
            raise Skipped(
                f"redis-cli {' '.join(args)} failed (exit {out.returncode}): "
                f"{out.stderr.strip()[:200]}"
            )
        return out.stdout


def measure_one(runner: Runner, query: Query, n1: int, n2: int) -> Measurement:
    """Instruction cost per execution for one query. Raises Skipped if unusable."""
    started = time.time()
    span = n2 - n1

    lo = runner.total(query.cypher, n1)
    hi = runner.total(query.cypher, n2)
    if hi <= lo:
        # Deterministic counts cannot go down when work is added; if this trips,
        # the two runs did not differ only by repeat count.
        raise Skipped(
            f"T(n2)={hi:,} <= T(n1)={lo:,} — the runs are not differing only by repeat count"
        )

    # Measure this engine's drift instead of assuming DRIFT_INSTR, which was
    # calibrated on the Rust module. The C engine cannot be pinned as tightly
    # (THREAD_COUNT 0 refuses to start, so a worker thread always exists and
    # valgrind schedules the handoff nondeterministically) and its real drift is
    # far larger. Assuming the calibrated value there produced `RETURN 1` at
    # 4,053,858 instructions against a true cost near 150k, reported as ±0.15%
    # because the error estimate divides by the very number the drift inflated:
    # two self-reinforcing failures, since a too-high cost also asks for a
    # too-small span so the widening never fires.
    #
    # `rep` re-runs n1. Identical work, so the whole difference is drift.
    rep = runner.total(query.cypher, n1)
    # One difference of two samples is a noisy estimator of the spread — roughly
    # |N(0, s*sqrt(2))|, which lands below the true s about half the time.
    # Measured: taking it at face value dropped `RETURN 1` from span 3193 to 991
    # and residual noise on a null comparison went from under 0.5% to 1.5%. So
    # DRIFT_INSTR stays a conservative floor and the measurement only takes over
    # when it is larger — the case this exists for.
    drift = max(abs(rep - lo), DRIFT_INSTR)
    base_lo = min(lo, rep)

    per = (hi - base_lo) / span
    used_span, widened_from = span, None

    if per > 0 and drift > 0:
        want = min(math.ceil(drift / (TARGET_REL * per)), MAX_SPAN)
        if want > span:
            try:
                hi2 = runner.total(query.cypher, n1 + want)
            except Skipped:
                hi2 = None
            if hi2 is not None and hi2 > base_lo:
                per = (hi2 - base_lo) / want
                used_span, widened_from = want, span

    rel = (drift / used_span / per) if per > 0 else float("inf")
    if rel > MAX_REL_ERR:
        raise Skipped(
            f"±{rel * 100:.1f}% at span {used_span} (drift {drift:,.0f} over two "
            f"identical n1={n1} runs) — too noisy on this host to report"
        )

    return Measurement(
        query=query.name,
        instr=per,
        span=used_span,
        rel_err=rel,
        drift=drift,
        seconds=time.time() - started,
        widened_from=widened_from,
    )


def shard(queries: Sequence[Query], spec: str, total_hint: int | None = None) -> list[Query]:
    """`I/N` -> shard I of N, round-robin.

    Round-robin rather than contiguous: per-query cost varies 11-17s and the
    expensive ones cluster by family, so contiguous chunks would finish at
    noticeably different times.
    """
    try:
        i_str, n_str = spec.split("/", 1)
        i, n = int(i_str), int(n_str)
    except ValueError as e:
        raise ValueError(f"--shard wants I/N, got {spec!r}") from e
    if not 1 <= i <= n:
        raise ValueError(f"--shard {spec}: need 1 <= I <= N")
    if total_hint is not None and total_hint != n:
        raise ValueError(
            f"--shard {spec} disagrees with the job total ({total_hint}); the shard "
            f"count must match the matrix or part of the subset is never measured"
        )
    return list(queries)[i - 1 :: n]


def require_tools() -> None:
    for tool in ("valgrind", "redis-server", "redis-cli"):
        if not shutil.which(tool):
            raise RuntimeError(
                f"{tool} not found. This needs valgrind and redis on PATH. Note valgrind "
                f"does not support macOS on Apple silicon, so this is Linux-only in practice."
            )


def default_outdir(root: Path) -> Path:
    return root / "results/callgrind"


def bare_payload() -> Query:
    """The payload used in `--bare` mode: a command any redis answers."""
    return Query(name="PING", write=False, cypher="ping", cg=True)


def resolve_module(module: str | None, bare: bool) -> Path | None:
    if bare:
        return None
    if module is None:
        raise RuntimeError("--module is required unless --bare is given")
    path = Path(module).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"module not found: {path}")
    return path


