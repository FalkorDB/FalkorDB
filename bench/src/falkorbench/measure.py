"""The full-set measurement loop.

For each query: warm it up (plan cache), snapshot jemalloc, run
`redis-benchmark -c 1 -n N` inside a counter window, snapshot jemalloc again.

`redis-benchmark` is the load generator and stays one — see `counters` for why
the thing inside the window has to be an external C process. That does mean the
`ms` column is a closed-loop measurement and a *mean*, not a latency
distribution; it is a coarse outlier net, never the gate. See bench/README.md.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from redis.exceptions import ResponseError

from falkorbench import client as client_mod
from falkorbench.counters import Backend
from falkorbench.metrics import Row
from falkorbench.metrics import read_rows
from falkorbench.metrics import write_rows
from falkorbench.model import CSV_FIELDS
from falkorbench.model import Query


@dataclass
class Calibration:
    """The server process's idle counter rate, subtracted per query.

    Background work (serverCron, bgsave) would otherwise be attributed to
    whichever query happened to be running.

    `usable` is False when a backend exists but returns nothing: `proc_pid_rusage`
    reports ri_instructions = 0 inside a virtualised macOS runner, and selecting
    the backend without checking made the harness emit a column of zeros that
    looked like measurements. A live redis process cannot execute zero
    instructions in three seconds, so zero here means "no counters".
    """

    usable: bool
    instr_per_s: float = 0.0
    cycles_per_s: float = 0.0
    event_rates: dict[str, float] | None = None

    @classmethod
    def measure(cls, backend: Backend, pid: int, seconds: int = 3) -> Calibration:
        reading = backend.run_and_count(pid, ["sleep", str(seconds)])
        if reading.instr is None or reading.instr <= 0 or reading.elapsed <= 0:
            return cls(usable=False)
        return cls(
            usable=True,
            instr_per_s=reading.instr / reading.elapsed,
            cycles_per_s=(reading.cycles or 0.0) / reading.elapsed,
            event_rates={k: v / reading.elapsed for k, v in reading.events.items()},
        )


def _adjust(total: float | None, rate: float, elapsed: float, n: int) -> float | None:
    """Per-execution cost with the idle rate removed.

    Clamped at 0: subtracting the idle rate can overshoot on a short, cheap
    query, and a negative value used to make the gate skip that metric entirely
    (it needs a positive baseline), silently disabling it for that row.
    """
    if total is None:
        return None
    return max(0.0, (total - rate * elapsed) / n)


def measure_queries(
    bench: client_mod.BenchClient,
    backend: Backend,
    queries: Sequence[Query],
    *,
    default_reps: int,
    c_compat: bool = False,
    echo=print,
) -> tuple[dict[str, Row], list[tuple[str, str]]]:
    """Measure each query. Returns (rows, failures).

    A query whose warmup errors is *not* measured. Without that check the
    harness would benchmark the error path and report a plausible, fast row for
    a query that never ran — worse than failing, because the number looks real.
    Failures are collected and reported at the end so one bad query does not
    cost the other 316.
    """
    if not shutil.which("redis-benchmark"):
        raise RuntimeError(
            "redis-benchmark not found on PATH. It is the load generator, not a "
            "convenience: the counter window is defined by its process lifetime."
        )

    pid = bench.pid
    cal = Calibration.measure(backend, pid)
    if cal.usable:
        echo(f"pid {pid}, idle rate {cal.instr_per_s / 1e6:.1f}M instr/s ({backend.name} backend)")
    else:
        echo(
            f"pid {pid}, per-process instruction counters unavailable on {sys.platform} "
            f"({backend.name} backend). Usually a virtualised host with no PMU exposed. "
            f"instr/cycles will be empty rather than zero; alloc_bytes/ms are unaffected."
        )

    # Exec redis-benchmark once and throw the result away, so the first *measured*
    # query is never the first *exec*.
    #
    # This protects the control row specifically. The first redis-benchmark in a
    # fresh session pays binary load, dynamic linking and a cold page cache, and
    # `RETURN 1` — the cheapest query in the set — is measured first, so it
    # absorbs all of it. Measured on two runs of the identical pre-refactor
    # harness: `RETURN 1` came out at 0.3476 ms on the first run and 0.0636 ms on
    # the second, a 5.5x difference, while every other row agreed to within 1%.
    #
    # That is not a cosmetic problem. `RETURN 1` is the control row the CI report
    # divides every wall-clock ratio by, so a one-time cost landing there scales
    # the entire wall-clock column: with the polluted control row, 316 of 317 rows
    # were flagged as having "moved more than 50%".
    warm = [
        "redis-benchmark",
        "-p",
        str(bench.server.port),
        "-c",
        "1",
        "-n",
        "1",
        "GRAPH.RO_QUERY",
        bench.graph_name,
        "RETURN 1",
    ]
    subprocess.run(warm, capture_output=True)

    rows: dict[str, Row] = {}
    failures: list[tuple[str, str]] = []

    for query in queries:
        n = query.reps if query.reps is not None else default_reps
        try:
            bench.run(query.cypher, write=query.write)  # warmup / plan cache
        except ResponseError as e:
            first = str(e).splitlines()[0][:100]
            if c_compat:
                # Expected: the C engine does not implement everything here
                # (UDFs are Rust-only), so skip the row and carry on.
                echo(f"{query.name:<20} SKIPPED (C error: {first})")
            else:
                echo(f"{query.name:<20} FAILED ({first})")
                failures.append((query.name, first))
            continue

        cmd = [
            "redis-benchmark",
            "-p",
            str(bench.server.port),
            "-c",
            "1",
            "-n",
            str(n),
            query.command,
            bench.graph_name,
            query.cypher,
        ]
        # Memory snapshots sit outside the counter window so the MALLOC-STATS
        # call's own work does not land in the instruction counts.
        alloc0, dealloc0 = bench.jemalloc_totals()
        reading = backend.run_and_count(pid, cmd)
        alloc1, dealloc1 = bench.jemalloc_totals()

        row = Row.fromkeys(CSV_FIELDS[1:], None)
        if cal.usable:
            row["instr"] = _adjust(reading.instr, cal.instr_per_s, reading.elapsed, n)
            row["cycles"] = _adjust(reading.cycles, cal.cycles_per_s, reading.elapsed, n)
        if reading.events and cal.event_rates:
            adj = {
                k: _adjust(v, cal.event_rates.get(k, 0.0), reading.elapsed, n) or 0.0
                for k, v in reading.events.items()
            }
            row["branches"] = adj.get("INST_BRANCH")
            row["br_miss"] = adj.get("BRANCH_MISPRED_NONSPEC")
            l1_ld = adj.get("L1D_CACHE_MISS_LD")
            l1_st = adj.get("L1D_CACHE_MISS_ST")
            if l1_ld is not None and l1_st is not None:
                row["l1d_miss"] = l1_ld + l1_st
        if alloc0 is not None and alloc1 is not None:
            row["alloc_bytes"] = (alloc1 - alloc0) / n
            row["dealloc_bytes"] = ((dealloc1 or 0) - (dealloc0 or 0)) / n
        row["ms"] = reading.elapsed / n * 1000

        rows[query.name] = row
        echo(_format_row(query.name, row))

    return rows, failures


def _format_row(name: str, row: Row) -> str:
    def num(key: str, width: int) -> str:
        v = row.get(key)
        return f"{v:>{width},.0f}" if v is not None else f"{'-':>{width}}"

    line = f"{name:<20} {num('instr', 13)} instr {num('cycles', 12)} cyc {row['ms']:>8.3f} ms"
    if row.get("alloc_bytes") is not None:
        line += f" {row['alloc_bytes']:>12,.0f} B alloc"
    return line


def run_once(
    bench: client_mod.BenchClient,
    queries: Sequence[Query],
    error_queries: Sequence[tuple[str, str, str]],
    *,
    include_errors: bool,
    echo=print,
) -> int:
    """Run each query exactly once, unmeasured. Returns the failure count.

    This is what `coverage.sh` drives, which is why it doubles as query
    validation: a non-zero return means the query set no longer runs clean.
    """
    fails = 0
    for query in queries:
        try:
            bench.run(query.cypher, write=query.write)
        except ResponseError as e:
            echo(f"FAIL {query.name}: {str(e).splitlines()[0][:120]}")
            fails += 1

    if include_errors:
        # Expected-error queries cover parse/bind/eval error paths and
        # constraint rollback. Passing means the reply *was* an error.
        for name, command, cypher in error_queries:
            try:
                if command == "GRAPH.RO_QUERY":
                    bench.graph.ro_query(cypher)
                else:
                    bench.graph.query(cypher)
            except ResponseError:
                continue
            except Exception as e:  # any other failure is still a failure
                echo(f"FAIL (expected error) {name}: unexpected {type(e).__name__}: {e}")
                fails += 1
                continue
            echo(f"FAIL (expected error) {name}: query succeeded")
            fails += 1

    echo(f"once-mode done, {fails} failures")
    return fails


def merge_into_csv(out: Path, rows: dict[str, Row]) -> None:
    """Write `rows` into `out`, preserving rows this run did not measure.

    Subset re-runs patch only the queries they measured, which is what makes
    `bench measure "CASE" "WITH pipeline"` useful against an existing CSV.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    merged = read_rows(str(out)) if out.exists() else {}
    merged.update(rows)
    write_rows(str(out), merged.items(), CSV_FIELDS)
