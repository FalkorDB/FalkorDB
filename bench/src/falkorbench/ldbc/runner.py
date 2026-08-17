"""Running the complex reads and recording per-query latency.

Latency percentiles, not the instruction counts the micro-benchmark harness
reports: LDBC's own metric is response time, and these queries are far too
large and too data-dependent for a deterministic instruction count to be the
useful number. The results therefore go to their own CSV and are deliberately
*not* merged into `results/current.csv`, whose thresholds gate the
micro-benchmarks.
"""

from __future__ import annotations

import contextlib
import csv
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from redis.exceptions import ResponseError

from falkorbench.client import BenchClient
from falkorbench.ldbc import params as params_mod
from falkorbench.ldbc import queries as query_mod

Echo = Callable[[str], None]

CSV_FIELDS = (
    "query",
    "runs",
    "rows_total",
    "empty_runs",
    "failures",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "mean_ms",
    "max_ms",
    "rewritten",
)


@dataclass
class QueryResult:
    """One query's timings across its whole parameter set."""

    name: str
    rewritten: bool
    latencies_ms: list[float] = field(default_factory=list)
    rows_total: int = 0
    empty_runs: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def runs(self) -> int:
        return len(self.latencies_ms)

    def pct(self, p: float) -> float | None:
        """The p'th percentile, nearest-rank.

        Nearest-rank rather than interpolation so a reported p99 is a latency
        that actually occurred.
        """
        if not self.latencies_ms:
            return None
        ordered = sorted(self.latencies_ms)
        rank = max(1, min(len(ordered), int(-(-p * len(ordered) // 100))))
        return ordered[rank - 1]

    def row(self) -> dict[str, Any]:
        def fmt(v: float | None) -> str:
            return "" if v is None else f"{v:.3f}"

        return {
            "query": self.name,
            "runs": self.runs,
            "rows_total": self.rows_total,
            "empty_runs": self.empty_runs,
            "failures": len(self.failures),
            "p50_ms": fmt(self.pct(50)),
            "p95_ms": fmt(self.pct(95)),
            "p99_ms": fmt(self.pct(99)),
            "mean_ms": fmt(statistics.fmean(self.latencies_ms) if self.latencies_ms else None),
            "max_ms": fmt(max(self.latencies_ms) if self.latencies_ms else None),
            "rewritten": "yes" if self.rewritten else "",
        }


def run(
    client: BenchClient,
    selected: list[query_mod.ComplexRead],
    param_set: params_mod.ParamSet,
    *,
    warmup: int = 1,
    echo: Echo = print,
) -> list[QueryResult]:
    """Run each query once per parameter row, returning per-query results."""
    results: list[QueryResult] = []
    for query in selected:
        rows = param_set.for_query(query.number)
        result = QueryResult(name=query.name, rewritten=query.rewritten)
        if not rows:
            result.failures.append("no parameters")
            echo(f"{result.name:<6} SKIPPED: no parameters")
            results.append(result)
            continue

        # GraphBLAS compiles kernels on first use, so an unwarmed first call
        # carries a one-off cost that has nothing to do with the query.
        for row in rows[:warmup]:
            with contextlib.suppress(ResponseError, OSError):
                client.graph.ro_query(query.cypher, dict(row), timeout=_TIMEOUT_MS)

        for row in rows:
            started = time.perf_counter()
            try:
                res = client.graph.ro_query(query.cypher, dict(row), timeout=_TIMEOUT_MS)
            except (ResponseError, OSError) as e:
                # Recorded per parameter row rather than aborting: one bad
                # parameter row must not cost the other thirteen queries.
                if len(result.failures) < 5:
                    result.failures.append(str(e)[:200])
                continue
            result.latencies_ms.append((time.perf_counter() - started) * 1000)
            count = len(res.result_set)
            result.rows_total += count
            if count == 0:
                result.empty_runs += 1

        echo(_summary_line(result))
        results.append(result)
    return results


#: Per-query server-side cap. IC14 at SF1 on a cold index can run for minutes,
#: and a hung query should be reported as a failure rather than stalling the run.
_TIMEOUT_MS = 120_000


def _summary_line(result: QueryResult) -> str:
    if not result.latencies_ms:
        return f"{result.name:<6} FAILED  {result.failures[0] if result.failures else ''}"
    flags = []
    if result.rewritten:
        flags.append("rewritten")
    if result.empty_runs:
        flags.append(f"{result.empty_runs}/{result.runs} empty")
    if result.failures:
        flags.append(f"{len(result.failures)} failed")
    suffix = f"   [{', '.join(flags)}]" if flags else ""
    return (
        f"{result.name:<6} p50 {result.pct(50):>9.2f}ms  p99 {result.pct(99):>9.2f}ms  "
        f"{result.rows_total:>8,} rows{suffix}"
    )


def write_csv(path: Path, results: list[QueryResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(result.row())


def problems(results: list[QueryResult]) -> list[str]:
    """Reasons the run should be considered failed.

    A query returning nothing on *every* parameter row counts: it did not error,
    and its latency looks excellent, but it measured nothing. That is the
    failure mode this whole exercise is most likely to produce silently.
    """
    out = []
    for result in results:
        if not result.latencies_ms:
            why = result.failures[0] if result.failures else "no runs"
            out.append(f"{result.name}: produced no measurement ({why})")
        elif result.empty_runs == result.runs:
            out.append(f"{result.name}: returned zero rows on all {result.runs} runs")
        elif result.failures:
            out.append(f"{result.name}: {len(result.failures)} parameter row(s) failed")
    return out
