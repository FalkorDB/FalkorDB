"""Value types shared across the harness. No I/O, no dependencies.

The important one is `Metric`. The pre-refactor code carried three different
spellings of "this number is absent" — `""` in a CSV cell, `None` from a failed
parse, and `0.0` from a counter backend that reported nothing — and re-derived
which was which with a `try/except ValueError` at every use. Two bugs came out
of that: a `0` read as a real measurement (making a row look infinitely
regressed) and a negative value silently disabling the gate for a row. So
absence has exactly one representation here, `None`, established once when a
CSV is parsed (`metrics.read_rows`) and never re-parsed downstream.
"""

from typing import NamedTuple

# A measurement, or None when the metric was not available on that host: no PMU
# for instructions, a libc-malloc redis for allocated bytes. Never 0.0 for
# "absent" — zero is a legitimate measurement and must stay distinguishable.
Metric = float | None

# Metrics in CSV column order. `ms` is last because it is the only non-
# deterministic one; see metrics.THRESHOLDS for how each is treated.
METRIC_NAMES = (
    "instr",
    "cycles",
    "branches",
    "br_miss",
    "l1d_miss",
    "alloc_bytes",
    "dealloc_bytes",
    "ms",
)

CSV_FIELDS = ("query", *METRIC_NAMES)


class Query(NamedTuple):
    """One benchmark query.

    Was a bare tuple unpacked as `for name, is_write, q, *rest` in three
    separate modules, which is what made the optional 4th element awkward and
    the 5th impossible.

    name    stable identifier; the CSV key and what `--subset` matches on
    write   send as GRAPH.QUERY rather than GRAPH.RO_QUERY
    cypher  the query text
    reps    override the default repeat count. The sized write queries set this
            so a 1M-row batch does not run 1000 times.
    cg      runnable under callgrind: needs nothing beyond CG_SETUP's 1,000
            :Person / :KNOWS ring / 5,000 :Tmp, and does not drain a pool
            faster than it is refilled. Running dry does not fail loudly — it
            quietly measures a no-op — hence the explicit opt-in.
    """

    name: str
    write: bool
    cypher: str
    reps: int | None = None
    cg: bool = False

    @property
    def command(self) -> str:
        return "GRAPH.QUERY" if self.write else "GRAPH.RO_QUERY"
