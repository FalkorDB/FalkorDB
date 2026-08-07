"""Reading measurements and turning them into ratios. Pure functions.

This module exists because there used to be two of it. `compare.py` (the local
gate) and `report.py` (the CI comment) each implemented CSV parsing, ratios and
thresholds independently, and disagreed on the same input: `compare.py` gated
raw wall-clock at 1.25x, while `report.py` had learned — from a null comparison
where the median PR/base `ms` ratio came out at 1.464 on byte-identical engines
— that raw wall-clock across two hosts is not comparable at all. Every guard
below was learned once, in one of the two, and is now available to both.

Nothing here does I/O beyond reading a CSV path, so all of it is testable
without a server.
"""

from __future__ import annotations

import csv
import glob
import math
from collections.abc import Iterable

from falkorbench.model import METRIC_NAMES
from falkorbench.model import Metric

# --- thresholds --------------------------------------------------------------

# Per-metric regression threshold for the local gate. Deterministic counters are
# gated tightly; the miss counters and wall-clock move run to run, so they get
# slack. Note `ms` is gated loosely *and* is never the only evidence — see
# `normalise_ms` for why a cross-host `ms` ratio needs a control row first.
THRESHOLDS: dict[str, float] = {
    "instr": 1.10,
    "cycles": 1.10,
    "branches": 1.10,
    "br_miss": 1.25,
    "l1d_miss": 1.25,
    "alloc_bytes": 1.10,
    "dealloc_bytes": 1.10,
    "ms": 1.25,
}

# What the gate actually fails on. `ms` is deliberately excluded.
#
# Wall-clock is never the gate — the harness says so everywhere else, and then
# the local gate failed builds on it anyway. Control-row normalisation does not
# rescue it either: that cancels a *uniform* per-host scale factor, and measured
# locally on two runs of identical code the control row agreed to 1.00x while
# individual rows still moved 4x. Most queries here cost 0.02-0.3 ms, so a
# ratio's denominator is a couple of hundred microseconds of process scheduling.
#
# It is still parsed, still displayed, and still reported as a coarse outlier net
# by the CI reporter. `--metrics ms` opts into gating on it deliberately.
GATED_BY_DEFAULT: tuple[str, ...] = tuple(m for m in THRESHOLDS if m != "ms")

# Ranking by allocated bytes needs a magnitude floor. Most rows allocate a
# couple of KB, and a ratio over a ~2 KB denominator is noise dressed as signal.
# Rows below the floor are still measured and still in the CSV; they are just
# not ranked.
ALLOC_FLOOR = 65536

# The fixed per-query floor. Its PR/base wall-clock ratio is the per-host speed
# difference, since it does almost no query work.
CONTROL_QUERY = "RETURN 1"

# How far a control-normalised wall-clock ratio must move to be worth printing.
# Measured on identical engines: median 0.973, p99 1.110, max 1.17. 50% is well
# clear of that.
MS_THRESHOLD = 0.50

# A C row is reported as a range when two passes disagree by more than this.
C_TOLERANCE = 0.02

# Below this, a callgrind instruction count for the C engine is an error reply
# rather than an executed query.
C_ERROR_FLOOR = 20000


# --- reading -----------------------------------------------------------------


class Row(dict[str, Metric]):
    """One CSV row: metric name -> value or None. Never a string, never "".

    Subclassing dict rather than wrapping it keeps `row["instr"]` working while
    guaranteeing the values have already been through `_parse`. The whole point
    is that absence is decided once, here, instead of at every use site.
    """


def _parse(raw: str | None) -> Metric:
    """A CSV cell as a float, or None when it holds no measurement.

    `""` is what the harness writes when a metric was unavailable on that host
    (no PMU, or a libc-malloc redis). It must not become 0.0: zero is a
    legitimate reading, and conflating them made a missing metric look like an
    infinite regression.
    """
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def read_rows(pattern: str | None) -> dict[str, Row]:
    """query name -> Row, merged over every file matching `pattern`.

    A glob rather than a path because the callgrind subset is sharded across
    parallel jobs and arrives as cg-<side>-1.csv .. cg-<side>-N.csv. A shard
    that failed contributes nothing and the rest still report.
    """
    out: dict[str, Row] = {}
    for path in sorted(glob.glob(pattern)) if pattern else []:
        try:
            with open(path, newline="") as f:
                for raw in csv.DictReader(f):
                    name = raw.get("query")
                    if name:
                        out[name] = Row((k, _parse(raw.get(k))) for k in METRIC_NAMES if k in raw)
        except OSError:
            continue
    return out


def write_rows(path: str, rows: Iterable[tuple[str, Row]], fields: Iterable[str]) -> None:
    """Write rows back out in the harness's CSV format.

    `None` is written as an empty cell, which is the wire format the CI
    artifacts and `read_rows` above agree on. Round-tripping is covered by
    tests/test_csv_schema.py.
    """
    fields = list(fields)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, row in rows:
            out: dict[str, object] = {"query": name}
            for k in fields[1:]:
                v = row.get(k)
                out[k] = "" if v is None else v
            w.writerow(out)


# --- comparing ---------------------------------------------------------------


def ratio(base: Row, cur: Row, key: str) -> float | None:
    """cur/base for one metric, or None when the pair cannot be compared.

    A non-positive baseline makes the ratio meaningless (and sign-flipped), so
    the pair is left uncompared rather than reported as a bogus improvement.
    """
    a, b = base.get(key), cur.get(key)
    if a is None or b is None or a <= 0:
        return None
    return b / a


def has_data(rows: Iterable[Row], key: str) -> bool:
    """True when at least one row carries this metric.

    Used to skip a metric entirely rather than print a column of blanks: on a
    host with no PMU, `instr` is absent for every row and gating on it would
    silently gate on nothing.
    """
    return any(r.get(key) is not None for r in rows)


def geomean(values: Iterable[float]) -> float | None:
    """Geometric mean, the right average for a set of ratios. None if empty."""
    vals = [v for v in values if v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def normalise_ms(pr: dict[str, Row], base: dict[str, Row]) -> float | None:
    """The per-host wall-clock offset, measured from the control query.

    Raw `ms` is not comparable when the two sides ran on different machines, and
    the numbers say so: on byte-identical engines the median PR/base ratio was
    1.464, with 316 of 317 rows off by more than 5%, purely because each side
    got its own hosted runner. The control query does almost no query work, so
    its ratio *is* that offset, and dividing every row by it cancels the part
    that hit all rows equally — which took the same null data to median 0.973,
    p99 1.110, nothing beyond 1.17x.

    Returns None when the control row is missing from either side, in which case
    callers must not report `ms` at all rather than report it uncorrected.
    """
    if CONTROL_QUERY not in pr or CONTROL_QUERY not in base:
        return None
    a = base[CONTROL_QUERY].get("ms")
    b = pr[CONTROL_QUERY].get("ms")
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    return b / a
