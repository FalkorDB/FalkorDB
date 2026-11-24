"""`bench ldbc check`: the LDBC complex reads, candidate image against baseline.

One command of the nightly perf run (contract v1.5):

    bench ldbc check --candidate <image@sha256:…> --baseline <image@sha256:…>
                     --out <dir> --mode <queue|nightly|long>
                     [--candidate-label …] [--baseline-label …]
                     [--candidate-commit …] [--baseline-commit …]

Both images run as their own server, side by side on this machine (see
`containers`). The dataset is loaded into each, the substitution parameters are
sampled once, from the baseline, and both sides then run the same parameter
rows: per row, one side and then the other, in two rounds whose order is
swapped. Every latency is one request on a connection that never re-sends.

What it reports, per query:

* `p50_ms` / `p95_ms`: the latency ratio candidate/baseline over the parameter
  rows both sides finished. Wall clock, so the band is wide: it crosses at
  `RATIO` and only when the difference is also at least the absolute floor
  (`FLOOR_MS`). A crossing is measured again (the re-run) and is a regression
  only when the re-run crosses too.
* `rows` (`mismatch`): a parameter row on which the two sides return a
  different number of rows. A regression at once.
* `completed`: 0 when the side timed out, errored, or its server went down on
  any parameter row. Candidate 0 where the baseline finished is a regression
  at once; 0 on both sides is `info`.
* `invariant` "returns rows": a query that returns zero rows on every
  parameter row, on both sides, measured nothing (`info`).

A query the baseline cannot run at all (an error on every parameter row: the C
engine's dialect gaps) is left out with an `info` finding. The load's
throughput is reported as `info`: it is one sample per side.

`result.json` is always written. It is written first as an `error` ("did not
finish"), so a run killed from outside still leaves one, and replaced at the
end.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import platform
import re
import signal
import subprocess
import time
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from falkordb import FalkorDB
from falkordb.helpers import stringify_param_value
from redis import Redis
from redis.backoff import NoBackoff
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError
from redis.exceptions import ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError
from redis.retry import Retry

from falkorbench.client import BenchClient
from falkorbench.client import Server
from falkorbench.ldbc import containers
from falkorbench.ldbc import dataset as dataset_mod
from falkorbench.ldbc import loader as loader_mod
from falkorbench.ldbc import params as params_mod
from falkorbench.ldbc import queries as query_mod

Echo = Callable[[str], None]

CHECK = "ldbc"
GRAPH = "ldbc"
SCHEMA = 1
BASELINE = "baseline"
CANDIDATE = "candidate"
SIDES = (BASELINE, CANDIDATE)

#: A latency ratio crosses at 2x either way. Wall-clock latency of one request
#: on a VM moves far more than an instruction count does, so this band is wide
#: on purpose; a real planner regression here is usually 10x or more.
RATIO = 2.0

#: ... and only when the difference is at least this many milliseconds, so a
#: 2x on a query of a few hundred microseconds is not read from scheduling
#: noise. Set from three A/A runs (main@4c1088949 twice, C v4.22.4 once,
#: unpinned, n4-standard-16): the worst ratio either way was 1.19x, the worst
#: p50 difference on a query under 5 ms 0.061 ms, the worst p95 difference on
#: a query under 10 ms 0.65 ms. 0.5 ms is 8x and 2 ms 3x those; the re-run
#: still has to repeat a crossing.
FLOOR_MS = {"p50_ms": 0.5, "p95_ms": 2.0}

#: The sampling seed. Fixed, so two nights sample the same parameter rows.
SEED = 1

#: Seconds a fresh server gets to answer PING.
STARTUP_S = 120.0

#: A side that times out on this many parameter rows of a query stops running
#: that query: a query that hangs on every row would otherwise cost one full
#: timeout per row.
MAX_TIMEOUTS = 2

#: After a client-side timeout, how long the server gets to answer PING
#: before it counts as down. A server finishing a long query can be slow to
#: answer for a while; one that is silent this long is not coming back.
PING_GRACE_S = 60.0

#: A client-side socket timeout this far above the server-side TIMEOUT: when
#: it fires, the server ignored its own timeout and is treated as hung.
SOCKET_SLACK_S = 30.0

# The contract validator's rules (tests/perf/validate.py), applied with
# fullmatch: a looser check lets the check write a result the validator
# rejects (an uppercase repository, a trailing newline).
_PATH_PART = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
_HOST_PART = r"(?:[A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9-]*[A-Za-z0-9])"
IMAGE_RE = re.compile(
    rf"(?:(?:localhost|{_HOST_PART}(?:\.{_HOST_PART})+)(?::[0-9]+)?/|{_HOST_PART}:[0-9]+/)?"
    rf"{_PATH_PART}(?:/{_PATH_PART})*(?::[A-Za-z0-9_][A-Za-z0-9_.-]{{0,127}})?@sha256:[0-9a-f]{{64}}"
)
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class Mode:
    """What `--mode` sets: only the length, never the method (contract §1)."""

    name: str
    sf: str
    params: int
    rounds: int
    rerun_rounds: int
    timeout_ms: int
    #: Seconds one query may take per round (both sides together) before its
    #: remaining parameter rows are skipped for that round. IC14 takes 0.2-5 s
    #: per call at SF0.1 on the Rust engine (main@4c1088949, n4-standard-16),
    #: about 125 s per round for both sides; the budget leaves it whole.
    query_budget_s: float
    #: Seconds from the start after which nothing new is measured; what is
    #: left unmeasured then makes the verdict `error`.
    deadline_s: float
    #: Seconds one load request (one CSV file) may take before the side's
    #: load counts as not finished. SF0.1 loads in about 70 s in all.
    load_request_s: float


MODES = {
    "queue": Mode("queue", "0.1", 10, 2, 2, 30_000, 150.0, 25 * 60.0, 600.0),
    "nightly": Mode("nightly", "0.1", 25, 2, 2, 60_000, 300.0, 35 * 60.0, 600.0),
    "long": Mode("long", "1", 25, 2, 2, 300_000, 1800.0, 150 * 60.0, 3600.0),
}


# --- samples -----------------------------------------------------------------

OK = "ok"
TIMEOUT = "timeout"
ERROR = "error"
DOWN = "down"
SKIPPED = "skipped"
FAILED = (TIMEOUT, ERROR, DOWN)


@dataclass
class Sample:
    """One request (or one that was not sent: `skipped`, `down`)."""

    phase: str  # warmup | main | rerun
    round: int
    query: str
    param: int
    side: str
    outcome: str
    latency_ms: float | None = None
    rows: int | None = None
    error: str = ""


SAMPLE_FIELDS = tuple(Sample.__dataclass_fields__)


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile: a value that actually occurred."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), -(-p * len(ordered) // 100)))
    return ordered[int(rank) - 1]


def params_header(params: dict[str, Any]) -> str:
    """`CYPHER k=v …`, without the backticks newer clients add (older engines
    do not accept a quoted parameter name)."""
    if not params:
        return ""
    parts = []
    for key, value in params.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"parameter name {key!r} is not a plain identifier")
        parts.append(f"{key}={stringify_param_value(value)}")
    return "CYPHER " + " ".join(parts) + " "


# --- one side ----------------------------------------------------------------


@dataclass
class Side:
    """One image's server, as the measurement sees it."""

    side: str
    image: str
    container: containers.Container | None = None
    container_name: str = ""
    conn: Redis | None = None
    down: str = ""  # why it stopped answering, once it has
    timeouts: dict[str, int] = field(default_factory=dict)  # per query

    def run(self, cypher: str, params: dict[str, Any], timeout_ms: int) -> Sample:
        """Send one GRAPH.RO_QUERY and classify what came back.

        The verbose reply (no `--compact`) needs no follow-up calls to
        resolve labels or keys, so the latency is exactly one round trip.
        """
        sample = Sample("", 0, "", 0, self.side, OK)
        if self.down:
            sample.outcome = DOWN
            sample.error = self.down
            return sample
        assert self.conn is not None
        command = ("GRAPH.RO_QUERY", GRAPH, params_header(params) + cypher, "TIMEOUT", timeout_ms)
        started = time.perf_counter()
        try:
            reply = self.conn.execute_command(*command)
        except ResponseError as e:
            return _error_sample(sample, str(e))
        except RedisTimeoutError:
            # The server ignored its own TIMEOUT. Its connection is mid-reply
            # and of no further use; whether the server still answers anything
            # decides between "this query hangs" and "the server is gone".
            self.conn.connection_pool.disconnect()
            sample.outcome = TIMEOUT
            sample.error = f"no reply within {timeout_ms / 1000 + SOCKET_SLACK_S:.0f}s"
            if not self._answers():
                self.down = f"no answer to PING for {PING_GRACE_S:.0f}s after a client-side timeout"
            return sample
        except (RedisConnectionError, OSError) as e:
            try:
                state = self.container.state() if self.container else "unknown"
            except containers.DockerError as inspect_error:
                state = f"unknown ({inspect_error})"
            self.down = f"connection lost ({e}); container {state}"
            sample.outcome = DOWN
            sample.error = self.down
            return sample
        sample.latency_ms = (time.perf_counter() - started) * 1000
        if isinstance(reply, list) and reply and isinstance(reply[-1], ResponseError):
            return _error_sample(sample, str(reply[-1]))
        rows = reply[1] if isinstance(reply, list) and len(reply) == 3 else []
        sample.rows = len(rows)
        return sample

    def _answers(self) -> bool:
        """Whether the server answers PING within `PING_GRACE_S`.

        One missed PING is not enough to call it down: a server still busy
        with the query that timed out can be slow to answer, and calling it
        down would turn every later request on this side into a failure.
        """
        assert self.container is not None
        deadline = time.monotonic() + PING_GRACE_S
        while (left := deadline - time.monotonic()) > 0:
            try:
                if self.container.state().startswith(("exited", "dead")):
                    return False
            except containers.DockerError:
                pass  # inspect failed: says nothing either way, keep probing
            left = deadline - time.monotonic()
            if left <= 0:
                break
            try:
                probe = Redis(
                    host="127.0.0.1",
                    port=self.container.port,
                    socket_timeout=min(5.0, left),
                    socket_connect_timeout=min(5.0, left),
                    retry=Retry(NoBackoff(), 0),
                )
                if probe.ping() and time.monotonic() <= deadline:
                    return True
            except RedisError:
                pass  # not answering yet: try again until the grace runs out
            time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
        return False


def _error_sample(sample: Sample, message: str) -> Sample:
    sample.latency_ms = None
    sample.outcome = TIMEOUT if "timed out" in message.lower() else ERROR
    sample.error = message[:300]
    return sample


# --- analysis ----------------------------------------------------------------


#: A request's place in the run: (phase, round, parameter row).
Key = tuple[str, int, int]


@dataclass
class SideStats:
    attempted: int = 0
    ok: int = 0
    timeouts: int = 0
    errors: int = 0
    down: int = 0
    first_failure: str = ""
    #: Requests (phase, round, parameter row) that finished, and that failed.
    ok_keys: set[Key] = field(default_factory=set)
    failed_keys: set[Key] = field(default_factory=set)
    #: Per request that finished, keyed by (phase, round, parameter row), so
    #: the two sides are compared request against request: a warm-up's row
    #: count never stands in for a later round's, and a latency only counts
    #: when the other side finished the same request too.
    rows: dict[Key, int] = field(default_factory=dict)
    latencies: dict[Key, float] = field(default_factory=dict)

    @property
    def failed(self) -> int:
        return self.timeouts + self.errors + self.down

    @property
    def completed(self) -> int | None:
        """1 finished everything it was sent, 0 did not, None never ran."""
        if self.attempted == 0:
            return None
        return 0 if self.failed else 1

    @property
    def cannot_run(self) -> bool:
        """It errored on every request and never timed out or went down: the
        query is outside what this engine accepts (a dialect gap)."""
        return self.attempted > 0 and self.errors == self.attempted


def completion(query: str, cand: SideStats, base: SideStats) -> dict[str, Any] | None:
    """The `completed` finding, decided request by request.

    The candidate regressed when it failed a request (a parameter row in a
    given round) that the baseline finished. Failures on both sides with no
    such request are `info`; the baseline failing a request the candidate
    finished is an improvement.
    """

    def only(a: SideStats, b: SideStats) -> list[Key]:
        return sorted(a.failed_keys & b.ok_keys)

    if only(cand, base):
        status, c, b = "regression", 0, 1
    elif cand.failed_keys and base.failed_keys:
        status, c, b = "info", 0, 0
    elif only(base, cand):
        status, c, b = "improvement", 1, 0
    else:
        return None
    return _finding(
        f"{query} completed", "completed", c, b, direction="lower_is_worse", status=status
    )


def side_stats(samples: list[Sample]) -> SideStats:
    stats = SideStats()
    for s in samples:
        if s.outcome == SKIPPED:
            continue
        stats.attempted += 1
        if s.outcome == OK:
            stats.ok += 1
            key = (s.phase, s.round, s.param)
            stats.ok_keys.add(key)
            stats.rows[key] = s.rows or 0
            if s.latency_ms is not None and s.phase != "warmup":
                stats.latencies[key] = s.latency_ms
            continue
        stats.failed_keys.add((s.phase, s.round, s.param))
        if s.outcome == TIMEOUT:
            stats.timeouts += 1
        elif s.outcome == DOWN:
            stats.down += 1
        else:
            stats.errors += 1
        if not stats.first_failure:
            stats.first_failure = f"{s.outcome}: {s.error}"
    return stats


@dataclass
class LatencyCompare:
    metric: str
    candidate: float
    baseline: float

    @property
    def ratio(self) -> float | None:
        return self.candidate / self.baseline if self.baseline > 0 else None

    def crossing(self) -> str | None:
        """`regression` or `improvement` when outside the band, else None.

        Strictly outside: ratio > RATIO and difference > floor (contract).
        """
        floor = FLOOR_MS[self.metric]
        delta = self.candidate - self.baseline
        if self.baseline <= 0:
            return "regression" if delta > floor else None
        ratio = self.candidate / self.baseline
        if ratio > RATIO and delta > floor:
            return "regression"
        if ratio < 1 / RATIO and -delta > floor:
            return "improvement"
        return None


def compare_latency(cand: SideStats, base: SideStats) -> list[LatencyCompare]:
    """p50 and p95 over the requests both sides finished: the same parameter
    row in the same round, so the two lists are drawn from the same work."""
    common = sorted(set(cand.latencies) & set(base.latencies))
    c = [cand.latencies[k] for k in common]
    b = [base.latencies[k] for k in common]
    if not c or not b:
        return []
    out = []
    for metric, p in (("p50_ms", 50), ("p95_ms", 95)):
        cv, bv = percentile(c, p), percentile(b, p)
        assert cv is not None and bv is not None
        out.append(LatencyCompare(metric, cv, bv))
    return out


def _finding(
    name: str,
    metric: str,
    candidate: float | None,
    baseline: float | None,
    *,
    direction: str,
    status: str,
    threshold: float | None = None,
) -> dict[str, Any]:
    def num(v: float | None) -> float | None:
        return None if v is None else round(float(v), 3)

    ratio = None
    if candidate is not None and baseline not in (None, 0) and metric not in ("rows",):
        ratio = round(float(candidate) / float(baseline), 3)
    if metric in ("completed", "invariant"):
        ratio = None
    return {
        "name": name,
        "metric": metric,
        "candidate": candidate if isinstance(candidate, int) else num(candidate),
        "baseline": baseline if isinstance(baseline, int) else num(baseline),
        "ratio": ratio,
        "threshold": threshold,
        "direction": direction,
        "status": status,
    }


@dataclass
class QueryVerdict:
    """What one query contributed: findings, and whether it was measured."""

    query: str
    findings: list[dict[str, Any]] = field(default_factory=list)
    measured: bool = True
    left_out: str = ""  # why the baseline cannot run it
    crossed: list[str] = field(default_factory=list)  # metrics to re-run
    note: str = ""
    #: The sides returned different row counts: their latencies measure
    #: different work (6.0.4's IC9 "ran" 93x faster by returning nothing), so
    #: no latency finding is made for this query.
    rows_differ: bool = False
    #: Every request both sides finished returned zero rows: the query
    #: measured nothing, so its latency is not compared either.
    no_rows: bool = False
    #: At least one parameter row finished on both sides, so the two were
    #: actually compared (a query broken on both sides is measured, not
    #: compared).
    compared: bool = False
    #: The main rounds' per-side statistics, kept for the re-run and the
    #: non-crossing rows.
    base: SideStats = field(default_factory=lambda: SideStats())
    cand: SideStats = field(default_factory=lambda: SideStats())


@dataclass
class RowCheck:
    findings: list[dict[str, Any]] = field(default_factory=list)
    compared: bool = False
    differ: bool = False
    empty: bool = False


def compare_rows(query: str, cand: SideStats, base: SideStats) -> RowCheck:
    """Row counts (and, between like engines, contents), request by request.

    A count that differs on any request both sides finished is a `rows`
    regression. Its values are the two sides' totals over one request per
    parameter row; when those happen to agree, the first differing request's
    counts are given instead, named after its parameter row.
    """
    out = RowCheck()
    common = sorted(set(cand.rows) & set(base.rows))
    out.compared = bool(common)
    mismatched = [k for k in common if cand.rows[k] != base.rows[k]]
    out.differ = bool(mismatched)
    if mismatched:
        first_per_row: dict[int, Key] = {}
        for k in common:
            first_per_row.setdefault(k[2], k)
        c_total = sum(cand.rows[k] for k in first_per_row.values())
        b_total = sum(base.rows[k] for k in first_per_row.values())
        name = f"{query} rows"
        if c_total == b_total:
            k = mismatched[0]
            name, c_total, b_total = f"{query} rows (param {k[2]})", cand.rows[k], base.rows[k]
        out.findings.append(
            _finding(name, "rows", c_total, b_total, direction="mismatch", status="regression")
        )
        return out
    # Whether the query returned anything is decided on the measured requests:
    # a warm-up that returned rows does not make empty rounds worth timing.
    measured = [k for k in common if k[0] != "warmup"] or common
    if measured and all(base.rows[k] == 0 for k in measured):
        out.empty = True
        out.findings.append(
            _finding(
                f"{query} returns rows",
                "invariant",
                0,
                0,
                direction="lower_is_worse",
                status="info",
            )
        )
    return out


def analyse(query: str, main: list[Sample]) -> QueryVerdict:
    """Findings from the main rounds (latency crossings still unconfirmed)."""
    base = side_stats([s for s in main if s.side == BASELINE])
    cand = side_stats([s for s in main if s.side == CANDIDATE])
    verdict = QueryVerdict(query, base=base, cand=cand)

    if base.attempted == 0 and cand.attempted == 0:
        verdict.measured = False
        verdict.note = "not measured"
        return verdict

    if base.cannot_run:
        verdict.left_out = base.first_failure
        verdict.findings.append(
            _finding(
                f"{query} completed",
                "completed",
                cand.completed,
                0,
                direction="lower_is_worse",
                status="info",
            )
        )
        verdict.note = "left out: the baseline cannot run it"
        return verdict

    done = completion(query, cand, base)
    if done is not None:
        verdict.findings.append(done)
        if done["status"] == "regression":
            verdict.note = f"candidate {cand.first_failure}"
        elif done["status"] == "info":
            verdict.note = "did not finish on either side"

    rows = compare_rows(query, cand, base)
    verdict.compared = rows.compared
    verdict.findings += rows.findings

    verdict.rows_differ = rows.differ
    verdict.no_rows = rows.empty
    for cmp in [] if verdict.rows_differ or verdict.no_rows else compare_latency(cand, base):
        crossing = cmp.crossing()
        if crossing == "regression":
            verdict.crossed.append(cmp.metric)
        elif crossing == "improvement":
            verdict.findings.append(_latency_finding(query, cmp, "improvement"))
    if not compare_latency(cand, base) and not verdict.findings:
        verdict.measured = False
        verdict.note = "no parameter row finished on both sides"
    return verdict


def _latency_finding(query: str, cmp: LatencyCompare, status: str) -> dict[str, Any]:
    label = cmp.metric.removesuffix("_ms")
    return _finding(
        f"{query} {label}",
        cmp.metric,
        cmp.candidate,
        cmp.baseline,
        direction="higher_is_worse",
        status=status,
        threshold=RATIO,
    )


@dataclass
class Confirmation:
    findings: list[dict[str, Any]] = field(default_factory=list)
    #: Why the re-run settles nothing, when it does not; the result is then
    #: an `error` unless something else regressed.
    inconclusive: str = ""


def confirm(
    query: str,
    crossed: list[str],
    rerun: list[Sample],
    timed_rows: set[int] | None = None,
) -> Confirmation:
    """The re-run's verdict on the metrics the main rounds crossed.

    A metric that crosses again is a regression (`confirmed`). One that does
    not is reported as `ok` with the re-run's numbers: the first crossing was
    noise, or too small to repeat. That claim needs the re-run to have timed
    every parameter row the main rounds timed (`timed_rows`), on both sides,
    and to have returned rows; otherwise the re-run is inconclusive.
    """
    base = side_stats([s for s in rerun if s.side == BASELINE])
    cand = side_stats([s for s in rerun if s.side == CANDIDATE])
    result = Confirmation()
    out = result.findings
    done = completion(query, cand, base)
    if done is not None and done["status"] == "regression":
        out.append(done)
    # A re-run that returns different rows is that regression, not a timing:
    # a candidate that returns nothing is fast for the wrong reason.
    rows = compare_rows(query, cand, base)
    if rows.differ:
        out += [f for f in rows.findings if f["status"] == "regression"]
        return result
    if out:
        return result  # the candidate stopped finishing: that is the finding
    if rows.empty:
        result.inconclusive = "the re-run returned no rows on either side"
        return result
    retimed = {k[2] for k in set(cand.latencies) & set(base.latencies)}
    missing = sorted((timed_rows or retimed) - retimed)
    if missing:
        result.inconclusive = f"the re-run did not time parameter row(s) {missing} on both sides"
        return result
    by_metric = {c.metric: c for c in compare_latency(cand, base)}
    for metric in crossed:
        cmp = by_metric.get(metric)
        if cmp is None:
            result.inconclusive = f"the re-run did not measure {metric}"
            continue
        status = "regression" if cmp.crossing() == "regression" else "ok"
        out.append(_latency_finding(query, cmp, status))
    return result


# --- findings and verdict ----------------------------------------------------

_STATUS_ORDER = {"regression": 0, "improvement": 1, "info": 2, "ok": 3}


def select_findings(
    findings: list[dict[str, Any]],
    ok_rows: list[dict[str, Any]],
    *,
    max_total: int = 50,
    max_informative: int = 10,
) -> list[dict[str, Any]]:
    """Every crossing first, worst ratio first; then the informative rows
    (`info` before the largest non-crossing ratios), at most ten of them."""

    def worst(f: dict[str, Any]) -> tuple[int, float]:
        ratio = f["ratio"]
        key = -ratio if isinstance(ratio, (int, float)) else 0.0
        if f["status"] == "improvement" and isinstance(ratio, (int, float)):
            key = ratio
        return (_STATUS_ORDER[f["status"]], key)

    crossing = sorted(
        (f for f in findings if f["status"] in ("regression", "improvement")), key=worst
    )
    informative = sorted((f for f in findings if f["status"] in ("info", "ok")), key=worst)
    extra = sorted(ok_rows, key=lambda f: -(f["ratio"] or 0.0))
    chosen = crossing + (informative + extra)[:max_informative]
    return chosen[:max_total]


def decide(
    findings: list[dict[str, Any]],
    problems: list[str],
) -> tuple[str, bool]:
    """(verdict, confirmed). A regression outranks an error (contract v1.4)."""
    if any(f["status"] == "regression" for f in findings):
        return "regression", True
    if problems:
        return "error", False
    return "pass", False


# --- the run -----------------------------------------------------------------


def _now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_label(image: str) -> str:
    """`<repo>@<first 12 hex of the digest>` (contract v1.2)."""
    name, _, dig = image.partition("@")
    repo = name.rsplit(":", 1)[0] if ":" in name.rsplit("/", 1)[-1] else name
    return f"{repo}@{dig.removeprefix('sha256:')[:12]}"


def machine() -> dict[str, Any]:
    """`PERF_MACHINE_JSON` on the VM (copied as is), else this host as `local`."""
    path = os.environ.get("PERF_MACHINE_JSON")
    if path:
        return json.loads(Path(path).read_text())
    cpu = platform.processor() or platform.machine()
    try:
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        elif platform.system() == "Darwin":
            cpu = (
                subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip()
                or cpu
            )
    except OSError:
        pass  # best effort: the CPU model falls back to platform's answer
    return {
        "runner": "local",
        "type": f"{platform.system().lower()}-{platform.machine()}",
        "cpu": cpu or "unknown",
        "cores": os.cpu_count() or 1,
    }


@dataclass
class Options:
    candidate: str
    baseline: str
    out: Path
    mode: Mode
    candidate_label: str | None = None
    baseline_label: str | None = None
    candidate_commit: str | None = None
    baseline_commit: str | None = None
    cache: Path = Path("ldbc-data")
    queries: tuple[str, ...] = ()
    param_count: int | None = None


class Check:
    """One run of the check, from images to result.json."""

    def __init__(self, opts: Options, echo: Echo = print) -> None:
        self.opts = opts
        self.echo = echo
        self.started = _now()
        self.t0 = time.monotonic()
        self.raw = opts.out / "raw"
        self.findings: list[dict[str, Any]] = []
        self.ok_rows: list[dict[str, Any]] = []
        self.problems: list[str] = []
        self.left_out: dict[str, str] = {}
        self.notes: list[str] = []
        self.samples: list[Sample] = []
        self.summary_rows: list[dict[str, Any]] = []
        self.artifacts: list[str] = []
        self.sides = {
            BASELINE: Side(BASELINE, opts.baseline),
            CANDIDATE: Side(CANDIDATE, opts.candidate),
        }
        self.queries: list[query_mod.ComplexRead] = []
        self.param_set: params_mod.ParamSet | None = None
        self.compared_queries = 0
        #: Parameter rows, per query, on which a side has timed out.
        self.timed_out: dict[str, set[int]] = {}
        #: Requests not sent because the deadline had passed.
        self.deadline_skips = 0
        #: Parameter-row runs not sent because a query used up its round's
        #: budget: the comparison covers fewer rows than the mode asks for.
        self.budget_skips = 0

    # -- result -----------------------------------------------------------

    def image_block(self, side: str) -> dict[str, Any]:
        o = self.opts
        image = o.candidate if side == CANDIDATE else o.baseline
        label = o.candidate_label if side == CANDIDATE else o.baseline_label
        commit = o.candidate_commit if side == CANDIDATE else o.baseline_commit
        if commit and not COMMIT_RE.fullmatch(commit):
            commit = None  # reported as a problem; null keeps result.json valid
        return {"image": image, "commit": commit or None, "label": label or default_label(image)}

    def result(self, *, error: str | None = None) -> dict[str, Any]:
        findings = select_findings(self.findings, self.ok_rows)
        problems = [*self.problems, *([error] if error else [])]
        verdict, confirmed = decide(findings, problems)
        return {
            "schema": SCHEMA,
            "check": CHECK,
            "verdict": verdict,
            "confirmed": confirmed,
            "summary": self.summary_line(verdict, findings, problems),
            "candidate": self.image_block(CANDIDATE),
            "baseline": self.image_block(BASELINE),
            "machine": self.machine(),
            "started": self.started,
            "finished": _now(),
            "findings": findings,
            "error": _one_line("; ".join(problems)) if verdict == "error" else None,
            "artifacts": sorted(set(self.artifacts)),
        }

    def summary_line(
        self, verdict: str, findings: list[dict[str, Any]], problems: list[str]
    ) -> str:
        mode = self.opts.mode
        scope = (
            f"{self.compared_queries} of {len(self.queries) or 14} LDBC complex reads compared "
            f"at SF{mode.sf} ({mode.name})"
        )
        parts = [scope]
        regressions = [f for f in findings if f["status"] == "regression"]
        if regressions:
            shown = ", ".join(describe(f) for f in regressions[:4])
            more = f" (+{len(regressions) - 4} more)" if len(regressions) > 4 else ""
            parts.append(f"regression: {shown}{more}")
        if self.left_out:
            parts.append(
                f"{', '.join(sorted(self.left_out, key=_qnum))} left out (baseline cannot run)"
            )
        both = [
            f["name"].split()[0]
            for f in findings
            if f["status"] == "info"
            and f["metric"] in ("completed", "invariant")
            and f["candidate"] == 0
            and f["baseline"] == 0
            and f["name"].split()[0] not in self.left_out
        ]
        if both:
            parts.append(f"broken on both sides (not new): {', '.join(both)}")
        if verdict == "error":
            parts.append(f"error: {problems[0]}")
        elif verdict == "pass":
            parts.append("no regression")
        elif problems:
            # A regression outranks an error, so `error` stays null; the
            # summary still says the run was cut short, not only raw/run.json.
            more = f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""
            parts.append(f"also not fully measured: {problems[0]}{more}")
        return _one_line("; ".join(parts))

    def machine(self) -> dict[str, Any]:
        try:
            return machine()
        except (OSError, ValueError) as e:
            note = f"PERF_MACHINE_JSON unreadable ({e}); recorded this host as local"
            if note not in self.notes:
                self.notes.append(note)
            os.environ.pop("PERF_MACHINE_JSON", None)
            return machine()

    def write(self, *, error: str | None = None) -> dict[str, Any]:
        self.opts.out.mkdir(parents=True, exist_ok=True)
        result = self.result(error=error)
        tmp = self.opts.out / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        tmp.replace(self.opts.out / "result.json")
        return result

    def problem(self, message: str) -> None:
        self.echo(f"error: {message}")
        self.problems.append(message)

    # -- artifacts ----------------------------------------------------------

    def artifact(self, rel: str, text: str) -> None:
        path = self.opts.out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        self.artifacts.append(rel)

    def write_samples(self) -> None:
        path = self.raw / "samples.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=SAMPLE_FIELDS)
            writer.writeheader()
            for s in self.samples:
                row = asdict(s)
                if s.latency_ms is not None:
                    row["latency_ms"] = f"{s.latency_ms:.3f}"
                writer.writerow(row)
        self.artifacts.append("raw/samples.csv")

    def write_summary(self) -> None:
        if not self.summary_rows:
            return
        path = self.raw / "summary.csv"
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self.summary_rows[0]))
            writer.writeheader()
            writer.writerows(self.summary_rows)
        self.artifacts.append("raw/summary.csv")

    # -- phases -------------------------------------------------------------

    def past_deadline(self) -> bool:
        return time.monotonic() - self.t0 > self.opts.mode.deadline_s

    def run(self) -> dict[str, Any]:
        o = self.opts
        self.write(error="the check did not finish (it was stopped, or it crashed)")
        try:
            self.queries = query_mod.select(o.queries)
        except ValueError as e:
            self.problem(str(e))
            return self.write()
        for side, image in ((CANDIDATE, o.candidate), (BASELINE, o.baseline)):
            if not IMAGE_RE.fullmatch(image):
                self.problem(f"--{side} {image!r} is not pinned by digest (name@sha256:<64 hex>)")
        for side, commit in ((CANDIDATE, o.candidate_commit), (BASELINE, o.baseline_commit)):
            if commit and not COMMIT_RE.fullmatch(commit):
                self.problem(f"--{side}-commit {commit!r} is not a 40-hex commit")
        if self.problems:
            return self.write()

        try:
            root = self.fetch()
            if root is not None:
                self.start_and_load(root)
            if not self.problems and not _regressed(self.findings):
                self.sample_params()
                self.explain()
                self.measure()
        finally:
            # A first SIGTERM arriving here is held until teardown is done,
            # so the containers are still removed and the logs still kept.
            with _sigterm_deferred():
                self.teardown()
        self.write_samples()
        self.write_summary()
        return self.write()

    def fetch(self) -> Path | None:
        sf = self.opts.mode.sf
        self.echo(f"dataset: SF{sf} into {self.opts.cache}")
        try:
            root = dataset_mod.fetch(self.opts.cache, sf, echo=self.echo)
            dataset_mod.prepare(root, echo=self.echo)
        except dataset_mod.DatasetError as e:
            self.problem(f"dataset: {e}")
            return None
        return root

    def start_and_load(self, root: Path) -> None:
        cpus = os.environ.get("PERF_SERVER_CPUS") or None
        for side in SIDES:
            image = self.sides[side].image
            self.echo(f"{side}: pulling {image}")
            try:
                containers.pull(image)
            except containers.DockerError as e:
                self.problem(f"{side} image: {e}")
        if self.problems:
            return

        load: dict[str, Any] = {}
        for side in SIDES:  # baseline first: a broken baseline stops the run early
            s = self.sides[side]
            name = f"perf-ldbc-{side}-{os.getpid()}"
            # Named before it exists, so teardown can remove it even when a
            # SIGTERM lands between `docker run` and `start` returning.
            s.container_name = name
            try:
                s.container = containers.start(s.image, name, self.opts.cache, cpus=cpus)
                client = self.connect(s)
            except (containers.DockerError, RuntimeError, RedisError) as e:
                self.side_failed(side, "server start", f"server did not start: {e}")
                return
            self.echo(f"{side}: port {s.container.port}")
            try:
                stats = loader_mod.load(client, root, import_root=self.opts.cache, echo=self.echo)
            except (loader_mod.LoadError, RedisError) as e:
                load[side] = {"error": str(e)}
                self.artifact("raw/load.json", json.dumps(load, indent=2) + "\n")
                self.side_failed(side, "load completed", f"could not load the dataset: {e}")
                return
            load[side] = {
                "nodes": stats.nodes,
                "edges": stats.edges,
                "seconds": round(stats.seconds, 3),
                "entities_per_s": round((stats.nodes + stats.edges) / stats.seconds, 1),
            }
        self.artifact("raw/load.json", json.dumps(load, indent=2) + "\n")
        b, c = load[BASELINE], load[CANDIDATE]
        self.findings.append(
            _finding(
                "load",
                "throughput",
                c["entities_per_s"],
                b["entities_per_s"],
                direction="lower_is_worse",
                status="info",
            )
        )

    def side_failed(self, side: str, name: str, why: str) -> None:
        """A side that could not start or load: the baseline's failure leaves
        nothing to compare against (error); the candidate's is a regression."""
        if side == BASELINE:
            self.problem(f"baseline {why}")
            return
        self.findings.append(
            _finding(name, "completed", 0, 1, direction="lower_is_worse", status="regression")
        )
        self.notes.append(f"candidate {why}")

    def connect(self, s: Side) -> BenchClient:
        """Wait for the server, switch off background saves, read its engine."""
        assert s.container is not None
        port = s.container.port
        deadline = time.monotonic() + STARTUP_S
        last: Exception | None = None
        while time.monotonic() < deadline:
            state = s.container.state()
            if state != "running":
                raise RuntimeError(f"container {state}: {s.container.logs()[-400:]}")
            try:
                # Bounded by what is left of the readiness window: a server
                # that accepts but never answers must not hold one attempt
                # past it.
                left = max(1.0, deadline - time.monotonic())
                FalkorDB(
                    host="127.0.0.1",
                    port=port,
                    retry=Retry(NoBackoff(), 0),
                    socket_timeout=min(5.0, left),
                    socket_connect_timeout=min(5.0, left),
                ).connection.ping()
                break
            except RedisError as e:
                last = e
                time.sleep(0.5)
        else:
            raise RuntimeError(f"no answer on :{port} within {STARTUP_S:.0f}s ({last})")
        db = FalkorDB(
            host="127.0.0.1",
            port=port,
            retry=Retry(NoBackoff(), 0),
            # A load or sampling request that never returns is a load that
            # did not finish, not a check that hangs until it is killed.
            socket_timeout=self.opts.mode.load_request_s,
        )
        # The images keep redis's default save points; one would fork a
        # BGSAVE a minute into the measurement, on one side only.
        db.connection.config_set("save", "")
        s.conn = Redis(
            host="127.0.0.1",
            port=port,
            retry=Retry(NoBackoff(), 0),
            socket_timeout=self.opts.mode.timeout_ms / 1000 + SOCKET_SLACK_S,
        )
        client = BenchClient(db=db, server=Server(port=port))
        client.graph_name = GRAPH
        return client

    def sample_params(self) -> None:
        o, mode = self.opts, self.opts.mode
        base = self.sides[BASELINE]
        assert base.container is not None
        client = BenchClient(
            db=FalkorDB(
                host="127.0.0.1",
                port=base.container.port,
                retry=Retry(NoBackoff(), 0),
                socket_timeout=mode.load_request_s,
            ),
            server=Server(port=base.container.port),
            graph_name=GRAPH,
        )
        count = o.param_count or mode.params
        try:
            self.param_set = params_mod.sample(client, count=count, seed=SEED, echo=self.echo)
        except (params_mod.ParamError, RedisError) as e:
            self.problem(f"sampling parameters on the baseline failed: {e}")
            return
        self.artifact(
            "raw/params.json",
            json.dumps(
                {
                    "source": self.param_set.source,
                    "sampled_on": BASELINE,
                    "rows": {f"IC{n}": r for n, r in sorted(self.param_set.rows.items())},
                },
                indent=2,
            )
            + "\n",
        )

    def explain(self) -> None:
        """Each query's plan on each side, for whoever reads a regression."""
        if self.param_set is None:
            return
        for side in SIDES:
            s = self.sides[side]
            for q in self.queries:
                rows = self.param_set.for_query(q.number)
                if not rows or s.conn is None or s.down:
                    continue
                try:
                    plan = s.conn.execute_command(
                        "GRAPH.EXPLAIN", GRAPH, params_header(rows[0]) + q.cypher
                    )
                    text = "\n".join(_text(line) for line in plan)
                except RedisError as e:
                    text = f"GRAPH.EXPLAIN failed: {e}"
                self.artifact(f"raw/explain/{side}/{q.name}.txt", text + "\n")

    def measure(self) -> None:
        if self.param_set is None:
            return
        mode = self.opts.mode
        main: dict[str, list[Sample]] = {q.name: [] for q in self.queries}
        for q in self.queries:
            main[q.name] += self.warmup(q)
        for rnd in range(mode.rounds):
            order = SIDES if rnd % 2 == 0 else SIDES[::-1]
            for q in self.queries:
                main[q.name] += self.round(q, "main", rnd, order)

        verdicts = {q.name: analyse(q.name, main[q.name]) for q in self.queries}
        reruns: dict[str, list[Sample]] = {}
        for q in self.queries:
            v = verdicts[q.name]
            if not v.crossed:
                continue
            self.echo(f"{q.name}: {', '.join(v.crossed)} crossed; re-running")
            reruns[q.name] = []
            for rnd in range(mode.rerun_rounds):
                order = SIDES[::-1] if rnd % 2 == 0 else SIDES
                reruns[q.name] += self.round(q, "rerun", rnd, order)
        self.judge(main, reruns, verdicts)

    def judge(
        self,
        main: dict[str, list[Sample]],
        reruns: dict[str, list[Sample]],
        verdicts: dict[str, QueryVerdict],
    ) -> None:
        """Findings and problems from the samples, apart from the servers.

        Separate from collecting them, so a run's raw/samples.csv can be
        judged again offline with the same code.
        """
        for q in self.queries:
            v = verdicts[q.name]
            self.findings += v.findings
            if v.left_out:
                self.left_out[q.name] = v.left_out
            if not v.measured:
                self.problem(f"{q.name} {v.note}")
                continue
            self.compared_queries += 1 if v.compared else 0
            if v.crossed:
                timed = {k[2] for k in set(v.cand.latencies) & set(v.base.latencies)}
                confirmed = confirm(q.name, v.crossed, reruns.get(q.name, []), timed)
                if confirmed.inconclusive:
                    self.problem(f"{q.name}: re-run inconclusive: {confirmed.inconclusive}")
                elif not confirmed.findings:
                    self.problem(f"{q.name}: the re-run did not finish on both sides")
                self.findings += confirmed.findings
            self.record_summary(q.name, main[q.name], reruns.get(q.name))

        if self.deadline_skips:
            self.problem(
                f"the {self.opts.mode.deadline_s / 60:.0f} min deadline passed before every "
                f"query had its rounds ({self.deadline_skips} parameter-row runs skipped)"
            )
        if self.budget_skips:
            self.problem(
                f"the {self.opts.mode.query_budget_s:.0f}s per-round budget cut "
                f"{self.budget_skips} parameter-row runs"
            )
        if self.compared_queries == 0:
            self.problem("no query could be compared on both sides")

        # The rows that did not cross, as candidates for the informative slots.
        for q in self.queries:
            v = verdicts[q.name]
            if v.left_out or not v.measured or v.rows_differ or v.no_rows:
                continue
            for cmp in compare_latency(v.cand, v.base):
                if cmp.metric not in v.crossed and cmp.crossing() is None:
                    self.ok_rows.append(_latency_finding(q.name, cmp, "ok"))

    def warmup(self, q: query_mod.ComplexRead) -> list[Sample]:
        """One untimed call per side: GraphBLAS compiles kernels on first use.

        A warm-up that times out does count: the query did not finish with
        these parameters, and that row is skipped on both sides from then on.
        """
        assert self.param_set is not None
        rows = self.param_set.for_query(q.number)
        out = []
        if not rows:
            return out
        if self.past_deadline():
            self.deadline_skips += 1
            return out
        for side in SIDES:
            out.append(self.one(q, "warmup", 0, 0, rows[0], self.sides[side]))
        return out

    def round(
        self,
        q: query_mod.ComplexRead,
        phase: str,
        rnd: int,
        order: tuple[str, ...],
    ) -> list[Sample]:
        assert self.param_set is not None
        mode = self.opts.mode
        rows = self.param_set.for_query(q.number)
        out: list[Sample] = []
        started = time.monotonic()
        for i, params in enumerate(rows):
            reason = ""
            # The timeout rules come first: a row they exclude would not run
            # with any budget, so it is not a row the budget cut.
            if i in self.timed_out.get(q.name, set()):
                reason = "a side timed out on this parameter row before"
            elif any(self.sides[s].timeouts.get(q.name, 0) >= MAX_TIMEOUTS for s in SIDES):
                reason = f"a side timed out {MAX_TIMEOUTS} times on this query"
            elif self.past_deadline():
                reason = "past the check's deadline"
                self.deadline_skips += 1
            elif time.monotonic() - started > mode.query_budget_s:
                reason = f"over the {mode.query_budget_s:.0f}s per-round budget"
                self.budget_skips += 1
            for side in order:
                if reason:
                    skipped = Sample(phase, rnd, q.name, i, side, SKIPPED, error=reason)
                    self.samples.append(skipped)
                    out.append(skipped)
                else:
                    out.append(self.one(q, phase, rnd, i, params, self.sides[side]))
        return out

    def one(
        self,
        q: query_mod.ComplexRead,
        phase: str,
        rnd: int,
        param: int,
        params: dict[str, Any],
        side: Side,
    ) -> Sample:
        sample = side.run(q.cypher, params, self.opts.mode.timeout_ms)
        sample.phase, sample.round, sample.query, sample.param = phase, rnd, q.name, param
        if sample.outcome == TIMEOUT:
            # Both sides skip this row from now on (the other side still runs
            # it in this round, so a timeout on one side only is seen as such).
            side.timeouts[q.name] = side.timeouts.get(q.name, 0) + 1
            self.timed_out.setdefault(q.name, set()).add(param)
        self.samples.append(sample)
        return sample

    def record_summary(self, query: str, main: list[Sample], rerun: list[Sample] | None) -> None:
        everything = main + (rerun or [])
        for phase in ("warmup", "main", "rerun"):
            for side in SIDES:
                st = side_stats([s for s in everything if s.side == side and s.phase == phase])
                if st.attempted == 0:
                    continue
                lat = list(st.latencies.values())
                self.summary_rows.append(
                    {
                        "query": query,
                        "phase": phase,
                        "side": side,
                        "attempted": st.attempted,
                        "ok": st.ok,
                        "timeouts": st.timeouts,
                        "errors": st.errors,
                        "down": st.down,
                        "rows_total": sum(st.rows.values()),
                        "p50_ms": _fmt(percentile(lat, 50)),
                        "p95_ms": _fmt(percentile(lat, 95)),
                        "p99_ms": _fmt(percentile(lat, 99)),
                        "first_failure": st.first_failure,
                    }
                )

    def teardown(self) -> None:
        started = {side for side in SIDES if self.sides[side].container is not None}
        for side in SIDES:
            s = self.sides[side]
            if s.container is None:
                if s.container_name:  # started, but stopped before it was ours
                    try:
                        containers.remove_by_name(s.container_name)
                    except containers.DockerError as e:
                        self.notes.append(f"{side} container not removed: {e}")
                continue
            try:
                self.artifact(f"raw/{side}-server.log", s.container.logs())
            except (containers.DockerError, OSError) as e:
                self.notes.append(f"{side} logs: {e}")
            try:
                state = s.container.state()
                # Only an exit is a server that went down. "gone" means
                # `docker inspect` itself failed, which says nothing about
                # the server (a daemon hiccup must not turn a night red).
                if state.startswith(("exited", "dead")) and not s.down:
                    s.down = f"container {state}"
                elif state not in ("running",) and not s.down:
                    self.notes.append(f"{side} container state unreadable at teardown: {state}")
            except (containers.DockerError, OSError) as e:
                self.notes.append(f"{side} state: {e}")
            if s.down:
                self.notes.append(f"{side} server: {s.down}")
            try:
                s.container.remove()
            except containers.DockerError as e:
                self.notes.append(f"{side} container not removed: {e}")
        self.server_findings(started)
        engines = {side: {"down": self.sides[side].down or None} for side in SIDES}
        self.artifact(
            "raw/run.json",
            json.dumps(
                {
                    "mode": asdict(self.opts.mode),
                    "ratio": RATIO,
                    "floor_ms": FLOOR_MS,
                    "seed": SEED,
                    "servers": engines,
                    "left_out": self.left_out,
                    "notes": self.notes,
                    "problems": self.problems,
                    "rewrites": {f"IC{n}": why for n, why in sorted(query_mod.REWRITES.items())},
                },
                indent=2,
            )
            + "\n",
        )

    def server_findings(self, started: set[str]) -> None:
        """A server that went down at any point, also after its last reply.

        The candidate's is a regression at once (a crash); the baseline's
        leaves the comparison without a reference, so it is an error.
        """
        if started != set(SIDES):
            return  # a side that never started is reported where it failed
        cand, base = self.sides[CANDIDATE].down, self.sides[BASELINE].down
        if cand:
            c, b, status = (0, 0, "info") if base else (0, 1, "regression")
            self.findings.append(
                _finding("server up", "completed", c, b, direction="lower_is_worse", status=status)
            )
        if base:
            self.problem(f"the baseline server went down: {base}")


def _regressed(findings: list[dict[str, Any]]) -> bool:
    return any(f["status"] == "regression" for f in findings)


def describe(f: dict[str, Any]) -> str:
    """One finding in a few words, for the summary line."""
    if f["metric"] == "completed":
        return f"{f['name'].removesuffix(' completed')} did not finish"
    if f["metric"] == "rows":
        return f"{f['name']} {f['candidate']} vs {f['baseline']}"
    if f["ratio"] is not None:
        return f"{f['name']} {f['ratio']}x"
    return f"{f['name']} {f['candidate']} vs {f['baseline']}"


def _text(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _one_line(text: str) -> str:
    return " | ".join(line.strip() for line in text.splitlines() if line.strip()) or "unknown"


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def _qnum(name: str) -> int:
    return int(name.removeprefix("IC"))


@contextmanager
def _term_as_interrupt() -> Iterator[None]:
    """`timeout` stops the command with SIGTERM; turn it into an exception so
    the containers are removed and result.json says what happened."""

    def handler(signum: int, frame: object) -> None:
        # `timeout` may signal the whole process group, and uv forwards the
        # signal too: one interrupt, then the rest are ignored so teardown
        # and the final write are not cut short.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt(f"signal {signum}")

    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


@contextmanager
def _sigterm_deferred() -> Iterator[None]:
    """Hold SIGTERM while the block runs; it is delivered when it ends."""
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})


def run_check(opts: Options, echo: Echo = print) -> dict[str, Any]:
    """Run the check and write `<out>/result.json`; never raises once that
    file can be written."""
    check = Check(opts, echo)
    # The handler stays in place through the final write below: after the
    # first SIGTERM it ignores the rest, so that write is not cut short.
    with _term_as_interrupt():
        try:
            return check.run()
        except (KeyboardInterrupt, Exception) as e:
            reason = "stopped" if isinstance(e, KeyboardInterrupt) else "crashed"
            try:
                check.write_samples()
                check.write_summary()
            except OSError:
                pass  # raw files are a bonus here; result.json below is the must
            return check.write(error=f"the check {reason}: {type(e).__name__}: {e}")
