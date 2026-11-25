"""`bench ldbc check`: the LDBC complex reads, candidate image against baseline.

The nightly perf check for LDBC (contract v1.12): each image runs as its own
server, both load the dataset and run the same sampled parameter rows, and the
result goes to `<out>/result.json`. What it measures, the findings, the
thresholds and the modes are in bench/README.md ("The nightly check").
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import re
import time
from collections.abc import Callable
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

from falkorbench import kit
from falkorbench.client import BenchClient
from falkorbench.client import Server
from falkorbench.ldbc import dataset as dataset_mod
from falkorbench.ldbc import loader as loader_mod
from falkorbench.ldbc import params as params_mod
from falkorbench.ldbc import queries as query_mod

Echo = Callable[[str], None]
say: Echo = functools.partial(print, flush=True)

CHECK = "ldbc"
GRAPH = "ldbc"
BASELINE = "baseline"
CANDIDATE = "candidate"
SIDES = (BASELINE, CANDIDATE)

#: A latency ratio crosses at 2x either way. Wall-clock latency of one request
#: on a VM moves far more than an instruction count does, so this band is wide
#: on purpose; a real planner regression here is usually 10x or more.
RATIO = 2.0

#: ... and only when the difference is more than this many milliseconds, so a
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

#: The dataset cache, beside the bench project (bench/ldbc-data).
CACHE = Path(__file__).resolve().parents[3] / "ldbc-data"

#: Each side is the image's own server (its redis or valkey, its allocator,
#: its entrypoint). The entrypoint appends FALKORDB_ARGS to `--loadmodule`;
#: the dataset is mounted read-only at /import, and the trailing slash
#: matters (the C module joins folder and file name with no separator).
#: BROWSER=0 keeps the images that bundle the browser from starting it.
SERVER_ENV = {"BROWSER": "0", "FALKORDB_ARGS": "IMPORT_FOLDER /import/"}


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
    container: kit.Container | None = None
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
                state = kit.state(self.container.name) if self.container else "unknown"
            except kit.DockerError as inspect_error:
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
        return kit.wait_ping(self.container.port, PING_GRACE_S, self.container)


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
    return kit.finding(f"{query} completed", "completed", c, b, status, direction="lower_is_worse")


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
        """`regression` or `improvement` when outside the band (strictly: ratio
        > RATIO and difference > the metric's floor), else None."""
        crossed = kit.crossed(self.candidate, self.baseline, RATIO, FLOOR_MS[self.metric])
        return {1: "regression", -1: "improvement"}.get(crossed)


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
    base: SideStats = field(default_factory=SideStats)
    cand: SideStats = field(default_factory=SideStats)


@dataclass
class RowCheck:
    findings: list[dict[str, Any]] = field(default_factory=list)
    compared: bool = False
    differ: bool = False
    empty: bool = False


def compare_rows(query: str, cand: SideStats, base: SideStats) -> RowCheck:
    """Row counts, request by request.

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
        mismatch = kit.finding(name, "rows", c_total, b_total, "regression", direction="mismatch")
        mismatch["ratio"] = None  # two row counts that differ: no slowdown to show as Nx
        out.findings.append(mismatch)
        return out
    # Whether the query returned anything is decided on the measured requests:
    # a warm-up that returned rows does not make empty rounds worth timing.
    measured = [k for k in common if k[0] != "warmup"] or common
    if measured and all(base.rows[k] == 0 for k in measured):
        out.empty = True
        name = f"{query} returns rows"
        out.findings.append(
            kit.finding(name, "invariant", 0, 0, "info", direction="lower_is_worse")
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
        name = f"{query} completed"
        verdict.findings.append(
            kit.finding(name, "completed", cand.completed, 0, "info", direction="lower_is_worse")
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


def latency_row(query: str, phase: str, cmp: LatencyCompare, status: str) -> dict[str, Any]:
    """One latency comparison for raw/rows.csv, whatever its outcome."""
    ratio = cmp.ratio
    return {
        "query": query,
        "phase": phase,
        "metric": cmp.metric,
        "candidate": round(cmp.candidate, 3),
        "baseline": round(cmp.baseline, 3),
        "ratio": "" if ratio is None else round(ratio, 3),
        "status": status,
    }


ROW_FIELDS = ("query", "phase", "metric", "candidate", "baseline", "ratio", "status")


def _latency_finding(query: str, cmp: LatencyCompare, status: str) -> dict[str, Any]:
    name = f"{query} {cmp.metric.removesuffix('_ms')}"
    return kit.finding(name, cmp.metric, cmp.candidate, cmp.baseline, status, threshold=RATIO)


@dataclass
class Confirmation:
    findings: list[dict[str, Any]] = field(default_factory=list)
    #: Every re-run comparison of a crossed metric, for raw/rows.csv.
    rows: list[dict[str, Any]] = field(default_factory=list)
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
    not was noise, or too small to repeat: no finding, and raw/rows.csv keeps
    the re-run's numbers as "not repeated". That claim needs the re-run to have timed
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
        repeated = cmp.crossing() == "regression"
        if repeated:
            out.append(_latency_finding(query, cmp, "regression"))
        result.rows.append(
            latency_row(query, "rerun", cmp, "regression" if repeated else "not repeated")
        )
    return result


# --- findings ----------------------------------------------------------------


def worst_first(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The order kit.select() keeps within each status: the worst ratio first
    (the highest for a regression, the lowest for an improvement)."""

    def worst(f: dict[str, Any]) -> float:
        ratio = f["ratio"]
        if not isinstance(ratio, (int, float)):
            return 0.0
        return ratio if f["status"] == "improvement" else -ratio

    return sorted(findings, key=worst)


# --- the run -----------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The contract's flags (kit.add_args), the dataset cache, and for
    development runs a parameter row count and a subset of the queries."""
    parser = argparse.ArgumentParser(
        prog="bench ldbc check",
        description="Compare a candidate image with a baseline image on the LDBC complex reads.",
    )
    kit.add_args(parser)
    parser.add_argument("--cache", type=Path, default=CACHE, help="dataset cache directory")
    parser.add_argument(
        "--param-count",
        type=_positive,
        default=None,
        help="parameter rows per query (default: the mode's); for development runs",
    )
    parser.add_argument("names", nargs="*", help="a subset of IC1..IC14 (default: all)")
    args = parser.parse_args(argv)
    args.cache = args.cache.resolve()
    return args


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{value} is not a positive count")
    return value


class Check:
    """One run of the check, from images to result.json."""

    def __init__(self, args: argparse.Namespace, echo: Echo = say) -> None:
        self.args = args
        self.mode = MODES[args.mode]
        self.echo = echo
        self.started = kit.utc_now()
        self.t0 = time.monotonic()
        self.findings: list[dict[str, Any]] = []
        #: Every latency comparison, crossing or not: raw/rows.csv.
        self.latency_rows: list[dict[str, Any]] = []
        self.problems: list[str] = []
        self.left_out: dict[str, str] = {}
        self.notes: list[str] = []
        self.samples: list[Sample] = []
        self.summary_rows: list[dict[str, Any]] = []
        self.artifacts: list[str] = []
        self.sides = {
            BASELINE: Side(BASELINE, args.baseline),
            CANDIDATE: Side(CANDIDATE, args.candidate),
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

    @property
    def out(self) -> Path:
        return Path(self.args.out)  # made absolute by kit.run_check()

    @property
    def raw(self) -> Path:
        return self.out / "raw"

    # -- result -----------------------------------------------------------

    def result(self, *, error: str | None = None) -> dict[str, Any]:
        """The result.json value: kit.result() over this run's findings and problems."""
        findings = kit.select(worst_first(self.findings))
        problems = [*self.problems, *([error] if error else [])]
        return kit.result(
            CHECK,
            self.args,
            started=self.started,
            findings=findings,
            summary=self.summary_line(kit.verdict(findings, problems), findings, problems),
            errors=problems,
            artifacts=self.artifacts,
        )

    def summary_line(
        self, verdict: str, findings: list[dict[str, Any]], problems: list[str]
    ) -> str:
        mode = self.mode
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
        return "; ".join(parts)

    def problem(self, message: str) -> None:
        self.echo(f"error: {message}")
        self.problems.append(message)

    # -- artifacts ----------------------------------------------------------

    def artifact(self, rel: str, text: str) -> None:
        path = self.out / rel
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
        if self.latency_rows:
            path = self.raw / "rows.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
                writer.writeheader()
                writer.writerows(self.latency_rows)
            self.artifacts.append("raw/rows.csv")
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
        return time.monotonic() - self.t0 > self.mode.deadline_s

    def work(self) -> None:
        """From the images to the samples; kit.run_check() runs it, then build()."""
        for error in kit.input_errors(self.args):
            self.problem(error)
        try:
            self.queries = query_mod.select(tuple(self.args.names))
        except ValueError as e:
            self.problem(str(e))
        if self.problems:
            return
        root = self.fetch()
        if root is not None:
            self.start_and_load(root)
        if not self.problems and not _regressed(self.findings):
            self.sample_params()
            self.explain()
            self.measure()

    def build(self, error: str | None) -> dict[str, Any]:
        """The result from what work() measured, also when it was stopped or crashed
        (`error`). kit.run_check() calls it with the stop signals ignored."""
        self.collect()
        self.write_samples()
        self.write_summary()
        return self.result(error=error)

    def cleanup(self) -> None:
        """Remove both servers, also one named but stopped before kit.start() returned."""
        kit.remove(*(s.container_name for s in self.sides.values() if s.container_name))

    def fetch(self) -> Path | None:
        sf = self.mode.sf
        self.echo(f"dataset: SF{sf} into {self.args.cache}")
        try:
            root = dataset_mod.fetch(self.args.cache, sf, echo=self.echo)
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
                kit.pull(image)
            except kit.DockerError as e:
                self.problem(f"{side} image: {e}")
        if self.problems:
            return

        load: dict[str, Any] = {}
        for side in SIDES:  # baseline first: a broken baseline stops the run early
            s = self.sides[side]
            name = f"perf-ldbc-{side}-{os.getpid()}"
            # Named before it exists, so cleanup() can remove it even when a
            # SIGTERM lands between `docker run` and `start` returning.
            s.container_name = name
            try:
                volumes = {self.args.cache: "/import:ro"}
                s.container = kit.start(s.image, name, env=SERVER_ENV, volumes=volumes, cpus=cpus)
                client = self.connect(s)
            except (kit.DockerError, RuntimeError, RedisError) as e:
                self.side_failed(side, "server start", f"server did not start: {e}")
                return
            self.echo(f"{side}: port {s.container.port}")
            try:
                stats = loader_mod.load(client, root, import_root=self.args.cache, echo=self.echo)
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
        c, b = load[CANDIDATE]["entities_per_s"], load[BASELINE]["entities_per_s"]
        self.findings.append(
            kit.finding("load", "throughput", c, b, "info", direction="lower_is_worse")
        )

    def side_failed(self, side: str, name: str, why: str) -> None:
        """A side that could not start or load: the baseline's failure leaves
        nothing to compare against (error); the candidate's is a regression."""
        if side == BASELINE:
            self.problem(f"baseline {why}")
            return
        self.findings.append(
            kit.finding(name, "completed", 0, 1, "regression", direction="lower_is_worse")
        )
        self.notes.append(f"candidate {why}")

    def connect(self, s: Side) -> BenchClient:
        """Wait for the server and switch off its background saves."""
        assert s.container is not None
        port = s.container.port
        if not port or not kit.wait_ping(port, STARTUP_S, s.container):
            # port 0: the server exited before docker published its port.
            raise RuntimeError(f"no answer to PING within {STARTUP_S:.0f}s; {_how_it_ended(s)}")
        db = FalkorDB(
            host="127.0.0.1",
            port=port,
            retry=Retry(NoBackoff(), 0),
            # A load or sampling request that never returns is a load that
            # did not finish, not a check that hangs until it is killed.
            socket_timeout=self.mode.load_request_s,
        )
        # The images keep redis's default save points; one would fork a
        # BGSAVE a minute into the measurement, on one side only.
        db.connection.config_set("save", "")
        s.conn = Redis(
            host="127.0.0.1",
            port=port,
            retry=Retry(NoBackoff(), 0),
            socket_timeout=self.mode.timeout_ms / 1000 + SOCKET_SLACK_S,
        )
        client = BenchClient(db=db, server=Server(port=port))
        client.graph_name = GRAPH
        return client

    def sample_params(self) -> None:
        mode = self.mode
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
        count = self.args.param_count or mode.params
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
        mode = self.mode
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
                elif not confirmed.findings and not confirmed.rows:
                    self.problem(f"{q.name}: the re-run did not finish on both sides")
                self.findings += confirmed.findings
                self.latency_rows += confirmed.rows
            self.record_summary(q.name, main[q.name], reruns.get(q.name))

        if self.deadline_skips:
            self.problem(
                f"the {self.mode.deadline_s / 60:.0f} min deadline passed before every "
                f"query had its rounds ({self.deadline_skips} parameter-row runs skipped)"
            )
        if self.budget_skips:
            self.problem(
                f"the {self.mode.query_budget_s:.0f}s per-round budget cut "
                f"{self.budget_skips} parameter-row runs"
            )
        if self.compared_queries == 0:
            self.problem("no query could be compared on both sides")

        # Every main-round comparison, crossing or not, for raw/rows.csv.
        for q in self.queries:
            v = verdicts[q.name]
            if v.left_out or not v.measured or v.rows_differ or v.no_rows:
                continue
            for cmp in compare_latency(v.cand, v.base):
                status = "crossed: re-run" if cmp.metric in v.crossed else cmp.crossing() or "ok"
                self.latency_rows.append(latency_row(q.name, "main", cmp, status))

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
        mode = self.mode
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
        sample = side.run(q.cypher, params, self.mode.timeout_ms)
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

    def collect(self) -> None:
        """Each server's log and how it ended, the server findings, raw/run.json."""
        started = {side for side in SIDES if self.sides[side].container is not None}
        for side in SIDES:
            s = self.sides[side]
            if s.container is None:
                continue
            try:
                self.artifact(f"raw/{side}-server.log", kit.logs(s.container.name))
            except (kit.DockerError, OSError) as e:
                self.notes.append(f"{side} logs: {e}")
            try:
                state = kit.state(s.container.name)
                # Only an exit is a server that went down; `gone` (no such
                # container) says nothing about how the server ran.
                if state.startswith(("exited", "dead")) and not s.down:
                    s.down = f"container {state}"
                elif state != "running" and not s.down:
                    self.notes.append(f"{side} container at the end: {state}")
            except kit.DockerError as e:
                self.notes.append(f"{side} state: {e}")
            if s.down:
                self.notes.append(f"{side} server: {s.down}")
        self.server_findings(started)
        engines = {side: {"down": self.sides[side].down or None} for side in SIDES}
        self.artifact(
            "raw/run.json",
            json.dumps(
                {
                    "mode": asdict(self.mode),
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
                kit.finding("server up", "completed", c, b, status, direction="lower_is_worse")
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
        return f"{f['name']} {round(f['ratio'], 3)}x"
    return f"{f['name']} {f['candidate']} vs {f['baseline']}"


def _text(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def _qnum(name: str) -> int:
    return int(name.removeprefix("IC"))


def _how_it_ended(s: Side) -> str:
    """The container's state and the end of its log, for a server that never answered."""
    assert s.container is not None
    try:
        return f"container {kit.state(s.container.name)}: {kit.logs(s.container.name)[-400:]}"
    except kit.DockerError as e:
        return f"container state unknown ({e})"


def run_check(args: argparse.Namespace, echo: Echo = say) -> dict[str, Any]:
    """Run the check through kit.run_check(): `<out>/result.json` is written
    whatever happens, and both servers are removed afterwards."""
    check = Check(args, echo)
    return kit.run_check(CHECK, args, check.work, check.build, check.cleanup)


def main(argv: list[str] | None = None) -> int:
    """`bench ldbc check`: exits 0 whenever result.json was written, whatever its verdict."""
    args = parse_args(argv)
    result = run_check(args)
    say(f"\n{result['verdict']}: {result['summary']}")
    say(f"wrote {Path(args.out) / 'result.json'}")
    return 0
