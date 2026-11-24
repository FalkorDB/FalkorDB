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
* `invariant` "same rows": the row contents differ although the counts agree,
  between two images of the same engine (`info` while it proves itself; see
  the README).

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
import hashlib
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

#: A client-side socket timeout this far above the server-side TIMEOUT: when
#: it fires, the server ignored its own timeout and is treated as hung.
SOCKET_SLACK_S = 30.0

IMAGE_RE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

#: Queries whose result legitimately depends on tie order: IC14 returns every
#: shortest path, and paths of equal weight have no defined order, so a
#: changed plan can list them differently. Row contents are compared as a
#: sorted multiset, which absorbs that; the note is kept for the reader of a
#: "same rows" finding.
TIE_DEPENDENT = frozenset({"IC14"})


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
    digest: str | None = None
    error: str = ""


SAMPLE_FIELDS = tuple(Sample.__dataclass_fields__)


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile: a value that actually occurred."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), -(-p * len(ordered) // 100)))
    return ordered[int(rank) - 1]


def normalise(value: Any) -> Any:
    """A reply value in a form both engines print the same way.

    Doubles come back as text, and the engines format them differently
    (`0.1` vs `0.10000000000000001`), so anything that reads as a
    non-integral number is rounded to nine significant digits. Both sides go
    through the same function, so a string that happens to look like a
    number is changed identically on each.
    """
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        if re.fullmatch(r"[-+]?(\d+\.\d*|\.\d+|\d+(\.\d*)?[eE][-+]?\d+|inf|nan)", value):
            return f"{float(value):.9g}"
        return value
    if isinstance(value, float):
        return f"{value:.9g}"
    if isinstance(value, (list, tuple)):
        return [normalise(v) for v in value]
    return value


def digest(rows: list[Any]) -> str:
    """A digest of the rows as a sorted multiset (order-independent)."""
    lines = sorted(json.dumps(normalise(row), separators=(",", ":")) for row in rows)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]


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
    conn: Redis | None = None
    engine: str = ""
    module_version: int | None = None
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
                self.down = "stopped answering after a client-side timeout"
            return sample
        except (RedisConnectionError, OSError) as e:
            state = self.container.state() if self.container else "unknown"
            self.down = f"connection lost ({e}); container {state}"
            sample.outcome = DOWN
            sample.error = self.down
            return sample
        sample.latency_ms = (time.perf_counter() - started) * 1000
        if isinstance(reply, list) and reply and isinstance(reply[-1], ResponseError):
            return _error_sample(sample, str(reply[-1]))
        rows = reply[1] if isinstance(reply, list) and len(reply) == 3 else []
        sample.rows = len(rows)
        sample.digest = digest(rows)
        return sample

    def _answers(self) -> bool:
        assert self.container is not None
        try:
            probe = Redis(
                host="127.0.0.1",
                port=self.container.port,
                socket_timeout=5,
                retry=Retry(NoBackoff(), 0),
            )
            return bool(probe.ping())
        except RedisError:
            return False


def _error_sample(sample: Sample, message: str) -> Sample:
    sample.latency_ms = None
    sample.outcome = TIMEOUT if "timed out" in message.lower() else ERROR
    sample.error = message[:300]
    return sample


def module_version(reply: Any) -> int | None:
    """The graph module's `ver` from a MODULE LIST reply.

    redis-py hands it back as a list of flat `[name, …, ver, …]` lists, or as
    dicts when a response callback has parsed it; both are accepted.
    """
    for module in reply or []:
        if isinstance(module, dict):
            info = {_text(k): v for k, v in module.items()}
        else:
            info = {_text(k): v for k, v in zip(module[::2], module[1::2], strict=False)}
        if _text(info.get("name", "")) == "graph":
            try:
                return int(_text(info.get("ver", "")))
            except ValueError:
                return None
    return None


def engine_of(version: int | None) -> str:
    """`c` or `rust`, from the module version MODULE LIST reports.

    The C engine is the 4.x line (42204 for v4.22.4); the Rust engine
    reports 5.x and up, and 999999 on a development build.
    """
    if version is None:
        return "unknown"
    return "c" if version < 50000 else "rust"


# --- analysis ----------------------------------------------------------------


@dataclass
class SideStats:
    attempted: int = 0
    ok: int = 0
    timeouts: int = 0
    errors: int = 0
    down: int = 0
    first_failure: str = ""
    ok_params: set[int] = field(default_factory=set)
    failed_params: set[int] = field(default_factory=set)
    rows: dict[int, int] = field(default_factory=dict)
    digests: dict[int, str] = field(default_factory=dict)
    latencies: dict[int, list[float]] = field(default_factory=dict)

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
    """The `completed` finding, decided per parameter row.

    The candidate regressed when it failed on a row the baseline always
    finished. Failures on both sides (on the same rows, or each on rows the
    other also failed) are `info`; the baseline failing where the candidate
    finished is an improvement.
    """

    def only(a: SideStats, b: SideStats) -> list[int]:
        return sorted(p for p in a.failed_params if p in b.ok_params and p not in b.failed_params)

    if only(cand, base):
        status, c, b = "regression", 0, 1
    elif cand.failed_params and base.failed_params:
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
            stats.ok_params.add(s.param)
            stats.rows.setdefault(s.param, s.rows or 0)
            if s.digest is not None:
                stats.digests.setdefault(s.param, s.digest)
            if s.latency_ms is not None and s.phase != "warmup":
                stats.latencies.setdefault(s.param, []).append(s.latency_ms)
            continue
        stats.failed_params.add(s.param)
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
        """`regression` or `improvement` when outside the band, else None."""
        floor = FLOOR_MS[self.metric]
        delta = self.candidate - self.baseline
        if self.baseline <= 0:
            return "regression" if delta >= floor else None
        ratio = self.candidate / self.baseline
        if ratio >= RATIO and delta >= floor:
            return "regression"
        if ratio <= 1 / RATIO and -delta >= floor:
            return "improvement"
        return None


def compare_latency(cand: SideStats, base: SideStats) -> list[LatencyCompare]:
    """p50 and p95 over the parameter rows both sides finished."""
    common = sorted(set(cand.latencies) & set(base.latencies))
    c = [v for p in common for v in cand.latencies[p]]
    b = [v for p in common for v in base.latencies[p]]
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
    #: At least one parameter row finished on both sides, so the two were
    #: actually compared (a query broken on both sides is measured, not
    #: compared).
    compared: bool = False


def analyse(
    query: str,
    main: list[Sample],
    *,
    same_engine: bool,
) -> QueryVerdict:
    """Findings from the main rounds (latency crossings still unconfirmed)."""
    verdict = QueryVerdict(query)
    base = side_stats([s for s in main if s.side == BASELINE])
    cand = side_stats([s for s in main if s.side == CANDIDATE])

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

    common = sorted(set(cand.rows) & set(base.rows))
    verdict.compared = bool(common)
    mismatched = [p for p in common if cand.rows[p] != base.rows[p]]
    if mismatched:
        c_total = sum(cand.rows[p] for p in common)
        b_total = sum(base.rows[p] for p in common)
        name = f"{query} rows"
        if c_total == b_total:
            first = mismatched[0]
            name, c_total, b_total = (
                f"{query} rows (param {first})",
                cand.rows[first],
                base.rows[first],
            )
        verdict.findings.append(
            _finding(name, "rows", c_total, b_total, direction="mismatch", status="regression")
        )
    elif common:
        if all(base.rows[p] == 0 for p in common):
            verdict.findings.append(
                _finding(
                    f"{query} returns rows",
                    "invariant",
                    0,
                    0,
                    direction="lower_is_worse",
                    status="info",
                )
            )
        differing = [
            p
            for p in common
            if p in cand.digests and p in base.digests and cand.digests[p] != base.digests[p]
        ]
        if differing and same_engine:
            verdict.findings.append(
                _finding(
                    f"{query} same rows",
                    "invariant",
                    None,
                    None,
                    direction="lower_is_worse",
                    status="info",
                )
            )

    verdict.rows_differ = bool(mismatched)
    for cmp in [] if verdict.rows_differ else compare_latency(cand, base):
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


def confirm(
    query: str,
    crossed: list[str],
    rerun: list[Sample],
) -> list[dict[str, Any]]:
    """The re-run's verdict on the metrics the main rounds crossed.

    A metric that crosses again is a regression (`confirmed`). One that does
    not is reported as `ok` with the re-run's numbers: the first crossing was
    noise, or too small to repeat.
    """
    base = side_stats([s for s in rerun if s.side == BASELINE])
    cand = side_stats([s for s in rerun if s.side == CANDIDATE])
    out = []
    done = completion(query, cand, base)
    if done is not None and done["status"] == "regression":
        out.append(done)
    by_metric = {c.metric: c for c in compare_latency(cand, base)}
    for metric in crossed:
        cmp = by_metric.get(metric)
        if cmp is None:
            continue
        status = "regression" if cmp.crossing() == "regression" else "ok"
        out.append(_latency_finding(query, cmp, status))
    return out


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
        pass
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

    # -- result -----------------------------------------------------------

    def image_block(self, side: str) -> dict[str, Any]:
        o = self.opts
        image = o.candidate if side == CANDIDATE else o.baseline
        label = o.candidate_label if side == CANDIDATE else o.baseline_label
        commit = o.candidate_commit if side == CANDIDATE else o.baseline_commit
        if commit and not COMMIT_RE.match(commit):
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
            if not IMAGE_RE.match(image):
                self.problem(f"--{side} {image!r} is not pinned by digest (name@sha256:<64 hex>)")
        for side, commit in ((CANDIDATE, o.candidate_commit), (BASELINE, o.baseline_commit)):
            if commit and not COMMIT_RE.match(commit):
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
            try:
                s.container = containers.start(s.image, name, self.opts.cache, cpus=cpus)
                client = self.connect(s)
            except (containers.DockerError, RuntimeError, RedisError) as e:
                if side == BASELINE:
                    self.problem(f"baseline server did not start: {e}")
                    return
                self.findings.append(
                    _finding(
                        "server start",
                        "completed",
                        0,
                        1,
                        direction="lower_is_worse",
                        status="regression",
                    )
                )
                self.notes.append(f"candidate server did not start: {e}")
                return
            self.echo(
                f"{side}: {s.engine} engine (module {s.module_version}), port {s.container.port}"
            )
            try:
                stats = loader_mod.load(client, root, import_root=self.opts.cache, echo=self.echo)
            except (loader_mod.LoadError, RedisError) as e:
                load[side] = {"error": str(e)}
                self.artifact("raw/load.json", json.dumps(load, indent=2) + "\n")
                if side == BASELINE:
                    self.problem(f"baseline could not load the dataset: {e}")
                    return
                self.findings.append(
                    _finding(
                        "load completed",
                        "completed",
                        0,
                        1,
                        direction="lower_is_worse",
                        status="regression",
                    )
                )
                self.notes.append(f"candidate could not load the dataset: {e}")
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
                db = FalkorDB(
                    host="127.0.0.1",
                    port=port,
                    retry=Retry(NoBackoff(), 0),
                    # A load or sampling request that never returns is a load
                    # that did not finish, not a check that hangs to its kill.
                    socket_timeout=self.opts.mode.load_request_s,
                )
                db.connection.ping()
                break
            except RedisError as e:
                last = e
                time.sleep(0.5)
        else:
            raise RuntimeError(f"no answer on :{port} within {STARTUP_S:.0f}s ({last})")
        # The images keep redis's default save points; one would fork a
        # BGSAVE a minute into the measurement, on one side only.
        db.connection.config_set("save", "")
        s.module_version = module_version(db.connection.execute_command("MODULE", "LIST"))
        s.engine = engine_of(s.module_version)
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

        same_engine = (
            self.sides[BASELINE].engine == self.sides[CANDIDATE].engine
            and self.sides[BASELINE].engine != "unknown"
        )
        verdicts = {
            q.name: analyse(q.name, main[q.name], same_engine=same_engine) for q in self.queries
        }
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
                confirmed = confirm(q.name, v.crossed, reruns.get(q.name, []))
                if not confirmed:
                    self.problem(f"{q.name}: the re-run did not finish on both sides")
                self.findings += confirmed
            self.record_summary(q.name, main[q.name], reruns.get(q.name))

        if self.deadline_skips:
            self.problem(
                f"the {self.opts.mode.deadline_s / 60:.0f} min deadline passed before every "
                f"query had its rounds ({self.deadline_skips} parameter rows not run)"
            )
        if self.compared_queries == 0:
            self.problem("no query could be compared on both sides")

        # The rows that did not cross, as candidates for the informative slots.
        for q in self.queries:
            v = verdicts[q.name]
            if v.left_out or not v.measured or v.rows_differ:
                continue
            base = side_stats([s for s in main[q.name] if s.side == BASELINE])
            cand = side_stats([s for s in main[q.name] if s.side == CANDIDATE])
            for cmp in compare_latency(cand, base):
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
            if self.past_deadline():
                reason = "past the check's deadline"
                self.deadline_skips += 1
            elif time.monotonic() - started > mode.query_budget_s:
                reason = f"over the {mode.query_budget_s:.0f}s per-round budget"
            elif i in self.timed_out.get(q.name, set()):
                reason = "a side timed out on this parameter row before"
            elif any(self.sides[s].timeouts.get(q.name, 0) >= MAX_TIMEOUTS for s in SIDES):
                reason = f"a side timed out {MAX_TIMEOUTS} times on this query"
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
        for phase, samples in (("main", main), ("rerun", rerun or [])):
            for side in SIDES:
                st = side_stats([s for s in samples if s.side == side])
                if st.attempted == 0:
                    continue
                lat = [v for vs in st.latencies.values() for v in vs]
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
                continue
            try:
                self.artifact(f"raw/{side}-server.log", s.container.logs())
                state = s.container.state()
                if state != "running" and not s.down:
                    s.down = f"container {state}"
            except (containers.DockerError, OSError) as e:
                self.notes.append(f"{side} logs: {e}")
            if s.down:
                self.notes.append(f"{side} server: {s.down}")
            try:
                s.container.remove()
            except containers.DockerError as e:
                self.notes.append(f"{side} container not removed: {e}")
        self.server_findings(started)
        engines = {
            side: {
                "engine": self.sides[side].engine,
                "module_version": self.sides[side].module_version,
                "down": self.sides[side].down or None,
            }
            for side in SIDES
        }
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
                    "tie_dependent": sorted(TIE_DEPENDENT),
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


def run_check(opts: Options, echo: Echo = print) -> dict[str, Any]:
    """Run the check and write `<out>/result.json`; never raises once that
    file can be written."""
    check = Check(opts, echo)
    try:
        with _term_as_interrupt():
            return check.run()
    except (KeyboardInterrupt, Exception) as e:
        reason = "stopped" if isinstance(e, KeyboardInterrupt) else "crashed"
        try:
            check.write_samples()
            check.write_summary()
        except OSError:
            pass
        return check.write(error=f"the check {reason}: {type(e).__name__}: {e}")
