"""`bench probe`: costs that grow with the graph (the nightly perf check `size-probe`).

The micro tier measures every query on a 10k-node graph (1k under callgrind).
A cost that grows with the node count hides there: 6.2.0's
`UNWIND … MATCH … MATCH … CREATE` planned a hash join over a full label scan
for every batch (#3887, #3965), 28.6 ms against 6.0.4's 0.8 ms per batch at
200k nodes, and at bench sizes one scan of 1k-10k nodes costs about what the
per-row seeks cost. So this check runs a few write and lookup shapes on graphs
of 10k, 100k and 500k nodes and asks one question per shape: does the
candidate's cost, relative to the baseline's, grow with the graph?

## How it measures

Both images run at once on this machine, each its own server (`docker run`,
pinned to `PERF_SERVER_CPUS` when set, on a free port, never 6379), with
`save` and `appendonly` off. Each server gets the same three graphs
(`probe_10k`, …): an index on `:N(id)` created on the empty label, N nodes,
and a ring of N edges. Nothing is measured until the index reports
OPERATIONAL on both servers.

Every (size, shape) pair is one unit. A unit runs a warm-up on each side, then
four blocks, A B B A, where A is the candidate in every other unit and the
baseline in the rest. Both sides get the same parameter sequence (seeded by
the unit), so request k on one side is request k on the other. A latency is
the client's wall clock around one request on a plain socket: one request,
never re-sent. The server's own "internal execution time" is kept beside it
in raw/rows.csv.

## What it decides

For each shape, the ratio candidate p50 / baseline p50 at every size, and

    growth_ratio = ratio at the largest size / ratio at the smallest size

which equals the candidate's own slowdown from the smallest to the largest
graph divided by the baseline's. Above `THRESHOLD` the shape is re-measured at
every size with the order swapped, and it is a regression only when the
re-run alone crosses again (else `info`). Below 1 / `THRESHOLD` it is an
improvement. A constant-factor change (the candidate 2x slower at every size)
does not move the growth ratio; the micro tier is where that shows.

Correctness needs no re-run: a shape that errors, times out or loses its
server on the candidate while the baseline answers is a `completed`
regression, and a request whose rows (or relationships created) differ
between the sides is a `rows` mismatch. A shape the baseline cannot run is
left out with an `info` finding (an older engine may lack it); a baseline
server that stops answering, or a setup that fails on the baseline, makes the
verdict `error`, since nothing can be compared after it.

Standard library only, apart from the click command in cli.py.
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import platform
import random
import re
import signal
import socket
import statistics
import subprocess
import time
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

CHECK = "size-probe"
SIDES = ("candidate", "baseline")

# Node counts. The same in every mode (contract table: 10k / 100k / 500k).
SIZES = (10_000, 100_000, 500_000)

# A shape crosses when its growth ratio passes this, or improves below its
# inverse. 6.2.0's regression grows by an order of magnitude over this span.
THRESHOLD = 1.5

# A block stops early once it has run this long, but never below MIN_REPS.
MIN_REPS = 5

REQUEST_TIMEOUT_S = 120.0  # one measured request; past it the shape did not finish
BUILD_TIMEOUT_S = 900.0  # one graph-building request
SETTLE_TIMEOUT_S = 600.0  # after a timed-out request, wait this long for the graph
START_TIMEOUT_S = 120.0  # docker run until PING answers
INDEX_TIMEOUT_S = 300.0  # until db.indexes() says OPERATIONAL
BUILD_BATCH = 50_000  # nodes (and edges) per building request

# Contract limits (START-HERE §2, v1.1).
MAX_FINDINGS = 50
MAX_QUIET = 10

DIGEST_RE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ModeSpec:
    """One `--mode`: how long each unit runs. Never what it measures."""

    warmup: int  # requests per side before the first block, unmeasured
    reps: int  # requests per block; each side runs two blocks per unit
    block_seconds: float  # a block stops after this long once it has MIN_REPS
    budget_minutes: float  # no unit starts after this; the run is then an error


# Measured on an n4-standard-16 (2026-11-24, nightly): a whole run takes 1-3
# minutes; 6.2.0's slowest unit, 85 ms per request at 500k nodes, fills a
# 60-request block in about 5 s. The caps bound a pathological candidate: at
# most 36 units (18 + 18 re-run) x 4 blocks x block_seconds, which is 10, 19
# and 48 minutes for queue, nightly and long, inside each budget.
MODES = {
    "queue": ModeSpec(warmup=3, reps=30, block_seconds=4.0, budget_minutes=20.0),
    "nightly": ModeSpec(warmup=5, reps=60, block_seconds=8.0, budget_minutes=35.0),
    "long": ModeSpec(warmup=10, reps=150, block_seconds=20.0, budget_minutes=100.0),
}


# --- the shapes --------------------------------------------------------------

INDEX_MATCH = "MATCH (a:N {id: e.s}) MATCH (b:N {id: e.d})"
ID_MATCH = "MATCH (a) WHERE id(a) = e.s MATCH (b) WHERE id(b) = e.d"


@dataclass(frozen=True)
class Shape:
    """One measured query, parameterised per request.

    The four UNWIND shapes are the micro tier's #3963 rows
    (bench/src/falkorbench/queries.py, `_unwind_edges`) with endpoints drawn
    from the whole graph instead of the first 1,000 nodes: each request creates
    its batch of edges and deletes them again, so the graph stays as it was.
    """

    name: str
    kind: str  # "edges" | "lookup" | "create"
    rows: int = 0  # rows per UNWIND batch ("edges" only)
    match: str = ""  # the two MATCH clauses ("edges" only)

    def params(self, rng: random.Random, size: int) -> Any:
        if self.kind == "edges":
            return [(rng.randrange(size), rng.randrange(size)) for _ in range(self.rows)]
        return rng.randrange(size)

    def query(self, p: Any) -> str:
        if self.kind == "edges":
            rows = ", ".join(f"{{s: {s}, d: {d}}}" for s, d in p)
            return (
                f"CYPHER rows=[{rows}] UNWIND $rows AS e {self.match} "
                "CREATE (a)-[r:UWE]->(b) WITH r DELETE r"
            )
        if self.kind == "lookup":
            return f"CYPHER id={p} MATCH (n:N {{id: $id}}) RETURN n.id"
        if self.kind == "create":
            return f"CYPHER v={p} CREATE (:W {{v: $v}})"
        raise ValueError(f"unknown shape kind {self.kind!r}")

    def count(self, reply: Reply) -> int:
        """What one request did: compared request by request between the sides."""
        if self.kind == "edges":
            return reply.stat_int("Relationships created")
        if self.kind == "lookup":
            return len(reply.rows)
        return reply.stat_int("Nodes created")


SHAPES = (
    Shape("UNWIND index MATCH edge 1 row", "edges", 1, INDEX_MATCH),
    Shape("UNWIND index MATCH edge 100 rows", "edges", 100, INDEX_MATCH),
    Shape("UNWIND id MATCH edge 1 row", "edges", 1, ID_MATCH),
    Shape("UNWIND id MATCH edge 100 rows", "edges", 100, ID_MATCH),
    # Controls: neither should grow with the graph.
    Shape("indexed lookup", "lookup"),
    Shape("create node", "create"),
)


def size_label(size: int) -> str:
    return f"{size // 1000}k" if size % 1000 == 0 else str(size)


def graph_name(size: int) -> str:
    return f"probe_{size_label(size)}"


# --- a RESP client that never re-sends ---------------------------------------


class QueryError(Exception):
    """The server answered with an error."""


class Timeout(Exception):
    """No reply within the request's timeout. The connection is closed: a
    re-sent request would measure two requests as one."""


class Gone(Exception):
    """The connection failed: refused, reset or closed by the server."""


class ErrorReply(str):
    """An error nested inside an array reply."""


def encode(words: Sequence[str]) -> bytes:
    out = [b"*%d\r\n" % len(words)]
    for word in words:
        data = word.encode()
        out.append(b"$%d\r\n%s\r\n" % (len(data), data))
    return b"".join(out)


class RespReader:
    """Parses RESP2 replies out of whatever `recv` returns."""

    def __init__(self, recv: Callable[[], bytes]) -> None:
        self.recv = recv
        self.buf = bytearray()
        self.pos = 0

    def _fill(self) -> None:
        chunk = self.recv()
        if not chunk:
            raise ConnectionError("server closed the connection")
        if self.pos:
            del self.buf[: self.pos]
            self.pos = 0
        self.buf += chunk

    def _line(self) -> bytes:
        while True:
            end = self.buf.find(b"\r\n", self.pos)
            if end >= 0:
                line = bytes(self.buf[self.pos : end])
                self.pos = end + 2
                return line
            self._fill()

    def _exact(self, n: int) -> bytes:
        while len(self.buf) - self.pos < n + 2:
            self._fill()
        data = bytes(self.buf[self.pos : self.pos + n])
        self.pos += n + 2
        return data

    def read(self) -> Any:
        line = self._line()
        kind, rest = line[:1], line[1:]
        if kind == b"+":
            return rest.decode(errors="replace")
        if kind == b"-":
            return ErrorReply(rest.decode(errors="replace"))
        if kind == b":":
            return int(rest)
        if kind == b"$":
            n = int(rest)
            return None if n < 0 else self._exact(n).decode(errors="replace")
        if kind == b"*":
            n = int(rest)
            return None if n < 0 else [self.read() for _ in range(n)]
        raise ConnectionError(f"not a RESP2 reply: {line[:40]!r}")


class Conn:
    def __init__(self, port: int, timeout: float) -> None:
        try:
            self.sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        except OSError as e:
            raise Gone(f"cannot connect to 127.0.0.1:{port}: {e}") from e
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.reader = RespReader(lambda: self.sock.recv(1 << 16))

    def call(self, words: Sequence[str], timeout: float) -> Any:
        """Send one command and return its reply. Raises QueryError for an
        error reply; Timeout and Gone close the connection."""
        try:
            self.sock.settimeout(timeout)
            self.sock.sendall(encode(words))
            reply = self.reader.read()
        except TimeoutError as e:
            self.close()
            raise Timeout(f"no reply within {timeout:.0f} s") from e
        except (ConnectionError, OSError, ValueError) as e:
            self.close()
            raise Gone(str(e) or type(e).__name__) from e
        if isinstance(reply, ErrorReply):
            raise QueryError(str(reply))
        return reply

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.sock.close()


@dataclass
class Reply:
    """A GRAPH.QUERY reply: [header, rows, stats] or [stats]."""

    header: list[str]
    rows: list[Any]
    stats: dict[str, str]

    @classmethod
    def parse(cls, raw: Any) -> Reply:
        if not isinstance(raw, list) or not raw:
            raise QueryError(f"unexpected reply: {str(raw)[:200]}")
        for item in raw:
            if isinstance(item, ErrorReply):
                raise QueryError(str(item))
        stats: dict[str, str] = {}
        for line in raw[-1] or []:
            key, _, value = str(line).partition(":")
            stats[key.strip()] = value.strip()
        if len(raw) == 3:
            header = [h[-1] if isinstance(h, list) else h for h in raw[0] or []]
            return cls([str(h) for h in header], list(raw[1] or []), stats)
        return cls([], [], stats)

    def stat_int(self, key: str) -> int:
        value = self.stats.get(key)
        return int(float(value.split()[0])) if value else 0

    def server_ms(self) -> float | None:
        value = self.stats.get("Query internal execution time")
        return float(value.split()[0]) if value else None


# --- one image's server ------------------------------------------------------


class StartFailed(Exception):
    pass


def free_port() -> int:
    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if port != 6379:
            return port


def docker(*args: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


class Server:
    """One image's own server in a container, and one connection to it."""

    def __init__(self, side: str, image: str, cpus: str | None) -> None:
        self.side = side
        self.image = image
        self.cpus = cpus
        self.name = f"perf-probe-{side}-{os.getpid()}"
        self.port: int | None = None
        self.conn: Conn | None = None
        self.down: str | None = None  # why this server can no longer be measured
        self.created = False

    def pull(self) -> str | None:
        """Pull by digest, three tries. Returns the failure, or None."""
        err = ""
        for attempt in range(3):
            if attempt:
                time.sleep(15 * attempt)
            try:
                done = docker("pull", "-q", self.image, timeout=600)
            except subprocess.TimeoutExpired:
                err = "docker pull took over 10 minutes"
                continue
            if done.returncode == 0:
                return None
            err = (done.stderr or done.stdout).strip()[-300:]
        return f"docker pull {self.image} failed: {err}"

    def start(self) -> None:
        self.port = free_port()
        args = ["run", "-d", "--name", self.name, "-p", f"127.0.0.1:{self.port}:6379"]
        if self.cpus:
            args += ["--cpuset-cpus", self.cpus]
        done = docker(*args, self.image)
        self.created = True
        if done.returncode != 0:
            raise StartFailed(f"docker run failed: {done.stderr.strip()[-300:]}")
        deadline = time.monotonic() + START_TIMEOUT_S
        while True:
            try:
                if self.call(["PING"], timeout=5) == "PONG":
                    break
            except (Gone, Timeout, QueryError) as e:
                state = self.state()
                if "running" not in state or time.monotonic() > deadline:
                    raise StartFailed(f"no PONG ({e}; {state})") from e
                time.sleep(0.5)
        # The idle server must not snapshot while the other one is measured.
        for key, value in (("save", ""), ("appendonly", "no")):
            try:
                self.call(["CONFIG", "SET", key, value], timeout=10)
            except QueryError as e:
                raise StartFailed(f"CONFIG SET {key} failed: {e}") from e

    def call(self, words: Sequence[str], timeout: float) -> Any:
        if self.conn is None:
            if self.port is None:
                raise Gone("server not started")
            self.conn = Conn(self.port, timeout=10)
        try:
            return self.conn.call(words, timeout)
        except (Timeout, Gone):
            self.conn = None
            raise

    def query(
        self, graph: str, cypher: str, timeout: float = REQUEST_TIMEOUT_S
    ) -> tuple[Reply, float]:
        """One GRAPH.QUERY and its round trip in milliseconds."""
        if self.down:
            raise Gone(self.down)
        try:
            if self.conn is None:
                self.conn = Conn(self.port or 0, timeout=10)
            t0 = time.perf_counter_ns()
            raw = self.call(["GRAPH.QUERY", graph, cypher], timeout)
        except Gone as e:
            self.down = f"server stopped answering ({e}; {self.state()})"
            raise Gone(self.down) from e
        wall_ms = (time.perf_counter_ns() - t0) / 1e6
        return Reply.parse(raw), wall_ms

    def settle(self, graph: str) -> None:
        """After a timed-out request: wait until the graph takes a write again,
        so the next unit is not measured behind the request still running."""
        try:
            self.call(["GRAPH.QUERY", graph, "CREATE (z:ProbeSync) DELETE z"], SETTLE_TIMEOUT_S)
        except Timeout:
            self.down = f"still busy {SETTLE_TIMEOUT_S:.0f} s after a request timed out"
        except Gone as e:
            self.down = f"server stopped answering ({e}; {self.state()})"
        except QueryError:
            pass

    def state(self) -> str:
        try:
            done = docker(
                "inspect", "-f", "{{.State.Status}} exit {{.State.ExitCode}}", self.name, timeout=30
            )
        except subprocess.TimeoutExpired:
            return "docker inspect timed out"
        return f"container {(done.stdout or done.stderr).strip()}"

    def save_logs(self, path: Path) -> None:
        if not self.created:
            return
        try:
            done = docker("logs", self.name, timeout=60)
            path.write_text(done.stdout + done.stderr)
        except (subprocess.TimeoutExpired, OSError):
            pass

    def stop(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        if self.created:
            with contextlib.suppress(subprocess.TimeoutExpired):
                docker("rm", "-f", self.name, timeout=60)


# --- measurement -------------------------------------------------------------


@dataclass
class SideRun:
    """One side's requests in one unit."""

    wall: list[float] = field(default_factory=list)  # measured blocks only
    server: list[float] = field(default_factory=list)
    counts: dict[int, int] = field(default_factory=dict)  # request index -> count, warm-up too
    failure: str | None = None
    failure_kind: str | None = None  # "error" | "timeout" | "down"

    def fail(self, kind: str, why: str) -> None:
        if self.failure is None:
            self.failure_kind, self.failure = kind, why


@dataclass
class Cell:
    """One unit: a shape at a size in one phase ("screen" or "confirm")."""

    shape: str
    size: int
    phase: str
    first: str
    runs: dict[str, SideRun] = field(default_factory=lambda: {s: SideRun() for s in SIDES})


def other(side: str) -> str:
    return SIDES[1] if side == SIDES[0] else SIDES[0]


ROW_FIELDS = (
    "phase",
    "size",
    "shape",
    "side",
    "block",
    "request",
    "wall_ms",
    "server_ms",
    "count",
    "error",
)


def run_unit(
    servers: dict[str, Server],
    shape: Shape,
    size: int,
    phase: str,
    first: str,
    spec: ModeSpec,
    record: Callable[[dict], None],
) -> Cell:
    """Warm-up on each side, then blocks A B B A. Request k uses parameters k
    on both sides; a side's first block takes the first `reps` parameters
    after the warm-up and its second block the next `reps`."""
    graph = graph_name(size)
    rng = random.Random(f"{phase}/{size}/{shape.name}")
    params = [shape.params(rng, size) for _ in range(spec.warmup + 2 * spec.reps)]
    cell = Cell(shape.name, size, phase, first)

    def block(side: str, number: int, indexes: range, cap: float | None) -> None:
        run, server = cell.runs[side], servers[side]
        if run.failure:
            return
        t0 = time.monotonic()
        for i, k in enumerate(indexes):
            if cap is not None and i >= MIN_REPS and time.monotonic() - t0 > cap:
                return
            row = {"phase": phase, "size": size, "shape": shape.name, "side": side}
            row |= {"block": number, "request": k}
            try:
                reply, wall_ms = server.query(graph, shape.query(params[k]))
            except QueryError as e:
                run.fail("error", f"error reply: {e}")
            except Timeout as e:
                run.fail("timeout", f"did not finish: {e}")
                server.settle(graph)
            except Gone as e:
                run.fail("down", str(e))
            if run.failure:
                record(row | {"error": run.failure})
                return
            count = shape.count(reply)
            run.counts[k] = count
            server_ms = reply.server_ms()
            if number:  # block 0 is the warm-up
                run.wall.append(wall_ms)
                if server_ms is not None:
                    run.server.append(server_ms)
            row |= {"wall_ms": f"{wall_ms:.4f}", "count": count}
            if server_ms is not None:
                row["server_ms"] = f"{server_ms:.4f}"
            record(row)

    second = other(first)
    for side in (first, second):
        block(side, 0, range(spec.warmup), None)
    base = spec.warmup
    for number, (side, half) in enumerate(((first, 0), (second, 0), (second, 1), (first, 1)), 1):
        start = base + half * spec.reps
        block(side, number, range(start, start + spec.reps), spec.block_seconds)
    return cell


# --- the verdict (pure) ------------------------------------------------------


def p50(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=20, method="inclusive")[18]


def first_mismatch(cell: Cell) -> tuple[int, int, int] | None:
    """The first request whose count differs between the sides."""
    cand, base = cell.runs["candidate"].counts, cell.runs["baseline"].counts
    for k in sorted(cand.keys() & base.keys()):
        if cand[k] != base[k]:
            return k, cand[k], base[k]
    return None


def measured(cell: Cell | None) -> bool:
    return cell is not None and all(
        cell.runs[s].failure is None and cell.runs[s].wall for s in SIDES
    )


def ratio(cand: float | None, base: float | None) -> float | None:
    if cand is None or base is None or base <= 0:
        return None
    return cand / base


def growth(small: Cell, large: Cell) -> tuple[float, float, float] | None:
    """(candidate growth, baseline growth, growth ratio) between two cells."""
    cs, cl = p50(small.runs["candidate"].wall), p50(large.runs["candidate"].wall)
    bs, bl = p50(small.runs["baseline"].wall), p50(large.runs["baseline"].wall)
    if not (cs and cl and bs and bl):
        return None
    return cl / cs, bl / bs, (cl / bl) / (cs / bs)


def crosses(g: float) -> str | None:
    if g > THRESHOLD:
        return "regression"
    if g < 1 / THRESHOLD:
        return "improvement"
    return None


def _num(x: float | None) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    return round(x, 4)


def finding(
    name: str,
    metric: str,
    cand: float | None,
    base: float | None,
    status: str,
    *,
    threshold: float | None = None,
    direction: str = "higher_is_worse",
    rate: float | None = None,
) -> dict:
    if rate is None:
        rate = ratio(cand, base)
    return {
        "name": name,
        "metric": metric,
        "candidate": _num(cand),
        "baseline": _num(base),
        "ratio": _num(rate),
        "threshold": threshold,
        "direction": direction,
        "status": status,
    }


@dataclass
class ShapeResult:
    shape: str
    status: str | None = None  # of the growth finding: regression/improvement/ok/info
    growth: tuple[float, float, float] | None = None
    confirm_growth: float | None = None


@dataclass
class Assessment:
    loud: list[dict]  # regression and improvement findings
    quiet: list[list[dict]]  # info/ok findings in priority groups; a group is kept whole
    errors: list[str]
    shapes: list[ShapeResult]


def completed_finding(name: str, cand_ok: bool, base_ok: bool) -> dict:
    """Contract v1.4: candidate 0 with baseline 1 is a regression, both 0 is
    info. Candidate 1 with baseline 0 is also info here: the shape is left out,
    not compared (rule 1, an engine that cannot run it)."""
    status = "regression" if (base_ok and not cand_ok) else "info"
    return finding(
        name,
        "completed",
        int(cand_ok),
        int(base_ok),
        status,
        direction="lower_is_worse",
        rate=(1.0 if cand_ok else 0.0) if base_ok else None,
    )


def assess(
    screen: dict[tuple[str, int], Cell],
    confirm: dict[tuple[str, int], Cell],
    shapes: Sequence[Shape],
    sizes: Sequence[int],
) -> Assessment:
    loud: list[dict] = []
    left_out: list[dict] = []
    unconfirmed: list[dict] = []
    triples: list[tuple[float, list[dict]]] = []
    oks: list[dict] = []
    errors: list[str] = []
    results: list[ShapeResult] = []
    small, large = sizes[0], sizes[-1]

    for shape in shapes:
        res = ShapeResult(shape.name)
        results.append(res)
        # Correctness, in every cell of both phases: needs no re-run.
        rerun_failed = False
        for phase, cells in (("screen", screen), ("confirm", confirm)):
            for size in sizes:
                cell = cells.get((shape.name, size))
                if cell is None:
                    continue
                where = f"{shape.name} @{size_label(size)}"
                if phase == "confirm":
                    where += " (re-run)"
                c, b = cell.runs["candidate"], cell.runs["baseline"]
                if b.failure_kind == "down":
                    errors.append(f"baseline {b.failure} ({where})")
                if c.failure and not b.failure:
                    loud.append(completed_finding(where, False, True))
                    rerun_failed |= phase == "confirm"
                elif c.failure or b.failure:
                    left_out.append(completed_finding(where, not c.failure, not b.failure))
                mismatch = first_mismatch(cell)
                if mismatch:
                    _, cc, bc = mismatch
                    loud.append(finding(where, "rows", cc, bc, "regression", direction="mismatch"))

        lo, hi = screen.get((shape.name, small)), screen.get((shape.name, large))
        res.growth = growth(lo, hi) if measured(lo) and measured(hi) else None
        if res.growth is None:
            continue
        cand_g, base_g, g = res.growth
        res.status = crosses(g) or "ok"
        if res.status == "regression":
            clo, chi = confirm.get((shape.name, small)), confirm.get((shape.name, large))
            again = growth(clo, chi) if measured(clo) and measured(chi) else None
            res.confirm_growth = again[2] if again else None
            if again is None or again[2] <= THRESHOLD:
                res.status = "info"
            if again is None and not rerun_failed:
                errors.append(
                    f"{shape.name} crossed (growth {g:.2f}x) and the re-run could not measure it"
                )
        row = finding(
            shape.name, "growth_ratio", cand_g, base_g, res.status, threshold=THRESHOLD, rate=g
        )
        if res.status in ("regression", "improvement"):
            loud.append(row)
        elif res.status == "info":
            unconfirmed.append(row)
        else:
            oks.append(row)
        if res.status != "ok":
            triple = []
            for size in sizes:
                cell = screen.get((shape.name, size))
                if measured(cell):
                    cp, bp = p50(cell.runs["candidate"].wall), p50(cell.runs["baseline"].wall)
                    name = f"{shape.name} @{size_label(size)}"
                    triple.append(finding(name, "p50_ms", cp, bp, "info"))
            triples.append((abs(math.log(g)), triple))

    def by_ratio(rows: list[dict], reverse: bool = True) -> list[dict]:
        return sorted(
            rows,
            key=lambda f: (
                f["ratio"] is None,
                -(f["ratio"] or 0) if reverse else (f["ratio"] or 0),
            ),
        )

    regressions = by_ratio([f for f in loud if f["status"] == "regression"])
    improvements = by_ratio([f for f in loud if f["status"] == "improvement"], reverse=False)
    quiet = [[f] for f in left_out]
    quiet += [[f] for f in by_ratio(unconfirmed)]
    quiet += [t for _, t in sorted(triples, key=lambda x: -x[0])]
    quiet += [[f] for f in by_ratio(oks)]
    return Assessment(regressions + improvements, quiet, errors, results)


def select_findings(assessment: Assessment) -> list[dict]:
    """Every loud finding, then whole quiet groups while they fit the cap."""
    loud = assessment.loud[:MAX_FINDINGS]
    room = min(MAX_QUIET, MAX_FINDINGS - len(loud))
    quiet: list[dict] = []
    for group in assessment.quiet:
        if len(quiet) + len(group) <= room:
            quiet += group
    return loud + quiet


def summarize(verdict: str, assessment: Assessment, sizes: Sequence[int], error: str | None) -> str:
    """One line in plain words."""
    span = f"{size_label(sizes[0])} to {size_label(sizes[-1])} nodes"
    shapes = [s for s in assessment.shapes if s.growth is not None]
    regressed = [s for s in shapes if s.status == "regression"]
    improved = [s for s in shapes if s.status == "improvement"]
    correctness = [
        f for f in assessment.loud if f["status"] == "regression" and f["metric"] != "growth_ratio"
    ]
    parts = []
    if verdict == "error":
        parts.append(f"could not measure: {(error or '')[:300]}")
    if regressed:
        worst = max(regressed, key=lambda s: s.growth[2])
        parts.append(
            f"{len(regressed)} shape(s) slow down with graph size on the candidate, worst "
            f"{worst.shape}: its candidate/baseline ratio grows {worst.growth[2]:.2f}x from "
            f"{span} (re-run {worst.confirm_growth:.2f}x)"
        )
    if correctness:
        first = correctness[0]
        parts.append(
            f"{len(correctness)} correctness finding(s), first {first['name']} ({first['metric']})"
        )
    if verdict == "pass" and shapes:
        worst = max(shapes, key=lambda s: s.growth[2])
        parts.append(
            f"no shape's candidate/baseline ratio grew more than {THRESHOLD}x from {span} "
            f"({len(shapes)} shapes, highest {worst.shape} {worst.growth[2]:.2f}x)"
        )
    if improved:
        best = min(improved, key=lambda s: s.growth[2])
        parts.append(
            f"{len(improved)} shape(s) improved with size, best {best.shape} {best.growth[2]:.3f}x"
        )
    unconfirmed = [s for s in shapes if s.status == "info"]
    if unconfirmed:
        parts.append(f"{len(unconfirmed)} crossed once and not on the re-run")
    left = sum(1 for g in assessment.quiet if g and g[0]["metric"] == "completed")
    if left:
        parts.append(f"{left} left out because an engine could not run them")
    return "; ".join(parts).replace("\n", " ").replace("\r", " ")


# --- result.json -------------------------------------------------------------


def now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_label(image: str) -> str:
    """`<repo>@<first 12 hex of the digest>`, the contract's label when none is given."""
    name, _, digest = image.partition("@sha256:")
    if name.rfind(":") > name.rfind("/"):
        name = name[: name.rfind(":")]
    return f"{name}@{digest[:12]}" if digest else name


def machine() -> dict:
    """The workflow's machine object (contract v1.3), copied as is; else local."""
    path = os.environ.get("PERF_MACHINE_JSON")
    if path:
        return json.loads(Path(path).read_text())
    return local_machine()


def local_machine() -> dict:
    return {
        "runner": "local",
        "type": f"{platform.system()}-{platform.machine()}",
        "cpu": platform.processor() or platform.machine() or "unknown",
        "cores": os.cpu_count() or 1,
    }


def side_ref(image: str, label: str | None, commit: str | None) -> dict:
    return {"image": image, "commit": commit or None, "label": label or default_label(image)}


def write_summary(path: Path, cells: Iterable[Cell]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "phase", "size", "shape", "first",
                "candidate_n", "candidate_p50_ms", "candidate_p95_ms", "candidate_server_p50_ms",
                "baseline_n", "baseline_p50_ms", "baseline_p95_ms", "baseline_server_p50_ms",
                "ratio_p50", "candidate_failure", "baseline_failure", "first_mismatch",
            ]
        )  # fmt: skip
        for cell in cells:
            c, b = cell.runs["candidate"], cell.runs["baseline"]

            def fmt(x: float | None) -> str:
                return "" if x is None else f"{x:.4f}"

            mismatch = first_mismatch(cell)
            w.writerow(
                [
                    cell.phase, cell.size, cell.shape, cell.first,
                    len(c.wall), fmt(p50(c.wall)), fmt(p95(c.wall)), fmt(p50(c.server)),
                    len(b.wall), fmt(p50(b.wall)), fmt(p95(b.wall)), fmt(p50(b.server)),
                    fmt(ratio(p50(c.wall), p50(b.wall))), c.failure or "", b.failure or "",
                    "" if mismatch is None else f"request {mismatch[0]}: {mismatch[1]} vs {mismatch[2]}",
                ]
            )  # fmt: skip


# --- the run -----------------------------------------------------------------


class Stopped(Exception):
    """SIGTERM arrived (the workflow's timeout)."""


class Abort(Exception):
    """Setup failed or the arguments are unusable: nothing more to measure."""


@dataclass
class Setup:
    """What setup found before any unit ran."""

    loud: list[dict] = field(default_factory=list)
    quiet: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def side_failed(self, what: str, failed: dict[str, str]) -> None:
        """A setup step failed on one side or both. The candidate alone is a
        regression at once (contract §1: a crash needs no re-run); anything
        that fails on the baseline leaves nothing to compare with."""
        if "baseline" not in failed:
            self.loud.append(completed_finding(what, False, True))
            return
        if "candidate" in failed:
            self.quiet.append(completed_finding(what, False, False))
        self.errors.append(f"{what} failed on the baseline: {failed['baseline']}")


def build_graph(server: Server, size: int, log: Callable[[dict], None]) -> None:
    """Index first (on the empty label, so nothing populates in the
    background), then nodes and a ring of edges in BUILD_BATCH requests."""
    graph = graph_name(size)
    steps = [("index", "CREATE INDEX FOR (n:N) ON (n.id)")]
    for lo in range(0, size, BUILD_BATCH):
        hi = min(size, lo + BUILD_BATCH) - 1
        steps.append(("nodes", f"UNWIND range({lo}, {hi}) AS i CREATE (:N {{id: i}})"))
    for lo in range(0, size, BUILD_BATCH):
        hi = min(size, lo + BUILD_BATCH) - 1
        steps.append(
            (
                "edges",
                f"UNWIND range({lo}, {hi}) AS i MATCH (x:N {{id: i}}) "
                f"MATCH (y:N {{id: (i + 1) % {size}}}) CREATE (x)-[:R]->(y)",
            )
        )
    for step, cypher in steps:
        _, ms = server.query(graph, cypher, timeout=BUILD_TIMEOUT_S)
        log({"side": server.side, "size": size, "step": step, "seconds": f"{ms / 1000:.3f}"})
    nodes, _ = server.query(graph, "MATCH (n:N) RETURN count(n)", timeout=BUILD_TIMEOUT_S)
    edges, _ = server.query(graph, "MATCH (:N)-[r:R]->() RETURN count(r)", timeout=BUILD_TIMEOUT_S)
    got = (int(nodes.rows[0][0]), int(edges.rows[0][0]))
    if got != (size, size):
        raise QueryError(f"built {got[0]} nodes and {got[1]} edges, expected {size} of each")


def index_operational(server: Server, size: int) -> bool:
    reply, _ = server.query(graph_name(size), "CALL db.indexes()", timeout=60)
    header = [h.lower() for h in reply.header]
    if "label" not in header:
        raise QueryError(f"db.indexes() has no label column: {reply.header}")
    li = header.index("label")
    si = header.index("status") if "status" in header else None
    for row in reply.rows:
        if row[li] == "N":
            return si is None or "operational" in str(row[si]).lower()
    return False


def run(
    *,
    candidate: str,
    baseline: str,
    out: Path,
    mode: str,
    candidate_label: str | None = None,
    baseline_label: str | None = None,
    candidate_commit: str | None = None,
    baseline_commit: str | None = None,
    sizes: Sequence[int] | None = None,
    reps: int | None = None,
    echo: Callable[[str], None] = print,
) -> dict:
    """Measure, decide and write `<out>/result.json`, whatever happens short
    of being unable to write it. Returns the result."""
    started = now_utc()
    clock0 = time.monotonic()
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    images = {"candidate": candidate, "baseline": baseline}
    commits = {"candidate": candidate_commit, "baseline": baseline_commit}
    sizes = tuple(sorted(sizes or SIZES))
    setup = Setup()
    errors: list[str] = []
    screen: dict[tuple[str, int], Cell] = {}
    confirm: dict[tuple[str, int], Cell] = {}
    servers: dict[str, Server] = {}
    artifacts = ["raw/rows.csv", "raw/summary.csv", "raw/setup.csv"]

    for side, image in images.items():
        if not DIGEST_RE.match(image):
            errors.append(f"--{side} must be pinned by digest (name@sha256:<64 hex>): {image}")
        if commits[side] and not COMMIT_RE.match(commits[side]):
            errors.append(f"--{side}-commit must be 40 lowercase hex: {commits[side]}")
            commits[side] = None
    spec = MODES.get(mode)
    if spec is None:
        errors.append(f"unknown --mode {mode!r} (queue, nightly or long)")
    elif reps:
        spec = ModeSpec(min(spec.warmup, reps), reps, spec.block_seconds, spec.budget_minutes)
    if len(sizes) < 2:
        errors.append(f"growth needs at least two sizes, got {sizes}")

    # SIGTERM (run-check.sh's timeout) stops the measurement once. The handler
    # disarms itself before it raises, and every way out of the measurement
    # disarms it too, so a second signal (`uv run` forwards one as well) cannot
    # land in the cleanup and lose result.json. timeout's KILL is the backstop.
    armed = [True]

    def on_term(signum: int, frame: Any) -> None:
        if armed[0]:
            armed[0] = False
            raise Stopped()

    rows_fh = (raw / "rows.csv").open("w", newline="")
    rows_csv = csv.DictWriter(rows_fh, ROW_FIELDS)
    rows_csv.writeheader()
    setup_fh = (raw / "setup.csv").open("w", newline="")
    setup_csv = csv.DictWriter(setup_fh, ("side", "size", "step", "seconds", "error"))
    setup_csv.writeheader()
    previous = signal.signal(signal.SIGTERM, on_term)
    try:
        try:
            if errors:
                raise Abort()
            cpus = os.environ.get("PERF_SERVER_CPUS") or None
            servers = {side: Server(side, images[side], cpus) for side in SIDES}

            pulled = {side: servers[side].pull() for side in SIDES}
            errors += [why for why in pulled.values() if why]
            if errors:
                raise Abort()

            failed = {}
            for side in SIDES:
                echo(f"starting {side}: {images[side]}")
                try:
                    servers[side].start()
                except StartFailed as e:
                    failed[side] = str(e)
                    setup_csv.writerow({"side": side, "step": "start", "error": str(e)})
            if failed:
                setup.side_failed("server starts and answers PING", failed)
                raise Abort()

            # Every graph on both servers, alternating which side builds first.
            for i, size in enumerate(sizes):
                failed = {}
                for side in SIDES if i % 2 == 0 else SIDES[::-1]:
                    echo(f"building {graph_name(size)} on {side}...")
                    try:
                        build_graph(servers[side], size, setup_csv.writerow)
                    except (QueryError, Timeout, Gone) as e:
                        failed[side] = str(e)
                        setup_csv.writerow(
                            {"side": side, "size": size, "step": "build", "error": str(e)}
                        )
                if failed:
                    setup.side_failed(f"build the {size_label(size)} graph", failed)
                    raise Abort()

            # Nothing is measured while either server still populates an index.
            failed = {}
            deadline = time.monotonic() + INDEX_TIMEOUT_S
            for side in SIDES:
                for size in sizes:
                    try:
                        while not index_operational(servers[side], size):
                            if time.monotonic() > deadline:
                                raise Timeout(
                                    f"index on :N(id) not OPERATIONAL within {INDEX_TIMEOUT_S:.0f} s"
                                )
                            time.sleep(0.5)
                    except (QueryError, Timeout, Gone) as e:
                        failed.setdefault(side, f"{graph_name(size)}: {e}")
            if failed:
                setup.side_failed("index becomes OPERATIONAL", failed)
                raise Abort()

            stats = docker("stats", "--no-stream", *(s.name for s in servers.values()), timeout=60)
            (raw / "docker-stats-before.txt").write_text(stats.stdout + stats.stderr)
            artifacts.append("raw/docker-stats-before.txt")
            write_plans(raw / "plans.txt", servers, sizes[-1])
            artifacts.append("raw/plans.txt")

            budget_end = clock0 + spec.budget_minutes * 60

            def units(phase: str, todo: list[tuple[Shape, int, str]], store: dict) -> None:
                for n, (shape, size, first) in enumerate(todo):
                    if time.monotonic() > budget_end:
                        errors.append(
                            f"the {mode} budget ({spec.budget_minutes:.0f} min) ran out with "
                            f"{n} of {len(todo)} {phase} units measured"
                        )
                        return
                    cell = run_unit(servers, shape, size, phase, first, spec, rows_csv.writerow)
                    store[(shape.name, size)] = cell
                    rows_fh.flush()
                    echo(progress_line(cell))

            # A alternates between units, so each side leads half of them.
            todo = [(shape, size) for size in sizes for shape in SHAPES]
            units("screen", [(sh, sz, SIDES[i % 2]) for i, (sh, sz) in enumerate(todo)], screen)

            crossed = []
            for shape in SHAPES:
                lo, hi = screen.get((shape.name, sizes[0])), screen.get((shape.name, sizes[-1]))
                g = growth(lo, hi) if measured(lo) and measured(hi) else None
                if g is not None and crosses(g[2]) == "regression":
                    crossed.append(shape)
            if crossed:
                echo(f"re-running {len(crossed)} crossed shape(s) at every size, order swapped")
                redo = [
                    (shape, size, other(screen[(shape.name, size)].first))
                    for size in sizes
                    for shape in crossed
                    if (shape.name, size) in screen
                ]
                units("confirm", redo, confirm)
            armed[0] = False
        except Abort:
            armed[0] = False
        except Stopped:
            errors.append("stopped by SIGTERM (the workflow's timeout) before finishing")
        except Exception as e:  # a bug here must still leave a result.json saying error
            armed[0] = False
            errors.append(f"the check failed: {type(e).__name__}: {e}")
    except Stopped:  # it landed in a handler above, before that disarmed it
        errors.append("stopped by SIGTERM (the workflow's timeout) before finishing")
    finally:
        armed[0] = False
        for side, server in servers.items():
            server.save_logs(raw / f"{side}.log")
            server.stop()
            if server.created:
                artifacts.append(f"raw/{side}.log")
        rows_fh.close()
        setup_fh.close()

    try:
        write_summary(raw / "summary.csv", [*screen.values(), *confirm.values()])
    except OSError as e:
        errors.append(f"could not write raw/summary.csv: {e}")
    try:
        machine_obj = machine()
    except (OSError, ValueError) as e:
        machine_obj = local_machine()
        errors.append(f"PERF_MACHINE_JSON unreadable: {e}")
    result = build_result(
        screen=screen,
        confirm=confirm,
        sizes=sizes,
        candidate=side_ref(candidate, candidate_label, commits["candidate"]),
        baseline=side_ref(baseline, baseline_label, commits["baseline"]),
        machine=machine_obj,
        started=started,
        artifacts=artifacts,
        setup=setup,
        errors=errors,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    signal.signal(signal.SIGTERM, previous)
    return result


def build_result(
    *,
    screen: dict[tuple[str, int], Cell],
    confirm: dict[tuple[str, int], Cell],
    sizes: Sequence[int],
    candidate: dict,
    baseline: dict,
    machine: dict,
    started: str,
    artifacts: list[str],
    setup: Setup | None = None,
    errors: Sequence[str] = (),
) -> dict:
    """The contract-v1 result from what was measured. Pure: no server, no files."""
    setup = setup or Setup()
    errors = list(errors)
    try:
        assessment = assess(screen, confirm, SHAPES, sizes)
    except Exception as e:  # still a result, saying error
        errors.append(f"the check failed while deciding: {type(e).__name__}: {e}")
        assessment = Assessment([], [], [], [])
    assessment.loud = setup.loud + assessment.loud
    assessment.quiet = [[f] for f in setup.quiet] + assessment.quiet
    all_errors = errors + setup.errors + assessment.errors
    planned = len(sizes) * len(SHAPES)
    if not all_errors and not setup.loud and len(screen) < planned:
        all_errors.append(f"only {len(screen)} of {planned} units were measured")
    if not all_errors and not setup.loud and not any(s.growth for s in assessment.shapes):
        all_errors.append("no shape was measured on both sides at the smallest and largest size")
    findings = select_findings(assessment)

    if any(f["status"] == "regression" for f in findings):
        verdict = "regression"
    elif all_errors:
        verdict = "error"
    else:
        verdict = "pass"
    error = "; ".join(dict.fromkeys(all_errors))[:2000] if verdict == "error" else None
    return {
        "schema": 1,
        "check": CHECK,
        "verdict": verdict,
        "confirmed": verdict == "regression",
        "summary": summarize(verdict, assessment, sizes, error),
        "candidate": candidate,
        "baseline": baseline,
        "machine": machine,
        "started": started,
        "finished": now_utc(),
        "findings": findings,
        "error": error,
        "artifacts": artifacts,
    }


def write_plans(path: Path, servers: dict[str, Server], size: int) -> None:
    """GRAPH.EXPLAIN of every shape on both sides, at the largest size: shows
    which plan each engine chose (a hash join over a label scan, or a seek)."""
    with path.open("w") as fh:
        for shape in SHAPES:
            cypher = shape.query(shape.params(random.Random(f"plan/{shape.name}"), size))
            for side in SIDES:
                try:
                    plan = servers[side].call(["GRAPH.EXPLAIN", graph_name(size), cypher], 60)
                    text = "\n".join(str(line) for line in plan or [])
                except (QueryError, Timeout, Gone) as e:
                    text = f"(no plan: {e})"
                fh.write(f"== {shape.name} [{side}]\n{text}\n\n")


def progress_line(cell: Cell) -> str:
    c, b = cell.runs["candidate"], cell.runs["baseline"]
    line = (
        f"{cell.phase:7} {size_label(cell.size):>5} {cell.shape:<34} "
        f"cand p50 {_fmt_ms(p50(c.wall))} ms ({len(c.wall)})  "
        f"base p50 {_fmt_ms(p50(b.wall))} ms ({len(b.wall)})"
    )
    if c.failure:
        line += f"  candidate FAILED: {c.failure}"
    if b.failure:
        line += f"  baseline FAILED: {b.failure}"
    return line


def _fmt_ms(x: float | None) -> str:
    return "      -" if x is None else f"{x:7.3f}"
