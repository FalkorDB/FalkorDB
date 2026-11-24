#!/usr/bin/env python3
"""Pokec check: FalkorDB/benchmark on the Pokec dataset, candidate against baseline.

    pokec.py --candidate <image@sha256:…> --baseline <image@sha256:…> \\
             --out <dir> --mode <queue|nightly|long> \\
             [--candidate-label …] [--baseline-label …] \\
             [--candidate-commit …] [--baseline-commit …]

One check of the nightly perf contract (tests/perf/README.md, tests/perf/pokec/README.md).
On one machine and in one run it starts each image's own server (`docker run`, pinned to
`PERF_SERVER_CPUS`), bulk-loads Pokec into a fresh server for every run, drives
FalkorDB/benchmark's mixed workload (read-only, then 5% writes) against it while a PING probe
watches for server freezes, and compares the two sides per query type. It always writes
`<out>/result.json`, also when it could not measure (`verdict: error`).

The tools are pinned by commit and set up by the check itself (README: "Tools"): the benchmark
is built from source, the bulk loader runs from its own uv.lock, the dataset is checked by
sha256. Standard library only; run it with `python3 -I`.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

CHECK = "pokec"

# --- pinned tools and data ---------------------------------------------------

BENCHMARK_REPO = "https://github.com/FalkorDB/benchmark.git"
BENCHMARK_COMMIT = "c366437bf462070d9da96b75634c975d57b01707"
LOADER_REPO = "https://github.com/FalkorDB/falkordb-bulk-loader.git"
LOADER_COMMIT = "d4e6718ebdc26b5dec4169d7dbb5a66f211e2701"
RUST_TOOLCHAIN = "1.100.0"
RUSTUP_URL = "https://sh.rustup.rs"
# Debian packages the benchmark's build needs on a bare GCE image: a C toolchain
# (ring, aws-lc-sys), protoc (prometheus' build script) and cmake.
BUILD_PACKAGES = ("build-essential", "pkg-config", "protobuf-compiler", "cmake")

DATA_URL = "https://s3.eu-west-1.amazonaws.com/deps.memgraph.io/dataset/pokec/benchmark/"


@dataclass(frozen=True)
class Dataset:
    name: str
    file: str
    sha256: str
    nodes: int
    edges: int
    # A run longer than this, with nothing to compare it with yet, did not finish.
    first_deadline_s: int
    # The least time any run is given.
    deadline_floor_s: int
    load_timeout_s: int


DATASETS = {
    "medium": Dataset(
        "medium",
        "pokec_medium_import.cypher",
        "3614ffa9fc03605a12b9838716b6b1f5ee5be96cfc0d2ab3744f4a74d402c664",
        100_000,
        1_768_515,
        first_deadline_s=900,
        deadline_floor_s=180,
        load_timeout_s=900,
    ),
    "large": Dataset(
        "large",
        "pokec_large.setup.cypher.gz",
        "9b6ebec0b40eefed7e4d86a8316758e7eb522ec63ac6b814294093674423e8d3",
        1_632_803,
        30_622_564,
        first_deadline_s=3600,
        deadline_floor_s=900,
        load_timeout_s=3600,
    ),
}

# --- the workload ------------------------------------------------------------

GRAPH = "falkor"  # the graph name FalkorDB/benchmark queries
QUERY_COUNT = 10_000
PARALLEL = 8
MPS = 2000
THREAD_COUNT = 10
# Per-query timeout the benchmark sends with every query (its default is 180 s).
# A query that stops finishing then ends as a counted timeout instead of holding
# the run (and, on 6.0.x, the whole server) for minutes. README: "Timeouts".
QUERY_TIMEOUT_MS = 15000


@dataclass(frozen=True)
class Workload:
    key: str
    label: str
    write_ratio: float


WORKLOADS = (
    Workload("read", "read-only", 0.0),
    Workload("write5", "5% writes", 0.05),
)

# The four whole-graph algorithm queries are left out of the workload (README: "Workload").
# They run for seconds, do not stop at the per-query timeout, and on the server's four
# cores turned a 10k-query run into ten minutes of contention (main vs itself, read-only:
# 589 s, 354 requests abandoned at the client's guard, harmonic centrality p50 4.1 s).
# 6.0.x and C v4.22 lack some of them anyway.
ALGORITHM_FLAGS = ("--enable-algo-pagerank", "--enable-algo-max-flow", "--enable-algo-msf", "--enable-algo-harmonic")

# --- thresholds --------------------------------------------------------------

RATIO_THRESHOLD = 1.3
# A latency row also has to move by at least this much (ms): below it a ratio is
# noise. Set from four A/A runs (main@4c1088949 and C v4.22.4, dev VM and the
# _perf-vm.yml acceptance run 45389482775; README: "Thresholds measured"). Over every
# 3-vs-3 split of a row's six first-pass readings, and every 2-vs-2 split for the
# re-run, floors of 3 / 10 ms gave 0.02-0.04 expected false regressions a night (the
# CI A/As: aggregate_expansion_4_with_filter p99 +14.7 ms at 1.47x on main,
# indexed_in_list_predicate p95 +4.0 ms on C); 5 / 15 ms gave none in all four.
LATENCY_FLOOR_MS = {"p50_ms": 1.0, "p95_ms": 5.0, "p99_ms": 15.0}
PING_LIMIT_MS = 1000.0
PING_INTERVAL_S = 0.2
PING_TIMEOUT_S = 30.0
PING_SLICE_S = 0.25  # the probe looks for a stop this often while it waits for a reply
# A run that takes this many times as long as the other side's slowest finished
# run of the same workload did not finish (README: "Did not finish").
DEADLINE_FACTOR = 3.0
# Timeouts of one query, or a PING freeze, count once they happen in this many
# separate runs of a side (each run is a fresh server and load).
REPEAT_RUNS = 2
PERCENTILES = ("p50_ms", "p95_ms", "p99_ms")


@dataclass(frozen=True)
class Mode:
    datasets: tuple[str, ...]
    reps: int
    rerun_reps: int
    # queue: latency rows are reported, never fail the run.
    latency_gates: bool


MODES = {
    "queue": Mode(("medium",), reps=1, rerun_reps=1, latency_gates=False),
    "nightly": Mode(("medium",), reps=3, rerun_reps=2, latency_gates=True),
    "long": Mode(("medium", "large"), reps=3, rerun_reps=2, latency_gates=True),
}

DIGEST_RE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class SetupError(Exception):
    """The check could not set up its tools or data: the result is an error."""


class Stopped(BaseException):
    """SIGTERM (the workflow's timeout) arrived. Not an Exception: no broad handler may eat it."""


# --- small helpers -----------------------------------------------------------


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_label(image: str) -> str:
    """`<repo>@<first 12 hex of the digest>`, the contract's label when none is given."""
    name, _, digest = image.partition("@sha256:")
    if name.rfind(":") > name.rfind("/"):
        name = name[: name.rfind(":")]
    return f"{name}@{digest[:12]}"


def median(values: Iterable[float]) -> float | None:
    data = [v for v in values if v is not None]
    return statistics.median(data) if data else None


def ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return round(candidate / baseline, 4)


def rounded(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


class Log:
    """Everything the check does, with timestamps, to stdout and raw/check.log."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a", encoding="utf-8")

    def __call__(self, message: str) -> None:
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        self.file.write(line + "\n")
        self.file.flush()


def run_cmd(
    cmd: list[str],
    log: Log,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    output: Path | None = None,
) -> subprocess.CompletedProcess:
    """Run a setup command; raise SetupError when it fails."""
    log("$ " + " ".join(cmd))
    try:
        if output is not None:
            with output.open("ab") as sink:
                proc = subprocess.run(
                    cmd, cwd=cwd, env=env, timeout=timeout, stdout=sink, stderr=subprocess.STDOUT
                )
        else:
            proc = subprocess.run(
                cmd, cwd=cwd, env=env, timeout=timeout, capture_output=True, text=True
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SetupError(f"{cmd[0]} failed: {exc}") from exc
    if proc.returncode != 0:
        tail = "" if output is not None else (proc.stderr or proc.stdout or "")[-2000:]
        raise SetupError(f"`{' '.join(cmd[:4])} …` exited {proc.returncode}: {tail.strip()}")
    return proc


# --- RESP: one request, one reply, no retries ---------------------------------


class RespError(Exception):
    """An error reply from the server."""


class Resp:
    """A minimal RESP2 client. It never re-sends: a timeout is the caller's to report."""

    def __init__(self, port: int, timeout: float) -> None:
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self.sock.settimeout(timeout)
        self.buf = b""

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass  # best effort: the socket is being dropped either way

    def shutdown(self) -> None:
        """Close from another thread: shutdown() wakes a recv() blocked on this socket."""
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # already closed or never connected
        self.close()

    def __enter__(self) -> "Resp":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def send(self, *words: str | int) -> None:
        payload = [f"*{len(words)}\r\n".encode()]
        for word in words:
            data = str(word).encode()
            payload.append(b"$%d\r\n%s\r\n" % (len(data), data))
        self.sock.sendall(b"".join(payload))

    def call(self, *words: str | int) -> Any:
        self.send(*words)
        return self.read()

    def _fill(self) -> None:
        chunk = self.sock.recv(1 << 16)
        if not chunk:
            raise ConnectionError("server closed the connection")
        self.buf += chunk

    def _line(self) -> bytes:
        while b"\r\n" not in self.buf:
            self._fill()
        line, _, self.buf = self.buf.partition(b"\r\n")
        return line

    def _exact(self, size: int) -> bytes:
        while len(self.buf) < size + 2:
            self._fill()
        data, self.buf = self.buf[:size], self.buf[size + 2 :]
        return data

    def read(self) -> Any:
        line = self._line()
        kind, rest = line[:1], line[1:]
        if kind == b"+":
            return rest.decode()
        if kind == b"-":
            return RespError(rest.decode(errors="replace"))
        if kind == b":":
            return int(rest)
        if kind == b"$":
            size = int(rest)
            return None if size < 0 else self._exact(size).decode(errors="replace")
        if kind == b"*":
            size = int(rest)
            return None if size < 0 else [self.read() for _ in range(size)]
        raise ConnectionError(f"unexpected RESP reply {line[:40]!r}")


def resp_call(port: int, *words: str | int, timeout: float = 30.0) -> Any:
    with Resp(port, timeout) as conn:
        reply = conn.call(*words)
    if isinstance(reply, RespError):
        raise reply
    return reply


def query_rows(port: int, query: str, *, graph: str = GRAPH, read_only: bool = True,
               timeout: float = 120.0) -> list[list[Any]]:
    """The rows of a (non-compact) GRAPH.QUERY reply. Scalars only: a list comes back as text."""
    command = "GRAPH.RO_QUERY" if read_only else "GRAPH.QUERY"
    reply = resp_call(port, command, graph, query, timeout=timeout)
    if not isinstance(reply, list) or len(reply) < 2 or not isinstance(reply[1], list):
        raise RespError(f"unexpected reply to {query!r}: {reply!r}"[:300])
    return reply[1]


def query_scalar(port: int, query: str, *, timeout: float = 120.0) -> Any:
    """The first value of the first row."""
    rows = query_rows(port, query, timeout=timeout)
    if not rows or not isinstance(rows[0], list) or not rows[0]:
        raise RespError(f"no rows from {query!r}: {rows!r}"[:300])
    return rows[0][0]


# --- the PING probe ----------------------------------------------------------


class PingProbe:
    """PING every PING_INTERVAL_S on its own connection, each timed from its own send.

    A reply that has not come after PING_TIMEOUT_S is recorded as a timeout and the
    connection replaced; a PING still waiting when the probe stops is recorded with the
    time it has waited so far (a freeze that outlasted the run). Each PING is recorded
    once: stop() and the probe thread claim the sample under one lock.
    """

    def __init__(self, port: int, path: Path) -> None:
        self.port = port
        self.path = path
        self.samples: list[tuple[int, float | None, str]] = []  # epoch ms, ms, kind
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._stopped = False
        self._inflight: tuple[float, int] | None = None  # (monotonic send time, epoch ms)
        self._conn: Resp | None = None
        self._thread = threading.Thread(target=self._loop, name="ping", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _record(self, sample: tuple[int, float | None, str]) -> bool:
        """Append unless stop() already did (it records the PING in flight itself)."""
        with self._lock:
            self._inflight = None
            if self._stopped:
                return False
            self.samples.append(sample)
            return True

    def _reply(self, conn: Resp, sent: float) -> Any:
        """Wait for the reply in short slices, so a stop() is seen within PING_SLICE_S."""
        while True:
            try:
                return conn.read()
            except socket.timeout:
                if self._stop.is_set():
                    return None  # stop() has recorded this PING
                if time.monotonic() - sent >= PING_TIMEOUT_S:
                    raise

    def _loop(self) -> None:
        conn: Resp | None = None
        while not self._stop.is_set():
            sent = time.monotonic()
            epoch = int(time.time() * 1000)
            try:
                if conn is None:
                    conn = Resp(self.port, PING_TIMEOUT_S)
                    conn.sock.settimeout(PING_SLICE_S)
                    self._conn = conn
                with self._lock:
                    if self._stopped:
                        break
                    self._inflight = (sent, epoch)
                conn.send("PING")
                reply = self._reply(conn, sent)
                ms = (time.monotonic() - sent) * 1000
                if not self._record((epoch, ms, "ok" if reply == "PONG" else "error")):
                    break
            except socket.timeout:
                self._record((epoch, PING_TIMEOUT_S * 1000, "timeout"))
                conn.close() if conn else None
                conn = None
            except OSError:
                self._record((epoch, None, "error"))
                conn.close() if conn else None
                conn = None
            wait = PING_INTERVAL_S - (time.monotonic() - sent)
            if wait > 0:
                self._stop.wait(wait)
        if conn is not None:
            conn.close()

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            inflight, self._inflight = self._inflight, None
            if inflight is not None:
                # Still waiting for a reply: the freeze lasted at least this long.
                sent, epoch = inflight
                self.samples.append((epoch, (time.monotonic() - sent) * 1000, "inflight"))
        self._stop.set()
        conn = self._conn
        if conn is not None:
            conn.shutdown()  # a probe blocked in recv() returns now
        self._thread.join(timeout=5.0)
        try:
            with self.path.open("w", encoding="utf-8") as sink:
                sink.write("epoch_ms,latency_ms,kind\n")
                for epoch, ms, kind in list(self.samples):
                    sink.write(f"{epoch},{'' if ms is None else round(ms, 3)},{kind}\n")
        except OSError as exc:
            # The samples (and summary()) stand without the file; an error here must not
            # replace a stop signal that is unwinding through the caller.
            print(f"ping probe: could not write {self.path}: {exc}", file=sys.stderr)

    def summary(self) -> dict[str, Any]:
        samples = list(self.samples)
        timed = [ms for _, ms, kind in samples if ms is not None]
        return {
            "ping_max_ms": rounded(max(timed)) if timed else None,
            "ping_samples": len(samples),
            "ping_over_100ms": sum(1 for ms in timed if ms > 100),
            "ping_over_1s": sum(1 for ms in timed if ms > 1000),
            "ping_timeouts": sum(1 for _, _, kind in samples if kind in ("timeout", "inflight")),
            "ping_errors": sum(1 for _, _, kind in samples if kind == "error"),
        }


# --- docker ------------------------------------------------------------------


def free_port() -> int:
    while True:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port != 6379:
            return port


def image_env(image: str, log: Log) -> dict[str, str]:
    proc = run_cmd(["docker", "image", "inspect", "--format", "{{json .Config.Env}}", image], log)
    env = {}
    for item in json.loads(proc.stdout.strip() or "[]") or []:
        key, _, value = item.partition("=")
        env[key] = value
    return env


@dataclass
class Server:
    name: str
    image: str
    port: int


class Docker:
    """Starts and removes this check's server containers (all labelled with run_id)."""

    def __init__(self, log: Log, run_id: str, server_cpus: str | None) -> None:
        self.log = log
        self.run_id = run_id
        self.server_cpus = server_cpus
        self.count = 0
        self.args_cache: dict[str, str] = {}

    def server_args(self, image: str) -> str:
        """The image's own FALKORDB_ARGS plus THREAD_COUNT: same thread count on every side."""
        if image not in self.args_cache:
            own = image_env(image, self.log).get("FALKORDB_ARGS", "")
            own = re.sub(r"\bTHREAD_COUNT\s+\S+", "", own).strip()
            self.args_cache[image] = f"{own} THREAD_COUNT {THREAD_COUNT}".strip()
        return self.args_cache[image]

    def start(self, image: str, tag: str) -> Server:
        self.count += 1
        name = f"perf-pokec-{self.run_id}-{self.count}-{tag}"
        port = free_port()
        cmd = ["docker", "run", "-d", "--name", name, "--label", f"perf-pokec={self.run_id}"]
        if self.server_cpus:
            cmd += ["--cpuset-cpus", self.server_cpus]
        cmd += ["-p", f"127.0.0.1:{port}:6379", "-e", f"FALKORDB_ARGS={self.server_args(image)}"]
        cmd.append(image)
        run_cmd(cmd, self.log, timeout=300)
        return Server(name, image, port)

    def running(self, server: Server) -> tuple[bool, str]:
        try:
            proc = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}} {{.State.ExitCode}}", server.name],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            return True, "docker inspect did not answer"
        state = proc.stdout.strip()
        return state.startswith("true"), state

    def logs(self, server: Server, path: Path) -> None:
        with path.open("wb") as sink:
            subprocess.run(
                ["docker", "logs", server.name], stdout=sink, stderr=subprocess.STDOUT, timeout=60
            )

    def remove(self, server: Server) -> None:
        try:
            subprocess.run(["docker", "rm", "-f", server.name], capture_output=True, timeout=120)
        except subprocess.TimeoutExpired:
            pass  # remove_all() tries again at the end

    def remove_all(self) -> None:
        proc = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"label=perf-pokec={self.run_id}"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        ids = proc.stdout.split()
        if ids:
            subprocess.run(["docker", "rm", "-f", *ids], capture_output=True, timeout=300)


def wait_ping(port: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if resp_call(port, "PING", timeout=5) == "PONG":
                return True
        except (OSError, RespError):
            pass  # not up yet (refused, reset or LOADING): try again until the deadline
        time.sleep(0.5)
    return False


# --- tools -------------------------------------------------------------------


@dataclass
class Tools:
    benchmark: Path
    loader_python: Path
    datasets: dict[str, tuple[Path, Path]] = field(default_factory=dict)  # name -> (users, friends)


def privileged() -> list[str] | None:
    """The prefix that runs apt-get as root here, or None when nothing can."""
    if os.name != "posix":
        return None
    if os.geteuid() == 0:
        return []
    if shutil.which("sudo") and subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0:
        return ["sudo", "-n"]
    return None


def ensure_build_packages(log: Log, setup_log: Path) -> None:
    missing = [tool for tool in ("cc", "protoc", "cmake", "pkg-config") if not shutil.which(tool)]
    if not missing:
        return
    prefix = privileged()
    if prefix is None or not shutil.which("apt-get"):
        raise SetupError(
            f"building FalkorDB/benchmark needs {', '.join(missing)}, and this machine cannot "
            f"install {' '.join(BUILD_PACKAGES)} (no apt-get as root)"
        )
    env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
    apt = prefix + ["apt-get", "-q", "-o", "DPkg::Lock::Timeout=300"]
    run_cmd(apt + ["update"], log, env=env, timeout=600, output=setup_log)
    run_cmd(
        apt + ["install", "-y", "--no-install-recommends", *BUILD_PACKAGES],
        log,
        env=env,
        timeout=900,
        output=setup_log,
    )


def ensure_rust(tools_dir: Path, log: Log, setup_log: Path) -> Path:
    """rustup with RUST_TOOLCHAIN installed; returns the rustup binary."""
    rustup = shutil.which("rustup") or str(Path.home() / ".cargo" / "bin" / "rustup")
    if not Path(rustup).exists():
        script = tools_dir / "rustup-init.sh"
        with urllib.request.urlopen(RUSTUP_URL, timeout=60) as response:
            script.write_bytes(response.read())
        run_cmd(
            ["sh", str(script), "-y", "--profile", "minimal", "--default-toolchain", "none", "--no-modify-path"],
            log,
            timeout=600,
            output=setup_log,
        )
        rustup = str(Path.home() / ".cargo" / "bin" / "rustup")
    run_cmd(
        [rustup, "toolchain", "install", RUST_TOOLCHAIN, "--profile", "minimal"],
        log,
        timeout=900,
        output=setup_log,
    )
    return Path(rustup)


def fetch_commit(repo: str, commit: str, dest: Path, log: Log, setup_log: Path) -> None:
    head = dest / ".git" / "HEAD"
    if dest.exists():
        proc = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip() == commit:
            return
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    run_cmd(["git", "init", "-q", str(dest)], log, output=setup_log)
    run_cmd(["git", "-C", str(dest), "fetch", "-q", "--depth", "1", repo, commit], log, timeout=600, output=setup_log)
    run_cmd(["git", "-C", str(dest), "checkout", "-q", "FETCH_HEAD"], log, output=setup_log)
    proc = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True)
    if proc.stdout.strip() != commit or not head.exists():
        raise SetupError(f"{repo} checked out {proc.stdout.strip()!r}, not {commit}")


def build_benchmark(tools_dir: Path, log: Log, setup_log: Path) -> Path:
    target = tools_dir / f"benchmark-target-{RUST_TOOLCHAIN}"
    binary = target / "release" / "benchmark"
    stamp = target / "built-from"
    if binary.exists() and stamp.exists() and stamp.read_text().strip() == BENCHMARK_COMMIT:
        log(f"benchmark {BENCHMARK_COMMIT[:7]} already built: {binary}")
        return binary
    source = tools_dir / "benchmark"
    fetch_commit(BENCHMARK_REPO, BENCHMARK_COMMIT, source, log, setup_log)
    ensure_build_packages(log, setup_log)
    rustup = ensure_rust(tools_dir, log, setup_log)
    started = time.monotonic()
    run_cmd(
        [str(rustup), "run", RUST_TOOLCHAIN, "cargo", "build", "--release", "--locked"],
        log,
        cwd=source,
        env=dict(os.environ, CARGO_TARGET_DIR=str(target)),
        timeout=3600,
        output=setup_log,
    )
    log(f"built benchmark {BENCHMARK_COMMIT[:7]} with Rust {RUST_TOOLCHAIN} in {time.monotonic() - started:.0f} s")
    if not binary.exists():
        raise SetupError(f"cargo build left no {binary}")
    stamp.write_text(BENCHMARK_COMMIT + "\n")
    return binary


def setup_loader(tools_dir: Path, log: Log, setup_log: Path) -> Path:
    source = tools_dir / "falkordb-bulk-loader"
    fetch_commit(LOADER_REPO, LOADER_COMMIT, source, log, setup_log)
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    if not Path(uv).exists():
        raise SetupError("uv not found: the bulk loader runs from its own uv.lock (uv sync --frozen)")
    run_cmd([uv, "sync", "--frozen", "--no-dev", "--project", str(source)], log, timeout=900, output=setup_log)
    python = source / ".venv" / "bin" / "python"
    if not python.exists():
        raise SetupError(f"uv sync left no {python}")
    return python


USER_RE = re.compile(r"^CREATE \(:User \{(.*)\}\);$")
EDGE_RE = re.compile(r"^MATCH \(n:User \{id: (\d+)\}\), \(m:User \{id: (\d+)\}\) CREATE \(n\)-\[e: Friend\]->\(m\);$")


def convert_dataset(source: Path, users: Path, friends: Path) -> tuple[int, int]:
    """The Cypher dump to the bulk loader's CSVs, as the Nov 23 run did (vm/setup.sh's perl).

    Friend edges carry `bench_capacity` = 1 + (31 src + 17 dst) mod 20, the property the
    benchmark otherwise adds with a write over every edge before it starts.
    """
    opener = gzip.open if source.suffix == ".gz" else open
    node_count = edge_count = 0
    with opener(source, "rt", encoding="utf-8") as lines, users.open("w", encoding="utf-8") as u, friends.open(
        "w", encoding="utf-8"
    ) as f:
        u.write("id,completion_percentage,gender,age\n")
        f.write("src_id,dst_id,bench_capacity\n")
        for raw in lines:
            line = raw.rstrip("\n")
            match = USER_RE.match(line)
            if match:
                props = {}
                for pair in re.split(r",\s*", match.group(1)):
                    key, _, value = pair.partition(":")
                    value = value.lstrip()
                    if value.startswith('"'):
                        value = value[1:]
                    if value.endswith('"'):
                        value = value[:-1]
                    props[key.strip()] = value
                if "id" in props:
                    u.write(
                        f"{props['id']},{props.get('completion_percentage', '')},"
                        f"{props.get('gender', '')},{props.get('age', '')}\n"
                    )
                    node_count += 1
                continue
            match = EDGE_RE.match(line)
            if match:
                src, dst = int(match.group(1)), int(match.group(2))
                f.write(f"{src},{dst},{1 + (src * 31 + dst * 17) % 20}\n")
                edge_count += 1
    return node_count, edge_count


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as data:
        for block in iter(lambda: data.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_dataset(dataset: Dataset, tools_dir: Path, log: Log) -> tuple[Path, Path]:
    folder = tools_dir / "datasets" / dataset.name
    folder.mkdir(parents=True, exist_ok=True)
    users, friends, stamp = folder / "User.csv", folder / "FRIEND.csv", folder / "converted-from"
    if users.exists() and friends.exists() and stamp.exists() and stamp.read_text().strip() == dataset.sha256:
        return users, friends
    source = folder / dataset.file
    if not source.exists() or sha256_of(source) != dataset.sha256:
        log(f"downloading {DATA_URL}{dataset.file}")
        partial = source.with_suffix(source.suffix + ".part")
        with urllib.request.urlopen(DATA_URL + dataset.file, timeout=120) as response, partial.open("wb") as sink:
            shutil.copyfileobj(response, sink, 1 << 20)
        got = sha256_of(partial)
        if got != dataset.sha256:
            raise SetupError(f"{dataset.file}: sha256 {got}, expected {dataset.sha256}")
        partial.rename(source)
    nodes, edges = convert_dataset(source, users, friends)
    if (nodes, edges) != (dataset.nodes, dataset.edges):
        raise SetupError(f"{dataset.file} converted to {nodes} nodes / {edges} edges, expected {dataset.nodes} / {dataset.edges}")
    stamp.write_text(dataset.sha256 + "\n")
    log(f"dataset {dataset.name}: {nodes} users, {edges} Friend edges")
    return users, friends


# --- queries -----------------------------------------------------------------


@dataclass
class QueryFile:
    path: Path
    counts: Counter  # q_name -> requests
    types: dict[str, str]  # q_name -> Read | Write
    # The exact cypher the benchmark sends -> q_name. Two query types can send the same text
    # (benchmark c366437: vertex_on_label_property and vertex_on_label_property_index share
    # a template, so the same random id collides); such a text maps to "a|b" and its
    # failures are split per type from metrics.prom (attribute_ambiguous).
    by_text: dict[str, str]


def read_query_file(path: Path) -> QueryFile:
    counts: Counter = Counter()
    types: dict[str, str] = {}
    names_by_text: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as lines:
        meta = json.loads(lines.readline())
        for entry in meta.get("catalog", []):
            types[entry["name"]] = entry["q_type"]
        for line in lines:
            if not line.strip():
                continue
            item = json.loads(line)
            counts[item["q_name"]] += 1
            types[item["q_name"]] = item["q_type"]
            names_by_text[item["cypher"]].add(item["q_name"])
    by_text = {text: "|".join(sorted(names)) for text, names in names_by_text.items()}
    return QueryFile(path, counts, types, by_text)


def attribute_ambiguous(
    errors: dict[str, dict[str, int]], counts: dict[str, int], rates: dict[str, dict[str, float]]
) -> tuple[dict[str, dict[str, int]], str | None]:
    """Split failures logged under a shared text ("a|b") between its query types.

    metrics.prom's per-type rates say how many requests of each type failed; what the
    unambiguous log lines do not account for belongs to the shared text. The split is
    exact per type in totals; when a shared text has both timeouts and other errors, which
    type got which kind is not known (the tool's own rates call its guard's `Elapsed` a
    failure), and timeouts are handed out first, in name order. Returns the errors and a
    problem when they disagree.
    """
    out = {name: dict(kinds) for name, kinds in errors.items() if "|" not in name}
    for key, kinds in errors.items():
        if "|" not in key:
            continue
        names = key.split("|")
        residual = {}
        for name in names:
            rate = rates.get(name, {})
            expected = round((rate.get("timeout", 0.0) + rate.get("failure", 0.0)) * counts.get(name, 0) / 100)
            residual[name] = expected - sum(out.get(name, {}).values())
        if any(v < 0 for v in residual.values()) or sum(residual.values()) != sum(kinds.values()):
            return out, f"{sum(kinds.values())} failures of the shared text {key} do not match metrics.prom ({residual})"
        left = {"timeouts": kinds.get("timeouts", 0), "failures": kinds.get("failures", 0)}
        for name in names:
            need = residual[name]
            if not need:
                continue
            slot = out.setdefault(name, {"timeouts": 0, "failures": 0})
            for kind in ("timeouts", "failures"):
                take = min(need, left[kind])
                slot[kind] = slot.get(kind, 0) + take
                left[kind] -= take
                need -= take
    return out, None


def hop_label(query: str) -> str:
    """A readable name: `aggregate_expansion_3_with_filter` reads `3-hop, age >= 18`."""
    match = re.fullmatch(r"aggregate_expansion_(\d)(_with_filter)?", query)
    if not match:
        return query
    hops = f"{match.group(1)}-hop expansion"
    return f"{hops}, age >= 18 ({query})" if match.group(2) else f"{hops} ({query})"


# --- the benchmark's output --------------------------------------------------

PROM_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)$")
LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse_prom(text: str) -> list[tuple[str, dict[str, str], float]]:
    samples = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = PROM_RE.match(line.strip())
        if not match:
            continue
        labels = dict(LABEL_RE.findall(match.group(2) or ""))
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        samples.append((match.group(1), labels, value))
    return samples


def parse_metrics(text: str) -> dict[str, Any]:
    """Per-query and global percentiles (ms) and per-query timeout/failure rates (%)."""
    per_query: dict[str, dict[str, float]] = defaultdict(dict)
    overall: dict[str, float] = {}
    rates: dict[str, dict[str, float]] = defaultdict(dict)
    global_rates: dict[str, float] = {}
    executed: Counter = Counter()  # requests a worker finished, per query type (any outcome)
    for name, labels, value in parse_prom(text):
        if name == "falkordb_query_latency_pct_us" and "query" in labels and "pct" in labels:
            if labels["pct"] in ("50", "95", "99"):
                per_query[labels["query"]][f"p{labels['pct']}_ms"] = value / 1000
        elif (match := re.fullmatch(r"falkordb_latency_(p\d+)_us", name)) and match.group(1) in ("p50", "p95", "p99"):
            overall[f"{match.group(1)}_ms"] = value / 1000
        elif name in ("benchmark_query_timeout_rate_pct", "benchmark_query_failure_rate_pct"):
            if labels.get("vendor", "falkor").lower().startswith("falkor") and "query" in labels:
                rates[labels["query"]]["timeout" if "timeout" in name else "failure"] = value
        elif name in ("benchmark_global_timeout_rate_pct", "benchmark_global_failure_rate_pct"):
            if labels.get("vendor", "falkor").lower().startswith("falkor"):
                global_rates["timeout" if "timeout" in name else "failure"] = value
        elif name == "operations_total" and labels.get("vendor", "falkor").lower().startswith("falkor"):
            # lib.rs OPERATION_COUNTER, one series per worker: counted once a request ends.
            executed[labels.get("name", "")] += int(value)
    return {"per_query": dict(per_query), "overall": overall, "rates": dict(rates), "global_rates": global_rates,
            "executed": dict(executed)}


def missing_requests(executed: dict[str, int], counts: dict[str, int]) -> str | None:
    """None when every generated request ended (with any outcome), else what is missing.

    The benchmark ignores its workers' join errors (main.rs run_falkor), so a worker that
    dies after taking a request can leave the run at exit 0 with the request unaccounted.
    """
    short = {q: (executed.get(q, 0), n) for q, n in counts.items() if executed.get(q, 0) != n}
    if not short:
        return None
    sample = ", ".join(f"{q} {got} of {want}" for q, (got, want) in sorted(short.items())[:3])
    return f"the benchmark ended {sum(executed.get(q, 0) for q in counts)} of {sum(counts.values())} requests ({sample})"


ERROR_RE = re.compile(r"Error executing query: (?P<query>.*?), the error is: (?P<error>.*)$")


def parse_errors(log_text: str, by_text: dict[str, str]) -> tuple[dict[str, dict[str, int]], list[str]]:
    """Per query, the requests that ended in a timeout or another error, from the run's log.

    The benchmark logs every failed request (`error!`, falkor_driver.rs read_reply) with the
    exact cypher it sent; it calls a failure a timeout when its message says "timeout"
    (main.rs is_timeout_error). Unlike metrics.prom, the log exists for a run that was
    stopped at its deadline too.
    """
    errors: dict[str, dict[str, int]] = defaultdict(lambda: {"timeouts": 0, "failures": 0})
    unmatched: list[str] = []
    for line in ANSI_RE.sub("", log_text).splitlines():
        if " ERROR " not in f" {line} " or "Error executing query: " not in line:
            continue
        match = ERROR_RE.search(line)
        if not match:
            continue
        name = by_text.get(match.group("query"))
        if name is None:
            unmatched.append(match.group("query")[:200])
            continue
        error = match.group("error")
        # The server says "Query timed out"; the tool's own guard (a tokio timeout at
        # QUERY_TIMEOUT_MS + 5 s) fails with `Elapsed(())`. Both are timeouts here.
        lowered = error.lower()
        timed_out = "timeout" in lowered or "timed out" in lowered or error.startswith("Elapsed")
        kind = "timeouts" if timed_out else "failures"
        errors[name][kind] += 1
    return {k: dict(v) for k, v in errors.items()}, unmatched


# --- one run -----------------------------------------------------------------


@dataclass
class Run:
    dataset: str
    workload: str  # Workload.key
    side: str  # candidate | baseline
    pass_no: int  # 1 = first pass, 2 = re-run
    rep: int
    finished: bool = False
    stop_reason: str | None = None
    wall_s: float | None = None
    elapsed_s: float | None = None  # the benchmark's own query phase
    load_s: float | None = None
    deadline_s: float | None = None
    used_memory_loaded: int | None = None
    ping: dict[str, Any] = field(default_factory=dict)
    per_query: dict[str, dict[str, float]] = field(default_factory=dict)
    overall: dict[str, float] = field(default_factory=dict)
    errors: dict[str, dict[str, int]] = field(default_factory=dict)
    unmatched_errors: int = 0
    dir: str = ""

    @property
    def throughput(self) -> float | None:
        return QUERY_COUNT / self.elapsed_s if self.finished and self.elapsed_s else None


# --- analysis ----------------------------------------------------------------


# The self-test sends a read that would run for hours with this per-query timeout. Measured
# on main, 6.0.4 and C v4.22.4: it stops at 2.50 s (and at 1.00 s with no timeout argument,
# the images' TIMEOUT 1000). A stop outside this window means another limit applied.
TIMEOUT_TEST_MS = 2500
TIMEOUT_TEST_WINDOW_S = (2.2, 4.0)


def timeout_problem(test: dict[str, Any], server_timeout: Any) -> str | None:
    """None when the per-query timeout demonstrably applied, else why it did not."""
    low, high = TIMEOUT_TEST_WINDOW_S
    if not test.get("timed_out"):
        return f"a read sent with timeout {TIMEOUT_TEST_MS} ms ended after {test.get('elapsed_s')} s ({test.get('reply')})"
    if not low <= test["elapsed_s"] <= high:
        return (f"a read sent with timeout {TIMEOUT_TEST_MS} ms timed out after {test['elapsed_s']} s, "
                f"outside {low}-{high} s")
    if isinstance(server_timeout, int) and low * 1000 <= server_timeout <= high * 1000:
        return f"the server's own TIMEOUT {server_timeout} ms falls inside the test's window, so the test cannot tell the two apart"
    return None


def make_finding(
    name: str,
    metric: str,
    candidate: float | None,
    baseline: float | None,
    threshold: float | None,
    direction: str,
    status: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "metric": metric,
        "candidate": candidate,
        "baseline": baseline,
        "ratio": ratio(candidate, baseline),
        "threshold": threshold,
        "direction": direction,
        "status": status,
    }


def repeated_level(values: list[float | None]) -> float | None:
    """The highest value reached in at least REPEAT_RUNS of the runs (None: too few runs)."""
    data = sorted(v for v in values if v is not None)
    return data[-REPEAT_RUNS] if len(data) >= REPEAT_RUNS else None


def completed_status(candidate: int, baseline: int) -> str:
    """Contract v1.4: 0 on the candidate only is a regression at once; on both, info."""
    if candidate == baseline:
        return "ok" if candidate else "info"
    return "improvement" if candidate else "regression"


def scope(dataset: str, workload: Workload) -> str:
    return workload.label if dataset == "medium" else f"{dataset}, {workload.label}"


def crossed(metric: str, candidate: float | None, baseline: float | None, direction: str) -> int:
    """+1 when the candidate is worse past the threshold (and floor), -1 better, else 0."""
    if candidate is None or baseline is None or baseline <= 0:
        return 0
    floor = LATENCY_FLOOR_MS.get(metric, 0.0)
    if direction == "lower_is_worse":
        candidate, baseline = baseline, candidate
    if candidate > RATIO_THRESHOLD * baseline and candidate - baseline > floor:
        return 1
    if baseline > RATIO_THRESHOLD * candidate and baseline - candidate > floor:
        return -1
    return 0


@dataclass
class Analysis:
    findings: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    rerun: set[tuple[str, str]]  # (dataset, workload) that need a re-run
    unmeasured: list[str]
    notes: list[str]
    rerun_why: dict[tuple[str, str], list[str]] = field(default_factory=dict)


def analyse(
    runs: list[Run],
    mode: Mode,
    query_files: dict[tuple[str, str], QueryFile],
    info: list[dict[str, Any]],
    rerun_done: set[tuple[str, str]],
) -> Analysis:
    """Every comparison the check makes, from the runs alone (pure; tested)."""
    findings: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    rerun: set[tuple[str, str]] = set()
    rerun_why: dict[tuple[str, str], list[str]] = defaultdict(list)
    unmeasured: list[str] = []
    notes: list[str] = []

    def want_rerun(key: tuple[str, str], why: str) -> None:
        rerun.add(key)
        rerun_why[key].append(why)

    groups: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for run in runs:
        groups[(run.dataset, run.workload)].append(run)

    for (dataset, wl_key), group in groups.items():
        workload = next(w for w in WORKLOADS if w.key == wl_key)
        where = scope(dataset, workload)
        side_runs = {side: [r for r in group if r.side == side] for side in ("candidate", "baseline")}
        finished = {side: [r for r in rs if r.finished] for side, rs in side_runs.items()}
        qfile = query_files.get((dataset, wl_key))

        def add_row(name: str, metric: str, cand: float | None, base: float | None, direction: str,
                    status: str, threshold: float | None, **extra: Any) -> dict[str, Any]:
            row = {"dataset": dataset, "workload": workload.label, "name": name, "metric": metric,
                   "candidate": cand, "baseline": base, "ratio": ratio(cand, base),
                   "direction": direction, "status": status, **extra}
            rows.append(row)
            finding = make_finding(name, metric, cand, base, threshold, direction, status)
            finding["_sort"] = extra.get("sort", 0.0)
            findings.append(finding)
            return row

        # Did every run finish?
        if not side_runs["candidate"] or not side_runs["baseline"]:
            continue
        done = {side: int(all(r.finished for r in rs)) for side, rs in side_runs.items()}
        reasons = sorted({r.stop_reason for r in group if r.stop_reason})
        add_row(
            f"benchmark run finished [{where}]",
            "completed",
            done["candidate"],
            done["baseline"],
            "lower_is_worse",
            completed_status(done["candidate"], done["baseline"]),
            None,
            sort=10.0,
            detail="; ".join(f"{r.side} {r.pass_no}.{r.rep}: {r.stop_reason}" for r in group if r.stop_reason),
        )
        if not finished["candidate"] or not finished["baseline"]:
            unmeasured.append(f"{where}: " + ("; ".join(reasons) or "no finished run"))
        elif not done["baseline"]:
            # The candidate is compared with fewer baseline runs than planned.
            unmeasured.append(f"{where}: the baseline did not finish every run (" + "; ".join(reasons) + ")")

        # Per query: did every request return? (from every run, finished or not)
        names = sorted(qfile.counts) if qfile else sorted({q for r in group for q in r.per_query})
        for query in names:
            label = f"{hop_label(query)} [{where}]"
            broken = {}
            counts = {}
            for side, rs in side_runs.items():
                failures = sum(r.errors.get(query, {}).get("failures", 0) for r in rs)
                timeouts = sum(r.errors.get(query, {}).get("timeouts", 0) for r in rs)
                timeout_runs = sum(1 for r in rs if r.errors.get(query, {}).get("timeouts", 0))
                counts[side] = (failures, timeouts, timeout_runs, len(rs))
                # An error reply is a correctness failure: at once. Timeouts are timing: they
                # count once they happen in REPEAT_RUNS separate runs.
                broken[side] = failures > 0 or timeout_runs >= REPEAT_RUNS
            cand_ok, base_ok = int(not broken["candidate"]), int(not broken["baseline"])
            status = completed_status(cand_ok, base_ok)
            detail = "; ".join(
                f"{side}: {f} errors, {t} timeouts in {tr} of {n} runs" for side, (f, t, tr, n) in counts.items()
            )
            if status != "ok":
                add_row(label, "completed", cand_ok, base_ok, "lower_is_worse", status, None,
                        sort=9.0, detail=detail)
            else:
                rows.append({"dataset": dataset, "workload": workload.label, "name": label,
                             "metric": "completed", "candidate": 1, "baseline": 1, "ratio": 1.0,
                             "direction": "lower_is_worse", "status": "ok", "detail": detail})
                # Timeouts in one candidate run: re-run to see whether they repeat. Also when the
                # baseline timed out once too: only a repeat decides which side is broken.
                f, t, tr, n = counts["candidate"]
                if tr == 1:
                    if (dataset, wl_key) not in rerun_done:
                        want_rerun((dataset, wl_key), f"{label}: timeouts in 1 candidate run")
                    else:
                        findings.append({**make_finding(label, "completed", 1, 1, None, "lower_is_worse", "info"),
                                         "_sort": 1.0})
                        notes.append(f"{label}: {t} timeouts in 1 of {n} candidate runs, not repeated")

        # Latency, throughput and PING: from the finished runs.
        if not finished["candidate"] or not finished["baseline"]:
            continue

        def side_value(side: str, getter: Callable[[Run], float | None], pass_no: int | None = None) -> float | None:
            return median(getter(r) for r in finished[side] if pass_no is None or r.pass_no == pass_no)

        def compare(name: str, metric: str, direction: str, getter: Callable[[Run], float | None],
                    use: str) -> None:
            """use: "gate" (can fail the check), "report" (info when it crosses), "row" (rows.csv only)."""
            cand, base = side_value("candidate", getter), side_value("baseline", getter)
            first = crossed(metric, side_value("candidate", getter, 1), side_value("baseline", getter, 1), direction)
            has_second = any(r.pass_no == 2 for r in finished["candidate"]) and any(
                r.pass_no == 2 for r in finished["baseline"])
            # A re-run with no value for this row (e.g. every request of the query failed on
            # one side) did not re-measure it: that stays None, never "did not cross".
            cand2, base2 = side_value("candidate", getter, 2), side_value("baseline", getter, 2)
            second = (crossed(metric, cand2, base2, direction)
                      if has_second and cand2 is not None and base2 is not None else None)
            r = ratio(cand, base)
            severity = 0.0 if r is None else (r if direction == "higher_is_worse" else (1 / r if r else 0.0))
            if use == "row":
                status = "ok"
            elif first and use == "report":
                status = "info"
            elif first == 1:
                if second == 1:
                    status = "regression"
                elif second is None and (dataset, wl_key) not in rerun_done:
                    want_rerun((dataset, wl_key), f"{name} {metric} {r}x")
                    status = "info"
                elif second is None:
                    # Crossed once and never re-measured: neither a pass nor a regression.
                    status = "info"
                    unmeasured.append(f"{name} {metric} crossed {RATIO_THRESHOLD}x in the first pass and was "
                                      f"never re-measured (no finished re-run on both sides, or none was run)")
                else:
                    status = "info"
                    notes.append(f"{name} {metric}: crossed {RATIO_THRESHOLD}x in the first pass, not in the re-run")
            elif first == -1 and second in (None, -1):
                status = "improvement"
            else:
                status = "ok"
            row = {"dataset": dataset, "workload": workload.label, "name": name, "metric": metric,
                   "candidate": rounded(cand), "baseline": rounded(base), "ratio": r, "direction": direction,
                   "status": status, "first_pass": first, "re_run": second, "use": use}
            rows.append(row)
            if status != "ok":
                finding = make_finding(name, metric, rounded(cand), rounded(base), RATIO_THRESHOLD, direction, status)
                # Crossed rows rank by ratio; an info row ranks below the other info lines.
                finding["_sort"] = severity if status != "info" else 0.4 + min(severity, 100.0) / 1000
                findings.append(finding)

        # queue: every latency row is reported, none fails the run. nightly/long: the
        # whole-workload rows gate in both workloads; per query, every row gates read-only,
        # and only a read's p50 gates under writes. There a read's tail is the writer gate
        # (#3816): whether 5% of a query's ~200 requests waited behind a writer decides its
        # p95, so one image against itself read 5.9 / 22.7 / 43.0 ms at p95 across runs. A
        # write's latency is mostly its wait for the write lock: single_vertex_write p50
        # 411-557 ms on one image (README: "Thresholds").
        whole = "gate" if mode.latency_gates else "report"
        for metric in PERCENTILES:
            compare(f"all queries [{where}]", metric, "higher_is_worse",
                    lambda r, m=metric: r.overall.get(m), whole)
        compare(f"throughput [{where}]", "throughput", "lower_is_worse", lambda r: r.throughput, whole)
        for query in names:
            for metric in PERCENTILES:
                is_write = qfile is not None and qfile.types.get(query) == "Write"
                if not mode.latency_gates:
                    use = "report"
                elif workload.write_ratio > 0 and (metric != "p50_ms" or is_write):
                    use = "row"
                else:
                    use = "gate"
                compare(f"{hop_label(query)} [{where}]", metric, "higher_is_worse",
                        lambda r, q=query, m=metric: r.per_query.get(q, {}).get(m), use)

    # PING: every run counts, finished or not (a freeze that ended the run counts most).
    for (dataset, wl_key), group in groups.items():
        workload = next(w for w in WORKLOADS if w.key == wl_key)
        where = scope(dataset, workload)
        values = {side: [r.ping.get("ping_max_ms") for r in group if r.side == side and r.ping] for side in
                  ("candidate", "baseline")}
        # A run that finished with no PING answered was not watched: "no freeze" is unknown.
        blind = sorted({r.side for r in group if r.finished and r.ping.get("ping_max_ms") is None})
        if blind:
            unmeasured.append(f"PING max [{where}]: no PING answered during a finished {' and '.join(blind)} run")
        if not values["candidate"]:
            continue
        # A side's value is the slowest PING reached in at least REPEAT_RUNS runs (the
        # second-highest run maximum): a freeze counts once it happened in two runs.
        cand, base = repeated_level(values["candidate"]), repeated_level(values["baseline"])
        froze = sum(1 for v in values["candidate"] if v is not None and v > PING_LIMIT_MS)
        base_froze = sum(1 for v in values["baseline"] if v is not None and v > PING_LIMIT_MS)
        name = f"PING max [{where}]"
        if froze >= REPEAT_RUNS and base is None:
            status = "info"  # nothing to compare with: fewer than two baseline runs probed
            unmeasured.append(f"{name}: the candidate froze in {froze} runs, the baseline has too few runs to compare")
        elif froze >= REPEAT_RUNS:
            # Frozen as long on the baseline is not new.
            status = "regression" if cand > RATIO_THRESHOLD * base else "info"
        elif froze == 1:
            # One freeze so far: another run decides.
            if (dataset, wl_key) not in rerun_done:
                want_rerun((dataset, wl_key), f"{name}: one candidate run froze")
            status = "info"
        elif base_froze >= REPEAT_RUNS:
            status = "improvement"
        else:
            status = "ok"
        rows.append({"dataset": dataset, "workload": workload.label, "name": name, "metric": "ping_max_ms",
                     "candidate": rounded(cand), "baseline": rounded(base), "ratio": ratio(cand, base),
                     "direction": "higher_is_worse", "status": status,
                     "detail": f"candidate froze (>1 s) in {froze} of {len(values['candidate'])} runs, "
                               f"baseline in {base_froze} of {len(values['baseline'])}"})
        if status != "ok":
            finding = make_finding(name, "ping_max_ms", rounded(cand), rounded(base), PING_LIMIT_MS,
                                   "higher_is_worse", status)
            finding["_sort"] = 8.0 if status == "regression" else 0.5
            findings.append(finding)

    # A workload whose candidate already did not finish a run is a regression; re-running it
    # would only spend more deadlines (6.0.4: up to 272 s a run) to confirm rows beside it.
    for key in list(rerun):
        if any(r.side == "candidate" and not r.finished for r in groups[key]):
            rerun.discard(key)
            notes.append(f"{key}: no re-run, the candidate did not finish a run; "
                         + "; ".join(rerun_why.pop(key, [])) + " stay unconfirmed")
    for item in info:
        findings.append({**item, "_sort": 0.0})
    return Analysis(findings, rows, rerun, unmeasured, notes, {k: v for k, v in rerun_why.items() if k in rerun})


def select_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Crossed rows first (regressions, then improvements), worst first; at most 10 others; 50 in all."""
    order = {"regression": 0, "improvement": 1, "info": 2, "ok": 3}
    ranked = sorted(findings, key=lambda f: (order[f["status"]], -f.get("_sort", 0.0), f["name"], f["metric"]))
    kept: list[dict[str, Any]] = []
    informative = 0
    for finding in ranked:
        if finding["status"] in ("info", "ok"):
            if informative >= 10:
                continue
            informative += 1
        kept.append({k: v for k, v in finding.items() if not k.startswith("_")})
        if len(kept) == 50:
            break
    return kept


def describe(finding: dict[str, Any]) -> str:
    metric, cand, base = finding["metric"], finding["candidate"], finding["baseline"]
    if metric == "completed":
        return f"{finding['name']} did not complete on the {'candidate' if cand == 0 else 'baseline'}"
    unit = " ms" if metric.endswith("_ms") else " q/s" if metric == "throughput" else ""
    r = finding.get("ratio")
    return f"{finding['name']} {metric} {cand}{unit} vs {base}{unit}" + (f" ({r}x)" if r else "")


def verdict_of(selected: list[dict[str, Any]], analysis: Analysis, error: str | None) -> tuple[str, str | None]:
    if any(f["status"] == "regression" for f in selected):
        return "regression", None
    if error:
        return "error", error
    if analysis.unmeasured:
        return "error", "not measured: " + " | ".join(analysis.unmeasured)
    return "pass", None


def summary_of(verdict: str, selected: list[dict[str, Any]], error: str | None, counts: dict[str, int],
               mode_name: str, findings: list[dict[str, Any]] | None = None) -> str:
    findings = findings if findings is not None else selected
    regressions = [f for f in selected if f["status"] == "regression"]
    if verdict == "regression":
        head = "; ".join(describe(f) for f in regressions[:3])
        more = f" (+{len(regressions) - 3} more)" if len(regressions) > 3 else ""
        return " ".join(f"Regression: {head}{more}.".split())
    if verdict == "error":
        return " ".join(f"Could not measure: {error}".split())[:500]
    improvements = sum(1 for f in selected if f["status"] == "improvement")
    both = [f["name"] for f in findings if f["metric"] == "completed" and f["candidate"] == 0 and f["baseline"] == 0]
    text = (
        f"Pass ({mode_name}): {counts['runs']} runs, {counts['rows']} rows compared; no row regressed past "
        f"{RATIO_THRESHOLD}x after a re-run, every run finished, no new PING freeze"
        + (f"; {improvements} improvements" if improvements else "")
        + (f"; {len(both)} queries failed on both sides (not new): {', '.join(both[:3])}" if both else "")
        + "."
    )
    return " ".join(text.split())


# --- the check ---------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    parser.add_argument("--candidate-label")
    parser.add_argument("--baseline-label")
    parser.add_argument("--candidate-commit")
    parser.add_argument("--baseline-commit")
    parser.add_argument(
        "--tools-dir",
        default=os.environ.get("PERF_POKEC_TOOLS") or str(Path.home() / ".cache" / "falkordb-perf" / "pokec"),
        help="where the built benchmark, the loader's venv and the datasets are kept between runs",
    )
    # Development only: shrink a run. A result made with these is not a contract run.
    parser.add_argument("--reps", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--workloads", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    # A development option must still schedule runs: zero runs would read as a pass.
    if args.reps is not None and args.reps < 1:
        parser.error(f"--reps must be at least 1, not {args.reps}")
    if args.workloads is not None:
        keys = [key for key in args.workloads.split(",") if key]
        known = [w.key for w in WORKLOADS]
        if not keys or any(key not in known for key in keys):
            parser.error(f"--workloads takes a comma-separated subset of {', '.join(known)}, not {args.workloads!r}")
    return args


def machine_object() -> dict[str, Any]:
    path = os.environ.get("PERF_MACHINE_JSON")
    if path:
        try:
            return json.loads(Path(path).read_text())
        except (OSError, ValueError) as exc:
            print(f"PERF_MACHINE_JSON {path}: {exc}; describing this machine instead", file=sys.stderr)
    cpu = platform.processor() or platform.machine() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.partition(":")[2].strip()
                break
    except OSError:
        pass  # no /proc (macOS): keep platform.processor()
    return {"runner": "local", "type": platform.machine() or "unknown", "cpu": cpu, "cores": os.cpu_count() or 1}


class Check:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.mode = MODES[args.mode]
        if args.reps:
            self.mode = Mode(self.mode.datasets, args.reps, self.mode.rerun_reps, self.mode.latency_gates)
        self.workloads = [w for w in WORKLOADS if not args.workloads or w.key in args.workloads.split(",")]
        # Absolute: the tools run from other directories (the workflow passes `--out out`).
        self.out = Path(args.out).resolve()
        self.raw = self.out / "raw"
        # Only this invocation's files may become its artifacts: drop an earlier run's.
        stale = self.raw.exists() or (self.out / "result.json").exists()
        cleanup_error = None
        try:
            if self.raw.exists():
                shutil.rmtree(self.raw)
            (self.out / "result.json").unlink(missing_ok=True)
        except OSError as exc:
            cleanup_error = f"could not remove an earlier run's files from {self.out}: {exc}"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.log = Log(self.raw / "check.log")
        if stale and not cleanup_error:
            self.log(f"removed an earlier run's raw/ and result.json from {self.out}")
        self.started = now()
        self.started_mono = time.monotonic()
        self.run_id = f"{os.getpid()}-{int(time.time())}"
        self.docker = Docker(self.log, self.run_id, os.environ.get("PERF_SERVER_CPUS") or None)
        self.images = {"candidate": args.candidate, "baseline": args.baseline}
        self.runs: list[Run] = []
        self.query_files: dict[tuple[str, str], QueryFile] = {}
        self.info: list[dict[str, Any]] = []
        self.servers: dict[str, Any] = {}
        self.rerun_done: set[tuple[str, str]] = set()
        self.error: str | None = None
        self.notes: list[str] = []
        # Set before main(): main() raises it, so the result is an error, not a traceback.
        self.cleanup_error = cleanup_error
        self.stopping = False
        self.machine: dict[str, Any] = machine_object()

    # -- setup --

    def setup(self) -> Tools:
        tools_dir = Path(self.args.tools_dir).resolve()
        tools_dir.mkdir(parents=True, exist_ok=True)
        setup_log = self.raw / "setup.log"
        self.log(f"tools in {tools_dir}")
        benchmark = build_benchmark(tools_dir, self.log, setup_log)
        loader = setup_loader(tools_dir, self.log, setup_log)
        tools = Tools(benchmark, loader)
        for name in self.mode.datasets:
            tools.datasets[name] = setup_dataset(DATASETS[name], tools_dir, self.log)
        for side, image in self.images.items():
            for attempt in range(3):
                proc = subprocess.run(["docker", "pull", "-q", image], capture_output=True, text=True, timeout=900)
                if proc.returncode == 0:
                    break
                self.log(f"docker pull {image} failed (attempt {attempt + 1}): {proc.stderr.strip()[-300:]}")
                time.sleep(10)
            else:
                raise SetupError(f"could not pull the {side} image {image}")
        return tools

    # -- probing an image --

    def probe(self, side: str) -> dict[str, Any] | None:
        """Start the image once, empty: server identity, config, procedures, timeout mechanism."""
        image = self.images[side]
        server = self.docker.start(image, f"probe-{side[:4]}")
        facts: dict[str, Any] = {"image": image, "falkordb_args": self.docker.server_args(image)}
        try:
            if not wait_ping(server.port, 120):
                self.docker.logs(server, self.raw / f"{side}-probe-server.log")
                facts["error"] = "did not answer PING within 120 s of docker run"
                return facts
            info = resp_call(server.port, "INFO", "server")
            fields = dict(line.split(":", 1) for line in info.splitlines() if ":" in line)
            facts["server"] = {k: fields.get(k) for k in ("server_name", "redis_version", "valkey_version",
                                                         "executable", "os", "arch_bits") if fields.get(k)}
            modules = resp_call(server.port, "MODULE", "LIST")
            facts["modules"] = [dict(zip(m[0::2], m[1::2])) for m in modules if isinstance(m, list)]
            for key in ("THREAD_COUNT", "TIMEOUT", "MAX_QUEUED_QUERIES", "RESULTSET_SIZE"):
                try:
                    reply = resp_call(server.port, "GRAPH.CONFIG", "GET", key)
                    facts[key] = reply[1] if isinstance(reply, list) and len(reply) == 2 else reply
                except RespError as exc:
                    facts[key] = f"error: {exc}"
            rows = query_rows(server.port, "CALL dbms.procedures() YIELD name RETURN toLower(name)",
                              graph="probe", read_only=False, timeout=60)
            facts["procedures"] = sorted(row[0] for row in rows)
            facts["timeout_test"] = self.timeout_test(server.port)
            return facts
        finally:
            if not self.stopping:
                self.docker.remove(server)

    def timeout_test(self, port: int) -> dict[str, Any]:
        """Does the per-query timeout the benchmark sends override the server's TIMEOUT?

        Every image ships TIMEOUT 1000 (ms) in FALKORDB_ARGS. A read sent with `timeout 2500`
        must run past 1 s and stop near 2.5 s, or QUERY_TIMEOUT_MS is not what applies.
        """
        # 10^10 rows. The filter needs both variables: with `WHERE x < 0` C v4.22.4 filtered
        # before the second UNWIND and answered in 12 ms.
        query = "UNWIND range(1, 100000) AS x UNWIND range(1, 100000) AS y WITH x + y AS z WHERE z < 0 RETURN count(z)"
        started = time.monotonic()
        answered = False
        try:
            with Resp(port, 60) as conn:
                reply = conn.call("GRAPH.RO_QUERY", "probe", query, "--compact", "timeout", TIMEOUT_TEST_MS)
            answered = True
        except OSError as exc:  # no reply within 60 s: the server did not stop it
            reply = f"no reply: {exc}"
        elapsed = time.monotonic() - started
        return {"elapsed_s": round(elapsed, 3), "reply": str(reply)[:200], "requested_ms": TIMEOUT_TEST_MS,
                "timed_out": answered and isinstance(reply, RespError) and "time" in str(reply).lower()}

    # -- one run --

    def deadline_for(self, dataset: Dataset, workload: Workload, side: str) -> float:
        other = "baseline" if side == "candidate" else "candidate"
        for who in (other, side):
            done = [r.wall_s for r in self.runs if r.dataset == dataset.name and r.workload == workload.key
                    and r.side == who and r.finished and r.wall_s]
            if done:
                return max(dataset.deadline_floor_s, DEADLINE_FACTOR * max(done))
        return float(dataset.first_deadline_s)

    def run_one(self, tools: Tools, dataset: Dataset, workload: Workload, side: str, pass_no: int, rep: int) -> Run:
        run = Run(dataset.name, workload.key, side, pass_no, rep)
        tag = f"{dataset.name}-{workload.key}-{side}-p{pass_no}r{rep}"
        run.dir = f"raw/runs/{tag}"
        folder = self.out / run.dir
        folder.mkdir(parents=True, exist_ok=True)
        image = self.images[side]
        self.log(f"== {tag}: {image}")
        server = self.docker.start(image, f"{side[:4]}-p{pass_no}r{rep}")
        try:
            self._run_on(server, run, tools, dataset, workload, folder)
        except (OSError, RespError, subprocess.TimeoutExpired) as exc:
            run.finished = False
            run.stop_reason = run.stop_reason or f"{type(exc).__name__}: {exc}"[:300]
        finally:
            # Stopping: write the result first; main() removes the containers after it.
            if not self.stopping:
                alive, state = self.docker.running(server)
                if not alive and run.finished:
                    run.finished, run.stop_reason = False, f"server exited ({state})"
                try:
                    self.docker.logs(server, folder / "server.log")
                except (OSError, subprocess.SubprocessError) as exc:
                    # Best effort: the run's result does not depend on the server log.
                    self.log(f"   could not save the server log: {exc}")
                self.docker.remove(server)
        self.runs.append(run)
        self.checkpoint(f"after {run.dir}")
        self.log(
            f"   {'finished' if run.finished else 'DID NOT FINISH: ' + str(run.stop_reason)}"
            f" | wall {run.wall_s} s, query phase {run.elapsed_s} s | PING max {run.ping.get('ping_max_ms')} ms"
            f" | errors {sum(e.get('failures', 0) for e in run.errors.values())},"
            f" timeouts {sum(e.get('timeouts', 0) for e in run.errors.values())}"
        )
        return run

    def _run_on(self, server: Server, run: Run, tools: Tools, dataset: Dataset, workload: Workload,
                folder: Path) -> None:
        if not wait_ping(server.port, 120):
            run.stop_reason = "server did not answer PING within 120 s of docker run"
            return
        users, friends = tools.datasets[dataset.name]
        started = time.monotonic()
        with (folder / "load.log").open("wb") as load_log:
            load = subprocess.run(
            [str(tools.loader_python), "-I", "-m", "falkordb_bulk_loader.bulk_insert", GRAPH,
             # redis-py 8 re-sends a request after a socket timeout (5 s by default); a
             # re-sent GRAPH.BULK batch would load it twice. No timeout, so no re-send.
             "-u", f"redis://127.0.0.1:{server.port}?socket_timeout=3600",
             "-n", str(users), "-R", "Friend", str(friends), "-j", "INTEGER", "-s",
             "-i", "User:id", "-i", "User:age", "-c", "128", "-b", "16", "-t", "16"],
            stdout=load_log, stderr=subprocess.STDOUT, timeout=dataset.load_timeout_s,
        )
        if load.returncode != 0:
            run.stop_reason = f"bulk load exited {load.returncode}"
            return
        statuses: list[str] = []
        deadline = time.monotonic() + dataset.load_timeout_s
        while True:
            try:
                statuses = [row[0] for row in
                            query_rows(server.port, "CALL db.indexes() YIELD status RETURN status", timeout=5)]
            except (OSError, RespError):
                statuses = []
            if statuses and all(s == "OPERATIONAL" for s in statuses):
                break
            if time.monotonic() > deadline:
                run.stop_reason = f"indexes not operational {dataset.load_timeout_s} s after the load: {statuses}"
                return
            time.sleep(0.5)
        nodes = query_scalar(server.port, "MATCH (n:User) RETURN count(n)")
        edges = query_scalar(server.port, "MATCH ()-[r:Friend]->() RETURN count(r)")
        run.load_s = round(time.monotonic() - started, 1)
        if (nodes, edges) != (dataset.nodes, dataset.edges):
            run.stop_reason = f"load produced {nodes} users / {edges} Friend edges, expected {dataset.nodes} / {dataset.edges}"
            return
        memory = resp_call(server.port, "INFO", "memory")
        match = re.search(r"^used_memory:(\d+)", memory, re.M)
        run.used_memory_loaded = int(match.group(1)) if match else None

        qfile = self.query_files[(dataset.name, workload.key)]
        run.deadline_s = round(self.deadline_for(dataset, workload, run.side), 1)
        results = folder / "bench"
        env = dict(os.environ, FALKOR_QUERY_TIMEOUT_MS=str(QUERY_TIMEOUT_MS), NO_COLOR="1", RUST_LOG="info")
        cmd = [str(tools.benchmark), "run", "--vendor", "falkor", "--name", str(qfile.path),
               "--parallel", str(PARALLEL), "--mps", str(MPS),
               "--endpoint", f"falkor://127.0.0.1:{server.port}", "--results-dir", str(results)]
        code = self._drive(run, server.port, cmd, env, folder)
        if code is None:
            run.stop_reason = f"stopped at the deadline ({run.deadline_s:.0f} s)"
            try:
                answer = resp_call(server.port, "PING", timeout=3)
            except (OSError, RespError) as exc:
                answer = f"no reply ({exc})"
            self.log(f"   deadline hit; PING now: {answer}")
        elif code != 0:
            run.stop_reason = f"benchmark exited {code}"
        self._read_output(run, qfile, folder, code)

    def _read_output(self, run: Run, qfile: QueryFile, folder: Path, code: int | None) -> None:
        """What the benchmark left in `folder`: failed requests from run.log, and for a run
        that exited 0, metrics.prom and meta.json. Marks the run finished only when every
        generated request ended and the log agrees with metrics.prom."""
        log_text = (folder / "run.log").read_text(encoding="utf-8", errors="replace")
        run.errors, unmatched = parse_errors(log_text, qfile.by_text)
        run.unmatched_errors = len(unmatched)
        prom = folder / "bench" / "falkor" / "metrics.prom"
        meta = folder / "bench" / "falkor" / "meta.json"
        if code == 0:
            if not prom.exists() or not meta.exists():
                run.stop_reason = "benchmark exited 0 but wrote no metrics.prom / meta.json"
                return
            metrics = parse_metrics(prom.read_text(encoding="utf-8"))
            run.per_query, run.overall = metrics["per_query"], metrics["overall"]
            run.elapsed_s = round(json.loads(meta.read_text())["elapsed_ms"] / 1000, 3)
            # First: with requests missing, metrics.prom's rates are over fewer requests
            # than generated, and the log cross-check below would call that a parser error.
            missing = missing_requests(metrics["executed"], qfile.counts)
            if missing:
                run.stop_reason = missing
                return
            run.errors, problem = attribute_ambiguous(run.errors, qfile.counts, metrics["rates"])
            if problem:
                raise SetupError(f"{run.dir}: {problem}; the log parser is out of date")
            self._cross_check(run, metrics, qfile, unmatched)
            run.finished = run.stop_reason is None
        elif any("|" in key for key in run.errors):
            # No metrics.prom to split a shared text by: those failures stay unattributed.
            self.log(f"   unattributed (shared query text): "
                     f"{ {k: v for k, v in run.errors.items() if '|' in k} }")

    def _drive(self, run: Run, port: int, cmd: list[str], env: dict[str, str], folder: Path) -> int | None:
        """Run the benchmark under the PING probe; None when it passed its deadline.

        The probe stops on every path, a failed start included: a probe left running
        would keep pinging whatever server takes this port next.
        """
        probe = PingProbe(port, folder / "ping.csv")
        probe.start()
        started = time.monotonic()
        try:
            with (folder / "run.log").open("wb") as sink:
                proc = subprocess.Popen(cmd, cwd=folder, env=env, stdout=sink, stderr=subprocess.STDOUT)
                try:
                    return proc.wait(timeout=run.deadline_s)
                except subprocess.TimeoutExpired:
                    return None
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=1 if self.stopping else 15)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
        finally:
            run.wall_s = round(time.monotonic() - started, 2)
            probe.stop()
            run.ping = probe.summary()

    def _cross_check(self, run: Run, metrics: dict[str, Any], qfile: QueryFile, unmatched: list[str]) -> None:
        """The failed requests parsed from the log must agree with metrics.prom's rates.

        Totals per query, not per kind: the tool calls its own guard's timeout a failure.
        """
        for query, count in qfile.counts.items():
            rates = metrics["rates"].get(query, {})
            expected = round((rates.get("timeout", 0.0) + rates.get("failure", 0.0)) * count / 100)
            got = sum(run.errors.get(query, {}).values())
            if expected != got:
                raise SetupError(
                    f"{run.dir}: {query} has {got} failed requests in run.log but {rates} % of {count} "
                    f"in metrics.prom; the log parser is out of date"
                )
        if unmatched:
            raise SetupError(f"{run.dir}: {len(unmatched)} failed requests in run.log match no generated query, "
                             f"e.g. {unmatched[0]!r}")

    # -- the plan --

    def generate_queries(self, tools: Tools, dataset: Dataset) -> None:
        for workload in self.workloads:
            path = self.raw / f"queries-{dataset.name}-{workload.key}.jsonl"
            cmd = [str(tools.benchmark), "generate-queries", "--vendor", "falkor", "--size", str(QUERY_COUNT),
                   "--dataset", dataset.name, "--name", str(path), "--write-ratio", str(workload.write_ratio)]
            for flag in ALGORITHM_FLAGS:
                cmd += [flag, "false"]
            run_cmd(cmd, self.log, cwd=self.raw, timeout=600, output=self.raw / "setup.log")
            self.query_files[(dataset.name, workload.key)] = read_query_file(path)

    def measure(self, tools: Tools, dataset: Dataset) -> None:
        for side in ("baseline", "candidate"):
            facts = self.probe(side)
            self.servers[side] = facts
            self.log(f"{side}: {json.dumps(facts)[:1500]}")
            if facts.get("error"):
                if side == "baseline":
                    raise SetupError(f"baseline server ({self.images[side]}) {facts['error']}")
                self.info.append(make_finding(f"server started [{dataset.name}]", "completed", 0, 1, None,
                                              "lower_is_worse", "regression"))
                return
            problem = timeout_problem(facts["timeout_test"], facts.get("TIMEOUT"))
            if problem:
                raise SetupError(f"{side}: {problem}: the benchmark's per-query timeout ({QUERY_TIMEOUT_MS} ms) "
                                 f"is not shown to be what this server applies")
            if str(facts.get("THREAD_COUNT")) != str(THREAD_COUNT):
                raise SetupError(f"{side}: THREAD_COUNT is {facts.get('THREAD_COUNT')}, not {THREAD_COUNT}")
        self.generate_queries(tools, dataset)

        # First pass: rep 1 runs the baseline first (so the candidate's deadline has a
        # reference); every rep after swaps the order.
        for rep in range(1, self.mode.reps + 1):
            for workload in self.workloads:
                for side in self.order(rep):
                    self.run_one(tools, dataset, workload, side, 1, rep)
        analysis = analyse(self.runs, self.mode, self.query_files, self.info, self.rerun_done)
        for note in analysis.notes:
            if "no re-run" in note:
                self.log(note)
                self.notes.append(note)
        todo = [w for w in self.workloads if (dataset.name, w.key) in analysis.rerun]
        for workload in todo:
            why = analysis.rerun_why.get((dataset.name, workload.key), [])
            self.log(f"re-run {dataset.name} {workload.label} to confirm: {'; '.join(why[:5])}"
                     + (f" (+{len(why) - 5} more)" if len(why) > 5 else ""))
        for rep in range(1, self.mode.rerun_reps + 1):
            for workload in todo:
                for side in self.order(rep + 1):
                    self.run_one(tools, dataset, workload, side, 2, rep)
        for workload in todo:
            self.rerun_done.add((dataset.name, workload.key))

    @staticmethod
    def order(rep: int) -> tuple[str, str]:
        return ("baseline", "candidate") if rep % 2 else ("candidate", "baseline")

    # -- result --

    def result(self, unfinished: str | None = None) -> dict[str, Any]:
        """The result from the runs so far; `unfinished` says why it is only a checkpoint."""
        analysis = analyse(self.runs, self.mode, self.query_files, self.info,
                           self.rerun_done | {(r.dataset, r.workload) for r in self.runs})
        selected = select_findings(analysis.findings)
        nothing = None if self.runs else "no benchmark run was made"
        verdict, error = verdict_of(selected, analysis, self.error or unfinished or nothing)
        counts = {"runs": len(self.runs), "rows": len(analysis.rows)}
        self.write_tables(analysis)
        artifacts = [p.relative_to(self.out).as_posix() for p in sorted(self.raw.glob("*")) if p.is_file()]
        if self.runs:
            artifacts.append("raw/runs")
        sides = {side: self.side_object(side) for side in ("candidate", "baseline")}
        return {
            "schema": 1,
            "check": CHECK,
            "verdict": verdict,
            "confirmed": verdict == "regression",
            "summary": summary_of(verdict, selected, error, counts, self.args.mode, analysis.findings),
            "candidate": sides["candidate"],
            "baseline": sides["baseline"],
            "machine": self.machine,
            "started": self.started,
            "finished": now(),
            "findings": selected,
            "error": error if verdict == "error" else None,
            "artifacts": artifacts,
        }

    def write_tables(self, analysis: Analysis) -> None:
        columns = ["dataset", "workload", "name", "metric", "candidate", "baseline", "ratio", "direction",
                   "status", "use", "first_pass", "re_run", "detail"]
        with (self.raw / "rows.csv").open("w", newline="", encoding="utf-8") as sink:
            writer = csv.DictWriter(sink, columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(analysis.rows)
        run_columns = ["dataset", "workload", "side", "pass_no", "rep", "finished", "stop_reason", "wall_s",
                       "elapsed_s", "load_s", "deadline_s", "used_memory_loaded", "ping_max_ms", "ping_samples",
                       "ping_over_100ms", "ping_over_1s", "ping_timeouts", "ping_errors", "p50_ms", "p95_ms",
                       "p99_ms", "throughput", "errors", "timeouts", "dir"]
        with (self.raw / "runs.csv").open("w", newline="", encoding="utf-8") as sink:
            writer = csv.DictWriter(sink, run_columns, extrasaction="ignore")
            writer.writeheader()
            for run in self.runs:
                writer.writerow({
                    **{k: getattr(run, k) for k in ("dataset", "workload", "side", "pass_no", "rep", "finished",
                                                    "stop_reason", "wall_s", "elapsed_s", "load_s", "deadline_s",
                                                    "used_memory_loaded", "dir")},
                    **run.ping, **{k: rounded(v) for k, v in run.overall.items()},
                    "throughput": rounded(run.throughput),
                    "errors": sum(e.get("failures", 0) for e in run.errors.values()),
                    "timeouts": sum(e.get("timeouts", 0) for e in run.errors.values()),
                })
        (self.raw / "servers.json").write_text(json.dumps(self.servers, indent=2) + "\n")
        (self.raw / "notes.txt").write_text("\n".join(self.notes + analysis.notes) + "\n")

    def side_object(self, side: str) -> dict[str, Any]:
        commit = getattr(self.args, f"{side}_commit")
        return {
            "image": self.images[side],
            "commit": commit if commit and COMMIT_RE.match(commit) else None,
            "label": getattr(self.args, f"{side}_label") or default_label(self.images[side]),
        }

    def write_result(self, unfinished: str | None = None) -> dict[str, Any]:
        try:
            result = self.result(unfinished)
        except Exception as exc:  # noqa: BLE001 - a result.json must exist whatever happened
            self.log(traceback.format_exc())
            result = self.bare_result(f"{self.error or unfinished or ''} (and building the result failed: "
                                      f"{type(exc).__name__}: {exc})".strip())
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        partial = self.out / "result.json.part"
        partial.write_text(text)
        partial.replace(self.out / "result.json")
        return result

    def bare_result(self, error: str) -> dict[str, Any]:
        return {
            "schema": 1, "check": CHECK, "verdict": "error", "confirmed": False,
            "summary": " ".join(f"Could not measure: {error}".split())[:500],
            "candidate": self.side_object("candidate"),
            "baseline": self.side_object("baseline"),
            "machine": self.machine, "started": self.started, "finished": now(), "findings": [],
            "error": error, "artifacts": [],
        }

    def checkpoint(self, where: str) -> None:
        """Rewrite result.json from the runs so far: if the check is killed, it stands."""
        try:
            self.write_result(f"the check did not finish; this result is its checkpoint {where}")
        except OSError as exc:
            self.log(f"checkpoint failed: {exc}")

    def main(self) -> None:
        if self.cleanup_error:
            raise SetupError(self.cleanup_error)
        self.log(f"pokec check, mode {self.args.mode}: candidate {self.images['candidate']}, "
                 f"baseline {self.images['baseline']}")
        for side, image in self.images.items():
            if not DIGEST_RE.match(image):
                raise SetupError(f"the {side} image {image!r} is not pinned by digest (name@sha256:<64 hex>)")
        tools = self.setup()
        for name in self.mode.datasets:
            self.measure(tools, DATASETS[name])


def main(argv: list[str] | None = None) -> int:
    check = Check(parse_args(argv))

    def on_term(signum: int, _frame: Any) -> None:
        # Once: `timeout --kill-after=60` leaves a minute to write the result.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        check.stopping = True
        raise Stopped(f"signal {signum}")

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    try:
        check.checkpoint("at start")
        check.main()
    except Stopped:
        minutes = (time.monotonic() - check.started_mono) / 60
        check.error = f"stopped by a signal after {minutes:.0f} min (the workflow's timeout?); partial data in raw/"
    except SetupError as exc:
        check.error = str(exc)
    except Exception as exc:  # noqa: BLE001 - always write result.json
        check.error = f"the check crashed: {type(exc).__name__}: {exc}"
        check.log(traceback.format_exc())
    # From here on a TERM must not interrupt the final write (`--kill-after` leaves 60 s).
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if check.error:
        check.log(f"ERROR: {check.error}")
    try:
        result = check.write_result()
        check.log(f"verdict {result['verdict']}: {result['summary']}")
    finally:
        # Also when the write failed (a full disk): the last checkpoint is the result then,
        # and a loaded server must not outlive the check.
        try:
            check.docker.remove_all()
        except (OSError, subprocess.SubprocessError) as exc:
            # Best effort, after the result: the containers carry the run's label for later cleanup.
            check.log(f"could not remove the containers: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
