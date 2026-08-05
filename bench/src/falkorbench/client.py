"""Server lifecycle and the control plane, over falkordb-py.

Replaces two hand-rolled `redis-cli` subprocess wrappers that had drifted apart
— one carried a timeout, the other carried server-log diagnostics, and both
detected success by looking for the substring `"execution time"` in stdout,
because `redis-cli` exits 0 even when the reply is an error. `Graph.query`
raises `ResponseError` instead, so failure is an exception rather than a string
that happens not to match.

Everything here runs *outside* measurement windows. The measured workload stays
`redis-benchmark` (and `redis-cli -r N` under callgrind) for the reasons in
`counters`.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from falkordb import FalkorDB
from falkordb import Graph
from redis.exceptions import RedisError
from redis.exceptions import ResponseError

from falkorbench.model import Metric

GRAPH_NAME = "bench"

# Panic/crash markers worth surfacing from a server log. When the module panics,
# redis prints the panic and backtrace to its log and dies, and every later
# command fails with the useless "Server closed the connection" — which was all
# CI ever reported before the log was kept.
_DEATH_MARKERS = (
    "panicked at",
    "FalkorDB panic",
    "Redis crashed",
    "signal:",
    "=== REDIS BUG REPORT",
)


class SetupFailed(RuntimeError):
    """Graph setup did not complete, so nothing measured afterwards is valid."""


@dataclass
class Server:
    """A redis-server this harness started, or one it attached to.

    `proc` and `log_path` are None when attached (`--reuse`): the process is
    someone else's and its log is not ours to read.
    """

    port: int
    proc: subprocess.Popen | None = None
    log_path: Path | None = None
    work_dir: Path | None = None

    def death_details(self) -> str:
        """Panic/crash lines from the log, for when a command failed because the
        server is gone. Empty string when there is nothing to add."""
        if self.log_path is None or not self.log_path.exists():
            return ""
        try:
            lines = self.log_path.read_text(errors="replace").splitlines()
        except OSError:
            return ""
        marked = [ln.rstrip() for ln in lines if any(m in ln for m in _DEATH_MARKERS)]
        tail = marked[:6] or [ln.rstrip() for ln in lines[-6:]]
        return "\n  server log: " + "\n              ".join(tail)

    def stop(self) -> None:
        if self.proc is None:
            return
        self.proc.send_signal(signal.SIGTERM)
        self.proc.wait()
        self.proc = None


@dataclass
class BenchClient:
    """The control plane for one server: setup, probes, allocation snapshots."""

    db: FalkorDB
    server: Server
    graph_name: str = GRAPH_NAME
    _graph: Graph | None = field(default=None, repr=False)

    @property
    def graph(self) -> Graph:
        if self._graph is None:
            self._graph = self.db.select_graph(self.graph_name)
        return self._graph

    # --- queries -------------------------------------------------------------

    def run(self, cypher: str, write: bool = True) -> None:
        """Execute one statement, raising ResponseError on an error reply."""
        if write:
            self.graph.query(cypher)
        else:
            self.graph.ro_query(cypher)

    def command(self, *args: object) -> object:
        """A raw redis command, for what the graph API does not cover:
        DEBUG RELOAD, GRAPH.CONSTRAINT, GRAPH.UDF, MEMORY MALLOC-STATS."""
        return self.db.execute_command(*args)

    # --- probes --------------------------------------------------------------

    @property
    def pid(self) -> int:
        """The server's own pid, from a parsed INFO map.

        Was recovered by string surgery on redis-cli output
        (`out.split("process_id:")[1].split()[0]`) in two different files.
        """
        return int(self.db.connection.info("server")["process_id"])

    def jemalloc_totals(self) -> tuple[Metric, Metric]:
        """Cumulative (allocated, deallocated) bytes from jemalloc's merged-arena
        stats, or (None, None) if the server is not jemalloc-built.

        Sums size*nmalloc / size*ndalloc over the `bins:` and `large:`
        size-class tables. Column positions are read from each table's own
        header rather than hardcoded: they *were* hardcoded, and the dealloc
        index was wrong — jemalloc 5.x emits
        `size ind allocated nmalloc ndalloc nrequests ...`, so index 5 is
        *nrequests*, not ndalloc. That inflated deallocated bytes and made the
        per-query deltas meaningless (absurd ratios, negative values). Reading
        the header also survives column changes between jemalloc versions.
        """
        try:
            out = str(self.command("MEMORY", "MALLOC-STATS"))
        except RedisError:
            return None, None
        if "Merged arenas stats:" not in out:
            return None, None

        alloc = dealloc = 0
        in_merged = in_table = False
        i_nmalloc = i_ndalloc = None
        for line in out.splitlines():
            if line.startswith("Merged arenas stats:"):
                in_merged = True
            elif line.startswith("arenas["):
                break
            elif in_merged:
                s = line.split()
                if not s:
                    continue
                if s[0] in ("bins:", "large:") and "size" in s:
                    in_table = True
                    # Header tokens include the leading "bins:"/"large:" label,
                    # which data rows do not, hence the -1.
                    try:
                        i_nmalloc = s.index("nmalloc") - 1
                        i_ndalloc = s.index("ndalloc") - 1
                    except ValueError:
                        i_nmalloc = i_ndalloc = None
                elif s[0] == "extents:":
                    in_table = False
                elif in_table and s[0].isdigit() and i_nmalloc is not None:
                    if len(s) <= max(i_nmalloc, i_ndalloc):
                        continue
                    size = int(s[0])
                    alloc += size * int(s[i_nmalloc])
                    dealloc += size * int(s[i_ndalloc])
        return alloc, dealloc

    def shutdown(self) -> None:
        """Graceful SHUTDOWN NOSAVE — which also flushes .profraw under
        coverage instrumentation, so it must not be replaced by SIGKILL."""
        # The server closing the connection mid-command is the success case here.
        with contextlib.suppress(RedisError):
            self.db.connection.shutdown(nosave=True)
        if self.server.proc is not None:
            self.server.proc.wait()
            self.server.proc = None


# --- lifecycle ---------------------------------------------------------------


def is_server_up(port: int) -> bool:
    """True when something answers on `port`."""
    try:
        FalkorDB(host="localhost", port=port).connection.ping()
        return True
    except RedisError:
        return False


def start_server(
    module: Path,
    port: int,
    work_dir: Path,
    import_dir: Path,
    *,
    appendonly: bool = False,
    module_args: Sequence[str] = (),
) -> Server:
    """Start a redis-server with the module, in an isolated data dir."""
    # DEBUG RELOAD during setup writes a dump.rdb into the server dir, and a
    # later server would reload it (making setup fail with "already indexed").
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)

    argv = [
        "redis-server",
        "--port",
        str(port),
        "--save",
        "",
        "--enable-debug-command",
        "local",
        "--dir",
        str(work_dir),
        # Trailing slash: the C module concatenates IMPORT_FOLDER + filename
        # directly, without inserting a separator.
        "--loadmodule",
        str(module),
        "IMPORT_FOLDER",
        f"{import_dir}/",
        *module_args,
    ]
    if appendonly:
        # AOF makes the module emit replication effects on every write
        # (Pending::build_effects_buffer), covering those paths.
        argv += ["--appendonly", "yes", "--appendfsync", "no"]

    log_path = work_dir / "server.log"
    # Popen dups the fd into the child, so close the parent's copy rather than
    # leaking it for the life of the run.
    with open(log_path, "w") as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
    return Server(port=port, proc=proc, log_path=log_path, work_dir=work_dir)


def connect(server: Server, *, timeout: float = 10.0, poll: float = 0.1) -> BenchClient:
    """Wait for `server` to answer, then return a client for it.

    Constructing a FalkorDB probes the server (it checks whether it is talking to
    a sentinel), so "not up yet" surfaces as a ConnectionError and is simply
    retried.
    """
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        if server.proc is not None and server.proc.poll() is not None:
            raise SetupFailed(f"server exited during startup{server.death_details()}")
        try:
            db = FalkorDB(host="localhost", port=server.port)
            db.connection.ping()
            return BenchClient(db=db, server=server)
        except RedisError as e:
            last = e
            time.sleep(poll)
    raise SetupFailed(f"server did not answer on :{server.port} ({last})")


def write_csv_fixtures(import_dir: Path, files: dict[str, str]) -> None:
    """Materialise the LOAD CSV corpus the query set imports."""
    import_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (import_dir / name).write_text(content)


def build_graph(
    client: BenchClient,
    setup: Sequence[str],
    setup_commands: Sequence[Sequence[str]],
    *,
    c_compat: bool = False,
) -> None:
    """Build the benchmark graph. Raises SetupFailed on anything unexpected.

    Graph setup is deliberately separate from server lifecycle: "do not start a
    server" must not silently mean "do not build the graph", or the harness
    benchmarks an empty database and reports numbers that look real.
    """
    for stmt in setup:
        try:
            client.graph.query(stmt)
        except ResponseError as e:
            raise SetupFailed(f"setup failed: {stmt[:60]}…: {e}") from e

    for cmd in setup_commands:
        if c_compat and _skip_for_c(cmd):
            continue
        args = [str(a).replace("{graph}", client.graph_name) for a in cmd]
        try:
            client.command(*args)
        except ResponseError as e:
            # Observational commands are tolerated against the C engine only.
            #
            # GRAPH.EXPLAIN / PROFILE / MEMORY do not shape the graph that gets
            # measured; they exist so the coverage run exercises plan rendering
            # and the memory report. The C engine legitimately rejects some of
            # what they contain (it refuses `ORDER BY` on a variable this set does
            # not project), and that must not abort a --c-compat run the way an
            # unbuildable graph would.
            #
            # For the Rust engine these stay fatal, deliberately. One of them had
            # been failing silently for as long as it existed — redis-cli writes
            # error replies to stderr and the pre-refactor harness inspected only
            # stdout, so its "did this command fail" check could never fire.
            if c_compat and _is_observational(cmd):
                print(f"  (skipped on C: {' '.join(args[:2])}: {e})", flush=True)
                continue
            raise SetupFailed(f"setup command failed: {' '.join(args[:2])}: {e}") from e


#: Setup commands that observe rather than build. They cover the engine's plan
#: rendering and memory-report paths for the coverage run; nothing the benchmark
#: measures depends on them having succeeded.
_OBSERVATIONAL = ("GRAPH.EXPLAIN", "GRAPH.PROFILE", "GRAPH.MEMORY")


def _is_observational(cmd: Sequence[str]) -> bool:
    return bool(cmd) and str(cmd[0]) in _OBSERVATIONAL


def _skip_for_c(cmd: Sequence[str]) -> bool:
    """Setup commands the C engine cannot take.

    Each of these is a real C-side behaviour, not a guess:
      - the async validation of a composite (2-property) unique constraint
        crashes the C server;
      - its RDB round-trip drops numeric 0 from the Person range index, so after
        DEBUG RELOAD `MATCH (:Person {id: 0})` finds nothing and every id-0 row
        silently becomes a no-op;
      - UDFs are Rust-only, and `ERR unknown command` would abort setup, which
        is how a --c-compat run once died before measuring a single query.
    """
    head = list(cmd[:2])
    if head == ["GRAPH.CONSTRAINT", "CREATE"] and "2" in cmd:
        return True
    if list(cmd) == ["DEBUG", "RELOAD"]:
        return True
    return cmd[0] == "GRAPH.UDF"


def find_module(explicit: str | None, root: Path) -> Path:
    """Resolve the module path, defaulting to this repo's release build."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    ext = "dylib" if os.uname().sysname == "Darwin" else "so"
    return root / f"target/release/libfalkordb.{ext}"
