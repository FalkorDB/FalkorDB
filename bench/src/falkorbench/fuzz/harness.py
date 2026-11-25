"""Drives one server: builds the graphs, runs epochs of long reads, writes and cheap reads, and
records everything `checks.analyse` judges.

Per server-hour: a fresh server from the image (THREAD_COUNT given), a plain server from the same
image without the module beside it (the PING control), 1-4 graphs at their own scales. Per epoch:
busy graphs, long-read streams, write and read rates, write pattern, indexes, constraints, write
timeouts and BGSAVE, all from the epoch's parameters (`model.epoch_params`).

Clients never re-send (no socket timeout, no retries) and speak RESP2. Reads and writes are open
loop: sent at their scheduled time whether or not earlier ones have returned, and timed from that
schedule. A separate process PINGs FalkorDB and the plain server every 10 ms, timed from the
scheduled send.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import gzip
import json
import multiprocessing as mp
import os
import random
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback

import redis
from redis.asyncio.retry import Retry as AsyncRetry
from redis.backoff import NoBackoff
from redis.exceptions import ResponseError
from redis.retry import Retry

from falkorbench import kit
from falkorbench.soak import soakkit as sk

from . import checks
from . import model
from .checks import gname

PING_INTERVAL_S = checks.PING_INTERVAL_S
HANG_S = 60  # a FalkorDB PING unanswered this long: the server is hung
MAX_INFLIGHT_WRITES = 900  # per graph; C crashes when a graph's 1,024-entry write queue overflows
CONTAINER_PORT = 6379  # the server's port inside its own network namespace (published mode)


def now() -> float:
    return time.monotonic()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def s(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


#: redis-py 8 speaks RESP3 unless told otherwise (HELLO 3 on connect); the engines are measured on
#: RESP2's replies, as the check's acceptance runs were
PROTOCOL = 2


def connect(port: int, host="127.0.0.1", timeout: float | None = None) -> redis.Redis:
    """Never re-sends: redis-py 8 retries a failed command by default, so retries are off
    explicitly. Measured requests use no socket timeout (a latency is one request, however long);
    setup and checking connections pass one, so a hung server cannot block the run forever."""
    return redis.Redis(
        host=host,
        port=port,
        socket_timeout=timeout,
        socket_connect_timeout=10,
        retry=Retry(NoBackoff(), 0),
        protocol=PROTOCOL,
    )


def async_pool(base, port: int, max_conns: int, **kw):
    """The async driver's pool (`base` a redis.asyncio pool class), set up as connect(): async
    connections take their retry and protocol from the pool, and the retry must be asyncio's."""
    return base(
        host="127.0.0.1",
        port=port,
        max_connections=max_conns,
        timeout=None,
        socket_timeout=None,
        socket_connect_timeout=10,
        retry=AsyncRetry(NoBackoff(), 0),
        protocol=PROTOCOL,
        **kw,
    )


ADMIN_TIMEOUT_S = 300
INDEX_WAIT_S = 300.0  # after the load, for every index still building, together


def server_hung(port: int, tries: int = 3) -> bool:
    """A running server that answers none of `tries` PINGs, each given 10 s."""
    for _ in range(tries):
        r = connect(port, timeout=10)
        try:
            if r.ping():
                return False
        except Exception:
            pass  # a timeout or a refused connection is no answer: try again
        finally:
            r.close()
    return True


def outcome_of(exc):
    return checks.outcome_of(exc, ResponseError)


def qargs(graph, cypher, params, write, timeout_ms):
    prefix = (
        "CYPHER " + " ".join(f"{k}={sk.lit(v)}" for k, v in params.items()) + " " if params else ""
    )
    a = ["GRAPH.QUERY" if write else "GRAPH.RO_QUERY", graph, prefix + cypher]
    if timeout_ms:
        a += ["TIMEOUT", str(int(timeout_ms))]
    return a


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# --- images and servers ----------------------------------------------------------------------------


def docker(*a, timeout=120) -> subprocess.CompletedProcess:
    """A docker call the kit has no helper for (stop, kill, a one-off `run --rm`). Its exit code is
    the caller's to read; docker missing or not returning in time is kit.DockerError."""
    try:
        return subprocess.run(["docker", *a], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise kit.DockerError(f"docker {a[0]} did not return within {timeout} s") from None
    except OSError as e:
        raise kit.DockerError(f"docker could not be started: {e}") from None


def probe_image(image: str) -> dict:
    """Pull the image and find its server and module. The server is the image's own default, as
    the soak picks it (soakkit.image_server: `${FALKORDB_SERVER:-redis}-server`, no fallback to the
    other binary); the module is where every falkordb-server image keeps it. Raises RuntimeError if
    either is missing."""
    kit.pull(image)
    binary = sk.image_server(image)
    script = f"command -v {binary}; if [ -f {sk.IMAGE_MODULE} ]; then echo {sk.IMAGE_MODULE}; fi"
    p = docker(
        "run", "--rm", "--network", "none", "--entrypoint", "sh", image, "-c", script, timeout=300
    )
    lines = [x.strip() for x in p.stdout.splitlines() if x.strip()]
    server = next((x for x in lines if x.endswith("/" + binary)), None)
    if not server:
        raise RuntimeError(f"image {image} has no {binary} (the server its FALKORDB_SERVER names)")
    if sk.IMAGE_MODULE not in lines:
        raise RuntimeError(f"image {image} has no {sk.IMAGE_MODULE}")
    return {"image": image, "server": server, "module": sk.IMAGE_MODULE}


def after_failure(server, exc, hang_path: str | None = None) -> dict:
    """What a failed step says about the engine: server_alive False (it exited, or it is hung: the
    prober's marker or no PONG to 3 PINGs), True, or None when Docker failed and nothing is known."""
    out = {"server_alive": None, "exit_code": None, "hung": False}
    if isinstance(exc, kit.DockerError):
        return out
    try:
        running, code = server.state()
        hung = running and (
            (hang_path is not None and os.path.exists(hang_path)) or server_hung(server.port)
        )
        out.update(
            server_alive=running and not hung, exit_code=None if running else code, hung=hung
        )
    except kit.DockerError:
        pass  # docker cannot say either: nothing is known about the engine
    return out


#: every server, with the module or without: no persistence, room for the drivers' connections
SERVER_FLAGS = {
    "--maxclients": "200000",
    "--save": "",
    "--appendonly": "no",
    "--dir": "/tmp",
    "--protected-mode": "no",
}


def _flags(pairs: dict) -> list[str]:
    return [x for pair in pairs.items() for x in pair]


class Server:
    """One container from the image: its own server binary, with the module (FalkorDB) or without
    (the plain server that tells a host stall from a FalkorDB freeze). On the host network it
    listens on a free port of 127.0.0.1; published (macOS), on CONTAINER_PORT behind a host port
    docker picks."""

    def __init__(
        self,
        info: dict,
        name: str,
        cpus: str | None,
        net: str,
        thread_count: int | None = None,
        module: bool = True,
        module_args: tuple = (),
        labels: dict | None = None,
    ):
        self.info, self.name, self.cpus, self.net = info, name, cpus, net
        self.tc, self.module, self.module_args = thread_count, module, module_args
        self.labels = labels
        self.port = 0  # set by start()

    def start(self, wait_s: float = 60) -> redis.Redis:
        host = self.net == "host"
        listen = free_port() if host else CONTAINER_PORT
        bind = "127.0.0.1" if host else "0.0.0.0"
        cmd = _flags({"--port": str(listen), "--bind": bind, **SERVER_FLAGS})
        if self.module:
            conf = {"THREAD_COUNT": str(self.tc), "CACHE_SIZE": "200", "EFFECTS_THRESHOLD": "0"}
            cmd += ["--loadmodule", self.info["module"], *_flags(conf), *self.module_args]
        container = kit.start(
            self.info["image"],
            self.name,
            labels=self.labels,
            cpus=self.cpus,
            entrypoint=self.info["server"],
            command=cmd,
            extra=["--ulimit", "nofile=1048576"],
            port=listen,
            host_network=host,
        )
        self.port = container.port
        if self.port and kit.wait_ping(self.port, wait_s, container):
            return connect(self.port, timeout=ADMIN_TIMEOUT_S)
        raise RuntimeError("server did not start: " + self.logs(50)[-1500:])

    def state(self) -> tuple[bool, int]:
        """(running, exit code); a container docker no longer knows is (False, -1). A docker that
        does not answer raises kit.DockerError: it never reads as a crash."""
        state = kit.state(self.name)
        if state == "running":
            return True, 0
        code = state.split()[1] if len(state.split()) > 1 else ""
        return False, int(code) if code.lstrip("-").isdigit() else -1

    def pid(self) -> int:
        return kit.host_pid(self.name)

    def logs(self, tail=400) -> str:
        try:
            return kit.logs(self.name, tail)
        except kit.DockerError as e:
            return f"(no log: {e})"

    def stop(self) -> str:
        """Stop and remove the container; returns its log tail. Best effort: never raises. Once the
        check is stopping (its time limit), the container goes at once, without a graceful stop."""
        try:
            if not kit.stopping():
                docker("stop", "-t", "20", self.name, timeout=60)
        except kit.DockerError:
            pass  # removed below either way
        tail = self.logs(200)
        with contextlib.suppress(kit.DockerError):  # the check's clean-up removes what is left
            kit.remove(self.name)
        return tail


def module_version(r) -> str:
    ver = "?"
    for m in r.execute_command("MODULE", "LIST"):
        d = (
            dict(zip(map(s, m[::2]), m[1::2], strict=False))
            if isinstance(m, list)
            else {s(k): v for k, v in m.items()}
        )
        if s(d.get("name", "")) in ("graph", "falkordb"):
            ver = s(d.get("ver"))
    return ver


def server_config(r) -> dict:
    try:
        flat = [s(x) for p in r.execute_command("GRAPH.CONFIG", "GET", "*") for x in p]
        return dict(zip(flat[::2], flat[1::2], strict=False))
    except Exception as e:
        return {"error": str(e)[:200]}


# --- PING prober process -------------------------------------------------------------------------


def prober_proc(ports: dict, out_path: str, hang_path: str, stop):
    asyncio.run(_prober(ports, out_path, hang_path, stop))


async def _prober(ports, out_path, hang_path, stop):
    # line-buffered: an epoch reads it while this writes
    with open(out_path, "a", buffering=1) as f:
        await asyncio.gather(*(_probe(f, n, p, hang_path, stop) for n, p in ports.items()))


async def _replies(f, name, reader, pending, alive):
    """Time each PONG from its PING's scheduled send; alive[0] goes False when the connection ends."""
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            t = now()
            if pending:
                sched = pending.popleft()
                if line.startswith(b"+PONG"):  # an error reply (-LOADING) is no sample of a PONG
                    f.write(f"{name},{sched:.4f},{(t - sched) * 1000:.2f}\n")
    finally:
        alive[0] = False


async def _probe(f, name, port, hang_path, stop):
    while not stop.is_set():
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
        except OSError:
            f.write(f"{name},{now():.4f},down\n")
            await asyncio.sleep(1)
            continue
        pending = collections.deque()
        alive = [True]
        rt = asyncio.create_task(_replies(f, name, reader, pending, alive))
        nxt = now()
        hung = False
        try:
            while alive[0] and not stop.is_set():
                nxt += PING_INTERVAL_S
                d = nxt - now()
                if d > 0:
                    await asyncio.sleep(d)
                else:
                    nxt = now()  # we fell behind (the prober itself stalled); do not burst
                pending.append(nxt)
                writer.write(b"PING\r\n")
                if name == "falkor" and pending and now() - pending[0] > HANG_S and not hung:
                    hung = True
                    with open(hang_path, "w") as h:
                        h.write(f"{now():.3f}\n")
        except (ConnectionError, OSError):
            pass  # the server dropped the connection: logged as down below, then reconnect
        finally:
            rt.cancel()
            # the transport is already broken: nothing to close
            with contextlib.suppress(Exception):
                writer.close()
        if not stop.is_set():
            f.write(f"{name},{now():.4f},down\n")
            await asyncio.sleep(1)


# --- long-read process (threads) ---------------------------------------------------------------------


class SnapshotGate:
    """Lets a write through only while its record can still reach the results: `admit` records it
    before it is sent, `close` takes the snapshot, and a write admitted after that is never sent."""

    def __init__(self):
        self.lock = threading.Lock()
        self.records: list = []
        self.closed = False

    def admit(self, rec) -> bool:
        with self.lock:
            if self.closed:
                return False
            self.records.append(rec)
            return True

    def shut(self):
        """Admit no more writes; those already admitted keep their records."""
        with self.lock:
            self.closed = True

    def close(self, write):
        """Stop admitting writes and call write(records) under the lock (no record changes meanwhile)."""
        with self.lock:
            self.closed = True
            write(self.records)


def finish_aligned(gate, timers, deadline) -> int:
    """Admit no more aligned writes, give those in flight until the drain's deadline, and return
    how many are still waiting: they count as unfinished, like a long read still running."""
    gate.shut()
    for t in timers:
        t.join(max(0.0, deadline - now()))
    return sum(t.is_alive() for t in timers)


def long_proc(cfg: dict, stop, scanning, out_path: str):
    ep, sizes, port = cfg["ep"], cfg["sizes"], cfg["port"]
    longs, died, timers = [], [], []
    lock = threading.Lock()
    gate = SnapshotGate()  # the aligned writes, each on a timer thread

    def aligned_write(g, rng):
        cypher, params, keys = model.make_upsert(sizes[g], rng)
        t0 = now()
        # recorded before sending (outcome 3, unknown), or not sent at all once the results are taken
        rec = [g, "scan-write", t0, t0, None, 3, "", keys, 0.0]
        if not gate.admit(rec):
            return
        r = connect(port)
        try:
            try:
                r.execute_command(*qargs(gname(g), cypher, params, True, None))
                exc = None
            except Exception as e:
                exc = e
            t1 = now()
            oc, msg = outcome_of(exc)
            with gate.lock:
                rec[4], rec[5], rec[6] = t1, oc, msg
        finally:
            r.close()

    def stream(g, j):
        rng = random.Random(f"{ep['seed']}-long-{g}-{j}")
        r = connect(port)
        try:
            stop.wait(rng.uniform(0, model.LONG_START_S))
            while not stop.is_set():
                kind = "full-scan" if rng.random() < model.SCAN_SHARE else "full-count"
                batch = model.make_long(kind)
                with scanning.get_lock():
                    scanning[g] += 1
                try:
                    if rng.random() < model.SCAN_WRITE_P:
                        timer = threading.Timer(
                            0.001, aligned_write, args=(g, random.Random(rng.random()))
                        )
                        with lock:
                            timers.append(timer)
                        timer.start()
                    for op, cypher, params in batch:
                        if stop.is_set():
                            break
                        t0 = now()
                        try:
                            r.execute_command(
                                *qargs(
                                    gname(g), cypher, params, False, model.LONG_TIMEOUT[op] * 1000
                                )
                            )
                            exc = None
                        except Exception as e:
                            exc = e
                        t1 = now()
                        oc, msg = outcome_of(exc)
                        with lock:
                            longs.append([g, op, t0, t1, oc, msg])
                        if oc == 4:
                            raise exc
                finally:
                    with scanning.get_lock():
                        scanning[g] -= 1
                stop.wait(rng.expovariate(1 / model.LONG_PAUSE_S))
        except Exception as e:
            with lock:
                died.append(f"long[{g}.{j}]: {type(e).__name__}: {e}")
        finally:
            r.close()

    threads = []
    for gs, k in ep["long_streams"].items():
        for j in range(k):
            threads.append(threading.Thread(target=stream, args=(int(gs), j), daemon=True))
    for t in threads:
        t.start()
    stop.wait()
    deadline = now() + cfg["drain_s"]
    for t in threads:
        t.join(max(0.1, deadline - now()))
    with lock:
        started = list(timers)
    unfinished = sum(t.is_alive() for t in threads) + finish_aligned(gate, started, deadline)

    def dump(writes):
        with lock, open(out_path, "w") as f:
            json.dump({"longs": longs, "writes": writes, "died": died, "unfinished": unfinished}, f)

    gate.close(dump)
    os._exit(0)


# --- open-loop reads + writes process (asyncio) ------------------------------------------------------


def async_proc(cfg: dict, stop, scanning, out_path: str):
    try:
        import resource

        _soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except Exception:
        pass  # keep the soft limit: requests past it fail as connection errors, which no_error_replies reports
    asyncio.run(_async_main(cfg, stop, scanning, out_path))
    os._exit(0)


def timed_pool_class(base, waits: dict):
    """A connection pool that records, per task, how long it waited for a free connection: the time
    in get_connection less the time spent opening a connection (a connect plus a handshake the
    server answers, which is server time, not the driver's)."""

    class TimedPool(base):
        async def get_connection(self, *a, **kw):
            task = asyncio.current_task()
            self._opening = getattr(self, "_opening", {})
            self._opening[task] = 0.0
            t0 = now()
            try:
                conn = await super().get_connection(*a, **kw)
            finally:
                opened = self._opening.pop(task, 0.0)
            waits[task] = max(0.0, now() - t0 - opened)
            return conn

        async def ensure_connection(self, connection):
            t0 = now()
            try:
                await super().ensure_connection(connection)
            finally:
                opening = getattr(self, "_opening", {})
                task = asyncio.current_task()
                if task in opening:
                    opening[task] += now() - t0

    return TimedPool


async def _async_main(cfg, stop, scanning, out_path):
    import redis.asyncio as aredis

    ep, sizes, port = cfg["ep"], cfg["sizes"], cfg["port"]
    ng = len(sizes)
    pool_waits = {}  # task -> seconds it waited for a free connection (the pool is full at max_conns)
    pool = async_pool(
        timed_pool_class(aredis.BlockingConnectionPool, pool_waits), port, cfg["max_conns"]
    )
    r = aredis.Redis(connection_pool=pool)
    reads, writes, errors, died = [], [], {}, []
    inflight = set()
    lag = {
        "late_sends": 0,
        "max_send_lag_ms": 0.0,
        "max_write_wait_s": 0.0,
        "max_pool_wait_s": 0.0,
        "pool_waits_over_50ms": 0,
    }
    wsem = {
        g: asyncio.Semaphore(cfg.get("max_inflight_writes", MAX_INFLIGHT_WRITES)) for g in range(ng)
    }

    async def do(kind, g, op, args, sched, keys=None):
        send = now()
        sl = (send - sched) * 1000
        if kind == "r":  # driver lag; writes also wait for the in-flight cap, reported apart
            if sl > lag["max_send_lag_ms"]:
                lag["max_send_lag_ms"] = sl
            if sl > 50:
                lag["late_sends"] += 1
        elif sl / 1000 > lag["max_write_wait_s"]:
            lag["max_write_wait_s"] = sl / 1000
        # recorded before sending: a request still open when the epoch ends keeps outcome 3 (unknown).
        # The last field is the time spent waiting for a free connection (driver-limited, not the server).
        if kind == "r":
            rec = [g, op, sched, send, None, 3, 0.0]
            reads.append(rec)
        else:
            rec = [g, op, sched, send, None, 3, "", keys, 0.0]
            writes.append(rec)
        try:
            await r.execute_command(*args)
            exc = None
        except Exception as e:
            exc = e
        t = now()
        pw = pool_waits.pop(asyncio.current_task(), 0.0)
        rec[-1] = pw
        lag["max_pool_wait_s"] = max(lag["max_pool_wait_s"], pw)
        if pw > checks.DRIVER_LAG_S:
            lag["pool_waits_over_50ms"] += 1
        oc, msg = outcome_of(exc)
        if msg:
            errors.setdefault(op, msg)
        rec[4], rec[5] = t, oc
        if kind == "w":
            rec[6] = msg

    async def capped(g, coro):
        try:
            await coro
        finally:
            wsem[g].release()

    def spawn(coro):
        task = asyncio.create_task(coro)
        inflight.add(task)
        task.add_done_callback(inflight.discard)

    async def read_stream(g, op, rate):
        rng = random.Random(f"{ep['seed']}-read-{g}-{op}")
        tmo = model.READ_OPS[op][1] * 1000
        nxt = now() + rng.uniform(0, 1 / rate)
        while not stop.is_set():
            d = nxt - now()
            if d > 0:
                await asyncio.sleep(min(d, 0.5))
                continue
            cypher, params = model.make_read(op, sizes[g], rng)
            spawn(do("r", g, op, qargs(gname(g), cypher, params, False, tmo), nxt))
            nxt += rng.expovariate(rate)

    async def write_stream(g, op, rate):
        rng = random.Random(f"{ep['seed']}-write-{g}-{op}")
        nxt = now() + rng.uniform(0, 1 / rate)
        holdoff = ep["write_pattern"] == "holdoff"
        while not stop.is_set():
            d = nxt - now()
            if d > 0:
                await asyncio.sleep(min(d, 0.5))
                continue
            # writes that stay clear of long reads: hold, then send (timed from the send)
            if holdoff:
                while scanning[g] > 0 and not stop.is_set():
                    await asyncio.sleep(0.02)
                if stop.is_set():
                    break
                sched = now()
            else:
                sched = nxt
            if op == "upsert":
                cypher, params, keys = model.make_upsert(sizes[g], rng)
            else:
                (cypher, params), keys = model.make_bulk_set(rng), []
            await wsem[g].acquire()
            spawn(
                capped(g, do("w", g, op, qargs(gname(g), cypher, params, True, None), sched, keys))
            )
            nxt += rng.expovariate(rate)
            if holdoff and nxt < now() - 1.0:  # no burst of the writes held back during a long read
                nxt = now()

    # warm the pool so the first second is not connection setup
    warm = [await pool.get_connection("PING") for _ in range(min(200, cfg["max_conns"]))]
    for c in warm:
        await pool.release(c)
    pool_waits.clear()

    tasks = []
    for g in range(ng):
        for op, (rt, _) in model.READ_OPS.items():
            tasks.append(asyncio.create_task(read_stream(g, op, rt * ep["read_mult"])))
    for g in ep["busy"]:
        for op, rt in model.WRITE_RATE.items():
            tasks.append(asyncio.create_task(write_stream(g, op, rt * ep["write_mult"])))

    while not stop.is_set():
        await asyncio.sleep(0.2)
    for t in tasks:
        t.cancel()
    t_stop = now()
    pending_at_stop = len(inflight)
    if inflight:
        await asyncio.wait(list(inflight), timeout=cfg["drain_s"])
    unfinished = len(inflight)
    drain_s = now() - t_stop
    # write results now: tearing down tens of thousands of open requests can outlast the join
    res = {
        "reads": reads,
        "writes": writes,
        "errors": errors,
        "died": died,
        "pending_at_stop": pending_at_stop,
        "unfinished": unfinished,
        "drain_s": drain_s,
        "harness": lag,
    }
    with open(out_path + ".tmp", "w") as f:
        json.dump(res, f)
    os.replace(out_path + ".tmp", out_path)
    os._exit(0)


# --- building ------------------------------------------------------------------------------------


def build_graph(
    r, graph: str, scale: float, batch: int = model.BUILD_BATCH
) -> tuple[dict, dict, dict]:
    """A fresh graph: labels get consecutive ids, so edges are wired by id. Returns the label sizes,
    the build's timing (wall seconds and query counts of the node phase and the edge phase) and
    each label's first node id."""
    n = model.label_sizes(scale)
    timing = {
        "nodes_s": 0.0,
        "edges_s": 0.0,
        "node_queries": 0,
        "edge_queries": 0,
        "nodes": sum(n.values()),
    }
    r.delete(graph)

    def run(phase, q):
        t0 = time.perf_counter()
        r.execute_command("GRAPH.QUERY", graph, q)
        timing[f"{phase}_s"] += time.perf_counter() - t0
        timing[f"{phase[:-1]}_queries"] += 1

    for phase, _label, q in model.build_plan(n, None, batch):
        run(phase, q)
    base = {}
    for label in n:
        lo_id, hi_id, c = r.execute_command(
            "GRAPH.RO_QUERY", graph, f"MATCH (x:{label}) RETURN min(id(x)), max(id(x)), count(x)"
        )[1][0]
        if hi_id - lo_id + 1 != c or c != n[label]:
            raise RuntimeError(
                f"{graph}: {label} ids not consecutive ({lo_id}..{hi_id}, {c} nodes)"
            )
        base[label] = lo_id
    for phase, _t, q in model.build_plan(n, base, batch):
        if phase == "edges":
            run(phase, q)
    timing = {k: round(v, 3) if isinstance(v, float) else v for k, v in timing.items()}
    return n, timing, base


# --- epoch events ----------------------------------------------------------------------------------


def wait_indexes(r, graph, timeout_s=300, stop=None) -> tuple[bool | None, float]:
    """True once no index is UNDER CONSTRUCTION, False after timeout_s, None if `stop` was set first."""
    t0 = now()
    while now() - t0 < timeout_s:
        rows = r.execute_command("GRAPH.RO_QUERY", graph, "CALL db.indexes()")[1]
        if not any(
            "UNDER CONSTRUCTION" in s(x) for row in rows for x in row if isinstance(x, (bytes, str))
        ):
            return True, now() - t0
        if stop is not None:
            if stop.wait(0.5):
                return None, now() - t0
        else:
            time.sleep(0.5)
    return False, now() - t0


def wait_pending_indexes(r, events, t_start, budget_s=INDEX_WAIT_S):
    """Indexes still building when the load stopped get budget_s more, shared by every graph
    (model.epoch_seconds reserves it once), before they count as stuck."""
    deadline = now() + budget_s
    for e in [x for x in events if x["event"] == "indexes_pending_at_epoch_end"]:
        try:
            ok, took = wait_indexes(r, gname(e["graph"]), max(0.0, deadline - now()))
            events.append(
                {
                    "t": round(now() - t_start, 2),
                    "event": "indexes_ready" if ok else "indexes_stuck",
                    "graph": e["graph"],
                    "seconds": round(e["seconds"] + took, 1),
                    "after_load": True,
                }
            )
        except Exception as exc:
            events.append(
                {"t": round(now() - t_start, 2), "event": "events_died", "msg": str(exc)[:200]}
            )


def constraint_status(r, graph) -> list[str]:
    rows = r.execute_command(
        "GRAPH.RO_QUERY", graph, "CALL db.constraints() YIELD status RETURN status"
    )[1]
    return [s(row[0]) for row in rows]


def idx_create(label, props):
    return f"CREATE INDEX FOR (n:{label}) ON (" + ", ".join(f"n.{p}" for p in props) + ")"


def idx_drop(label, props):
    return f"DROP INDEX FOR (n:{label}) ON (" + ", ".join(f"n.{p}" for p in props) + ")"


def unique_constraint(op, graph, label, props) -> list:
    """GRAPH.CONSTRAINT CREATE|DROP of a unique node constraint."""
    return [
        "GRAPH.CONSTRAINT",
        op,
        graph,
        "UNIQUE",
        "NODE",
        label,
        "PROPERTIES",
        str(len(props)),
        *props,
    ]


def epoch_events(port, epoch_s, ep, ng, stop, events: list, t_start: float, constraint_info: dict):
    """Index creation, constraints, deliberate violations and BGSAVE, during the epoch."""
    r = connect(port, timeout=ADMIN_TIMEOUT_S)
    rng = random.Random(f"{ep['seed']}-events")
    plan = []
    if ep["indexes"]:
        plan.append((rng.uniform(0.05, 0.2) * epoch_s, "indexes"))
    if ep["constraints"]:
        plan.append((rng.uniform(0.1, 0.4) * epoch_s, "constraints"))
    if ep["bgsave"]:
        plan.append((rng.uniform(0.1, 0.8) * epoch_s, "bgsave"))
    plan.sort()

    def ev(kind, **kw):
        events.append({"t": round(now() - t_start, 2), "event": kind, **kw})

    try:
        for at, kind in plan:
            if stop.wait(max(0, t_start + at - now())):
                return
            if kind == "indexes":
                t_created = now()
                for g in range(ng):
                    if stop.is_set():  # no new schema commands once the epoch has ended
                        break
                    for label, props in model.EPOCH_INDEXES:
                        try:
                            r.execute_command("GRAPH.QUERY", gname(g), idx_create(label, props))
                        except ResponseError as e:
                            ev("index_error", graph=g, label=label, msg=str(e)[:200])
                # every graph gets an event; once the epoch ends the rest are pending
                for g in range(ng):
                    ok, took = (
                        (None, 0.0) if stop.is_set() else wait_indexes(r, gname(g), stop=stop)
                    )
                    kind_ = {
                        True: "indexes_ready",
                        False: "indexes_stuck",
                        None: "indexes_pending_at_epoch_end",
                    }[ok]
                    ev(kind_, graph=g, seconds=round(now() - t_created if ok is None else took, 1))
            elif kind == "constraints":
                created = []  # the graphs the constraints were created on (the epoch may end first)
                for g in range(ng):
                    if stop.is_set():
                        break
                    created.append(g)
                    for label, props in model.CONSTRAINTS:
                        try:
                            r.execute_command("GRAPH.QUERY", gname(g), idx_create(label, props))
                            r.execute_command(*unique_constraint("CREATE", gname(g), label, props))
                        except ResponseError as e:
                            ev("constraint_error", graph=g, label=label, msg=str(e)[:200])
                t0 = now()
                states = {}
                while now() - t0 < 180:
                    states = {g: constraint_status(r, gname(g)) for g in created}
                    if all(
                        st and all(x != "UNDER CONSTRUCTION" for x in st) for st in states.values()
                    ):
                        break
                    if stop.wait(1):
                        break
                took = now() - t0
                constraint_info["states"] = {str(g): st for g, st in states.items()}
                constraint_info["seconds"] = round(took, 1)
                ev(
                    "constraints_state",
                    seconds=round(took, 1),
                    states={str(g): st for g, st in states.items()},
                )
                if stop.is_set() and any("UNDER CONSTRUCTION" in st for st in states.values()):
                    # run_epoch waits for them after the load
                    constraint_info["pending_at_epoch_end"] = True
                    return
                operational = {
                    g for g, st in states.items() if st and all(x == "OPERATIONAL" for x in st)
                }
                # deliberate violations on operational graphs: existing keys, must be rejected
                label = model.CONSTRAINTS[0][0]
                for g in sorted(operational):
                    if stop.is_set():
                        break
                    for k in range(3):
                        try:
                            r.execute_command(
                                "GRAPH.QUERY",
                                gname(g),
                                f"CREATE (:{label} {{k: {k}, g: {k % model.GROUPS}}})",
                            )
                            constraint_info.setdefault("violations_accepted", []).append([g, k])
                            ev("violation_accepted", graph=g, key=k)
                        except ResponseError as e:
                            constraint_info.setdefault("violations_rejected", 0)
                            constraint_info["violations_rejected"] += 1
                            if "constraint" not in str(e).lower():
                                ev("violation_odd_error", graph=g, msg=str(e)[:200])
            elif kind == "bgsave":
                try:
                    r.execute_command("BGSAVE")
                    t0 = now()
                    info = {}
                    while now() - t0 < 300:
                        info = r.info("persistence")
                        if not info.get("rdb_bgsave_in_progress") or stop.wait(0.5):
                            break
                    ev(
                        "bgsave_done",
                        seconds=round(now() - t0, 1),
                        status=info.get("rdb_last_bgsave_status"),
                    )
                except Exception as e:
                    ev("bgsave_error", msg=str(e)[:200])
    except Exception as e:
        ev("events_died", msg=f"{type(e).__name__}: {e}"[:300])
    finally:
        r.close()


def end_of_epoch_cleanup(r, ep, ng, notes):
    for g in range(ng):
        if ep["constraints"]:
            for label, props in model.CONSTRAINTS:
                try:
                    r.execute_command(*unique_constraint("DROP", gname(g), label, props))
                except ResponseError as e:
                    notes.append(f"g{g} drop constraint {label}: {str(e)[:120]}")
                drop_index(r, g, label, props, notes)
        if ep["indexes"]:
            for label, props in model.EPOCH_INDEXES:
                drop_index(r, g, label, props, notes)


def drop_index(r, g, label, props, notes):
    """C's DROP INDEX with several properties drops only the first one: drop them one by one."""
    for p_ in props:
        try:
            r.execute_command("GRAPH.QUERY", gname(g), idx_drop(label, (p_,)))
        except ResponseError as e:
            notes.append(f"g{g} drop index {label}.{p_}: {str(e)[:120]}")


def index_vs_scan(r, ng, sizes, seed) -> list[str]:
    rng = random.Random(f"{seed}-ivs")
    problems = []
    for g in range(ng):
        for label, prop in model.INDEX_PROBES:
            for _ in range(5):
                v = rng.randrange(sizes[g][label])
                a = r.execute_command(
                    "GRAPH.RO_QUERY", gname(g), f"MATCH (n:{label} {{{prop}: {v}}}) RETURN count(n)"
                )[1][0][0]
                b = r.execute_command(
                    "GRAPH.RO_QUERY",
                    gname(g),
                    f"MATCH (n:{label}) WHERE n.{prop} + 0 = {v} RETURN count(n)",
                )[1][0][0]
                if a != b:
                    problems.append(f"g{g} {label}.{prop}={v}: index {a}, scan {b}")
    return problems


def calibrate(r, ng, sizes, seed) -> list[dict]:
    """Each cheap read's cost alone (median of 7, ms), per graph, before any load."""
    rng = random.Random(f"{seed}-cal")
    cal = []
    for g in range(ng):
        c = {}
        for op in model.READ_OPS:
            ts_ = []
            for _ in range(7):
                cypher, params = model.make_read(op, sizes[g], rng)
                t0 = now()
                r.execute_command(*qargs(gname(g), cypher, params, False, None))
                ts_.append((now() - t0) * 1000)
            c[op] = round(statistics.median(ts_), 2)
        cal.append(c)
    return cal


# --- running one epoch ----------------------------------------------------------------------------


def driver_results(edir) -> tuple[str, str]:
    """Where the long-read and the async driver write their results; a file an earlier attempt
    left there is removed, so a driver that writes nothing is never read as this epoch's."""
    paths = os.path.join(edir, "long.json"), os.path.join(edir, "async.json")
    for p in paths:
        with contextlib.suppress(FileNotFoundError):
            os.remove(p)
    return paths


def run_epoch(
    opts: dict,
    server: Server,
    hp,
    ep,
    sizes,
    cal,
    ledger: checks.Ledger,
    ping_path,
    hang_path,
    hdir,
) -> dict:
    port = server.port
    r = connect(port, timeout=ADMIN_TIMEOUT_S)
    ng = hp["graphs"]
    edir = os.path.join(hdir, f"e{ep['epoch']}")
    os.makedirs(edir, exist_ok=True)
    log(f"epoch {ep['seed']}: {json.dumps(ep)}")
    if ep["write_timeout_ms"]:
        r.execute_command("GRAPH.CONFIG", "SET", "TIMEOUT_MAX", "10000")
        r.execute_command("GRAPH.CONFIG", "SET", "TIMEOUT_DEFAULT", str(ep["write_timeout_ms"]))
    stop = mp.Event()
    scanning = mp.Array("i", ng)
    base_cfg = {
        "ep": ep,
        "sizes": sizes,
        "port": port,
        "drain_s": opts["drain_s"],
        "max_conns": opts["max_conns"],
        "max_inflight_writes": ep.get("max_inflight_writes", MAX_INFLIGHT_WRITES),
    }
    lp, ap = driver_results(edir)
    p_long = mp.Process(target=long_proc, args=(base_cfg, stop, scanning, lp), daemon=True)
    p_async = mp.Process(target=async_proc, args=(base_cfg, stop, scanning, ap), daemon=True)
    t_start = now()
    p_long.start()
    p_async.start()
    events, constraint_info = [], {}
    ev_stop = threading.Event()
    ev_thread = threading.Thread(
        target=epoch_events,
        args=(port, opts["epoch_s"], ep, ng, ev_stop, events, t_start, constraint_info),
        daemon=True,
    )
    ev_thread.start()
    crash = hang = None
    try:
        while now() - t_start < opts["epoch_s"]:
            time.sleep(1)
            running, code = server.state()
            if not running:
                crash = {"exit_code": code, "t": round(now() - t_start, 1)}
                break
            if os.path.exists(hang_path):
                hang = {"t": round(now() - t_start, 1)}
                break
            if not p_async.is_alive() or not p_long.is_alive():
                log("driver process died early")
                break
    except BaseException:
        # stopped (SIGTERM) or docker failed: the drivers must not outlive the epoch
        stop.set()
        ev_stop.set()
        for p in (p_async, p_long):
            p.kill()
        raise
    stop.set()
    ev_stop.set()
    t_stopped = now()
    if hang:
        hang["backtrace_file"] = grab_backtrace(server, edir)
        docker("kill", server.name)
    p_async.join(opts["drain_s"] + 180)
    p_long.join(opts["drain_s"] + 60)
    for p in (p_async, p_long):
        if p.is_alive():
            p.kill()
    ev_thread.join(60)
    reusable = True
    if ev_thread.is_alive():  # still inside a schema command: the server must not run another epoch
        events.append({"t": round(now() - t_start, 2), "event": "events_stuck"})
        reusable = False
    # a constraint still building when the load stopped gets 300 s more, then its state is judged
    if constraint_info.get("pending_at_epoch_end"):
        try:
            t0 = now()
            while now() - t0 < 300:
                states = {g: constraint_status(r, gname(int(g))) for g in constraint_info["states"]}
                if all("UNDER CONSTRUCTION" not in st for st in states.values()):
                    break
                time.sleep(1)
            constraint_info["states"] = states
            events.append(
                {
                    "t": round(now() - t_start, 2),
                    "event": "constraints_state",
                    "states": states,
                    "seconds": round(now() - t0, 1),
                    "after_load": True,
                }
            )
        except Exception as exc:
            events.append(
                {"t": round(now() - t_start, 2), "event": "events_died", "msg": str(exc)[:200]}
            )
    wait_pending_indexes(r, events, t_start)
    t_end = now()
    data = {
        "reads": [],
        "writes": [],
        "longs": [],
        "errors": {},
        "died": [],
        "unfinished": 0,
        "pending_at_stop": 0,
        "drain_s": 0.0,
        "harness": {},
        "window_s": t_end - t_start,
    }
    for path, keys in (
        (
            ap,
            (
                "reads",
                "writes",
                "errors",
                "died",
                "unfinished",
                "pending_at_stop",
                "drain_s",
                "harness",
            ),
        ),
        (lp, ("longs", "writes", "died", "unfinished")),
    ):
        try:
            with open(path) as f:
                d = json.load(f)
            for k in keys:
                if k in ("writes", "died"):
                    data[k] += d.get(k, [])
                elif k == "unfinished":
                    data[k] += d.get(k, 0)
                else:
                    data[k] = d.get(k, data[k])
        except (FileNotFoundError, json.JSONDecodeError) as e:
            data["died"].append(f"no results from {os.path.basename(path)}: {e}")
            ledger.incomplete = True
    for w in data["writes"]:
        ledger.add(w[0], w[7], w[5])
    for g, _key in constraint_info.get("violations_accepted", []):
        ledger.extra[g][model.CONSTRAINTS[0][0]] += 1
    pings, downs = checks.read_pings(ping_path, t_start, t_end)

    result = {
        "params": ep,
        "hour": hp,
        "calibration": cal,
        "events": events,
        "duration_s": round(t_stopped - t_start, 1),
    }
    if crash or hang:
        result["crash"] = crash
        result["hang"] = hang
        result["checks"] = {"server_stays_up": {"pass": False, **(crash or {}), "hang": hang}}
        an = checks.analyse(ep, hp, cal, data, pings, downs, events, constraint_info, [])
        result["checks"].update({k: v for k, v in an["checks"].items() if k in ("no_freeze",)})
        result["info"] = an["info"]
    else:
        end = end_of_epoch(r, ep, ng, sizes, ledger, events, t_start)
        ledger_problems, ivs, notes = end["ledger_problems"], end["ivs"], end["notes"]
        reusable = reusable and end["reusable"]
        if end["memory"]:
            result["memory"] = end["memory"]
        an = checks.analyse(
            ep, hp, cal, data, pings, downs, events, constraint_info, ledger_problems
        )
        running, code = server.state()
        # a server that stopped answering after the load (the prober's marker, or no PONG now) did not stay up
        hung = running and (os.path.exists(hang_path) or server_hung(port))
        an["checks"]["server_stays_up"] = {
            "pass": running and not hung,
            "exit_code": None if running else code,
            "hang": hung,
        }
        if ledger.incomplete:
            for k in ("writes_add_up", "upsert_edges_complete", "timed_out_writes_leave_nothing"):
                an["checks"][k] = {
                    "pass": None,
                    "skipped": "a driver result file was lost earlier on this server",
                }
        an["checks"]["index_matches_scan"] = index_check(ep, ivs)
        result.update(an)
        result["notes"] = notes
    result["reusable"] = reusable
    result["failed"] = sorted(
        k for k, v in result["checks"].items() if v.get("pass", True) is False
    )
    with open(os.path.join(edir, "epoch.json"), "w") as f:
        json.dump(result, f, indent=1, default=str)
    with gzip.open(os.path.join(edir, "raw.json.gz"), "wt") as f:
        json.dump({"reads": data["reads"], "writes": data["writes"], "longs": data["longs"]}, f)
    for p in (lp, ap):
        # that driver never wrote its file (already reported as "no results from ...")
        with contextlib.suppress(FileNotFoundError):
            os.remove(p)
    log(
        f"epoch {ep['seed']} failed={result['failed']} "
        f"worst_ping={result['checks'].get('no_freeze', {}).get('worst_ping_ms')}ms "
        f"read_timeouts={result['checks'].get('no_timeouts_while_waiting', {}).get('all_read_timeouts')}"
    )
    return result


def schema_left(r, ng) -> dict:
    """Indexes and constraints still on each graph ({graph: count}, non-zero only). The built graphs
    have none, so after an epoch's clean-up there should be none either."""
    left = {}
    for g in range(ng):
        n = len(r.execute_command("GRAPH.RO_QUERY", gname(g), "CALL db.indexes()")[1]) + len(
            r.execute_command("GRAPH.RO_QUERY", gname(g), "CALL db.constraints()")[1]
        )
        if n:
            left[g] = n
    return left


def index_check(ep, ivs) -> dict:
    """index_matches_scan from index_vs_scan's problems; None (the comparison failed) is unmeasured."""
    return {
        "pass": None if ivs is None else not ivs,
        "applies": ep["indexes"] or ep["constraints"],
        "problems": ivs or [],
    }


def end_of_epoch(r, ep, ng, sizes, ledger, events, t_start) -> dict:
    """After the load: count what the graphs hold (the ledger), compare index lookups with scans,
    then put the schema back. The clean-up runs even when the counting failed, and `reusable` is
    False when the schema could not be put back, so the server-hour stops instead of running an
    epoch on a schema it did not ask for."""
    out = {"ledger_problems": [], "ivs": [], "notes": [], "memory": None, "reusable": True}
    try:
        r.execute_command("GRAPH.CONFIG", "SET", "TIMEOUT_DEFAULT", "0")
        r.execute_command("GRAPH.CONFIG", "SET", "TIMEOUT_MAX", "0")
        out["ledger_problems"] = ledger.check(r)
    except Exception as e:
        out["ledger_problems"] = [
            ("harness", f"end-of-epoch checks failed: {type(e).__name__}: {e}")
        ]
        out["notes"].append(traceback.format_exc()[-500:])
    if ep["indexes"] or ep["constraints"]:
        try:
            out["ivs"] = index_vs_scan(r, ng, sizes, ep["seed"])
        except Exception as e:  # not compared: index_matches_scan is not measured (None)
            out["ivs"] = None
            out["notes"].append(f"index comparison failed: {type(e).__name__}: {e}"[:300])
    try:
        end_of_epoch_cleanup(r, ep, ng, out["notes"])
        left = schema_left(r, ng)
    except Exception as e:
        left = {"error": f"{type(e).__name__}: {e}"[:200]}
    if left:
        out["reusable"] = False
        events.append({"t": round(now() - t_start, 2), "event": "cleanup_incomplete", "left": left})
    try:
        mem = r.info("memory")
        out["memory"] = {
            "used_memory": mem.get("used_memory"),
            "used_memory_rss": mem.get("used_memory_rss"),
        }
    except Exception as e:
        out["notes"].append(f"no memory figure: {type(e).__name__}: {e}"[:200])
    return out


def grab_backtrace(server: Server, edir: str) -> str:
    path = os.path.join(edir, "hang-backtrace.txt")
    try:
        pid = server.pid()
        with open(path, "w") as f:
            subprocess.run(
                ["gdb", "-p", str(pid), "-batch", "-ex", "thread apply all bt"],
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
    except Exception as e:
        with open(path, "a") as f:
            f.write(f"\nbacktrace failed: {e}\n")
    return os.path.basename(path)


# --- server-hour --------------------------------------------------------------------------------


def run_server_hour(
    opts: dict,
    info: dict,
    tag: str,
    hp: dict,
    epochs: list[dict],
    hdir: str,
    deadline: float | None = None,
) -> dict:
    """Start a server and its plain PING control from the image, build the hour's graphs, run its
    epochs. Returns {"hour", "build": [...], "epochs": [...], "module_version", "config"}, plus
    "error" and "server_alive" when something failed before the epochs finished (a server that
    would not start, died during the build, or a harness fault)."""
    os.makedirs(hdir, exist_ok=True)
    log(f"{tag}: server-hour {hp}")
    server = Server(
        info,
        f"{opts['name_prefix']}-{tag}",
        opts.get("server_cpus"),
        opts["net"],
        hp["thread_count"],
        labels=opts.get("labels"),
    )
    control = Server(
        info,
        f"{opts['name_prefix']}-{tag}-plain",
        opts.get("client_cpus"),
        opts["net"],
        module=False,
        labels=opts.get("labels"),
    )
    out = {"hour": hp, "tag": tag, "build": [], "epochs": []}
    prober = None
    stop = mp.Event()
    ping_path = os.path.join(hdir, "pings.csv")
    hang_path = os.path.join(hdir, "HANG")
    # a marker left by an earlier run in this directory is not this server's
    if os.path.exists(hang_path):
        os.remove(hang_path)
    try:
        r = server.start()
        control.start()
        out["module_version"] = module_version(r)
        out["config"] = server_config(r)
        prober = mp.Process(
            target=prober_proc,
            args=({"falkor": server.port, "control": control.port}, ping_path, hang_path, stop),
            daemon=True,
        )
        prober.start()
        sizes = []
        for g, sc in enumerate(hp["scales"]):
            n, timing, _base = build_graph(r, gname(g), sc)
            sizes.append(n)
            out["build"].append({"graph": g, "scale": sc, **timing})
        cal = calibrate(r, hp["graphs"], sizes, f"{hp.get('seed', 'soak')}-h{hp['hour']}")
        out["calibration"] = cal
        out["used_memory_after_build"] = r.info("memory").get("used_memory")
        log(f"{tag}: built {out['build']}, cal={cal}")
        ledger = checks.Ledger(sizes)
        for ep in epochs:
            if (
                deadline
                and time.time() + model.epoch_seconds(opts["epoch_s"], opts["drain_s"], ep)
                > deadline
            ):
                log("deadline: no time for another epoch")
                break
            res = run_epoch(opts, server, hp, ep, sizes, cal, ledger, ping_path, hang_path, hdir)
            out["epochs"].append(res)
            if res.get("crash") or res.get("hang") or not res.get("reusable", True):
                break
        res_list = out["epochs"]
        mems = [x.get("memory", {}).get("used_memory") for x in res_list if x.get("memory")]
        if len(mems) >= 2 and mems[0]:
            growth = (mems[-1] - mems[0]) / mems[0]
            res_list[-1]["checks"]["memory_steady"] = {
                "pass": growth < 0.10,
                "growth": round(growth, 4),
                "used_memory_by_epoch": mems,
            }
        for x in res_list:
            x["failed"] = sorted(
                k for k, v in x.get("checks", {}).items() if v.get("pass", True) is False
            )
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:500]
        out.update(after_failure(server, e, hang_path))
        log(f"{tag}: {out['error']} (server alive: {out['server_alive']})")
        with open(os.path.join(hdir, "error.txt"), "w") as f:
            f.write(traceback.format_exc())
    finally:
        stop.set()
        if prober is not None:
            prober.join(5)
            if prober.is_alive():
                prober.kill()
        tail = server.stop()  # first: both servers go even if the log cannot be written
        control.stop()
        with open(os.path.join(hdir, "server-tail.log"), "w") as f:
            f.write(tail)
    with open(os.path.join(hdir, "hour.json"), "w") as f:
        json.dump({k: v for k, v in out.items() if k != "epochs"}, f, indent=1, default=str)
    return out


def set_start_method():
    if sys.platform != "win32":
        with contextlib.suppress(RuntimeError):  # already set by an earlier call in this process
            mp.set_start_method("fork")
