"""Drives one server: builds the graphs, runs epochs of long reads, writes and cheap reads, and
records everything `checks.analyse` judges.

Per server-hour: a fresh server from the image (THREAD_COUNT given), a plain server from the same
image without the module beside it (the PING control), 1-4 graphs at their own scales. Per epoch:
busy graphs, long-read streams, write and read rates, write pattern, indexes, constraints, write
timeouts and BGSAVE, all from the epoch's parameters (`model.epoch_params`).

Clients never re-send (no socket timeout, no retries). Reads and writes are open loop: sent at
their scheduled time whether or not earlier ones have returned, and timed from that schedule. A
separate process PINGs FalkorDB and the plain server every 10 ms, timed from the scheduled send.
"""

from __future__ import annotations

import asyncio
import collections
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
from redis.backoff import NoBackoff
from redis.exceptions import ResponseError
from redis.retry import Retry

from . import checks, model
from .checks import gname

MODULE_PATH = "/var/lib/falkordb/bin/falkordb.so"
PING_INTERVAL_S = checks.PING_INTERVAL_S
HANG_S = 60  # a FalkorDB PING unanswered this long: the server is hung
MAX_INFLIGHT_WRITES = 900  # per graph; C crashes when a graph's 1,024-entry write queue overflows


def now() -> float:
    return time.monotonic()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def s(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def connect(port: int, host="127.0.0.1", timeout: float | None = None) -> redis.Redis:
    """No retries. Measured requests use no socket timeout (a latency is one request, however long);
    setup and checking connections pass one, so a hung server cannot block the run forever."""
    return redis.Redis(host=host, port=port, socket_timeout=timeout, socket_connect_timeout=10,
                       retry=Retry(NoBackoff(), 0))


ADMIN_TIMEOUT_S = 300


class DockerError(RuntimeError):
    """Docker itself failed or timed out (pull, run, inspect): nothing about the engine was learnt."""


#: set by the check's SIGTERM handler: containers are removed at once, without a graceful stop
STOPPING = False


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


def cypher_params(params: dict) -> str:
    def lit(v):
        if v is None:
            return "null"
        if isinstance(v, str):
            return "'" + v.replace("\\", "\\\\").replace("'", "\\'") + "'"
        return str(v)
    return "CYPHER " + " ".join(f"{k}={lit(v)}" for k, v in params.items()) + " " if params else ""


def qargs(graph, cypher, params, write, timeout_ms):
    a = ["GRAPH.QUERY" if write else "GRAPH.RO_QUERY", graph, cypher_params(params) + cypher]
    if timeout_ms:
        a += ["TIMEOUT", str(int(timeout_ms))]
    return a


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# --- images and servers ----------------------------------------------------------------------------


def docker(*a, check=True, timeout=120):
    try:
        return subprocess.run(["docker", *a], check=check, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise DockerError(f"docker {a[0]} did not return within {timeout} s") from None


def probe_image(image: str) -> dict:
    """Pull the image and find its server and module. The server is the image's own default, as its
    run.sh picks it: `${FALKORDB_SERVER:-redis}-server` from the image's config env (empty counts as
    unset), with no fallback to the other binary. Raises RuntimeError if either is missing."""
    p = docker("pull", "-q", image, check=False, timeout=900)
    if p.returncode != 0:
        raise DockerError(f"docker pull failed: {(p.stderr or p.stdout).strip()[:300]}")
    p = docker("image", "inspect", "--format", "{{json .Config.Env}}", image, check=False, timeout=60)
    if p.returncode != 0:
        raise DockerError(f"docker image inspect failed: {(p.stderr or p.stdout).strip()[:300]}")
    env = dict(x.split("=", 1) for x in (json.loads(p.stdout or "null") or []) if "=" in x)
    flavour = env.get("FALKORDB_SERVER") or None
    binary = f"{flavour or 'redis'}-server"
    script = (f"command -v {binary}; "
              f"if [ -f {MODULE_PATH} ]; then echo {MODULE_PATH}; else find / -name falkordb.so 2>/dev/null | head -1; fi")
    p = docker("run", "--rm", "--entrypoint", "sh", image, "-c", script, check=False, timeout=300)
    lines = [x.strip() for x in p.stdout.splitlines() if x.strip()]
    server = next((x for x in lines if x.endswith("/" + binary)), None)
    module = next((x for x in lines if x.endswith("falkordb.so")), None)
    if not server:
        raise RuntimeError(f"image {image} has no {binary} (FALKORDB_SERVER={flavour or 'unset'})")
    if not module:
        raise RuntimeError(f"image {image} has no falkordb.so")
    return {"image": image, "server": server, "module": module}


def after_failure(server, exc, hang_path: str | None = None) -> dict:
    """What a failed step says about the engine: server_alive False (it exited, or it is hung: the
    prober's marker or no PONG to 3 PINGs), True, or None when Docker failed and nothing is known."""
    out = {"server_alive": None, "exit_code": None, "hung": False}
    if isinstance(exc, DockerError):
        return out
    try:
        running, code = server.state()
        hung = running and ((hang_path is not None and os.path.exists(hang_path)) or server_hung(server.port))
        out.update(server_alive=running and not hung, exit_code=None if running else code, hung=hung)
    except DockerError:
        pass  # docker cannot say either: nothing is known about the engine
    return out


class Server:
    """One container from the image: its own server binary, with the module (FalkorDB) or without
    (the plain server that tells a host stall from a FalkorDB freeze)."""

    def __init__(self, info: dict, name: str, cpus: str | None, net: str, thread_count: int | None = None,
                 module: bool = True, module_args: tuple = ()):
        self.info, self.name, self.cpus, self.net = info, name, cpus, net
        self.tc, self.module, self.module_args = thread_count, module, module_args
        self.port = free_port()

    def start(self, wait_s: float = 60) -> redis.Redis:
        docker("rm", "-f", self.name, check=False)
        net = ["--network", "host"] if self.net == "host" else ["-p", f"127.0.0.1:{self.port}:{self.port}"]
        cpus = ["--cpuset-cpus", self.cpus] if self.cpus else []
        cmd = ["run", "-d", "--name", self.name, *net, *cpus, "--ulimit", "nofile=1048576",
               "--entrypoint", self.info["server"], self.info["image"],
               "--port", str(self.port), "--bind", "127.0.0.1" if self.net == "host" else "0.0.0.0", "--maxclients", "200000", "--save", "",
               "--appendonly", "no", "--dir", "/tmp", "--protected-mode", "no"]
        if self.module:
            cmd += ["--loadmodule", self.info["module"], "THREAD_COUNT", str(self.tc), "CACHE_SIZE", "200",
                    "EFFECTS_THRESHOLD", "0", *self.module_args]
        p = docker(*cmd, check=False)
        if p.returncode != 0:
            raise DockerError(f"docker run failed: {p.stderr.strip()[:300]}")
        r = connect(self.port, timeout=ADMIN_TIMEOUT_S)
        t0 = now()
        while now() - t0 < wait_s:
            try:
                r.ping()
                return r
            except Exception:
                running, _ = self.state()
                if not running:
                    break
                time.sleep(0.1)
        raise RuntimeError("server did not start: " + self.logs(50)[-1500:])

    def state(self) -> tuple[bool, int]:
        """(running, exit code). A container Docker no longer knows is (False, -1); any other failure
        of `docker inspect` (its daemon unreachable) raises DockerError, never reads as a crash."""
        p = docker("inspect", "-f", "{{.State.Running}} {{.State.ExitCode}}", self.name, check=False)
        if p.returncode != 0:
            if any(s_ in (p.stderr + p.stdout).lower() for s_ in ("no such object", "no such container")):
                return False, -1
            raise DockerError(f"docker inspect failed: {(p.stderr or p.stdout).strip()[:200]}")
        run, code = p.stdout.split()
        return run == "true", int(code)

    def pid(self) -> int:
        return int(docker("inspect", "-f", "{{.State.Pid}}", self.name).stdout.strip())

    def logs(self, tail=400) -> str:
        p = docker("logs", "--tail", str(tail), self.name, check=False)
        return (p.stdout or "") + (p.stderr or "")

    def stop(self) -> str:
        """Stop and remove the container; returns its log tail. Best effort: never raises."""
        tail = ""
        try:
            if not STOPPING:
                docker("stop", "-t", "20", self.name, check=False, timeout=60)
            tail = self.logs(200)
        except DockerError as e:
            tail = f"(no log: {e})"
        try:
            docker("rm", "-f", self.name, check=False)
        except DockerError:
            pass  # best effort; the check's final clean-up removes what is left
        return tail


def module_version(r) -> str:
    ver = "?"
    for m in r.execute_command("MODULE", "LIST"):
        d = dict(zip(map(s, m[::2]), m[1::2])) if isinstance(m, list) else {s(k): v for k, v in m.items()}
        if s(d.get("name", "")) in ("graph", "falkordb"):
            ver = s(d.get("ver"))
    return ver


def server_config(r) -> dict:
    try:
        flat = [s(x) for p in r.execute_command("GRAPH.CONFIG", "GET", "*") for x in p]
        return dict(zip(flat[::2], flat[1::2]))
    except Exception as e:
        return {"error": str(e)[:200]}


# --- PING prober process -------------------------------------------------------------------------


def prober_proc(ports: dict, out_path: str, hang_path: str, stop):
    asyncio.run(_prober(ports, out_path, hang_path, stop))


async def _prober(ports, out_path, hang_path, stop):
    f = open(out_path, "a", buffering=1)  # line-buffered: an epoch reads the file while this writes

    async def one(name, port):
        while not stop.is_set():
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
            except OSError:
                f.write(f"{name},{now():.4f},down\n")
                await asyncio.sleep(1)
                continue
            pending = collections.deque()
            alive = [True]

            async def read_loop():
                try:
                    while True:
                        line = await reader.readline()
                        if not line:
                            break
                        t = now()
                        if pending:
                            sched = pending.popleft()
                            f.write(f"{name},{sched:.4f},{(t - sched) * 1000:.2f}\n")
                finally:
                    alive[0] = False

            rt = asyncio.create_task(read_loop())
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
                try:
                    writer.close()
                except Exception:
                    pass  # the transport is already broken: nothing left to close
            if not stop.is_set():
                f.write(f"{name},{now():.4f},down\n")
                await asyncio.sleep(1)

    await asyncio.gather(*(one(n, p) for n, p in ports.items()))
    f.close()


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

    def close(self, write):
        """Stop admitting writes and call write(records) under the lock (no record changes meanwhile)."""
        with self.lock:
            self.closed = True
            write(self.records)


def long_proc(cfg: dict, stop, scanning, out_path: str):
    ep, sizes, port = cfg["ep"], cfg["sizes"], cfg["port"]
    longs, died = [], []
    lock = threading.Lock()
    gate = SnapshotGate()  # the aligned writes: their timer threads are not joined

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
                        threading.Timer(0.001, aligned_write, args=(g, random.Random(rng.random()))).start()
                    for op, cypher, params in batch:
                        if stop.is_set():
                            break
                        t0 = now()
                        try:
                            r.execute_command(*qargs(gname(g), cypher, params, False,
                                                     model.LONG_TIMEOUT[op] * 1000))
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
    unfinished = sum(t.is_alive() for t in threads)

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
    pool = timed_pool_class(aredis.BlockingConnectionPool, pool_waits)(host="127.0.0.1", port=port, max_connections=cfg["max_conns"], timeout=None,
                     socket_timeout=None, socket_connect_timeout=10, retry=Retry(NoBackoff(), 0))
    r = aredis.Redis(connection_pool=pool)
    reads, writes, errors, died = [], [], {}, []
    inflight = set()
    lag = {"late_sends": 0, "max_send_lag_ms": 0.0, "max_write_wait_s": 0.0, "max_pool_wait_s": 0.0,
           "pool_waits_over_50ms": 0}
    wsem = {g: asyncio.Semaphore(cfg.get("max_inflight_writes", MAX_INFLIGHT_WRITES)) for g in range(ng)}

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
            if holdoff:  # writes that stay clear of long reads: hold, then send (timed from the send)
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
            spawn(capped(g, do("w", g, op, qargs(gname(g), cypher, params, True, None), sched, keys)))
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
    res = {"reads": reads, "writes": writes, "errors": errors, "died": died,
           "pending_at_stop": pending_at_stop, "unfinished": unfinished, "drain_s": drain_s,
           "harness": lag}
    with open(out_path + ".tmp", "w") as f:
        json.dump(res, f)
    os.replace(out_path + ".tmp", out_path)
    os._exit(0)


# --- building ------------------------------------------------------------------------------------


def build_graph(r, graph: str, scale: float, batch: int = model.BUILD_BATCH) -> tuple[dict, dict, dict]:
    """A fresh graph: labels get consecutive ids, so edges are wired by id. Returns the label sizes,
    the build's timing (wall seconds and query counts of the node phase and the edge phase) and
    each label's first node id."""
    n = model.label_sizes(scale)
    timing = {"nodes_s": 0.0, "edges_s": 0.0, "node_queries": 0, "edge_queries": 0, "nodes": sum(n.values())}
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
        lo_id, hi_id, c = r.execute_command("GRAPH.RO_QUERY", graph,
                                            f"MATCH (x:{label}) RETURN min(id(x)), max(id(x)), count(x)")[1][0]
        if hi_id - lo_id + 1 != c or c != n[label]:
            raise RuntimeError(f"{graph}: {label} ids not consecutive ({lo_id}..{hi_id}, {c} nodes)")
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
        if not any("UNDER CONSTRUCTION" in s(x) for row in rows for x in row if isinstance(x, (bytes, str))):
            return True, now() - t0
        if stop is not None:
            if stop.wait(0.5):
                return None, now() - t0
        else:
            time.sleep(0.5)
    return False, now() - t0


def constraint_status(r, graph) -> list[str]:
    rows = r.execute_command("GRAPH.RO_QUERY", graph, "CALL db.constraints() YIELD status RETURN status")[1]
    return [s(row[0]) for row in rows]


def idx_create(label, props):
    return f"CREATE INDEX FOR (n:{label}) ON (" + ", ".join(f"n.{p}" for p in props) + ")"


def idx_drop(label, props):
    return f"DROP INDEX FOR (n:{label}) ON (" + ", ".join(f"n.{p}" for p in props) + ")"


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
                for g in range(ng):  # every graph gets an event; once the epoch ends the rest are pending
                    ok, took = (None, 0.0) if stop.is_set() else wait_indexes(r, gname(g), stop=stop)
                    kind_ = {True: "indexes_ready", False: "indexes_stuck", None: "indexes_pending_at_epoch_end"}[ok]
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
                            r.execute_command("GRAPH.CONSTRAINT", "CREATE", gname(g), "UNIQUE", "NODE", label,
                                              "PROPERTIES", str(len(props)), *props)
                        except ResponseError as e:
                            ev("constraint_error", graph=g, label=label, msg=str(e)[:200])
                t0 = now()
                states = {}
                while now() - t0 < 180:
                    states = {g: constraint_status(r, gname(g)) for g in created}
                    if all(st and all(x != "UNDER CONSTRUCTION" for x in st) for st in states.values()):
                        break
                    if stop.wait(1):
                        break
                took = now() - t0
                constraint_info["states"] = {str(g): st for g, st in states.items()}
                constraint_info["seconds"] = round(took, 1)
                ev("constraints_state", seconds=round(took, 1), states={str(g): st for g, st in states.items()})
                if stop.is_set() and any("UNDER CONSTRUCTION" in st for st in states.values()):
                    constraint_info["pending_at_epoch_end"] = True  # run_epoch waits for them after the load
                    return
                operational = {g for g, st in states.items() if st and all(x == "OPERATIONAL" for x in st)}
                # deliberate violations on operational graphs: existing keys, must be rejected
                label = model.CONSTRAINTS[0][0]
                for g in sorted(operational):
                    if stop.is_set():
                        break
                    for k in range(3):
                        try:
                            r.execute_command("GRAPH.QUERY", gname(g),
                                              f"CREATE (:{label} {{k: {k}, g: {k % model.GROUPS}}})")
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
                    ev("bgsave_done", seconds=round(now() - t0, 1), status=info.get("rdb_last_bgsave_status"))
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
                    r.execute_command("GRAPH.CONSTRAINT", "DROP", gname(g), "UNIQUE", "NODE", label,
                                      "PROPERTIES", str(len(props)), *props)
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
                a = r.execute_command("GRAPH.RO_QUERY", gname(g), f"MATCH (n:{label} {{{prop}: {v}}}) RETURN count(n)")[1][0][0]
                b = r.execute_command("GRAPH.RO_QUERY", gname(g), f"MATCH (n:{label}) WHERE n.{prop} + 0 = {v} RETURN count(n)")[1][0][0]
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


def run_epoch(opts: dict, server: Server, hp, ep, sizes, cal, ledger: checks.Ledger, ping_path, hang_path,
              hdir) -> dict:
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
    base_cfg = {"ep": ep, "sizes": sizes, "port": port, "drain_s": opts["drain_s"],
                "max_conns": opts["max_conns"], "max_inflight_writes": ep.get("max_inflight_writes", MAX_INFLIGHT_WRITES)}
    lp, ap = os.path.join(edir, "long.json"), os.path.join(edir, "async.json")
    p_long = mp.Process(target=long_proc, args=(base_cfg, stop, scanning, lp), daemon=True)
    p_async = mp.Process(target=async_proc, args=(base_cfg, stop, scanning, ap), daemon=True)
    t_start = now()
    p_long.start()
    p_async.start()
    events, constraint_info = [], {}
    ev_stop = threading.Event()
    ev_thread = threading.Thread(target=epoch_events,
                                 args=(port, opts["epoch_s"], ep, ng, ev_stop, events, t_start, constraint_info),
                                 daemon=True)
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
    except BaseException:  # stopped (SIGTERM) or docker failed: the drivers must not outlive the epoch
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
        docker("kill", server.name, check=False)
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
            events.append({"t": round(now() - t_start, 2), "event": "constraints_state", "states": states,
                           "seconds": round(now() - t0, 1), "after_load": True})
        except Exception as exc:
            events.append({"t": round(now() - t_start, 2), "event": "events_died", "msg": str(exc)[:200]})
    # an index still building when the load stopped gets 300 s more before it counts as stuck
    for e in [x for x in events if x["event"] == "indexes_pending_at_epoch_end"]:
        try:
            ok, took = wait_indexes(r, gname(e["graph"]), 300)
            events.append({"t": round(now() - t_start, 2), "event": "indexes_ready" if ok else "indexes_stuck",
                           "graph": e["graph"], "seconds": round(e["seconds"] + took, 1), "after_load": True})
        except Exception as exc:
            events.append({"t": round(now() - t_start, 2), "event": "events_died", "msg": str(exc)[:200]})
    t_end = now()
    data = {"reads": [], "writes": [], "longs": [], "errors": {}, "died": [], "unfinished": 0,
            "pending_at_stop": 0, "drain_s": 0.0, "harness": {}, "window_s": t_end - t_start}
    for path, keys in ((ap, ("reads", "writes", "errors", "died", "unfinished", "pending_at_stop", "drain_s", "harness")),
                       (lp, ("longs", "writes", "died", "unfinished"))):
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

    result = {"params": ep, "hour": hp, "calibration": cal, "events": events,
              "duration_s": round(t_stopped - t_start, 1)}
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
        an = checks.analyse(ep, hp, cal, data, pings, downs, events, constraint_info, ledger_problems)
        running, code = server.state()
        # a server that stopped answering after the load (the prober's marker, or no PONG now) did not stay up
        hung = running and (os.path.exists(hang_path) or server_hung(port))
        an["checks"]["server_stays_up"] = {"pass": running and not hung, "exit_code": None if running else code,
                                           "hang": hung}
        if ledger.incomplete:
            for k in ("writes_add_up", "upsert_edges_complete", "timed_out_writes_leave_nothing"):
                an["checks"][k] = {"pass": None, "skipped": "a driver result file was lost earlier on this server"}
        an["checks"]["index_matches_scan"] = {"pass": not ivs, "applies": ep["indexes"] or ep["constraints"],
                                              "problems": ivs}
        result.update(an)
        result["notes"] = notes
    result["reusable"] = reusable
    result["failed"] = sorted(k for k, v in result["checks"].items() if v.get("pass", True) is False)
    with open(os.path.join(edir, "epoch.json"), "w") as f:
        json.dump(result, f, indent=1, default=str)
    with gzip.open(os.path.join(edir, "raw.json.gz"), "wt") as f:
        json.dump({"reads": data["reads"], "writes": data["writes"], "longs": data["longs"]}, f)
    for p in (lp, ap):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass  # that driver never wrote its file (already reported as "no results from ...")
    log(f"epoch {ep['seed']} failed={result['failed']} "
        f"worst_ping={result['checks'].get('no_freeze', {}).get('worst_ping_ms')}ms "
        f"read_timeouts={result['checks'].get('no_timeouts_while_waiting', {}).get('all_read_timeouts')}")
    return result


def schema_left(r, ng) -> dict:
    """Indexes and constraints still on each graph ({graph: count}, non-zero only). The built graphs
    have none, so after an epoch's clean-up there should be none either."""
    left = {}
    for g in range(ng):
        n = (len(r.execute_command("GRAPH.RO_QUERY", gname(g), "CALL db.indexes()")[1])
             + len(r.execute_command("GRAPH.RO_QUERY", gname(g), "CALL db.constraints()")[1]))
        if n:
            left[g] = n
    return left


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
        if ep["indexes"] or ep["constraints"]:
            out["ivs"] = index_vs_scan(r, ng, sizes, ep["seed"])
    except Exception as e:
        out["ledger_problems"] = [("harness", f"end-of-epoch checks failed: {type(e).__name__}: {e}")]
        out["notes"].append(traceback.format_exc()[-500:])
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
        out["memory"] = {"used_memory": mem.get("used_memory"), "used_memory_rss": mem.get("used_memory_rss")}
    except Exception as e:
        out["notes"].append(f"no memory figure: {type(e).__name__}: {e}"[:200])
    return out


def grab_backtrace(server: Server, edir: str) -> str:
    path = os.path.join(edir, "hang-backtrace.txt")
    try:
        pid = server.pid()
        with open(path, "w") as f:
            subprocess.run(["gdb", "-p", str(pid), "-batch", "-ex", "thread apply all bt"], stdout=f,
                           stderr=subprocess.STDOUT, timeout=120)
    except Exception as e:
        with open(path, "a") as f:
            f.write(f"\nbacktrace failed: {e}\n")
    return os.path.basename(path)


# --- server-hour --------------------------------------------------------------------------------


def run_server_hour(opts: dict, info: dict, tag: str, hp: dict, epochs: list[dict], hdir: str,
                    deadline: float | None = None) -> dict:
    """Start a server and its plain PING control from the image, build the hour's graphs, run its
    epochs. Returns {"hour", "build": [...], "epochs": [...], "module_version", "config"}, plus
    "error" and "server_alive" when something failed before the epochs finished (a server that
    would not start, died during the build, or a harness fault)."""
    os.makedirs(hdir, exist_ok=True)
    log(f"{tag}: server-hour {hp}")
    server = Server(info, f"{opts['name_prefix']}-{tag}", opts.get("server_cpus"), opts["net"], hp["thread_count"])
    control = Server(info, f"{opts['name_prefix']}-{tag}-plain", opts.get("client_cpus"), opts["net"], module=False)
    out = {"hour": hp, "tag": tag, "build": [], "epochs": []}
    prober = None
    stop = mp.Event()
    ping_path = os.path.join(hdir, "pings.csv")
    hang_path = os.path.join(hdir, "HANG")
    if os.path.exists(hang_path):  # a marker left by an earlier run in this directory is not this server's
        os.remove(hang_path)
    try:
        r = server.start()
        control.start()
        out["module_version"] = module_version(r)
        out["config"] = server_config(r)
        prober = mp.Process(target=prober_proc, args=({"falkor": server.port, "control": control.port},
                                                      ping_path, hang_path, stop), daemon=True)
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
            if deadline and time.time() + model.epoch_seconds(opts["epoch_s"], opts["drain_s"], ep) > deadline:
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
            res_list[-1]["checks"]["memory_steady"] = {"pass": growth < 0.10, "growth": round(growth, 4),
                                                      "used_memory_by_epoch": mems}
        for x in res_list:
            x["failed"] = sorted(k for k, v in x.get("checks", {}).items() if v.get("pass", True) is False)
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
        with open(os.path.join(hdir, "server-tail.log"), "w") as f:
            f.write(server.stop())
        control.stop()
    with open(os.path.join(hdir, "hour.json"), "w") as f:
        json.dump({k: v for k, v in out.items() if k != "epochs"}, f, indent=1, default=str)
    return out


def set_start_method():
    if sys.platform != "win32":
        try:
            mp.set_start_method("fork")
        except RuntimeError:
            pass  # already set by an earlier call in this process
