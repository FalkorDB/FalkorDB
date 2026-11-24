"""soakkit — shared harness for FalkorDB release soak scenarios.

Every scenario uses this so all results share one format:
  <out>/monitor.csv       1 Hz server samples: cpu%, rss, used_memory, threads, alive
  <out>/lat-*.jsonl       per-worker, per-second, per-op latency histograms
  <out>/checks.json       correctness / invariant checks the scenario recorded
  <out>/summary.json      produced by analyze(): throughput, p50/p95/p99/p999,
                          drift, cpu, memory growth, crashes, errors, checks

The server under test is either a module path, run with a local redis-server, or a
falkordb-server image (Image), whose own server binary and module run in docker; a
scenario takes either through add_engine_args().

Usage sketch (see scenario template in README.md):
    srv = Server(module=..., port=..., workdir=out)   # starts redis + module
    mon = Monitor(srv, out); mon.start()
    ... load data with srv.client() ...
    run_load(srv, out, workers=[WorkerSpec(fn, rate=..)], duration=..)
    checks.add("name", ok, detail)
    mon.stop(); srv.stop(); analyze(out)
"""
from __future__ import annotations

import argparse
import atexit
import functools
import json
import math
import multiprocessing as mp
import os
import random
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import psutil
import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

REDIS_BIN = os.environ.get("SOAK_REDIS", "redis-server")
RUST_MODULE = os.environ.get("SOAK_RUST_MODULE", "")
C_MODULE = os.environ.get("SOAK_C_MODULE", "")
# Where every falkordb-server image keeps the module (Rust and C, Redis and Valkey).
IMAGE_MODULE = "/var/lib/falkordb/bin/falkordb.so"
# Every container a Server starts carries this label, so a driver can remove what a
# scenario that died left behind (check.py sets SOAK_RUN_ID per run).
DOCKER_LABEL = "falkordb-soak=" + os.environ.get("SOAK_RUN_ID", "local")
CHECK_LABEL = "falkordb-soak-check=" + os.environ.get("SOAK_CHECK_ID", "local")


# ---------------------------------------------------------------- engine under test
@dataclass(frozen=True)
class Image:
    """A falkordb-server image (name@sha256:...) to run instead of a local module."""
    ref: str

    def __str__(self):
        return self.ref


def add_engine_args(ap: argparse.ArgumentParser) -> None:
    """--module PATH | --image REF, both stored in args.module (a str or an Image)."""
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--module", help="module path, loaded into $SOAK_REDIS (default redis-server)")
    g.add_argument("--image", dest="module", type=Image,
                   help="falkordb-server image; runs the image's own server and module in docker "
                        "(Linux only)")


def _docker(*args: str, check: bool = True) -> str:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check).stdout


@functools.lru_cache(maxsize=None)
def image_server(ref: str) -> str:
    """The server binary an image ships, picked the way its run.sh picks it:
    ${FALKORDB_SERVER:-redis}-server (6.2+ Rust images set valkey; C and older ones none)."""
    try:
        env_json = _docker("image", "inspect", "--format", "{{json .Config.Env}}", ref)
    except subprocess.CalledProcessError:
        _docker("pull", "--quiet", ref)
        env_json = _docker("image", "inspect", "--format", "{{json .Config.Env}}", ref)
    env = dict(e.split("=", 1) for e in (json.loads(env_json) or []) if "=" in e)
    return f"{env.get('FALKORDB_SERVER') or 'redis'}-server"


def image_server_version(ref: str) -> str:
    """`<server> --version` of the binary image_server() names, run from the image."""
    return _docker("run", "--rm", "--network", "none", "--entrypoint", image_server(ref), ref,
                   "--version").strip()


_CONTAINERS: set[str] = set()


@atexit.register
def _remove_containers():
    # a scenario that raised leaves its servers running; never leave a container behind
    for name in list(_CONTAINERS):
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)



# ---------------------------------------------------------------- client utils
def connect(port: int, timeout: float | None = None) -> redis.Redis:
    # No retries ever: redis-py's default timeout + retry silently re-issues slow commands
    # (a slow CREATE could apply twice) — never in a soak. No client timeout either, unless
    # the caller bounds one probe on purpose (it then sees redis.exceptions.TimeoutError).
    return redis.Redis(port=port, socket_timeout=timeout, retry=Retry(NoBackoff(), 0),
                       decode_responses=False)


def lit(v) -> str:
    """Cypher literal for a CYPHER-prefix parameter."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(lit(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(f"{k}:{lit(x)}" for k, x in v.items()) + "}"
    raise TypeError(type(v))


def query(r: redis.Redis, graph: str, q: str, params: dict | None = None, ro=False,
          timeout_ms: int | None = None):
    if params:
        q = "CYPHER " + " ".join(f"{k}={lit(v)}" for k, v in params.items()) + " " + q
    args = ["GRAPH.RO_QUERY" if ro else "GRAPH.QUERY", graph, q, "--compact"]
    if timeout_ms:
        args += ["timeout", timeout_ms]
    return r.execute_command(*args)


def rows(res):
    """Result rows of a --compact reply (list of lists of [type, value])."""
    if not isinstance(res, list) or len(res) < 3:
        return []
    return res[1]


def cval(v):
    """Normalize one compact cell [type, value]: 1 null, 2 str, 3 int, 4 bool, 5 double."""
    if isinstance(v, list) and len(v) == 2 and isinstance(v[0], int):
        t, x = v
        if t == 1:
            return None
        if t == 3:
            return int(x)
        if t == 5:
            return float(x)
        if t == 4:
            return x in (b"true", "true", 1, True)
        if t == 2:
            return x.decode() if isinstance(x, bytes) else x
        return x
    return v


def scalar(res):
    """First cell of the first row, normalized; numbers that are integral become int."""
    rs = rows(res)
    if not rs:
        return None
    v = cval(rs[0][0])
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


# ---------------------------------------------------------------- server
class Server:
    """One server under test. `module` is a module path (a local $SOAK_REDIS loads it) or an
    Image: then the image's own server binary and module run in a container with host
    networking, the workdir bind-mounted at the same path (RDB/AOF/log land where they do
    locally), as the calling user, pinned to $PERF_SERVER_CPUS when set. The image's
    run.sh and its default FALKORDB_ARGS are deliberately bypassed: both sides of a
    comparison get exactly the scenario's arguments. `docker run` stays in the
    foreground, so self.proc tracks the container; pid is the server's host PID."""

    def __init__(self, module: str | Image, port: int, workdir: Path | str,
                 redis_bin: str = REDIS_BIN, module_args: list[str] | None = None,
                 extra: list[str] | None = None, name: str = "server"):
        self.module, self.port = module, port
        self.image = module if isinstance(module, Image) else None
        if self.image:
            if not sys.platform.startswith("linux"):
                raise RuntimeError("--image needs Linux (host networking, host PIDs); use --module here")
            redis_bin = image_server(self.image.ref)
        self.redis_bin = redis_bin
        self.workdir = Path(workdir).resolve()  # redis --dir/--logfile need absolute paths
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.module_args = module_args or []
        self.extra = extra or []
        self.name = name
        self.log = self.workdir / f"{name}-{port}.log"
        self.container = f"soak-{name}-{port}-{os.getpid()}" if self.image else None
        self.proc: subprocess.Popen | None = None
        self._host_pid: int | None = None
        self.start()

    def argv(self) -> list[str]:
        module = IMAGE_MODULE if self.image else self.module
        return [self.redis_bin, "--port", str(self.port), "--dir", str(self.workdir),
                "--dbfilename", f"{self.name}-{self.port}.rdb", "--save", "",
                "--appendonly", "no", "--logfile", str(self.log), "--protected-mode", "no",
                "--loadmodule", module, *self.module_args, *self.extra]

    def start(self, timeout: float = 60):
        # refuse to attach to a stale server left on this port by a crashed earlier run
        try:
            if connect(self.port).ping():
                raise RuntimeError(f"port {self.port} already has a server (stale run?); stop it first "
                                   f"(docker rm -f $(docker ps -q --filter label=falkordb-soak) for "
                                   f"image runs, pkill -f 'redis-server.*--port {self.port}' otherwise)")
        except redis.exceptions.ConnectionError:
            pass
        cmd = self.argv()
        if self.image:
            subprocess.run(["docker", "rm", "-f", self.container], capture_output=True)
            cpus = os.environ.get("PERF_SERVER_CPUS")
            # host networking, so listen on loopback only: no password, protected mode off
            cmd = ["docker", "run", "--rm", "--name", self.container, "--label", DOCKER_LABEL,
                   "--label", CHECK_LABEL,
                   "--network", "host", "--user", f"{os.getuid()}:{os.getgid()}",
                   "--ulimit", "nofile=65536:65536", "-v", f"{self.workdir}:{self.workdir}",
                   *(["--cpuset-cpus", cpus] if cpus else []),
                   "--entrypoint", cmd[0], self.image.ref, *cmd[1:3], "--bind", "127.0.0.1",
                   *cmd[3:]]
            _CONTAINERS.add(self.container)
        self._host_pid = None
        stderr = self.workdir / f"{self.name}-{self.port}.stderr"
        with open(stderr, "ab") as err:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                tail = stderr.read_text(errors="replace")[-400:].strip()
                raise RuntimeError(f"server exited at start rc={self.proc.returncode}; see {self.log}"
                                   + (f"; stderr: {tail}" if tail else ""))
            if self.image and not self._host_pid:
                # as soon as the container runs, so the monitor follows the server through a
                # long RDB/AOF load (it answers LOADING, which redis-py reports as an error)
                state = subprocess.run(["docker", "inspect", "--format", "{{.State.Pid}}", self.container],
                                       capture_output=True, text=True).stdout.strip()
                self._host_pid = int(state) if state.isdigit() and int(state) > 0 else None
            try:
                if connect(self.port).ping():
                    if self.image and not self._host_pid:
                        self._host_pid = int(_docker("inspect", "--format", "{{.State.Pid}}",
                                                     self.container).strip())
                    return
            except redis.exceptions.ConnectionError:
                time.sleep(0.1)
        raise RuntimeError(f"server did not start in {timeout:.0f}s")

    @property
    def pid(self):
        """Host PID of the server process (psutil works on it in both modes)."""
        if self.image:
            return self._host_pid
        return self.proc.pid if self.proc else None

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def client(self):
        return connect(self.port)

    def engine(self) -> str:
        """'rust' or 'c', from the graph module's version as the server reports it
        (C is 4.x = 4xxxx; Rust is 6.x and later, a local dev build 99.99.99)."""
        for m in self.client().execute_command("MODULE", "LIST"):
            # redis-py parses each module into a dict; a raw reply is a flat key/value list
            pairs = m.items() if isinstance(m, dict) else zip(m[::2], m[1::2])
            d = {(k.decode() if isinstance(k, bytes) else k): v for k, v in pairs}
            name = d.get("name")
            if (name.decode() if isinstance(name, bytes) else name) == "graph":
                return "rust" if int(d["ver"]) >= 50000 else "c"
        raise RuntimeError("graph module not loaded")

    def restart(self, save=True):
        """SAVE (optional), SIGTERM, start again from the RDB; returns restart seconds."""
        r = self.client()
        if save:
            r.execute_command("SAVE")
        self.stop()
        t = time.time()
        self.start()
        return time.time() - t

    def _signal(self, sig: signal.Signals):
        if self.image:
            subprocess.run(["docker", "kill", "--signal", sig.name, self.container], capture_output=True)
        else:
            self.proc.send_signal(sig)

    def kill9(self):
        if self.proc:
            if self.proc.poll() is None:
                self._signal(signal.SIGKILL)
            self.proc.wait()

    def terminate(self):
        """SIGTERM (the server saves per its config and exits); wait() for it."""
        if self.alive():
            self._signal(signal.SIGTERM)

    def wait(self, timeout: float | None = None):
        """Wait for the server to exit; raises subprocess.TimeoutExpired."""
        return self.proc.wait(timeout)

    def stop(self):
        if self.alive():
            try:
                self.client().execute_command("SHUTDOWN", "NOSAVE")
            except Exception:
                pass
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.kill9()

    def crashed(self) -> list[str]:
        """Crash / panic markers in the server log."""
        if not self.log.exists():
            return []
        bad = []
        for line in self.log.read_text(errors="replace").splitlines():
            l = line.lower()
            if any(k in l for k in ("panic", "crashed by signal", "bug report", "segmentation",
                                    "assertion", "stack trace", "abort")):
                bad.append(line[:300])
        return bad[:50]


# ---------------------------------------------------------------- monitor
class Monitor(threading.Thread):
    """1 Hz: process cpu% (all threads; 100 = one core), rss, INFO used_memory,
    thread count, connected clients, alive. Follows restarts of the same Server."""

    def __init__(self, server: Server, out: Path | str, interval: float = 1.0):
        super().__init__(daemon=True)
        self.server, self.out, self.interval = server, Path(out), interval
        self._halt = threading.Event()
        self.events: list[dict] = []
        self.f = open(self.out / "monitor.csv", "w")
        self.f.write("t,cpu_pct,rss_mb,used_mem_mb,threads,clients,alive,phase\n")
        self.phase = "setup"
        self.t0 = time.time()

    def mark(self, phase: str):
        self.phase = phase
        self.events.append({"t": round(time.time() - self.t0, 1), "phase": phase})

    def run(self):
        proc, pid = None, None
        r = None
        while not self._halt.is_set():
            t = time.time() - self.t0
            alive = self.server.alive()
            cpu = rss = um = th = cl = float("nan")
            # pid is None while an image's container is starting: psutil.Process(None) would
            # be this process, so skip the sample
            if alive and self.server.pid:
                try:
                    if pid != self.server.pid:
                        pid = self.server.pid
                        proc = psutil.Process(pid)
                        proc.cpu_percent(None)
                        r = connect(self.server.port)
                    cpu = proc.cpu_percent(None)
                    mi = proc.memory_info()
                    rss = mi.rss / 2**20
                    th = proc.num_threads()
                    info = r.info()
                    um = info.get("used_memory", 0) / 2**20
                    cl = info.get("connected_clients", 0)
                except Exception:
                    pass
            self.f.write(f"{t:.1f},{cpu:.1f},{rss:.1f},{um:.1f},{th},{cl},{int(alive)},{self.phase}\n")
            self.f.flush()
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(5)
        self.f.close()
        (self.out / "phases.json").write_text(json.dumps(self.events))


# ---------------------------------------------------------------- latency histograms
# log buckets, 2% resolution, microseconds
_B = math.log(1.02)


def bucket(us: float) -> int:
    return int(math.log(max(us, 1.0)) / _B)


def bucket_val(b: int) -> float:
    return math.exp((b + 0.5) * _B)


def hpct(h: dict, p: float) -> float:
    n = sum(h.values())
    if not n:
        return float("nan")
    k = p * n
    c = 0
    for b in sorted(h):
        c += h[b]
        if c >= k:
            return bucket_val(b)
    return bucket_val(max(h))


class Recorder:
    """Per-worker recorder; flushes one JSON line per (second, op)."""

    def __init__(self, path: Path, t0: float):
        self.f = open(path, "w")
        self.t0 = t0
        self.sec = None
        self.cur: dict[str, dict] = {}
        self.errmsgs: dict[str, int] = {}

    def record(self, op: str, us: float, ok: bool = True, err: str | None = None):
        s = int(time.time() - self.t0)
        if s != self.sec:
            self.flush()
            self.sec = s
        d = self.cur.setdefault(op, {"n": 0, "err": 0, "h": {}})
        d["n"] += 1
        if not ok:
            d["err"] += 1
            if err:
                k = err[:160]
                self.errmsgs[k] = self.errmsgs.get(k, 0) + 1
        else:
            b = bucket(us)
            d["h"][b] = d["h"].get(b, 0) + 1

    def flush(self):
        for op, d in self.cur.items():
            self.f.write(json.dumps({"s": self.sec, "op": op, **d}) + "\n")
        self.cur = {}
        self.f.flush()

    def close(self):
        self.flush()
        self.f.write(json.dumps({"errmsgs": self.errmsgs}) + "\n")
        self.f.close()


@dataclass
class WorkerSpec:
    """fn(ctx) performs ONE operation and returns (op_name, extra) or op_name.
    ctx has: r (redis client), rng, wid, port, state (dict, per worker), rec.
    rate: target ops/s for this worker (open loop with catch-up; latency is measured from the
    actual send, and sends >50 ms behind schedule are counted in op "_late_start");
    None = closed loop."""
    fn: Callable
    rate: float | None = None
    count: int = 1
    name: str = "w"
    init: Callable | None = None  # init(ctx) once per worker


class Ctx:
    def __init__(self, port, wid, seed):
        self.r = connect(port)
        self.port = port
        self.wid = wid
        self.rng = random.Random(seed)
        self.state: dict = {}


def _worker(spec_bytes, port, wid, out, t0, t_end, seed):
    import pickle
    spec: WorkerSpec = pickle.loads(spec_bytes)
    ctx = Ctx(port, wid, seed)
    rec = Recorder(Path(out) / f"lat-{spec.name}-{wid}.jsonl", t0)
    ctx.rec = rec
    if spec.init:
        spec.init(ctx)
    interval = 1.0 / spec.rate if spec.rate else 0
    nxt = time.time()
    while True:
        now = time.time()
        if now >= t_end:
            break
        if interval:
            if now < nxt:
                time.sleep(min(nxt - now, t_end - now))
                if time.time() >= t_end:
                    break
            sched = nxt
            nxt += interval
            # never queue more than 1s of backlog: drop it and count as missed
            if time.time() - nxt > 1.0:
                missed = int((time.time() - nxt) / interval)
                nxt += missed * interval
                for _ in range(min(missed, 100000)):
                    rec.record("_missed_schedule", 0, ok=False)
        start = time.time()  # service latency from actual send (macOS sleep overshoots ~1 ms)
        if interval and start - sched > 0.05:
            # client fell >50 ms behind schedule: server backpressure (coordinated omission signal)
            rec.record("_late_start", (start - sched) * 1e6)
        op = "op"
        try:
            ret = spec.fn(ctx)
            op = ret[0] if isinstance(ret, tuple) else (ret or "op")
            rec.record(op, (time.time() - start) * 1e6)
        except redis.exceptions.ConnectionError as e:
            rec.record(op if op != "op" else spec.name, 0, ok=False, err=f"ConnectionError: {e}")
            time.sleep(0.2)
            try:
                ctx.r = connect(port)
            except Exception:
                pass
        except Exception as e:  # query errors are data, not worker death
            rec.record(getattr(e, "op", spec.name), 0, ok=False, err=f"{type(e).__name__}: {e}")
    rec.close()


def _worker_group(jobs):
    ths = [threading.Thread(target=_worker, args=j) for j in jobs]
    for t in ths:
        t.start()
    for t in ths:
        t.join()


def run_load(server: Server, out: Path | str, workers: list[WorkerSpec], duration: float,
             seed: int = 1, tag: str = "") -> None:
    """Run all worker specs concurrently (one OS process per worker) for `duration` s."""
    import pickle
    out = Path(out)
    t0 = time.time()
    t_end = t0 + duration
    procs = []
    wid = 0
    _cap = int(os.environ.get("SOAK_MAX_WORKERS", "0") or 0)  # cap OS processes; extra workers become threads
    jobs = []
    for spec in workers:
        if tag:
            spec.name = f"{tag}{spec.name}"
        sb = pickle.dumps(spec)
        for _ in range(spec.count):
            jobs.append((sb, server.port, wid, str(out), t0, t_end, seed * 1000 + wid))
            wid += 1
    if _cap and len(jobs) > _cap:
        groups = [jobs[i::_cap] for i in range(_cap)]
        for g in groups:
            p = mp.get_context("fork").Process(target=_worker_group, args=(g,))
            p.start()
            procs.append(p)
    else:
        for j in jobs:
            p = mp.get_context("fork").Process(target=_worker, args=j)
            p.start()
            procs.append(p)
    for p in procs:
        p.join(duration + 600)
        if p.is_alive():
            p.kill()
    meta = out / "load_runs.json"
    runs = json.loads(meta.read_text()) if meta.exists() else []
    runs.append({"tag": tag, "start": t0, "duration": duration,
                 "workers": [{"name": s.name, "count": s.count, "rate": s.rate} for s in workers]})
    meta.write_text(json.dumps(runs))


# ---------------------------------------------------------------- checks
class Checks:
    def __init__(self, out: Path | str):
        self.path = Path(out) / "checks.json"
        self.items: list[dict] = []

    def add(self, name: str, ok: bool, detail: str = "", severity: str = "major"):
        """severity: critical (data loss/wrong result/crash) | major | minor."""
        self.items.append({"name": name, "ok": bool(ok), "detail": str(detail)[:2000],
                           "severity": severity})
        self.path.write_text(json.dumps(self.items, indent=1))
        print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail and not ok else ""),
              flush=True)


# ---------------------------------------------------------------- analysis
def _linreg(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if not sxx:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def analyze(out: Path | str, server: Server | None = None, warmup_frac: float = 0.15) -> dict:
    out = Path(out)
    per_sec: dict[tuple[str, int], dict] = {}
    total: dict[str, dict] = {}
    errmsgs: dict[str, int] = {}
    for f in sorted(out.glob("lat-*.jsonl")):
        for line in f.read_text().splitlines():
            d = json.loads(line)
            if "errmsgs" in d:
                for k, v in d["errmsgs"].items():
                    errmsgs[k] = errmsgs.get(k, 0) + v
                continue
            op, s = d["op"], d["s"]
            for tgt in (per_sec.setdefault((op, s), {"n": 0, "err": 0, "h": {}}),
                        total.setdefault(op, {"n": 0, "err": 0, "h": {}, "smin": s, "smax": s})):
                tgt["n"] += d["n"]
                tgt["err"] += d["err"]
                for b, c in d["h"].items():
                    b = int(b)
                    tgt["h"][b] = tgt["h"].get(b, 0) + c
            t = total[op]
            t["smin"], t["smax"] = min(t["smin"], s), max(t["smax"], s)

    ops = {}
    ts_rows = []
    for op, t in total.items():
        secs = max(1, t["smax"] - t["smin"] + 1)
        h = t["h"]
        # drift: p99 over first vs last 10% of seconds (after warmup)
        ss = sorted(s for (o, s) in per_sec if o == op)
        drift = None
        if len(ss) >= 30:
            w = ss[int(len(ss) * warmup_frac):]
            k = max(3, len(w) // 10)
            def merged(sec_list):
                m = {}
                for s in sec_list:
                    for b, c in per_sec[(op, s)]["h"].items():
                        m[b] = m.get(b, 0) + c
                return m
            a, z = hpct(merged(w[:k]), .99), hpct(merged(w[-k:]), .99)
            drift = {"p99_early_ms": a / 1e3, "p99_late_ms": z / 1e3,
                     "ratio": (z / a) if a and a == a and z == z else None}
        ops[op] = {"count": t["n"], "errors": t["err"],
                   "err_rate": t["err"] / t["n"] if t["n"] else 0,
                   "throughput_ops_s": (t["n"] - t["err"]) / secs,
                   "p50_ms": hpct(h, .5) / 1e3, "p95_ms": hpct(h, .95) / 1e3,
                   "p99_ms": hpct(h, .99) / 1e3, "p999_ms": hpct(h, .999) / 1e3,
                   "max_ms": (bucket_val(max(h)) / 1e3) if h else None, "drift": drift}
    for (op, s), d in sorted(per_sec.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        ts_rows.append([s, op, d["n"] - d["err"], d["err"], round(hpct(d["h"], .5) / 1e3, 3),
                        round(hpct(d["h"], .99) / 1e3, 3)])
    with open(out / "latency_ts.csv", "w") as f:
        f.write("sec,op,ok,err,p50_ms,p99_ms\n")
        for r in ts_rows:
            f.write(",".join(str(x) for x in r) + "\n")

    # monitor
    mon = {}
    mp_ = out / "monitor.csv"
    if mp_.exists():
        lines = mp_.read_text().splitlines()[1:]
        rec = []
        for l in lines:
            p = l.split(",")
            try:
                rec.append((float(p[0]), float(p[1]), float(p[2]), float(p[3]), p[6] == "1", p[7]))
            except (ValueError, IndexError):
                continue
        alive = [x for x in rec if x[4] and x[2] == x[2]]
        load = [x for x in alive if x[5].startswith("load") or x[5].startswith("soak")] or alive
        if alive:
            cpu = [x[1] for x in load if x[1] == x[1]]
            rss = [x[2] for x in load]
            um = [x[3] for x in load if x[3] == x[3]]
            w = load[int(len(load) * warmup_frac):]
            xs = [x[0] / 3600 for x in w]
            mon = {"cpu_avg_pct": sum(cpu) / len(cpu) if cpu else None,
                   "cpu_max_pct": max(cpu) if cpu else None,
                   "rss_start_mb": rss[0], "rss_end_mb": rss[-1], "rss_max_mb": max(rss),
                   "used_mem_end_mb": um[-1] if um else None,
                   "rss_slope_mb_per_h": _linreg(xs, [x[2] for x in w]),
                   "used_mem_slope_mb_per_h": _linreg(xs, [x[3] for x in w if x[3] == x[3]])
                   if len(w) == len([x for x in w if x[3] == x[3]]) else None,
                   "samples": len(rec), "dead_samples": len(rec) - len(alive)}
    checks = json.loads((out / "checks.json").read_text()) if (out / "checks.json").exists() else []
    summary = {"ops": ops, "monitor": mon, "checks": checks,
               "checks_failed": [c for c in checks if not c["ok"]],
               "top_errors": sorted(errmsgs.items(), key=lambda kv: -kv[1])[:20],
               "server_log_crash_markers": server.crashed() if server else []}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def print_summary(s: dict):
    print(f"{'op':34} {'count':>9} {'err':>6} {'ops/s':>9} {'p50':>8} {'p99':>8} {'p999':>8} {'drift':>6}")
    for op, d in sorted(s["ops"].items()):
        dr = d["drift"]["ratio"] if d["drift"] and d["drift"]["ratio"] else float("nan")
        print(f"{op[:34]:34} {d['count']:9} {d['errors']:6} {d['throughput_ops_s']:9.1f} "
              f"{d['p50_ms']:8.2f} {d['p99_ms']:8.2f} {d['p999_ms']:8.2f} {dr:6.2f}")
    print("monitor:", json.dumps(s["monitor"], default=str))
    print("failed checks:", [c["name"] for c in s["checks_failed"]])
    print("top errors:", s["top_errors"][:5])
    print("crash markers:", s["server_log_crash_markers"][:5])
