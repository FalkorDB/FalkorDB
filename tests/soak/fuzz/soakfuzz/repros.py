"""Targeted checks for long mode, each on a fresh server with a graph of its own (`q`, label N).

Each returns {"pass": True/False/None, "detail": ...}:

* write-<none|create|set>: a write sent to a graph while a ~30 s read runs on it must not freeze
  the server (no PING over 500 ms during the read). `none` is the read alone.
* constraint-<none|create|drop|mandatory>: the same for GRAPH.CONSTRAINT CREATE / DROP (#3763).
* index-build: an index that finishes populating while a long read holds the graph must not freeze
  the server (#3705).
* writer-starvation: at THREAD_COUNT 10, four writers behind 20 clients of back-to-back ~300 ms
  reads must finish within 120 s, every acknowledged write exactly once.
* reads-behind-writer: cheap reads with TIMEOUT 1000 ms, sent while a write waits for a long read,
  must not time out (#3685).
"""

from __future__ import annotations

import threading
import time

from redis.exceptions import ResponseError

from . import harness

FREEZE_MS = 500
WAIT_S = 300  # every request and wait in a repro gives up after this (a hung server)
LONG_READ = "MATCH (n:N) WITH n LIMIT {limit} UNWIND range(1, 200000) AS x RETURN count(x)"


class Incomplete(RuntimeError):
    """A request the repro depends on did not complete; run_repro decides crash or unmeasured."""


def _conn(port):
    return harness.connect(port, timeout=WAIT_S)


def _no_timeouts(r):
    for k in ("TIMEOUT", "TIMEOUT_MAX", "TIMEOUT_DEFAULT"):
        try:
            r.execute_command("GRAPH.CONFIG", "SET", k, "0")
        except ResponseError:
            pass  # an engine without this setting has no such timeout to turn off


def _fill(r, nodes):
    r.delete("q")
    for lo in range(0, nodes, 500_000):
        r.execute_command("GRAPH.QUERY", "q", f"UNWIND range({lo}, {min(nodes, lo + 500_000) - 1}) AS i CREATE (:N {{v: i}})")


def _wait_status(r, proc):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < WAIT_S:
        rows = r.execute_command("GRAPH.RO_QUERY", "q", f"CALL {proc}() YIELD status RETURN status")[1]
        if rows and all(harness.s(x[0]) == "OPERATIONAL" for x in rows):
            return
        time.sleep(0.05)
    raise TimeoutError(f"{proc}: not OPERATIONAL after {WAIT_S} s")


def _join(*threads):
    for t in threads:
        t.join(WAIT_S)
        if t.is_alive():
            raise TimeoutError(f"a client thread still waits after {WAIT_S} s")


class _Pinger:
    def __init__(self, port):
        self.port, self.pings, self.stop = port, [], threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        p = _conn(self.port)
        while not self.stop.is_set():
            t = time.perf_counter()
            p.ping()
            self.pings.append((t, (time.perf_counter() - t) * 1000))
            time.sleep(0.01)

    def __enter__(self):
        self.t.start()
        time.sleep(0.5)
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.t.join(5)

    def during(self, a, b):
        return [lat for t, lat in self.pings if a <= t <= b]


def _long_read(port, limit, out):
    """Appends (start, end, error or None): a failed read's interval is no evidence of anything."""
    c = _conn(port)
    t = time.perf_counter()
    err = None
    try:
        c.execute_command("GRAPH.RO_QUERY", "q", LONG_READ.format(limit=limit))
    except Exception as e:
        err = f"{type(e).__name__}: {e}"[:200]
    out.append((t, time.perf_counter(), err))


def _freeze_verdict(pinger, lr, extra):
    if not lr or lr[0][2] is not None:
        raise Incomplete(f"the long read did not complete: {lr[0][2] if lr else 'no reply'}")
    a, b, _ = lr[0]
    during = pinger.during(a, b)
    late = sum(1 for x in during if x > FREEZE_MS)
    return {"pass": late == 0 if during else None,
            "detail": {"long_read_s": round(b - a, 1), "worst_ping_ms": round(max(during, default=0), 1),
                       "pings_over_500ms": late, **extra}}


def write_during_long_read(port, mode, nodes=1_000_000):
    r = _conn(port)
    _no_timeouts(r)
    _fill(r, nodes)
    lr, extra = [], {}
    with _Pinger(port) as pinger:
        th = threading.Thread(target=_long_read, args=(port, 2000, lr))
        th.start()
        time.sleep(1.0)
        q = {"create": "CREATE (:W {i: 1})", "set": "MATCH (n:N {v: 1}) SET n.w = 1"}.get(mode)
        if q:
            t_w = time.perf_counter()
            r.execute_command("GRAPH.QUERY", "q", q)
            extra["write_s"] = round(time.perf_counter() - t_w, 1)
        _join(th)
        time.sleep(0.5)
    return _freeze_verdict(pinger, lr, extra)


def constraint_during_long_read(port, mode, nodes=1_000_000):
    r = _conn(port)
    _no_timeouts(r)
    _fill(r, nodes)
    r.execute_command("GRAPH.QUERY", "q", "CREATE INDEX FOR (n:N) ON (n.v)")
    _wait_status(r, "db.indexes")
    constraint = ("q", "MANDATORY" if mode == "mandatory" else "UNIQUE", "NODE", "N", "PROPERTIES", "1", "v")
    if mode == "drop":  # the constraint must exist (and be enforced) before the long read starts
        r.execute_command("GRAPH.CONSTRAINT", "CREATE", *constraint)
        _wait_status(r, "db.constraints")
    lr, extra = [], {}
    with _Pinger(port) as pinger:
        th = threading.Thread(target=_long_read, args=(port, 2000, lr))
        th.start()
        time.sleep(1.0)
        if mode != "none":
            t_c = time.perf_counter()
            rep = r.execute_command("GRAPH.CONSTRAINT", "DROP" if mode == "drop" else "CREATE", *constraint)
            extra.update(reply=harness.s(rep), command_s=round(time.perf_counter() - t_c, 1))
        _join(th)
        time.sleep(0.5)
    return _freeze_verdict(pinger, lr, extra)


def index_build_behind_long_read(port, nodes=2_000_000):
    r = _conn(port)
    _no_timeouts(r)
    _fill(r, nodes)
    lr = []
    with _Pinger(port) as pinger:
        reader = threading.Thread(target=_long_read, args=(port, 3000, lr))
        # create the index first (the CREATE itself waits for the write lock), then start the long
        # read at once, so the index finishes populating while the read holds the graph's read lock
        t_idx = time.perf_counter()
        r.execute_command("GRAPH.QUERY", "q", "CREATE INDEX FOR (n:N) ON (n.v)")
        reader.start()
        _wait_status(r, "db.indexes")
        t_ready = time.perf_counter()
        _join(reader)
        time.sleep(0.5)
    return _freeze_verdict(pinger, lr, {"index_ready_s": round(t_ready - t_idx, 1),
                                        "ready_before_read_ended": t_ready < lr[0][1] if lr else None})


def writer_starvation(port, readers=20, writers=4, per=25, limit_s=120):
    r = _conn(port)
    r.delete("q")
    r.execute_command("GRAPH.QUERY", "q", "UNWIND range(0, 199999) AS i CREATE (:N {v: i})")

    def read_q(k):
        return f"MATCH (n:N) WITH n LIMIT {k} UNWIND range(1, 100000) AS x RETURN count(x)"

    k = 50
    for _ in range(4):  # calibrate one read to about 300 ms alone on this server
        t = time.perf_counter()
        r.execute_command("GRAPH.RO_QUERY", "q", read_q(k))
        d = time.perf_counter() - t
        k = max(1, min(20000, int(k * 0.3 / max(d, 1e-3))))
    stop, writes, errs = threading.Event(), [], []

    def reader():
        c = _conn(port)
        while not stop.is_set():
            try:
                c.execute_command("GRAPH.RO_QUERY", "q", read_q(k))
            except Exception as e:
                errs.append(f"read: {e}")

    def writer(w):
        c = _conn(port)
        for i in range(per):
            a = time.perf_counter()
            try:
                c.execute_command("GRAPH.QUERY", "q", f"CREATE (:W {{w: {w}, i: {i}}})")
                writes.append((w, i, time.perf_counter() - a))
            except Exception as e:
                errs.append(f"write: {e}")

    rs = [threading.Thread(target=reader, daemon=True) for _ in range(readers)]
    for t in rs:
        t.start()
    time.sleep(3)
    t0 = time.perf_counter()
    ws = [threading.Thread(target=writer, args=(w,), daemon=True) for w in range(writers)]
    for t in ws:
        t.start()
    for t in ws:
        t.join(timeout=max(0.0, limit_s - (time.perf_counter() - t0)))
    starved = any(t.is_alive() for t in ws)
    took = time.perf_counter() - t0
    stop.set()
    for t in rs:
        t.join(timeout=60)
    n_w = r.execute_command("GRAPH.RO_QUERY", "q", "MATCH (n:W) RETURN count(n)")[1][0][0]
    n_d = r.execute_command("GRAPH.RO_QUERY", "q", "MATCH (n:W) RETURN count(DISTINCT [n.w, n.i])")[1][0][0]
    lat = sorted(x[2] for x in writes)
    ok = not starved and not errs and n_w == n_d == len(writes) == writers * per
    return {"pass": ok, "detail": {"acked": len(writes), "took_s": round(took, 1), "starved": starved,
                                   "write_max_ms": round(lat[-1] * 1000, 1) if lat else None,
                                   "count_w": n_w, "distinct_w": n_d, "errors": errs[:3]}}


def reads_behind_writer(port, nodes=1_000_000, n_reads=8):
    r = _conn(port)
    _no_timeouts(r)
    _fill(r, nodes)
    cheap_q = "MATCH (n:N) WHERE n.v = 7 RETURN n.v"
    out, lock, lr, wr = [], threading.Lock(), [], []

    def cheap(tag, t0):
        a = time.perf_counter()
        try:
            _conn(port).execute_command("GRAPH.RO_QUERY", "q", cheap_q, "TIMEOUT", "1000")
            res = "ok"
        except ResponseError as e:
            res = "timeout" if "timed out" in str(e).lower() else f"error: {str(e)[:80]}"
        except Exception as e:  # no reply at all (connection lost): the run is incomplete
            res = f"failed: {type(e).__name__}"
        with lock:
            out.append((tag, round((time.perf_counter() - a) * 1000, 1), res))

    def write(t0):
        a = time.perf_counter()
        try:
            _conn(port).execute_command("GRAPH.QUERY", "q", "CREATE (:W {i: 1})")
            wr.append((round(a - t0, 1), round(time.perf_counter() - a, 1), None))
        except Exception as e:
            wr.append((round(a - t0, 1), round(time.perf_counter() - a, 1), f"{type(e).__name__}: {e}"[:120]))

    t0 = time.perf_counter()
    ths = [threading.Thread(target=_long_read, args=(port, 2000, lr))]
    ths[-1].start()
    time.sleep(0.5)
    for _ in range(n_reads):
        ths.append(threading.Thread(target=cheap, args=("before-writer", t0)))
        ths[-1].start()
    time.sleep(0.5)
    ths.append(threading.Thread(target=write, args=(t0,)))
    ths[-1].start()
    time.sleep(1.0)
    for _ in range(n_reads):
        ths.append(threading.Thread(target=cheap, args=("behind-writer", t0)))
        ths[-1].start()
    _join(*ths)
    return judge_reads(out, wr, lr, n_reads)


def judge_reads(out, wr, lr, n_reads) -> dict:
    """reads-behind-writer's verdict: every cheap read answered (no timeout, no error reply). It is
    judged only when all 2 * n_reads reads, the write and the long read completed; otherwise the
    run is incomplete (a crash or an unmeasured repro, as run_repro finds)."""
    failed = [x for x in out if x[2].startswith("failed")]
    if len(out) != 2 * n_reads or failed:
        raise Incomplete(f"{len(out) - len(failed)} of {2 * n_reads} cheap reads got a reply")
    if not wr or wr[0][2] is not None:
        raise Incomplete(f"the write did not complete: {wr[0][2] if wr else 'no reply'}")
    if not lr or lr[0][2] is not None:
        raise Incomplete(f"the long read did not complete: {lr[0][2] if lr else 'no reply'}")
    timeouts = sum(1 for x in out if x[2] == "timeout")
    other = [x for x in out if x[2] not in ("ok", "timeout")]
    return {"pass": timeouts == 0 and not other,
            "detail": {"timeouts": timeouts, "other": other[:3], "write": wr[:1],
                       "long_read_s": round(lr[0][1] - lr[0][0], 1), "reads": sorted(out)}}


#: name -> (THREAD_COUNT, function, args)
REPROS = {
    **{f"write-{m}": (32, write_during_long_read, (m,)) for m in ("none", "create", "set")},
    **{f"constraint-{m}": (32, constraint_during_long_read, (m,)) for m in ("none", "create", "drop", "mandatory")},
    "index-build": (32, index_build_behind_long_read, ()),
    "writer-starvation": (10, writer_starvation, ()),
    "reads-behind-writer": (32, reads_behind_writer, ()),
}


def run_repro(opts: dict, info: dict, tag: str, name: str) -> dict:
    """One repro on a fresh server: {"pass", "detail", "crashed"}. A server that exits or stops
    answering is pass False with crashed True; a repro that could not run is pass None."""
    tc, fn, args = REPROS[name]
    server = harness.Server(info, f"{opts['name_prefix']}-{tag}", opts.get("server_cpus"), opts["net"], tc)
    try:
        server.start()
        res = {**fn(server.port, *args), "crashed": False}
    except Exception as e:
        res = {"pass": None, "crashed": False, "error": f"{type(e).__name__}: {e}"[:300]}
        if not isinstance(e, harness.DockerError):
            try:
                running, code = server.state()
                if not running or harness.server_hung(server.port):
                    res.update(server_alive=False, exit_code=None if running else code)
                    res["pass"], res["crashed"] = False, True
            except harness.DockerError:
                pass  # docker cannot say either: the repro stays unmeasured (pass None)
    finally:
        res_tail = server.stop()
    res["server_tail"] = res_tail[-2000:]
    harness.log(f"{tag}: repro {name}: pass={res.get('pass')} {str(res.get('detail', res.get('error')))[:300]}")
    return res
