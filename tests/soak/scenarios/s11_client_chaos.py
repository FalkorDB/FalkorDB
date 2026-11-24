"""s11_client_chaos — misbehaving clients must not hurt well-behaved ones.

A steady open-loop "victim" workload of cheap, exactly-verifiable queries runs against a
~500k node graph (scale 1.0) while chaos workers (fixed intensity, not scaled) abuse the server:
connection churn, 300-query pipelines, 200k-row results, slow/never-reading raw sockets,
CLIENT KILL mid-query, timeouts, memory hogs vs QUERY_MEM_CAPACITY, malformed queries at a high
rate, parameter edge cases, RO_QUERY with writes, 600 idle connections, UDF throw/loop/flush.

CLI: python -I s11_client_chaos.py --module M --port P --out DIR --duration SEC --scale F
"""
import argparse
import functools
import os
import random
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import soakkit as sk  # noqa: E402
import redis  # noqa: E402

ap = argparse.ArgumentParser()
sk.add_engine_args(ap)  # --module PATH | --image REF
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--duration", type=float, default=60)
ap.add_argument("--scale", type=float, default=0.05)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

G = "victim"          # victim graph
CG = "chaos"          # chaos graph (fixed size)
N = max(5000, int(500_000 * a.scale))
CN = 200_000          # chaos graph nodes (fixed)
PORT = a.port
MEM_CAP = 256 * 1024 * 1024
TIMEOUT_CFG_MS = 5000
WRONG_LOG = out / "wrong_results.log"
PANIC_WORDS = ("panic", "unwrap", "backtrace", "thread '", "rust_backtrace", "index out of bounds",
               "assertion", "segfault")


class WrongResult(Exception):
    pass


def wrong(ctx, what):
    with open(WRONG_LOG, "a") as f:
        f.write(f"{time.time():.1f} {what}\n")
    e = WrongResult(what)
    e.op = "WRONG_RESULT"
    raise e


# ---------------------------------------------------------------- victim ground truth
def nbrs(i):
    return sorted([(i * 7919 + 1) % N, (i * 104729 + 3) % N])


def val(i):
    return i * 3 + 7


def victim_check_point(r, rng):
    i = rng.randrange(N)
    res = sk.query(r, G, "MATCH (n:V {id: $i}) RETURN n.id, n.val", {"i": i}, ro=True)
    rs = sk.rows(res)
    got = [[sk.cval(c) for c in row] for row in rs]
    return i, got == [[i, val(i)]], got


def victim_check_hop(r, rng):
    i = rng.randrange(N)
    res = sk.query(r, G, "MATCH (a:V {id: $i})-[:E]->(b) RETURN b.id ORDER BY b.id", {"i": i}, ro=True)
    got = [sk.cval(row[0]) for row in sk.rows(res)]
    return i, got == nbrs(i), got


def victim_check_range(r, rng):
    s = rng.randrange(N)
    e = min(N, s + 50)
    res = sk.query(r, G, "MATCH (n:V) WHERE n.id >= $s AND n.id < $e RETURN count(n), sum(n.val)",
                   {"s": s, "e": e}, ro=True)
    got = [sk.cval(c) for c in sk.rows(res)[0]]
    exp = [e - s, sum(val(i) for i in range(s, e))]
    return s, got == exp, got


def victim_op(ctx, prefix=""):
    x = ctx.rng.random()
    if x < 0.55:
        op, f = "point", victim_check_point
    elif x < 0.85:
        op, f = "hop", victim_check_hop
    else:
        op, f = "range50", victim_check_range
    k, ok, got = f(ctx.r, ctx.rng)
    if not ok:
        wrong(ctx, f"victim {op} key={k} got={got}")
    return prefix + "victim_" + op


# ---------------------------------------------------------------- chaos helpers
def resp(*args):
    b = [b"*%d\r\n" % len(args)]
    for x in args:
        if not isinstance(x, bytes):
            x = str(x).encode()
        b.append(b"$%d\r\n%s\r\n" % (len(x), x))
    return b"".join(b)


def raw_sock(port, rcvbuf=None):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if rcvbuf:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
    s.settimeout(10)
    s.connect(("127.0.0.1", port))
    return s


def bad(e, extra=""):
    """An error reply that looks like an engine bug (panic text in message)."""
    m = str(e).lower()
    return any(w in m for w in PANIC_WORDS)


class ChaosFail(Exception):
    pass


def fail(op, msg):
    e = ChaosFail(msg)
    e.op = op
    raise e


def tryq(ctx, op, q, ro=False, timeout_ms=None, expect_err=None):
    """Run a query that may legitimately error. Returns (ok, reply_or_msg). Raises on bad error text."""
    try:
        res = sk.query(ctx.r, CG, q, None, ro=ro, timeout_ms=timeout_ms)
        return True, res
    except redis.exceptions.ResponseError as e:
        if bad(e):
            fail(op, f"panic-like error text for {q[:80]!r}: {e}")
        if not str(e).strip():
            fail(op, f"empty error message for {q[:80]!r}")
        return False, str(e)


# ---------------------------------------------------------------- chaos ops
def churn_op(ctx):
    c = sk.connect(PORT)
    c.ping()
    i, ok, got = victim_check_point(c, ctx.rng)
    if not ok:
        wrong(ctx, f"churn point {i} got {got}")
    c.connection_pool.disconnect()
    return "chaos_churn"


def pipeline_op(ctx):
    p = ctx.r.pipeline(transaction=False)
    exp = []
    n = 300
    for _ in range(n):
        i = ctx.rng.randrange(N)
        p.execute_command("GRAPH.RO_QUERY", G,
                          f"CYPHER i={i} MATCH (n:V {{id: $i}}) RETURN n.val", "--compact")
        exp.append(val(i))
    res = p.execute()
    for e, r in zip(exp, res):
        if sk.scalar(r) != e:
            wrong(ctx, f"pipeline got {sk.scalar(r)} expected {e}")
    if len(res) != n:
        wrong(ctx, "pipeline length")
    return "chaos_pipeline300"


def bigresult_op(ctx):
    k = ctx.rng.random()
    if k < 0.4:
        res = sk.query(ctx.r, CG, "MATCH (n:C) RETURN n.id, n.s", ro=True)
        rs = sk.rows(res)
        if len(rs) != CN:
            wrong(ctx, f"bigresult rows {len(rs)}")
        row = rs[ctx.rng.randrange(len(rs))]
        i = sk.cval(row[0])
        if sk.cval(row[1]) != cstr(i):
            wrong(ctx, f"bigresult row {i} string mismatch")
        return "chaos_big_200k_rows"
    if k < 0.7:
        res = sk.query(ctx.r, CG, "UNWIND range(1, 150000) AS x RETURN x, range(1, 20)", ro=True)
        rs = sk.rows(res)
        if len(rs) != 150000:
            wrong(ctx, f"biglist rows {len(rs)}")
        return "chaos_big_150k_lists"
    if k < 0.85:
        res = sk.query(ctx.r, CG, "MATCH (n:C) RETURN collect(n.s)", ro=True)
        v = sk.rows(res)[0][0]
        # a 20 MB single value; check its length only (raw compact array is nested)
        if not v:
            wrong(ctx, "big collect empty")
        return "chaos_big_collect_20MB"
    res = sk.query(ctx.r, CG, "UNWIND range(1, 100) AS x RETURN x, $s + toString(x)",
                   {"s": "ab" * 10000}, ro=True)
    rs = sk.rows(res)
    if len(rs) != 100 or any(len(sk.cval(r_[1])) != 20000 + len(str(sk.cval(r_[0]))) for r_ in rs):
        wrong(ctx, "big string rows")
    return "chaos_big_20KB_strings"


def cstr(i):
    return ("%08d" % i) * 12  # 96 chars


def slowread_op(ctx):
    s = raw_sock(PORT, rcvbuf=4096)
    try:
        s.sendall(resp("GRAPH.QUERY", CG, "MATCH (n:C) RETURN n.id, n.s", "--compact"))
        mode = ctx.rng.choice(["never", "slow", "slow", "partial"])
        if mode == "never":
            time.sleep(ctx.rng.uniform(0.5, 3))
        elif mode == "partial":
            s.recv(100)
            time.sleep(ctx.rng.uniform(0.2, 1.5))
        else:
            end = time.time() + ctx.rng.uniform(1, 4)
            s.settimeout(1)
            while time.time() < end:
                try:
                    if not s.recv(512):
                        break
                except socket.timeout:
                    pass
                time.sleep(0.05)
    except socket.timeout:
        mode = "stalled"  # server did not answer within 10 s (queue backlog): not a protocol failure
    finally:
        s.close()
    return "chaos_slowreader_" + mode


def killmid_op(ctx):
    c = sk.connect(PORT)
    cid = c.client_id()
    q = ctx.rng.choice(["UNWIND range(1, 300000000) AS x RETURN sum(x)",
                        "MATCH (n:C) RETURN n.id, n.s",
                        "UNWIND range(1, 2000000) AS x RETURN x, toString(x)"])
    box = {}

    def run():
        try:
            box["r"] = sk.query(c, CG, q, ro=True, timeout_ms=3000)
        except Exception as e:  # noqa: BLE001
            box["e"] = e

    th = threading.Thread(target=run)
    th.start()
    time.sleep(ctx.rng.uniform(0.005, 0.3))
    ctx.r.execute_command("CLIENT", "KILL", "ID", cid)
    th.join(30)
    if th.is_alive():
        fail("chaos_killmid", "killed client's query thread still blocked after 30s")
    e = box.get("e")
    if e is not None and not isinstance(e, (redis.exceptions.ConnectionError, redis.exceptions.ResponseError,
                                            redis.exceptions.TimeoutError)):
        fail("chaos_killmid", f"unexpected {type(e).__name__}: {e}")
    if isinstance(e, redis.exceptions.ResponseError) and bad(e):
        fail("chaos_killmid", f"panic-like error: {e}")
    return "chaos_kill_midquery"


def timeout_op(ctx):
    t = ctx.rng.random()
    if t < 0.6:
        ms = ctx.rng.choice([1, 10, 50, 100])
        ok, r = tryq(ctx, "chaos_timeout_arg", "UNWIND range(1, 300000000) AS x RETURN sum(x)", ro=True,
                     timeout_ms=ms)
        if ok:
            fail("chaos_timeout_arg", f"query with timeout {ms}ms returned a result")
        if "timed out" not in r.lower() and "mem consumption exceeded" not in r.lower():  # C materialises range()
            fail("chaos_timeout_arg", f"expected timeout (or mem-cap) error, got {r}")
        return "chaos_timeout_arg_read"
    # config default timeout (TIMEOUT_CFG_MS) must cut a runaway query; allow generous slack
    t0 = time.time()
    ok, r = tryq(ctx, "chaos_timeout_cfg", "UNWIND range(1, 100000) AS a UNWIND range(1, 100000) AS b RETURN sum(a + b)", ro=True)
    dt = time.time() - t0
    if ok:
        fail("chaos_timeout_cfg", "runaway query succeeded")
    if "timed out" not in r.lower():
        fail("chaos_timeout_cfg", f"expected timeout error, got {r}")
    if dt > TIMEOUT_CFG_MS / 1000 * 3 + 5:
        fail("chaos_timeout_cfg", f"config timeout {TIMEOUT_CFG_MS}ms enforced only after {dt:.1f}s")
    return "chaos_timeout_cfg_read"


def memhog_op(ctx):
    q = ctx.rng.choice(["UNWIND range(1, 50000000) AS x RETURN collect(x)",
                        "RETURN range(0, 100000000)",
                        "UNWIND range(1, 30000000) AS x RETURN x, count(x)",
                        "MATCH (a:C), (b:C) RETURN count(a)"])
    ok, r = tryq(ctx, "chaos_memhog", q, ro=True, timeout_ms=3000)
    if ok:
        if "count(a)" in q:
            return "chaos_memhog_ok"  # cartesian count may legitimately stream under the cap
        fail("chaos_memhog", f"memory hog not stopped by QUERY_MEM_CAPACITY: {q}")
    low = r.lower()
    if "mem consumption exceeded" not in low and "timed out" not in low:
        fail("chaos_memhog", f"unexpected error for {q}: {r}")
    return "chaos_memhog_stopped"


# (query, must_fail)
ERR_QUERIES = [
    ("RETRN 1", True), ("RETURN nofunc(1)", True), ("RETURN abs(1,2)", True), ("RETURN 1/0", True),
    ("RETURN 'a' - 1", True), ("MATCH (n RETURN n", True), ("RETURN $missing", True),
    ("UNWIND 5 AS x RETURN x", False), ("RETURN 1 +", True), ("", True), ("RETURN range(1,2,0)", True),
    ("MATCH (n:V) RETURN n.id ORDER BY", True), ("CREATE INDEX ON :", True),
    ("RETURN substring('abc', -1)", True), ("MATCH (n) WHERE n RETURN n", True),
    ("RETURN [1,2,3][1.5]", True), ("RETURN {a:1}[0]", True), ("RETURN toUpper(1)", True),
    ("MATCH (a)-[r:E*..]->(b) RETURN r.x.y", False), ("RETURN size(5)", True),
    ("CALL db.nosuchproc()", True), ("CALL db.labels() YIELD nosuch RETURN nosuch", True),
    ("MATCH (n:V) DELETE m", True), ("WITH 1 AS x WITH y RETURN y", True), ("RETURN 1 AS x, 2 AS x", True),
    ("RETURN datetime('garbage')", True),
    ("UNWIND range(1,3) AS x RETURN x LIMIT -1", True), ("RETURN toInteger('x') + 1", False),
    ("RETURN point({latitude: 1})", True), ("RETURN reduce(s=0, x IN 5 | s+x)", True),
    ("RETURN [x IN 5 WHERE x]", True), ("RETURN collect()", True), ("RETURN 1;;;", False),
]
RO_WRITES = ["CREATE (:Z)", "MATCH (n:C) WHERE n.id = 1 SET n.x = 1", "MATCH (n:C {id: 1}) DELETE n",
             "MERGE (:Z {a: 1})", "CREATE INDEX FOR (n:Z) ON (n.a)", "MATCH (n:C {id:1}) REMOVE n.s",
             "UNWIND range(1,3) AS x CREATE (:Z {x: x})", "MATCH (n:C {id:1}) SET n:Z"]
GOOD = ["MATCH (n:C {id: 5}) RETURN n.id", "RETURN 1", "UNWIND range(1,5) AS x RETURN sum(x)"]


def errors_op(ctx):
    rng = ctx.rng
    k = rng.random()
    if k < 0.55:
        q, must = rng.choice(ERR_QUERIES)
        ok, r = tryq(ctx, "chaos_err_query", q)
        if ok and must:
            fail("chaos_err_query", f"query expected to fail but succeeded: {q!r}")
        return "chaos_err_query"
    if k < 0.75:
        q = rng.choice(RO_WRITES)
        ok, r = tryq(ctx, "chaos_ro_write", q, ro=True)
        if ok:
            fail("chaos_ro_write", f"RO_QUERY accepted a write: {q!r}")
        return "chaos_ro_write_rejected"
    if k < 0.9:
        # truncated / mutated valid query
        base = rng.choice(["MATCH (a:C {id: 1})-[:X]->(b) WHERE a.id > 3 RETURN a, count(b) ORDER BY a.id LIMIT 5",
                           "UNWIND range(1,10) AS x WITH x, x*2 AS y RETURN collect({x:x, y:y})",
                           "MERGE (n:Z {a: 1}) ON CREATE SET n.b = 2 RETURN n"])
        cut = rng.randrange(1, len(base))
        q = base[:cut] if rng.random() < 0.6 else base[:cut] + rng.choice("(){}[]'\"`,:.|*-<>$")
        ok, r = tryq(ctx, "chaos_err_mutated", q, ro=True)
        return "chaos_err_mutated"
    # random garbage bytes
    g = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 200)))
    try:
        ctx.r.execute_command("GRAPH.RO_QUERY", CG, g)
    except redis.exceptions.ResponseError as e:
        if bad(e):
            fail("chaos_err_garbage", str(e))
    q = rng.choice(GOOD)
    ok, r = tryq(ctx, "chaos_err_after", q, ro=True)  # connection still healthy after errors
    if not ok:
        fail("chaos_err_after", f"good query failed after garbage: {r}")
    # wrong arity / unknown command forms
    for args in (("GRAPH.QUERY",), ("GRAPH.QUERY", CG), ("GRAPH.RO_QUERY", CG, "RETURN 1", "timeout"),
                 ("GRAPH.RO_QUERY", CG, "RETURN 1", "timeout", "abc"), ("GRAPH.EXPLAIN", CG),
                 ("GRAPH.QUERY", CG, "RETURN 1", "bogusflag"), ("GRAPH.CONFIG", "GET", "NOSUCH"),
                 ("GRAPH.UDF", "BOGUS")):
        try:
            ctx.r.execute_command(*args)
        except redis.exceptions.ResponseError as e:
            if bad(e):
                fail("chaos_err_arity", f"{args}: {e}")
    return "chaos_err_garbage_arity"


def raw_cypher(ctx, op, q):
    """Run a raw (possibly huge) query string; return (ok, reply|msg)."""
    try:
        return True, ctx.r.execute_command("GRAPH.RO_QUERY", CG, q, "--compact")
    except redis.exceptions.ResponseError as e:
        if bad(e):
            fail(op, f"panic-like error text for {q[:80]!r}: {e}")
        return False, str(e)


def nest(depth, open_, close, leaf="1"):
    return open_ * depth + leaf + close * depth


def params_op(ctx):
    rng = ctx.rng
    k = rng.randrange(8)
    op = "chaos_params"
    if k == 0:
        n = rng.choice([100_000, 1_000_000, 4_000_000])
        ok, r = raw_cypher(ctx, op, f'CYPHER p="{"x" * n}" RETURN size($p)')
        if not ok or sk.scalar(r) != n:
            fail(op, f"huge string param size {n}: {str(r)[:100]}")
    elif k == 1:
        s = "héllo 世界 😀 \\u00e9 \\\\ ñ " + chr(0x1F600) * 3
        ok, r = raw_cypher(ctx, op, f'CYPHER p="{s}" RETURN $p')
        if not ok:
            fail(op, f"unicode param rejected: {r}")
        got = sk.scalar(r)
        if "世界" not in got or "😀" not in got:
            wrong(ctx, f"unicode param mangled: {got!r}")
    elif k == 2:
        for q in ("RETURN 0.0/0.0", "RETURN 1e308*10", "RETURN -1e308*10", "CYPHER p=1e999 RETURN $p",
                  "UNWIND [1.0, 0.0/0.0, 2.0] AS x RETURN max(x), min(x), sum(x)",
                  "UNWIND [1.0, 0.0/0.0, 2.0] AS x RETURN x ORDER BY x",
                  "RETURN toInteger(0.0/0.0), toInteger(1e300)", "RETURN round(0.0/0.0), ceil(1e308*10)"):
            raw_cypher(ctx, op, q)
    elif k == 3:
        for q in ("RETURN 9223372036854775808", "RETURN -9223372036854775808", "RETURN -9223372036854775809",
                  "RETURN 9223372036854775807 * 2", "RETURN toInteger('99999999999999999999')",
                  "RETURN abs(-9223372036854775807 - 1)", "RETURN 0x7fffffffffffffff + 1",
                  "RETURN range(9223372036854775800, 9223372036854775807)",
                  "RETURN 2 ^ 1000", "RETURN 5 % 0", "RETURN -9223372036854775808 / -1",
                  "CYPHER p=99999999999999999999 RETURN $p"):
            raw_cypher(ctx, op, q)
    elif k == 4:
        d = rng.choice([10, 50, 100])
        ok, r = raw_cypher(ctx, op, "CYPHER p=" + nest(d, "[", "]") + " RETURN $p")
        if not ok:
            fail(op, f"nested list depth {d} rejected: {r}")
        ok, r = raw_cypher(ctx, op, "CYPHER p=" + nest(d, "{a:", "}") + " RETURN $p")
        if not ok:
            fail(op, f"nested map depth {d} rejected: {r}")
    elif k == 5:
        n = 100_000
        ok, r = raw_cypher(ctx, op, "CYPHER p=[" + ",".join(map(str, range(n))) + "] RETURN size($p), reduce(s=0, x IN $p | s+x)")
        if not ok:
            fail(op, f"100k-element list param rejected: {r}")
        v = [sk.cval(c) for c in sk.rows(r)[0]]
        if v != [n, n * (n - 1) // 2]:
            wrong(ctx, f"big list param got {v}")
    elif k == 6:
        n = 5000
        ok, r = raw_cypher(ctx, op, "CYPHER p={" + ",".join(f"k{i}:{i}" for i in range(n)) + "} RETURN size(keys($p)), $p.k4999")
        if not ok:
            fail(op, f"5000-key map param rejected: {r}")
        v = [sk.cval(c) for c in sk.rows(r)[0]]
        if v != [n, 4999]:
            wrong(ctx, f"big map param got {v}")
    else:
        for q in ('CYPHER p="a\\"b\'c" RETURN $p', 'CYPHER p=null RETURN $p IS NULL', "CYPHER p=[] RETURN size($p)",
                  "CYPHER p={} RETURN keys($p)", 'CYPHER p="" RETURN size($p)', "CYPHER p= RETURN 1",
                  "CYPHER p=1 p=2 RETURN $p", "CYPHER 1p=1 RETURN 1", 'CYPHER p="unterminated RETURN 1'):
            raw_cypher(ctx, op, q)
    return op


def flood_init(ctx):
    ctx.state["conns"] = []
    for _ in range(300):
        c = sk.connect(PORT)
        c.ping()
        ctx.state["conns"].append(c)


def flood_op(ctx):
    cs = ctx.state["conns"]
    for c in ctx.rng.sample(cs, 20):
        if not c.ping():
            fail("chaos_flood_ping", "ping false")
    c = ctx.rng.choice(cs)
    i, ok, got = victim_check_point(c, ctx.rng)
    if not ok:
        wrong(ctx, f"flood point {i} got {got}")
    return "chaos_flood_conn_query"


UDF_LIB = """
function Add(a, b) { return a + b; }
function Thrower() { throw new Error('chaos boom'); }
function Looper() { while (true) {} }
function Big() { var o = []; for (var i = 0; i < 1000; i++) o.push({i: i, s: 'x'.repeat(100)}); return o; }
function Echo(x) { return x; }
falkor.register('Add', Add); falkor.register('Thrower', Thrower); falkor.register('Looper', Looper);
falkor.register('Big', Big); falkor.register('Echo', Echo);
"""


def udf_init(ctx):
    ctx.state["n"] = 0
    ctx.state["loop_ok"] = UDF_LOOP
    try:
        ctx.r.execute_command("GRAPH.UDF", "LOAD", "REPLACE", "chaoslib", UDF_LIB)
        ctx.state["ok"] = True
    except redis.exceptions.ResponseError:
        ctx.state["ok"] = False


def udf_op(ctx):
    if not ctx.state["ok"]:
        time.sleep(1)
        return "chaos_udf_unsupported"
    ctx.state["n"] += 1
    n = ctx.state["n"]
    r = ctx.r
    x = ctx.rng.randrange(1000)
    ok, res = tryq(ctx, "chaos_udf", f"RETURN chaoslib.Add({x}, 5)", ro=True)
    if not ok or sk.scalar(res) != x + 5:
        fail("chaos_udf", f"Add({x},5) -> {res}")
    ok, res = tryq(ctx, "chaos_udf", "RETURN chaoslib.Thrower()", ro=True)
    if ok or "boom" not in res:
        fail("chaos_udf", f"Thrower did not raise 'boom': {res}")
    ok, res = tryq(ctx, "chaos_udf", "RETURN size(chaoslib.Big())", ro=True)
    if not ok or sk.scalar(res) != 1000:
        fail("chaos_udf", f"Big -> {res}")
    ok, res = tryq(ctx, "chaos_udf", 'RETURN chaoslib.Echo("é世")', ro=True)
    if not ok or sk.scalar(res) != "é世":
        fail("chaos_udf", f"Echo -> {res}")
    if n % 20 == 0 and ctx.state["loop_ok"]:  # runaway UDF: needs a timeout to terminate
        t0 = time.time()
        ok, res = tryq(ctx, "chaos_udf_loop", "RETURN chaoslib.Looper()", ro=True)
        dt = time.time() - t0
        if ok:
            fail("chaos_udf_loop", "looping UDF returned a result")
        if dt > 120:
            fail("chaos_udf_loop", f"looping UDF terminated only after {dt:.0f}s: {res}")
        ctx.state.setdefault("loop_s", []).append(dt)
    if n % 7 == 0:
        r.execute_command("GRAPH.UDF", "FLUSH")
        ok, res = tryq(ctx, "chaos_udf", f"RETURN chaoslib.Add(1, 2)", ro=True)
        if ok:
            fail("chaos_udf", "UDF callable after FLUSH")
        r.execute_command("GRAPH.UDF", "LOAD", "REPLACE", "chaoslib", UDF_LIB)
    return "chaos_udf_cycle"


# ---------------------------------------------------------------- setup
# A looping UDF is only safe where UDFs have a timeout (Rust: 30 s cap). The C module has none:
# the loop would pin a thread forever and block every writer. Override with S11_UDF_LOOP=0/1.
srv = sk.Server(a.module, PORT, out, extra=["--maxclients", "10000"])
UDF_LOOP = os.environ.get("S11_UDF_LOOP", "1" if srv.engine() == "rust" else "0") == "1"
mon = sk.Monitor(srv, out)
mon.start()
checks = sk.Checks(out)
r = srv.client()
meta = {"scenario": "s11_client_chaos", "scale": a.scale, "victim_nodes": N, "victim_edges": 2 * N,
        "chaos_nodes": CN, "duration": a.duration}

mon.mark("ingest")
t_ing = time.time()
B = 5000
for i in range(0, N, B):
    sk.query(r, G, "UNWIND range($a, $b) AS i CREATE (:V {id: i, val: i * 3 + 7})",
             {"a": i, "b": min(N, i + B) - 1})
sk.query(r, G, "CREATE INDEX FOR (n:V) ON (n.id)")
for i in range(0, N, B):
    for mul, add in ((7919, 1), (104729, 3)):
        sk.query(r, G, f"UNWIND range($a, $b) AS i MATCH (u:V {{id: i}}) MATCH (v:V {{id: (i * {mul} + {add}) % $n}}) "
                       f"CREATE (u)-[:E]->(v)", {"a": i, "b": min(N, i + B) - 1, "n": N})
for i in range(0, CN, B):
    sk.query(r, CG,
             "UNWIND range($a, $b) AS i WITH i, ('00000000' + toString(i)) AS z "
             "WITH i, substring(z, size(z) - 8, 8) AS p "
             "CREATE (:C {id: i, s: p + p + p + p + p + p + p + p + p + p + p + p})",
             {"a": i, "b": min(CN, i + B) - 1})
sk.query(r, CG, "CREATE INDEX FOR (n:C) ON (n.id)")
meta["ingest_s"] = time.time() - t_ing
nodes = sk.scalar(sk.query(r, G, "MATCH (n:V) RETURN count(n)"))
edges = sk.scalar(sk.query(r, G, "MATCH ()-[e:E]->() RETURN count(e)"))
cn = sk.scalar(sk.query(r, CG, "MATCH (n:C) RETURN count(n)"))
checks.add("ingest victim node count", nodes == N, f"{nodes} != {N}", "critical")
checks.add("ingest victim edge count", edges == 2 * N, f"{edges} != {2 * N}", "critical")
checks.add("ingest chaos node count", cn == CN, f"{cn} != {CN}", "critical")
cs = sk.scalar(sk.query(r, CG, "MATCH (n:C {id: 777}) RETURN n.s"))
checks.add("chaos string ground truth", cs == cstr(777), f"{cs!r}", "critical")

rl = sk.scalar(sk.query(r, G, "RETURN size(reduce(s='', i IN range(1,10000) | s+'ab'))", ro=True))
meta["reduce_string_len"] = rl
checks.add("reduce() string concat result length exact (20000)", rl == 20000, f"got {rl}", "minor")

# server limits for the chaos phase
for k, v in (("QUERY_MEM_CAPACITY", MEM_CAP), ("TIMEOUT", TIMEOUT_CFG_MS)):
    try:
        r.execute_command("GRAPH.CONFIG", "SET", k, v)
        got = int(r.execute_command("GRAPH.CONFIG", "GET", k)[1])
        checks.add(f"GRAPH.CONFIG SET {k} takes effect", got == v, f"got {got}", "major")
    except redis.exceptions.ResponseError as e:
        checks.add(f"GRAPH.CONFIG SET {k} supported", False, str(e), "minor")
try:
    r.execute_command("GRAPH.UDF", "LOAD", "REPLACE", "probe", "function F(){return 1}\nfalkor.register('F',F);")
    r.execute_command("GRAPH.UDF", "FLUSH")
    meta["udf_supported"] = True
    meta["udf_loop_tested"] = UDF_LOOP
except redis.exceptions.ResponseError as e:
    meta["udf_supported"] = False
    checks.add("UDF supported", False, str(e), "minor")

# ---------------------------------------------------------------- victim baseline (no chaos)
base_s = min(60.0, max(10.0, 0.15 * a.duration))
VICT = dict(rate=100, count=4)
mon.mark("baseline")
sk.run_load(srv, out, [sk.WorkerSpec(functools.partial(victim_op, prefix="base_"), name="vbase", **VICT)],
            base_s, tag="")
rss_base = srv_rss = None
try:
    import psutil
    rss_base = psutil.Process(srv.pid).memory_info().rss / 2**20
except Exception:  # noqa: BLE001
    pass
um_base = r.info().get("used_memory", 0) / 2**20
meta["rss_baseline_mb"], meta["used_mem_baseline_mb"] = rss_base, um_base

# ---------------------------------------------------------------- chaos + victim
W = sk.WorkerSpec
workers = [
    W(victim_op, name="victim", **VICT),
    W(churn_op, rate=60, count=2, name="churn"),
    W(pipeline_op, rate=2, count=2, name="pipeline"),
    W(bigresult_op, rate=0.5, count=1, name="bigresult"),
    W(slowread_op, rate=0.5, count=2, name="slowread"),
    W(killmid_op, rate=1, count=1, name="killmid"),
    W(timeout_op, rate=2, count=1, name="timeouts"),
    W(memhog_op, rate=0.3, count=1, name="memhog"),
    W(errors_op, rate=150, count=2, name="errors"),
    W(params_op, rate=15, count=1, name="params"),
    W(flood_op, rate=20, count=2, name="flood", init=flood_init),
    W(udf_op, rate=1, count=1, name="udf", init=udf_init),
]
_only = os.environ.get("S11_ONLY")  # diagnostics: comma list of worker names to keep (victim always kept)
if _only:
    workers = [w for w in workers if w.name == "victim" or w.name in _only.split(",")]
mon.mark("soak")
t_soak0 = time.time()
sk.run_load(srv, out, workers, a.duration, seed=2)
soak_s = time.time() - t_soak0

# ---------------------------------------------------------------- verify
mon.mark("verify")
alive = srv.alive()
checks.add("server alive after chaos", alive, "", "critical")
if alive:
    r = srv.client()
    summ = sk.analyze(out, srv)
    ops = summ["ops"]
    # chaos op errors = real failures (all chaos ops handle their own expected errors)
    for op, d in sorted(ops.items()):
        if op == "WRONG_RESULT":
            checks.add("victim/chaos results always exactly right", False,
                       f"{d['count']} wrong results; see {WRONG_LOG}", "critical")
    if "WRONG_RESULT" not in ops:
        checks.add("victim/chaos results always exactly right", True, "", "critical")
    vops = {o: d for o, d in ops.items() if o.startswith("victim_")}
    vbase = {o: d for o, d in ops.items() if o.startswith("base_victim_")}
    verr = sum(d["errors"] for d in vops.values()) + sum(d["errors"] for d in vbase.values())
    checks.add("victim queries never error", verr == 0, f"{verr} errors; top={summ['top_errors'][:3]}", "major")
    other_err = {o: d["errors"] for o, d in ops.items()
                 if o.startswith("chaos_") and d["errors"] and not o.startswith("chaos_kill")}
    chaos_errs = [(m, c) for m, c in summ["top_errors"] if "WrongResult" not in m]
    checks.add("chaos workers saw no unexpected failures", not other_err,
               f"{other_err}; top={chaos_errs[:4]}", "major")
    mis = {o: d["errors"] for o, d in ops.items() if o in ("_missed_schedule",)}
    # victim SLO: p99 under chaos vs baseline
    for kind in ("point", "hop", "range50"):
        b, c = ops.get(f"base_victim_{kind}"), ops.get(f"victim_{kind}")
        if not b or not c:
            continue
        bp, cp = b["p99_ms"], c["p99_ms"]
        limit = max(10 * bp, 100.0)
        checks.add(f"victim {kind} p99 under chaos within SLO", cp <= limit,
                   f"p99 {cp:.1f} ms vs baseline {bp:.1f} ms (limit {limit:.0f} ms)", "major")
        meta[f"victim_{kind}_p99_base_ms"], meta[f"victim_{kind}_p99_chaos_ms"] = bp, cp
        meta[f"victim_{kind}_p50_base_ms"], meta[f"victim_{kind}_p50_chaos_ms"] = b["p50_ms"], c["p50_ms"]
    late = ops.get("_late_start", {}).get("count", 0)
    missed = ops.get("_missed_schedule", {}).get("count", 0)
    meta["late_start_ops"], meta["missed_schedule_ops"] = late, missed
    vtotal = sum(d["count"] for d in vops.values())
    checks.add("victim open-loop schedule kept (few late/missed sends)", missed < 0.02 * max(vtotal, 1),
               f"late={late} missed={missed} victim_ops={vtotal}", "major")
    meta["chaos_op_counts"] = {o: d["count"] for o, d in ops.items() if o.startswith("chaos_")}
    # checks that the chaos really ran at intended intensity
    churn_n = ops.get("chaos_churn", {}).get("count", 0)
    checks.add("connection churn volume >= 1000 connects", churn_n >= min(1000, 20 * soak_s * 0.5), f"{churn_n}", "minor")
    meta["max_clients_seen"] = None
    try:
        mc = [float(l.split(",")[5]) for l in (out / "monitor.csv").read_text().splitlines()[1:] if l.split(",")[5] != "nan"]
        meta["max_clients_seen"] = max(mc)
        checks.add("500+ concurrent connections were reached", max(mc) >= 600, f"max {max(mc)}", "minor")
    except Exception:  # noqa: BLE001
        pass

    # exhaustive-ish victim re-verification
    rng = random.Random(99)
    bad_n = 0
    for _ in range(2000):
        for f in (victim_check_point, victim_check_hop, victim_check_range):
            k, ok, got = f(r, rng)
            if not ok:
                bad_n += 1
    checks.add("post-chaos victim data intact (6000 spot checks)", bad_n == 0, f"{bad_n} mismatches", "critical")
    nodes = sk.scalar(sk.query(r, G, "MATCH (n:V) RETURN count(n)"))
    edges = sk.scalar(sk.query(r, G, "MATCH ()-[e:E]->() RETURN count(e)"))
    checks.add("post-chaos victim counts intact", nodes == N and edges == 2 * N, f"{nodes} {edges}", "critical")
    cn = sk.scalar(sk.query(r, CG, "MATCH (n:C) RETURN count(n)"))
    checks.add("post-chaos chaos-graph :C count intact (RO ops never wrote)", cn == CN, f"{cn} != {CN}", "critical")
    z = sk.scalar(sk.query(r, CG, "MATCH (n:Z) RETURN count(n)"))
    # :Z can only come from the mutating MERGE in ERR_QUERIES / mutated queries on the RW path (none): RO only
    checks.add("no :Z nodes (RO_QUERY writes never applied)", z == 0, f"{z} :Z nodes", "critical")
    tmp = sk.scalar(sk.query(r, CG, "MATCH (n:Tmp) RETURN count(n)"))
    meta["tmp_nodes_after_timed_out_writes"] = tmp
    sk.query(r, CG, "MATCH (n:Tmp) DELETE n")
    # write behind a slow reader: known design (C behaves the same) - measure and record, no failure
    try:
        res = {}

        def tq(label, delay, f):
            time.sleep(delay)
            c = sk.connect(PORT)
            t = time.time()
            try:
                f(c)
            except Exception:  # noqa: BLE001
                pass
            res[label] = time.time() - t

        ths = [threading.Thread(target=tq, args=("slow_read_s", 0, lambda c: sk.query(
                   c, CG, "UNWIND range(1, 100000) AS a UNWIND range(1, 1500) AS b RETURN sum(a + b)",
                   ro=True, timeout_ms=15000))),
               threading.Thread(target=tq, args=("write_s", 0.5, lambda c: sk.query(
                   c, CG, "MATCH (n:C {id: 1}) SET n.touch = 1"))),
               threading.Thread(target=tq, args=("read_after_pending_write_s", 1.0, lambda c: sk.query(
                   c, G, "MATCH (n:V {id: 1}) RETURN n.val", ro=True)))]
        [t.start() for t in ths]
        [t.join(60) for t in ths]
        meta["write_behind_slow_read"] = {k: round(v, 2) for k, v in res.items()}
    except Exception as e:  # noqa: BLE001
        meta["write_behind_slow_read"] = repr(e)
    # kill-mid-query: how long does the server keep burning CPU for an orphaned query?
    try:
        c = sk.connect(PORT)
        cid = c.client_id()
        th = threading.Thread(target=lambda: _ignore(lambda: sk.query(
            c, CG, "UNWIND range(1, 1000000000) AS x RETURN sum(x)", ro=True, timeout_ms=15000)))
        def _ignore(f):
            try:
                f()
            except Exception:  # noqa: BLE001
                pass
        th.start()
        time.sleep(0.5)
        r.execute_command("CLIENT", "KILL", "ID", cid)
        th.join(10)
        t0 = time.time()
        pr = psutil.Process(srv.pid)
        pr.cpu_percent(None)
        busy = 0
        idle_at = None
        while time.time() - t0 < 30:
            time.sleep(1)
            if pr.cpu_percent(None) < 30:
                idle_at = time.time() - t0
                break
        meta["orphaned_query_busy_s_after_client_kill"] = idle_at if idle_at is not None else ">30"
        checks.add("CLIENT KILLed query is cancelled (CPU idle within 5s of the kill)",
                   idle_at is not None and idle_at <= 5, f"idle after {idle_at} (query timeout was 15s)", "minor")
    except Exception as e:  # noqa: BLE001
        checks.add("orphaned query probe ran", False, repr(e), "minor")

    # quiet period #1, then a second (shorter) chaos round, quiet #2: no growth between rounds = no leak
    def measure(label, quiet):
        mon.mark(label)
        time.sleep(quiet)
        pr = psutil.Process(srv.pid)
        inf = r.info()
        return pr.memory_info().rss / 2**20, inf.get("used_memory", 0) / 2**20, inf.get("connected_clients", 0)

    quiet = min(60.0, max(10.0, 0.05 * a.duration))
    rss1, um1, clients = measure("quiet", quiet)
    rss_max = max((float(l.split(",")[2]) for l in (out / "monitor.csv").read_text().splitlines()[1:]
                   if l.split(",")[2] != "nan"), default=0)
    meta.update({"rss_after_quiet_mb": rss1, "used_mem_after_quiet_mb": um1,
                 "connected_clients_after_quiet": clients, "quiet_s": quiet, "rss_peak_mb": rss_max})
    checks.add("used_memory returns near baseline after quiet period", um1 <= um_base * 1.3 + 100,
               f"{um1:.0f} MB vs baseline {um_base:.0f} MB (peak RSS {rss_max:.0f})", "major")
    checks.add("RSS back within 1.5x+300MB of baseline after quiet (allocator retention tolerated?)",
               rss1 <= rss_base * 1.5 + 300,
               f"{rss1:.0f} MB vs baseline {rss_base:.0f} MB (peak RSS {rss_max:.0f})", "minor")
    checks.add("no leaked client connections after chaos", clients <= 5, f"{clients} connected", "major")
    k, ok, got = victim_check_point(r, rng)
    checks.add("victim query correct after quiet", ok, str(got), "critical")
    r2_s = 3.0 if os.environ.get("S11_ONLY") else max(15.0, 0.2 * a.duration)
    mon.mark("soak2")
    sk.run_load(srv, out, workers, r2_s, seed=3, tag="r2_")
    rss2, um2, clients2 = measure("quiet2", quiet)
    meta.update({"rss_after_quiet2_mb": rss2, "used_mem_after_quiet2_mb": um2, "round2_s": r2_s})
    checks.add("no memory growth between chaos rounds (RSS)", rss2 <= rss1 * 1.15 + 150,
               f"after round 1: {rss1:.0f} MB, after round 2: {rss2:.0f} MB", "major")
    checks.add("no memory growth between chaos rounds (used_memory)", um2 <= um1 * 1.15 + 50,
               f"{um1:.0f} -> {um2:.0f} MB", "major")
    checks.add("no leaked client connections after round 2", clients2 <= 5, f"{clients2}", "major")
    bad_n = 0
    for _ in range(300):
        for f in (victim_check_point, victim_check_hop, victim_check_range):
            if not f(r, rng)[1]:
                bad_n += 1
    checks.add("victim data intact after round 2", bad_n == 0, f"{bad_n} mismatches", "critical")
    try:
        r.execute_command("GRAPH.CONFIG", "SET", "QUERY_MEM_CAPACITY", 0)
        r.execute_command("GRAPH.CONFIG", "SET", "TIMEOUT", 0)
        r.execute_command("GRAPH.UDF", "FLUSH")
    except redis.exceptions.ResponseError:
        pass
crash = srv.crashed()
checks.add("no crash/panic markers in server log", not crash, "; ".join(crash[:3]), "critical")
checks.add("server alive at end", srv.alive(), "", "critical")
mon.stop()
s = sk.analyze(out, srv)
srv.stop()

# ---------------------------------------------------------------- extreme-depth probes on a throwaway server
# (a C build overflows its stack on deeply nested expressions; Rust must reject them cleanly)
probe_dir = out / "depthprobe"
PROBE_TIMEOUT_S = 60
res_probe = {}
for name, q in (("parens5000", "RETURN " + "(" * 5000 + "1" + ")" * 5000),
                ("lists5000", "RETURN " + "[" * 5000 + "1" + "]" * 5000),
                ("plus20000", "RETURN " + "+".join(["1"] * 20000)),
                ("and20000", "RETURN " + " AND ".join(["true"] * 20000)),
                ("maps2000", "RETURN " + "{a:" * 2000 + "1" + "}" * 2000)):
    try:
        ps = sk.Server(a.module, PORT + 5, probe_dir, name=f"probe-{name}")
        # bounded: C v4.22.4 spins for minutes on parens5000 without answering, which would
        # hold the whole scenario to its cap
        pc = sk.connect(PORT + 5, timeout=PROBE_TIMEOUT_S)
        try:
            pc.execute_command("GRAPH.QUERY", "p", q)
            res_probe[name] = "ok"
        except redis.exceptions.ResponseError as e:
            res_probe[name] = "error-reply: " + str(e)[:60]
        except redis.exceptions.TimeoutError:
            res_probe[name] = f"no reply in {PROBE_TIMEOUT_S} s (hung)"
        except redis.exceptions.ConnectionError:
            res_probe[name] = "CONNECTION LOST (crash)"
        pa = ps.alive()
        pcr = ps.crashed()
        ps.stop()
        if not pa or pcr:
            res_probe[name] += " / server dead" if not pa else " / panic logged"
    except Exception as e:  # noqa: BLE001
        res_probe[name] = f"harness error {e!r}"
meta["depth_probe"] = res_probe
dead = [k for k, v in res_probe.items() if "crash" in v or "dead" in v or "panic" in v]
cc = sk.Checks(out)
cc.items = list(checks.items)
cc.add("deeply nested expressions never crash the server", not dead, f"{res_probe}", "critical")
hung = [k for k, v in res_probe.items() if "hung" in v]
cc.add(f"deeply nested expressions are answered within {PROBE_TIMEOUT_S} s", not hung, f"{res_probe}", "critical")
s = sk.analyze(out, None)
s["server_log_crash_markers"] = crash
s["meta"] = meta
(out / "summary.json").write_text(sk.json.dumps(s, indent=1, default=str))
sk.print_summary(s)
print("META", sk.json.dumps(meta, default=str))
