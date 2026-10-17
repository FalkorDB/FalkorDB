"""Live Rust-vs-C repros for proofs/redis_layer (see RedisLayer.lean header).

Start Rust on 18430 and C on 18431, e.g.
  redis-server --port 18430 --save "" --loadmodule target/release/libfalkordb.dylib
  redis-server --port 18431 --save "" --loadmodule bin/macos-arm64v8-release/falkordb.so
then: venv/bin/python proofs/redis_layer/repro_live.py
"""
import re, struct, redis

R = redis.Redis(port=18430, socket_timeout=60, retry_on_timeout=False)
C = redis.Redis(port=18431, socket_timeout=60, retry_on_timeout=False)


def run(r, *a):
    try:
        return r.execute_command(*a)
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


def show(title, *a, setup=()):
    for s in setup:
        run(R, *s); run(C, *s)
    x, y = run(R, *a), run(C, *a)
    strip = lambda v: re.sub(r"time: [0-9.]+", "time: T", repr(v))
    print(("SAME " if strip(x) == strip(y) else "DIFF ") + title)
    print("   Rust:", strip(x)[:300]); print("   C   :", strip(y)[:300])


def hdr(label, props):
    return label.encode() + b"\0" + struct.pack("<I", len(props)) + b"".join(p.encode() + b"\0" for p in props)


LONG = lambda v: b"\x04" + struct.pack("<q", v)

for r in (R, C):
    r.flushall()
    r.execute_command("GRAPH.QUERY", "g", "RETURN 1")

show("A1 non-UTF-8 flag ends Rust's flag loop, --compact dropped", "GRAPH.QUERY", "g", "RETURN 1", b"\xff", "--compact")
show("A2 negative TIMEOUT", "GRAPH.QUERY", "g", "RETURN 1", "TIMEOUT", "-5")
show("A3 +10 TIMEOUT", "GRAPH.QUERY", "g", "RETURN 1", "TIMEOUT", "+10")
show("A4 RO_QUERY garbage TIMEOUT", "GRAPH.RO_QUERY", "g", "RETURN 1", "TIMEOUT", "abc")
show("A5 version > UINT_MAX", "GRAPH.QUERY", "g", "RETURN 1", "version", "4294967296")
show("A6 >8 argv", "GRAPH.QUERY", "g", "RETURN 1", "a", "b", "c", "d", "e", "f")
show("C1 VKEY_MAX_ENTITY_COUNT -5", "GRAPH.CONFIG", "SET", "VKEY_MAX_ENTITY_COUNT", "-5")
show("C2 CMD_INFO 1", "GRAPH.CONFIG", "SET", "CMD_INFO", "1")
show("C3 JS_HEAP_SIZE 5", "GRAPH.CONFIG", "SET", "JS_HEAP_SIZE", "5")
show("C4 dotless-i config name", "GRAPH.CONFIG", "GET", "tımeout")
show("C5 GET arity", "GRAPH.CONFIG", "GET", "TIMEOUT", "extra")
show("C6 ASYNC_DELETE default", "GRAPH.CONFIG", "GET", "ASYNC_DELETE")
show("C7 DEFAULT>MAX in one batch (C breaks invariant)", "GRAPH.CONFIG", "SET", "TIMEOUT_DEFAULT", "10", "TIMEOUT_MAX", "5")
show("C7b", "GRAPH.CONFIG", "GET", "TIMEOUT_DEFAULT")
for r in (R, C):
    r.execute_command("GRAPH.CONFIG", "SET", "TIMEOUT_MAX", "0"); r.execute_command("GRAPH.CONFIG", "SET", "TIMEOUT_DEFAULT", "0")
    r.execute_command("GRAPH.CONFIG", "SET", "VKEY_MAX_ENTITY_COUNT", "100000")
    r.execute_command("GRAPH.CONFIG", "SET", "JS_HEAP_SIZE", "268435456")
show("B1 bulk header with property p twice", "GRAPH.BULK", "d", "BEGIN", "1", "0", "1", "0",
     hdr("L", ["p", "q", "p"]) + LONG(1) + LONG(2) + LONG(3))
show("B1b", "GRAPH.QUERY", "d", "MATCH (n) RETURN n.p, properties(n)")
show("B1c", "GRAPH.QUERY", "d", "MATCH (n {p:3}) RETURN count(n)")
show("B2 bulk label L:L", "GRAPH.BULK", "e", "BEGIN", "1", "0", "1", "0", hdr("L:L", ["p"]) + LONG(1))
show("B2b", "GRAPH.QUERY", "e", "MATCH (n:L) RETURN count(n)")
show("V1 verbose top-level point", "GRAPH.QUERY", "g", "RETURN point({latitude:1, longitude:2})")
show("V2 verbose NaN inside a list", "GRAPH.QUERY", "g", "RETURN [0.0/0.0]")
show("V3 deleted node labels", "GRAPH.QUERY", "g", "CREATE (a:A) DELETE a RETURN a")
show("M1 GRAPH.MEMORY SAMPLES 0", "GRAPH.MEMORY", "USAGE", "g", "SAMPLES", "0")
show("K1 CONSTRAINT count +1", "GRAPH.CONSTRAINT", "CREATE", "g", "MANDATORY", "NODE", "L", "PROPERTIES", "+1", "q")
show("K2 CONSTRAINT entity LABEL", "GRAPH.CONSTRAINT", "CREATE", "g", "MANDATORY", "LABEL", "L", "PROPERTIES", "1", "z")
show("T1 write ignores TIMEOUT (TIMEOUT_MAX set)", "GRAPH.QUERY", "t", "UNWIND range(1,300000) AS x CREATE (:T)", "TIMEOUT", "1",
     setup=[("GRAPH.CONFIG", "SET", "TIMEOUT_MAX", "100000")])
show("T1b", "GRAPH.QUERY", "t", "MATCH (n) RETURN count(n)")
