"""Live reproductions for proofs/optimizer_scan.

Needs two redis-servers: Rust module on 18200, C module
(bin/macos-arm64v8-release/falkordb.so) on 18201. Each query runs on a graph
with the index (gi) and without (gn) on Rust, and with the index on C.
Run: venv/bin/python proofs/optimizer_scan/repro.py
"""
import time
import redis

rs = redis.Redis(port=18200, socket_timeout=30)
c = redis.Redis(port=18201, socket_timeout=30)


def run(r, g, q):
    try:
        res = r.execute_command("GRAPH.QUERY", g, q)
        return res[1] if len(res) == 3 else res[:-1]
    except Exception as e:  # noqa: BLE001
        return "ERR: " + str(e)


def setup(g, qs):
    for r in (rs, c):
        try:
            r.execute_command("GRAPH.DELETE", g)
        except Exception:  # noqa: BLE001
            pass
        for q in qs:
            r.execute_command("GRAPH.QUERY", g, q)


def compare(qs):
    for q in qs:
        a, b, cc = run(rs, "gi", q), run(rs, "gn", q), run(c, "gi", q)
        tag = "ok   " if a == b == cc else ("BUG  " if a != b and b == cc else "DIFF ")
        print(f"[{tag}] {q}\n   rust+index: {a}\n   rust      : {b}\n   C+index   : {cc}")


DATA = ["CREATE (:L {k:1, a:1, v:1}), (:L {k:2, a:-1, v:2}), (:L {k:3, a:2, b:7}),"
        " (:L {k:4, arr:[true, 'X']}), (:L {k:5, arr:[1, 'x']}), (:L {k:6, v:'B'})"]
setup("gi", DATA + ["CREATE INDEX FOR (n:L) ON (n.a)", "CREATE INDEX FOR (n:L) ON (n.v)",
                    "CREATE INDEX FOR (n:L) ON (n.arr)"])
setup("gn", DATA)
time.sleep(1)
compare([
    # C4: computed constant >= 2^52 -> filter dropped, fallback label scan
    "MATCH (n:L) WHERE n.v = 4503599627370495 + 4503599627370495 + 3 RETURN n.k ORDER BY n.k",
    "MATCH (n:L) WHERE n.v > 4503599627370495 * 2 RETURN n.k ORDER BY n.k",
    # C7: IN with a computed left side pushed as `n.a IN [...]`
    "MATCH (n:L) WHERE n.a + 1 IN [2] RETURN n.k ORDER BY n.k",
    "MATCH (n:L) WHERE abs(n.a) IN [1] RETURN n.k ORDER BY n.k",
    "MATCH (n:L) WHERE toString(n.a) IN ['1'] RETURN n.k ORDER BY n.k",
    # C8: `x IN [n.a, n.b]` pushed as array-contains on n.a
    "MATCH (n:L) WHERE 2 IN [n.a, n.b] RETURN n.k ORDER BY n.k",
    "MATCH (n:L) WHERE -1 IN [n.a] RETURN n.k ORDER BY n.k",
    # C10: array-contains inside AND loses its post-filter
    "MATCH (n:L) WHERE 1 IN n.arr RETURN n.k ORDER BY n.k",
    "MATCH (n:L) WHERE 1 IN n.arr AND n.k > 0 RETURN n.k ORDER BY n.k",
    # C11 (also proofs/index_layer bug 6)
    "MATCH (n:L) WHERE n.v > 'B' AND n.v < 'B' RETURN n.k ORDER BY n.k",
])

# Phantom node 0 on a graph that never held a node (no index involved)
for r in (rs, c):
    try:
        r.execute_command("GRAPH.DELETE", "gempty")
    except Exception:  # noqa: BLE001
        pass
    r.execute_command("GRAPH.QUERY", "gempty", "RETURN 1")
for q in ["MATCH (n) WHERE id(n) = 0 RETURN id(n), labels(n)",
          "MATCH (n) WHERE id(n) = 0 SET n.x = 1, n:Z RETURN id(n)",
          "CREATE (m:Q) RETURN id(m)",
          "MATCH (n) RETURN id(n), labels(n), properties(n)"]:
    print(f"[phantom] {q}\n   rust: {run(rs, 'gempty', q)}\n   C   : {run(c, 'gempty', q)}")
