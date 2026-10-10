"""Live repro of the index-layer bugs: each case runs the query on a graph
without an index (the full-scan reference) and with a range index, on the
Rust module (port 18300) and the C module (port 18301).

  redis-server --port 18300 --dir . --loadmodule target/release/libfalkordb.dylib
  redis-server --port 18301 --dir . --loadmodule bin/macos-arm64v8-release/falkordb.so
  venv/bin/python proofs/index_layer/repro.py
"""
import redis

PORTS = {"rust": 18300, "c": 18301}

CASES = [
    # (name, index stmts, setup, query)
    ("bool_int_conflated", ["CREATE INDEX FOR (n:L) ON (n.v)"],
     "CREATE (:L {v:1, k:'int'}), (:L {v:true, k:'bool'})", "MATCH (n:L) WHERE n.v = 1 RETURN n.k"),
    ("temporal_in_numeric_range", ["CREATE INDEX FOR (n:L) ON (n.v)"],
     "CREATE (:L {v:5, k:'int'}), (:L {v:date('2020-01-01'), k:'date'})", "MATCH (n:L) WHERE n.v > 0 RETURN n.k"),
    ("folded_temporal_constant_drops_filter", ["CREATE INDEX FOR (n:L) ON (n.v)"],
     "CREATE (:L {v:5, k:'int'}), (:L {v:date('2020-01-01'), k:'date'})",
     "MATCH (n:L) WHERE n.v = date('2020-01-01') RETURN n.k"),
    ("inline_temporal_constant", ["CREATE INDEX FOR (n:L) ON (n.v)"],
     "CREATE (:L {v:5, k:'int'}), (:L {v:date('2020-01-01'), k:'date'})",
     "MATCH (n:L {v: date('2020-01-01')}) RETURN n.k"),
    ("point_equality_empty", ["CREATE INDEX FOR (n:L) ON (n.v)"],
     "CREATE (:L {v:point({latitude:1.0, longitude:2.0}), k:'p'})",
     "MATCH (n:L) WHERE n.v = point({latitude:1.0, longitude:2.0}) RETURN n.k"),
    ("in_list_drops_temporal_item", ["CREATE INDEX FOR (n:L) ON (n.v)"],
     "CREATE (:L {v:2, k:'int'}), (:L {v:date('2020-01-01'), k:'date'})",
     "MATCH (n:L) WHERE n.v IN [date('2020-01-01'), 2] RETURN n.k"),
    ("empty_string_not_indexed", ["CREATE INDEX FOR (n:L) ON (n.v)"],
     "CREATE (:L {v:'', k:'empty'})", "MATCH (n:L) WHERE n.v = '' RETURN n.k"),
    ("string_range_encoding_order", ["CREATE INDEX FOR (n:L) ON (n.v)"],
     "CREATE (:L {v:'John Smith', k:'js'}), (:L {v:'Johnny', k:'jy'})", "MATCH (n:L) WHERE n.v < 'JohnA' RETURN n.k"),
    ("string_range_encoding_order_gt", ["CREATE INDEX FOR (n:L) ON (n.v)"],
     "CREATE (:L {v:'a b', k:'ab'})", "MATCH (n:L) WHERE n.v > 'a!' RETURN n.k"),
    ("string_exclusive_equal_bounds", ["CREATE INDEX FOR (n:L) ON (n.v)"],
     "CREATE (:L {v:'a', k:'a'})", "MATCH (n:L) WHERE n.v > 'a' AND n.v < 'a' RETURN n.k"),
    ("open_bound_excludes_inf", ["CREATE INDEX FOR (n:L) ON (n.v)"],
     "CREATE (:L {v:1.0/0.0, k:'inf'}), (:L {v:-1.0/0.0, k:'-inf'})",
     "MATCH (n:L) WHERE n.v > 0 OR n.v < 0 RETURN n.k"),
    ("multilabel_and", ["CREATE INDEX FOR (n:A) ON (n.x)", "CREATE INDEX FOR (n:B) ON (n.y)"],
     "CREATE (:A:B {x:1, y:2, k:'ab'})", "MATCH (n:A:B) WHERE n.x = 1 AND n.y = 2 RETURN n.k"),
    ("multilabel_or", ["CREATE INDEX FOR (n:A) ON (n.x)", "CREATE INDEX FOR (n:B) ON (n.y)"],
     "CREATE (:A:B {x:1, y:2, k:'ab'}), (:A:B {x:1, y:3, k:'ab2'})", "MATCH (n:A:B) WHERE n.y = 2 OR n.x = 1 RETURN n.k"),
    ("multilabel_inline", ["CREATE INDEX FOR (n:A) ON (n.x)", "CREATE INDEX FOR (n:B) ON (n.y)"],
     "CREATE (:A:B {x:1, y:2, k:'ab'})", "MATCH (n:A:B {x:1, y:2}) RETURN n.k"),
    ("geo_radius_near_pole", ["CREATE INDEX FOR (n:P) ON (n.p)"],
     "UNWIND range(0, 20) AS i CREATE (:P {k: toString(i), p: point({latitude: 89.99, longitude: -179.9 + i*0.001})})",
     "MATCH (n:P) WHERE distance(n.p, point({latitude: 89.99, longitude: -179.9})) < 100 RETURN count(n)"),
    ("edge_undirected_index_scan", ["CREATE INDEX FOR ()-[r:R]-() ON (r.v)"],
     "CREATE ({k:1})-[:R {v:2, k:'e'}]->({k:2})", "MATCH ()-[r:R]-() WHERE r.v >= 1 RETURN r.k"),
]


def run(r, g, q):
    try:
        res = r.execute_command("GRAPH.QUERY", g, q)
        return sorted(x[0].decode() if isinstance(x[0], bytes) else repr(x[0]) for x in res[1])
    except Exception as e:
        return "ERR " + str(e)[:60]


for name, idx, setup, q in CASES:
    line = {}
    for side, port in PORTS.items():
        r = redis.Redis(port=port, socket_timeout=30)
        for mode in ("scan", "index"):
            g = "repro_%s_%s" % (name, mode)
            try:
                r.execute_command("GRAPH.DELETE", g)
            except Exception:
                pass
            if mode == "index":
                for s in idx:
                    r.execute_command("GRAPH.QUERY", g, s)
            r.execute_command("GRAPH.QUERY", g, setup)
            line[(side, mode)] = run(r, g, q)
    rust_bad = line[("rust", "scan")] != line[("rust", "index")]
    c_bad = line[("c", "scan")] != line[("c", "index")]
    print("%-40s rust scan=%s index=%s | c scan=%s index=%s | %s" % (
        name, line[("rust", "scan")], line[("rust", "index")], line[("c", "scan")], line[("c", "index")],
        ("RUST-WRONG" if rust_bad else "rust-ok") + ("/C-WRONG" if c_bad else "/c-ok")))
