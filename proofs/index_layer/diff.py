"""Differential test: index scan vs full scan, Rust vs C.

For each (setup, query): run on a fresh graph without index, then with a
range index on :L(v) (and :L(w), :B(w) for multi-label cases).  Report any
case where Rust(index) != Rust(noindex) or C(index) != C(noindex), plus
Rust(noindex) vs C(noindex) as context.
"""
import redis, sys, json

RUST, C = 18300, 18301
conns = {p: redis.Redis(port=p, socket_timeout=60) for p in (RUST, C)}

VALUES = [
    "1", "1.0", "-0.0", "0", "2", "true", "false", "9007199254740993",
    "0.0/0.0", "1.0/0.0", "-1.0/0.0",
    "''", "' '", "'a'", "'A'", "'a b'", "'a!'", "'aB'", "'a1'", "'a_'", "'a\\\\'", "'a]'",
    "'_'", "'\\t'", "'é'", "'😀'", "'z'", "'John Smith'", "'JohnA'",
    "[1,2]", "['a']", "[true]", "[1.0]", "[[1]]",
    "date('2020-01-01')", "localdatetime('2020-01-01T00:00:00')", "duration('P1D')",
    "point({latitude:1.0, longitude:2.0})",
]


def setup(r, g):
    try:
        r.execute_command("GRAPH.DELETE", g)
    except Exception:
        pass
    q = "UNWIND range(0, %d) AS i CREATE (:L {i:i})" % (len(VALUES) - 1)
    r.execute_command("GRAPH.QUERY", g, q)
    for i, v in enumerate(VALUES):
        r.execute_command("GRAPH.QUERY", g, "MATCH (n:L {i:%d}) SET n.v = %s" % (i, v))


def run(r, g, q):
    try:
        res = r.execute_command("GRAPH.QUERY", g, q)
        rows = res[1] if len(res) >= 3 else []
        return sorted(repr(x) for x in rows)
    except Exception as e:
        return ["ERR " + str(e)[:80]]


def explain(r, g, q):
    try:
        return " | ".join(x.decode() if isinstance(x, bytes) else str(x)
                          for x in r.execute_command("GRAPH.EXPLAIN", g, q))
    except Exception as e:
        return "ERR " + str(e)[:80]


def main(queries, idx_stmts=("CREATE INDEX FOR (n:L) ON (n.v)",), setup_fn=setup):
    bad = 0
    res = {}
    for p, r in conns.items():
        g = "diff%d" % p
        setup_fn(r, g)
        res[p] = {"noidx": [run(r, g, q) for q in queries]}
        for s in idx_stmts:
            r.execute_command("GRAPH.QUERY", g, s)
        # wait for index population
        for _ in range(100):
            out = r.execute_command("GRAPH.QUERY", g, "CALL db.indexes() YIELD status RETURN collect(status)")
            if b"UNDER CONSTRUCTION" not in repr(out).encode() and "UNDER" not in repr(out):
                break
        res[p]["idx"] = [run(r, g, q) for q in queries]
        res[p]["plan"] = [explain(r, g, q) for q in queries]
    for k, q in enumerate(queries):
        rn, ri = res[RUST]["noidx"][k], res[RUST]["idx"][k]
        cn, ci = res[C]["noidx"][k], res[C]["idx"][k]
        tag = []
        if rn != ri:
            tag.append("RUST-IDX-DIFF")
        if cn != ci:
            tag.append("C-IDX-DIFF")
        if ri != ci:
            tag.append("RUST!=C(idx)")
        if tag:
            bad += 1
            print("==", " ".join(tag), "::", q)
            print("   rust noidx:", rn)
            print("   rust idx  :", ri)
            print("   c    noidx:", cn)
            print("   c    idx  :", ci)
            print("   rust plan :", res[RUST]["plan"][k])
    print("cases", len(queries), "flagged", bad)


if __name__ == "__main__":
    qs = [l.rstrip("\n") for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
    main(qs)
