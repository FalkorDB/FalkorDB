"""Batch 2: multi-label indexes and index maintenance, Rust vs C.

Each scenario: list of setup statements (run after index creation, so the
index must be *maintained*), list of check queries.  Baseline = same on a
graph without any index.
"""
import redis, sys

RUST, C = 18300, 18301
conns = {p: redis.Redis(port=p, socket_timeout=60) for p in (RUST, C)}


def q(r, g, s):
    try:
        res = r.execute_command("GRAPH.QUERY", g, s)
        return sorted(repr(x) for x in (res[1] if len(res) >= 3 else []))
    except Exception as e:
        return ["ERR " + str(e)[:100]]


def scenario(name, indexes, stmts, checks):
    out = {}
    for p, r in conns.items():
        for mode in ("noidx", "idx"):
            g = "m_%s_%d_%s" % (name, p, mode)
            try:
                r.execute_command("GRAPH.DELETE", g)
            except Exception:
                pass
            r.execute_command("GRAPH.QUERY", g, "RETURN 1")
            if mode == "idx":
                for s in indexes:
                    q(r, g, s)
            for s in stmts:
                res = q(r, g, s)
                if res and res[0].startswith("ERR"):
                    print("   setup err", p, mode, s, res)
            out[(p, mode)] = [q(r, g, c) for c in checks]
    bad = False
    for k, c in enumerate(checks):
        rn, ri = out[(RUST, "noidx")][k], out[(RUST, "idx")][k]
        cn, ci = out[(C, "noidx")][k], out[(C, "idx")][k]
        tags = []
        if rn != ri:
            tags.append("RUST-IDX-DIFF")
        if cn != ci:
            tags.append("C-IDX-DIFF")
        if ri != ci:
            tags.append("RUST!=C")
        if tags:
            bad = True
            print("==", name, " ".join(tags), "::", c)
            print("   rust noidx", rn, "idx", ri)
            print("   c    noidx", cn, "idx", ci)
    print("--", name, "BAD" if bad else "ok")


IDX_LV = ["CREATE INDEX FOR (n:L) ON (n.v)"]

scenario("multilabel_and",
         ["CREATE INDEX FOR (n:A) ON (n.x)", "CREATE INDEX FOR (n:B) ON (n.y)"],
         ["CREATE (:A:B {x:1, y:2, k:'ab'}), (:A:B {x:1, y:3, k:'ab2'}), (:A {x:1, y:2, k:'a'})"],
         ["MATCH (n:A:B) WHERE n.x = 1 AND n.y = 2 RETURN n.k",
          "MATCH (n:A:B) WHERE n.x = 1 OR n.y = 2 RETURN n.k",
          "MATCH (n:A:B) WHERE n.y = 2 OR n.x = 1 RETURN n.k",
          "MATCH (n:B:A) WHERE n.y = 2 AND n.x = 1 RETURN n.k",
          "MATCH (n:A:B) WHERE n.y = 3 OR n.x = 5 RETURN n.k",
          "MATCH (n:A:B {x:1, y:2}) RETURN n.k",
          "MATCH (n:A) WHERE n.x = 1 AND n.y = 2 RETURN n.k"])

scenario("update_value", IDX_LV,
         ["CREATE (:L {v:1, k:1}), (:L {v:2, k:2})",
          "MATCH (n:L {k:1}) SET n.v = 5",
          "MATCH (n:L {k:2}) SET n.v = 'x'"],
         ["MATCH (n:L) WHERE n.v = 1 RETURN n.k", "MATCH (n:L) WHERE n.v = 5 RETURN n.k",
          "MATCH (n:L) WHERE n.v = 2 RETURN n.k", "MATCH (n:L) WHERE n.v = 'x' RETURN n.k",
          "MATCH (n:L) WHERE n.v > 0 RETURN n.k"])

scenario("remove_prop", IDX_LV,
         ["CREATE (:L {v:1, k:1}), (:L {v:1, k:2, w:3})",
          "MATCH (n:L {k:1}) REMOVE n.v",
          "MATCH (n:L {k:2}) SET n.v = null"],
         ["MATCH (n:L) WHERE n.v = 1 RETURN n.k", "MATCH (n:L) WHERE n.v >= 0 RETURN n.k"])

scenario("set_map", IDX_LV,
         ["CREATE (:L {v:1, k:1}), (:L {v:1, k:2})",
          "MATCH (n:L {k:1}) SET n = {k:1}",
          "MATCH (n:L {k:2}) SET n += {v:7}"],
         ["MATCH (n:L) WHERE n.v = 1 RETURN n.k", "MATCH (n:L) WHERE n.v = 7 RETURN n.k"])

scenario("type_change_list", IDX_LV,
         ["CREATE (:L {v:1, k:1}), (:L {v:'a', k:2})",
          "MATCH (n:L {k:1}) SET n.v = [1]",
          "MATCH (n:L {k:2}) SET n.v = ['a']"],
         ["MATCH (n:L) WHERE n.v = 1 RETURN n.k", "MATCH (n:L) WHERE n.v = 'a' RETURN n.k",
          "MATCH (n:L) WHERE 1 IN n.v RETURN n.k", "MATCH (n:L) WHERE 'a' IN n.v RETURN n.k",
          "MATCH (n:L) WHERE n.v > 0 RETURN n.k"])

scenario("list_to_scalar", IDX_LV,
         ["CREATE (:L {v:[1], k:1}), (:L {v:['a'], k:2})",
          "MATCH (n:L {k:1}) SET n.v = 2",
          "MATCH (n:L {k:2}) SET n.v = 'b'"],
         ["MATCH (n:L) WHERE 1 IN n.v RETURN n.k", "MATCH (n:L) WHERE 'a' IN n.v RETURN n.k",
          "MATCH (n:L) WHERE n.v = 2 RETURN n.k", "MATCH (n:L) WHERE n.v = 'b' RETURN n.k"])

scenario("delete_node", IDX_LV,
         ["CREATE (:L {v:1, k:1}), (:L {v:1, k:2})",
          "MATCH (n:L {k:1}) DELETE n",
          "CREATE (:L {v:1, k:3})"],
         ["MATCH (n:L) WHERE n.v = 1 RETURN n.k"])

scenario("label_remove", IDX_LV,
         ["CREATE (:L {v:1, k:1}), (:L {v:1, k:2})",
          "MATCH (n:L {k:1}) REMOVE n:L"],
         ["MATCH (n:L) WHERE n.v = 1 RETURN n.k"])

scenario("label_remove_readd_same_query", IDX_LV,
         ["CREATE (:L {v:1, k:1}), (:L {v:1, k:2})",
          "MATCH (n:L {k:1}) REMOVE n:L SET n:L"],
         ["MATCH (n:L) WHERE n.v = 1 RETURN n.k", "MATCH (n:L) RETURN n.k"])

scenario("label_remove_readd_with", IDX_LV,
         ["CREATE (:L {v:1, k:1}), (:L {v:1, k:2})",
          "MATCH (n:L {k:1}) REMOVE n:L WITH n SET n:L"],
         ["MATCH (n:L) WHERE n.v = 1 RETURN n.k", "MATCH (n:L) RETURN n.k"])

scenario("label_add", IDX_LV,
         ["CREATE (:M {v:1, k:1}), (:L {v:1, k:2})",
          "MATCH (n:M {k:1}) SET n:L"],
         ["MATCH (n:L) WHERE n.v = 1 RETURN n.k"])

scenario("label_add_then_set_prop", IDX_LV,
         ["CREATE (:M {k:1}), (:L {v:1, k:2})",
          "MATCH (n:M {k:1}) SET n:L SET n.v = 1"],
         ["MATCH (n:L) WHERE n.v = 1 RETURN n.k"])

scenario("set_prop_then_label_remove", IDX_LV,
         ["CREATE (:L {v:1, k:1}), (:L {v:1, k:2})",
          "MATCH (n:L {k:1}) SET n.v = 2 REMOVE n:L"],
         ["MATCH (n:L) WHERE n.v = 2 RETURN n.k", "MATCH (n:L) WHERE n.v = 1 RETURN n.k"])

scenario("create_delete_same_query", IDX_LV,
         ["CREATE (:L {v:1, k:2})",
          "CREATE (n:L {v:1, k:1}) DELETE n"],
         ["MATCH (n:L) WHERE n.v = 1 RETURN n.k"])

scenario("delete_then_create_reuse", IDX_LV,
         ["CREATE (:L {v:1, k:1}), (:L {v:2, k:2})",
          "MATCH (n:L {k:1}) DELETE n",
          "CREATE (:L {v:3, k:3})"],
         ["MATCH (n:L) WHERE n.v = 1 RETURN n.k", "MATCH (n:L) WHERE n.v = 3 RETURN n.k",
          "MATCH (n:L) WHERE n.v > 0 RETURN n.k"])

scenario("delete_other_label", ["CREATE INDEX FOR (n:L) ON (n.v)", "CREATE INDEX FOR (n:M) ON (n.v)"],
         ["CREATE (:L:M {v:1, k:1})",
          "MATCH (n:L {k:1}) REMOVE n:M"],
         ["MATCH (n:L) WHERE n.v = 1 RETURN n.k", "MATCH (n:M) WHERE n.v = 1 RETURN n.k"])

scenario("merge_on_match", IDX_LV,
         ["CREATE (:L {v:1, k:1})",
          "MERGE (n:L {k:1}) ON MATCH SET n.v = 9",
          "MERGE (n:L {k:2}) ON CREATE SET n.v = 9"],
         ["MATCH (n:L) WHERE n.v = 9 RETURN n.k", "MATCH (n:L) WHERE n.v = 1 RETURN n.k"])

scenario("rollback_on_error", IDX_LV,
         ["CREATE (:L {v:1, k:1})",
          "MATCH (n:L {k:1}) SET n.v = 2 WITH n RETURN 1/0",
          "MATCH (n:L {k:1}) SET n.v = 3 WITH n RETURN toInteger({})"],
         ["MATCH (n:L) WHERE n.v = 1 RETURN n.k", "MATCH (n:L) WHERE n.v = 2 RETURN n.k",
          "MATCH (n:L) WHERE n.v = 3 RETURN n.k", "MATCH (n:L) RETURN n.v"])

IDX_E = ["CREATE INDEX FOR ()-[r:R]-() ON (r.v)"]
scenario("edge_updates", IDX_E,
         ["CREATE (a {k:1})-[:R {v:1, k:1}]->(b {k:2}), (a)-[:R {v:1, k:2}]->(b), (b)-[:R {v:1, k:3}]->(a)",
          "MATCH ()-[r:R {k:1}]->() SET r.v = 2",
          "MATCH ()-[r:R {k:2}]->() DELETE r",
          "MATCH ({k:2})-[r:R {k:3}]->() REMOVE r.v"],
         ["MATCH ()-[r:R]->() WHERE r.v = 1 RETURN r.k", "MATCH ()-[r:R]->() WHERE r.v = 2 RETURN r.k",
          "MATCH ()-[r:R]-() WHERE r.v >= 1 RETURN r.k"])

scenario("edge_delete_node", IDX_E,
         ["CREATE (a {k:1})-[:R {v:1, k:1}]->(b {k:2})",
          "MATCH (a {k:1}) DETACH DELETE a",
          "CREATE (c {k:3})-[:R {v:1, k:4}]->(d {k:4})"],
         ["MATCH ()-[r:R]->() WHERE r.v = 1 RETURN r.k"])

scenario("edge_types", IDX_E,
         ["CREATE ()-[:R {v:date('2020-01-01'), k:1}]->(), ()-[:R {v:1, k:2}]->(), ()-[:R {v:true, k:3}]->()"],
         ["MATCH ()-[r:R]->() WHERE r.v = date('2020-01-01') RETURN r.k",
          "MATCH ()-[r:R]->() WHERE r.v = 1 RETURN r.k",
          "MATCH ()-[r:R]->() WHERE r.v > 0 RETURN r.k"])

scenario("unwind_param_like", IDX_LV,
         ["CREATE (:L {v:1, k:1}), (:L {v:date('2020-01-01'), k:2}), (:L {v:[1], k:3})"],
         ["UNWIND [1, date('2020-01-01'), [1]] AS x MATCH (n:L) WHERE n.v = x RETURN n.k",
          "WITH date('2020-01-01') AS d MATCH (n:L) WHERE n.v = d RETURN n.k",
          "WITH [1] AS d MATCH (n:L) WHERE n.v = d RETURN n.k",
          "MATCH (n:L) WHERE n.v = [1] RETURN n.k",
          "MATCH (n:L {v: date('2020-01-01')}) RETURN n.k",
          "MATCH (n:L {v: [1]}) RETURN n.k"])
