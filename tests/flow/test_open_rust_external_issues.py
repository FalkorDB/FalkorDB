"""Reproductions of open issues reported by people outside the FalkorDB team
that affect the Rust engine.

One class per issue (testIssue<N>). Each docstring links the issue, names the
trigger and the behaviour the asserts expect, so a test fails while its issue is
open and passes once it is fixed. Several classes take the server down (crash,
OOM), so run each class on its own; the DoS classes (#1482, #1483, #1485, #1486)
end with a method that runs the reporter's query on the default configuration
and should be run in a memory-capped container (e.g. `docker run --memory 2g`).

Five classes already pass on the Rust engine (#627, #2098, #2099, #2179, #2733):
those issues look fixed and the classes are kept as regression tests.

Like test_open_rust_s1_issues.py, this file is deliberately NOT listed in
flow_tests_done.txt, so CI does not run it. When an issue is fixed, move its
class into the suite that covers the feature (and into CI), and delete it here.
"""

from common import Env
from index_utils import create_node_range_index, wait_for_indices_to_sync

import redis
from redis import ResponseError

# A valid Rust-engine GRAPH.RESTORE payload (see test_open_rust_s1_issues.py).
from test_open_rust_s1_issues import _good_payload


def _exta_expect_error(env, graph, query, *fragments):
    """Run `query`, expect a ResponseError whose message contains every
    fragment, then check the server is still up."""
    try:
        graph.query(query)
        env.assertTrue(False, message=f"expected an error for: {query[:80]}")
    except ResponseError as e:
        for f in fragments:
            env.assertContains(f, str(e))
    env.assertTrue(env.getConnection().ping())


class testIssue551():
    """https://github.com/FalkorDB/FalkorDB/issues/551
    GRAPH.CONSTRAINT rejects labels that are not plain identifiers (e.g.
    `prefix:suffix`, and on Rust also `Lab$el`) although the same labels can be
    created and indexed through Cypher. Creating the constraint must succeed."""

    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph("issue_551")

    def _create_unique(self, label):
        self.graph.query(f"CREATE (:`{label}` {{id: 1}})")
        self.graph.query(f"CREATE INDEX FOR (n:`{label}`) ON (n.id)")
        res = self.conn.execute_command("GRAPH.CONSTRAINT", "CREATE", "issue_551",
                                        "UNIQUE", "NODE", label, "PROPERTIES", 1, "id")
        self.env.assertEqual(res, "PENDING")

    def test01_constraint_on_label_with_colon(self):
        self._create_unique("prefix:suffix")

    def test02_constraint_on_label_with_dollar(self):
        self._create_unique("Lab$el")


class testIssue602():
    """https://github.com/FalkorDB/FalkorDB/issues/602
    A CALL {} body whose leading WITH introduces a fresh binding (imports
    nothing) and has ORDER BY is rejected as an invalid importing WITH. The
    query is valid and must run."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_602")

    def test01_leading_with_order_by_in_call(self):
        res = self.graph.query("CALL { WITH 0 AS n1 ORDER BY 0 RETURN 0 AS n0 } RETURN 0")
        self.env.assertEqual(res.result_set, [[0]])
        res = self.graph.query("CALL { WITH 1 AS n1 ORDER BY n1 RETURN n1 AS n0 } RETURN n0")
        self.env.assertEqual(res.result_set, [[1]])


class testIssue627():
    """https://github.com/FalkorDB/FalkorDB/issues/627
    A node returned after DELETE picks up the properties of an unrelated node
    created (and deleted) later in the same query, because it is re-read from
    its recycled slot. `x` must come back without properties."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_627")

    def test01_deleted_node_does_not_leak_properties(self):
        res = self.graph.query("CREATE (x) DELETE x CREATE (y {n0: 0}) DELETE y RETURN properties(x)")
        self.env.assertEqual(res.result_set, [[{}]])
        res = self.graph.query("CREATE (x) DELETE x CREATE (y {n0: 0}) DELETE y RETURN x")
        self.env.assertEqual(res.result_set[0][0].properties, {})


class testIssue647():
    """https://github.com/FalkorDB/FalkorDB/issues/647
    With a range index on a property, `WHERE x.p = ''` served from the index
    returns no rows, although the entity holds the empty string (same family as
    #282/#294). The empty-string row must be found, for edges and nodes."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_647")

    def test01_edge_index_empty_string(self):
        self.graph.query("CREATE (:per {id:'x1'})-[:fl {addr:''}]->(:per {id:'x2'})")
        self.graph.query("CREATE (:per {id:'x3'})-[:fl {addr:'hk'}]->(:per {id:'x4'})")
        q = "MATCH (src)-[r:fl]->(dst) WHERE r.addr = '' RETURN src.id, r.addr, dst.id"
        self.env.assertEqual(self.graph.query(q).result_set, [['x1', '', 'x2']])
        self.graph.query("CREATE INDEX FOR ()-[r:fl]->() ON (r.addr)")
        self.env.assertEqual(self.graph.query(q).result_set, [['x1', '', 'x2']])

    def test02_node_index_empty_string(self):
        self.graph.query("CREATE (:P {id:'a', s:''}), (:P {id:'b', s:'hk'})")
        self.graph.query("CREATE INDEX FOR (n:P) ON (n.s)")
        res = self.graph.query("MATCH (n:P) WHERE n.s = '' RETURN n.id")
        self.env.assertEqual(res.result_set, [['a']])


class testIssue694():
    """https://github.com/FalkorDB/FalkorDB/issues/694
    On a single self-loop, `MATCH (y)<--(y)<--(y)` matches by binding the same
    relationship twice in one MATCH. openCypher relationship uniqueness forbids
    that, so the pattern must produce no rows."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_694")

    def test01_self_loop_not_traversed_twice(self):
        res = self.graph.query("CREATE (x)-[:n0]->(x) WITH * MATCH (y)<--(y)<--(y) RETURN 0")
        self.env.assertEqual(res.result_set, [])
        res = self.graph.query("MATCH (y)<--(y)<--(y) RETURN count(*)")
        self.env.assertEqual(res.result_set, [[0]])
        res = self.graph.query("MATCH (y)-[a]->(y)-[b]->(y) RETURN count(*)")
        self.env.assertEqual(res.result_set, [[0]])


class testIssue748():
    """https://github.com/FalkorDB/FalkorDB/issues/748
    A MERGE inside CALL {} is skipped when the outer query continues with a
    constant-false filter (`MATCH () WHERE false`), but runs when `WITH *` is
    inserted. The write must always happen: one node in both cases."""

    def __init__(self):
        self.env, self.db = Env()

    def test01_merge_in_call_runs_before_false_filter(self):
        for i, q in enumerate([
                "WITH 0 AS n0 CALL { MERGE () RETURN 0 AS n1 } MATCH () WHERE false RETURN 0",
                "WITH 0 AS n0 CALL { MERGE () RETURN 0 AS n1 } WITH * MATCH () WHERE false RETURN 0"]):
            g = self.db.select_graph(f"issue_748_{i}")
            res = g.query(q)
            self.env.assertEqual(res.result_set, [])
            res = g.query("MATCH (n) RETURN count(n)")
            self.env.assertEqual(res.result_set, [[1]], message=q)


class testIssue752():
    """https://github.com/FalkorDB/FalkorDB/issues/752
    Adding an irrelevant `UNWIND [1] AS a0` between two MATCH clauses changes
    the result; on Rust the UNWIND-free query returns no rows because
    relationship uniqueness is enforced across MATCH clauses (#2534). Both
    variants must return the single row (4, 10, 7)."""

    SETUP = [
        "CREATE (n0 {id : 0})",
        "CREATE (n0 :L2{k17 : false, k16 : '4NUdD', id : 1, k13 : -1476973449, k12 : false, k15 : 'CNX922GS', k14 : '6ksmzAx'})",
        "CREATE (n0 :L2{k17 : false, k16 : 'peDDcpg', id : 2, k13 : -1918217236, k12 : true, k15 : 'w3AvR', k14 : 'WmlON'})",
        "CREATE (n0 :L1:L3{k20 : 'srKIiKuV', k5 : false, k11 : false, k22 : false, k6 : false, k10 : 1241021292, k21 : false, k7 : true, k8 : 'Tq2Dzmi', k23 : false, k9 : -134369876, k19 : true, k18 : -1993341303, id : 3})",
        "CREATE (n0 :L1:L2{k5 : false, k11 : true, k6 : true, k10 : -2079152779, k7 : true, k13 : -305188500, k8 : 'Qs3ERl0', k12 : false, k9 : 1979613674, k15 : 'mbZkbYR', k14 : 'pjYauSb', k17 : true, k16 : 'c9mE1', id : 4})",
        "CREATE (n0 :L1:L2{k5 : true, k11 : true, k6 : false, k7 : true, k13 : -1094893368, k8 : 'O0BlTl', k12 : false, k9 : 880680659, k15 : 'gUcnZHGc', k14 : 'gwHgQvl', k17 : true, k16 : 'RQ14LGcAE', id : 5})",
        "CREATE (n0 :L1:L0{k0 : 1577689414, k1 : true, k2 : 'mSMuFiu9', k3 : true, k4 : -788505078, k5 : false, k11 : true, k6 : true, k10 : -1050191560, k7 : true, k8 : '6NJrXC', k9 : -1540567757, id : 7})",
        "MATCH (n0 {id : 7}), (n1 {id : 4}) MERGE (n0)-[r :T6{k70 : -1053273747, k66 : false, id : 10, k68 : false, k67 : 934917662, k69 : false}]->(n1)",
        "MATCH (n0 {id : 7}), (n1 {id : 1}) MERGE (n0)-[r :T6{k70 : -232155708, k66 : true, id : 13, k68 : false, k69 : true}]->(n1)",
        "MATCH (n0 {id : 3}), (n1 {id : 5}) MERGE (n0)-[r :T7{k71 : 'COuumxi8T', k73 : 'Y9w1dRBI', k72 : -1356324883, k75 : 'UTR9R', k74 : -1588955651, k76 : true, id : 15}]->(n1)",
        "MATCH (n0 {id : 0}), (n1 {id : 2}) MERGE (n0)-[r :T5{k60 : 'bOfRnoR', k62 : 427092828, k61 : -40891162, k64 : -1196059165, k63 : 1449751216, k65 : false, id : 17, k59 : true}]->(n1)",
        "MATCH (n0 {id : 5}), (n1 {id : 7}) MERGE (n0)-[r :T7{k71 : 'd2TzdXQ', k73 : 'ugtzJA', k75 : '2ZEkayjbA', k74 : 1146341737, k76 : true, id : 20}]->(n1)",
        "MATCH (n0 {id : 2}), (n1 {id : 0}) MERGE (n0)-[r :T0{k31 : false, k32 : true, id : 24, k35 : 1385927747, k34 : 1149172548, k36 : -1849713195}]->(n1)",
    ]

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_752")
        for q in self.SETUP:
            self.graph.query(q)

    def test01_reported_queries_agree(self):
        first = "MATCH (n0)<-[r0]-(n1)<-[r1]-(n2)<-[r2]-(n3) WHERE true "
        second = ("MATCH (n4)<-[r3]-(n5{k12:true})<-[r4]-(n6), "
                  "(n0{k6:true})<-[r5]-(n1)<-[r6]-(n2)<-[r7]-(n3 :L3{k23:false}) "
                  "WHERE true RETURN n0.id, r5.id, n1.id")
        for mid in ["", "UNWIND [1] AS a0 "]:
            q = first + mid + second
            res = self.graph.query(q)
            self.env.assertEqual(res.result_set, [[4, 10, 7]], message=q)

    def test02_same_relationship_across_match_clauses(self):
        g = self.db.select_graph("issue_752_min")
        g.query("CREATE (a {id:1}), (b {id:2}), (c {id:3}) WITH * "
                "CREATE (a)-[:R]->(b), (b)-[:R]->(c)")
        res = g.query("MATCH (x)-[r1]->(y) MATCH (x)-[r2]->(y) RETURN x.id, y.id ORDER BY x.id")
        self.env.assertEqual(res.result_set, [[1, 2], [2, 3]])
        res = g.query("MATCH (x)-[r1]->(y)-[r2]->(z) MATCH (x)-[r3]->(y)-[r4]->(z) RETURN x.id, y.id, z.id")
        self.env.assertEqual(res.result_set, [[1, 2, 3]])


class testIssue946():
    """https://github.com/FalkorDB/FalkorDB/issues/946
    A map literal mixing an aggregation with a non-aggregated expression over
    the same variable (`{a: l.v, b: collect(l.v)}`) is invalid Cypher (implicit
    grouping key). It must be rejected with an error (C leaks an internal
    error; Rust silently returns `a: NULL`), never answered with a NULL key."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_946")
        self.graph.query("CREATE (:S {v:1}), (:S {v:2})")

    def _no_silent_null(self, q):
        try:
            res = self.graph.query(q)
        except ResponseError:
            return
        for row in res.result_set:
            self.env.assertNotEqual(row[0].get('a'), None, message=q)

    def test01_aggregation_in_map_after_with(self):
        self._no_silent_null("MATCH (l:S) WITH {a: l.v, b: collect(l.v)} AS m RETURN m")

    def test02_aggregation_in_map_in_return(self):
        self._no_silent_null("MATCH (l:S) RETURN {a: l.v, c: count(l)} AS m")


class testIssue997():
    """https://github.com/FalkorDB/FalkorDB/issues/997
    A CREATE inside CALL {} is skipped when a later filter is the constant
    `false`, but runs when the filter is an equivalent non-foldable expression.
    The write must always happen: one node in both cases."""

    def __init__(self):
        self.env, self.db = Env()

    def test01_create_in_call_runs_before_false_filter(self):
        for i, q in enumerate([
                "CALL { CREATE () } UNWIND 0 AS x MATCH () WHERE false RETURN 0",
                "CALL { CREATE () } UNWIND 0 AS x MATCH () WHERE (CASE WHEN true THEN false ELSE x END) RETURN 0"]):
            g = self.db.select_graph(f"issue_997_{i}")
            res = g.query(q)
            self.env.assertEqual(res.result_set, [])
            res = g.query("MATCH (n) RETURN count(n)")
            self.env.assertEqual(res.result_set, [[1]], message=q)


class testIssue1315():
    """https://github.com/FalkorDB/FalkorDB/issues/1315
    `exists(<pattern>)` (openCypher 9, and the docs at the time) is rejected on
    both engines. `WHERE exists((n)--())` must return the nodes that have at
    least one relationship, like the bare pattern predicate does."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_1315")
        self.graph.query("CREATE (:N {v:1})-[:R]->(:N {v:2}), (:N {v:3})")

    def test01_exists_pattern(self):
        res = self.graph.query("MATCH (n:N) WHERE exists((n)-[]-()) RETURN n.v ORDER BY n.v")
        self.env.assertEqual(res.result_set, [[1], [2]])
        res = self.graph.query("MATCH (n:N) WHERE exists((n)-->()) RETURN n.v ORDER BY n.v")
        self.env.assertEqual(res.result_set, [[1]])


class testIssue1334():
    """https://github.com/FalkorDB/FalkorDB/issues/1334
    A WHERE pattern predicate that introduces a new variable (`() -- (n1)`) is
    executed instead of being rejected; Rust turns it into an uncorrelated
    global existence test. The query must fail with an undefined-variable error."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_1334")
        self.graph.query("CREATE (:X {n:1})-[:E]->(:X {n:2})")

    def test01_pattern_predicate_cannot_declare_variable(self):
        _exta_expect_error(self.env, self.graph,
                           "MATCH (n0) WITH * WHERE () -- (n1) RETURN *", "n1")
        _exta_expect_error(self.env, self.graph,
                           "MATCH (n0) WHERE (n0)--(n1) RETURN n0.n")


class testIssue1393():
    """https://github.com/FalkorDB/FalkorDB/issues/1393
    `MATCH (n) DELETE n RETURN n.k1` still returns the deleted node's property.
    Reading a property of an entity deleted earlier in the same query must
    raise an error (as Neo4j/Memgraph do). Note test_access_del_entity.py pins
    the current behaviour."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_1393")

    def test01_property_of_deleted_node(self):
        self.graph.query("CREATE (:l1 {k1:1})")
        _exta_expect_error(self.env, self.graph,
                           "MATCH (n:l1) DELETE n RETURN n.k1", "deleted")

    def test02_property_of_deleted_relationship(self):
        self.graph.query("CREATE (:P)-[:R {w:9}]->(:P)")
        _exta_expect_error(self.env, self.graph,
                           "MATCH ()-[r:R]->() DELETE r RETURN r.w", "deleted")


MEM_CAP_ERR = "Query's mem consumption exceeded capacity"


def _extb_expect_error(env, graph, query, *fragments, params=None):
    """Run `query`, expect a ResponseError whose message contains every
    fragment, then check the server is still up."""
    try:
        graph.query(query, params)
        env.assertTrue(False, message=f"expected an error for: {query[:80]}")
    except ResponseError as e:
        for f in fragments:
            env.assertContains(f, str(e))
    env.assertTrue(env.getConnection().ping())


def _extb_expect_error_under_mem_cap(env, db, graph, query, cap=32 * 1024 * 1024):
    """With QUERY_MEM_CAPACITY set to `cap`, `query` must fail with the
    memory-capacity error (not succeed, not kill the server)."""
    db.config_set("QUERY_MEM_CAPACITY", cap)
    try:
        _extb_expect_error(env, graph, query, MEM_CAP_ERR)
    finally:
        try:
            db.config_set("QUERY_MEM_CAPACITY", 0)
        except Exception:
            pass


def _extb_expect_survives(env, graph, query):
    """Default configuration: `query` may fail with a clean error, but must not
    take the server down."""
    try:
        graph.query(query)
    except ResponseError:
        pass
    except (redis.exceptions.ConnectionError, ConnectionError):
        pass  # reported by the ping below
    alive = False
    try:
        alive = env.getConnection().ping()
    except Exception:
        alive = False
    env.assertTrue(alive, message="server died running: " + query[:80])


class testIssue1451():
    """https://github.com/FalkorDB/FalkorDB/issues/1451
    `RETURN p.n + COUNT(p)` mixes a non-grouped variable with an aggregate in one
    expression and silently returns a single null. It must be rejected at
    compile time (openCypher: AmbiguousAggregationExpression)."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("ext_b_issue_1451")
        self.graph.query("CREATE (:P {n:'a'})-[:R]->(:P {n:'b'})")

    def test01_mixed_aggregate_expression_is_rejected(self):
        _extb_expect_error(self.env, self.graph, "MATCH (p) RETURN p.n + COUNT(p)")

    def test02_grouped_workaround_still_works(self):
        res = self.graph.query(
            "MATCH (p) WITH p.n AS n, COUNT(p) AS c RETURN n + c ORDER BY n + c")
        self.env.assertEqual(res.result_set, [['a1'], ['b1']])


class testIssue1482():
    """https://github.com/FalkorDB/FalkorDB/issues/1482
    `CALL { UNWIND range(...) CREATE ... DETACH DELETE ... }` buffers every write
    until the subquery finishes, so one query can exhaust memory and OOM-kill the
    server. Expected: the query is cut off by the per-query memory cap with a
    clean error, and on the default configuration it must not kill the server."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("ext_b_issue_1482")

    def test01_call_subquery_writes_respect_mem_cap(self):
        # ~300 MB unbounded on the Rust engine; must stop at a 32 MB cap.
        _extb_expect_error_under_mem_cap(
            self.env, self.db, self.graph,
            "CALL {UNWIND range(0, 1000000) AS v1 CREATE (n:Node) DETACH DELETE n} "
            "RETURN count(*)")

    def test02_reporter_query_does_not_kill_server(self):
        # Last: OOM-kills an unfixed server in a memory-capped container.
        _extb_expect_survives(
            self.env, self.graph,
            "CALL {UNWIND range(0, 2147483647) AS v1 CREATE (n:Node) DETACH DELETE n} "
            "RETURN count(*)")


class testIssue1483():
    """https://github.com/FalkorDB/FalkorDB/issues/1483
    Three chained `UNWIND range(1, 1000)` feeding CREATE build a 10^9-row
    cartesian product that is buffered until the server is OOM-killed. Expected:
    a clean memory-cap error, and no server death on the default configuration."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("ext_b_issue_1483")

    def test01_unwind_chain_create_respects_mem_cap(self):
        # 100^3 = 1M creations (~250 MB) must stop at a 32 MB cap.
        _extb_expect_error_under_mem_cap(
            self.env, self.db, self.graph,
            "WITH range(1, 100) AS list UNWIND list AS a UNWIND list AS b "
            "UNWIND list AS c CREATE (:CartesianNode {id: a + b + c})")

    def test02_reporter_query_does_not_kill_server(self):
        # Last: OOM-kills an unfixed server in a memory-capped container.
        _extb_expect_survives(
            self.env, self.graph,
            "WITH range(1, 1000) AS list UNWIND list AS a UNWIND list AS b "
            "UNWIND list AS c CREATE (:CartesianNode {id: a + b + c})")


class testIssue1485():
    """https://github.com/FalkorDB/FalkorDB/issues/1485
    `MATCH p=(:VLP)-[*1..10]->(:VLP) RETURN collect(p)` on a small complete graph
    materialises an exponential number of paths in collect() until the server is
    OOM-killed. Expected: a clean memory-cap error, and no server death on the
    default configuration."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("ext_b_issue_1485")

    def _complete_graph(self, n):
        self.graph.query("MATCH (n) DETACH DELETE n")
        self.graph.query(f"UNWIND range(1, {n}) AS i CREATE (:VLP {{id: i}})")
        self.graph.query("MATCH (a:VLP), (b:VLP) WHERE a.id <> b.id "
                         "CREATE (a)-[:NEXT]->(b)")

    def test01_collect_paths_respects_mem_cap(self):
        # 10-node complete graph, paths up to 5 hops: ~200 MB unbounded.
        self._complete_graph(10)
        _extb_expect_error_under_mem_cap(
            self.env, self.db, self.graph,
            "MATCH p=(:VLP)-[*1..5]->(:VLP) RETURN size(collect(p))")

    def test02_reporter_query_does_not_kill_server(self):
        # Last: OOM-kills an unfixed server in a memory-capped container.
        self._complete_graph(50)
        _extb_expect_survives(
            self.env, self.graph,
            "MATCH p=(:VLP)-[*1..10]->(:VLP) RETURN collect(p)")


class testIssue1486():
    """https://github.com/FalkorDB/FalkorDB/issues/1486
    A nested list comprehension `[x IN axis | [y IN axis | 1]]` over
    range(1, 100000) allocates a 10^10-element matrix with no size check and the
    server is OOM-killed; the allocation is not charged to QUERY_MEM_CAPACITY.
    Expected: a clean memory-cap error, and no server death by default."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("ext_b_issue_1486")

    def test01_nested_comprehension_respects_mem_cap(self):
        # 3000x3000 matrix (~230 MB) built and reduced to its size inside the
        # query, so only the comprehension itself can trip the 32 MB cap.
        _extb_expect_error_under_mem_cap(
            self.env, self.db, self.graph,
            "WITH range(1, 3000) AS axis RETURN size([x IN axis | [y IN axis | 1]])")

    def test02_reporter_query_does_not_kill_server(self):
        # Last: OOM-kills an unfixed server in a memory-capped container.
        _extb_expect_survives(
            self.env, self.graph,
            "WITH range(1, 100000) AS axis RETURN [x IN axis | [y IN axis | 1]] AS Matrix")


class testIssue1848():
    """https://github.com/FalkorDB/FalkorDB/issues/1848
    Two parallel edges of different types between the same endpoints, matched by
    `-[:FRIENDS_WITH|WORKS_WITH]->` and grouped with count(*), are counted once.
    Each relationship is a distinct match: expected `Bob, 2`."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("ext_b_issue_1848")
        self.graph.query("CREATE (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}), "
                         "(a)-[:FRIENDS_WITH]->(b), (a)-[:WORKS_WITH]->(b)")

    def test01_parallel_edges_of_different_types_counted(self):
        res = self.graph.query(
            "MATCH (p:Person) WHERE p.name = 'Alice' "
            "MATCH (p)-[:FRIENDS_WITH|WORKS_WITH]->(friend) "
            "WITH friend, count(*) AS connectionCount "
            "RETURN friend.name AS friendName, connectionCount ORDER BY friendName")
        self.env.assertEqual(res.result_set, [['Bob', 2]])


class testIssue1916():
    """https://github.com/FalkorDB/FalkorDB/issues/1916
    A map parameter used as a pattern's inlined properties (`CREATE (n:t $p)`,
    MERGE, MATCH, `[r:R $p]`) fails with "Encountered unhandled type in inlined
    properties". It must behave like the same map written literally."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("ext_b_issue_1916")
        self.p = {'p': {'name': 'a'}}

    def test01_create_node_with_param_properties(self):
        res = self.graph.query("CREATE (n:T1 $p) RETURN n.name", self.p)
        self.env.assertEqual(res.nodes_created, 1)
        self.env.assertEqual(res.result_set, [['a']])

    def test02_merge_node_with_param_properties(self):
        res = self.graph.query("MERGE (n:T2 $p) RETURN n.name", self.p)
        self.env.assertEqual(res.nodes_created, 1)
        self.env.assertEqual(res.result_set, [['a']])

    def test03_match_node_with_param_properties(self):
        self.graph.query("CREATE (:T3 {name: 'a'}), (:T3 {name: 'b'})")
        res = self.graph.query("MATCH (n:T3 $p) RETURN n.name", self.p)
        self.env.assertEqual(res.result_set, [['a']])

    def test04_create_edge_with_param_properties(self):
        res = self.graph.query("CREATE (:X)-[r:R $p]->(:Y) RETURN r.name", self.p)
        self.env.assertEqual(res.relationships_created, 1)
        self.env.assertEqual(res.result_set, [['a']])


class testIssue1944():
    """https://github.com/FalkorDB/FalkorDB/issues/1944
    In a converging pattern with anonymous relationships,
    `(a)-[:FRIEND]->(b)<-[:FRIEND]-(c)`, one physical edge fills both slots and
    fabricates rows. Relationship uniqueness forbids that: 2 rows for two edges,
    0 rows for a single edge."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("ext_b_issue_1944")

    def test01_converging_pattern_does_not_reuse_edge(self):
        self.graph.query("MATCH (n) DETACH DELETE n")
        self.graph.query("CREATE (:Person {name:'Alice'})-[:FRIEND]->(:Person {name:'Bob'})"
                         "<-[:FRIEND]-(:Person {name:'Charlie'})")
        res = self.graph.query(
            "MATCH (a:Person)-[:FRIEND]->(b:Person)<-[:FRIEND]-(c:Person) "
            "RETURN a.name AS a, b.name AS b, c.name AS c ORDER BY a, c")
        self.env.assertEqual(res.result_set, [['Alice', 'Bob', 'Charlie'],
                                              ['Charlie', 'Bob', 'Alice']])

    def test02_single_edge_cannot_fill_both_slots(self):
        self.graph.query("MATCH (n) DETACH DELETE n")
        self.graph.query("CREATE (:Person {name:'Alice'})-[:FRIEND]->(:Person {name:'Bob'})")
        res = self.graph.query(
            "MATCH (a:Person)-[:FRIEND]->(b:Person)<-[:FRIEND]-(c:Person) "
            "RETURN a.name AS a, b.name AS b, c.name AS c")
        self.env.assertEqual(res.result_set, [])


class testIssue1946():
    """https://github.com/FalkorDB/FalkorDB/issues/1946
    `MATCH (p:Person) DELETE p RETURN p` returns the deleted node with its
    pre-delete properties. Expected (Neo4j, per triage): an empty node value.
    Note: tests/flow/test_access_del_entity.py testAccessDelNode.test02 asserts
    the opposite (current) behaviour."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("ext_b_issue_1946")

    def test01_deleted_node_returned_without_properties(self):
        self.graph.query("CREATE (:Person {name:'Alice', age:30})")
        res = self.graph.query("MATCH (p:Person) DELETE p RETURN p")
        self.env.assertEqual(res.nodes_deleted, 1)
        self.env.assertEqual(len(res.result_set), 1)
        self.env.assertEqual(res.result_set[0][0].properties, {})


class testIssue1948():
    """https://github.com/FalkorDB/FalkorDB/issues/1948
    Three parallel `[:WORKS_FOR]` edges between the same two nodes collapse into
    one row when aggregated grouped by an endpoint (`WITH c, count(*) ...`).
    Expected: `TechCorp, 3, 3, [Alice, Alice, Alice]`."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("ext_b_issue_1948")
        self.graph.query("CREATE (p:Person {name: 'Alice', age: 30}), "
                         "(c:Company {name: 'TechCorp', industry: 'Technology'}), "
                         "(p)-[:WORKS_FOR]->(c), (p)-[:WORKS_FOR]->(c), (p)-[:WORKS_FOR]->(c)")

    def test01_grouped_aggregation_keeps_parallel_edges(self):
        res = self.graph.query(
            "MATCH (c:Company {name: 'TechCorp'})<-[r:WORKS_FOR]-(p:Person) "
            "WITH c, count(*) AS cstar, count(p) AS cp, collect(p.name) AS names "
            "RETURN c.name AS company_name, cstar, cp, names")
        self.env.assertEqual(res.result_set,
                             [['TechCorp', 3, 3, ['Alice', 'Alice', 'Alice']]])


class testIssue1953():
    """https://github.com/FalkorDB/FalkorDB/issues/1953
    A pattern comprehension whose pattern matches but whose projection is null,
    `[(n)-[:FRIEND_OF]-(m) | m.name]` with `m` lacking `name`, drops the element.
    Expected: `[null]`, one element per match."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("ext_b_issue_1953")
        self.graph.query("CREATE (:Person {name:'Alice'})-[:FRIEND_OF]->(:Person)")

    def test01_null_projection_kept(self):
        res = self.graph.query(
            "MATCH (n:Person {name:'Alice'}) "
            "RETURN [(n)-[:FRIEND_OF]-(m) | m.name] AS friendNames")
        self.env.assertEqual(res.result_set, [[[None]]])


def _extc_expect_error(env, graph, query, *fragments, params=None):
    """Run `query`, expect a ResponseError whose message contains every
    fragment (case-insensitive), then check the server is still up."""
    try:
        graph.query(query, params)
        env.assertTrue(False, message=f"expected an error for: {query[:80]}")
    except ResponseError as e:
        for f in fragments:
            env.assertContains(f.lower(), str(e).lower())
    env.assertTrue(env.getConnection().ping())


class testIssue2000():
    """https://github.com/FalkorDB/FalkorDB/issues/2000
    A relationship variable bound by one MATCH and carried through WITH must
    keep its identity in a later MATCH: `(x)-[r]->(y)` may only match the
    already-bound KNOWS edge, not Alice's other (FRIEND) edge."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2000")
        self.graph.query(
            "CREATE (a:Person {name:'Alice'}), (b:Person {name:'Bob'}), "
            "(c:Person {name:'Charlie'}), (a)-[:KNOWS]->(b), (a)-[:FRIEND]->(c)")

    def test01_bound_relationship_keeps_identity(self):
        q = """MATCH ()-[r:KNOWS]->()
               WITH r
               MATCH (x:Person)-[r]->(y)
               RETURN x.name AS x, y.name AS y
               ORDER BY x, y"""
        res = self.graph.query(q).result_set
        self.env.assertEqual(res, [['Alice', 'Bob']])


class testIssue2001():
    """https://github.com/FalkorDB/FalkorDB/issues/2001
    Durations are not ordered in Cypher: `<`, `>`, `<=`, `>=` between two
    durations must evaluate to null (as Neo4j does), while `=` still compares
    them. Both engines return booleans."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2001")

    def test01_duration_ordering_is_null(self):
        q = """RETURN duration('P2Y') > duration('P1Y') AS gt,
                      duration('P6M') > duration('P1Y') AS gt2,
                      duration('P1Y') < duration('P2Y') AS lt,
                      duration('P1Y') >= duration('P1Y') AS ge,
                      duration('P2Y') = duration('P2Y') AS eq"""
        res = self.graph.query(q).result_set
        self.env.assertEqual(res, [[None, None, None, None, True]])


class testIssue2003():
    """https://github.com/FalkorDB/FalkorDB/issues/2003
    A list comprehension over `collect(...)` whose projection also reads an
    outer row variable (`x + n.age`) yields null instead of the sum. With
    collect(v.age) = [30] and n.age = 25 the result must be [55]."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2003")
        self.graph.query(
            "CREATE (:Person {name:'Alice', age:30})-[:KNOWS]->"
            "(:Person {name:'Bob', age:25})")

    def test01_comprehension_over_collect_reads_outer_variable(self):
        q = """MATCH (v:Person {name:'Alice'})--(n:Person)
               RETURN n.age AS age,
                      [x IN collect(v.age) WHERE x > 20 | x + n.age] AS vals
               ORDER BY age"""
        res = self.graph.query(q).result_set
        self.env.assertEqual(res, [[25, [55]]])


class testIssue2007():
    """https://github.com/FalkorDB/FalkorDB/issues/2007
    An anonymous undirected relationship slot between two bound-by-property
    endpoints collapses two parallel KNOWS edges into one match. Each edge is
    a distinct match, so count(*) must be 2 and the projection must yield 2
    rows."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2007")
        self.graph.query(
            "CREATE (a:Person {name:'Alice'}), (b:Person {name:'Bob'}), "
            "(a)-[:KNOWS]->(b), (a)-[:KNOWS]->(b)")

    def test01_parallel_edges_counted_separately(self):
        q = """MATCH (:Person {name:'Alice'})-[:KNOWS]-(:Person {name:'Bob'})
               RETURN count(*) AS cnt"""
        self.env.assertEqual(self.graph.query(q).result_set, [[2]])

    def test02_parallel_edges_projected_separately(self):
        q = """MATCH (p:Person {name:'Alice'})-[:KNOWS]-(friend:Person {name:'Bob'})
               RETURN p.name, friend.name"""
        self.env.assertEqual(self.graph.query(q).result_set,
                             [['Alice', 'Bob'], ['Alice', 'Bob']])


class testIssue2098():
    """https://github.com/FalkorDB/FalkorDB/issues/2098
    `=` between Vectorf32 values was always true from dimension 5 up (C
    4.18), and later always false at every dimension (Rust, #2604). A stored
    vector must equal only an element-wise identical vector, at any
    dimension."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2098")

    def test01_vector_equality_filter_by_dimension(self):
        for dim in (4, 5, 8, 16):
            label = f"V{dim}"
            a = [0.1] * dim
            b = [0.2] * (dim - 1) + [0.9]
            self.graph.query(
                f"CREATE (:{label} {{id:'a', v: vecf32($a)}}), "
                f"(:{label} {{id:'b', v: vecf32($b)}})", {'a': a, 'b': b})
            res = self.graph.query(
                f"MATCH (n:{label}) WHERE n.v = vecf32($a) RETURN n.id",
                {'a': a}).result_set
            self.env.assertEqual(res, [['a']], message=f"dim {dim}")
            res = self.graph.query(
                f"MATCH (n:{label}) RETURN n.id, n.v = n.v, n.v <> vecf32($a) "
                "ORDER BY n.id", {'a': a}).result_set
            self.env.assertEqual(res, [['a', True, False], ['b', True, True]],
                                 message=f"dim {dim}")

    def test02_vector_literal_equality(self):
        q = """RETURN vecf32([1.0,2.0]) = vecf32([1.0,2.0]) AS eq,
                      vecf32([1.0,2.0]) <> vecf32([1.0,2.0]) AS neq,
                      vecf32([0.1,0.1,0.1,0.1,0.1]) =
                      vecf32([0.2,0.2,0.2,0.2,0.9]) AS eq5"""
        self.env.assertEqual(self.graph.query(q).result_set,
                             [[True, False, False]])


class testIssue2099():
    """https://github.com/FalkorDB/FalkorDB/issues/2099
    SET of a Vectorf32 property that already holds a different Vectorf32
    silently does nothing (properties_set 0, old value kept). The new vector
    must be written and reported as one property set."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2099")
        self.graph.query("CREATE (:S {id:'a', v: vecf32([1.0,0.0,0.0,0.0])})")

    def test01_set_vector_over_vector(self):
        res = self.graph.query(
            "MATCH (n:S {id:'a'}) SET n.v = vecf32([0.5,0.5,0.0,0.0]) RETURN n.id")
        self.env.assertEqual(res.result_set, [['a']])
        self.env.assertEqual(res.properties_set, 1)
        d = self.graph.query(
            "MATCH (n:S {id:'a'}) "
            "RETURN vec.cosineDistance(n.v, vecf32([0.5,0.5,0.0,0.0]))"
        ).result_set[0][0]
        self.env.assertTrue(abs(d) < 1e-5, message=f"distance {d}")

    def test02_set_vector_over_vector_with_param(self):
        res = self.graph.query("MATCH (n:S {id:'a'}) SET n.v = vecf32($v)",
                               {'v': [0.0, 0.0, 1.0, 0.0]})
        self.env.assertEqual(res.properties_set, 1)
        d = self.graph.query(
            "MATCH (n:S {id:'a'}) "
            "RETURN vec.cosineDistance(n.v, vecf32([0.0,0.0,1.0,0.0]))"
        ).result_set[0][0]
        self.env.assertTrue(abs(d) < 1e-5, message=f"distance {d}")


class testIssue2179():
    """https://github.com/FalkorDB/FalkorDB/issues/2179
    Re-running a `MERGE ... ON MATCH SET c.label_vector = vecf32($v)` with a
    new embedding keeps the old vector, so a vector-index query with the new
    embedding no longer finds the node (duplicate root cause of #2099). The
    index must return the updated node at distance 0."""

    MERGE = """MERGE (c:BaseClass {cid: $cid})
               ON CREATE SET c.name = $name, c.label_vector = vecf32($label_vector)
               ON MATCH SET c.name = $name, c.label_vector = vecf32($label_vector)"""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2179")
        self.graph.create_node_vector_index("BaseClass", "label_vector", dim=2,
                                            similarity_function="cosine")
        wait_for_indices_to_sync(self.graph)

    def _top(self, v):
        q = """CALL db.idx.vector.queryNodes('BaseClass', 'label_vector', 1, vecf32($v))
               YIELD node, score RETURN node.cid, score"""
        return self.graph.query(q, {'v': v}).result_set

    def test01_merge_on_match_set_updates_vector_index(self):
        self.graph.query(self.MERGE, {'cid': 1, 'name': 'one', 'label_vector': [1.0, 0.0]})
        self.graph.query(self.MERGE, {'cid': 2, 'name': 'two', 'label_vector': [0.0, 1.0]})
        res = self._top([1.0, 0.0])
        self.env.assertEqual(res[0][0], 1)

        # re-ingest cid 1 with a new embedding through the ON MATCH branch
        res = self.graph.query(self.MERGE, {'cid': 1, 'name': 'one',
                                            'label_vector': [-1.0, 0.0]})
        self.env.assertEqual(res.nodes_created, 0)
        self.env.assertEqual(res.properties_set, 1)

        res = self._top([-1.0, 0.0])
        self.env.assertEqual(len(res), 1)
        self.env.assertEqual(res[0][0], 1)
        self.env.assertTrue(abs(res[0][1]) < 1e-5, message=f"score {res}")


class testIssue2105():
    """https://github.com/FalkorDB/FalkorDB/issues/2105
    Relationship uniqueness holds across a whole MATCH, so the split form
    `(a)-[*1..1]-()-[*1..1]-(b)` must not reuse a relationship ([r1,r1],
    [r3,r3]); it must agree with `[*2..2]` and with the explicit two-hop form:
    exactly the paths [r1,r2] and [r2,r1]."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2105")
        self.graph.query(
            "CREATE (a:N {id:1}), (b:N {id:2}), (a)-[:R {eid:'r1'}]->(b), "
            "(b)-[:R {eid:'r2'}]->(a), (a)-[:R {eid:'r3'}]->(a)")

    def test01_split_var_len_enforces_relationship_uniqueness(self):
        q = """MATCH p=(a:N {id:1})-[*1..1]-()-[*1..1]-(b:N {id:1})
               RETURN [rel IN relationships(p) | rel.eid] AS eids ORDER BY eids"""
        res = self.graph.query(q).result_set
        self.env.assertEqual(res, [[['r1', 'r2']], [['r2', 'r1']]])

    def test02_explicit_two_hop_enforces_relationship_uniqueness(self):
        q = """MATCH (a:N {id:1})-[r]-(m)-[s]-(b:N {id:1})
               RETURN r.eid, s.eid ORDER BY r.eid, s.eid"""
        res = self.graph.query(q).result_set
        self.env.assertEqual(res, [['r1', 'r2'], ['r2', 'r1']])

    def test03_exact_length_var_len(self):
        # Crashes the C server (SIGSEGV, #2542, still on v4.22.0); kept last.
        q = """MATCH p=(a:N {id:1})-[*2..2]-(b:N {id:1})
               RETURN [rel IN relationships(p) | rel.eid] AS eids ORDER BY eids"""
        res = self.graph.query(q).result_set
        self.env.assertEqual(res, [[['r1', 'r2']], [['r2', 'r1']]])
        self.env.assertTrue(self.env.getConnection().ping())


class testIssue2134():
    """https://github.com/FalkorDB/FalkorDB/issues/2134
    A large single-query edge delete `UNWIND $ids AS pid MATCH (d:D {did:pid})
    -[r:L]->() DELETE r` crashed C 4.16. Triage found Rust drops the index scan
    when the inline value is UNWIND-bound and scans every edge per row. The plan
    must use the index, and a scaled-down bulk delete must succeed."""

    N = 3000
    FANOUT = 3

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2134")
        self.graph.query("CREATE INDEX FOR (d:D) ON (d.did)")
        wait_for_indices_to_sync(self.graph)
        self.graph.query(
            "UNWIND range(0, $n - 1) AS i CREATE (d:D {did: i}) "
            "WITH d UNWIND range(1, $f) AS j CREATE (d)-[:L]->(:T)",
            {'n': self.N, 'f': self.FANOUT})

    def test01_unwind_bound_inline_filter_uses_index(self):
        for q in ["UNWIND range(0, 9) AS pid MATCH (d:D {did: pid})-[r:L]->() RETURN count(r)",
                  "UNWIND range(0, 9) AS pid MATCH (d:D)-[r:L]->() WHERE d.did = pid RETURN count(r)"]:
            plan = str(self.graph.explain(q))
            self.env.assertContains("Node By Index Scan", plan)
            self.env.assertEqual(self.graph.query(q).result_set, [[10 * self.FANOUT]])

    def test02_bulk_edge_delete_single_query(self):
        q = """UNWIND $doc_ids AS pid
               MATCH (d:D {did: pid})-[r:L]->()
               DELETE r"""
        res = self.graph.query(q, {'doc_ids': list(range(self.N))})
        self.env.assertEqual(res.relationships_deleted, self.N * self.FANOUT)
        self.env.assertTrue(self.env.getConnection().ping())
        res = self.graph.query("MATCH ()-[r:L]->() RETURN count(r)").result_set
        self.env.assertEqual(res, [[0]])


class testIssue2180():
    """https://github.com/FalkorDB/FalkorDB/issues/2180
    64-bit integer arithmetic silently wraps (INT64_MAX + 1 returns INT64_MIN),
    also when the result is written by SET. Overflowing +, -, * must fail the
    query with an overflow error and write nothing."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2180")

    def test01_overflowing_arithmetic_raises(self):
        for q in ["RETURN 9223372036854775807 + 1 AS result",
                  "RETURN 9223372036854775807 + 2 AS result",
                  "RETURN 9223372036854775807 * 2 AS result",
                  "RETURN -9223372036854775808 - 1 AS result"]:
            _extc_expect_error(self.env, self.graph, q, "overflow")

    def test02_overflowing_set_raises_and_writes_nothing(self):
        self.graph.query("CREATE (:Oracle {_id:'overflow-set', x:9223372036854775807})")
        _extc_expect_error(self.env, self.graph,
                           "MATCH (n:Oracle {_id:'overflow-set'}) SET n.y = n.x + 1",
                           "overflow")
        res = self.graph.query(
            "MATCH (n:Oracle {_id:'overflow-set'}) RETURN n.x, n.y").result_set
        self.env.assertEqual(res, [[9223372036854775807, None]])


class testIssue2182():
    """https://github.com/FalkorDB/FalkorDB/issues/2182
    Finite subnormal float literals such as 2.2e-308 and 1.0e-308 are rejected
    with `Float overflow`. They are valid binary64 values and must be accepted
    and returned (the same value `1.0e-307 / 10` already produces)."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2182")

    def test01_subnormal_float_literals_accepted(self):
        for lit, expected in [("2.3e-308", 2.3e-308), ("2.2e-308", 2.2e-308),
                              ("1.0e-308", 1.0e-308), ("5e-320", 5e-320)]:
            try:
                v = self.graph.query(f"RETURN {lit}").result_set[0][0]
            except ResponseError as e:
                self.env.assertTrue(False, message=f"{lit} rejected: {e}")
                continue
            self.env.assertTrue(v > 0 and abs(v - expected) <= expected * 1e-6,
                                message=f"{lit} -> {v}")
        self.env.assertTrue(self.env.getConnection().ping())


class testIssue2211():
    """https://github.com/FalkorDB/FalkorDB/issues/2211
    trim(), lTrim() and rTrim() without a trim character only strip U+0020.
    They must strip all leading/trailing whitespace: tab, newline, carriage
    return and form feed too."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2211")

    def test01_trim_strips_all_whitespace(self):
        q = r"""RETURN trim('\thello\t') = 'hello',
                       trim('\nhello\n') = 'hello',
                       trim('\rhello\r') = 'hello',
                       trim('\fhello\f') = 'hello',
                       trim(' \t\n hello \r\n') = 'hello'"""
        self.env.assertEqual(self.graph.query(q).result_set,
                             [[True, True, True, True, True]])

    def test02_ltrim_rtrim_strip_all_whitespace(self):
        q = r"""RETURN lTrim('\thello') = 'hello',
                       lTrim('\nhello') = 'hello',
                       rTrim('hello\t') = 'hello',
                       rTrim('hello\n') = 'hello'"""
        self.env.assertEqual(self.graph.query(q).result_set,
                             [[True, True, True, True]])


def _extd_expect_error(env, graph, query, *fragments):
    """Run `query`, expect a ResponseError whose message contains every
    fragment, then check the server is still up."""
    try:
        graph.query(query)
        env.assertTrue(False, message=f"expected an error for: {query[:80]}")
    except ResponseError as e:
        for f in fragments:
            env.assertContains(f, str(e))
    env.assertTrue(env.getConnection().ping())


class testIssue2213():
    """https://github.com/FalkorDB/FalkorDB/issues/2213
    sum() over integer-only input returns a FLOAT, so `sum(x) / 2` becomes a
    float division. It must return INTEGER 5 and `sum(x) / 2` must be 2."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_ext_2213")

    def test01_sum_of_integers_is_integer(self):
        res = self.graph.query(
            "UNWIND [1, 4] AS x "
            "RETURN sum(x) AS total, sum(x) / 2 AS half, typeOf(sum(x)) AS t"
        ).result_set
        total, half, t = res[0]
        self.env.assertEqual(t, "Integer")
        self.env.assertTrue(isinstance(total, int))
        self.env.assertEqual(total, 5)
        self.env.assertEqual(half, 2)


class testIssue2217():
    """https://github.com/FalkorDB/FalkorDB/issues/2217
    Relationship uniqueness is applied across two separate MATCH clauses that
    share a node variable, so a later MATCH may not re-traverse an edge bound
    by an earlier one and rows are dropped. Uniqueness is scoped to one MATCH
    clause: each of these queries must return its row (C does)."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_ext_2217")

    def test01_reported_fixture(self):
        for q in [
            'CREATE (:L0 {k4: true, id: 30})',
            'CREATE (:L0 {k4: true, id: 46})',
            'CREATE ({id: 50})',
            'CREATE (:L3:L2 {id: 81})',
            'CREATE (:L1:L5:L2 {id: 105})',
            'CREATE (:L1 {id: 125})',
            'MATCH (a {id: 50}), (b {id: 105}) CREATE (a)-[:T3 {id: 147}]->(b)',
            'MATCH (a {id: 125}), (b {id: 105}) CREATE (a)-[:T6 {id: 182}]->(b)',
            'MATCH (a {id: 81}), (b {id: 105}) CREATE (a)-[:T6 {id: 195}]->(b)',
            'MATCH (a {id: 125}), (b {id: 30}) CREATE (a)-[:T3 {id: 198}]->(b)',
            'MATCH (a {id: 50}), (b {id: 46}) CREATE (a)-[:T6 {id: 204}]->(b)',
        ]:
            self.graph.query(q)
        q = ("MATCH (n0:L0)<-[r0:T6]-(n1) "
             "MATCH (n0)<-[]-(n1)-[]->(n2:L1), "
             "(n6:L3:L2)-[r4:T6]->(n2)<-[r5:T6]-(n7:L1), "
             "(n7)-[r6:T3]->(n5:L0) RETURN 2 AS a8")
        self.env.assertEqual(self.graph.query(q).result_set, [[2]])

    def test02_edge_reused_across_match_clauses(self):
        self.graph.delete()
        self.graph.query("CREATE (:A {id: 1})-[:R]->(:B {id: 2})")
        for q in [
            "MATCH (a)-[r1:R]->(b) MATCH (a)-[r2:R]->(b) RETURN count(*)",
            "MATCH (a)-[r1:R]->(b) MATCH (a)-[r2:R]->(c) RETURN count(*)",
            "MATCH (a)-[r1:R]->(b) MATCH (c)-[r2:R]->(b) RETURN count(*)",
            "MATCH (a)-[r1:R]->(b) MATCH (a)-[r2:R]-(b) RETURN count(*)",
        ]:
            res = self.graph.query(q).result_set
            self.env.assertEqual(res, [[1]], message=q)


class testIssue2220():
    """https://github.com/FalkorDB/FalkorDB/issues/2220
    percentileDisc() selects an input value but returns it as FLOAT for
    integer input, so `percentileDisc(x, 0.5) / 2` is 0.5. It must return the
    INTEGER 1 and the division must be 0."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_ext_2220")

    def test01_percentile_disc_keeps_integer_type(self):
        res = self.graph.query(
            "UNWIND [1, 4] AS x "
            "RETURN percentileDisc(x, 0.5) AS p, percentileDisc(x, 0.5) / 2 AS half, "
            "typeOf(percentileDisc(x, 0.5)) AS t"
        ).result_set
        p, half, t = res[0]
        self.env.assertEqual(t, "Integer")
        self.env.assertTrue(isinstance(p, int))
        self.env.assertEqual(p, 1)
        self.env.assertEqual(half, 0)


class testIssue2222():
    """https://github.com/FalkorDB/FalkorDB/issues/2222
    Calendar month/year duration components are treated as fixed day counts:
    2024-01-31 + P1M gives 2024-03-02 and P1Y = P365D, P1M = P31D are true.
    Month addition must clamp to the month end (2024-02-29) and calendar
    durations must not equal fixed-day ones."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_ext_2222")

    def test01_calendar_duration_semantics(self):
        res = self.graph.query(
            "RETURN toString(date('2024-01-31') + duration('P1M')) AS m, "
            "toString(date('2023-03-01') + duration('P1Y')) AS y, "
            "duration('P1Y') = duration('P365D') AS eq_year, "
            "duration('P1M') = duration('P31D') AS eq_month"
        ).result_set
        self.env.assertEqual(res, [["2024-02-29", "2024-03-01", False, False]])


class testIssue2224():
    """https://github.com/FalkorDB/FalkorDB/issues/2224
    round() rounds negative .5 ties away from zero. Cypher rounds ties of the
    single-argument round() toward positive infinity: round(-2.5) = -2,
    round(-0.5) = 0, round(-3.5) = -3."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_ext_2224")

    def test01_negative_half_rounds_up(self):
        res = self.graph.query(
            "RETURN round(-2.5), round(-0.5), round(-3.5), round(2.5), "
            "round(-2.4), round(-2.6)").result_set
        self.env.assertEqual(res, [[-2, 0, -3, 3, -2, -3]])


class testIssue2233():
    """https://github.com/FalkorDB/FalkorDB/issues/2233
    Writing a plain list to a vector-indexed property succeeds silently and the
    node is left out of the vector index (also when SET replaces an indexed
    vecf32). The write must either be rejected or the list coerced and
    indexed; silently un-indexing it is wrong."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_ext_2233")

    def _indexed_ids(self):
        res = self.graph.query(
            "CALL db.idx.vector.queryNodes('Doc', 'embedding', 10, vecf32([1.0, 0.0])) "
            "YIELD node RETURN node.id ORDER BY node.id").result_set
        return [r[0] for r in res]

    def _rejected_or_indexed(self, write, node_id):
        try:
            self.graph.query(write)
        except ResponseError:
            return  # rejected at write time: acceptable
        self.env.assertContains(node_id, self._indexed_ids())

    def test01_plain_list_create_is_indexed_or_rejected(self):
        self.graph.query("CREATE VECTOR INDEX FOR (n:Doc) ON (n.embedding) "
                         "OPTIONS {dimension: 2, similarityFunction: 'cosine'}")
        self.graph.query("CREATE (:Doc {id: 'wrapped', embedding: vecf32([1.0, 0.0])})")
        wait_for_indices_to_sync(self.graph)
        self.env.assertEqual(self._indexed_ids(), ["wrapped"])
        self._rejected_or_indexed(
            "CREATE (:Doc {id: 'plain', embedding: [1.0, 0.0]})", "plain")

    def test02_plain_list_set_does_not_evict(self):
        self._rejected_or_indexed(
            "MATCH (n:Doc {id: 'wrapped'}) SET n.embedding = [1.0, 0.0]", "wrapped")


class testIssue2274():
    """https://github.com/FalkorDB/FalkorDB/issues/2274
    `CREATE (n),(n)` silently creates one node, and the Rust engine also
    accepts `CREATE (n) CREATE (n)`, which C rejects. Both redeclare `n` in a
    CREATE and must be rejected without creating anything."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_ext_2274")

    def _count(self):
        return self.graph.query("MATCH (n) RETURN count(n)").result_set[0][0]

    def test01_separate_create_clauses_rejected(self):
        _extd_expect_error(self.env, self.graph, "CREATE (n) CREATE (n)",
                           "redeclared")
        self.env.assertEqual(self._count(), 0)

    def test02_comma_separated_patterns_rejected(self):
        _extd_expect_error(self.env, self.graph, "CREATE (n),(n)")
        self.env.assertEqual(self._count(), 0)


class testIssue2282():
    """https://github.com/FalkorDB/FalkorDB/issues/2282
    toInteger(' 42 ') returns null although toInteger('42') is 42. Whitespace
    around a valid integer string must be ignored and return 42."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_ext_2282")

    def test01_padded_string_converts(self):
        res = self.graph.query(
            "RETURN toInteger(' 42 ') AS v, toInteger('42') AS w").result_set
        self.env.assertEqual(res, [[42, 42]])


class testIssue2283():
    """https://github.com/FalkorDB/FalkorDB/issues/2283
    Relationships created earlier in the same query (behind a WITH barrier) are
    invisible to a later variable-length MATCH, which finds no paths; it must
    see them (900 paths). On Rust the reported query fails one step earlier:
    a cross-variable WHERE (a.id <> b.id) over nodes created earlier in the
    query drops every row, so no relationships are created at all."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_ext_2283")

    def test01_reported_query(self):
        q = ("UNWIND range(1, 6) AS i CREATE (:N {id: i}) "
             "WITH count(*) AS ignored "
             "MATCH (a:N), (b:N) WHERE a.id <> b.id CREATE (a)-[:R]->(b) "
             "WITH count(*) AS ignored "
             "MATCH p=(:N)-[:R*1..3]->(:N) RETURN count(p) AS paths")
        res = self.graph.query(q)
        self.env.assertEqual(res.relationships_created, 30)
        self.env.assertEqual(res.result_set, [[900]])

    def test02_var_len_match_sees_earlier_creates(self):
        # C fails here: the var-len MATCH does not see the new relationships.
        self.graph.delete()
        q = ("UNWIND range(1, 6) AS i CREATE (:N {id: i}) "
             "WITH count(*) AS ignored "
             "MATCH (a:N), (b:N) WITH a, b WHERE a <> b CREATE (a)-[:R]->(b) "
             "WITH count(*) AS ignored "
             "MATCH p=(:N)-[:R*1..3]->(:N) RETURN count(p) AS paths")
        res = self.graph.query(q)
        self.env.assertEqual(res.relationships_created, 30)
        self.env.assertEqual(res.result_set, [[900]])

    def test03_cross_variable_filter_sees_earlier_creates(self):
        # Rust fails here: the predicate drops every row although projecting
        # a.id < b.id in the same position yields true for one pair.
        self.graph.delete()
        res = self.graph.query(
            "UNWIND range(1, 2) AS i CREATE (:N {id: i}) "
            "WITH count(*) AS ignored "
            "MATCH (a:N), (b:N) WHERE a.id < b.id RETURN count(*)").result_set
        self.env.assertEqual(res, [[1]])


class testIssue2284():
    """https://github.com/FalkorDB/FalkorDB/issues/2284
    A relationship-dependent UNION branch inside CALL {} returns no row when
    the relationships were created earlier in the same query. Both branches
    must contribute: collect(s) is [1, 2]."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_ext_2284")

    def test01_union_branch_sees_earlier_creates(self):
        q = ("CREATE (x:N {id: 2})-[:R]->(a:N {id: 1}), (a)-[:R]->(y:N {id: 3}) "
             "WITH a CALL { WITH a RETURN 1 AS s "
             "UNION MATCH (x:N {id: 2})-[:R]-(a)-[:R]-(y:N {id: 3}) RETURN 2 AS s } "
             "WITH s ORDER BY s RETURN collect(s) AS values")
        res = self.graph.query(q)
        self.env.assertEqual(res.relationships_created, 2)
        self.env.assertEqual(res.result_set, [[[1, 2]]])


class testIssue2286():
    """https://github.com/FalkorDB/FalkorDB/issues/2286
    After CREATE / var-len MATCH / SET in one query, `CASE WHEN paths > 0 AND
    count(n) > 0` yields FAIL on its own but PASS once sibling columns are
    projected. Projecting extra columns must not change it: both are PASS."""

    PREFIX = ("CREATE (a:N {id: 1, v: 0})-[:R]->(b:N {id: 2}), (b)-[:R]->(a) "
              "WITH count(*) AS ignored "
              "MATCH p=(n:N)-[:R*2..2]->(n) SET n.v = 1 "
              "WITH count(p) AS paths MATCH (n:N {v: 1}) ")
    CASE = "CASE WHEN paths > 0 AND count(n) > 0 THEN 'PASS' ELSE 'FAIL' END AS value"

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_ext_2286")

    def test01_with_diagnostic_columns(self):
        res = self.graph.query(
            self.PREFIX + "RETURN paths, count(n) AS updated, " + self.CASE).result_set
        self.env.assertEqual(res, [[2, 2, "PASS"]])

    def test02_case_alone(self):
        self.graph.delete()
        res = self.graph.query(self.PREFIX + "RETURN " + self.CASE).result_set
        self.env.assertEqual(res, [["PASS"]])


class testIssue2293():
    """https://github.com/FalkorDB/FalkorDB/issues/2293
    Relationship uniqueness is not enforced across comma-separated pattern
    parts of one MATCH, so a variable-length path (or another hop) reuses an
    edge already bound in the same clause. Such matches must be discarded;
    separate MATCH clauses may still reuse edges."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_ext_2293")

    def test01_var_len_path_cannot_reuse_bound_edge(self):
        self.graph.query(
            "CREATE (:A {id: 1})-[:T0 {rid: 10}]->(:L1 {id: 2})"
            "-[:X {rid: 11}]->(:C {id: 3})-[:Y {rid: 12}]->(:D {id: 4})")
        res = self.graph.query(
            "MATCH (n0:A)-[r0:T0]-(n1:L1), p=(n2:A)-[*3]-(n3:D) "
            "RETURN id(r0), [r IN relationships(p) | id(r)]").result_set
        self.env.assertEqual(res, [])

    def test02_single_edge_graph(self):
        self.graph.delete()
        self.graph.query("CREATE (:A {n: 1})-[:R]->(:A {n: 2})")
        for q in ["MATCH (x)-[r1]-(y), (a)-[r2]-(b) RETURN count(*)",
                  "MATCH p=(x)-[*1]-(y), (a)-[r2]-(b) RETURN count(*)"]:
            self.env.assertEqual(self.graph.query(q).result_set, [[0]], message=q)
        # control: separate MATCH clauses may reuse the edge
        res = self.graph.query(
            "MATCH (x)-[r1]-(y) MATCH (a)-[r2]-(b) RETURN count(*)").result_set
        self.env.assertEqual(res, [[4]])


class testIssue2324():
    """https://github.com/FalkorDB/FalkorDB/issues/2324
    `MATCH (n)` with `n` already bound to a STRING passes the string through
    as if it matched a node. A bound non-node value in a node pattern must
    raise a type error (as `MATCH (n:Label)` already does)."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_ext_2324")

    def test01_bound_string_in_node_pattern_is_rejected(self):
        _extd_expect_error(self.env, self.graph,
                           "UNWIND ['zzz'] AS n MATCH (n) RETURN n")
        _extd_expect_error(self.env, self.graph,
                           "WITH 'zzz' AS n MATCH (n) RETURN n")


def _exte_scalar(graph, query):
    return graph.query(query).result_set[0][0]


class testIssue2325():
    """https://github.com/FalkorDB/FalkorDB/issues/2325
    Two parallel R0 edges between the same pair of nodes: when the edge
    variable is matched but never read downstream, the rows it contributes are
    collapsed into one. Row multiplicity must not depend on whether the edge is
    projected, so both queries must return 2 rows."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2325")
        self.graph.query(
            "CREATE (m:M {k3: 1, k11: 5}), (n0:N {id: 1}), (n2:N {id: 2, k11: 7}), "
            "(m)-[:R0 {x: 1}]->(n0), (m)-[:R0 {x: 2}]->(n0), "
            "(m)-[:rt5 {k8: true}]->(n2)")

    def test01_unread_edge_keeps_row_multiplicity(self):
        q = ("MATCH (n0)<-[r0:R0]-(n1 {k3: 1})-[r1:rt5 {k8: true}]->(n2) "
             "RETURN n2.k11 AS a6, n1.k11 AS a7")
        self.env.assertEqual(self.graph.query(q).result_set, [[7, 5], [7, 5]])

    def test02_unread_edge_count_star(self):
        q = ("MATCH (n0)<-[r0:R0]-(n1 {k3: 1})-[r1:rt5 {k8: true}]->(n2) "
             "RETURN count(*)")
        self.env.assertEqual(_exte_scalar(self.graph, q), 2)


class testIssue2435():
    """https://github.com/FalkorDB/FalkorDB/issues/2435
    OPTIONAL MATCH (b),(c) where b is bound to null: the null element is
    dropped and c is matched against every node, inflating the result. A null
    bound variable must make the whole comma-separated pattern fail, keeping
    one row per input row with b and c null."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2435")
        self.graph.query("CREATE (:A), (:A)")

    def test01_null_bound_variable_fails_whole_pattern(self):
        q = ("MATCH (a) OPTIONAL MATCH (a)--(b) OPTIONAL MATCH (b),(c) "
             "RETURN count(*) AS n")
        self.env.assertEqual(_exte_scalar(self.graph, q), 2)

        q = ("MATCH (a) OPTIONAL MATCH (a)--(b) OPTIONAL MATCH (b),(c) "
             "RETURN b, c")
        self.env.assertEqual(self.graph.query(q).result_set,
                             [[None, None], [None, None]])


class testIssue2441():
    """https://github.com/FalkorDB/FalkorDB/issues/2441
    In a single MATCH path the same relationship must not be bound twice. With
    anonymous hops `(a)-[:R]-(b)-[:R]-(c)` the same edge is reused for both
    hops, returning impossible rows like (1,2,1). Only (1,2,3) and (3,2,1)
    may be returned on a 3-node chain."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2441")
        self.graph.query("CREATE (:N {n:1})-[:R]->(:N {n:2})-[:R]->(:N {n:3})")

    def test01_anonymous_relationships_are_unique(self):
        q = ("MATCH (a)-[:R]-(b)-[:R]-(c) RETURN a.n, b.n, c.n "
             "ORDER BY a.n, b.n, c.n")
        self.env.assertEqual(self.graph.query(q).result_set,
                             [[1, 2, 3], [3, 2, 1]])

    def test02_mixed_named_anonymous_relationships_are_unique(self):
        q = "MATCH (a)-[:R]-(b)-[r2:R]-(c) RETURN count(*)"
        self.env.assertEqual(_exte_scalar(self.graph, q), 2)
        q = "MATCH p=(a)-[:R]-(b)-[:R]-(c) RETURN count(p)"
        self.env.assertEqual(_exte_scalar(self.graph, q), 2)

    def test03_named_relationships_are_unique(self):
        q = "MATCH (a)-[r1:R]-(b)-[r2:R]-(c) RETURN count(*)"
        self.env.assertEqual(_exte_scalar(self.graph, q), 2)


class testIssue2522():
    """https://github.com/FalkorDB/FalkorDB/issues/2522
    Variables imported with `CALL { WITH s, p ... UNION ALL ... }` are visible
    only in the first branch; that matches Neo4j's branch-local import, so the
    actionable part is the scoped `CALL (s, p) { ... UNION ALL ... }` form,
    which must make the imports visible in every branch (both engines reject
    the syntax today)."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2522")
        self.graph.query("CREATE (:l1 {n:'s'})-[:l2]->(:l1 {n:'p'})")

    def test01_scoped_call_imports_into_every_union_branch(self):
        q = ("MATCH (s:l1 {n:'s'}), (p:l1 {n:'p'}) "
             "CALL (s, p) { RETURN s AS z UNION ALL RETURN p AS z } "
             "RETURN z.n ORDER BY z.n")
        self.env.assertEqual(self.graph.query(q).result_set, [['p'], ['s']])


class testIssue2573():
    """https://github.com/FalkorDB/FalkorDB/issues/2573
    A plain DELETE of a node that still has relationships silently deletes the
    relationships too (acts as DETACH DELETE). openCypher requires an error and
    no change to the graph. The Rust engine also under-reports the statistics
    of a CREATE ... DETACH DELETE in one query."""

    def __init__(self):
        self.env, self.db = Env()

    def test01_plain_delete_of_connected_node_errors(self):
        graph = self.db.select_graph("issue_2573_a")
        graph.query("CREATE (a:A {id: 1}), (b:B), (a)-[:R]->(b)")
        try:
            graph.query("MATCH (a:A) DELETE a")
            self.env.assertTrue(False, message="DELETE of a connected node must fail")
        except ResponseError:
            pass
        self.env.assertEqual(_exte_scalar(graph, "MATCH (n) RETURN count(n)"), 2)
        self.env.assertEqual(
            _exte_scalar(graph, "MATCH ()-[r]->() RETURN count(r)"), 1)

    def test02_same_query_create_detach_delete_stats(self):
        graph = self.db.select_graph("issue_2573_b")
        res = graph.query("CREATE (a), (b), (a)-[:R]->(b) DETACH DELETE a")
        self.env.assertEqual(res.nodes_created, 2)
        self.env.assertEqual(res.relationships_created, 1)
        self.env.assertEqual(res.nodes_deleted, 1)
        self.env.assertEqual(res.relationships_deleted, 1)


class testIssue2574():
    """https://github.com/FalkorDB/FalkorDB/issues/2574
    LIMIT/SKIP accept only an integer literal or a parameter; any other
    expression that does not reference row variables (1+1, CASE, toInteger)
    is rejected at parse time. It must be evaluated once and used."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2574")

    def test01_limit_skip_accept_constant_expressions(self):
        q = "UNWIND range(1, 5) AS x RETURN x LIMIT 1 + 1"
        self.env.assertEqual(self.graph.query(q).result_set, [[1], [2]])
        q = "UNWIND range(1, 5) AS x RETURN x LIMIT toInteger('2')"
        self.env.assertEqual(self.graph.query(q).result_set, [[1], [2]])
        q = "UNWIND range(1, 5) AS x RETURN x SKIP CASE WHEN true THEN 3 END"
        self.env.assertEqual(self.graph.query(q).result_set, [[4], [5]])


class testIssue2585():
    """https://github.com/FalkorDB/FalkorDB/issues/2585
    localtime()/localdatetime() drop millisecond/microsecond/nanosecond and the
    fraction of a parsed second, so 12:31:14.645 equals 12:31:14. The
    sub-second component must be kept, as the TCK expects."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2585")

    def test01_fractional_seconds_are_kept(self):
        cases = [
            ("toString(localtime({hour: 12, minute: 31, second: 14, millisecond: 645}))",
             "12:31:14.645"),
            ("toString(localdatetime({year: 1984, month: 10, day: 11, hour: 12, "
             "minute: 31, second: 14, microsecond: 645876}))",
             "1984-10-11T12:31:14.645876"),
            ("toString(localdatetime('2015-07-21T21:40:32.142'))",
             "2015-07-21T21:40:32.142"),
            ("localtime('12:31:14.645') = localtime('12:31:14')", False),
            ("localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, "
             "second: 14, millisecond: 645}).millisecond", 645),
        ]
        for expr, expected in cases:
            self.env.assertEqual(_exte_scalar(self.graph, f"RETURN {expr}"),
                                 expected, message=expr)


class testIssue2587():
    """https://github.com/FalkorDB/FalkorDB/issues/2587
    The ISO extended ordinal date form YYYY-DDD is misparsed: C reads the
    day-of-year as a month, Rust reads it as a month or errors. date('2015-202')
    must be 2015-07-21 (and the same for localdatetime)."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2587")

    def test01_ordinal_date_is_parsed(self):
        cases = [
            ("toString(date('2015-202'))", "2015-07-21"),
            ("toString(date('2015-060'))", "2015-03-01"),
            ("toString(date('2015-012'))", "2015-01-12"),
            ("toString(localdatetime('2015-202T21:40:32'))", "2015-07-21T21:40:32"),
        ]
        for expr, expected in cases:
            try:
                actual = _exte_scalar(self.graph, f"RETURN {expr}")
            except ResponseError as e:
                actual = f"error: {e}"
            self.env.assertEqual(actual, expected, message=expr)


class testIssue2595():
    """https://github.com/FalkorDB/FalkorDB/issues/2595
    duration() on ISO 8601 strings with a fractional non-second unit returns
    null (C), or truncates/errors (Rust). The fraction must carry into the
    smaller units: PT0.75M = PT45S, P5M1.5D = P5M1DT12H, P2.5W = P17DT12H."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2595")

    def test01_fractional_duration_components(self):
        cases = [
            ("PT0.75M", "PT45S"),
            ("PT0.5H", "PT30M"),
            ("P5M1.5D", "P5M1DT12H"),
            ("P2.5W", "P17DT12H"),
        ]
        for literal, expected in cases:
            try:
                actual = _exte_scalar(self.graph,
                                      f"RETURN toString(duration('{literal}'))")
            except ResponseError as e:
                actual = f"error: {e}"
            self.env.assertEqual(actual, expected, message=literal)


class testIssue2607():
    """https://github.com/FalkorDB/FalkorDB/issues/2607
    An upper-bounded variable-length pattern with an empty interval (`*..0`)
    raises a hops-range error. TCK Match5 [13] expects an empty result.
    (tests/flow/test_relation_patterns.py test12 asserts the error today.)"""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2607")
        self.graph.query("CREATE (:L {i: 1})-[:R]->(:L {i: 2})-[:R]->(:L {i: 3})")

    def test01_empty_upper_bounded_interval_returns_no_rows(self):
        q = "MATCH (a:L) MATCH (a)-[:R*..0]->(c) RETURN c.i"
        self.env.assertEqual(self.graph.query(q).result_set, [])
        q = "MATCH p=(a:L)-[r*..0]->(b:L) RETURN a.i AS k, b.i AS v"
        self.env.assertEqual(self.graph.query(q).result_set, [])


class testIssue2652():
    """https://github.com/FalkorDB/FalkorDB/issues/2652
    GRAPH.LIST ignores ACL key patterns: a user restricted to ~tenant_a still
    sees every graph name. GRAPH.LIST must only return graphs whose keys the
    calling user may access."""

    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.prefix = "issue_2652_"
        for name in ("tenant_a", "tenant_b", "secret_payroll"):
            self.db.select_graph(self.prefix + name).query("CREATE ()")

    def _user_conn(self, user):
        return redis.Redis(host=self.env.host, port=self.env.port,
                           username=user, password="pw2652",
                           decode_responses=True)

    def test01_graph_list_honours_key_acl(self):
        alice = "issue_2652_alice"
        bob = "issue_2652_bob"
        self.conn.execute_command("ACL", "SETUSER", alice, "reset", "on",
                                  ">pw2652", f"~{self.prefix}tenant_a", "+@all")
        self.conn.execute_command("ACL", "SETUSER", bob, "reset", "on",
                                  ">pw2652", "+@all")
        try:
            a = self._user_conn(alice)
            # control: key ACL is enforced for GRAPH.QUERY
            a.execute_command("GRAPH.QUERY", self.prefix + "tenant_a",
                              "MATCH (n) RETURN count(n)")
            try:
                a.execute_command("GRAPH.QUERY", self.prefix + "secret_payroll",
                                  "MATCH (n) RETURN count(n)")
                self.env.assertTrue(False, message="alice must not query secret_payroll")
            except ResponseError as e:
                self.env.assertContains("No permissions", str(e))

            listed = a.execute_command("GRAPH.LIST")
            self.env.assertContains(self.prefix + "tenant_a", listed)
            self.env.assertNotContains(self.prefix + "tenant_b", listed)
            self.env.assertNotContains(self.prefix + "secret_payroll", listed)

            listed = self._user_conn(bob).execute_command("GRAPH.LIST")
            self.env.assertEqual(
                [g for g in listed if g.startswith(self.prefix)], [])
        finally:
            self.conn.execute_command("ACL", "DELUSER", alice, bob)
        self.env.assertTrue(self.conn.ping())


class testIssue2659():
    """https://github.com/FalkorDB/FalkorDB/issues/2659
    A WITH with ORDER BY and SKIP followed by a RETURN with SKIP and LIMIT
    returns no rows: the top-N budget given to Sort ignores the outer SKIP.
    The rows must be 11..20 (and 13..17 for the second query)."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2659")

    def test01_stacked_skip_with_sort(self):
        q = ("UNWIND range(1, 20) AS x WITH x ORDER BY x SKIP 0 "
             "RETURN x SKIP 10 LIMIT 10")
        self.env.assertEqual(self.graph.query(q).result_set,
                             [[x] for x in range(11, 21)])
        q = ("UNWIND range(1, 20) AS x WITH x ORDER BY x SKIP 2 "
             "RETURN x SKIP 10 LIMIT 5")
        self.env.assertEqual(self.graph.query(q).result_set,
                             [[x] for x in range(13, 18)])


class testIssue2724():
    """https://github.com/FalkorDB/FalkorDB/issues/2724
    The Rust engine rejects the documented prev() function with "Unknown
    function 'prev'"; C supports it. prev(x) must return the previous row's
    value (null on the first row), also inside a variable-length filter."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2724")
        self.graph.query("CREATE (:A)-[:R {k0: 1}]->(:A)-[:R {k0: 2}]->(:A)")

    def test01_prev_over_rows(self):
        q = "UNWIND range(1, 5) AS x RETURN prev(x)"
        self.env.assertEqual(self.graph.query(q).result_set,
                             [[None], [1], [2], [3], [4]])

    def test02_prev_in_var_len_filter(self):
        q = ("MATCH p=(:A)-[r:R*1..2]->(:A) "
             "WHERE coalesce(prev(r.k0), r.k0) <= r.k0 "
             "RETURN length(p) ORDER BY length(p)")
        self.env.assertEqual(self.graph.query(q).result_set, [[1], [1], [2]])


def _extf_alive(env):
    env.assertTrue(env.getConnection().ping())


class testIssue2725():
    """https://github.com/FalkorDB/FalkorDB/issues/2725
    A MATCH that names a single-node path over a variable bound before a
    projection (CALL algo.BFS, WITH) and filters it fails with "Variable n0 not
    found", and without the filter it drops the row. The variable must stay
    bound and the path must match its node."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2725")
        self.graph.query("CREATE (:A {k11: 1, klist: []})-[:rt0]->(:B)")

    def test01_issue_query_after_bfs(self):
        # the WHERE is null for every row (n0.k9 is missing), so C returns no rows
        q = """MATCH (n0 {k11: 1})
               WHERE size(n0.klist) >= 0
               CALL algo.BFS(n0, 2, 'rt0') YIELD nodes AS alias0
               UNWIND alias0 AS alias1
               MATCH (n0), p0 = (n0), (n0)
               WHERE (((n0.k9 = false) OR true)
                      XOR ((n0.k9 = false) XOR (n0.k9 = false)))
               RETURN *"""
        self.env.assertEqual(self.graph.query(q).result_set, [])

    def test02_bfs_then_path_over_bound_node(self):
        q = """MATCH (n0 {k11: 1})
               CALL algo.BFS(n0, 2, 'rt0') YIELD nodes AS alias0
               UNWIND alias0 AS alias1
               MATCH (n0), p0 = (n0), (n0)
               WHERE n0.k11 = 1
               RETURN n0.k11, labels(alias1), length(p0)"""
        self.env.assertEqual(self.graph.query(q).result_set, [[1, ['B'], 0]])

    def test03_with_then_path_over_bound_node(self):
        q = "MATCH (n0:A) WITH n0 MATCH p0 = (n0) WHERE n0.k11 = 1 RETURN count(*)"
        self.env.assertEqual(self.graph.query(q).result_set, [[1]])
        q = "MATCH (n0:A) WITH n0 MATCH p0 = (n0) RETURN count(*)"
        self.env.assertEqual(self.graph.query(q).result_set, [[1]])


class testIssue2727():
    """https://github.com/FalkorDB/FalkorDB/issues/2727
    After CALL algo.BFS, a WITH that projects `n0 AS alias4` (with ORDER BY and
    WHERE) followed by `MATCH (alias4), p1 = (alias4)` fails with "Variable
    alias4 not found". The alias must stay bound; C returns the row."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2727")
        self.graph.query("CREATE (:A {k8: -956555468, k2: '5', k0: 'x'})-[:rt0]->(:B)")

    def test01_issue_query(self):
        # the OPTIONAL MATCH filter compares null to null, so n0 is null and
        # the query returns no rows (as C does), but it must not fail
        q = """OPTIONAL MATCH p0 = (n0 {k8: -956555468, k2: '5'})
               WHERE toStringOrNull(n0.k3) = toStringOrNull(n0.k3)
               WITH * WHERE n0 IS NOT NULL
               CALL algo.BFS(n0, 2, 'rt0') YIELD nodes AS alias0
               WITH alias0, n0.k2 AS alias1,
                    pow(toFloatOrNull(reduce(acc = 0, n IN labels(n0) | acc + 1)), 2) AS alias2,
                    toFloatOrNull(reduce(acc = 0, n IN labels(n0) | acc + 1)) AS alias3,
                    n0 AS alias4, p0
               ORDER BY alias1, alias2, alias3
               WHERE (true XOR false)
               MATCH (alias4), p1 = (alias4)
               RETURN alias4, p1, alias3, alias0, alias2, alias1,
                      toStringOrNull(alias4.k0) AS alias5"""
        self.env.assertEqual(self.graph.query(q).result_set, [])

    def test02_alias_reaches_match_with_rows(self):
        q = """OPTIONAL MATCH p0 = (n0 {k8: -956555468, k2: '5'})
               WITH * WHERE n0 IS NOT NULL
               CALL algo.BFS(n0, 2, 'rt0') YIELD nodes AS alias0
               WITH alias0, n0.k2 AS alias1, n0 AS alias4, p0
               ORDER BY alias1
               WHERE (true XOR false)
               MATCH (alias4), p1 = (alias4)
               RETURN toStringOrNull(alias4.k0), length(p1), size(alias0), alias1"""
        self.env.assertEqual(self.graph.query(q).result_set, [['x', 0, 1, '5']])


class testIssue2728():
    """https://github.com/FalkorDB/FalkorDB/issues/2728
    An OPTIONAL MATCH with a WHERE inside a CALL {} body that follows outer
    clauses fails with "Variable n1 not found" (the filter is pushed below the
    scan of its own variable). The query must run, as C does."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2728")

    def test01_issue_query(self):
        q = """UNWIND [1] AS alias0
               CALL { WITH * RETURN alias0 AS alias1, range(0, 4) AS alias2,
                      range(9, 16, 1) AS alias3 SKIP 0 }
               CREATE ({id: 135})
               WITH alias0, alias3, alias2, alias1 AS alias4,
                    toBooleanOrNull(range(23, 9, -3)) AS alias5,
                    localTime('12:00:00') AS alias6
               WHERE true
               CALL {
                   WITH alias3, alias2, alias6
                   OPTIONAL MATCH (n0)<-[r0:rt11]-(n1 {k6: '9'}), (n0),
                                  p0 = (n0)<-[:rt3]-()
                   WHERE size(r0.k9) >= 0
                   RETURN n1 AS n2, r0 AS r1, p0 AS p1, alias6 AS alias7,
                          alias3 AS alias8, alias2 AS alias9,
                          list.insertListElements(labels(n1), [0], 0, false) AS alias10
               }
               CALL db.relationshipTypes() YIELD relationshipType AS alias11
               RETURN n2"""
        # the graph has no relationship types, so db.relationshipTypes() yields nothing
        res = self.graph.query(q)
        self.env.assertEqual(res.result_set, [])
        self.env.assertEqual(res.nodes_created, 1)

    def test02_smaller_shape(self):
        self.graph.query("UNWIND range(1,3) AS i CREATE (:A:B {id: i})")
        q = """MATCH (x:A:B)
               CALL { OPTIONAL MATCH (n:A:B {id:3}) RETURN n.id AS nid }
               RETURN x.id, nid ORDER BY x.id"""
        self.env.assertEqual(self.graph.query(q).result_set,
                             [[1, 3], [2, 3], [3, 3]])


class testIssue2729():
    """https://github.com/FalkorDB/FalkorDB/issues/2729
    Wrapping a query that matches paths, OPTIONAL MATCHes a node, creates a
    self-loop on it and DETACH DELETEs it in CALL {} fails with "Failed to
    create relationship; endpoint was not found". It must return the same 25
    rows as the unwrapped query, as C does."""

    BODY = """MATCH p0=()-[*0..3]->({id:87})
              OPTIONAL MATCH p1=(n0)-[r0:rt5 {k4:798157397,k2:"E"}]->(n1 {k4:193904108,k1:"NX0"}),(n1),(n0)
              WHERE toStringOrNull(n1.k11) CONTAINS toStringOrNull(n1.k11)
              WITH * WHERE r0 IS NOT NULL
              SET r0.k3=848731825
              WITH * WHERE n0 IS NOT NULL
              CREATE (n0)-[:rt8]->(n0)
              DETACH DELETE n0
              RETURN {p0:p0,n0:n0,n1:n1,r0:r0,p1:p1} AS __layer_row"""

    def __init__(self):
        self.env, self.db = Env()

    def _populate(self, g):
        g.query("""UNWIND [0,1,2,3,6,19,22,26,27,30,43,51,53,55,64,69,82,87,91,96,101,104,126] AS i
                   CREATE (:A {id:i})""")
        g.query("UNWIND range(1,11) AS i MATCH (n:A {id:1}) CREATE (n)-[:D]->(n)")
        g.query('MATCH (n1:A {id:26}) SET n1.k4=193904108, n1.k1="NX0", n1.k11="x"')
        g.query('MATCH (n0:A {id:69}), (n1:A {id:26}) CREATE (n0)-[:rt5 {k4:798157397, k2:"E"}]->(n1)')
        g.query("""UNWIND [[6,87],[43,6],[30,6],[64,6],[91,6],[51,6],[126,6],[6,126],[126,126],
                           [19,51],[19,51],[87,51],[104,51],[55,51],[53,91],[0,64],[101,64],[19,30],
                           [3,30],[22,30],[96,30],[27,30],[2,30],[82,43]] AS e
                   MATCH (a:A {id:e[0]}), (b:A {id:e[1]}) CREATE (a)-[:P]->(b)""")

    def test01_unwrapped_query(self):
        g = self.db.select_graph("issue_2729_left")
        self._populate(g)
        res = g.query(self.BODY)
        self.env.assertEqual(len(res.result_set), 25)
        self.env.assertEqual(g.query("MATCH (n) RETURN count(n)").result_set, [[22]])

    def test02_wrapped_in_call(self):
        g = self.db.select_graph("issue_2729_right")
        self._populate(g)
        res = g.query("CALL { " + self.BODY + " } RETURN __layer_row")
        self.env.assertEqual(len(res.result_set), 25)
        self.env.assertEqual(g.query("MATCH (n) RETURN count(n)").result_set, [[22]])


class testIssue2733():
    """https://github.com/FalkorDB/FalkorDB/issues/2733
    `RETURN 1 LIMIT 0` was reported to return one row. LIMIT 0 must return
    no rows (the result header is still sent)."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2733")

    def test01_limit_zero_returns_no_rows(self):
        res = self.graph.query("RETURN 1 LIMIT 0")
        self.env.assertEqual(res.result_set, [])
        self.env.assertEqual(self.graph.query("UNWIND [1, 2] AS x RETURN x LIMIT 0").result_set, [])
        self.env.assertEqual(self.graph.query("WITH 1 AS x LIMIT 0 RETURN x").result_set, [])


class testIssue2737():
    """https://github.com/FalkorDB/FalkorDB/issues/2737
    An OPTIONAL MATCH that binds a single-node path over an already bound node
    and has a WHERE: returning the path inside a map fails with "Variable p6
    not found", and otherwise (or after `WITH *`) the path comes back null.
    The path must be bound (a zero-length path), as in C."""

    def __init__(self):
        self.env, self.db = Env()

    def test01_path_after_procedure_call(self):
        g = self.db.select_graph("issue_2737_a")
        g.query("""CREATE (n1:X {k8: []})-[:R]->(n0 {id: 39, klist: []})
                          -[:rt7 {k0: true, k1: -39985978, k6: 1}]->(n2:l8:l1:l10 {k10: 2})""")
        prefix = """MATCH p0 = (n1 {k8: []})-[r0]->(n0 {id: 39})
                          -[r1:rt7 {k0: true, k1: -39985978}]->(n2:l8:l1:l10)
                    WHERE all(item IN n0.klist WHERE item IS NOT NULL)
                    CALL db.propertyKeys() YIELD propertyKey AS alias0
                    OPTIONAL MATCH p6 = (n1)
                    WHERE NOT (toFloatOrNull(r1.k6) = toFloatOrNull(n2.k10)) """
        # the issue's own RETURN (a map holding the path) fails on LEFT
        ret = ("RETURN {n1:n1, n0:n0, n2:n2, r0:r0, r1:r1, p0:p0, alias0:alias0, p6:p6} AS row")
        left = g.query(prefix + ret).result_set
        right = g.query(prefix + "WITH * " + ret).result_set
        self.env.assertEqual(len(left), 7)
        self.env.assertEqual(len(right), 7)
        for row in left + right:
            self.env.assertNotEqual(row[0]['p6'], None)
        # the path is bound to n1, a zero-length path
        ret = "RETURN alias0, length(p6) ORDER BY alias0"
        for res in (g.query(prefix + ret).result_set,
                    g.query(prefix + "WITH * " + ret).result_set):
            self.env.assertEqual(len(res), 7)
            self.env.assertEqual([r[1] for r in res], [0] * 7)

    def test02_path_after_projection(self):
        # the issue's variant B starts with LOAD CSV; UNWIND triggers the same bug
        g = self.db.select_graph("issue_2737_b")
        prefix = """UNWIND ['one', 'three'] AS alias0
                    CREATE (n0:l10:l2 {k3:false, k8:true, id:135})
                    WITH alias0, n0
                    OPTIONAL MATCH p0 = (n0), (n0) """
        where = "WHERE ((n0.k8 = true) OR (n0.k3 = false)) "
        ret = "RETURN {alias0:alias0, n0:n0, p0:p0} AS row"
        for q in (prefix + where + ret, prefix + "WITH * " + where + ret):
            res = g.query(q).result_set
            self.env.assertEqual(len(res), 2)
            for row in res:
                self.env.assertNotEqual(row[0]['p0'], None)
        ret = "RETURN alias0, length(p0) ORDER BY alias0"
        for q in (prefix + where + ret, prefix + "WITH * " + where + ret):
            self.env.assertEqual(g.query(q).result_set, [['one', 0], ['three', 0]])


class testIssue2738():
    """https://github.com/FalkorDB/FalkorDB/issues/2738
    A CALL {} body that matches a pattern and creates relationships which the
    same pattern matches, run once per outer row, does not see the writes of
    earlier rows unless identity `CALL { RETURN 0 }` barriers are added. Both
    forms must create the same 96 relationships, as C does. (The issue's LOAD
    CSV rows are replaced by UNWINDs of the same size.)"""

    LEFT = """UNWIND range(1, 3) AS alias3
              CALL {
                MATCH p0 = (n5 {k4:true})<-[*1]-(n6:l4:l6:l0:l8:l11)
                UNWIND range(1, 2) AS alias4
                FOREACH (elem IN nodes(p0) | SET elem.k2=[])
                WITH n5.id AS alias47, n6.k2[0] AS alias48, n5, n6 AS alias50, p0
                WHERE ((n5.k9=true) AND (n5.k9=true))
                CREATE (n5)-[:rt11 {id:615}]->(n5)
                RETURN {alias47:alias47, alias48:alias48, n5:n5, alias50:alias50, p0:p0} AS row
              }
              RETURN {alias3:alias3, row:row} AS row"""

    RIGHT = """UNWIND range(1, 3) AS alias3
               WITH alias3
               CALL { RETURN 0 AS b }
               WITH alias3
               CALL {
                 MATCH p0 = (n5 {k4:true})<-[*1]-(n6:l4:l6:l0:l8:l11)
                 UNWIND range(1, 2) AS alias4
                 FOREACH (elem IN nodes(p0) | SET elem.k2=[])
                 WITH n5.id AS alias47, n6.k2[0] AS alias48, n5, n6 AS alias50, p0
                 WHERE ((n5.k9=true) AND (n5.k9=true))
                 CREATE (n5)-[:rt11 {id:615}]->(n5)
                 WITH alias47, alias48, n5, alias50, p0
                 CALL { RETURN 0 AS c }
                 WITH alias47, alias48, n5, alias50, p0
                 RETURN {alias47:alias47, alias48:alias48, n5:n5, alias50:alias50, p0:p0} AS row
               }
               RETURN {alias3:alias3, row:row} AS row"""

    def __init__(self):
        self.env, self.db = Env()

    def _populate(self, g):
        g.query("""CREATE (:N {id:29, k4:true, k9:true, k2:[]}),
                          (:l4:l6:l0:l8:l11 {id:33, k4:true, k9:true, k2:[]}),
                          (:N {id:39, k4:true, k9:true, k2:[]}),
                          (:l4:l6:l0:l8:l11 {id:41, k4:true, k9:true, k2:[]}),
                          (:l4:l6:l0:l8:l11 {id:71, k4:true, k9:true, k2:[]}),
                          (:l4:l6:l0:l8:l11 {id:96, k4:false, k9:false, k2:[]}),
                          (:l4:l6:l0:l8:l11 {id:90, k4:false, k9:false, k2:[]})""")
        g.query("""UNWIND [[33,29],[90,29],[41,39],[71,41],[96,71],[33,33]] AS e
                   MATCH (a {id:e[0]}), (b {id:e[1]}) CREATE (a)-[:R]->(b)""")

    def test01_identity_barriers_do_not_change_writes(self):
        gl = self.db.select_graph("issue_2738_left")
        self._populate(gl)
        left = gl.query(self.LEFT)
        gr = self.db.select_graph("issue_2738_right")
        self._populate(gr)
        right = gr.query(self.RIGHT)
        self.env.assertEqual(len(right.result_set), 96)
        self.env.assertEqual(right.relationships_created, 96)
        self.env.assertEqual(len(left.result_set), len(right.result_set))
        self.env.assertEqual(left.relationships_created, right.relationships_created)


class testIssue2742():
    """https://github.com/FalkorDB/FalkorDB/issues/2742
    An OPTIONAL MATCH of anonymous nodes with a WHERE inside a CALL {} (here a
    UNION branch) after an outer MATCH fails with "Variable _anon_0 not found".
    It must run, as C does, and as it does with an extra nested CALL."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2742")
        self.graph.query("CREATE (n0:L {k0: true})")

    def test01_issue_query(self):
        q = """MATCH (n0:L)
               CALL { WITH * RETURN n0 AS n1 }
               CALL {
                 WITH n0, n1
                 RETURN n0 AS n2
                 UNION
                 OPTIONAL MATCH ({k0: true}), ()
                 WHERE true
                 RETURN null AS n2
               }
               RETURN n0.k0, n1.k0, n2.k0 ORDER BY n2.k0"""
        self.env.assertEqual(self.graph.query(q).result_set,
                             [[True, True, True], [True, True, None]])

    def test02_smaller_shape(self):
        q = """MATCH (n0:L)
               CALL { OPTIONAL MATCH ({k0: true}) WHERE true RETURN null AS n2 }
               RETURN n0.k0, n2"""
        self.env.assertEqual(self.graph.query(q).result_set, [[True, None]])


class testIssue2744():
    """https://github.com/FalkorDB/FalkorDB/issues/2744
    After a WITH that carries n0, `MATCH p0=(n0), p1=(n0), (n0) WHERE ...`
    followed by LOAD CSV fails with "Variable n0 not found". The binding must be
    kept and the rows returned, as C does. LOAD CSV is not needed for the bug,
    so an UNWIND of the same three CSV rows stands in for it."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2744")
        self.graph.query("CREATE (n0:A {k8: 1, k5: 1, k10: 'x', k2: true})")

    def test01_binding_survives(self):
        q = """MATCH (n0:A)
               WITH n0.k8 AS alias0,
                    radians(toFloatOrNull(n0.k5)) AS alias1,
                    tan(toFloatOrNull(n0.k8)) AS alias2,
                    n0
               WHERE n0.k10 ENDS WITH ''
               MATCH p0=(n0), p1=(n0), (n0)
               WHERE n0.k10 STARTS WITH ''
               UNWIND [['alpha', 'beta'], ['one', 'two'], ['three', 'four']] AS alias3
               RETURN n0.k10, length(p1), length(p0), alias0, alias3, n0.k2 AS alias4
               ORDER BY alias1 DESC, alias0 DESC, alias4 DESC, alias3
               SKIP 0"""
        self.env.assertEqual(self.graph.query(q).result_set, [
            ['x', 0, 0, 1, ['alpha', 'beta'], True],
            ['x', 0, 0, 1, ['one', 'two'], True],
            ['x', 0, 0, 1, ['three', 'four'], True]])


class testIssue2745():
    """https://github.com/FalkorDB/FalkorDB/issues/2745
    A negative duration of one day (duration({days: 2}) + duration({days: -3}),
    or duration({days: -1})) is normalized and rendered as P-1Y11M30D. It must
    stay -1 day: P-1D."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2745")

    def test01_negative_day_duration(self):
        q = """RETURN toString(duration({days: 2}) + duration({days: -3})),
                      toString(duration({days: -1})),
                      duration({days: 2}) + duration({days: -3}) = duration({days: -1})"""
        self.env.assertEqual(self.graph.query(q).result_set, [['P-1D', 'P-1D', True]])


class testIssue2746():
    """https://github.com/FalkorDB/FalkorDB/issues/2746
    Adding a huge number of days to a date (1,000,000,000 days) gives a wrong
    date (9933-01-03) whose components are null, and dates past year 9999 print
    only four year digits. The result must be 2739933-01-03, as in C."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2746")

    def test01_large_day_count(self):
        q = """WITH date('2026-01-01') + duration({days: 1000000000}) AS d
               RETURN toString(d), d.year, d.month, d.day"""
        self.env.assertEqual(self.graph.query(q).result_set,
                             [['2739933-01-03', 2739933, 1, 3]])

    def test02_five_digit_year(self):
        q = """WITH date('2026-01-01') + duration({days: 3000000}) AS d
               RETURN toString(d), d.year"""
        self.env.assertEqual(self.graph.query(q).result_set, [['10239-09-21', 10239]])


class testIssue2759():
    """https://github.com/FalkorDB/FalkorDB/issues/2759
    Creating and deleting a node, then creating a new node in the same query
    (which reuses the freed id) returns the new node without its labels and
    properties. It must return label V, {id: 1, k1: 2} and k = 2."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2759")

    def test01_new_node_after_delete(self):
        q = """CREATE (__d) DELETE __d WITH 1 AS dummy
               CREATE (n:V {id: 1, k1: 2})
               RETURN n.k1 AS k, labels(n) AS labels, properties(n) AS props"""
        self.env.assertEqual(self.graph.query(q).result_set,
                             [[2, ['V'], {'id': 1, 'k1': 2}]])
        q = "MATCH (n:V) RETURN n.k1, labels(n), properties(n)"
        self.env.assertEqual(self.graph.query(q).result_set,
                             [[2, ['V'], {'id': 1, 'k1': 2}]])


class testIssue3117():
    """https://github.com/FalkorDB/FalkorDB/issues/3117
    duration() accepts a string whose last number has no unit ('P5', 'P1Y5',
    'PT4H5') and silently drops that number. Such a string must be rejected:
    C returns null, Neo4j raises an error; either is accepted here."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_3117")

    def test01_trailing_number_without_unit_is_rejected(self):
        for s in ['P5', 'P-', 'P1Y5', 'P1M2', 'PT4H5']:
            try:
                res = self.graph.query(f"RETURN toString(duration('{s}'))").result_set
                self.env.assertEqual(res, [[None]], message=f"duration('{s}')")
            except ResponseError:
                pass
        _extf_alive(self.env)


def _ext_g_raw_conn(env):
    """A client that does not decode replies, for byte-level payloads."""
    return redis.Redis(host=env.host, port=env.port, decode_responses=False)


def _ext_g_plan(conn, key, q):
    return "\n".join(conn.execute_command("GRAPH.EXPLAIN", key, q))


class testIssue3121():
    """https://github.com/FalkorDB/FalkorDB/issues/3121
    algo.SPpaths with a configured costProp that an edge on the path lacks:
    that edge must cost 1 (as in C), so A-[cost:2]->B-[no cost]->C has
    pathCost 3. Rust's default single-path result charges 0 (pathCost 2)."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("ext_g_3121")
        self.graph.query("CREATE (:L {id:'A'})-[:E {weight:1, cost:2}]->"
                         "(:L {id:'B'})-[:E2 {weight:1}]->(:L {id:'C'})")

    def test01_missing_cost_property_costs_one(self):
        for extra in ["", ", pathCount: 1", ", pathCount: 0", ", maxCost: 3"]:
            q = ("MATCH (s:L {id:'A'}), (t:L {id:'C'}) "
                 "CALL algo.SPpaths({sourceNode: s, targetNode: t, "
                 f"weightProp: 'weight', costProp: 'cost'{extra}}}) "
                 "YIELD pathWeight, pathCost RETURN pathWeight, pathCost")
            res = self.graph.query(q).result_set
            self.env.assertEqual(res, [[2, 3]], message=f"options: {extra!r}")


class testIssue3164():
    """https://github.com/FalkorDB/FalkorDB/issues/3164
    A CREATE in the second branch of a `CALL { ... UNION ... }` whose first
    branch imports a variable creates its node with the wrong label (the
    imported node's label) and never adds the new label. The node must carry
    the label the CREATE names, as in C."""

    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph("ext_g_3164")
        self.graph2 = self.db.select_graph("ext_g_3164_orig")

    def test01_create_in_second_union_branch_keeps_its_label(self):
        self.graph.query("CREATE (:N {i:1})")
        res = self.graph.query("MATCH (n:N) CALL { WITH n RETURN 1 AS x UNION "
                               "CREATE (m:M {v:5}) RETURN m.v AS x } "
                               "RETURN x ORDER BY x")
        self.env.assertEqual(res.result_set, [[1], [5]])
        self.env.assertEqual(res.labels_added, 1)
        self.env.assertEqual(self.graph.query("MATCH (m:M) RETURN m.v").result_set, [[5]])
        self.env.assertEqual(self.graph.query("MATCH (n:N) RETURN count(n)").result_set, [[1]])

    def test02_issue_query_creates_correctly_labelled_nodes(self):
        g = self.graph2
        g.query("CREATE (s:S {id:93}), (v:V {id:61,k9:true}), (n0:N {id:16,k7:'jUKnRFZAA'}), "
                "(n1:N {id:120,k6:'c'}), (n2:N {id:40}), (u:U {id:94,k5:false,klist:['Glgbw']}), "
                "(m:l2:l7:l3:l5:l1:l9 {id:96})")
        g.query("MATCH (s {id:93}), (v {id:61}), (n0 {id:16}), (n1 {id:120}), (n2 {id:40}), "
                "(u {id:94}), (m {id:96}) "
                "CREATE (s)-[:rt7 {id:3,k8:'GI'}]->(v), (n0)-[:rt0 {id:72,k8:'XL'}]->(v), "
                "(n0)-[:rt9 {id:19,k8:'a'}]->(n1), (n2)-[:R {id:50}]->(u), "
                "(n2)-[:R3 {id:160,klist:[]}]->(n2), (n2)-[:R {id:51}]->(n0), "
                "(m)-[:rt10 {k2:'wbXu',k6:'c'}]->(n1)")
        g.query("""MATCH p0=({id:93})-[r0:rt7 {id:3}]->({k9:true})<-[r1:rt0 {id:72}]-
                         (n0 {k7:'jUKnRFZAA'})-[r2:rt9]->(n1),
                         ({k5:false,klist:['Glgbw']})<-[]-(n2)<-[r3]-(n2)-[]->(n0),(n1)
                   CALL db.relationshipTypes() YIELD relationshipType AS alias0
                   CALL algo.BFS(n0,2,'rt0') YIELD nodes AS alias1
                   UNWIND CASE WHEN r3.klist IS NULL THEN [] ELSE [r3.klist] END AS alias2
                   WITH *
                   MATCH (:l2:l7:l3:l5:l1:l9)-[:rt10 {k2:'wbXu',k6:'c'}]->(n1)
                   WHERE NOT (((true XOR true) OR (toStringOrNull(n1.k6) <= toStringOrNull(r2.k8)))
                              OR (toStringOrNull(r0.k8) CONTAINS 'hD'))
                   CALL {
                     WITH n0,n1,n2,alias1
                     DETACH DELETE n2,n0
                     RETURN n1 AS n3, alias1 AS alias3, toFloatOrNull(n1.k3) AS alias4,
                            toIntegerOrNull(n1.k5) AS alias5, right(toStringOrNull(n1.k2),1) AS alias6
                     UNION
                     FOREACH (elem IN [true,false] | CREATE (:l0 {k11:elem}))
                     MERGE (:l6 {id:130})-[:rt8 {id:514}]->(:l8 {id:131})
                     CREATE (:l6 {id:132})
                     RETURN null AS n3, null AS alias3, null AS alias4, null AS alias5, null AS alias6
                   }
                   RETURN count(*)""")
        res = g.query("MATCH (n) WHERE n.k11 IS NOT NULL RETURN DISTINCT labels(n)")
        self.env.assertEqual(res.result_set, [[['l0']]])
        res = g.query("MATCH (n {id:132}) RETURN DISTINCT labels(n)")
        self.env.assertEqual(res.result_set, [[['l6']]])
        # MERGE finds its own path after the first input row: exactly one path
        res = g.query("MATCH (:l6 {id:130})-[:rt8 {id:514}]->(:l8 {id:131}) RETURN count(*)")
        self.env.assertEqual(res.result_set, [[1]])
        self.env.assertTrue(self.conn.ping())


class testIssue3166():
    """https://github.com/FalkorDB/FalkorDB/issues/3166
    CREATE FULLTEXT INDEX ... OPTIONS {stopwords} on a label that already has a
    range index reports success but the stopwords are not applied, so a query
    for a stopword still matches. The stopwords must apply (as in C)."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("ext_g_3166")

    def test01_stopwords_apply_when_range_index_exists(self):
        self.graph.query("CREATE INDEX FOR (n:L) ON (n.uuid)")
        self.graph.query("CREATE FULLTEXT INDEX FOR (n:L) ON (n.t) OPTIONS {stopwords: ['router']}")
        wait_for_indices_to_sync(self.graph)
        self.graph.query("CREATE (:L {uuid:'u1', t:'router restarted'})")
        res = self.graph.query("CALL db.idx.fulltext.queryNodes('L','restarted') "
                               "YIELD node RETURN count(node)")
        self.env.assertEqual(res.result_set, [[1]])
        res = self.graph.query("CALL db.idx.fulltext.queryNodes('L','router') "
                               "YIELD node RETURN count(node)")
        self.env.assertEqual(res.result_set, [[0]])


class testIssue3185():
    """https://github.com/FalkorDB/FalkorDB/issues/3185
    (1) The C-style `db.idx.fulltext.createNodeIndex(label, prop, ...)` must
    create the index (Rust rejects >1 argument, and a 1-argument call over
    GRAPH.QUERY is refused as a GRAPH.RO_QUERY write). (2) A mangled query
    (non-UTF-8 + NUL inside the errCtx span) must be an error, not a panic."""

    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph("ext_g_3185")

    def test01_create_node_index_procedure(self):
        self.graph.query("CREATE (:Person {name:'alice', title:'engineer', org:'acme'})")
        res = self.graph.query(
            "CALL db.idx.fulltext.createNodeIndex('Person','name','title','org')")
        self.env.assertEqual(res.indices_created, 3)
        wait_for_indices_to_sync(self.graph)
        res = self.graph.query("CALL db.idx.fulltext.queryNodes('Person','engineer') "
                               "YIELD node RETURN node.name")
        self.env.assertEqual(res.result_set, [['alice']])
        # a write procedure run with GRAPH.QUERY must never be called read-only
        try:
            self.graph.query("CALL db.idx.fulltext.createNodeIndex('Doc')")
        except ResponseError as e:
            self.env.assertNotContains("RO_QUERY", str(e))

    def test02_mangled_query_is_an_error_not_a_crash(self):
        raw = _ext_g_raw_conn(self.env)
        queries = [
            # the bytes from the issue's panic log
            b"MATCH (a:Person {name:'\xd1\xea'}),(b:Company {name:'Y\x00;b'}) "
            b"CREATE (a)-[:\xe5\\\x8e]->(b)",
            b"MATCH (a:Person {name:'\xd1\xea'}) CREATE (a)-[:\xe5\\\x8e]->(a)",
            b"CREATE (a)-[:\xe5\x8e]->(b)",
            # error messages that embed a string holding a NUL
            b"RETURN date('a\\u0000b')",
            b"RETURN duration('a\\u0000b')",
        ]
        for q in queries:
            try:
                raw.execute_command("GRAPH.QUERY", "ext_g_3185_raw", q)
            except ResponseError:
                pass
            self.env.assertTrue(self.env.getConnection().ping(), message=repr(q))


class testIssue3187():
    """https://github.com/FalkorDB/FalkorDB/issues/3187
    Wrapping an `id(n) = x` predicate in parentheses (or ANDing it with a
    property predicate on n) makes the planner fall back to All Node Scan +
    Filter. The plan must use NodeByIdSeek, as for the bare predicate (and C)."""

    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.key = "ext_g_3187"
        self.graph = self.db.select_graph(self.key)
        self.graph.query("UNWIND range(1, 100) AS i CREATE (:N {i: i})")

    def _check(self, q, expected):
        plan = _ext_g_plan(self.conn, self.key, q)
        self.env.assertContains("NodeByIdSeek", plan, message=q)
        self.env.assertNotContains("All Node Scan", plan, message=q)
        self.env.assertEqual(self.graph.query(q).result_set, expected, message=q)

    def test01_parenthesized_id_predicate_uses_seek(self):
        self._check("MATCH (n) WHERE (id(n) = 42) RETURN n.i", [[43]])
        self._check("MATCH (n) WHERE ((id(n) = 42)) RETURN n.i", [[43]])
        self._check("MATCH (n) WHERE (42 = id(n)) RETURN n.i", [[43]])
        self._check("MATCH (a), (b) WHERE ((id(a) = 1) AND (id(b) = 2)) RETURN a.i, b.i", [[2, 3]])
        self._check("MATCH (a) WHERE (id(a) = 1) MATCH (b) WHERE (id(b) = 2) RETURN a.i, b.i", [[2, 3]])

    def test02_id_predicate_and_property_predicate_uses_seek(self):
        self._check("MATCH (n) WHERE id(n) = 42 AND n.i > 0 RETURN n.i", [[43]])
        self._check("MATCH (n) WHERE n.i > 0 AND id(n) = 42 RETURN n.i", [[43]])


class testIssue3194():
    """https://github.com/FalkorDB/FalkorDB/issues/3194
    With 5 parallel edges between two nodes, `MATCH (a)<-[e:R0]-(c)` whose edge
    variable is not otherwise referenced returns 1 row. openCypher returns one
    row per matched relationship (5), the same as `[:R0*1..1]`."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("ext_g_3194")
        self.graph.query("CREATE (n30:V {id: 30}), (n31:V {id: 31}), "
                         "(n30)-[:R0]->(n31), (n30)-[:R0]->(n31), (n30)-[:R0]->(n31), "
                         "(n30)-[:R0]->(n31), (n30)-[:R0]->(n31)")

    def test01_parallel_edges_keep_multiplicity(self):
        res = self.graph.query("MATCH p = (a:V)<-[:R0*1..1]-(c:V) RETURN a.id, c.id")
        self.env.assertEqual(res.result_set, [[31, 30]] * 5)
        res = self.graph.query("MATCH (a:V)<-[e:R0]-(c:V) RETURN a.id, c.id")
        self.env.assertEqual(res.result_set, [[31, 30]] * 5)
        res = self.graph.query("MATCH (a:V)-[:R0]->(c:V) RETURN count(*)")
        self.env.assertEqual(res.result_set, [[5]])


class testIssue3195():
    """https://github.com/FalkorDB/FalkorDB/issues/3195
    A two-hop pattern over 2 parallel first-hop edges and 1 second-hop edge
    must yield 2 paths (as `[:KNOWS*2]` does); the explicit
    `(a)-[:KNOWS]->(m)-[:KNOWS]->(c)` form collapses them to 1."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("ext_g_3195")
        self.graph.query("CREATE (n31:V {id: 31, score: 9886}), (n313:V {id: 313, score: 6537}), "
                         "(n223:V {id: 223, score: 6810}), (n31)-[:KNOWS {w: 5328}]->(n313), "
                         "(n313)-[:KNOWS {w: 395}]->(n223), (n31)-[:KNOWS {w: 8819}]->(n313)")

    def test01_multi_hop_keeps_multiplicity(self):
        res = self.graph.query("MATCH p = (a:V)-[:KNOWS*2]->(c:V) RETURN count(p)")
        self.env.assertEqual(res.result_set, [[2]])
        res = self.graph.query(
            "MATCH (a:V)-[:KNOWS]->(m1:V)-[:KNOWS]->(c:V) "
            "WHERE (a.id >= 28) AND (c.score <> 9970) "
            "RETURN a.id AS start_id, c.id AS end_id, count(*) AS path_count")
        self.env.assertEqual(res.result_set, [[31, 223, 2]])


class testIssue3196():
    """https://github.com/FalkorDB/FalkorDB/issues/3196
    A pattern comprehension whose projection is null (missing property) drops
    the element: `[(g)-[:HAS]->(m) | m.status]` is [] instead of [null], so
    NONE(s IN ... WHERE s = 'banned') is true instead of null and the row passes
    the WHERE. Expected: [null], and only group 1 is returned."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("ext_g_3196")
        self.graph.query("CREATE (g1:Group {id: 1}), (m1:Member {id: 11, status: 'active'}), "
                         "(g2:Group {id: 2}), (m2:Member {id: 12, status: 'banned'}), "
                         "(g3:Group {id: 3}), (m3:Member {id: 13}), "
                         "(g1)-[:HAS]->(m1), (g2)-[:HAS]->(m2), (g3)-[:HAS]->(m3)")

    def test01_pattern_comprehension_keeps_nulls(self):
        res = self.graph.query("MATCH (g:Group {id: 3}) "
                               "RETURN [(g)-[:HAS]->(m:Member) | m.status]")
        self.env.assertEqual(res.result_set, [[[None]]])

    def test02_none_over_null_filters_the_row(self):
        res = self.graph.query(
            "MATCH (g:Group) WITH g, [(g)-[:HAS]->(m:Member) | m.status] AS statuses "
            "WHERE NONE(s IN statuses WHERE s = 'banned') "
            "RETURN g.id AS group_id ORDER BY group_id ASC")
        self.env.assertEqual(res.result_set, [[1]])


class testIssue3197():
    """https://github.com/FalkorDB/FalkorDB/issues/3197
    Within one MATCH, comma-separated patterns may not bind the same
    relationship twice (relationship isomorphism). With a single :E edge,
    `MATCH p1=(..)-[:E]->(..), p2=(..)-[:E]->(..)` must return no rows."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("ext_g_3197")
        self.graph.query("CREATE (c1:C {id: 1}), (c2:C {id: 2}), (c1)-[:E {w: 5}]->(c2)")

    def test01_relationship_is_not_bound_twice(self):
        res = self.graph.query("MATCH p1=(:C {id:1})-[:E]->(:C {id:2}), "
                               "p2=(:C {id:1})-[:E]->(:C {id:2}) RETURN p1 = p2 AS is_eq")
        self.env.assertEqual(res.result_set, [])
        res = self.graph.query("MATCH (a)-[r1:E]->(b), (c)-[r2:E]->(d) RETURN count(*)")
        self.env.assertEqual(res.result_set, [[0]])
        res = self.graph.query("MATCH (a)-[r1:E]->(b), (a)-[r2:E]->(b) RETURN count(*)")
        self.env.assertEqual(res.result_set, [[0]])


class testIssue3198():
    """https://github.com/FalkorDB/FalkorDB/issues/3198
    A relationship variable bound earlier and passed through WITH is treated as
    a fresh variable in a later `MATCH (x)-[r]->(y)`, so every row re-scans all
    edges (25 rows for 5 edges). `r` must stay bound: 5 rows."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("ext_g_3198")
        self.graph.query("CREATE (n1:V {id: 1}), (n2:V {id: 2}), (n3:V {id: 3}), (n4:V {id: 4}), "
                         "(n1)-[:E {w:10}]->(n2), (n1)-[:E {w:20}]->(n3), (n2)-[:E {w:30}]->(n4), "
                         "(n3)-[:E {w:40}]->(n4), (n4)-[:E {w:50}]->(n4)")

    def test01_bound_relationship_stays_bound(self):
        res = self.graph.query("MATCH (a:V)-[r:E]->(b:V) WITH r MATCH (x)-[r]->(y) "
                               "RETURN x.id, y.id ORDER BY x.id, y.id")
        self.env.assertEqual(res.result_set, [[1, 2], [1, 3], [2, 4], [3, 4], [4, 4]])
        for q in ["MATCH ()-[r:E]->() WITH r MATCH (x)-[r:E]->(y) RETURN count(*)",
                  "MATCH ()-[r:E]->() WITH r MATCH (x)<-[r]-(y) RETURN count(*)"]:
            self.env.assertEqual(self.graph.query(q).result_set, [[5]], message=q)


class testIssue3199():
    """https://github.com/FalkorDB/FalkorDB/issues/3199
    sum() over integers returns a float (serialized as a double string), while
    reduce() over the same integers returns an integer. openCypher: the sum of
    integers is an integer; a float input makes the sum a float."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("ext_g_3199")
        self.graph.query("CREATE (a:Node {id: 1}), (b:Node {id: 2}), (a)-[:LINK {w: 1}]->(b)")

    def test01_sum_of_integers_is_an_integer(self):
        res = self.graph.query("MATCH p=(a:Node)-[:LINK]->(b:Node) "
                               "RETURN reduce(acc = 0, r IN relationships(p) | acc + coalesce(r.w, 0))")
        self.env.assertEqual(res.result_set, [[1]])
        self.env.assertTrue(type(res.result_set[0][0]) is int)
        for q, expected in [
                ("MATCH p=(a:Node)-[:LINK]->(b:Node) UNWIND relationships(p) AS r "
                 "WITH sum(coalesce(r.w, 0)) AS path_weight RETURN path_weight", 1),
                ("UNWIND [1, 2, 3] AS x RETURN sum(x)", 6)]:
            v = self.graph.query(q).result_set[0][0]
            self.env.assertEqual(v, expected, message=q)
            self.env.assertTrue(type(v) is int, message=f"{q} -> {v!r}")
        v = self.graph.query("UNWIND [1, 2.5] AS x RETURN sum(x)").result_set[0][0]
        self.env.assertTrue(type(v) is float and v == 3.5)


def _exth_expect_error(env, graph, query, *fragments):
    """Run `query`, expect a ResponseError whose message contains every
    fragment (case-insensitive), then check the server is still up."""
    try:
        graph.query(query)
        env.assertTrue(False, message=f"expected an error for: {query[:80]}")
    except ResponseError as e:
        for f in fragments:
            env.assertContains(f.lower(), str(e).lower())
    env.assertTrue(env.getConnection().ping())


class testIssue3200():
    """https://github.com/FalkorDB/FalkorDB/issues/3200
    A linear multi-hop match over two parallel WORKS_AT edges collapses them
    into one path, and a second pattern over the same pair reuses the edge the
    first one bound. Both queries must count 2 (one row per distinct edge
    assignment, with r1 <> r2)."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_3200")
        self.graph.query(
            "CREATE (u1:User {id: 1}), (c1:Company {id: 1}), (city1:City {id: 1}), "
            "(u1)-[:WORKS_AT]->(c1), (u1)-[:WORKS_AT]->(c1), (c1)-[:LOCATED_IN]->(city1)")

    def test01_linear_path_keeps_parallel_edges(self):
        res = self.graph.query(
            "MATCH (n1:User)-[:WORKS_AT]->(n2:Company)-[:LOCATED_IN]->(n3:City) "
            "RETURN count(n1) AS res")
        self.env.assertEqual(res.result_set, [[2]])

    def test02_comma_patterns_bind_distinct_edges(self):
        res = self.graph.query(
            "MATCH (n1:User)-[r1:WORKS_AT]->(n2:Company)-[:LOCATED_IN]->(n3:City), "
            "(n1)-[r2:WORKS_AT]->(n2) RETURN count(n1) AS res")
        self.env.assertEqual(res.result_set, [[2]])


class testIssue3201():
    """https://github.com/FalkorDB/FalkorDB/issues/3201
    `RETURN count1 * count(*)` where count1 comes from an earlier WITH and is
    not a grouping key returns NULL. It must either be rejected as a
    non-grouping sub-expression (Neo4j) or evaluate to 1 (Memgraph)."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_3201")
        self.graph.query(
            "CREATE (v1:Node {id: 1}), (v1)-[:LOOP]->(v1), (x1:Node {id: 2}), "
            "(y1:Node {id: 3}), (x1)-[:DIR]->(y1)")

    def test01_control_query(self):
        res = self.graph.query(
            "MATCH (a)-[:DIR]->(b) WITH count(*) AS count1 MATCH (c)-[:LOOP]->(d) "
            "WITH count1, count(*) AS count2 RETURN count1 * count2 AS result")
        self.env.assertEqual(res.result_set, [[1]])

    def test02_non_grouping_variable_in_aggregate_expression(self):
        q = ("MATCH (a)-[:DIR]->(b) WITH count(*) AS count1 MATCH (c)-[:LOOP]->(d) "
             "RETURN count1 * count(*) AS result")
        try:
            res = self.graph.query(q)
            # implicit grouping is accepted only if it is evaluated correctly
            self.env.assertEqual(res.result_set, [[1]])
        except ResponseError as e:
            self.env.assertContains("count1", str(e))
        self.env.assertTrue(self.env.getConnection().ping())


class testIssue3202():
    """https://github.com/FalkorDB/FalkorDB/issues/3202
    An explicit two-hop pattern over parallel edges loses cardinality (or, on
    C, reuses a self-loop edge twice) while `-[:R*2]->` counts correctly. Both
    must give dst 2 -> 2 paths and dst 3 -> 4 paths."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_3202")
        self.graph.query(
            "CREATE (a:V {id:1}), (b:V {id:2}), (c:V {id:3}), "
            "(a)-[:R {eid:1}]->(b), (a)-[:R {eid:2}]->(b), (b)-[:R {eid:3}]->(c), "
            "(b)-[:R {eid:4}]->(c), (c)-[:R {eid:5}]->(a), (a)-[:R {eid:6}]->(a)")

    def test01_var_length_baseline(self):
        res = self.graph.query(
            "MATCH (a:V {id:1})-[:R*2]->(b:V) RETURN b.id AS dst, count(*) AS n ORDER BY dst")
        self.env.assertEqual(res.result_set, [[2, 2], [3, 4]])

    def test02_explicit_two_hop_matches_var_length(self):
        res = self.graph.query(
            "MATCH (a:V {id:1})-[r1:R]->(v:V)-[r2:R]->(b:V) "
            "RETURN b.id AS dst, count(*) AS n ORDER BY dst")
        self.env.assertEqual(res.result_set, [[2, 2], [3, 4]])


class testIssue3203():
    """https://github.com/FalkorDB/FalkorDB/issues/3203
    A three-hop path over parallel edges and self-loops undercounts, and an
    untyped edge filtered by `type(x)` gives a different count than the inline
    `[:S]`/`[:T]` type. Both forms must give S -> 10 and T -> 6."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_3203")
        self.graph.query(
            "CREATE (a:V {id:1}), (b:V {id:2}), (c:V {id:3}), (d:V {id:4}), "
            "(a)-[:R]->(b), (a)-[:R]->(b), (a)-[:R]->(c), "
            "(b)-[:S]->(c), (b)-[:S]->(c), (b)-[:T]->(c), "
            "(c)-[:S]->(c), (c)-[:T]->(c), "
            "(c)-[:U]->(d), (c)-[:U]->(d), (b)-[:U]->(d)")

    def test01_where_type_filter(self):
        res = self.graph.query(
            "MATCH (a:V {id:1})-[:R]->(b)-[x]->(c)-[:U]->(d:V) "
            "WHERE type(x) IN ['S','T'] "
            "RETURN type(x) AS branch, count(*) AS n ORDER BY branch")
        self.env.assertEqual(res.result_set, [['S', 10], ['T', 6]])

    def test02_inline_type(self):
        res = self.graph.query(
            "MATCH (a:V {id:1})-[:R]->(b)-[:S]->(c)-[:U]->(d:V) "
            "RETURN 'S' AS branch, count(*) AS n "
            "UNION ALL "
            "MATCH (a:V {id:1})-[:R]->(b)-[:T]->(c)-[:U]->(d:V) "
            "RETURN 'T' AS branch, count(*) AS n")
        self.env.assertEqual(sorted(res.result_set), [['S', 10], ['T', 6]])


class testIssue3205():
    """https://github.com/FalkorDB/FalkorDB/issues/3205
    INT64_MIN / -1 is not representable as a 64-bit integer; Rust wraps it to
    INT64_MIN and C saturates to INT64_MAX. It must raise an overflow error."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_3205")

    def test01_int64_min_div_minus_one_overflows(self):
        for q in ["RETURN -9223372036854775808 / -1",
                  "RETURN (-9223372036854775807 - 1) / -1",
                  "WITH -9223372036854775808 AS x, -1 AS y RETURN x / y"]:
            _exth_expect_error(self.env, self.graph, q, "overflow")

    def test02_int64_min_mod_minus_one_is_zero(self):
        res = self.graph.query("RETURN -9223372036854775808 % -1")
        self.env.assertEqual(res.result_set, [[0]])


class testIssue3226():
    """https://github.com/FalkorDB/FalkorDB/issues/3226
    A write query whose range-index scan is still open at commit (scan below
    LIMIT, or feeding a CALL {} subquery) aborts the server with "attempted to
    upgrade a spec read lock to a write lock". It must return its count."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_3226")
        self.graph.query("UNWIND range(1, 2000) AS i CREATE (:W1 {v: i}), (:W2 {v: i})")
        create_node_range_index(self.graph, 'W1', 'v', sync=True)
        create_node_range_index(self.graph, 'W2', 'v', sync=True)

    def test01_write_under_limit_over_index_scan(self):
        res = self.graph.query(
            "MATCH (n:W1) WHERE n.v > 0 WITH n LIMIT 10 SET n.v = n.v + 100000 RETURN count(n)")
        self.env.assertEqual(res.result_set, [[10]])
        res = self.graph.query("MATCH (n:W1) WHERE n.v > 100000 RETURN count(n)")
        self.env.assertEqual(res.result_set, [[10]])
        self.env.assertTrue(self.env.getConnection().ping())

    def test02_write_in_call_subquery_over_index_scan(self):
        res = self.graph.query(
            "MATCH (n:W2) WHERE n.v > 0 CALL { WITH n SET n.v = n.v + 100000 } RETURN count(*)")
        self.env.assertEqual(res.result_set, [[2000]])
        res = self.graph.query("MATCH (n:W2) WHERE n.v > 100000 RETURN count(n)")
        self.env.assertEqual(res.result_set, [[2000]])
        self.env.assertTrue(self.env.getConnection().ping())


class testIssue3168():
    """https://github.com/FalkorDB/FalkorDB/issues/3168

    GRAPH.RESTORE crashes the server on an empty or a truncated payload. The
    empty payload is covered by test_open_rust_s1_issues.py testIssue2537 (the
    same bug, #2537); this class covers the truncated one. Expected: every
    truncated payload is refused with a clean error, no key is created, and the
    server stays up. The payload is in the Rust engine's format, so this test
    does not apply to the C engine.
    """

    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()

    def test01_truncated_payloads_are_rejected_not_fatal(self):
        payload = _good_payload()
        for cut in (1, 16, len(payload) // 2, len(payload) - 1):
            key = f"truncated_restore_{cut}"
            try:
                self.conn.execute_command("GRAPH.RESTORE", key, payload[:cut])
                raised = False
            except (ResponseError, redis.ConnectionError) as e:
                raised = True
                # A ResponseError is the correct outcome; a dropped connection
                # means the server crashed, which is the bug.
                self.env.assertIsInstance(
                    e, ResponseError,
                    message=f"RESTORE truncated to {cut} bytes must be a clean "
                            f"error, got {e!r}")
            self.env.assertTrue(
                raised,
                message=f"RESTORE truncated to {cut} of {len(payload)} bytes "
                        f"must be refused")
            self.env.assertEqual(self.env.getConnection().exists(key), 0)
        self.env.assertTrue(self.env.getConnection().ping())
