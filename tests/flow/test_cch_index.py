"""CCH indexes, end to end.

The CCH index exists only in a module built with `--features index-falkordb`,
so every class here skips unless `FALKORDB_FEATURES` declares it (see
`common.py`), and the file is not in `flow_tests_done.txt`, the default build's
CI matrix: it runs from `flow_tests_index_falkordb.txt`, against a module built
with the feature. All CCH flow classes share this one file.

Effects -- CCH index DDL, which is not replicated yet. See `effects_common.py`
for the shared fixture.
"""

import struct

from common import *

from effects_common import _EffectsBase


#-----------------------------------------------------------------------------
# 1. CCH DDL is not replicated, so it must never create schema
#-----------------------------------------------------------------------------


class testEffects_01_CchDdlCreatesNoSchema(_EffectsBase):
    """CREATE CCH INDEX emits no effects yet, so it must not create schema.

    A relationship type or attribute it created would exist on the primary
    only. The next write naming it carries an id no earlier effect announced,
    the replica refuses it ("relationship type id 0 out of range") and
    resyncs. So the statement refuses to run until the type and the weight
    attribute exist, and a replica that never sees CCH DDL still resolves every
    id the primary uses afterwards.
    """

    GRAPH_ID = "effects_cch_ddl"

    def __init__(self):
        if not INDEX_FALKORDB:
            Environment.skip(None)
        self._setup()

    @staticmethod
    def _schema_of(graph):
        """(relationship types, property keys), in id order."""
        return tuple([r[0] for r in graph.ro_query(q).result_set]
                     for q in ("CALL db.relationshipTypes()",
                               "CALL db.propertyKeys()"))

    def _schema(self):
        """The primary's schema, asserted equal (same ids) on the replica."""
        m = self._schema_of(self.master_graph)
        self.env.assertEqual(self._schema_of(self.replica_graph), m)
        return m

    def _refused(self, q, needle):
        try:
            self.master_graph.query(q)
        except ResponseError as e:
            self.env.assertContains(needle, str(e))
            self.env.assertContains("first", str(e))
            return
        self.env.assertTrue(False, message=f"{q} was not refused")

    def test01_ddl_over_missing_schema_is_refused_and_writes_still_apply(self):
        self.set_effects_config()
        full_before = self.master.info()["sync_full"]
        failures_before = self.effect_failures()
        out_of_range_before = self.replica_log_count(b"out of range")

        # Nothing exists yet: refused, and nothing was created. The query left
        # an empty graph on the primary, which no effect announces, so the
        # replica has no key to read until the first write below.
        self._refused("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)",
                      "relationship type 'ROAD' does not exist")
        self.env.assertEqual(self._schema_of(self.master_graph), ([], []))

        # A normal write creates the type and the attribute, as an effect.
        self.query_and_sync("CREATE (:N)-[:ROAD {time: 1.0}]->(:N)")
        self._refused("CREATE CCH INDEX FOR ()-[e:ROAD|FERRY]->() ON (e.time)",
                      "relationship type 'FERRY' does not exist")
        self._refused("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.length)",
                      "attribute 'length' does not exist")
        self.wait_for_replica_offset()
        self.env.assertEqual(self._schema(), (['ROAD'], ['time']))

        # Over existing schema the index is created.
        res = self.query_and_sync(
            "CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")
        self.env.assertEqual(res.indices_created, 1)

        # A DROP naming a type or an attribute that does not exist matches no
        # index, and is refused like the drop of an index that is not held.
        for q in ("DROP CCH INDEX FOR ()-[e:TRAIN]->() ON (e.time)",
                  "DROP CCH INDEX FOR ()-[e:ROAD]->() ON (e.speed)"):
            try:
                self.master_graph.query(q)
                self.env.assertTrue(False, message=f"{q} was not refused")
            except ResponseError as e:
                self.env.assertContains("no CCH index over these relationship "
                                        "types and weight attribute", str(e))

        # Later writes apply on the replica: the indexed type and its weight,
        # then a new type and a new attribute.
        self.query_and_sync("CREATE (:N)-[:ROAD {time: 2.0}]->(:N)")
        self.query_and_sync("MATCH ()-[r:ROAD]->() SET r.time = 3.0")
        self.query_and_sync("CREATE (:N)-[:FERRY {time: 4.0, length: 5}]->(:N)")
        self.env.assertEqual(self._schema(),
                             (['ROAD', 'FERRY'], ['time', 'length']))
        self.assert_agree(
            "MATCH ()-[r]->() RETURN type(r), r.time, r.length "
            "ORDER BY type(r), r.time",
            [['FERRY', 4.0, 5], ['ROAD', 3.0, None], ['ROAD', 3.0, None]])
        # Not `assert_graph_eq`: it compares index lists, and the CCH index is
        # on the primary only until CCH DDL is replicated.

        # And none of it was refused: no resync, no refused GRAPH.EFFECT.
        self.env.assertEqual(self.master.info()["sync_full"], full_before,
            message="the replica was resynced")
        self.env.assertEqual(self.effect_failures(), failures_before,
            message="the replica refused a GRAPH.EFFECT")
        out_of_range = self.replica_log_count(b"out of range")
        if out_of_range is not None:
            self.env.assertEqual(out_of_range, out_of_range_before,
                message="the replica could not resolve an id")


#-----------------------------------------------------------------------------
# 2. An indexed weight is a constraint: refused at the write
#-----------------------------------------------------------------------------


class testCchWeightConstraint():
    """A negative (-inf included) or NaN weight on an edge of an indexed type fails
    the write that sets it, through Cypher and GRAPH.BULK alike, and leaves the graph
    as it was. Other types and attributes are not checked. +inf is legal: a closed
    edge."""

    def __init__(self):
        if not INDEX_FALKORDB:
            Environment.skip(None)
        self.env, self.db = Env()
        self.con = self.env.getConnection()
        self.graph = self.db.select_graph("cch_weight_constraint")
        self.graph.query(
            "CREATE (a:N {id: 0}), (b:N {id: 1}), (c:N {id: 2}), "
            "(a)-[:ROAD {time: 1.0}]->(b), (b)-[:ROAD {time: 2.0}]->(c), "
            "(a)-[:FERRY {time: 9.0}]->(c)")
        self.graph.query("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")

    def _refused(self, run):
        try:
            run()
        except ResponseError as e:
            self.env.assertContains("CCH weight constraint violation", str(e))
            self.env.assertContains("(:ROAD {time})", str(e))
            self.env.assertContains(
                "needs weights that are >= 0 (+inf marks a closed edge)", str(e))
            return
        self.env.assertTrue(False, message="the write was not refused")

    def _unchanged(self):
        self.env.assertEquals(self.graph.ro_query(
            "MATCH ()-[r]->() RETURN type(r), r.time ORDER BY type(r), r.time").result_set,
            [["FERRY", 9.0], ["ROAD", 1.0], ["ROAD", 2.0]])

    def test01_cypher(self):
        for q in ["MATCH ()-[r:ROAD]->() SET r.time = -1",
                  "MATCH (a:N {id: 0}), (c:N {id: 2}) CREATE (a)-[:ROAD {time: -0.5}]->(c)",
                  "MATCH ()-[r:ROAD]->() SET r.time = 1 WITH r SET r.time = -1"]:
            self._refused(lambda: self.graph.query(q))
            self._unchanged()
        # Not indexed: another type. Then put it back.
        self.graph.query("MATCH ()-[r:FERRY]->() SET r.time = -1")
        self.graph.query("MATCH ()-[r:FERRY]->() SET r.time = 9.0")
        self._unchanged()

    def test02_bulk(self):
        # Append one edge 0 -> 2 to the existing graph: type, one double property `time`.
        def edge(rel, w):
            return (rel.encode() + b"\0" + struct.pack("=I", 1) + b"time\0" +
                    struct.pack("=QQ", 0, 2) + struct.pack("=Bd", 2, w))
        bulk = lambda rel, w: self.con.execute_command(
            "GRAPH.BULK", "cch_weight_constraint", 0, 1, 0, 1, edge(rel, w))
        self._refused(lambda: bulk("ROAD", -1.0))
        self._unchanged()
        bulk("FERRY", -1.0)
        self.env.assertEquals(self.graph.ro_query(
            "MATCH ()-[r:FERRY]->() RETURN count(r)").result_set, [[2]])

    def test03_closed_edge(self):
        # +inf (1.0/0.0 in Cypher) is a closed edge: legal through SET, CREATE and
        # GRAPH.BULK, and an index can be created over it. -inf is refused.
        g = self.db.select_graph("cch_closed_edge")
        g.query("CREATE (a:N {id: 0}), (b:N {id: 1}), (c:N {id: 2}), "
                "(a)-[:ROAD {time: 1.0}]->(b), (b)-[:ROAD {time: 1.0 / 0.0}]->(c)")
        g.query("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")
        g.query("MATCH (:N {id: 0})-[r:ROAD]->() SET r.time = 1.0 / 0.0")
        g.query("MATCH (a:N {id: 0}), (c:N {id: 2}) CREATE (a)-[:ROAD {time: 1.0 / 0.0}]->(c)")
        edge = (b"ROAD\0" + struct.pack("=I", 1) + b"time\0" + struct.pack("=QQ", 2, 0) +
                struct.pack("=Bd", 2, float("inf")))
        self.con.execute_command("GRAPH.BULK", "cch_closed_edge", 0, 1, 0, 1, edge)
        self.env.assertEquals(g.ro_query(
            "MATCH ()-[r:ROAD]->() RETURN toString(r.time), count(r)").result_set,
            [["inf", 4]])
        try:
            g.query("MATCH (:N {id: 0})-[r:ROAD]->(:N {id: 1}) SET r.time = -1.0 / 0.0")
            self.env.assertTrue(False, message="-inf was not refused")
        except ResponseError as e:
            self.env.assertContains("CCH weight constraint violation", str(e))

        # CREATE CCH INDEX over data: +inf is fine, a negative weight names the rule.
        g.query("MATCH ()-[r:ROAD]->() SET r.len = 1.0 / 0.0")
        g.query("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.len)")
        g.query("MATCH ()-[r:ROAD]->() SET r.cost = -2")
        try:
            g.query("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.cost)")
            self.env.assertTrue(False, message="the index was created over -2")
        except ResponseError as e:
            self.env.assertContains("edge weights must be >= 0 (+inf marks a closed edge)",
                                    str(e))

    def test04_a_closed_edge_is_routed_around(self):
        # 0 -> 1 -> 2 costs 2, the direct 0 -> 2 costs 10.
        g = self.db.select_graph("cch_closed_route")
        g.query("CREATE (a:N {id: 0}), (b:N {id: 1}), (c:N {id: 2}), "
                "(a)-[:ROAD {time: 1.0}]->(b), (b)-[:ROAD {time: 1.0}]->(c), "
                "(a)-[:ROAD {time: 10.0}]->(c)")
        g.query("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")
        q = ("MATCH (a:N {id: $s}), (b:N {id: $t}) "
             "CALL db.idx.cch.query({sourceNode: a, targetNode: b, relTypes: ['ROAD'], "
             "weightProp: 'time'}) YIELD pathWeight, path "
             "RETURN pathWeight, [n IN nodes(path) | n.id]")
        route = lambda: g.ro_query(q, {"s": 0, "t": 2}).result_set
        self.env.assertEquals(route(), [[2.0, [0, 1, 2]]])
        # Close 1 -> 2: the detour is taken.
        g.query("MATCH (:N {id: 1})-[r:ROAD]->(:N {id: 2}) SET r.time = 1.0 / 0.0")
        self.env.assertEquals(route(), [[10.0, [0, 2]]])
        # A finite edge parallel to the closed one wins the pair.
        g.query("MATCH (b:N {id: 1}), (c:N {id: 2}) CREATE (b)-[:ROAD {time: 3.0}]->(c)")
        self.env.assertEquals(route(), [[4.0, [0, 1, 2]]])
        # Every way into 2 closed: no row.
        g.query("MATCH ()-[r:ROAD]->(:N {id: 2}) SET r.time = 1.0 / 0.0")
        self.env.assertEquals(route(), [])
        # Reopen every edge into 2 at 1.0: the direct edge now wins.
        g.query("MATCH ()-[r:ROAD]->(:N {id: 2}) SET r.time = 1.0")
        self.env.assertEquals(route(), [[1.0, [0, 2]]])


#-----------------------------------------------------------------------------
# 3. GRAPH.MEMORY USAGE counts the hierarchy
#-----------------------------------------------------------------------------


class testCchMemoryUsage():
    """A CCH index's hierarchy counts in indices_sz_mb, and DROP CCH INDEX releases it."""

    def __init__(self):
        if not INDEX_FALKORDB:
            Environment.skip(None)
        self.env, self.db = Env()
        self.con = self.env.getConnection()
        self.graph = self.db.select_graph("cch_memory_usage")

    def _indices_mb(self):
        res = self.con.execute_command("GRAPH.MEMORY", "USAGE", "cch_memory_usage")
        return dict(zip(res[0::2], res[1::2]))['indices_sz_mb']

    def test01_counted_and_released(self):
        # A 100 x 100 grid: its hierarchy takes a few MiB, enough to show in whole MiB.
        side = 100
        params = {'n': side * side, 'side': side}
        self.graph.query("UNWIND range(0, $n - 1) AS x CREATE (:J)", params)
        self.graph.query("""UNWIND range(0, $n - 1) AS x
                            MATCH (a) WHERE id(a) = x
                            MATCH (b) WHERE id(b) = x + 1 AND x % $side <> $side - 1
                            CREATE (a)-[:ROAD {time: 1.0}]->(b)""", params)
        self.graph.query("""UNWIND range(0, $n - $side - 1) AS x
                            MATCH (a) WHERE id(a) = x
                            MATCH (b) WHERE id(b) = x + $side
                            CREATE (a)-[:ROAD {time: 1.0}]->(b)""", params)
        self.env.assertEqual(self._indices_mb(), 0)

        self.graph.query("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")
        self.env.assertGreater(self._indices_mb(), 0)

        self.graph.query("DROP CCH INDEX FOR ()-[e:ROAD]->() ON (e.time)")
        self.env.assertEqual(self._indices_mb(), 0)
