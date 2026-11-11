"""CCH indexes, end to end.

The CCH index exists only in a module built with `--features index-falkordb`,
so every class here skips unless `FALKORDB_FEATURES` declares it (see
`common.py`), and the file is not in `flow_tests_done.txt`, the default build's
CI matrix: CI runs it from `flow_tests_index_falkordb.txt`, against a module
built with the feature. All CCH flow classes share this one file.

Effects -- CCH index DDL, which is not replicated yet. See `effects_common.py`
for the shared fixture.
"""

from common import *

from effects_common import _EffectsBase


#-----------------------------------------------------------------------------
# 1. CCH DDL is not replicated, so it must never create schema
#-----------------------------------------------------------------------------


NOT_HELD = "no CCH index over these relationship types and weight attribute"
ALREADY_HELD = ("a CCH index already exists over these relationship types and "
                "weight attribute")


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

    def _refused(self, q, *needles, read_only=False):
        """`q` fails on the primary with an error containing every needle."""
        run = (self.master_graph.ro_query if read_only
               else self.master_graph.query)
        try:
            run(q)
        except ResponseError as e:
            for needle in needles:
                self.env.assertContains(needle, str(e))
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
                      "relationship type 'ROAD' does not exist", "first")
        self.env.assertEqual(self._schema_of(self.master_graph), ([], []))

        # A normal write creates the type and the attribute, as an effect.
        self.query_and_sync("CREATE (:N)-[:ROAD {time: 1.0}]->(:N)")
        self._refused("CREATE CCH INDEX FOR ()-[e:ROAD|FERRY]->() ON (e.time)",
                      "relationship type 'FERRY' does not exist", "first")
        self._refused("CREATE CCH INDEX FOR ()-[e:ROAD]->() ON (e.length)",
                      "attribute 'length' does not exist", "first")
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
            self._refused(q, NOT_HELD)

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

    def test02_create_drop_and_refusals(self):
        self.set_effects_config()
        self.query_and_sync("CREATE (:N)-[:A {w: 1.0}]->(:N)-[:B {w: 2.0}]->(:N)")

        # The types are a set: repeats and order do not make a new index.
        res = self.query_and_sync("CREATE CCH INDEX FOR ()-[e:A]->() ON (e.w)")
        self.env.assertEqual(res.indices_created, 1)
        self._refused("CREATE CCH INDEX FOR ()-[e:A|A]->() ON (e.w)",
                      ALREADY_HELD)
        res = self.query_and_sync(
            "CREATE CCH INDEX FOR ()-[e:A|B]->() ON (e.w)")
        self.env.assertEqual(res.indices_created, 1)
        self._refused("CREATE CCH INDEX FOR ()-[e:B|A]->() ON (e.w)",
                      ALREADY_HELD)

        # CCH DDL is a write: GRAPH.RO_QUERY refuses it.
        for verb in ("CREATE", "DROP"):
            self._refused(f"{verb} CCH INDEX FOR ()-[e:A]->() ON (e.w)",
                          "graph.RO_QUERY is to be executed only on read-only "
                          "queries", read_only=True)

        # A drop names the same set in any spelling, and frees it for a create.
        res = self.query_and_sync(
            "DROP CCH INDEX FOR ()-[e:B|A|B]->() ON (e.w)")
        self.env.assertEqual(res.indices_deleted, 1)
        self._refused("DROP CCH INDEX FOR ()-[e:A|B]->() ON (e.w)", NOT_HELD)
        res = self.query_and_sync(
            "CREATE CCH INDEX FOR ()-[e:A|B]->() ON (e.w)")
        self.env.assertEqual(res.indices_created, 1)
