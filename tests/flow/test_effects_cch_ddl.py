"""Effects -- CCH index DDL, which is not replicated yet.

See `effects_common.py` for the shared fixture and why these are split. This
file is its own because `CCH_INDEX` is an immutable load-time argument: the
literal below routes the file to CI's spawn bucket, and keeping it apart leaves
the other effects files in the shared-services bucket.
"""

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

    `CCH_INDEX` is immutable, so it is a load-time argument, on both sides: the
    literal below is also what routes this file to CI's spawn bucket.
    """

    GRAPH_ID = "effects_cch_ddl"

    def __init__(self):
        self._setup(moduleArgs="CCH_INDEX yes")

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
        for con in (self.master, self.replica):
            self.env.assertEqual(
                con.execute_command("GRAPH.CONFIG", "GET", "CCH_INDEX"),
                ["CCH_INDEX", 1])
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
