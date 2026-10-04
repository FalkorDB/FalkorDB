"""Does every value survive the wire, in both payload versions?

Values are the half of a record the format spends most of its bytes on, so this
is where a width code or a type tag being wrong shows up as a live divergence
rather than a refused payload.

The non-deterministic case is the one to read twice. `rand()` and `timestamp()`
are why effects replication exists at all: replaying the query text on the
replica re-evaluates them and produces a DIFFERENT graph. So that test asserts
agreement under effects, and is the one case the original suite refused to
re-run with effects disabled -- not an oversight, the point.
"""

from common import *
from effects_common import _EffectsBase


class test01_Values(_EffectsBase):
    GRAPH_ID = "effects_values"

    def __init__(self):
        self._setup(version=2, threshold=0)
        self.start_monitor("GRAPH.EFFECT", "GRAPH.QUERY")

    def test01_scalar_types(self):
        def body(v):
            res = self.query_and_sync(
                "CREATE ({i: 1, d: 1.5, s: 'str', b: true, n: null})")
            self.env.assertEquals(res.nodes_created, 1)
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()
        self.for_each_version(body)

    def test02_array_and_nested(self):
        def body(v):
            self.query_and_sync("CREATE ({a: [1, 2, 3], e: []})")
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()
        self.for_each_version(body)

    def test03_empty_vector(self):
        def body(v):
            res = self.query_and_sync("CREATE ({v: vecf32([])})")
            self.env.assertEquals(res.nodes_created, 1)
            self.env.assertEquals(res.properties_set, 1)
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()
        self.for_each_version(body)

    def test04_vector(self):
        def body(v):
            self.query_and_sync("CREATE ({v: vecf32([0.1, 0.2, 0.3])})")
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()
        self.for_each_version(body)

    def test05_null_is_a_removal(self):
        def body(v):
            self.query_and_sync("CREATE (:N {keep: 1, drop: 2})")
            self.assert_effect_emitted()
            self.query_and_sync("MATCH (n:N) SET n.drop = null")
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()
            row = self.master_graph.query(
                "MATCH (n:N) RETURN n.drop").result_set[0][0]
            self.env.assertIsNone(row)
        self.for_each_version(body)

    def test06_non_deterministic_values(self):
        """rand() and timestamp() -- the reason effects exist.

        Replaying the query text on the replica would re-evaluate these and
        produce different values, so agreement here is evidence the VALUES
        travelled rather than the statement.
        """
        def body(v):
            res = self.query_and_sync("CREATE ({r: rand(), t: timestamp()})")
            self.env.assertEquals(res.nodes_created, 1)
            self.env.assertEquals(res.properties_set, 2)
            self.assert_wire_version(self.assert_effect_emitted())
            self.assert_graph_eq()
        self.for_each_version(body)
