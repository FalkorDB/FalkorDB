"""Index and constraint DDL as effects, in both payload versions.

The largest group in the original suite and the one most worth running twice:
the index record carries the most structure of any record on the wire -- a
schema list, a field list, and per-type option blocks each behind a presence
byte -- so it is where a version's layout differs most.

Waiting here needs two separate things, and the original suite's comment is
worth keeping: observing the command on MONITOR says it was SENT, not that the
replica finished applying it. Index population is asynchronous on both sides, so
every assertion on index state waits for the replica to ack AND for both sides
to finish populating.
"""

from common import *
from index_utils import *
from constraint_utils import *
from effects_common import _EffectsBase


class test01_IndexDDL(_EffectsBase):
    GRAPH_ID = "effects_ddl_index"

    def __init__(self):
        self._setup(version=2, threshold=0)
        self.start_monitor("GRAPH.EFFECT", "GRAPH.QUERY")

    def _settle(self):
        self.master.execute_command("WAIT", "1", "400")
        wait_for_indices_to_sync(self.master_graph)
        wait_for_indices_to_sync(self.replica_graph)

    def test01_range_index_node_and_edge(self):
        def body(v):
            create_node_range_index(self.master_graph, "IdxL", "a")
            self.assert_wire_version(self.assert_effect_emitted())
            create_edge_range_index(self.master_graph, "IdxR", "a", "b")
            self.assert_wire_version(self.assert_effect_emitted())
            self._settle()
            self.env.assertEquals(list_indicies(self.master_graph).result_set,
                                  list_indicies(self.replica_graph).result_set)
        self.for_each_version(body)

    def test02_fulltext_index_with_options(self):
        def body(v):
            create_node_fulltext_index(self.master_graph, "IdxL", "c",
                                       language="english", stopwords=["the", "a"])
            self.assert_wire_version(self.assert_effect_emitted())
            self._settle()
            self.env.assertEquals(list_indicies(self.master_graph).result_set,
                                  list_indicies(self.replica_graph).result_set)
        self.for_each_version(body)

    def test03_vector_index_with_params(self):
        def body(v):
            create_node_vector_index(self.master_graph, "IdxL", "v", dim=4, m=8,
                                     efConstruction=100, efRuntime=5)
            self.assert_wire_version(self.assert_effect_emitted())
            self._settle()
            self.env.assertEquals(list_indicies(self.master_graph).result_set,
                                  list_indicies(self.replica_graph).result_set)
        self.for_each_version(body)

    def test04_drop_index(self):
        def body(v):
            create_node_range_index(self.master_graph, "Dropped", "a", sync=True)
            self.assert_effect_emitted()
            drop_node_range_index(self.master_graph, "Dropped", "a")
            self.assert_wire_version(self.assert_effect_emitted())
            self._settle()
            self.env.assertEquals(list_indicies(self.master_graph).result_set,
                                  list_indicies(self.replica_graph).result_set)
        self.for_each_version(body)

    def test05_index_correct_after_mutation(self):
        """An index the replica built from an effect must answer the same
        queries as the primary's after the data moves under it."""
        def body(v):
            create_node_range_index(self.master_graph, "Mut", "n", sync=True)
            self.assert_effect_emitted()
            self.query_and_sync("UNWIND range(0, 99) AS i CREATE (:Mut {n: i})")
            self.assert_effect_emitted()
            self.query_and_sync("MATCH (m:Mut) WHERE m.n < 50 SET m.n = m.n + 1000")
            self.assert_effect_emitted()
            self._settle()
            q = "MATCH (m:Mut) WHERE m.n > 500 RETURN count(m)"
            self.env.assertEquals(
                self.master_graph.query(q).result_set[0][0],
                self.replica_graph.ro_query(q).result_set[0][0])
            self.assert_graph_eq()
        self.for_each_version(body)


class test02_ConstraintDDL(_EffectsBase):
    GRAPH_ID = "effects_ddl_constraint"

    def __init__(self):
        self._setup(version=2, threshold=0)
        self.start_monitor("GRAPH.EFFECT", "GRAPH.QUERY")

    def test01_create_and_drop_unique_constraint(self):
        def body(v):
            create_node_range_index(self.master_graph, "C", "a", sync=True)
            self.assert_effect_emitted()
            create_unique_node_constraint(self.master_graph, "C", "a", sync=True)
            self.assert_wire_version(self.assert_effect_emitted())
            self.master.execute_command("WAIT", "1", "400")
            self.env.assertEquals(
                len(list_constraints(self.master_graph)),
                len(list_constraints(self.replica_graph)))

            drop_constraint(self.master_graph, "UNIQUE", "NODE", "C", "a")
            self.assert_wire_version(self.assert_effect_emitted())
            self.master.execute_command("WAIT", "1", "400")
            self.env.assertEquals(
                len(list_constraints(self.master_graph)),
                len(list_constraints(self.replica_graph)))
        self.for_each_version(body)

    def test02_mandatory_constraint(self):
        def body(v):
            create_mandatory_node_constraint(self.master_graph, "M", "a", sync=True)
            self.assert_wire_version(self.assert_effect_emitted())
            self.master.execute_command("WAIT", "1", "400")
            self.env.assertEquals(
                len(list_constraints(self.master_graph)),
                len(list_constraints(self.replica_graph)))
        self.for_each_version(body)
