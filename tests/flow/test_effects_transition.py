"""A graph written at v2 and continued at v3, on one live pair.

THE CASE NEITHER ENGINE TESTS TODAY, and it is only reachable because
EFFECTS_VERSION takes effect on the next query rather than at module load. That
makes it the actual upgrade path an operator takes: a running primary is
reconfigured, and the replica -- which never restarted -- has to apply both
formats from the same connection, in order, against one graph.

Kept out of `for_each_version` deliberately. Every other suite starts each
version on a FRESH graph so that v3 assertions are not quietly reading v2 state.
Here the shared graph is the point.
"""

from common import *
from effects_common import _EffectsBase


class test01_VersionTransition(_EffectsBase):
    GRAPH_ID = "effects_transition"

    def __init__(self):
        self._setup(version=2, threshold=0)
        self.start_monitor("GRAPH.EFFECT", "GRAPH.QUERY")

    def test01_v2_then_v3_on_one_graph(self):
        # written at v2
        self.set_effects_config(version=2)
        self.query_and_sync("CREATE (:P {n: 1}), (:P {n: 2})")
        self.assert_wire_version(self.assert_effect_emitted(), version=2)

        # flip the primary mid-stream; the replica is not restarted
        self.set_effects_config(version=3)
        self.query_and_sync("CREATE (:P {n: 3})")
        self.assert_wire_version(self.assert_effect_emitted(), version=3)

        # the replica applied both formats into one graph
        self.assert_graph_eq()
        self.env.assertEquals(
            self.master_graph.query(
                "MATCH (p:P) RETURN count(p)").result_set[0][0], 3)

    def test02_v3_then_v2_still_converges(self):
        # the reverse, which a downgrade or a rollback produces
        self.set_effects_config(version=3)
        self.query_and_sync("CREATE (:Q {n: 1})")
        self.assert_wire_version(self.assert_effect_emitted(), version=3)

        self.set_effects_config(version=2)
        self.query_and_sync("MATCH (q:Q {n: 1}) SET q.n = 2")
        self.assert_wire_version(self.assert_effect_emitted(), version=2)

        self.assert_graph_eq()
        self.env.assertEquals(
            self.master_graph.query(
                "MATCH (q:Q) RETURN q.n").result_set[0][0], 2)

    def test03_entities_written_at_one_version_are_updatable_at_the_other(self):
        # an id minted under v2 addressed by a v3 record, and the reverse --
        # the case where a version change could plausibly break continuity
        self.set_effects_config(version=2)
        self.query_and_sync("CREATE (:X {tag: 'from_v2'})")
        self.assert_effect_emitted()

        self.set_effects_config(version=3)
        self.query_and_sync("MATCH (x:X {tag: 'from_v2'}) SET x.touched = 1")
        self.assert_wire_version(self.assert_effect_emitted(), version=3)
        self.assert_graph_eq()

        self.set_effects_config(version=2)
        self.query_and_sync("MATCH (x:X {tag: 'from_v2'}) SET x.touched = 2")
        self.assert_wire_version(self.assert_effect_emitted(), version=2)
        self.assert_graph_eq()
