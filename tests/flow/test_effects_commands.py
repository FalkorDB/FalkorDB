"""The commands that still replicate verbatim, and the threshold only v2 honours.

VERSION-PINNED ON PURPOSE. Everything here is about *not* using effects, and
"not using effects" is not a property both versions have:

  EFFECTS_THRESHOLD disables effects under v2. Under v3 it does nothing --
  _should_replicate_effects returns true on the version check before the
  threshold is ever read (cmd_query.c). So a `for_each_version` loop over these
  would assert something false of v3, and the v3 half would either fail for the
  right reason by luck or pass while measuring nothing.

The original suite re-ran thirteen opcode tests with `expect_effect=False`, and
that flag only SKIPPED the effect assertion -- it did not assert absence. Those
re-runs therefore passed whether or not effects were actually disabled, and the
only thing standing between the suite and a silent wrong-path run was a single
trailing assertFalse. Here absence is asserted directly, per write.
"""

from common import *
from effects_common import _EffectsBase, WIRE


class test01_ThresholdDisablesV2(_EffectsBase):
    """v2 only: raising the threshold puts writes back on the query path."""

    GRAPH_ID = "effects_cmd_v2"

    def __init__(self):
        self._setup(version=2, threshold=0)
        self.start_monitor("GRAPH.EFFECT", "GRAPH.QUERY")

    def test01_threshold_disables_effects(self):
        self.env.assertTrue(WIRE[2]["threshold_disables"])

        # baseline: at threshold 0 the write ships as an effect
        self.query_and_sync("CREATE (:T {a: 1})")
        self.assert_effect_emitted()

        # raised: the same write must ship as a query, and NOT as an effect.
        # assert_replicated_verbatim checks both halves -- skipping the effect
        # check instead would pass either way.
        self.set_effects_config(threshold=999999)
        self.query_and_sync("CREATE (:T {a: 2})")
        self.assert_replicated_verbatim()
        self.assert_graph_eq()

        self.set_effects_config(threshold=0)

    def test02_threshold_restored_resumes_effects(self):
        # the pair to test01: a disable that cannot be undone would pass test01
        # and leave every later test measuring the query path
        self.set_effects_config(threshold=0)
        self.query_and_sync("CREATE (:T {a: 3})")
        self.assert_effect_emitted()
        self.assert_graph_eq()


class test02_ThresholdIgnoredV3(_EffectsBase):
    """v3 only: the threshold is inert, and that is asserted rather than assumed.

    This is the test the original suite could not have: it pins the *difference*
    between the versions rather than working around it. If v3 ever starts
    honouring the threshold, this fails and says so.
    """

    GRAPH_ID = "effects_cmd_v3"

    def __init__(self):
        self._setup(version=3, threshold=0)
        self.start_monitor("GRAPH.EFFECT", "GRAPH.QUERY")

    def test01_threshold_does_not_disable_v3(self):
        self.env.assertFalse(WIRE[3]["threshold_disables"])

        self.set_effects_config(threshold=999999)
        self.query_and_sync("CREATE (:T {a: 1})")

        # still an effect, and still a v3 one
        window = self.assert_effect_emitted()
        self.assert_wire_version(window, version=3)
        self.assert_graph_eq()

        self.set_effects_config(threshold=0)
