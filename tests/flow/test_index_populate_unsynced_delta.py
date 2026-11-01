from common import *
from index_utils import *

GRAPH_ID = "index_populate_unsynced_delta"

# Regression test for a batch-resume bug in index population, for both node
# and edge indexes.
#
# _Index_PopulateNodeIndex / _Index_PopulateEdgeIndex (src/index/index_construct.c)
# scan a label/relation Delta_Matrix in batches, releasing the graph read
# lock between batches. Node population resumes the next batch with
# `rowIdx = id + 1`, relying on the comment "this is true because we're
# iterating over a diagonal matrix" - i.e. it assumes entity ids stream out
# in one globally ascending order. Edge population resumes similarly, keyed
# off the relation matrix's row (source id).
#
# That assumption only holds if the label/relation matrix has no pending
# delta-plus (DP) entries: Delta_MatrixTupleIter_next_BOOL/_UINT64
# (src/graph/delta_matrix/delta_matrix_iter.c) drains the *entire* main
# matrix (M) across an attached row range before it ever yields from DP, so
# within one attached range the id stream is "all of M ascending, then all
# of DP ascending" - not one merged ascending stream. If a batch boundary
# lands while DP still holds entries whose row is below the batch's resume
# point, the next batch re-attaches at [last_row + 1, MAX) and those DP rows
# are excluded from every later range - permanently, not just delayed.
#
# The cleanest way to land DP entries below a later batch boundary doesn't
# need any deletion: label/connect a high range of pre-existing node ids
# first (flushed into M by a subsequent touch), then a low range of ids
# second (stays in delta-plus, far below DELTA_MAX_PENDING_CHANGES). The
# first population batch drains part of M - which starts above the low
# range - without ever reaching delta-plus, and resumes past the low range
# forever.
#
# Until v4.20.3 the RDB decoder called Graph_ApplyAllPending(g, true) before
# constructing indexes, forcing a full DP/DM -> M merge so the matrix was
# always fully synced by the time population ran, which incidentally masked
# this pre-existing bug on every DEBUG RELOAD / replica full sync. v4.20.4
# ("remove full sync from decoder") correctly stopped force-merging on
# decode - the decoder is expected to preserve M/DP/DM as encoded rather
# than merge them (see
# test_encode_decode.py::test_18_encode_decode_preserves_pending_deltas) -
# which removed that incidental masking.
#
# Node case verified against the released images directly (single node, no
# replication involved):
#   falkordb/falkordb:v4.20.1 - CREATE INDEX against the live un-synced
#     graph already undercounts (this bug predates the decoder change);
#     DEBUG RELOAD repairs it (the old forced flush runs before population).
#   falkordb/falkordb:v4.20.4 - same live undercount, and DEBUG RELOAD no
#     longer repairs it - the freshly reloaded graph populates the index
#     against the same kind of un-synced state.
#
# Edge case: _Index_PopulateEdgeIndex's do-while batching condition
# (`(indexed < batch_size || (prev_src_id == src_id && prev_dest_id == dest_id))`)
# has a tautology - prev_src_id/prev_dest_id are set equal to src_id/dest_id
# on the immediately preceding lines, so the right-hand disjunct is always
# true at the point it's evaluated, and the loop never actually stops at
# batch_size; it only stops when the relation-matrix range is genuinely
# exhausted. That accidentally prevents this same bug from ever surfacing
# as missing entries in practice (population never resumes a partial
# range), at the cost of never releasing the read lock either - a large
# relation type populates in one uninterrupted pass, stalling concurrent
# writers for the full duration (measured: ~1.5s of continuous stall
# indexing 1,000,000 edges, vs 5ms once real batching is restored).
#
# NOTE - there used to be a separate bug layered on top of this: repopulating
# an index via DEBUG RELOAD dropped the entity whose indexed numeric value was
# 0. It was originally misattributed to "the smallest id" because here the
# smallest id (`v == 0`) is also the only value-0 entity. The real root cause
# was a union-aliasing field-skip in RediSearch - a numeric +0.0 is all-zero
# bytes, so the field's value pointer reads as NULL and the field was skipped
# during the post-reload populate - fixed in FalkorDB/RediSearch#31. With that
# fix `v == 0` is indexed like any other value, so the post-reload counts below
# are asserted at the full `total`. test_03 / test_04 are the focused
# regression for that numeric-0 fix.


def _build_node_delta_gap(graph, label, prop, spare_prop, total=100000, gap=1000):
    # `total` plain, unlabeled nodes, ids 0..total-1 via the `v` property.
    graph.query(f"UNWIND range(0, {total - 1}) AS i CREATE (:Base {{v: i}})")

    # label everything except v in [1, gap] with `label` (i.e. v == 0 or
    # v > gap) - this is the only labeled content so far, flushed into M by
    # the touch query below.
    graph.query(
        f"MATCH (n:Base) WHERE n.v = 0 OR n.v > {gap} "
        f"SET n:{label}, n.{prop} = n.v, n.{spare_prop} = n.v"
    )

    # a write that touches the label matrix but matches nothing: forces
    # Delta_Matrix_synchronize to run, and DP (total - gap entries) is over
    # DELTA_MAX_PENDING_CHANGES (10,000 by default), so the additions flush
    # into M.
    graph.query(f"MATCH (n:{label}) WHERE n.{prop} = -1 SET n.touch = 1")

    # label v in [1, gap] second - far below DELTA_MAX_PENDING_CHANGES,
    # these stay unflushed in delta-plus indefinitely. Their row is below
    # where M now starts (gap + 1), so the first population batch - which
    # drains part of M without ever reaching delta-plus - resumes past them
    # for good.
    graph.query(
        f"MATCH (n:Base) WHERE n.v >= 1 AND n.v <= {gap} "
        f"SET n:{label}, n.{prop} = n.v, n.{spare_prop} = n.v"
    )


def _build_edge_delta_gap(graph, base_label, reltype, prop, spare_prop,
                           total=100000, gap=1000):
    # `total` plain nodes, used as both endpoints of self-loop edges - row in
    # the relation matrix is the source node's internal id, exposed here via
    # the `v` property.
    graph.query(f"UNWIND range(0, {total - 1}) AS i CREATE (:{base_label} {{v: i}})")

    # self-loop edges sourced from v == 0 or v > gap - the only
    # relation-matrix content so far.
    graph.query(
        f"MATCH (a:{base_label}) WHERE a.v = 0 OR a.v > {gap} "
        f"CREATE (a)-[:{reltype} {{{prop}: a.v, {spare_prop}: a.v}}]->(a)"
    )

    # a write that touches the relation matrix but matches nothing: forces
    # the flush of the (total - gap) pending additions into M, same
    # mechanism as the node case.
    graph.query(f"MATCH ()-[e:{reltype}]->() WHERE e.{prop} = -1 SET e.touch = 1")

    # self-loop edges sourced from v in [1, gap] second - stays unflushed in
    # delta-plus. Their row is below where M now starts, so the first
    # population batch resumes past them for good.
    graph.query(
        f"MATCH (a:{base_label}) WHERE a.v >= 1 AND a.v <= {gap} "
        f"CREATE (a)-[:{reltype} {{{prop}: a.v, {spare_prop}: a.v}}]->(a)"
    )


class testIndexPopulateUnsyncedDelta():
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)
        self.redis_con = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        try:
            self.graph.delete()
        except Exception as e:
            msg = str(e).lower()
            if "empty key" not in msg and "connection" not in msg:
                raise

    def test_01_node_index_misses_delta_plus_rows_below_batch_boundary(self):
        g = self.graph
        label = "L"
        prop, spare_prop = "expired", "expired2"
        total, gap = 100000, 1000

        _build_node_delta_gap(g, label, prop, spare_prop, total, gap)

        labeled_total = g.query(f"MATCH (n:{label}) RETURN count(n)").result_set[0][0]
        self.env.assertEquals(labeled_total, total)

        #----------------------------------------------------------------
        # path 1: index created directly against the live, un-synced graph
        #----------------------------------------------------------------
        create_node_range_index(g, label, prop, sync=True)

        plan = str(g.explain(f"MATCH (n:{label}) WHERE n.{prop} >= 0 RETURN count(n)"))
        self.env.assertIn("Node By Index Scan", plan)

        indexed = g.query(
            f"MATCH (n:{label}) WHERE n.{prop} >= 0 RETURN count(n)"
        ).result_set[0][0]
        self.env.assertEquals(indexed, labeled_total)

        low_range_indexed = g.query(
            f"MATCH (n:{label}) WHERE n.{prop} >= 1 AND n.{prop} <= {gap} RETURN count(n)"
        ).result_set[0][0]
        self.env.assertEquals(low_range_indexed, gap)

        # dropping and recreating the index in the same delta state is not a
        # repair - population runs again against the same un-synced matrix
        g.query(f"DROP INDEX FOR (n:{label}) ON (n.{prop})")
        create_node_range_index(g, label, prop, sync=True)

        indexed = g.query(
            f"MATCH (n:{label}) WHERE n.{prop} >= 0 RETURN count(n)"
        ).result_set[0][0]
        self.env.assertEquals(indexed, labeled_total)

        #----------------------------------------------------------------
        # path 2: DEBUG RELOAD - the decoder now preserves the M/DP split
        # (see test_18_encode_decode_preserves_pending_deltas) instead of
        # merging it, so the existing index is repopulated against the same
        # kind of un-synced state on load
        #----------------------------------------------------------------
        self.redis_con.execute_command("DEBUG", "RELOAD")

        labeled_total = g.query(f"MATCH (n:{label}) RETURN count(n)").result_set[0][0]
        self.env.assertEquals(labeled_total, total)

        # value 0 (v == 0, in the high/M range) is now indexed across reload
        # too - see the module docstring note and test_03 (FalkorDB/RediSearch#31)
        indexed = g.query(
            f"MATCH (n:{label}) WHERE n.{prop} >= 0 RETURN count(n)"
        ).result_set[0][0]
        self.env.assertEquals(indexed, labeled_total)

        low_range_indexed = g.query(
            f"MATCH (n:{label}) WHERE n.{prop} >= 1 AND n.{prop} <= {gap} RETURN count(n)"
        ).result_set[0][0]
        self.env.assertEquals(low_range_indexed, gap)

        # path 3: a brand new index, created for the first time after the
        # reload, populated purely from the decoder-restored (un-synced)
        # state - no prior live population involved at all
        create_node_range_index(g, label, spare_prop, sync=True)

        indexed_spare = g.query(
            f"MATCH (n:{label}) WHERE n.{spare_prop} >= 1 AND n.{spare_prop} <= {gap} "
            f"RETURN count(n)"
        ).result_set[0][0]
        self.env.assertEquals(indexed_spare, gap)

    def test_02_edge_index_misses_delta_plus_rows_below_batch_boundary(self):
        g = self.graph
        base_label = "B"
        reltype = "E"
        prop, spare_prop = "expired", "expired2"
        total, gap = 100000, 1000

        _build_edge_delta_gap(g, base_label, reltype, prop, spare_prop, total, gap)

        total_edges = g.query(
            f"MATCH ()-[e:{reltype}]->() RETURN count(e)"
        ).result_set[0][0]
        self.env.assertEquals(total_edges, total)

        #----------------------------------------------------------------
        # path 1: index created directly against the live, un-synced graph
        #----------------------------------------------------------------
        create_edge_range_index(g, reltype, prop, sync=True)

        plan = str(g.explain(
            f"MATCH ()-[e:{reltype}]->() WHERE e.{prop} >= 0 RETURN count(e)"
        ))
        self.env.assertIn("Edge By Index Scan", plan)

        indexed = g.query(
            f"MATCH ()-[e:{reltype}]->() WHERE e.{prop} >= 0 RETURN count(e)"
        ).result_set[0][0]
        self.env.assertEquals(indexed, total_edges)

        low_range_indexed = g.query(
            f"MATCH ()-[e:{reltype}]->() WHERE e.{prop} >= 1 AND e.{prop} <= {gap} "
            f"RETURN count(e)"
        ).result_set[0][0]
        self.env.assertEquals(low_range_indexed, gap)

        # dropping and recreating the index in the same delta state is not a
        # repair - population runs again against the same un-synced matrix
        g.query(f"DROP INDEX FOR ()-[e:{reltype}]-() ON (e.{prop})")
        create_edge_range_index(g, reltype, prop, sync=True)

        indexed = g.query(
            f"MATCH ()-[e:{reltype}]->() WHERE e.{prop} >= 0 RETURN count(e)"
        ).result_set[0][0]
        self.env.assertEquals(indexed, total_edges)

        #----------------------------------------------------------------
        # path 2: DEBUG RELOAD
        #----------------------------------------------------------------
        self.redis_con.execute_command("DEBUG", "RELOAD")

        total_edges = g.query(
            f"MATCH ()-[e:{reltype}]->() RETURN count(e)"
        ).result_set[0][0]
        self.env.assertEquals(total_edges, total)

        # value 0 (v == 0, in the high/M range) is now indexed across reload
        # too - see the module docstring note and test_04 (FalkorDB/RediSearch#31)
        indexed = g.query(
            f"MATCH ()-[e:{reltype}]->() WHERE e.{prop} >= 0 RETURN count(e)"
        ).result_set[0][0]
        self.env.assertEquals(indexed, total_edges)

        low_range_indexed = g.query(
            f"MATCH ()-[e:{reltype}]->() WHERE e.{prop} >= 1 AND e.{prop} <= {gap} "
            f"RETURN count(e)"
        ).result_set[0][0]
        self.env.assertEquals(low_range_indexed, gap)

        # path 3: a brand new edge index, created for the first time after
        # reload, populated purely from the decoder-restored state
        create_edge_range_index(g, reltype, spare_prop, sync=True)

        indexed_spare = g.query(
            f"MATCH ()-[e:{reltype}]->() WHERE e.{spare_prop} >= 1 AND e.{spare_prop} <= {gap} "
            f"RETURN count(e)"
        ).result_set[0][0]
        self.env.assertEquals(indexed_spare, gap)

    def test_03_node_numeric_zero_survives_reload(self):
        # Focused regression for FalkorDB/RediSearch#31: a range-indexed numeric
        # property equal to 0 (int 0 / float +0.0) must stay in the index across
        # an RDB reload (the restart / failover / replica full-sync path, where
        # the index is rebuilt from the decoded graph). Before the fix the
        # document was created on reload but its numeric record was never written,
        # so the node vanished from index scans while remaining in the graph.
        g      = self.graph
        label  = "Z"
        prop   = "val"

        # int 0 and float 0.0 (both affected), plus non-zero controls
        g.query(f"CREATE (:{label} {{{prop}: 0}}), (:{label} {{{prop}: 7}}), "
                f"(:{label} {{{prop}: 0.0}}), (:{label} {{{prop}: -3}})")
        create_node_range_index(g, label, prop, sync=True)

        q_zero = f"MATCH (n:{label} {{{prop}: 0}}) RETURN count(n)"
        self.env.assertIn("Node By Index Scan", str(g.explain(q_zero)))

        def zero_via_index():
            return g.query(q_zero).result_set[0][0]

        # before reload: both the int 0 and the float 0.0 node match
        self.env.assertEquals(zero_via_index(), 2)

        # reload rebuilds the index from RDB - value 0 must survive
        self.redis_con.execute_command("DEBUG", "RELOAD")
        self.env.assertEquals(zero_via_index(), 2)

        # a second reload must remain stable (not an accumulating drift)
        self.redis_con.execute_command("DEBUG", "RELOAD")
        self.env.assertEquals(zero_via_index(), 2)

        # and a brand new index, created fresh after the reload, must also
        # index value 0
        g.query(f"DROP INDEX FOR (n:{label}) ON (n.{prop})")
        create_node_range_index(g, label, prop, sync=True)
        self.env.assertEquals(zero_via_index(), 2)

    def test_04_edge_numeric_zero_survives_reload(self):
        # Edge-index counterpart of test_03.
        g          = self.graph
        base_label = "ZB"
        reltype    = "ZE"
        prop       = "val"

        g.query(f"CREATE (a:{base_label}) "
                f"CREATE (a)-[:{reltype} {{{prop}: 0}}]->(a), "
                f"(a)-[:{reltype} {{{prop}: 5}}]->(a), "
                f"(a)-[:{reltype} {{{prop}: 0.0}}]->(a)")
        create_edge_range_index(g, reltype, prop, sync=True)

        q_zero = f"MATCH ()-[e:{reltype}]->() WHERE e.{prop} = 0 RETURN count(e)"
        self.env.assertIn("Edge By Index Scan", str(g.explain(q_zero)))

        def zero_via_index():
            return g.query(q_zero).result_set[0][0]

        # before reload: both the int 0 and the float 0.0 edge match
        self.env.assertEquals(zero_via_index(), 2)

        # reload rebuilds the index from RDB - value 0 must survive
        self.redis_con.execute_command("DEBUG", "RELOAD")
        self.env.assertEquals(zero_via_index(), 2)

    def test_05_node_boolean_false_survives_reload(self):
        # A boolean `false` is indexed through the same numeric path as 0
        # (SI_GET_NUMERIC(false) == 0), so its document-field bytes are all-zero
        # and it hit the exact same union-aliasing skip - a node with an indexed
        # `false` was dropped from the index on reload before the fix.
        g     = self.graph
        label = "BZ"
        prop  = "flag"

        g.query(f"CREATE (:{label} {{{prop}: false}}), (:{label} {{{prop}: true}}), "
                f"(:{label} {{{prop}: false}})")
        create_node_range_index(g, label, prop, sync=True)

        q_false = f"MATCH (n:{label} {{{prop}: false}}) RETURN count(n)"
        self.env.assertIn("Node By Index Scan", str(g.explain(q_false)))

        def false_via_index():
            return g.query(q_false).result_set[0][0]

        self.env.assertEquals(false_via_index(), 2)
        self.redis_con.execute_command("DEBUG", "RELOAD")
        self.env.assertEquals(false_via_index(), 2)

    def test_06_edge_boolean_false_survives_reload(self):
        # Edge-index counterpart of test_05.
        g          = self.graph
        base_label = "BB"
        reltype    = "BE"
        prop       = "flag"

        g.query(f"CREATE (a:{base_label}) "
                f"CREATE (a)-[:{reltype} {{{prop}: false}}]->(a), "
                f"(a)-[:{reltype} {{{prop}: true}}]->(a), "
                f"(a)-[:{reltype} {{{prop}: false}}]->(a)")
        create_edge_range_index(g, reltype, prop, sync=True)

        q_false = f"MATCH ()-[e:{reltype}]->() WHERE e.{prop} = false RETURN count(e)"
        self.env.assertIn("Edge By Index Scan", str(g.explain(q_false)))

        def false_via_index():
            return g.query(q_false).result_set[0][0]

        self.env.assertEquals(false_via_index(), 2)
        self.redis_con.execute_command("DEBUG", "RELOAD")
        self.env.assertEquals(false_via_index(), 2)
