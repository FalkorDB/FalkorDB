import time
from common import *
from graph_utils import graph_eq

GRAPH_ID = "effects_compression"

# The acceptance test for v3 payload compression: a C master emits a compressed
# v3 payload and a C replica inflates and applies it.
#
# READ THIS BEFORE WEAKENING ANY ASSERTION HERE. The obvious version of this
# test - "the replica ends up with the right data, and the compressed run moved
# fewer bytes" - passes with EITHER HOOK DELETED. Both were verified by
# deleting them:
#
#   * delete the READ hook and the replica refuses the payload, diverges, and
#     takes a FULL RESYNC. The resync restores the data, so the node count and
#     graph_eq both pass, and the resync moved 176,000 bytes against 215,292 -
#     still fewer, so a direction-only byte assertion passed too.
#   * delete the WRITE hook and the two runs move 215,269 and 215,292 bytes.
#     A direction-only assertion passed on 23 bytes of NOISE.
#
# So correctness of the replica proves nothing here: divergence is self-healing
# and hides exactly the failure this test exists to catch. Three mechanism
# checks are what give it teeth, and each one fails on a different break:
#
#   1. failed_calls on the replica's GRAPH.EFFECT must not move - a refused
#      effect increments it
#   2. sync_full on the MASTER must not move - a refusal forces a resync, and
#      this is what catches divergence being papered over
#   3. the compressed run must move a FRACTION of the bytes, not merely fewer
#
# Deliberately NOT asserted: a specific ratio. zstd level 1's ratio on effects
# payloads is a coin flip on record alignment - the same body measures 10,399
# bytes at one alignment and ~31,100 at the other eight - so a tight threshold
# would pin one payload's luck rather than a property of the format. The
# measured ratio here is ~35x; the assertion is a floor of 4x, which clears the
# worst alignment by a wide margin and still fails both breaks above.
class testEffectsCompression():
    def __init__(self):
        self.env, self.db = Env(env='oss', useSlaves=True)
        self.master  = self.env.getConnection()
        self.replica = self.env.getSlaveConnection()
        self.mg = Graph(self.master,  GRAPH_ID)
        self.rg = Graph(self.replica, GRAPH_ID)

        # replicate via effects, and emit v3
        self.db.config_set("EFFECTS_THRESHOLD", 0)
        self.db.config_set("EFFECTS_VERSION", 3)

        # block until the link is up. A replica still in wait_bgsave answers
        # queries with an empty graph, which looks exactly like broken
        # replication and has misled people here before
        self._wait_for_link()

    def _wait_for_link(self):
        for _ in range(200):
            if self.replica.info().get("master_link_status") == "up":
                return
            time.sleep(0.05)
        raise AssertionError("replica link never came up")

    # GRAPH.EFFECT is counted on the REPLICA, not the master - the master runs
    # GRAPH.QUERY. Reading the master's commandstats concludes effects are off
    # when they are on
    def _effect_stats(self):
        row = self.replica.info("commandstats").get("cmdstat_graph.EFFECT")
        if row is None:
            return (0, 0)
        return (row["calls"], row["failed_calls"])

    def _sync_full(self):
        return self.master.info("stats")["sync_full"]

    def _replica_bytes_in(self):
        return self.replica.info("stats")["total_net_input_bytes"]

    def _wait_for_count(self, n, q="MATCH (n:L) RETURN count(n)"):
        # GRAPH.QUERY is a write command and refused on a read-only replica, so
        # this reads with ro_query.
        #
        # AND THE GRAPH KEY MAY NOT EXIST ON THE REPLICA YET. `master_link_status`
        # being "up" says the connection is established, not that any write has
        # replicated - the key appears only once the first one lands. Querying a
        # graph that is not there raises "Invalid graph operation on empty key"
        # rather than returning 0, so without this the test failed roughly half
        # the time, and failed as an ERROR rather than as an assertion, which
        # made it look like a product fault instead of a race in the test.
        last = 0
        for _ in range(200):
            try:
                last = self.rg.ro_query(q).result_set[0][0]
                if last == n:
                    return n
            except redis.exceptions.ResponseError as e:
                # only the not-yet-replicated case is tolerated; anything else
                # is a real failure and must not be swallowed by a poll loop
                if "empty key" not in str(e):
                    raise
                last = 0
            time.sleep(0.05)
        return last

    # the stats-wrapped write both tests share: run q on the master, let
    # settle() wait for the replica and check what landed, and hold the three
    # mechanism checks around it. Returns the bytes the replica read.
    def _measured(self, min_bytes, q, settle):
        self.db.config_set("EFFECTS_COMPRESSION", min_bytes)

        calls0, failed0 = self._effect_stats()
        sync0           = self._sync_full()
        bytes0          = self._replica_bytes_in()

        settle(self.mg.query(q))
        bytes1 = self._replica_bytes_in()

        calls1, failed1 = self._effect_stats()

        # the replica applied it through the effects path
        self.env.assertGreater(calls1, calls0)

        # and did NOT refuse it. Without this, a refusal followed by a resync
        # looks identical to success
        self.env.assertEquals(failed1, failed0)

        # and no full resync happened, which is the other way a refusal hides
        self.env.assertEquals(self._sync_full(), sync0)

        return bytes1 - bytes0

    def _write_run(self, min_bytes, n):
        q = f"UNWIND range(1, {n}) AS i CREATE (:L {{v: i, s: 'padpadpadpadpadpad'}})"

        def settle(res):
            self.env.assertEquals(res.nodes_created, n)
            self.env.assertEquals(self._wait_for_count(n), n)

        return self._measured(min_bytes, q, settle)

    def test01_compressed_payload_round_trips(self):
        n = 5000

        uncompressed = self._write_run(0, n)

        # the same write again, this time compressed
        self.mg.delete()
        self._wait_for_link()
        compressed = self._write_run(64, n)

        print(f"\n    {n} nodes: replica read {uncompressed:,} bytes uncompressed, "
              f"{compressed:,} compressed ({uncompressed/compressed:.1f}x)")

        # a fraction, not merely fewer - see the class comment
        self.env.assertLess(compressed * 4, uncompressed)

        # and the graphs agree, so the inflated records applied correctly.
        # Necessary but NOT sufficient on its own, which is the whole point
        self.env.assertTrue(graph_eq(self.mg, self.rg))

    # A SECOND SHAPE, because test01 cannot fail on a whole class of bug.
    #
    # EffectsV3_ReaderNext ends the walk on ftell(stream) >= r->n, so after the
    # reader swaps its stream to the inflated plaintext, r->n has to be the
    # PLAINTEXT length. Left at the compressed length it stops the walk at the
    # first record boundary past it and drops the rest - silently: the records
    # that did arrive apply cleanly, so there is no refusal, no failed_calls,
    # and no resync for test01's mechanism checks to catch.
    #
    # test01 cannot reach it. Its payload is four records at offsets 0, 22, 38
    # and 54, and the fourth is the whole 215,082-byte node group, so every
    # boundary sits below the 5,880-byte compressed length and the terminator
    # is never consulted again before EOF. Measured, not assumed.
    #
    # This shape puts a boundary above it: two node groups and an edge group,
    # each large, so the second and third boundaries land tens of thousands of
    # bytes in. Verified by deleting `r->n = r->plain_len` - this test fails
    # and test01 still passes.
    def test02_every_group_in_a_compressed_payload_applies(self):
        n = 4000

        self.mg.delete()
        self._wait_for_link()

        pad = "padpadpadpadpadpad"
        q   = (f"UNWIND range(1, {n}) AS i CREATE "
               f"(:L {{v: i, s: '{pad}'}})-[:R {{w: i, t: '{pad}'}}]->(:M {{v: i, s: '{pad}'}})")

        def settle(res):
            self.env.assertEquals(res.nodes_created, 2 * n)
            self.env.assertEquals(res.relationships_created, n)
            self.env.assertEquals(
                self._wait_for_count(n, "MATCH ()-[r:R]->() RETURN count(r)"), n)

        self._measured(64, q, settle)

        # the LAST group in the payload, which is the one a short walk loses
        self.env.assertEquals(
            self.rg.ro_query("MATCH (:L)-[:R]->(:M) RETURN count(*)").result_set[0][0], n)
        self.env.assertTrue(graph_eq(self.mg, self.rg))
