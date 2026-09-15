from common import *
from packaging.version import Version

GRAPH_ID = "slowlog_test"

# Rows in each `populate_slowlog` query, and in the queries that have to
# out-rank the entries it leaves behind.
#
# These are expressed as one constant and a multiple of it on purpose. They used
# to be independent literals, and #2628 raised the populate size tenfold
# (250,000 -> 2,500,000) without touching the queries that had to beat it, which
# silently removed most of their headroom and left three comments describing the
# old size. Tying them together makes that drift impossible rather than merely
# unlikely.
POPULATE_ROWS = 2_500_000
DISPLACER_ROWS = POPULATE_ROWS * 10


def slow_query(rows, divisor):
    """A query whose cost is proportional to `rows`.

    Every query in this file that must be slower than a `populate_slowlog`
    entry is built from this one shape, so the ratio between them is fixed by
    row count alone. A differently-shaped query (the old nested `UNWIND ...
    RETURN SUM(i + j)`) leaves the margin at the mercy of whichever expression
    path the engine optimizes next.
    """
    return (f"UNWIND range(0, {rows}) AS x "
            f"WITH x WHERE x % {divisor} = 0 RETURN count(x)")

class testSlowLog():
    def __init__(self):
        self.env, self.db = Env()
        self.redis_con = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def populate_slowlog(self, n):
        """Fill the slowlog with `n - 1` distinct entries, run SERIALLY.

        These used to run concurrently through an async pool, and that is what
        made the tests below flaky. A slowlog entry records the query's own
        latency, so 19 queries competing for the same cores recorded times far
        above their actual cost: measured under coverage instrumentation, each
        query does ~105ms of work but the retained entries read 5,600-8,771ms —
        a ~37x inflation. Every later query in this file has to out-rank those
        entries while running *alone*, so it was racing an inflation factor
        that depends on the runner's core count and on whether instrumentation
        is enabled. No choice of query size is safe against that.

        Run serially, an entry's recorded latency is its own work (98.9-154.5ms
        under the same instrumentation), which is a bar a query can be sized
        against. It is also faster in wall-clock than the contended version,
        because the concurrency was never buying throughput here.

        Nothing asserts on concurrent behaviour; the assertions are about
        slowlog semantics (how many entries are kept, which ones, and that
        repeated reads agree). Concurrency belongs in tests/test_concurrency.py.

        POPULATE_ROWS is sized to run an order of magnitude past the slowlog's
        10ms floor (SLOW_LOG_MIN_REQ_LATENCY). range(0, 250000) used to measure
        ~11ms, which stopped qualifying once `WHERE x % i = 0` moved onto the
        columnar expression path and dropped to ~4.7ms — the entries silently
        stopped being logged and the assertions started reading 2 instead of 10.
        """
        g = self.db.select_graph(GRAPH_ID)
        for i in range(1, n):
            g.query(slow_query(POPULATE_ROWS, i))

    def test01_slowlog(self):
        # Slowlog should fail when graph doesn't exists
        try:
            slowlog = self.redis_con.execute_command("GRAPH.SLOWLOG", "NONE_EXISTING_GRAPH")
        except ResponseError as e:
            self.env.assertContains("Invalid graph operation on empty key", str(e))

        # issue the same query twice
        # the range is sized to run an order of magnitude past the slowlog's
        # 10ms floor (SLOW_LOG_MIN_REQ_LATENCY). range(0, 500000) used to sit
        # right on it — ~12ms cold but ~7.9ms once warm, and 8.8-13.8ms under
        # the load `--parallelism` creates — so whether the entry got logged
        # at all was a coin flip. range(0, 5000000) measures 70-84ms.
        q = "UNWIND range (0, 5000000) AS x RETURN max(x)"
        self.graph.query(q)
        self.graph.query(q)

        # slow log should contain a single entry, no duplicates
        slowlog = self.graph.slowlog()
        self.env.assertEqual(len(slowlog), 1)

        # saturate slowlog
        self.populate_slowlog(20)
        A = self.graph.slowlog()
        B = self.graph.slowlog()

        # calling slowlog multiple times should preduce the same result
        self.env.assertEqual(A, B)
        self.env.assertEqual(len(A), 10)

        server = self.redis_con.info("Server")
        if Version(server["redis_version"]) < Version("6.2.0"):
            # redis < 6.2.0 not support slowlog time measure
            return

        # Issue a long running query, this should replace an existing entry in
        # the slowlog.
        #
        # Displacement evicts the *fastest* retained entry, so this only has to
        # beat that one — but it has to beat it reliably. Built from the same
        # shape as the populate queries at ten times the rows, the margin is
        # fixed by row count and cannot be tipped by an optimization to one
        # expression path. Measured under coverage instrumentation: the fastest
        # retained entry is ~99ms and this query runs ~1,443ms, a ~14x margin.
        # The previous nested-UNWIND query measured ~377ms against a contended
        # bar of ~5,600ms, i.e. 0.07x — it could not displace anything.
        q = slow_query(DISPLACER_ROWS, 1)

        self.graph.query(q)
        B = self.graph.slowlog()

        self.env.assertNotEqual(A, B)

        # get redis slowlog
        slowlog = self.redis_con.slowlog_get()
        slowlog_commands = [log["command"].decode('ascii') for log in slowlog]

        # validate the command added to redis slowlog
        self.env.assertGreater(len(slowlog), 0)
        self.env.assertContains(f"GRAPH.QUERY slowlog_test {q} --compact", slowlog_commands)

    def test02_slowlog_reset(self):
        # reset none existing slowlog
        try:
            slowlog = self.redis_con.execute_command("GRAPH.SLOWLOG", "NONE_EXISTING_GRAPH", "RESET")
        except ResponseError as e:
            self.env.assertContains("Invalid graph operation on empty key", str(e))

        # issue an unknown slowlog sub command
        try:
            slowlog = self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID, "UNKNOW_SUB_CMD")
        except ResponseError as e:
            self.env.assertContains("Unknown subcommand", str(e))

        # populate slowlog
        self.populate_slowlog(36)
        slowlog = self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID)
        self.env.assertGreater(len(slowlog), 0)

        # clear slowlog
        # make sure there's no harm in double reseting
        self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID, "RESET")
        self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID, "RESET")

        # expecting an empty slowlog
        slowlog = self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID)
        self.env.assertEqual(len(slowlog), 0)

        # make sure slowlog repopulates after RESET
        self.populate_slowlog(36)
        slowlog = self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID)
        self.env.assertGreater(len(slowlog), 0)

    def test03_cap_entry(self):
        # make sure slowlog entries are capped

        # clear slowlog
        self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID, "RESET")

        #-----------------------------------------------------------------------
        # truncated query
        #-----------------------------------------------------------------------

        # NOTE: the query body must be heavy enough to deterministically exceed
        # the slowlog MIN_LATENCY_MS (10ms) threshold even on a fast engine /
        # under coverage. A single UNWIND range(0, 200000) dropped below 10ms
        # once the engine got faster, leaving the slowlog empty.
        #
        # Unlike test01 and test06 this is measured against a *fixed* 10ms
        # floor rather than against other entries, so it does not need to
        # out-rank anything: at ~377ms under instrumentation it has ~37x
        # headroom on a threshold that does not move. Kept as the nested-UNWIND
        # body it has always had; test01 no longer uses that shape.
        long_string = 'a' * 4000
        query = f"WITH '{long_string}' AS str UNWIND range(0, 2500) AS i UNWIND range(0, 2500) AS j WITH i, j WHERE i > 0 AND j < 500 RETURN SUM(i + j)"
        self.graph.query(query)

        slowlog = self.graph.slowlog()
        entry = slowlog[0]
        cmd     = entry[1]
        q       = entry[2]
        latency = entry[3]
        params  = entry[4]

        self.env.assertEqual(cmd, "GRAPH.QUERY")
        self.env.assertEqual(params, None)

        # reported query should be truncated
        self.env.assertContains("...", q)
        self.env.assertLess(len(q), len(query))

        #-----------------------------------------------------------------------
        # truncated params
        #-----------------------------------------------------------------------

        # clear slowlog
        self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID, "RESET")

        query = "WITH $long_string AS str UNWIND range(0, 2500) AS i UNWIND range(0, 2500) AS j WITH i, j WHERE i > 0 AND j < 500 RETURN SUM(i + j)"
        self.graph.query(query, {'long_string': long_string})

        slowlog = self.graph.slowlog()
        entry = slowlog[0]
        cmd     = entry[1]
        q       = entry[2]
        latency = entry[3]
        params  = entry[4]

        self.env.assertEqual(cmd, "GRAPH.QUERY")
        self.env.assertEqual(query, q)
        self.env.assertContains("...", params)

        # reported param should be truncated
        self.env.assertLess(len(params), len(long_string))

        #-----------------------------------------------------------------------
        # truncated query & params
        #-----------------------------------------------------------------------

        # clear slowlog
        self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID, "RESET")

        query = f"WITH $long_string as long_param, '{long_string}' AS long_string UNWIND range(0, 2500) AS i UNWIND range(0, 2500) AS j WITH i, j WHERE i > 0 AND j < 500 RETURN SUM(i + j)"
        self.graph.query(query, {'long_string': long_string})

        slowlog = self.graph.slowlog()
        entry = slowlog[0]
        cmd     = entry[1]
        q       = entry[2]
        latency = entry[3]
        params  = entry[4]

        self.env.assertEqual(cmd, "GRAPH.QUERY")

        # reported query should be truncated
        self.env.assertContains("...", q)
        self.env.assertLess(len(q), len(query))

        # reported param should be truncated
        self.env.assertContains("...", params)
        self.env.assertLess(len(params), len(long_string))

    def test04_same_query_diff_params(self):
        # make sure no new entries are added when the query remains the same
        # but the params change

        # clear slowlog
        self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID, "RESET")

        # $i drives how many rows the second UNWIND produces, so the same query
        # TEXT (one slowlog entry) does ~10x the work on the second run below.
        # The entry's params are only refreshed on a strictly greater latency
        # (slow_log.rs MIN_LATENCY_MS path; C's slow_log.c returns early when
        # latency <= existing), so run 2 must be the slower of the two. The
        # previous body differed by only ~31ms on a ~400ms baseline and lost
        # that ordering to scheduling noise under --parallelism, failing ~1 run
        # in 12; this pair keeps a ~700ms gap even on a fully loaded machine.
        query = "UNWIND range(0, 500000) AS x UNWIND range(0, $i) AS y RETURN SUM(x + y)"
        self.graph.query(query, {'i': 4})

        slowlog = self.graph.slowlog()
        self.env.assertEqual(len(slowlog), 1)

        entry = slowlog[0]
        q0 = entry[2]
        latency0 = entry[3]
        p0 = entry[4]

        # re-issue the same query but with different params
        query = "UNWIND range(0, 500000) AS x UNWIND range(0, $i) AS y RETURN SUM(x + y)"
        self.graph.query(query, {'i': 49})

        slowlog = self.graph.slowlog()
        self.env.assertEqual(len(slowlog), 1)

        entry = slowlog[0]
        q1 = entry[2]
        latency1 = entry[3]
        p1 = entry[4]

        # expecting the same query
        self.env.assertEqual(q0, q1)

        # expecting params to update
        self.env.assertNotEqual(p0, p1)
        self.env.assertContains('49', p1)

    def test05_fast_queries(self):
        # make sure fast queries do not enter the slowlog

        # clear slowlog
        self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID, "RESET")

        # query too fast for slowlog
        q = "RETURN 1"
        self.graph.query(q)

        slowlog = self.graph.slowlog()
        self.env.assertEqual(len(slowlog), 0)

    def test06_force_replace(self):
        # make sure slowlog entries get replcaed

        # clear slowlog
        self.redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID, "RESET")

        # fill slowlog
        self.populate_slowlog(20)
        entries = self.graph.slowlog()

        # expecting 10 entries
        self.env.assertEqual(len(entries), 10)

        # issue 2 slower queries
        # expecting to have them replace existing entries
        #
        # Same reasoning as test01: built from the populate shape at ten times
        # the rows, so they out-rank the entries by row count rather than by
        # luck. The divisors differ only to make the two query texts distinct,
        # since the slowlog keys on text.
        q0 = slow_query(DISPLACER_ROWS, 2)
        self.graph.query(q0)

        q1 = slow_query(DISPLACER_ROWS, 3)
        self.graph.query(q1)

        entries = self.graph.slowlog()

        # expecting 10 entries
        self.env.assertEqual(len(entries), 10)

        # make sure both q0 & q1 are in entries
        queries = [entry[2] for entry in entries]
        self.env.assertContains (q0, queries)
        self.env.assertContains (q1, queries)
