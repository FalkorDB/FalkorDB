import asyncio
from common import *
from index_utils import *
from falkordb.asyncio import FalkorDB
from redis.asyncio import BlockingConnectionPool

GRAPH_ID = "timeout"


class testQueryTimeout():
    def __init__(self):
        self.env, self.db = Env(moduleArgs="TIMEOUT 1000")

        # skip test if we're running under Valgrind
        if VALGRIND or SANITIZER:
            self.env.skip() # valgrind is not working correctly with replication

        self.graph = self.db.select_graph(GRAPH_ID)

    def test01_read_write_query_timeout(self):
        query = "UNWIND range(0, 1000000) AS x WITH x AS x WHERE x = 10000 RETURN x"
        try:
            # The query is expected to timeout
            self.graph.query(query, timeout=1)
            self.env.assertTrue(False)
        except ResponseError as error:
            self.env.assertContains("Query timed out", str(error))

        try:
            # The query is expected to succeed
            self.graph.query(query, timeout=2000)
        except:
            self.env.assertTrue(False)

        query = """UNWIND range(0, 1000000) AS x CREATE (p:Person {age: x%90, height: x%200, weight: x%80})"""
        try:
            # The query is expected to succeed
            self.graph.query(query, timeout=1)
            self.env.assertTrue(True)
        except:
            self.env.assertTrue(False)

    def test02_configured_timeout(self):
        # Verify that the module-level timeout is set to the default of 0
        timeout = self.db.config_get("timeout")
        self.env.assertEquals(timeout, 1000)

        # Set a default timeout of 1 millisecond
        self.db.config_set("timeout", 1)
        timeout = self.db.config_get("timeout")
        self.env.assertEquals(timeout, 1)

        # Validate that a read query times out
        query = "UNWIND range(0,1000000) AS x WITH x AS x WHERE x = 10000 RETURN x"
        try:
            self.graph.query(query)
            self.env.assertTrue(False)
        except ResponseError as error:
            self.env.assertContains("Query timed out", str(error))

    def test03_timeout_index_scan(self):
        # set timeout to unlimited
        self.db.config_set("timeout", 0)

        create_node_range_index(self.graph, 'Person', 'age', 'height', 'weight', sync=True)

        queries = [
                # full scan
                "MATCH (a) RETURN a",
                # ID scan
                "MATCH (a) WHERE ID(a) > 20 RETURN a",
                # label scan
                "MATCH (a:Person) RETURN a",
                # single index scan
                "MATCH (a:Person) WHERE a.age > 40 RETURN a",
                # index scan + full scan
                "MATCH (a:Person), (b) WHERE a.age > 40 RETURN a, b",
                # index scan + ID scan
                "MATCH (a:Person), (b) WHERE a.age > 40 AND ID(b) > 20 RETURN a, b",
                # index scan + label scan
                "MATCH (a:Person), (b:Person) WHERE a.age > 40 RETURN a, b",
                # multi full and index scans
                "MATCH (a:Person), (b:Person), (c), (d) WHERE a.age > 40 AND b.height < 150 RETURN a,b,c,d",
                # multi ID and index scans
                "MATCH (a:Person), (b:Person), (c:Person), (d) WHERE a.age > 40 AND b.height < 150 AND ID(c) > 20 AND ID(d) > 30 RETURN a,b,c,d",
                # multi label and index scans
                "MATCH (a:Person), (b:Person), (c:Person), (d:Person) WHERE a.age > 40 AND b.height < 150 RETURN a,b,c,d",
                # multi index scans
                "MATCH (a:Person), (b:Person), (c:Person) WHERE a.age > 40 AND b.height < 150 AND c.weight = 50 RETURN a,b,c"
                ]

        timeouts = []

        # run each query with timeout and limit
        # expecting queries to run to completion
        for q in queries:
            q += " LIMIT 10"
            try:
                res = self.graph.query(q, timeout=5)
                timeouts.append(res.run_time_ms)
            except:
                timeouts.append(res.run_time_ms)

        for i, q in enumerate(queries):
            try:
                # query is expected to timeout
                timeout = min(max(int(timeouts[i]), 1), 10)
                res = self.graph.query(q, timeout=timeout)
                self.env.assertTrue(False)
            except ResponseError as error:
                self.env.assertContains("Query timed out", str(error))

    def test04_query_timeout_free_resultset(self):
        query = "UNWIND range(0,3000000) AS x RETURN toString(x)"

        res = None
        try:
            # The query is expected to succeed
            res = self.graph.query(query + " LIMIT 1000", timeout=3000)
        except:
            self.env.assertTrue(False)

        try:
            # The query is expected to timeout
            res = self.graph.query(query, timeout=int(res.run_time_ms))
            self.env.assertTrue(False)
        except ResponseError as error:
            self.env.assertContains("Query timed out", str(error))

    def test05_invalid_loadtime_config(self):
        try:
            env, db = Env(moduleArgs="TIMEOUT 10 TIMEOUT_DEFAULT 10 TIMEOUT_MAX 10")
            env.getConnection().ping()
            self.env.assertTrue(False)
        except:
            self.env.assertTrue(True)

    def test06_error_timeout_default_higher_than_timeout_max(self):
        self.env, self.db = Env(moduleArgs="TIMEOUT_DEFAULT 10 TIMEOUT_MAX 10")

        # get current timeout configuration
        max_timeout = self.db.config_get("TIMEOUT_MAX")
        default_timeout = self.db.config_get("TIMEOUT_DEFAULT")

        self.env.assertEquals(max_timeout, 10)
        self.env.assertEquals(default_timeout, 10)

        # try to set default-timeout to a higher value than max-timeout
        try:
            self.db.config_set("TIMEOUT_DEFAULT", max_timeout + 1)
            self.env.assertTrue(False)
        except ResponseError as error:
            self.env.assertContains("TIMEOUT_DEFAULT configuration parameter cannot be set to a value higher than TIMEOUT_MAX", str(error))

        # try to set max-timeout to a lower value then default-timeout
        try:
            self.db.config_set("TIMEOUT_MAX", default_timeout - 1)
            self.env.assertTrue(False)
        except ResponseError as error:
            self.env.assertContains("TIMEOUT_MAX configuration parameter cannot be set to a value lower than TIMEOUT_DEFAULT", str(error))

        # disable max timeout
        try:
            self.db.config_set("TIMEOUT_MAX", 0)
            self.env.assertTrue(True)
            # revert timeout_max to 10
            self.db.config_set("TIMEOUT_MAX", 10)
        except ResponseError as error:
            self.env.assertTrue(False)

        # disable default timeout
        try:
            self.db.config_set("TIMEOUT_DEFAULT", 0)
            self.env.assertTrue(True)
            # revert timeout_default to 5
            self.db.config_set("TIMEOUT_DEFAULT", 5)
        except ResponseError as error:
            self.env.assertTrue(False)

    def test07_read_write_query_timeout_default(self):
        queries = [
            "UNWIND range(0,1000000) AS x WITH x AS x WHERE x = 10000 RETURN x",
            "UNWIND range(0,1000000) AS x CREATE (:N {v: x})"
        ]

        for _ in range(1, 2):
            for query in queries:
                try:
                    # The query is expected to timeout
                    self.graph.query(query)
                    self.env.assertTrue(False)
                except ResponseError as error:
                    self.env.assertContains("Query timed out", str(error))

            # disable timeout_default, timeout_max should be enforced
            self.db.config_set("TIMEOUT_DEFAULT", 0)

        # revert timeout_default to 10
        self.db.config_set("TIMEOUT_DEFAULT", 10)

    def test08_enforce_timeout_configuration(self):
        read_q = "RETURN 1"
        write_q = "CREATE ()"
        queries = [read_q, write_q]

        max_timeout = self.db.config_get("TIMEOUT_MAX")

        for query in queries:
            try:
                # query is expected to fail
                self.graph.query(query, timeout=max_timeout+1)
                self.env.assertTrue(False)
            except ResponseError as error:
                self.env.assertContains("The query TIMEOUT parameter value cannot exceed the TIMEOUT_MAX configuration parameter value", str(error))

    def test09_fallback(self):
        self.env.stop()
        self.env, self.db = Env(moduleArgs="TIMEOUT 1")

        configs = ["TIMEOUT_DEFAULT", "TIMEOUT_MAX"]

        for config in configs:
            # enable/disable config expecting to fallback to the old timeout
            self.db.config_set(config, 10)
            self.db.config_set(config, 0)

            query = "UNWIND range(0,1000000) AS x WITH x AS x WHERE x = 10000 RETURN x"
            try:
                # The query is expected to timeout
                self.graph.query(query)
                self.env.assertTrue(False)
            except ResponseError:
                self.env.assertTrue(True)

            query = "UNWIND range(0, 1000000) AS x CREATE (:N {v: x})"
            try:
                # The query is expected to succeed
                self.graph.query(query)
                self.env.assertTrue(True)
            except:
                self.env.assertTrue(False)

    def test10_set_old_timeout_when_new_config_set(self):
        self.db.config_set("TIMEOUT_DEFAULT", 10)

        # try to set timeout
        try:
            self.db.config_set("TIMEOUT", 20)
            self.env.assertTrue(False)
        except ResponseError as error:
            self.env.assertContains("The TIMEOUT configuration parameter is deprecated. Please set TIMEOUT_MAX and TIMEOUT_DEFAULT instead", str(error))

    # When timeout occurs while executing a PROFILE command, only the error-message
    # should return to user
    def test11_profile_no_double_response(self):
        # reset timeout params to default
        self.env.stop()
        self.env, self.db = Env()

        # Set timeout parameters to small values (1 millisecond)
        self.db.config_set("TIMEOUT_MAX", 1)
        self.db.config_set("TIMEOUT_DEFAULT", 1)

        # Issue a profile query, expect a timeout error
        try:
            q = """UNWIND range(0, 10000) AS x
                   UNWIND range(0, 10000) AS y
                   CREATE (:P{v:x * y + y})"""
            self.graph.profile(q)
            self.env.assertTrue(False)
        except redis.exceptions.ResponseError as e:
            # expecting to get a timeout error
            self.env.assertIn("Query timed out", str(e))

        # make sure no pending result exists
        res = self.graph.query("RETURN 1")
        self.env.assertEquals(res.result_set[0][0], 1)

    def test12_concurrent_timeout(self):
        self.env.stop()
        self.env, self.db = Env()

        self.graph.query("UNWIND range(1, 1000) AS x CREATE (:N {v:x})")

        async def query():
            # connection pool with 16 connections
            # blocking when there's no connections available
            pool = BlockingConnectionPool(max_connections=16, timeout=None,
                                          port=self.env.port,
                                          decode_responses=True)

            db = FalkorDB(connection_pool=pool)
            g = db.select_graph(GRAPH_ID)

            tasks = []
            for i in range(1, 10000):
                q = f"MATCH (n:N) WHERE n.v > $i RETURN count(1)"
                tasks.append(asyncio.create_task(g.query(q, {'i': i}, timeout=1000)))

            # wait for tasks to finish
            await asyncio.gather(*tasks)

            # close the connection pool
            await pool.aclose()

        asyncio.run(query())

