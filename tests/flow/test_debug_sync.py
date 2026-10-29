import time

import redis
from common import *

GRAPH_ID = "debug_sync"


# GRAPH.DEBUG SYNC <key> forces a full flush of a graph's delta matrices -
# it IS replicated (RedisModule_ReplicateVerbatim), so one invocation against
# the master reaches every attached replica, while staying "readonly"-
# flagged so it can also be invoked directly against a replica. see the comment
# above Debug_Sync in src/commands/cmd_debug.c.
class testDebugSync:
    def __init__(self):
        self.env, self.db = Env(useSlaves=True)
        self.master = self.env.getConnection()
        self.slave = self.env.getSlaveConnection()

    def _populate(self, conn):
        g = conn.select_graph(GRAPH_ID)
        g.query("UNWIND range(0, 999) AS x CREATE (:N {v: x})-[:R]->(:N {v: x})")
        return g

    def test_sync_on_master(self):
        g = self._populate(self.master)

        # force a full delta matrix sync on the master
        result = self.master.execute_command("GRAPH.DEBUG", "SYNC", GRAPH_ID)
        self.env.assertEquals(result, "OK")

        # data must be unaffected
        result = g.query("MATCH (n:N) RETURN count(n)")
        self.env.assertEquals(result.result_set[0][0], 2000)

        # replica must still be consistent
        self.master.execute_command("WAIT", "1", "0")
        g_slave = self.slave.select_graph(GRAPH_ID)
        result = g_slave.ro_query("MATCH (n:N) RETURN count(n)")
        self.env.assertEquals(result.result_set[0][0], 2000)

    def test_sync_is_replicated(self):
        self._populate(self.master)

        # observe the replica's incoming command stream directly to prove
        # SYNC issued against the master is propagated to it
        kwargs = dict(self.slave.connection_pool.connection_kwargs)
        kwargs["socket_timeout"] = 5
        mon_conn = redis.Redis(**kwargs)

        try:
            seen = False
            with mon_conn.monitor() as mon:
                self.master.execute_command("GRAPH.DEBUG", "SYNC", GRAPH_ID)
                self.master.execute_command("WAIT", "1", "0")

                deadline = time.time() + 5
                try:
                    for entry in mon.listen():
                        cmd = entry.get("command", "")
                        if "GRAPH.DEBUG" in cmd and "SYNC" in cmd:
                            seen = True
                            break
                        if time.time() > deadline:
                            break
                except redis.exceptions.TimeoutError:
                    pass

            self.env.assertTrue(seen)
        finally:
            mon_conn.close()

    def test_sync_directly_on_replica(self):
        self._populate(self.master)
        self.master.execute_command("WAIT", "1", "0")

        # GRAPH.DEBUG is a "readonly" command, so SYNC must be invocable
        # directly against a read-only replica, independent of the master
        result = self.slave.execute_command("GRAPH.DEBUG", "SYNC", GRAPH_ID)
        self.env.assertEquals(result, "OK")

    def test_sync_missing_graph(self):
        try:
            self.master.execute_command("GRAPH.DEBUG", "SYNC", "no_such_graph")
            self.env.assertTrue(False)
        except Exception as e:
            self.env.assertIn("Invalid graph operation on empty key", str(e))

    def test_sync_wrong_arity(self):
        try:
            self.master.execute_command("GRAPH.DEBUG", "SYNC")
            self.env.assertTrue(False)
        except Exception:
            pass
