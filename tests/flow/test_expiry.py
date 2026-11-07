from common import *
import time

GRAPH_ID = "expire"

class testExpiry():
    def test01_expire_graph(self):
        # create a redisgraph object
        env, db = Env()
        redis_con = env.getConnection()
        redis_graph = db.select_graph(GRAPH_ID)

        # create a single node
        redis_graph.query("create ()")

        # expire key in 100ms
        expire_in_ms = 100
        assert redis_con.pexpire(GRAPH_ID, expire_in_ms)

        # wait for key to expire
        time.sleep((expire_in_ms * 3) / 1000) # convert from sec to ms

        # key should have been evicted by now
        assert not redis_con.exists(GRAPH_ID)
        try:
            slowlog = redis_con.execute_command("GRAPH.SLOWLOG", GRAPH_ID)
        except ResponseError as e:
            assert "Invalid graph operation on empty key" in str(e)

