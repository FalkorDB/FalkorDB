from common import *

GRAPH_ID = "G"


class testHashJoin(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()

    def test_multi_hashjoins(self):
        # See issue https://github.com/RedisGraph/RedisGraph/issues/1124
        # Construct a 4 node graph, (v1),(v2),(v3),(v4)
        graph = Graph(self.env.getConnection(), GRAPH_ID)
        a = Node(properties = {"val": 1})
        b = Node(properties = {"val": 2})
        c = Node(properties = {"val": 3})
        d = Node(properties = {"val": 4})

        graph.add_node(a)
        graph.add_node(b)
        graph.add_node(c)
        graph.add_node(d)
        graph.flush()

        # Find nodes a,b,c such that a.v = 1, a.v = b.v-1 and b.v = c.v-1
        q = "MATCH (a {val:1}), (b), (c) WHERE a.val = b.val-1 AND b.val = c.val-1 RETURN a.val, b.val, c.val"
        plan = str(graph.explain(q))

        # Make sure plan contains 2 Value Hash Join operations
        self.env.assertEquals(plan.count("Value Hash Join"), 2)

        # Validate results
        expected_result = [[1,2,3]]
        actual_result = graph.query(q)

        self.env.assertEquals(actual_result.result_set, expected_result)

