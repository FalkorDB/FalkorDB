from common import *

GRAPH_ID = "graph_list"


# tests the GRAPH.LIST command
class testGraphList(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()

    def create_graph(self, graph_name, con):
        con.execute_command("GRAPH.QUERY", graph_name, "RETURN 1")

    def test_graph_list(self):
        # no graphs, expecting an empty array
        con = self.env.getConnection()
        graphs = self.db.list_graphs()
        self.env.assertEquals(graphs, [])

        # create graph key GRAPH_ID
        self.create_graph(GRAPH_ID, con)
        graphs = self.db.list_graphs()
        self.env.assertEquals(graphs, [GRAPH_ID])

        # create a second graph key "X"
        self.create_graph("X", con)
        graphs = self.db.list_graphs()
        graphs.sort()
        self.env.assertEquals(graphs, ["X", GRAPH_ID])

        # create a string key "str", graph list shouldn't be effected
        con.set("str", "some string")
        graphs = self.db.list_graphs()
        graphs.sort()
        self.env.assertEquals(graphs, ["X", GRAPH_ID])

        # delete graph key GRAPH_ID
        con.delete(GRAPH_ID)
        graphs = self.db.list_graphs()
        self.env.assertEquals(graphs, ["X"])

        # rename graph key X to Z
        con.rename("X", "Z")
        graphs = self.db.list_graphs()
        self.env.assertEquals(graphs, ["Z"])

        # delete graph key "Z", no graph keys in the keyspace
        con.execute_command("GRAPH.DELETE", "Z")
        graphs = self.db.list_graphs()
        self.env.assertEquals(graphs, [])

