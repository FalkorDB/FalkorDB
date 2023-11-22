from common import *
from redis import ResponseError

VERSION = 0
GRAPH_ID = "GraphVersion"


class testGraphVersioning(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()

    # Make sure graph version changes once a new label is created
    def test01_version_update_on_label_creation(self):
        global VERSION
        con = self.env.getConnection()

        # Adding a node without a label shouldn't update graph version.
        q = """CREATE ()"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertFalse(isinstance(res[0], ResponseError))

        # Adding a labeled node should update graph version.
        q = """CREATE (:L)"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertFalse(isinstance(res[0], ResponseError))

        q = """RETURN 1"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertTrue(isinstance(res[0], ResponseError))

        # Update version
        VERSION = int(res[1])

        # Adding a node with an existing label shouldn't update graph version
        q = """CREATE (:L)"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertFalse(isinstance(res[0], ResponseError))

        q = """RETURN 1"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertFalse(isinstance(res[0], ResponseError))

    # Make sure graph version changes once a new relationship type is created
    def test02_version_update_on_relation_creation(self):
        global VERSION
        con = self.env.getConnection()

        # Adding edge with a new relationship type should update graph version
        q = """CREATE ()-[:R]->()"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertFalse(isinstance(res[0], ResponseError))

        q = """RETURN 1"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertTrue(isinstance(res[0], ResponseError))

        # Update version
        VERSION = int(res[1])

        # Adding edge with existing relationship type shouldn't update graph version
        q = """CREATE ()-[:R]->()"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertFalse(isinstance(res[0], ResponseError))

        q = """RETURN 1"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertFalse(isinstance(res[0], ResponseError))

    # Make sure graph version changes once a new attribute is created
    def test03_version_update_on_attribute_creation(self):
        global VERSION
        con = self.env.getConnection()

        # Adding a new attribute should update graph version
        q = """CREATE ({v:1})"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertFalse(isinstance(res[0], ResponseError))

        q = """RETURN 1"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertTrue(isinstance(res[0], ResponseError))

        # Update version
        VERSION = int(res[1])

        # Adding a new node with existing attribute shouldn't update graph version
        q = """CREATE ({v:1})"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertFalse(isinstance(res[0], ResponseError))

        # Adding a new edge with a new attribute should update graph version
        q = """CREATE ()-[:R {q:1}]->()"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertFalse(isinstance(res[0], ResponseError))

        q = """RETURN 1"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertTrue(isinstance(res[0], ResponseError))

        # Update version
        VERSION = int(res[1])

        # Adding a new edge with existing attribute shouldn't update graph version
        q = """CREATE ()-[:R {v:1}]->()"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        self.env.assertFalse(isinstance(res[0], ResponseError))
