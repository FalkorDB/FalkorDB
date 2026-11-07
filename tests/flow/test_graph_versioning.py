from common import *
from redis import ResponseError

VERSION = 0
GRAPH_ID = "GraphVersion"


class testGraphVersioning(FlowTestsBase):
    @classmethod
    def setup_class(cls):
        cls.env, cls.db = Env(env='oss', useSlaves=True)
        cls.master = cls.env.getConnection()
        cls.replica = cls.env.getSlaveConnection()
        cls.master_graph = Graph(cls.master, GRAPH_ID)
        cls.replica_graph = Graph(cls.replica, GRAPH_ID)
        cls.master.wait(1, 0)

    # Make sure graph version changes once a new label is created
    def test01_version_update_on_label_creation(self):
        global VERSION

        con = self.master

        # Adding a node without a label shouldn't update graph version.
        q = """CREATE ()"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        assert not isinstance(res[0], ResponseError)

        # Adding a labeled node should update graph version.
        q = """CREATE (:L)"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        assert not isinstance(res[0], ResponseError)

        q = """RETURN 1"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        assert isinstance(res[0], ResponseError)

        # Update version
        VERSION = int(res[1])

        # Adding a node with an existing label shouldn't update graph version
        q = """CREATE (:L)"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        assert not isinstance(res[0], ResponseError)

        q = """RETURN 1"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        assert not isinstance(res[0], ResponseError)

    # Make sure graph version changes once a new relationship type is created
    def test02_version_update_on_relation_creation(self):
        global VERSION
        con = self.master

        # Adding edge with a new relationship type should update graph version
        q = """CREATE ()-[:R]->()"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        assert not isinstance(res[0], ResponseError)

        q = """RETURN 1"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        assert isinstance(res[0], ResponseError)

        # Update version
        VERSION = int(res[1])

        # Adding edge with existing relationship type shouldn't update graph version
        q = """CREATE ()-[:R]->()"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        assert not isinstance(res[0], ResponseError)

        q = """RETURN 1"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        assert not isinstance(res[0], ResponseError)

    # Make sure graph version changes once a new attribute is created
    def test03_version_update_on_attribute_creation(self):
        global VERSION
        con = self.master

        # Adding a new attribute should update graph version
        q = """CREATE ({v:1})"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        assert not isinstance(res[0], ResponseError)

        q = """RETURN 1"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        assert isinstance(res[0], ResponseError)

        # Update version
        VERSION = int(res[1])

        # Adding a new node with existing attribute shouldn't update graph version
        q = """CREATE ({v:1})"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        assert not isinstance(res[0], ResponseError)

        # Adding a new edge with a new attribute should update graph version
        q = """CREATE ()-[:R {q:1}]->()"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        assert not isinstance(res[0], ResponseError)

        q = """RETURN 1"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        assert isinstance(res[0], ResponseError)

        # Update version
        VERSION = int(res[1])

        # Adding a new edge with existing attribute shouldn't update graph version
        q = """CREATE ()-[:R {v:1}]->()"""
        res = con.execute_command("GRAPH.QUERY", GRAPH_ID, q, "version", VERSION)
        assert not isinstance(res[0], ResponseError)

    # make sure the graph version on the replica matches the one on the master
    # after each schema-changing operation (new label, relation type, attribute).
    def test04_version_sync_master_replica(self):
        global VERSION

        def get_version(con):
            """
            Execute a no-op query and read the graph version
            """
            res = con.execute_command("GRAPH.RO_QUERY", GRAPH_ID, "RETURN 1", "version", VERSION)
            return int(res[1])

        # ------------------------------------------------------------------ #
        # 1. New label
        # ------------------------------------------------------------------ #
        q = """CREATE (:NewLabel)"""
        self.master_graph.query(q)
        # Wait for the replica to catch up before reading its version.
        self.master.wait(1, 0)

        master_version  = get_version(self.master)
        replica_version = get_version(self.replica)
        assert master_version == replica_version
        VERSION = master_version

        # ------------------------------------------------------------------ #
        # 2. New relationship type
        # ------------------------------------------------------------------ #
        q = """CREATE ()-[:NewRelType]->()"""
        self.master_graph.query(q)
        self.master.wait(1, 0)

        master_version  = get_version(self.master)
        replica_version = get_version(self.replica)
        assert master_version == replica_version
        VERSION = master_version

        # ------------------------------------------------------------------ #
        # 3. New attribute key
        # ------------------------------------------------------------------ #
        q = """CREATE ({newAttr: 42})"""
        self.master_graph.query(q)
        self.master.wait(1, 0)

        master_version  = get_version(self.master)
        replica_version = get_version(self.replica)
        assert master_version == replica_version
        VERSION = master_version

        # ------------------------------------------------------------------ #
        # 4. Multiple schema changes
        # ------------------------------------------------------------------ #
        q = """CREATE (:A {a:1})-[:R {b:2, c:3}]->(:Z {z:4, y:5})"""
        self.master_graph.query(q)
        self.master.wait(1, 0)

        master_version  = get_version(self.master)
        replica_version = get_version(self.replica)
        assert master_version == replica_version
        VERSION = master_version
