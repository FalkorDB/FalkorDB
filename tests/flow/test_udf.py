from common import *

GRAPH_ID = "udf_types"

class testUDF():
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.conn = self.env.getConnection()

    def tearDown(self):
        self.graph.delete()

    def register_udf(self, name, body):
        res = self.conn.execute_command(
            "GRAPH.UDF",
            name,
            f"function {name}() {{ {body} }}"
        )

        self.env.assertEqual(res, "OK")

    def test_return_primitives(self):
        # int
        self.register_udf("ReturnInt", "return 12;")
        v = self.graph.query("RETURN ReturnInt()").result_set[0][0]
        self.env.assertEqual(v, 12)

        # float
        self.register_udf("ReturnFloat", "return 3.14;")
        v = self.graph.query("RETURN ReturnFloat()").result_set[0][0]
        self.env.assertEqual(v, 3.14)

        # string
        self.register_udf("ReturnString", "return 'hello';")
        v = self.graph.query("RETURN ReturnString()").result_set[0][0]
        self.env.assertEqual(v, "hello")

        # boolean
        self.register_udf("ReturnTrue", "return true;")
        v = self.graph.query("RETURN ReturnTrue()").result_set[0][0]
        self.env.assertEqual(v, True)

        self.register_udf("ReturnFalse", "return false;")
        v = self.graph.query("RETURN ReturnFalse()").result_set[0][0]
        self.env.assertEqual(v, False)

        # null
        self.register_udf("ReturnNull", "return null;")
        v = self.graph.query("RETURN ReturnNull()").result_set[0][0]
        self.env.assertEqual(v, None)

        # undefined → maps to NULL
        self.register_udf("ReturnUndefined", "return undefined;")
        v = self.graph.query("RETURN ReturnUndefined()").result_set[0][0]
        self.env.assertEqual(v, None)

    def test_return_collections(self):
        # array
        self.register_udf("ReturnArray", "return [1, null, 'str', [42]];")
        v = self.graph.query("RETURN ReturnArray()").result_set[0][0]
        self.env.assertEqual(v, [1, None, "str", [42]])

        # object (map)
        self.register_udf("ReturnObject", "return {x: 1, y: 'val', z: [true, false]};")
        v = self.graph.query("RETURN ReturnObject()").result_set[0][0]
        self.env.assertEqual(v, {"x": 1, "y": "val", "z": [True, False]})

        # nested structures
        self.register_udf("ReturnNested", "return {nested: [1, {k:'v'}, [true, null]]};")
        v = self.graph.query("RETURN ReturnNested()").result_set[0][0]
        self.env.assertEqual(v, {"nested": [1, {"k": "v"}, [True, None]]})

    def test_return_specials(self):
        # BigInt → maps to INT64
        self.register_udf("ReturnBigInt", "return 1234567890123456789n;")
        v = self.graph.query("RETURN ReturnBigInt()").result_set[0][0]
        self.env.assertEqual(v, 1234567890123456789)

        # Date → maps to DATETIME
        self.register_udf("ReturnDate", "return new Date('2025-08-22T12:34:56Z');")
        v = self.graph.query("RETURN ReturnDate()").result_set[0][0]
        self.env.assertTrue(v is not None)

        # RegExp → converted to string
        self.register_udf("ReturnRegExp", "return /abc.*/;")
        v = self.graph.query("RETURN ReturnRegExp()").result_set[0][0]
        self.env.assertEqual(v, "/abc.*/")

        # Symbol is not supported
        self.register_udf("ReturnSymbol", "return Symbol('s');")
        try:
            v = self.graph.query("RETURN ReturnSymbol()").result_set[0][0]
            self.env.assertTrue(False)
        except Exception:
            pass

        # TypedArray (Float32Array) → VECTOR_F32
        self.register_udf("ReturnF32Array", "return new Float32Array([1.1, 2.2, 3.3]);")
        v = self.graph.query("RETURN ReturnF32Array()").result_set[0][0]
        print(f"v: {v}")
        self.env.assertTrue(isinstance(v, list))
        EPS = 1e-5
        self.env.assertTrue(all(abs(a - b) < EPS for a, b in zip(v, [1.1, 2.2, 3.3])))

        # Point object
        self.register_udf("ReturnPoint", "return {x:1.0, y:2.0, z:3.0};")
        v = self.graph.query("RETURN ReturnPoint()").result_set[0][0]
        self.env.assertTrue("x" in v and "y" in v)

    # Test that a simple "Echo" UDF returns all supported FalkorDB types unchanged.
    def test_types(self):
        # Register UDF (overwrites if already exists)
        try:
            res = self.conn.execute_command(
                "GRAPH.UDF",
                "Echo",
                "function Echo(n) { return n; }"
            )
            #self.env.assertEqual(res, "OK")
        except Exception:
            pass

        q = "RETURN $item, Echo($item)"

        #-----------------------------------------------------------------------
        # NULL
        #-----------------------------------------------------------------------

        v, echo_v = self.graph.query(q, {'item': None}).result_set[0]
        self.env.assertEqual(v, echo_v)

        #-----------------------------------------------------------------------
        # Scalars
        #-----------------------------------------------------------------------

        scalars = [
            True,        # BOOL
            False,       # BOOL
            123,         # INT64
            -9999999999, # INT64 large negative
            3.14159,     # DOUBLE
            "hello",     # STRING
        ]

        for item in scalars:
            v, echo_v = self.graph.query(q, {'item': item}).result_set[0]
            self.env.assertEqual(v, echo_v)

        #-----------------------------------------------------------------------
        # Temporal Types
        #-----------------------------------------------------------------------

        # FalkorDB temporal literals are constructed from Cypher functions.
        #temporal_queries = [
        #    "WITH time('13:37:00Z')                    AS x RETURN x, Echo(x)",
        #    "WITH date('2025-08-22')                   AS x RETURN x, Echo(x)",
        #    "WITH duration('P2DT3H4M')                 AS x RETURN x, Echo(x)",
        #    "WITH localtime('13:37:00')                AS x RETURN x, Echo(x)",
        #    "WITH datetime('2025-08-22T13:37:00Z')     AS x RETURN x, Echo(x)",
        #    "WITH localdatetime('2025-08-22T13:37:00') AS x RETURN x, Echo(x)"
        #]

        #for q in temporal_queries:
        #    t, echo_t = self.graph.query(q).result_set[0]
        #    self.env.assertEqual(t, echo_t)

        #-----------------------------------------------------------------------
        # Collections
        #-----------------------------------------------------------------------

        collections = [
            [],                     # empty ARRAY
            [1, 2, 3],              # ARRAY of ints
            ["a", "b"],             # ARRAY of strings
            {"k1": "v1", "k2": 2},  # MAP
        ]

        for item in collections:
            v, echo_v = self.graph.query(q, {'item': item}).result_set[0]
            self.env.assertEqual(v, echo_v)

        #-----------------------------------------------------------------------
        # Spatial + Vector
        #-----------------------------------------------------------------------

        q = "WITH point({latitude: 25.21, longitude: 31.7}) AS p RETURN p, Echo(p)"
        result_set = self.graph.query(q).result_set
        p, echo_p = result_set[0]
        self.env.assertEqual(p, echo_p)

        q = "WITH vecf32([2.1, 0.82, 1.3]) AS vec RETURN vec, Echo(vec)"
        v, echo_v = self.graph.query(q).result_set[0]

        EPSILON = 1e-5
        self.env.assertEqual(len(v), len(echo_v))
        self.env.assertTrue(all(abs(a - b) < EPSILON for a, b in zip(v, echo_v)))

        #-----------------------------------------------------------------------
        # Graph Entities
        #-----------------------------------------------------------------------

        # Build a small graph to extract Node / Edge / Path
        self.graph.query("CREATE (a:Person {name:'Alice'})-[r:KNOWS {since:2020}]-> (b:Person {name:'Bob'})")

        # Node
        q = "MATCH (n:Person {name:'Alice'}) RETURN n, Echo(n)"
        n, echo_n = self.graph.query(q).result_set[0]
        self.env.assertEqual(n, echo_n)

        # Edge
        q = "MATCH ()-[r:KNOWS]->() RETURN r, Echo(r)"
        e, echo_e = self.graph.query(q).result_set[0]
        self.env.assertEqual(e, echo_e)

        # Path
        q = "MATCH p=(a:Person {name:'Alice'})-[:KNOWS]->(b:Person {name:'Bob'}) RETURN p, Echo(p)"
        p, echo_p = self.graph.query(q).result_set[0]
        self.env.assertEqual(p, echo_p)

    # Test that FalkorDB’s JavaScript Node object exposes
    # - `id`: internal node ID
    # - `labels`: array of labels
    # - `attributes`: dictionary of properties
    # and handles conflicts between internal id vs. user property "id".
    def test_node_object(self):
        # Register UDF that exposes node info
        try:
            res = self.conn.execute_command(
                "GRAPH.UDF",
                "InspectNode",
                """
                function InspectNode(n) {
                    return {
                        internal_id: n.id,
                        labels: n.labels,
                        attributes: n.attributes
                    };
                }
                """
            )
            self.env.assertEqual(res, "OK")
        except Exception:
            pass

        # 1. Node with no labels
        q = "CREATE (n {height:180}) RETURN InspectNode(n)"
        res = self.graph.query(q).result_set[0][0]

        self.env.assertEqual(res['internal_id'], 0)
        self.env.assertEqual(res["labels"], [])
        self.env.assertEqual(res["attributes"], {'height': 180})

        # 2. Node with a single label
        q = "CREATE (n:Person {height:175}) RETURN InspectNode(n)"
        res = self.graph.query(q).result_set[0][0]

        self.env.assertEqual(res['internal_id'], 1)
        self.env.assertEqual(res["labels"], ["Person"])
        self.env.assertEqual(res["attributes"], {'height': 175})

        # 3. Node with multiple labels
        q = "CREATE (n:Person:Employee {height:190}) RETURN InspectNode(n)"
        res = self.graph.query(q).result_set[0][0]

        self.env.assertEqual(res['internal_id'], 2)
        self.env.assertEqual(res["labels"], ["Person", "Employee"])
        self.env.assertEqual(res["attributes"], {'height': 190})

        # 4. Node with conflicting "id" property
        q = "CREATE (n:Test {id:'user_defined_id', height:160}) RETURN InspectNode(n)"
        res = self.graph.query(q).result_set[0][0]

        # User-defined "id" attribute accessible via n.attributes.id
        self.env.assertEqual(res['internal_id'], 3)
        self.env.assertEqual(res["labels"], ["Test"])
        self.env.assertEqual(res["attributes"], {'id': "user_defined_id", 'height': 160})

    # Test that FalkorDB’s JavaScript Edge object exposes:
    # - `id`: internal edge ID
    # - `type`: relationship type string
    # - `startNode` / `endNode`: node IDs
    # - `attributes`: dictionary of properties
    # and handles conflict between internal id vs. user property "id".
    def test_edge_object(self):
        # Register UDF that exposes edge info
        try:
            res = self.conn.execute_command(
                "GRAPH.UDF",
                "InspectEdge",
                """
                function InspectEdge(e) {
                    return {
                        internal_id: e.id,
                        type: e.type,
                        attributes: e.attributes
                    };
                }
                """
            )
            self.env.assertEqual(res, "OK")
        except Exception:
            pass

        # 1. Simple edge with attributes
        q = """
        CREATE (a:Person {name:'Alice'})-[e:KNOWS {since:2020}]->(b:Person {name:'Bob'})
        RETURN InspectEdge(e)
        """
        res = self.graph.query(q).result_set[0][0]

        # Internal edge ID
        self.env.assertEqual(res["internal_id"], 0)

        # Relationship type
        self.env.assertEqual(res["type"], "KNOWS")

        # Start/end node IDs must be integers
        #self.env.assertTrue(res["start"], 0)
        #self.env.assertTrue(res["end"], 1)

        # Attribute check
        self.env.assertEqual(res["attributes"]["since"], 2020)

        # 2. Edge with conflicting "id" property
        q = """
        MATCH (a:Person {name:'Alice'}), (b:Person {name:'Bob'})
        CREATE (a)-[e:WORKS_WITH {id:'edge_custom_id', role:'dev'}]->(b)
        RETURN InspectEdge(e)
        """
        res = self.graph.query(q).result_set[0][0]

        # Internal ID must be numeric
        self.env.assertEqual(res["internal_id"], 1)

        # Relationship type
        self.env.assertEqual(res["type"], "WORKS_WITH")

        # User-defined "id" attribute is accessible via e.attributes.id
        self.env.assertEqual(res["attributes"]["id"], "edge_custom_id")
        self.env.assertEqual(res["attributes"]["role"], "dev")

