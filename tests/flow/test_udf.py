from common import *

GRAPH_ID = "udfs"

def udf_list(db, lib=None, with_code=None):
    libs = []
    res = db.udf_list(lib, with_code)

    for l in res:
        lib_name  = l[1]
        lib_funcs = l[3]

        lib_script = None
        if with_code:
            lib_script = l[5]

        libs.append({'library_name': lib_name, 'functions': lib_funcs, 'library_code': lib_script })

    return libs

class testUDF():
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.conn = self.env.getConnection()


    def _assert_udf_exists(self, lib, funcs, with_code=False, script=None):
        """
        Helper to assert that a UDF library is loaded correctly.

        Args:
            lib (str): The library name to check.
            funcs (list[str]): Expected list of function names.
            with_code (bool): If True, also checks for presence of the code in response.
            script (str|None): Optional script to compare against if with_code is True.
        """

        res = udf_list(self.db, lib, with_code)
        self.env.assertEqual(len(res), 1)

        res = res[0]

        lib_name   = res["library_name"]
        lib_funcs  = res["functions"]
        lib_script = res["library_code"]

        self.env.assertEqual(lib_name, lib)
        self.env.assertEqual(sorted(lib_funcs), sorted(funcs))

        if with_code:
            self.env.assertEqual(lib_script, script.strip())

    def _assert_any_udfs_missing(self):
        """Helper to assert that no UDF libraries are loaded at all."""

        res = udf_list(self.db)
        self.env.assertEqual(len(res), 0)

    def _assert_udf_missing(self, lib):
        """Helper to assert that a UDF library does not exist."""

        res = udf_list(self.db, lib)
        self.env.assertEqual(len(res), 0)

    def tearDown(self):
        self.db.udf_flush()
        self.conn.flushall()

    def test_return_primitives(self):
        """
        test the returning JS primitives back to FalkorDB works as expected
        """

        script ="""
        function ReturnInt       () { return 12;        }
        function ReturnFloat     () { return 3.14;      }
        function ReturnTrue      () { return true;      }
        function ReturnFalse     () { return false;     }
        function ReturnString    () { return 'hello';   }
        function ReturnNull      () { return null;      }
        function ReturnUndefined () { return undefined; }

        falkor.register ('ReturnInt',       ReturnInt);
        falkor.register ('ReturnFloat',     ReturnFloat);
        falkor.register ('ReturnTrue',      ReturnTrue);
        falkor.register ('ReturnString',    ReturnString);
        falkor.register ('ReturnFalse',     ReturnFalse);
        falkor.register ('ReturnNull',      ReturnNull);
        falkor.register ('ReturnUndefined', ReturnUndefined);
        """

        self.db.udf_load("ReturnTypes", script, True)

        # int
        v = self.graph.query("RETURN ReturnTypes.ReturnInt()").result_set[0][0]
        self.env.assertEqual(v, 12)

        # float
        v = self.graph.query("RETURN ReturnTypes.ReturnFloat()").result_set[0][0]
        self.env.assertEqual(v, 3.14)

        # string
        v = self.graph.query("RETURN ReturnTypes.ReturnString()").result_set[0][0]
        self.env.assertEqual(v, "hello")

        # boolean
        v = self.graph.query("RETURN ReturnTypes.ReturnTrue()").result_set[0][0]
        self.env.assertEqual(v, True)

        v = self.graph.query("RETURN ReturnTypes.ReturnFalse()").result_set[0][0]
        self.env.assertEqual(v, False)

        # null
        v = self.graph.query("RETURN ReturnTypes.ReturnNull()").result_set[0][0]
        self.env.assertEqual(v, None)

        # undefined → maps to NULL
        v = self.graph.query("RETURN ReturnTypes.ReturnUndefined()").result_set[0][0]
        self.env.assertEqual(v, None)

    def test_return_collections(self):
        script ="""
        function ReturnArray  () { return [1, null, 'str', [42]]; }
        function ReturnObject () { return {x: 1, y: 'val', z: [true, false]}; }
        function ReturnNested () { return {nested: [1, {k:'v'}, [true, null]]}; }

        falkor.register ('ReturnArray',  ReturnArray);
        falkor.register ('ReturnObject', ReturnObject);
        falkor.register ('ReturnNested', ReturnNested);
        """

        self.db.udf_load("ReturnCollections", script, True)

        # array
        v = self.graph.query("RETURN ReturnCollections.ReturnArray()").result_set[0][0]
        self.env.assertEqual(v, [1, None, "str", [42]])

        # object (map)
        v = self.graph.query("RETURN ReturnCollections.ReturnObject()").result_set[0][0]
        self.env.assertEqual(v, {"x": 1, "y": "val", "z": [True, False]})

        # nested structures
        v = self.graph.query("RETURN ReturnCollections.ReturnNested()").result_set[0][0]
        self.env.assertEqual(v, {"nested": [1, {"k": "v"}, [True, None]]})

    def test_return_specials(self):
        script ="""
        function ReturnBigInt  () { return 1234567890123456789n; }
        function ReturnDate    () { return new Date('2025-08-22T12:34:56Z'); }
        function ReturnRegExp  () { return /abc.*/; }
        function ReturnSymbol  () { return Symbol('s'); }
        function ReturnF32Array() { return new Float32Array([1.1, 2.2, 3.3]); }

        falkor.register ('ReturnBigInt',   ReturnBigInt);
        falkor.register ('ReturnDate',     ReturnDate);
        falkor.register ('ReturnRegExp',   ReturnRegExp);
        falkor.register ('ReturnSymbol',   ReturnSymbol);
        falkor.register ('ReturnF32Array', ReturnF32Array);
        """

        self.db.udf_load("ReturnSpecials", script, True)

        # BigInt → maps to INT64
        v = self.graph.query("RETURN ReturnSpecials.ReturnBigInt()").result_set[0][0]
        self.env.assertEqual(v, 1234567890123456789)

        # Date → maps to DATETIME
        v = self.graph.query("RETURN ReturnSpecials.ReturnDate()").result_set[0][0]
        self.env.assertTrue(v is not None)

        # RegExp → converted to string
        v = self.graph.query("RETURN ReturnSpecials.ReturnRegExp()").result_set[0][0]
        self.env.assertEqual(v, "/abc.*/")

        # Symbol is not supported
        try:
            v = self.graph.query("RETURN ReturnSpecials.ReturnSymbol()").result_set[0][0]
            self.env.assertTrue(False)
        except Exception:
            pass

        # TypedArray (Float32Array) → VECTOR_F32
        #v = self.graph.query("RETURN ReturnF32Array()").result_set[0][0]
        #print(f"v: {v}")
        #self.env.assertTrue(isinstance(v, list))
        #EPS = 1e-5
        #self.env.assertTrue(all(abs(a - b) < EPS for a, b in zip(v, [1.1, 2.2, 3.3])))

    # Test that a simple "Echo" UDF returns all supported FalkorDB types unchanged
    def test_types(self):
        # Register UDF (overwrites if already exists)
        script ="""
        function Echo(x) { return x; }

        falkor.register ('Echo', Echo);
        """

        self.db.udf_load("Echo", script, True)

        q = "RETURN $item, Echo.Echo($item)"

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

        q = "WITH point({latitude: 25.21, longitude: 31.7}) AS p RETURN p, Echo.Echo(p)"
        result_set = self.graph.query(q).result_set
        p, echo_p = result_set[0]
        self.env.assertEqual(p, echo_p)

        q = "WITH vecf32([2.1, 0.82, 1.3]) AS vec RETURN vec, Echo.Echo(vec)"
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
        q = "MATCH (n:Person {name:'Alice'}) RETURN n, Echo.Echo(n)"
        n, echo_n = self.graph.query(q).result_set[0]
        self.env.assertEqual(n, echo_n)

        # Edge
        q = "MATCH ()-[e:KNOWS]->() RETURN e, Echo.Echo(e)"
        e, echo_e = self.graph.query(q).result_set[0]
        self.env.assertEqual(e, echo_e)

        # Path
        q = "MATCH p=(a:Person {name:'Alice'})-[:KNOWS]->(b:Person {name:'Bob'}) RETURN p, Echo.Echo(p)"
        p, echo_p = self.graph.query(q).result_set[0]
        self.env.assertEqual(p, echo_p)

        #-----------------------------------------------------------------------
        # Temporal Types
        #-----------------------------------------------------------------------

        unsupported_temporal_types = [
            "WITH duration('P2DT3H4M')  AS x RETURN x, Echo.Echo(x)",
            "WITH localtime('13:37:00') AS x RETURN x, Echo.Echo(x)",
        ]

        for q in unsupported_temporal_types:
            try:
                self.graph.query(q)
                assert False, "Expected failure on missing args"
            except ResponseError as e:
                self.env.assertIn("UDF Exception:", str(e))

        temporal_queries = [
            """WITH date('2025-08-22') AS x
               WITH x, Echo.Echo(x) AS echo_x
               RETURN x.year  = echo_x.year  AND
                      x.month = echo_x.month AND
                      x.day   = echo_x.day""",

            """WITH localdatetime('2025-08-22T13:37:00') AS x
               WITH x, Echo.Echo(x) AS echo_x
               RETURN x.year  = echo_x.year  AND
                      x.month = echo_x.month AND
                      x.day   = echo_x.day"""
        ]

        for q in temporal_queries:
            res = self.graph.query(q).result_set[0][0]
            self.env.assertTrue(res)

    # Test that FalkorDB’s JavaScript Node object exposes
    # - `id`: internal node ID
    # - `labels`: array of labels
    # - `attributes`: dictionary of properties
    # and handles conflicts between internal id vs. user property "id".
    def test_node_object(self):
        # Register UDF that exposes node info
        script ="""
        function InspectNode(n) { return { internal_id: n.id,
                                         labels: n.labels,
                                         attributes: n.attributes}; }

        falkor.register ('InspectNode', InspectNode);
        """

        self.db.udf_load("InspectNode", script, True)

        # 1. Node with no labels
        q = "CREATE (n {height:180}) RETURN InspectNode.InspectNode(n)"
        res = self.graph.query(q).result_set[0][0]

        self.env.assertEqual(res['internal_id'], 0)
        self.env.assertEqual(res["labels"], [])
        self.env.assertEqual(res["attributes"], {'height': 180})

        # 2. Node with a single label
        q = "CREATE (n:Person {height:175}) RETURN InspectNode.InspectNode(n)"
        res = self.graph.query(q).result_set[0][0]

        self.env.assertEqual(res['internal_id'], 1)
        self.env.assertEqual(res["labels"], ["Person"])
        self.env.assertEqual(res["attributes"], {'height': 175})

        # 3. Node with multiple labels
        q = "CREATE (n:Person:Employee {height:190}) RETURN InspectNode.InspectNode(n)"
        res = self.graph.query(q).result_set[0][0]

        self.env.assertEqual(res['internal_id'], 2)
        self.env.assertEqual(res["labels"], ["Person", "Employee"])
        self.env.assertEqual(res["attributes"], {'height': 190})

        # 4. Node with conflicting "id" property
        q = "CREATE (n:Test {id:'user_defined_id', height:160}) RETURN InspectNode.InspectNode(n)"
        res = self.graph.query(q).result_set[0][0]

        # User-defined "id" attribute accessible via n.attributes.id
        self.env.assertEqual(res['internal_id'], 3)
        self.env.assertEqual(res["labels"], ["Test"])
        self.env.assertEqual(res["attributes"], {'id': "user_defined_id", 'height': 160})

    # Test that FalkorDB’s JavaScript Edge object exposes:
    # - `id`: internal edge ID
    # - `type`: relationship type string
    # - `source` / `target`: node IDs
    # - `attributes`: dictionary of properties
    # and handles conflict between internal id vs. user property "id".
    def test_edge_object(self):
        # Register UDF that exposes edge info
        script ="""
        function InspectEdge(e) { return {internal_id: e.id,
                                         type: e.type,
                                         source: e.source,
                                         target: e.target,
                                         attributes: e.attributes}; }

        falkor.register ('InspectEdge', InspectEdge);
        """

        self.db.udf_load("InspectEdge", script, True)


        # 1. Simple edge with attributes
        q = """
        CREATE (a:Person {name:'Alice'})-[e:KNOWS {since:2020}]->(b:Person {name:'Bob'})
        RETURN InspectEdge.InspectEdge(e) AS inspect, a AS source, b AS target
        """
        res = self.graph.query(q).result_set[0]
        inspect = res[0]
        source  = res[1]
        target  = res[2]

        # Internal edge ID
        self.env.assertEqual(inspect["internal_id"], 0)

        # Relationship type
        self.env.assertEqual(inspect["type"], "KNOWS")

        # source/target node IDs must be integers
        self.env.assertEqual(inspect["source"], source)
        self.env.assertEqual(inspect["target"], target)

        # attribute check
        self.env.assertEqual(inspect["attributes"]["since"], 2020)

        # 2. Edge with conflicting "id" property
        q = """
        MATCH (a:Person {name:'Alice'}), (b:Person {name:'Bob'})
        CREATE (a)-[e:WORKS_WITH {id:'edge_custom_id', role:'dev'}]->(b)
        RETURN InspectEdge.InspectEdge(e)
        """
        res = self.graph.query(q).result_set[0][0]

        # Internal ID must be numeric
        self.env.assertEqual(res["internal_id"], 1)

        # Relationship type
        self.env.assertEqual(res["type"], "WORKS_WITH")

        # User-defined "id" attribute is accessible via e.attributes.id
        self.env.assertEqual(res["attributes"]["id"], "edge_custom_id")
        self.env.assertEqual(res["attributes"]["role"], "dev")

    def test_load_invalid_invocations(self):
        """
        Test invalid invocations of GRAPH.UDF LOAD:
        1. Missing arguments (should error).
        2. Invalid trailing option (should error).
        """

        try:
            self.db.execute_command("GRAPH.UDF", "LOAD")
            assert False, "Expected failure on missing args"
        except ResponseError as e:
            self.env.assertIn("wrong number of arguments", str(e).lower())

        try:
            self.db.execute_command("GRAPH.UDF", "LOAD", "lib", "script", "INVALID")
            assert False, "Expected failure on invalid option"
        except ResponseError as e:
            self.env.assertIn("unknown option given", str(e).lower())

    def test_load_udf_lib(self):
        """
        Test successful loading of a UDF library:
        - Loads a script with two functions (Foo, Bar).
        - Verifies presence via GRAPH.UDF LIST.
        - Executes the functions and validates results.
        """

        script = """
        function Foo() { return 123; }
        function Bar(x) { return x + 1; }
        falkor.register("Foo", Foo);
        falkor.register("Bar", Bar);
        """

        res = self.db.udf_load("mylib", script)
        self.env.assertEqual(res, "OK")

        self._assert_udf_exists("mylib", ["Foo", "Bar"])

        v = self.graph.query("RETURN mylib.Foo()").result_set[0][0]
        self.env.assertEqual(v, 123)

        v = self.graph.query("RETURN mylib.Bar(41)").result_set[0][0]
        self.env.assertEqual(v, 42)

    def test_replace_library(self):
        """
        Test library replacement with REPLACE flag:
        - Load library 'replace_lib' with one version of function X.
        - Replace it with a new version of the same function.
        - Validate the updated function behavior.
        """

        script1 = """
        function X() { return "one"; }
        falkor.register("X", X);
        """

        script2 = """
        function X() { return "two"; }
        falkor.register("X", X);
        """

        self.db.udf_load("replace_lib", script1)
        v = self.graph.query("RETURN replace_lib.X()").result_set[0][0]
        self.env.assertEqual(v, "one")

        self.db.udf_load("replace_lib", script2, True)
        v = self.graph.query("RETURN replace_lib.X()").result_set[0][0]
        self.env.assertEqual(v, "two")

    def test_conflict_on_existing_lib(self):
        """
        Test conflict handling:
        - Load a library normally.
        - Attempt to load the same library again without REPLACE.
        - Expect a conflict error indicating the library already exists.
        """

        script = """
        function Y() { return 1; }
        falkor.register("Y", Y);
        """
        self.db.udf_load("conflict_lib", script)

        try:
            self.db.udf_load("conflict_lib", script, False)
            assert False, "Expected conflict error"
        except ResponseError as e:
            self.env.assertIn("already registered", str(e).lower())

        y = self.graph.query("RETURN conflict_lib.Y()").result_set[0][0]
        self.env.assertEqual(y, 1)

    def test_load_large_script(self):
        """
        Test loading a large JS script without registrating any UDF
        validate its registration by a follow up lib which calls one of the
        previously loaded functions
        """

        template_function = "function {name}() {{ return {result}; }}"

        # Collect all function definitions efficiently in a list
        functions = [
            template_function.format(name=f"func_{i}", result=i)
            for i in range(20000)
        ]

        # Join them once into a single script
        script = "\n".join(functions)

        res = self.db.udf_load("large_lib", script)
        self.env.assertEqual(res, "OK")

        # register a second library which uses the previous one
        script = "falkor.register('proxy_func_515', function() { return func_515();})"
        res = self.db.udf_load("proxy_lib", script)
        self.env.assertEqual(res, "OK")

        # try calling one of large_lib functions
        res = self.graph.query("RETURN proxy_lib.proxy_func_515()").result_set[0][0]
        self.env.assertEqual(res, 515)

    def test_invalid_js_script(self):
        """
        Test invalid JavaScript script:
        - Try loading a malformed script.
        - Expect a parsing/execution error from the engine.
        """

        bad_script = "function Bad() { return ;"
        try:
            self.db.udf_load("badlib", bad_script)
            assert False, "Expected JS parse error"
        except ResponseError as e:
            self.env.assertIn("SyntaxError:", str(e))

    def test_persistence_and_replication(self):
        """
        Test persistence and replication:
        - Load a UDF library.
        - Restart the server (simulate persistency check).
        - Ensure the UDF is still available and functional.
        """

        script = """
        function Persist() { return "I persist"; }
        falkor.register("Persist", Persist);
        """

        self.db.udf_load("persist_lib", script)

        # restart server (test harness utility)
        self.env.restart_and_reload()

        self._assert_udf_exists("persist_lib", ["Persist"])
        v = self.graph.query("RETURN persist_lib.Persist()").result_set[0][0]
        self.env.assertEqual(v, "I persist")

    def test_delete_invalid_invocations(self):
        """
        Test invalid invocations of GRAPH.UDF DELETE:
        - Missing arguments (should error).
        - Extra arguments (should error).
        """

        try:
            self.db.execute_command("GRAPH.UDF", "DELETE")
            assert False, "Expected failure on missing args"
        except ResponseError as e:
            self.env.assertIn("wrong number of arguments", str(e).lower())

        try:
            self.db.execute_command("GRAPH.UDF", "DELETE", "lib", "extra")
            assert False, "Expected failure on extra args"
        except ResponseError as e:
            self.env.assertIn("wrong number of arguments", str(e).lower())

    def test_delete_existing_library(self):
        """
        Test deletion of an existing library:
        - Load a library with a UDF.
        - Verify it exists and works.
        - Delete it using GRAPH.UDF DELETE.
        - Verify it no longer exists and calling its function fails.
        """

        script = """
        function DelTest() { return "bye"; }
        falkor.register("DelTest", DelTest);
        """
        self.db.udf_load("del_lib", script)
        self._assert_udf_exists("del_lib", ["DelTest"])

        v = self.graph.query("RETURN del_lib.DelTest()").result_set[0][0]
        self.env.assertEqual(v, "bye")

        # delete the library
        res = self.db.udf_delete("del_lib")
        self.env.assertTrue(res)

        # verify it's gone
        self._assert_udf_missing("del_lib")

        # function should now error
        try:
            self.graph.query("RETURN del_lib.DelTest()")
            assert False, "Expected failure calling deleted function"
        except ResponseError as e:
            print (f"str(e).lower(): {str(e).lower()}")
            self.env.assertIn("unknown function 'del_lib.deltest'", str(e).lower())

    def test_delete_nonexistent_library(self):
        """
        Test deletion of a non-existing library:
        - Attempt to delete a library that was never loaded.
        - Expect an error indicating the library does not exist.
        """

        try:
            self.db.udf_delete("no_such_lib")
            assert False, "Expected error deleting nonexistent library"
        except ResponseError as e:
            self.env.assertIn("does not exist", str(e).lower())

    def test_multiple_libraries_deletion(self):
        """
        Test deletion when multiple libraries exist:
        - Load two libraries (lib1, lib2).
        - Delete only one of them.
        - Verify the other one remains intact and functional.
        """

        script1 = "function F1() { return 'f1'; } falkor.register('F1', F1);"
        script2 = "function F2() { return 'f2'; } falkor.register('F2', F2);"

        self.db.udf_load("lib1", script1)
        self.db.udf_load("lib2", script2)

        # delete lib1
        self.db.udf_delete("lib1")
        self._assert_udf_missing("lib1")
        self._assert_udf_exists("lib2", ["F2"])

        # verify lib2 still works
        v = self.graph.query("RETURN lib2.F2()").result_set[0][0]
        self.env.assertEqual(v, "f2")

    def test_persistence_delete(self):
        """
        Test persistence of deletion:
        - Load a library.
        - Delete it.
        - Restart the server.
        - Verify the library does not reappear.
        """

        script = """function PersistDel() { return 99; }
                    falkor.register('PersistDel', PersistDel);"""

        self.db.udf_load("persist_del_lib", script)
        self._assert_udf_exists("persist_del_lib", ["PersistDel"])

        # save DB
        self.conn.save()

        # delete library
        self.db.udf_delete("persist_del_lib")
        self._assert_udf_missing("persist_del_lib")

        # restart server
        self.env.restart_and_reload()

        # confirm library did not come back
        self._assert_udf_missing("persist_del_lib")

        # confirm removed function isn't callable
        try:
            self.graph.query ("RETURN persist_del_lib.PersistDel()")
            assert False, f"Expected failure calling deleted function {f}"
        except ResponseError as e:
            self.env.assertIn("unknown function", str(e).lower())

    def test_delete_library_with_multiple_functions(self):
        """
        Test deleting a library with multiple functions:
        - Load a library with two functions.
        - Delete the library.
        - Verify all functions are removed together.
        """

        script = """
        function F3() { return 3; }
        function F4() { return 4; }
        falkor.register("F3", F3);
        falkor.register("F4", F4);
        """

        self.db.udf_load("multi_func_lib", script)
        self._assert_udf_exists("multi_func_lib", ["F3", "F4"])

        # delete library
        self.db.udf_delete("multi_func_lib")
        self._assert_udf_missing("multi_func_lib")

        # calling functions should fail
        for f in ["F3", "F4"]:
            try:
                self.graph.query(f"RETURN multi_func_lib.{f}()")
                assert False, f"Expected failure calling deleted function {f}"
            except ResponseError as e:
                self.env.assertIn("unknown function", str(e).lower())

    def test_flush_invalid_invocations(self):
        """
        Test invalid invocations of GRAPH.UDF FLUSH:
        - Extra arguments (should error).
        """

        try:
            self.db.execute_command("GRAPH.UDF", "FLUSH", "extra")
            assert False, "Expected failure on extra args"
        except ResponseError as e:
            self.env.assertIn("wrong number of arguments", str(e).lower())

    def test_flush_removes_all_libraries(self):
        """
        Test basic functionality of GRAPH.UDF FLUSH:
        - Load multiple libraries.
        - Verify they exist.
        - Flush all libraries.
        - Verify no libraries remain and functions are unavailable.
        """

        script1 = "function A() { return 'a'; } falkor.register('A', A);"
        script2 = "function B() { return 'b'; } falkor.register('B', B);"

        self.db.udf_load("lib1", script1)
        self.db.udf_load("lib2", script2)

        # sanity check they exist
        res = udf_list(self.db)
        libs = [r["library_name"] for r in res]
        self.env.assertIn("lib1", libs)
        self.env.assertIn("lib2", libs)

        # flush all UDFs
        res = self.db.udf_flush()
        self.env.assertEqual(res, "OK")

        # verify empty registry
        self._assert_any_udfs_missing()

        # functions should now fail
        for f in ["lib1.A", "lib2.B"]:
            try:
                self.graph.query(f"RETURN {f}()")
                assert False, f"Expected failure calling flushed function {f}"
            except ResponseError as e:
                self.env.assertIn("unknown function", str(e).lower())

    def test_flush_on_empty_registry(self):
        """
        Test flushing when no libraries:
        - Ensure FLUSH is safe and succeeds even when registry is already empty.
        """

        # make sure registry is empty
        self.db.udf_flush()
        self._assert_any_udfs_missing()

        # flush again should still succeed
        res = self.db.udf_flush()
        self.env.assertEqual(res, "OK")

    def test_persistence_of_flush(self):
        """
        Test persistence of flush:
        - Load a library.
        - Flush all libraries.
        - Restart the server.
        - Ensure no libraries are restored after restart.
        """

        script = "function PersistFlush() { return 'stay?'; } falkor.register('PersistFlush', PersistFlush);"
        self.db.udf_load("flush_lib", script)

        # save
        self.conn.save()

        # flush all
        self.db.udf_flush()
        self._assert_any_udfs_missing()

        # restart server (test infra helper)
        self.env.restart_and_reload()

        # still no libraries
        self._assert_any_udfs_missing()

    def test_list_invalid_invocations(self):
        """
        GRAPH.UDF LIST should not accept extra arguments.
        """

        try:
            self.db.execute_command("GRAPH.UDF", "LIST", "lib", "extra")
            assert False, "Expected LIST with extra args to fail"
        except ResponseError as e:
            self.env.assertIn("unknown option given", str(e).lower())

    def test_list_empty_registry(self):
        """
        When no UDF libraries are loaded, LIST should return an empty result.
        """

        self.db.udf_flush()  # ensure clean state
        res = self.db.udf_list()
        self.env.assertEqual(len(res), 0)

class test_udf_javascript():
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.conn = self.env.getConnection()

    def tearDown(self):
        self.db.udf_flush()
        self.conn.flushall()

    def test_register_existing_udf(self):
        """
        Registering the same UDF name twice within the same library should fail.
        """

        script = """
        function f() { return 1; }
        falkor.register('f', f);
        falkor.register('f', f);
        """

        try:
            self.db.udf_load("dup", script)
            assert False, "Expected duplicate registration to fail"
        except ResponseError as e:
            self.env.assertIn("Failed to register UDF library", str(e))

        #-----------------------------------------------------------------------

        script = """
        function f() { return 1; }
        falkor.register('f', f);
        """
        self.db.udf_load("lib1", script)

        # redeclare same function name under a different lib
        script2 = """
        function f() { return 2; }
        falkor.register('f', f);
        """
        self.db.udf_load("lib1_dup", script2)

        # call both libs
        result = self.graph.query("RETURN lib1.f() + lib1_dup.f()").result_set[0][0]
        self.env.assertEqual(result, 3)

    def test_call_non_existing_function(self):
        """
        Calling a non-existent UDF should raise an error.
        """

        self.db.udf_flush()
        try:
            self.graph.query("RETURN lib.DoesNotExist()")
            assert False, "Expected error when calling unknown UDF"
        except ResponseError as e:
            self.env.assertIn("unknown function 'lib.doesnotexist'", str(e).lower())

    def test_redeclaration_non_exposed_function(self):
        """
        Redeclaring a plain JS function that is NOT registered should be fine,
        since it's internal to the library.
        """

        script = """
        function helper() { return 1; }
        function helper() { return 2; }  // redeclaration
        function exposed() { return helper(); }
        falkor.register('exposed', exposed);
        """

        self.db.udf_load("lib_redecl", script)
        v = self.graph.query("RETURN lib_redecl.exposed()").result_set[0][0]
        self.env.assertEqual(v, 2)

        #-----------------------------------------------------------------------

        # overwrite helper once again, see that the latest version takes precedence
        script = """
        function helper() { return 3; }
        """

        self.db.udf_load("lib_redecl_new", script)
        v = self.graph.query("RETURN lib_redecl.exposed()").result_set[0][0]
        self.env.assertEqual(v, 3)

    def test_cross_library_calls(self):
        """
        Verify cross-library visibility:
        - Functions in other libraries are accessible to one another
        """

        script1 = """
        function base() { return 100; }
        falkor.register('base', base);
        """
        self.db.udf_load("lib_base", script1)

        script2 = """
        function wrapper() {
            return base();
        }
        falkor.register('wrapper', wrapper);
        """

        self.db.udf_load("lib_wrapper", script2, replace=True)
        res = self.graph.query("RETURN lib_wrapper.wrapper()").result_set[0][0]
        self.env.assertEqual(res, 100)

    def test_invalid_function_access(self):
        """
        Attempting to access built-in globals like console.log should fail
        (sandbox enforcement).
        """

        script = """
        function bad() { console.log('oops'); }
        falkor.register('bad', bad);
        """

        self.db.udf_load("lib_invalid", script)
        try:
            res = self.graph.query("RETURN lib_invalid.bad()").result_set
            assert False, "Expected error when calling unknown UDF"
        except ResponseError as e:
            self.env.assertIn("udf exception: 'console' is not defined", str(e).lower())

    def test_global_variable_usage(self):
        """
        Globals inside a library should persist across calls within the same lib.
        """

        script = """
        var counter = 0;
        function inc() { counter += 1; return counter; }
        falkor.register('inc', inc);
        """
        self.db.udf_load("lib_globals", script)

        v1 = self.graph.query("RETURN lib_globals.inc()").result_set[0][0]
        v2 = self.graph.query("RETURN lib_globals.inc()").result_set[0][0]
        self.env.assertEqual(v1, 1)
        self.env.assertGreaterEqual(v2, 1) # depending on the executing thread and it's js context

    def test_register_anonymous_function(self):
        """
        Registering an anonymous function should work normally.
        """

        script = """
        falkor.register('anon', function(x) { return x * 2; });
        """
        self.db.udf_load("lib_anon", script)

        v = self.graph.query("RETURN lib_anon.anon(5)").result_set[0][0]
        self.env.assertEqual(v, 10)

    def test_exception_propagation(self):
        """
        Exceptions in UDFs should propagate as Cypher errors.
        """

        script = """
        function fail() { throw new Error('boom'); }
        falkor.register('fail', fail);
        """
        self.db.udf_load("lib_fail", script)

        try:
            self.graph.query("RETURN lib_fail.fail()")
            assert False, "Expected JS exception to propagate"
        except ResponseError as e:
            self.env.assertIn("boom", str(e).lower())

    def test_argument_handling(self):
        """
        Validate argument passing:
        - Too few arguments should yield undefined for missing ones.
        - Too many arguments should be ignored.
        """

        script = """
        function f(x, y) { return [x, y]; }
        falkor.register('f', f);
        """
        self.db.udf_load("lib_args", script)

        v1 = self.graph.query("RETURN lib_args.f(1)").result_set[0][0]
        self.env.assertEqual(v1, [1, None])

        v2 = self.graph.query("RETURN lib_args.f(1, 2, 3)").result_set[0][0]
        self.env.assertEqual(v2, [1, 2])

    def test_returning_undefined(self):
        """
        UDFs returning `undefined` should map to Cypher NULL.
        """

        script = """
        function undef() { return undefined; }
        falkor.register('undef', undef);
        """
        self.db.udf_load("lib_undef", script)

        v = self.graph.query("RETURN lib_undef.undef()").result_set[0][0]
        self.env.assertEqual(v, None)

    def test_runtime_interrupt(self):
        """
        UDFs should be time bounded, avoiding infinity loop
        and long computations
        """

        script = """
        function infinity() {
            while(1) {
                var a = 1;
            }
            return 1;
        }

        falkor.register('infinity', infinity);
        """

        self.db.udf_load("infinity", script)

        try:
            v = self.graph.query("RETURN infinity.infinity()")
            assert False, "Expected JS exception to propagate"
        except ResponseError as e:
            self.env.assertIn("UDF Exception: interrupted", str(e))

class testUDFCluster():
    def __init__(self):
        self.env, self.db = Env(env='oss-cluster', shardsCount=3)
        self.master_1 = self.env.getConnection(shardId=1)
        self.master_2 = self.env.getConnection(shardId=2)
        self.master_3 = self.env.getConnection(shardId=3)
        self.shards = [self.master_1, self.master_2, self.master_3]

    def tearDown(self):
        for shard in self.shards:
            shard.execute_command("FLUSHALL")
            shard.execute_command("GRAPH.UDF", "FLUSH")

    def test_udf_load_propagation(self):
        """
        load UDF to one master shard and make sure
        that on success the UDF is propagated to the rest of the cluster
        """

        # make sure all shards are clean
        for shard in self.shards:
            udfs = shard.execute_command("GRAPH.UDF", "LIST")
            self.env.assertEqual(len(udfs), 0)

        # load UDF to master_1
        script = """
        falkor.register('add', function(a,b) {return a + b;});
        """

        res = self.db.udf_load("math", script)
        self.env.assertEqual(res, "OK")

        # collect UDFs from master_1
        master_1_udfs = self.master_1.execute_command("GRAPH.UDF", "LIST")
        self.env.assertNotEqual(len(master_1_udfs), 0)

        # make sure UDFs been propagated to the rest of the cluster
        for shard in self.shards:
            udfs = shard.execute_command("GRAPH.UDF", "LIST")
            self.env.assertEqual(master_1_udfs, udfs)

        # update UDFs on master_2
        script = """
        falkor.register('add', function(a,b) {return a + b;});
        falkor.register('sub', function(a,b) {return a - b;});
        """

        res = self.db.udf_load("math", script, replace=True)
        self.env.assertEqual(res, "OK")

        # collect UDFs from master_2
        master_2_udfs = self.master_2.execute_command("GRAPH.UDF", "LIST")
        self.env.assertNotEqual(len(master_2_udfs), 0)

        # make sure UDFs been propagated to the rest of the cluster
        for shard in self.shards:
            udfs = shard.execute_command("GRAPH.UDF", "LIST")
            self.env.assertEqual(master_2_udfs, udfs)

        # make sure a failed load doesn't effects the cluster
        try:
            # LOAD should fail as 'math' lib already exists and we did not
            # specified REPLACE
            self.db.udf_load("math", script)
            self.env.assertFalse(True)
        except Exception:
            pass

        # make sure UDFs remaind as before the failed call
        for shard in self.shards:
            udfs = shard.execute_command("GRAPH.UDF", "LIST")
            self.env.assertEqual(master_2_udfs, udfs)

    def test_udf_delete_propagation(self):
        """
        delete UDF from one master shard and make sure
        that on success the UDF is deleted from the rest of the cluster
        """

        # load 3 libraries
        libs    = ["A", "B", "C"]
        scripts = ["falkor.register('a', function(a) {return a;});",
                   "falkor.register('b', function(b) {return b;});",
                   "falkor.register('c', function(c) {return c;});"]

        for i in range(0, 3):
            lib    = libs[i]
            script = scripts[i]

            res = self.db.udf_load(lib, script)
            self.env.assertEqual(res, "OK")

        # make sure all 3 libs are available throughout the cluster
        master_1_udfs = self.master_1.execute_command("GRAPH.UDF", "LIST")
        self.env.assertEqual(len(master_1_udfs), 3)

        # make sure UDFs been propagated to the rest of the cluster
        for shard in self.shards:
            udfs = shard.execute_command("GRAPH.UDF", "LIST")
            self.env.assertEqual(master_1_udfs, udfs)

        # start removing libs
        remove_sequance = [(self.master_2, "B"),  # remove B from master 2
                           (self.master_3, "A"),  # remove A from master 3
                           (self.master_1, "C")]  # remove C from master 1

        for shard, lib in remove_sequance:
            res = self.db.udf_delete(lib)
            self.env.assertEqual(res, "OK")

            # make sure all nodes in the cluster has the same view over UDFs
            udfs = shard.execute_command("GRAPH.UDF", "LIST")
            for s in self.shards:
                s_udfs = s.execute_command("GRAPH.UDF", "LIST")
                self.env.assertEqual(udfs, s_udfs)

        # all shards should have no UDFs
        for s in self.shards:
            udfs = s.execute_command("GRAPH.UDF", "LIST")
            self.env.assertEqual(len(udfs), 0)

    def test_udf_flush_propagation(self):
        """
        flush UDFs from one master shard and make sure
        that on success the UDFs been flushed from the rest of the cluster
        """
        # load 3 libraries
        libs    = ["A", "B", "C"]
        scripts = ["falkor.register('a', function(a) {return a;});",
                   "falkor.register('b', function(b) {return b;});",
                   "falkor.register('c', function(c) {return c;});"]

        for i in range(0, 3):
            lib    = libs[i]
            script = scripts[i]

            res = self.db.udf_load(lib, script)
            self.env.assertEqual(res, "OK")

        # make sure all 3 libs are available throughout the cluster
        master_1_udfs = self.master_1.execute_command("GRAPH.UDF", "LIST")
        self.env.assertEqual(len(master_1_udfs), 3)

        # make sure UDFs been propagated to the rest of the cluster
        for shard in self.shards:
            udfs = shard.execute_command("GRAPH.UDF", "LIST")
            self.env.assertEqual(master_1_udfs, udfs)

        # flush UDFs
        res = self.db.udf_flush()
        self.env.assertEqual(res, "OK")

        # all shards should have no UDFs
        for s in self.shards:
            udfs = s.execute_command("GRAPH.UDF", "LIST")
            self.env.assertEqual(len(udfs), 0)

