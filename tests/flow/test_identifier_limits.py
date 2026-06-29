from common import *
import struct

GRAPH_ID = "identifier_limits"


class testIdentifierLimits():
    def __init__(self):
        self.env, self.db = Env()
        self.con = self.env.getConnection()
        self.con.delete(GRAPH_ID)
        self.graph = self.db.select_graph(GRAPH_ID)
        self.graph.query("CREATE (:Seed {v: 1})")

    def _assert_identifier_too_long(self, fn):
        try:
            fn()
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("exceeds maximum length of 512", str(e))

    # GRAPH.CONSTRAINT consumes command arguments directly (non-AST identifier path).
    def test01_constraint_command_identifier_length_limit(self):
        valid_label = "l" * 512
        valid_prop = "p" * 512
        long_name = "a" * 513

        # 512-byte label and property names are accepted.
        self.con.execute_command(
            "GRAPH.CONSTRAINT",
            "CREATE",
            GRAPH_ID,
            "MANDATORY",
            "NODE",
            valid_label,
            "PROPERTIES",
            1,
            "p",
        )
        self.con.execute_command(
            "GRAPH.CONSTRAINT",
            "CREATE",
            GRAPH_ID,
            "MANDATORY",
            "NODE",
            "L2",
            "PROPERTIES",
            1,
            valid_prop,
        )

        # 513-byte label and property names fail with an explicit query error.
        self._assert_identifier_too_long(
            lambda: self.con.execute_command(
                "GRAPH.CONSTRAINT",
                "CREATE",
                GRAPH_ID,
                "MANDATORY",
                "NODE",
                long_name,
                "PROPERTIES",
                1,
                "p",
            )
        )
        self._assert_identifier_too_long(
            lambda: self.con.execute_command(
                "GRAPH.CONSTRAINT",
                "CREATE",
                GRAPH_ID,
                "MANDATORY",
                "NODE",
                "L3",
                "PROPERTIES",
                1,
                long_name,
            )
        )

    # Property-map updates consume runtime map keys (non-AST identifier path).
    def test02_set_map_identifier_length_limit(self):
        valid_key_add = "k" * 512
        valid_key_replace = "m" * 512
        long_key = "z" * 513

        # 512-byte map keys are accepted.
        result = self.graph.query(
            f"CYPHER props={{`{valid_key_add}`: 1}} MATCH (n) SET n += $props RETURN n",
        )
        self.env.assertEqual(result.properties_set, 1)

        result = self.graph.query(
            f"CYPHER props={{`{valid_key_replace}`: 2}} MATCH (n) SET n = $props RETURN n",
        )
        self.env.assertEqual(result.properties_set, 1)

        # 513-byte map keys fail with an explicit query error.
        self._assert_identifier_too_long(
            lambda: self.graph.query(
                f"CYPHER props={{`{long_key}`: 1}} MATCH (n) SET n += $props RETURN n",
            )
        )
        self._assert_identifier_too_long(
            lambda: self.graph.query(
                f"CYPHER props={{`{long_key}`: 1}} MATCH (n) SET n = $props RETURN n",
            )
        )

    # Fulltext create procedures read label/field strings from runtime args.
    def test03_fulltext_procedure_identifier_length_limit(self):
        valid_label = "f" * 512
        valid_field = "g" * 512
        long_name = "h" * 513

        # 512-byte procedure string args are accepted.
        result = self.graph.query(
            f"CALL db.idx.fulltext.createNodeIndex('{valid_label}', 'field')"
        )
        self.env.assertEqual(result.indices_created, 1)

        result = self.graph.query(
            f"CALL db.idx.fulltext.createNodeIndex('L_proc', '{valid_field}')"
        )
        self.env.assertEqual(result.indices_created, 1)

        # 513-byte procedure string args fail with an explicit query error.
        self._assert_identifier_too_long(
            lambda: self.graph.query(
                f"CALL db.idx.fulltext.createNodeIndex('{long_name}', 'field')"
            )
        )
        self._assert_identifier_too_long(
            lambda: self.graph.query(
                f"CALL db.idx.fulltext.createNodeIndex('L_proc2', '{long_name}')"
            )
        )

    # CREATE INDEX path should enforce the same 512/513 boundary.
    def test04_create_index_identifier_length_limit(self):
        valid_prop = "i" * 512
        long_prop = "i" * 513

        # 512-byte property name is accepted.
        result = self.graph.query(
            f"CREATE INDEX FOR (n:IndexLabel) ON (n.`{valid_prop}`)"
        )
        self.env.assertEqual(result.indices_created, 1)

        # 513-byte property name fails.
        self._assert_identifier_too_long(
            lambda: self.graph.query(
                f"CREATE INDEX FOR (n:IndexLabel2) ON (n.`{long_prop}`)"
            )
        )

    # Function-name validation should accept length 512 and reject 513.
    def test05_function_name_identifier_length_limit(self):
        self.db.udf_flush()

        lib = "Lib"
        valid_func = "f" * 508  # len("Lib") + 1 + 508 = 512
        long_func = "f" * 509   # len("Lib") + 1 + 509 = 513
        script = f"""
        function valid() {{ return 1; }}
        falkor.register('{valid_func}', valid);
        """

        # Register and call a function with a 512-byte qualified name.
        self.db.udf_load(lib, script, True)
        result = self.graph.query(f"RETURN {lib}.{valid_func}()")
        self.env.assertEqual(result.result_set, [[1]])

        # Calling a 513-byte function name fails at identifier-length validation.
        self._assert_identifier_too_long(
            lambda: self.graph.query(f"RETURN {lib}.{long_func}()")
        )

    # GRAPH.UDF LOAD should enforce 512/513 on library names.
    def test06_udf_library_name_identifier_length_limit(self):
        self.db.udf_flush()

        valid_lib = "L" * 512
        long_lib = "L" * 513

        # Use invalid JS script so both calls fail, but only 513 should fail
        # due to identifier length.
        invalid_script = "function broken( {"

        try:
            self.db.udf_load(valid_lib, invalid_script, True)
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertFalse(
                "Library name exceeds maximum length of 512" in str(e)
            )

        self._assert_identifier_too_long(
            lambda: self.db.udf_load(long_lib, invalid_script, True)
        )

    # GRAPH.BULK headers should reject overlong labels before schema creation.
    def test07_bulk_insert_label_identifier_length_limit(self):
        bulk_graph = f"{GRAPH_ID}_bulk"
        self.con.delete(bulk_graph)

        long_label = b"a" * 513
        node_header = long_label + b"\x00" + struct.pack("I", 0)

        try:
            self.con.execute_command(
                "GRAPH.BULK",
                bulk_graph,
                "BEGIN",
                0,  # node count
                0,  # edge count
                1,  # node token count
                0,  # relation token count
                node_header,
            )
            self.env.assertTrue(False)
        except ResponseError as e:
            self.env.assertContains("Label name exceeds maximum length of 512", str(e))

