import csv

from common import *
from constraint_utils import create_constraint
from index_utils import *
from click.testing import CliRunner
from falkordb_bulk_loader.bulk_insert import bulk_insert

GRAPH_ID = "identifier_limits"


class testIdentifierLimits:
    def __init__(self):
        self.env, self.db = Env()
        self.con = self.env.getConnection()
        self.con.delete(GRAPH_ID)
        self.graph = self.db.select_graph(GRAPH_ID)
        self.graph.query("CREATE (:Seed {v: 1})")
        self.port = self.env.envRunner.port

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
        create_constraint(self.graph, "MANDATORY", "NODE", valid_label, "p")
        create_constraint(self.graph, "MANDATORY", "NODE", "L2", valid_prop)

        # 513-byte label and property names fail with an explicit query error.
        self._assert_identifier_too_long(
            lambda: create_constraint(self.graph, "MANDATORY", "NODE", long_name, "p")
        )
        self._assert_identifier_too_long(
            lambda: create_constraint(self.graph, "MANDATORY", "NODE", "L3", long_name)
        )

    # Property-map updates consume runtime map keys (non-AST identifier path).
    def test02_set_map_identifier_length_limit(self):
        valid_key_add = "k" * 512
        valid_key_replace = "m" * 512
        long_key = "z" * 513

        # 512-byte map keys are accepted.
        result = self.graph.query(
            "MATCH (n) SET n += $props RETURN n",
            params={"props": {valid_key_add: 1}},
        )
        self.env.assertEqual(result.properties_set, 1)

        result = self.graph.query(
            "MATCH (n) SET n = $props RETURN n",
            params={"props": {valid_key_replace: 2}},
        )
        self.env.assertEqual(result.properties_set, 1)

        # 513-byte map keys fail with an explicit query error.
        self._assert_identifier_too_long(
            lambda: self.graph.query(
                "MATCH (n) SET n += $props RETURN n",
                params={"props": {long_key: 1}},
            )
        )
        self._assert_identifier_too_long(
            lambda: self.graph.query(
                "MATCH (n) SET n = $props RETURN n",
                params={"props": {long_key: 1}},
            )
        )

    # Fulltext index creation enforces the same 512/513 boundary on label/field names.
    def test03_fulltext_procedure_identifier_length_limit(self):
        valid_label = "f" * 512
        valid_field = "g" * 512
        long_name = "h" * 513

        # 512-byte label and field names are accepted.
        result = create_node_fulltext_index(self.graph, valid_label, 'field')
        self.env.assertEqual(result.indices_created, 1)

        result = create_node_fulltext_index(self.graph, 'L_proc', valid_field)
        self.env.assertEqual(result.indices_created, 1)

        # 513-byte names fail with an explicit query error.
        self._assert_identifier_too_long(
            lambda: create_node_fulltext_index(self.graph, long_name, 'field')
        )
        self._assert_identifier_too_long(
            lambda: create_node_fulltext_index(self.graph, 'L_proc2', long_name)
        )

    # CREATE INDEX path should enforce the same 512/513 boundary.
    def test04_create_index_identifier_length_limit(self):
        valid_prop = "i" * 512
        long_prop = "i" * 513

        # 512-byte property name is accepted.
        result = create_node_range_index(self.graph, 'IndexLabel', valid_prop)
        self.env.assertEqual(result.indices_created, 1)

        # 513-byte property name fails.
        self._assert_identifier_too_long(
            lambda: create_node_range_index(self.graph, 'IndexLabel2', long_prop)
        )

    # Function-name validation should accept length 512 and reject 513.
    def test05_function_name_identifier_length_limit(self):
        self.db.udf_flush()

        lib = "Lib"
        valid_func = "f" * 508  # len("Lib") + 1 + 508 = 512
        long_func = "f" * 509  # len("Lib") + 1 + 509 = 513
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
            self.env.assertFalse("Library name exceeds maximum length of 512" in str(e))

        self._assert_identifier_too_long(
            lambda: self.db.udf_load(long_lib, invalid_script, True)
        )

    # GRAPH.BULK headers should reject overlong label / rel-type / attribute names.
    def test07_bulk_insert_label_identifier_length_limit(self):
        runner = CliRunner()
        bulk_graph = f"{GRAPH_ID}_bulk"
        long_name = "a" * 513

        def assert_bulk_fail(res):
            self.env.assertNotEqual(res.exit_code, 0)
            combined = res.output + (str(res.exception) if res.exception else "")
            self.env.assertContains("exceeds maximum length of 512", combined)

        with runner.isolated_filesystem():
            # long label via --nodes-with-label
            with open('nodes.csv', 'w') as f:
                out = csv.writer(f)
                out.writerow(["name"])
                out.writerow(["node1"])

            res = runner.invoke(bulk_insert, [
                '--server-url', f"redis://localhost:{self.port}",
                '--nodes-with-label', long_name, 'nodes.csv',
                bulk_graph,
            ])
            assert_bulk_fail(res)

            # long attribute name via CSV column header
            with open('nodes_long_attr.csv', 'w') as f:
                out = csv.writer(f)
                out.writerow([long_name])
                out.writerow(["val1"])

            res = runner.invoke(bulk_insert, [
                '--server-url', f"redis://localhost:{self.port}",
                '--nodes', 'nodes_long_attr.csv',
                bulk_graph,
            ])
            assert_bulk_fail(res)

            # long rel-type via --relations-with-type
            with open('seed.csv', 'w') as f:
                out = csv.writer(f)
                out.writerow(["_identifier", "name"])
                out.writerow([0, "a"])
                out.writerow([1, "b"])

            with open('rels.csv', 'w') as f:
                out = csv.writer(f)
                out.writerow(["src", "dest"])
                out.writerow([0, 1])

            res = runner.invoke(bulk_insert, [
                '--server-url', f"redis://localhost:{self.port}",
                '--nodes', 'seed.csv',
                '--relations-with-type', long_name, 'rels.csv',
                bulk_graph,
            ])
            assert_bulk_fail(res)

