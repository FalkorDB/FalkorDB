from common import *

GRAPH_ID = "redundant_ops"

class testRedundantOps(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        self.graph.delete()

    def test01_redundant_optional_cartesian_product(self):
        query = "CREATE (:Person {name:'Alice'}), (:Person {name:'Bob'})"
        self.graph.query(query)

        # both `a` is already matched upon reaching the OPTIONAL MATCH
        # part of the query, as such that part is redundant and could be removed
        query = """MATCH (a:Person {name:'Alice'}), (b:Person {name:'Bob'})
                   OPTIONAL MATCH (a)
                   RETURN count(*) AS cnt"""

        # validate that OPTIONAL MATCH isn't included in the generated plan
        plan = self.graph.explain(query)
        root = plan.structured_plan
        self.env.assertEquals(root.name, "")
        root.children[0]

        # validate results
        res = self.graph.query(query).result_set
        self.env.assertEquals(res[0][0], 1)

    def test02_redundant_optional_cartesian_product(self):
        query = "CREATE (:Person {name:'Alice'}), (:Person {name:'Bob'})"
        self.graph.query(query)

        # both `a` and `b` are already matched upon reaching the OPTIONAL MATCH
        # part of the query, as such that part is redundant and could be removed
        query = """MATCH (a:Person {name:'Alice'}), (b:Person {name:'Bob'})
                   OPTIONAL MATCH (a), (b)
                   RETURN count(*) AS cnt"""

        # validate that OPTIONAL MATCH isn't included in the generated plan
        plan = self.graph.explain(query)
        root = plan.structured_plan
        self.env.assertEquals(root.name, "")
        root.children[0]

        # validate results
        res = self.graph.query(query).result_set
        self.env.assertEquals(res[0][0], 1)







from common import *

GRAPH_ID = "redundant_ops"


class testRedundantOps(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        # Delete the graph after every test so each test starts with a clean slate.
        self.graph.delete()

    # =========================================================================
    # Plan-shape assertion helper
    # =========================================================================

    def _assert_plan_shape(self, op, expected):
        """Recursively assert that the execution-plan tree rooted at `op`
        exactly matches the `expected` shape.

        `expected` is a nested tuple: (operation_name, [child_spec, ...])
        """
        exp_name, exp_children = expected

        self.env.assertEquals(op.name, exp_name)
        self.env.assertEquals(len(op.children), len(exp_children))
        for child, exp_child in zip(op.children, exp_children):
            self._assert_plan_shape(child, exp_child)

    def test01_redundant_optional_bare_node(self):
        # OPTIONAL MATCH (a)
        # 'a' is already bound; the clause adds no new variables → redundant.
        self.graph.query("CREATE (:L {v:1})")

        query = """MATCH (a)
                   OPTIONAL MATCH (a)
                   RETURN 1"""

        plan = self.graph.explain(query)

        # The Optional and any Argument it contained must be gone entirely.
        #   Results
        #     Project          ← RETURN 1
        #       All Node Scan  ← MATCH (a)
        self._assert_plan_shape(
            plan.structured_plan,
            ("Results", [
                ("Project", [
                    ("All Node Scan", [])
                ])
            ])
        )

        res = self.graph.query(query).result_set
        self.env.assertEquals(res[0][0], 1)

    def test02_redundant_optional_node_with_label(self):
        # OPTIONAL MATCH (a:L)
        # 'a' is already bound; the extra label constraint on an already-bound
        # variable cannot introduce a new binding, so the clause is redundant.
        self.graph.query("CREATE (:L {v:1})")

        query = """MATCH (a)
                   OPTIONAL MATCH (a:L)
                   RETURN 1"""

        plan = self.graph.explain(query)

        # Results
        #   Project
        #     All Node Scan
        self._assert_plan_shape(
            plan.structured_plan,
            ("Results", [
                ("Project", [
                    ("All Node Scan", [])
                ])
            ])
        )

        res = self.graph.query(query).result_set
        self.env.assertEquals(res[0][0], 1)

    def test03_redundant_optional_node_with_property(self):
        # OPTIONAL MATCH (a {v:1})
        # 'a' is already bound; the property predicate on a bound variable
        # introduces no new bindings → clause is redundant.
        self.graph.query("CREATE (:L {v:1})")

        query = """MATCH (a)
                   OPTIONAL MATCH (a {v:1})
                   RETURN 1"""

        plan = self.graph.explain(query)

        # Results
        #   Project
        #     All Node Scan
        self._assert_plan_shape(
            plan.structured_plan,
            ("Results", [
                ("Project", [
                    ("All Node Scan", [])
                ])
            ])
        )

        res = self.graph.query(query).result_set
        self.env.assertEquals(res[0][0], 1)

    def test04_redundant_optional_node_with_label_and_property(self):
        # OPTIONAL MATCH (a:L {v:1})
        # 'a' is already bound; combining a label and a property on a bound
        # variable still introduces no new bindings → clause is redundant.
        self.graph.query("CREATE (:L {v:1})")

        query = """MATCH (a)
                   OPTIONAL MATCH (a:L {v:1})
                   RETURN 1"""

        plan = self.graph.explain(query)

        # Results
        #   Project
        #     All Node Scan
        self._assert_plan_shape(
            plan.structured_plan,
            ("Results", [
                ("Project", [
                    ("All Node Scan", [])
                ])
            ])
        )

        res = self.graph.query(query).result_set
        self.env.assertEquals(res[0][0], 1)

    def test05_redundant_optional_same_node_twice_bare(self):
        # OPTIONAL MATCH (a), (a)
        # Referencing the same already-bound variable twice in a single
        # OPTIONAL MATCH still introduces no new bindings → redundant.
        self.graph.query("CREATE (:L {v:1})")

        query = """MATCH (a)
                   OPTIONAL MATCH (a), (a)
                   RETURN 1"""

        plan = self.graph.explain(query)

        # Results
        #   Project
        #     All Node Scan
        self._assert_plan_shape(
            plan.structured_plan,
            ("Results", [
                ("Project", [
                    ("All Node Scan", [])
                ])
            ])
        )

        res = self.graph.query(query).result_set
        self.env.assertEquals(res[0][0], 1)

    def test06_redundant_optional_same_node_bare_and_labeled(self):
        # OPTIONAL MATCH (a), (a:L)
        # One pattern is bare, the other adds a label — both reference only
        # the already-bound 'a', so the clause is still fully redundant.
        self.graph.query("CREATE (:L {v:1})")

        query = """MATCH (a)
                   OPTIONAL MATCH (a), (a:L)
                   RETURN 1"""

        plan = self.graph.explain(query)

        # Results
        #   Project
        #     All Node Scan
        self._assert_plan_shape(
            plan.structured_plan,
            ("Results", [
                ("Project", [
                    ("All Node Scan", [])
                ])
            ])
        )

        res = self.graph.query(query).result_set
        self.env.assertEquals(res[0][0], 1)

    def test07_redundant_optional_same_node_property_and_labeled(self):
        # OPTIONAL MATCH (a {v:1}), (a:L)
        # One pattern carries a property filter, the other a label — both still
        # reference only the already-bound 'a' → clause is redundant.
        self.graph.query("CREATE (:L {v:1})")

        query = """MATCH (a)
                   OPTIONAL MATCH (a {v:1}), (a:L)
                   RETURN 1"""

        plan = self.graph.explain(query)

        # Results
        #   Project
        #     All Node Scan
        self._assert_plan_shape(
            plan.structured_plan,
            ("Results", [
                ("Project", [
                    ("All Node Scan", [])
                ])
            ])
        )

        res = self.graph.query(query).result_set
        self.env.assertEquals(res[0][0], 1)

    def test08_redundant_optional_same_node_all_constraints(self):
        # OPTIONAL MATCH (a:L {v:1}), (a:L), (a {v:1})
        # Multiple patterns, all with different combinations of label /
        # property constraints, but every one refers only to already-bound
        # 'a' → the entire OPTIONAL MATCH is redundant.
        self.graph.query("CREATE (:L {v:1})")

        query = """MATCH (a)
                   OPTIONAL MATCH (a:L {v:1}), (a:L), (a {v:1})
                   RETURN 1"""

        plan = self.graph.explain(query)

        # Results
        #   Project
        #     All Node Scan
        self._assert_plan_shape(
            plan.structured_plan,
            ("Results", [
                ("Project", [
                    ("All Node Scan", [])
                ])
            ])
        )

        res = self.graph.query(query).result_set
        self.env.assertEquals(res[0][0], 1)

    def test09_redundant_optional_two_distinct_bound_nodes(self):
        # OPTIONAL MATCH (a), (b)
        # Two different already-bound variables — both are bound before the
        # OPTIONAL MATCH is reached, so the clause is redundant for both.
        self.graph.query("CREATE (:L {v:1}), (:L {v:2})")

        query = """MATCH (a:L {v:1}), (b:L {v:2})
                   OPTIONAL MATCH (a), (b)
                   RETURN 1"""

        plan = self.graph.explain(query)

        # Results
        #   Aggregate          ← count(*) collapsed to scalar; RETURN 1 is Project
        #     Cartesian Product
        #       Filter
        #         Node By Label Scan
        #       Filter
        #         Node By Label Scan
        self._assert_plan_shape(
            plan.structured_plan,
            ("Results", [
                ("Project", [
                    ("Cartesian Product", [
                        ("Filter", [("Node By Label Scan", [])]),
                        ("Filter", [("Node By Label Scan", [])]),
                    ])
                ])
            ])
        )

        res = self.graph.query(query).result_set
        self.env.assertEquals(res[0][0], 1)

    def test10_call_redundant_optional_bare_node(self):
        # CALL { WITH a  OPTIONAL MATCH (a)  RETURN 1 AS x }
        # 'a' is imported via WITH and re-used bare in OPTIONAL MATCH.
        # No new variable is introduced → redundant.
        self.graph.query("CREATE (:L {v:1})")

        query = """MATCH (a)
                   CALL {
                     WITH a
                     OPTIONAL MATCH (a)
                     RETURN 1 AS x
                   }
                   RETURN x"""

        plan = self.graph.explain(query)

        # The Optional sub-tree is removed; the CallSubquery shell and its
        # Argument (for WITH a) remain untouched.
        #   Results
        #     Project
        #       CallSubquery
        #         Project
        #           Argument
        #         All Node Scan
        self._assert_plan_shape(
            plan.structured_plan,
            ("Results", [
                ("Project", [
                    ("Apply", [
                        ("All Node Scan", []),
                        ("Project", [
                            ("Project", [
                                ("Argument", [])
                            ])
                        ])
                    ])
                ])
            ])
        )

        res = self.graph.query(query).result_set
        self.env.assertEquals(res[0][0], 1)

    def test11_call_redundant_optional_node_with_label(self):
        # CALL { WITH a  OPTIONAL MATCH (a:L)  RETURN 1 AS x }
        # Adding a label constraint on the already-imported 'a' does not
        # introduce a new variable → still redundant.
        self.graph.query("CREATE (:L {v:1})")

        query = """MATCH (a)
                   CALL {
                     WITH a
                     OPTIONAL MATCH (a:L)
                     RETURN 1 AS x
                   }
                   RETURN x"""

        plan = self.graph.explain(query)

        #   Results
        #     Project
        #       CallSubquery
        #         Project
        #           Argument
        #         All Node Scan
        self._assert_plan_shape(
            plan.structured_plan,
            ("Results", [
                ("Project", [
                    ("Apply", [
                        ("All Node Scan", []),
                        ("Project", [
                            ("Project", [
                                ("Argument", [])
                            ])
                        ])
                    ])
                ])
            ])
        )

        res = self.graph.query(query).result_set
        self.env.assertEquals(res[0][0], 1)

    def test12_call_redundant_optional_node_with_property(self):
        # CALL { WITH a  OPTIONAL MATCH (a {v:1})  RETURN 1 AS x }
        # A property predicate on the already-imported 'a' introduces no new
        # binding → redundant.
        self.graph.query("CREATE (:L {v:1})")

        query = """MATCH (a)
                   CALL {
                     WITH a
                     OPTIONAL MATCH (a {v:1})
                     RETURN 1 AS x
                   }
                   RETURN x"""

        plan = self.graph.explain(query)

        #   Results
        #     Project
        #       CallSubquery
        #         Project
        #           Argument
        #         All Node Scan
        self._assert_plan_shape(
            plan.structured_plan,
            ("Results", [
                ("Project", [
                    ("Apply", [
                        ("All Node Scan", []),
                        ("Project", [
                            ("Project", [
                                ("Argument", [])
                            ])
                        ])
                    ])
                ])
            ])
        )

        res = self.graph.query(query).result_set
        self.env.assertEquals(res[0][0], 1)

    def test13_call_redundant_optional_node_with_label_and_property(self):
        # CALL { WITH a  OPTIONAL MATCH (a:L {v:1})  RETURN 1 AS x }
        # Combining a label and a property on the already-imported 'a' still
        # introduces no new binding → redundant.
        self.graph.query("CREATE (:L {v:1})")

        query = """MATCH (a)
                   CALL {
                     WITH a
                     OPTIONAL MATCH (a:L {v:1})
                     RETURN 1 AS x
                   }
                   RETURN x"""

        plan = self.graph.explain(query)

        #   Results
        #     Project
        #       CallSubquery
        #         Project
        #           Argument
        #         All Node Scan
        self._assert_plan_shape(
            plan.structured_plan,
            ("Results", [
                ("Project", [
                    ("Apply", [
                        ("All Node Scan", []),
                        ("Project", [
                            ("Project", [
                                ("Argument", [])
                            ])
                        ])
                    ])
                ])
            ])
        )

        res = self.graph.query(query).result_set
        self.env.assertEquals(res[0][0], 1)

    def test14_call_redundant_optional_same_node_bare_and_labeled(self):
        # CALL { WITH a  OPTIONAL MATCH (a), (a:L)  RETURN 1 AS x }
        # Two patterns in one OPTIONAL MATCH, both referencing only the
        # already-imported 'a' → the whole clause is redundant.
        self.graph.query("CREATE (:L {v:1})")

        query = """MATCH (a)
                   CALL {
                     WITH a
                     OPTIONAL MATCH (a), (a:L)
                     RETURN 1 AS x
                   }
                   RETURN x"""

        plan = self.graph.explain(query)

        #   Results
        #     Project
        #       CallSubquery
        #         Project
        #           Argument
        #         All Node Scan
        self._assert_plan_shape(
            plan.structured_plan,
            ("Results", [
                ("Project", [
                    ("Apply", [
                        ("All Node Scan", []),
                        ("Project", [
                            ("Project", [
                                ("Argument", [])
                            ])
                        ])
                    ])
                ])
            ])
        )

        res = self.graph.query(query).result_set
        self.env.assertEquals(res[0][0], 1)

    def test15_call_redundant_optional_same_node_property_and_labeled(self):
        # CALL { WITH a  OPTIONAL MATCH (a {v:1}), (a:L)  RETURN 1 AS x }
        # Property filter on one pattern, label on the other — all variables
        # are the already-imported 'a' → redundant.
        self.graph.query("CREATE (:L {v:1})")

        query = """MATCH (a)
                   CALL {
                     WITH a
                     OPTIONAL MATCH (a {v:1}), (a:L)
                     RETURN 1 AS x
                   }
                   RETURN x"""

        plan = self.graph.explain(query)

        #   Results
        #     Project
        #       CallSubquery
        #         Project
        #           Argument
        #         All Node Scan
        self._assert_plan_shape(
            plan.structured_plan,
            ("Results", [
                ("Project", [
                    ("Apply", [
                        ("All Node Scan", []),
                        ("Project", [
                            ("Project", [
                                ("Argument", [])
                            ])
                        ])
                    ])
                ])
            ])
        )

        res = self.graph.query(query).result_set
        self.env.assertEquals(res[0][0], 1)

