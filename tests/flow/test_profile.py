from common import *

GRAPH_ID = "profile"

class testProfile(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def test01_profile(self):
        q = """UNWIND range(1, 3) AS x CREATE (p:Person {v:x})"""
        profile = self.graph.profile(q)

        # A write query's plan is rooted at Commit: it builds a CommitOp and
        # runs, so the profile lists it like any other operator.
        commit_op = profile.structured_plan
        self.env.assertEqual(commit_op.name, 'Commit')

        create_op = commit_op.children[0]
        self.env.assertEqual(create_op.name, 'Create')

        unwind_op = create_op.children[0]
        self.env.assertEqual(unwind_op.name, 'Unwind')
        self.env.assertEqual(unwind_op.records_produced, 3)

        #-----------------------------------------------------------------------

        q = "MATCH (p:Person) WHERE p.v > 1 RETURN p"
        profile = self.graph.profile(q)

        project_op = profile.structured_plan
        self.env.assertEqual(project_op.name, 'Project')
        self.env.assertEqual(project_op.records_produced, 2)

        filter_op = project_op.children[0]
        self.env.assertEqual(filter_op.name, 'Filter')
        self.env.assertEqual(filter_op.records_produced, 2)

        node_by_label_scan_op = filter_op.children[0]
        self.env.assertEqual(node_by_label_scan_op.name, 'Node By Label Scan')
        self.env.assertEqual(node_by_label_scan_op.records_produced, 3)

    def test02_profile_after_op_reset(self):
        #validate that profile works properly on reset operations
        q = """MATCH (a:L)-[*]->() SET a.v = 5"""
        profile = self.graph.profile(q)

        commit_op = profile.structured_plan
        self.env.assertEqual(commit_op.name, 'Commit')

        update_op = commit_op.children[0]
        self.env.assertEqual(update_op.name, 'Update')
        self.env.assertEqual(update_op.records_produced, 0)

        traverse_op = update_op.children[0]
        self.env.assertEqual(traverse_op.name, 'Conditional Variable Length Traverse')
        self.env.assertEqual(traverse_op.records_produced, 0)

        scan_op = traverse_op.children[0]
        self.env.assertEqual(scan_op.name, 'Node By Label Scan')
        self.env.assertEqual(scan_op.records_produced, 0)

    def test03_limit_caps_variable_length_traverse(self):
        # A LIMIT caps how many rows the operator under it packs into its first
        # batch. Operators call the emitter once before seeding their first
        # input; that empty call must not double the cap, or a var-length
        # traverse under LIMIT n walks 2n paths.
        g = self.db.select_graph("profile_limit_cap")
        try:
            g.query("""UNWIND range(1, 60) AS i
                       CREATE (:Hub {id: i})-[:R]->(:Leaf)""")
            g.query("""MATCH (h:Hub), (l:Leaf) WHERE h.id % 3 = 0
                       CREATE (h)-[:R]->(l)""")

            def find(op, name):
                if op.name == name:
                    return op
                for child in op.children:
                    found = find(child, name)
                    if found is not None:
                        return found
                return None

            for n in [1, 10, 37, 100]:
                for q in [f"MATCH (s:Hub)-[:R*1..2]->(t) RETURN t LIMIT {n}",
                          f"MATCH (s:Hub)-[:R*1..2]->(t) WITH t LIMIT {n} RETURN count(t)"]:
                    plan = g.profile(q).structured_plan
                    traverse = find(plan, 'Conditional Variable Length Traverse')
                    self.env.assertEqual(traverse.records_produced, n, message=q)
        finally:
            g.delete()
