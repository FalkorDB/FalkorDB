from common import *

GRAPH_ID = "profile"

class testProfile(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def test01_profile(self):
        q = """UNWIND range(1, 3) AS x CREATE (p:Person {v:x})"""
        profile = self.graph.profile(q)

        create_op = profile.structured_plan
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

        update_op = profile.structured_plan
        self.env.assertEqual(update_op.name, 'Update')
        self.env.assertEqual(update_op.records_produced, 0)

        traverse_op = update_op.children[0]
        self.env.assertEqual(traverse_op.name, 'Conditional Variable Length Traverse')
        self.env.assertEqual(traverse_op.records_produced, 0)

        scan_op = traverse_op.children[0]
        self.env.assertEqual(scan_op.name, 'Node By Label Scan')
        self.env.assertEqual(scan_op.records_produced, 0)
