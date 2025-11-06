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
        self.env.assertEquals(create_op.name, 'Create')

        unwind_op = create_op.children[0]
        self.env.assertEquals(unwind_op.name, 'Unwind')
        self.env.assertEquals(unwind_op.records_produced, 3)

        #-----------------------------------------------------------------------

        q = "MATCH (p:Person) WHERE p.v > 1 RETURN p"
        profile = self.graph.profile(q)

        results_op = profile.structured_plan
        self.env.assertEquals(results_op.name, 'Results')
        self.env.assertEquals(results_op.records_produced, 2)

        project_op = results_op.children[0]
        self.env.assertEquals(project_op.name, 'Project')
        self.env.assertEquals(project_op.records_produced, 2)

        filter_op = project_op.children[0]
        self.env.assertEquals(filter_op.name, 'Filter')
        self.env.assertEquals(filter_op.records_produced, 2)

        node_by_label_scan_op = filter_op.children[0]
        self.env.assertEquals(node_by_label_scan_op.name, 'Node By Label Scan')
        self.env.assertEquals(node_by_label_scan_op.records_produced, 3)

    def test02_profile_after_op_reset(self):
        #validate that profile works properly on reset operations
        q = """MATCH (a:L)-[*]->() SET a.v = 5"""
        profile = self.graph.profile(q)

        update_op = profile.structured_plan
        self.env.assertEquals(update_op.name, 'Update')
        self.env.assertEquals(update_op.records_produced, 0)

        traverse_op = update_op.children[0]
        self.env.assertEquals(traverse_op.name, 'Conditional Variable Length Traverse')
        self.env.assertEquals(traverse_op.records_produced, 0)

        scan_op = traverse_op.children[0]
        self.env.assertEquals(scan_op.name, 'Node By Label Scan')
        self.env.assertEquals(scan_op.records_produced, 0)
