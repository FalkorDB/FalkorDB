from common import *

GRAPH_ID = "profile"

class testProfile(FlowTestsBase):
    @classmethod
    def setup_class(cls):
        cls.env, cls.db = Env()
        cls.graph = cls.db.select_graph(GRAPH_ID)

    def test01_profile(self):
        q = """UNWIND range(1, 3) AS x CREATE (p:Person {v:x})"""
        profile = self.graph.profile(q)

        create_op = profile.structured_plan
        assert create_op.name == 'Create'

        unwind_op = create_op.children[0]
        assert unwind_op.name == 'Unwind'
        assert unwind_op.records_produced == 3

        #-----------------------------------------------------------------------

        q = "MATCH (p:Person) WHERE p.v > 1 RETURN p"
        profile = self.graph.profile(q)

        project_op = profile.structured_plan
        assert project_op.name == 'Project'
        assert project_op.records_produced == 2

        filter_op = project_op.children[0]
        assert filter_op.name == 'Filter'
        assert filter_op.records_produced == 2

        node_by_label_scan_op = filter_op.children[0]
        assert node_by_label_scan_op.name == 'Node By Label Scan'
        assert node_by_label_scan_op.records_produced == 3

    def test02_profile_after_op_reset(self):
        #validate that profile works properly on reset operations
        q = """MATCH (a:L)-[*]->() SET a.v = 5"""
        profile = self.graph.profile(q)

        update_op = profile.structured_plan
        assert update_op.name == 'Update'
        assert update_op.records_produced == 0

        traverse_op = update_op.children[0]
        assert traverse_op.name == 'Conditional Variable Length Traverse'
        assert traverse_op.records_produced == 0

        scan_op = traverse_op.children[0]
        assert scan_op.name == 'Node By Label Scan'
        assert scan_op.records_produced == 0
