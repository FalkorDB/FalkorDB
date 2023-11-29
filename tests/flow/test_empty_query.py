from common import *

class testEmptyQuery(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph('G')

    def test01_empty_query(self):
        try:
            # execute empty query
            self.graph.query("")
        except ResponseError as e:
            self.env.assertIn("Error: empty query.", str(e))

    #def test02_query_with_only_params(self):
    #    try:
    #        self.graph.query("CYPHER v=1")
    #    except ResponseError as e:
    #        self.env.assertIn("Error: could not parse query", str(e))
