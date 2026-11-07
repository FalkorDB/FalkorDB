from common import *

class testEmptyQuery(FlowTestsBase):
    @classmethod
    def setup_class(cls):
        cls.env, cls.db = Env()
        cls.graph = cls.db.select_graph('G')

    def test01_empty_query(self):
        try:
            # execute empty query
            self.graph.query("")
        except ResponseError as e:
            assert "Error: empty query." in str(e)

    def test02_whitespace_and_semicolon_queries(self):
        for query in [" ", ";"]:
            try:
                self.graph.query(query)
                assert False
            except ResponseError as e:
                assert "Error: empty query." in str(e)

    #def test02_query_with_only_params(self):
    #    try:
    #        self.graph.query("CYPHER v=1")
    #    except ResponseError as e:
    #        self.env.assertContains("Error: could not parse query", str(e))
