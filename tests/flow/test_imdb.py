from common import *

from index_utils import *
from reversepattern import ReversePattern

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../demo/imdb')
import imdb_queries
import imdb_utils

class testImdbFlow(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()

    def setUp(self):
        self.graph     = self.db.select_graph(imdb_utils.graph_name)
        actors, movies = imdb_utils.populate_graph(self.db, self.graph)
        self.imdb      = imdb_queries.IMDBQueries(actors, movies)
        self.queries   = self.imdb.queries()

    def tearDown(self):
        self.env.cmd('flushall')

    def assert_reversed_pattern(self, query, resultset):
        # Test reversed pattern query.
        reversed_query = ReversePattern().reverse_query_pattern(query)
        # print "reversed_query: %s" % reversed_query
        actual_result = self.graph.query(reversed_query)

        # assert result set
        self.env.assertEqual(resultset.result_set, actual_result.result_set)

    def test_imdb(self):
        for q in self.queries:
            query = q.query
            actual_result = self.graph.query(query)

            # assert result set
            self._assert_only_expected_results_are_in_actual_results(actual_result, q)

            if q.reversible:
                # assert reversed pattern.
                self.assert_reversed_pattern(query, actual_result)

    def test_index_scan_actors_over_85(self):
        # skip test if we're running under Valgrind
        # drop index is an async operation which can cause Valgraind
        # to wrongfully report as a leak
        if VALGRIND:
            self.env.skip()

        # Execute this command directly, as its response does not contain the result set that
        # 'self.graph.query()' expects
        create_node_range_index(self.graph, 'actor', 'age', sync=True)

        q = self.imdb.actors_over_85_index_scan.query
        execution_plan = str(self.graph.explain(q))
        self.env.assertIn('Index Scan', execution_plan)

        actual_result = self.graph.query(q)

        self.graph.drop_node_range_index("actor", "age")

        # assert result set
        self._assert_only_expected_results_are_in_actual_results(
            actual_result,
            self.imdb.actors_over_85_index_scan)

        # assert reversed pattern.
        self.assert_reversed_pattern(q, actual_result)

    def test_index_scan_eighties_movies(self):
        # skip test if we're running under Valgrind
        # drop index is an async operation which can cause Valgraind
        # to wrongfully report as a leak
        if VALGRIND:
            self.env.skip()

        # Execute this command directly, as its response does not contain the result set that
        # 'self.graph.query()' expects
        create_node_range_index(self.graph, 'movie', 'year', sync=True)

        q = self.imdb.eighties_movies_index_scan.query
        execution_plan = str(self.graph.explain(q))
        self.env.assertIn('Index Scan', execution_plan)

        actual_result = self.graph.query(q)

        self.graph.drop_node_range_index("movie", "year")

        # assert result set
        self._assert_only_expected_results_are_in_actual_results(
            actual_result,
            self.imdb.eighties_movies_index_scan)

        # assert reversed pattern.
        self.assert_reversed_pattern(q, actual_result)

