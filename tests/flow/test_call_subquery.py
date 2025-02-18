from common import *
from collections import OrderedDict
from index_utils import create_node_range_index
from execution_plan_util import locate_operation, count_operation

GRAPH_ID = "call_subquery"

def _assert_subquery_contains_single(plan: ExecutionPlan, operation_name: str, env):
    """Asserts that the sub-plan embedded in a CallSubquery contains a single
    operation with name `operation_name`. If `plan` contains more than one
    CallSubquery operations, only the first will be checked."""

    callsubquery = locate_operation(plan.structured_plan, "CallSubquery")
    env.assertIsNotNone(callsubquery)
    env.assertEquals(count_operation(callsubquery, operation_name), 1)

class testCallSubqueryFlow():
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
    
    def get_res_and_assertEquals(self, query, expected_result):
        """Run the query and assert the result set is as expected"""

        actual_result = self.graph.query(query)
        self.env.assertEquals(actual_result.result_set, expected_result)
        return actual_result

    def expect_error(self, query, expected_err_msg):
        """Run the query and expect an error that contains the given message"""

        try:
            self.graph.query(query)
            assert(False)
        except redis.exceptions.ResponseError as e:
            self.env.assertIn(expected_err_msg, str(e))

    def test01_test_validations(self):
        """Make sure we fail on invalid queries"""

        import_error = "WITH imports in CALL {} must consist of only simple references to outside variables"
        match_after_updating = "A WITH clause is required to introduce MATCH after an updating clause"
        union_column_name_error = "All sub queries in a UNION must have the same column names."
        queries_errors = [
            ("WITH 1 AS a CALL {WITH a+1 AS b RETURN b} RETURN b", import_error),
            ("WITH {a: 1} AS map CALL {WITH map.a AS b RETURN b} RETURN b", import_error),
            ("WITH [1, 2, 3] AS list CALL {WITH list[0] AS b RETURN b} RETURN b", import_error),
            ("WITH 'RAZ' AS str CALL {WITH toUpper(str) AS b RETURN b} RETURN b", import_error),
            ("WITH 1 AS a CALL {WITH a AS b RETURN b} RETURN b", import_error),
            ("WITH 1 AS a CALL {WITH a LIMIT 5 RETURN a} RETURN a", import_error),
            ("WITH 1 AS a CALL {WITH a ORDER BY a.v RETURN a} RETURN a", import_error),
            ("WITH 1 AS a CALL {WITH a WHERE a > 5 RETURN a} RETURN a", import_error),
            ("WITH 1 AS a CALL {WITH a SKIP 5 RETURN a} RETURN a", import_error),
            ("WITH true AS a CALL {WITH NOT(a) AS b RETURN b} RETURN b", import_error),
            ("CALL {CREATE (n:N) MATCH (n:N) RETURN n} RETURN 1", match_after_updating),
            ("WITH 1 AS a CALL {WITH a CREATE (n:N) MATCH (n:N) RETURN n} RETURN a", match_after_updating),
            ("CALL {MATCH (n:N) CREATE (n:N2)} RETURN 1 ", "The bound variable 'n' can't be redeclared in a CREATE clause"),
            ("MATCH (n) CALL {WITH n AS n1 RETURN n1 UNION WITH n RETURN n1} RETURN n, n1", import_error),
            ("MATCH (n) CALL {WITH n RETURN n AS n1 UNION WITH n AS n1 RETURN n1} RETURN n, n1", import_error),
            ("CALL {RETURN 1 AS one UNION RETURN 2 AS two} RETURN 1", union_column_name_error),
            ("MATCH (n) CALL {RETURN 1 AS one UNION RETURN 2 AS two} RETURN 1", union_column_name_error)
        ]
        for query, err in queries_errors:
            self.expect_error(query, err)

        # invalid queries: import an undefined identifier
        queries = [
            # There is no variable 'a' to import from the outer scope
            """
            CALL {
                WITH a
                RETURN 1 AS one
            } 
            RETURN one
            """,
            # 'a' is not defined in the first `WITH`
            """
            WITH a
            CALL {
                WITH a 
                RETURN a AS b
            } 
            RETURN b
            """,
            # 'a' in the inner RETURN is not defined
            """
            MATCH (a:N)
            CALL {
                RETURN a.v AS INNERRETURN
            }
            RETURN 1
            """,
            # 'a' in the outer RETURN is not defined
            """
            CALL {
                MATCH (a:N)
                RETURN 1 as innerReturn
            }
            RETURN a
            """,
            # 'a' in inner RETURN is not defined
            """
            UNWIND [1, 2, 3, 4] AS a
            MATCH (n)
            CALL {
                WITH n
                RETURN a + 1
            }
            RETURN a + n.v
            """
        ]
        for query in queries:
            self.expect_error(query, "'a' not defined")

        # outer scope variables (bound) should not be returnable from the sq
        query = "MATCH (n:N) CALL {RETURN 1 AS n} RETURN n"
        self.expect_error(query, "Variable `n` already declared in outer scope")

        # a CALL {} after an updating clause requires a separating WITH
        query = "CREATE (n:N) CALL {RETURN 1} RETURN 1"
        self.expect_error(query,
            "A WITH clause is required to introduce CALL SUBQUERY after an \
updating clause.")

        # a query can not be terminated by a returning subquery
        query = "MATCH (n:N) CALL {WITH n CREATE (m:M {n: n.v}) RETURN m}"
        self.expect_error(query, "Query cannot conclude with a returning subquery \
(must be a RETURN clause, an update clause, a procedure call or a non-returning\
 subquery)")

    def test02_simple_scan_return(self):
        """Tests a simple scan and return subquery"""

        # the graph is empty
        # create a node
        res = self.graph.query("CREATE (n:N {name: 'Raz'})")
        self.env.assertEquals(res.nodes_created, 1)

        # find and return a node via CALL {}
        res = self.graph.query(
            """CALL
                {
                    MATCH (n:N {name: 'Raz'})
                    RETURN n
                }
                RETURN n
            """
        )

        # one node should have been found
        self.env.assertEquals(len(res.result_set), 1)
        # node labels and props should match created node
        self.env.assertEquals(res.result_set[0][0],
                              Node(labels='N', properties={'name': 'Raz'}))

    def test03_filter_results(self):
        """Tests that no records return from the subquery for scans that found
        nothing."""

        # the graph has one node with label `N` and prop `name` with val `Raz`
        # add the `v` property to the one node currently existing, with val 4
        res = self.graph.query("MATCH (n:N) SET n.v = 4")
        self.env.assertEquals(res.properties_set, 1)

        # if a returning subquery doesn't return records, the input record is
        # not passed as well. Here, only one record should be returned
        res = self.graph.query(
            """
            CALL {
                UNWIND [1, 2, 3, 4] AS x
                MATCH (n:N {v: x})
                RETURN n
            }
            RETURN n
            """
        )

        # assert correctness of the result
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(res.result_set[0][0],
            Node(labels='N', properties={'name': 'Raz', 'v': 4}))

    def test04_reference_innerReturn_alias(self):
        """Tests that the outer scope can reference the returned projections
        from the subquery"""

        res = self.graph.query(
            """
            UNWIND [1, 2, 3, 4] AS x
            MATCH (n)
            CALL {
                WITH n
                RETURN n.v as v
            }
            RETURN x + v AS ret ORDER BY ret ASC
            """
        )

        self.env.assertEquals(res.result_set, [[5], [6], [7], [8]])

        res = self.graph.query (
            """
            UNWIND range(1,4) AS i
            CALL {
                WITH i
                UNWIND range(1, 4) AS j
                WITH j WHERE j >= i
                RETURN count(1) AS v
            }
            RETURN i, v ORDER BY i ASC
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 4)
        for i in range(0, 4):
            self.env.assertEquals(res.result_set[i][0], i + 1)
            self.env.assertEquals(res.result_set[i][1], 4 - i)

    def test05_many_to_one_not_first(self):
        """Tests the case Call {} gets several records, and returns only one,
        that corresponds to a record that is not the first one it gets"""

        # first input record to sq yields no records, next inputs do
        res = self.graph.query(
            """
            UNWIND ['Omer', 'Raz', 'Moshe'] as name
            CALL {
                WITH name
                MATCH (n:N {name: name})
                RETURN n
            }
            RETURN n
            """
        )
        # assert correctness of the result
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(res.result_set[0][0],
            Node(labels='N', properties={'name': 'Raz', 'v': 4}))

    def test06_one_to_many(self):
        """Tests the case Call {} changes the amount of records such that there
        are more output records than input"""

        # the graph contains the following node (:N {name: 'Raz', v: 4}})

        res = self.graph.query(
            """
            WITH 1 as one
            CALL {
                UNWIND [1, 2] AS x
                RETURN x
            }
            RETURN x, one
            """
        )

        # validate the results
        self.env.assertEquals(len(res.result_set), 2)
        self.env.assertEquals(res.result_set[0][0], 1)
        self.env.assertEquals(res.result_set[0][1], 1)
        self.env.assertEquals(res.result_set[1][0], 2)
        self.env.assertEquals(res.result_set[1][1], 1)

        # more than one component in the outer loop
        res = self.graph.query(
            """
            UNWIND [1, 2, 3, 4] AS x
            CALL {
                UNWIND [1, 2] AS y
                RETURN y
            }
            RETURN x, y
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 8)
        for i in range(0, 4):
            for j in range(0, 2):
                self.env.assertEquals(res.result_set[i * 2 + j][0], i + 1)
                self.env.assertEquals(res.result_set[i * 2 + j][1], j + 1)

    def test07_optional_match(self):
        """Tests that OPTIONAL MATCH within the subquery works appropriately,
        i.e., passes records with empty columns in case no match was found"""

        self.graph.query("CREATE (:N {name: 'Raz', v: 1})")

        res = self.graph.query(
            """
            CALL {
                UNWIND [1, 2, 3, 4] AS x
                OPTIONAL MATCH (n:N {v: x})
                RETURN n
            }
            RETURN n ORDER BY n.v ASC
            """
        )

        # validate the results
        self.env.assertEquals(len(res.result_set), 4)
        self.env.assertEquals(res.result_set[0][0],
        Node(labels='N', properties={'name': 'Raz', 'v': 1}))
        self.env.assertEquals(res.result_set[1][0],
        Node(labels='N', properties={'name': 'Raz', 'v': 4}))
        self.env.assertEquals(res.result_set[2][0], None)
        self.env.assertEquals(res.result_set[3][0], None)

    def test08_filtering(self):
        """Tests filtering within the subquery"""

        # the graph has two nodes, with `name` 'Raz' and `v` 1 and 4
        # test filter using WHERE
        res = self.graph.query(
            """
            CALL {
                MATCH (n:N {v: 1})
                RETURN n
            }
            RETURN n
            """
        )

        # assert the correctness of the results
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(res.result_set[0][0],
        Node(labels='N', properties={'name': 'Raz', 'v': 1}))

    def test09_simple_ret_eager(self):
        """Simple test for the eager and returning subquery case"""

        # the graph has two nodes, with `name` 'Raz' and `v` 1 and 4
        # execute a query with an eager operation and returning subquery inside
        res = self.graph.query(
            """
            CALL {
                MATCH (n:N)
                SET n.v = n.v + 1
                RETURN n
            }
            RETURN n ORDER BY n.v ASC
            """
        )

        # make sure the wanted actions occurred
        self.env.assertEquals(res.properties_set, 2)
        self.env.assertEquals(res.result_set[0][0], Node(labels='N',
            properties={'name': 'Raz', 'v': 2}))
        self.env.assertEquals(res.result_set[1][0], Node(labels='N',
            properties={'name': 'Raz', 'v': 5}))

        # import outer-scope aliases, reference them inside and outside the sq
        res = self.graph.query(
            """
            WITH 2 AS x
            CALL {
                WITH x
                MATCH (n:N {v: x})
                SET n.v = 2 * n.v
                RETURN n
            }
            RETURN n, x
            """
        )

        # make sure the wanted actions occurred
        self.env.assertEquals(res.properties_set, 1)
        self.env.assertEquals(res.result_set[0][0], Node(labels='N', properties={'name': 'Raz', 'v': 4}))
        self.env.assertEquals(res.result_set[0][1], 2)

    # test FOREACH within CALL {} (updating (eager), returning)
    def test10_embedded_foreach(self):
        """Tests that FOREACH works properly when used inside a subquery"""

        # the graph has two nodes with label `N`, with `name` 'Raz' and `v` 4
        # and 5

        res = self.graph.query(
            """
            CALL {
                MATCH (n:N)
                FOREACH (i in [1] |
                    SET n.v = n.v + 1
                )
                RETURN n
            }
            RETURN n ORDER BY n.v ASC
            """
        )

        # assert the correctness of the results
        self.env.assertEquals(res.result_set[0][0],
            Node(labels='N', properties={'name': 'Raz', 'v': 5}))
        self.env.assertEquals(res.result_set[1][0],
            Node(labels='N', properties={'name': 'Raz', 'v': 6}))

        # Test with a non-returning subquery
        # Update properties using FOREACH in subquery
        self.graph.query(
            """
            CALL {
                MATCH (n:N)
                FOREACH (m in [n] |
                    MERGE (:TEMP {v: m.v})
                )
            }
            """
        )

        # assert the correctness of the results
        res = self.graph.query("MATCH(n:TEMP) RETURN n ORDER BY n.v ASC")
        self.env.assertEquals(len(res.result_set), 2)
        self.env.assertEquals(res.result_set[0][0],
            Node(labels='TEMP', properties={'v': 5}))
        self.env.assertEquals(res.result_set[1][0],
            Node(labels='TEMP', properties={'v': 6}))

        # delete the nodes with label :TEMP
        res = self.graph.query("MATCH (n:TEMP) DELETE n")
        self.env.assertEquals(res.nodes_deleted, 2)

        # FOREACH is first clause inside {}
        self.graph.query(
            """
            CALL {
                FOREACH (m in [1, 2] |
                    CREATE (:TEMP)
                )
            }
            """
        )

        # assert the correctness of the results
        res = self.graph.query("MATCH(n:TEMP) RETURN n ORDER BY n.v ASC")
        self.env.assertEquals(len(res.result_set), 2)

        # delete the nodes with label :TEMP
        res = self.graph.query("MATCH (n:TEMP) DELETE n")
        self.env.assertEquals(res.nodes_deleted, 2)

        # tests that `FOREACH` is interpreted as an updating clause, properly
        res = self.graph.query(
            """
            MATCH (n)
            CALL {
                FOREACH (m in [1] |
                    CREATE (:TEMP)
                )
                RETURN 1 AS one
            }
            RETURN n, one ORDER BY n.v ASC
            """
        )

        # assert results
        self.env.assertEquals(res.nodes_created, 2)
        self.env.assertEquals(len(res.result_set), 2)
        self.env.assertEquals(res.result_set[0][0], Node(labels='N',
            properties={'name': 'Raz', 'v': 5}))
        self.env.assertEquals(res.result_set[0][1], 1)
        self.env.assertEquals(res.result_set[1][0], Node(labels='N',
            properties={'name': 'Raz', 'v': 6}))
        self.env.assertEquals(res.result_set[1][1], 1)

        # delete the nodes with label :TEMP
        res = self.graph.query("MATCH (n:TEMP) DELETE n")

    def test11_skip_limit(self):
        """Tests that SKIP and LIMIT work properly when placed inside a
        subquery"""

        # the graph has two nodes, with `name` 'Raz' and `v` 5 and 6

        # Test LIMIT
        res = self.graph.query(
            """
            CALL {
                MATCH (n:N)
                RETURN n
                ORDER BY n.v ASC
                LIMIT 1
            }
            RETURN n
            """
        )

        # assert that only one record was returned
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(res.result_set[0][0],
            Node(labels='N', properties={'name': 'Raz', 'v': 5}))

        # Test SKIP
        res = self.graph.query(
            """
            CALL {
                MATCH (n:N)
                RETURN n
                ORDER BY n.v ASC
                SKIP 1
            }
            RETURN n
            """
        )

        # assert that only one record was returned
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(res.result_set[0][0],
            Node(labels='N', properties={'name': 'Raz', 'v': 6}))

        # Tests SKIP and LIMIT together

        # Create 10 nodes with label :W
        res = self.graph.query("UNWIND range(1,10) AS i CREATE(:W {name:tostring(i), value:i})")
        self.env.assertEquals(res.nodes_created, 10)

        res = self.graph.query(
            """
            CALL {
                MATCH (n:W)
                RETURN n
                ORDER BY n.value ASC
                SKIP 5
                LIMIT 2
            }
            RETURN n
            """
        )

        # assert that only two records were returned
        self.env.assertEquals(len(res.result_set), 2)
        self.env.assertEquals(res.result_set[0][0],
            Node(labels='W', properties={'name': '6', 'value': 6}))
        self.env.assertEquals(res.result_set[1][0],
            Node(labels='W', properties={'name': '7', 'value': 7}))

        # delete the nodes with label :W created only for LIMIT tests purpose
        res = self.graph.query("MATCH (n:W) DELETE n")
        self.env.assertEquals(res.nodes_deleted, 10)

    def test12_order_by(self):
        """Tests the ordering of the output of the sq, and the outer query"""

        # the graph has two nodes: (:N {name: 'Raz', v: 2}) and
        # (:N {name: 'Raz', v: 5})

        res = self.graph.query(
            """
            CALL {
                UNWIND range(1, 7) AS x
                MATCH (n:N {v: x})
                RETURN n ORDER BY n.v ASC
            }
            RETURN n
            """
        )

        # validate the results
        self.env.assertEquals(len(res.result_set), 2)
        self.env.assertEquals(res.result_set[0][0],
        Node(labels='N', properties={'name': 'Raz', 'v': 5}))
        self.env.assertEquals(res.result_set[1][0],
        Node(labels='N', properties={'name': 'Raz', 'v': 6}))

        # same query, but with descending order
        res = self.graph.query(
            """
            CALL {
                UNWIND range(1, 7) AS x
                MATCH (n:N {v: x})
                RETURN n ORDER BY n.v DESC
            }
            RETURN n
            """
        )

        # validate the results
        self.env.assertEquals(len(res.result_set), 2)
        self.env.assertEquals(res.result_set[0][0],
        Node(labels='N', properties={'name': 'Raz', 'v': 6}))
        self.env.assertEquals(res.result_set[1][0],
        Node(labels='N', properties={'name': 'Raz', 'v': 5}))

        # multiple Sort operations, in and outside the CallSubquery op
        res = self.graph.query(
            """
            CALL {
                UNWIND range(1, 7) AS x
                MATCH (n:N {v: x})
                RETURN n ORDER BY n.v ASC
            }
            RETURN n ORDER BY n.v ASC
            """
        )

        # validate the results
        self.env.assertEquals(len(res.result_set), 2)
        self.env.assertEquals(res.result_set[0][0],
        Node(labels='N', properties={'name': 'Raz', 'v': 5}))
        self.env.assertEquals(res.result_set[1][0],
        Node(labels='N', properties={'name': 'Raz', 'v': 6}))

    def test13_midrun_fail(self):
        """Tests that mid-run fails are recovered correctly (free'd)"""

        query = """
        UNWIND [1, 0, 3] as x
        CALL {
            WITH x
            CREATE (:TEMP {v: 5 / x})
            RETURN x AS innerReturn
        }
        RETURN innerReturn
        """
        self.expect_error(query, "Division by zero")

        # make sure the TEMP node was deleted
        res = self.graph.query("MATCH (n:TEMP) RETURN n")
        self.env.assertEquals(len(res.result_set), 0)

        # eager non-returning case
        query = """
        UNWIND [1, 0, 3] as x
        CALL {
            WITH x
            CREATE (:TEMP {v: 5 / x})
        }
        """
        self.expect_error(query, "Division by zero")

        # make sure the TEMP node was deleted
        res = self.graph.query("MATCH (n:TEMP) RETURN n")
        self.env.assertEquals(len(res.result_set), 0)

    def test14_nested_call_subquery(self):
        """Tests that embedded call subqueries are handled correctly."""

        query_to_expected_result = [
            ("""
            UNWIND [0, 1, 2] AS x
            CALL {
                CALL {
                    RETURN 1 AS one
                }
                RETURN 2 AS two, one
            }
            RETURN one, two, x
            """
            , [[1, 2, 0],[1, 2, 1],[1, 2, 2]]),
            ("""
            UNWIND [0, 1, 2] AS x 
            CALL {
                WITH x
                CALL {
                    RETURN 1 AS one
                }
                RETURN one, 2 AS two
            } 
            RETURN one, two, x
            """
            , [[1, 2, 0],[1, 2, 1],[1, 2, 2]]),
            ("""
            UNWIND [0, 1, 2] AS x
            CALL {
                CALL {
                    RETURN 1 AS x
                }
                RETURN max(x) AS y
            }
            RETURN y, x ORDER BY x ASC
            """
            , [[1, 0], [1, 1], [1, 2]])
        ]
        for query, expected_result in query_to_expected_result:
            self.get_res_and_assertEquals(query, expected_result)

    def test15_complex(self):
        """A more complex test, using more data and inner Call {} calls"""

        # clean db
        self.graph = self.db.select_graph(GRAPH_ID)

        # create data
        self.graph.query("""
            CREATE (english: Language {name: 'English'}),
                   (spanish: Language {name: 'Spanish'}),
                   (french: Language {name: 'French'}),
                   (coastal: Property {name: 'Coastal'}),
                   (usa: Country {name: 'USA', extra_data: 'foo'}),
                   (cali: State {name: 'California'}),
                   (sacramento: City {name: 'Sacramento', extra_data: 'bar'}),
                   (ny: State {name: 'New York'}),
                   (nyc: City {name: 'New York City'}),
                   (sacramento)-[:CITY_IN_STATE]->(cali)-[:STATE_IN_COUNTRY]->(usa),
                   (cali)-[:HAS_STATE_PROP]->(coastal),
                   (nyc)-[:CITY_IN_STATE]->(ny)-[:STATE_IN_COUNTRY]->(usa),
                   (ny)-[:HAS_STATE_PROP]->(coastal),
                   (nyc)-[:HAS_CITY_PROP]->(coastal),
                   (canada: Country {name: 'Canada'}),
                   (ontario: State {name: 'Ontario', extra_data: 'baz'}),
                   (toronto: City {name: 'Toronto'}),
                   (bc: State {name: 'British Columbia'}),
                   (victoria: City {name: 'Victoria'}),
                   (toronto)-[:CITY_IN_STATE]->(ontario)-[:STATE_IN_COUNTRY]->(canada),
                   (victoria)-[:CITY_IN_STATE]->(bc)-[:STATE_IN_COUNTRY]->(canada),
                   (bc)-[:HAS_STATE_PROP]->(coastal),
                   (victoria)-[:HAS_CITY_PROP]->(coastal),
                   (canada)-[:OFFICIAL_LANGUAGE]->(english),
                   (canada)-[:OFFICIAL_LANGUAGE]->(french),
                   (mexico: Country {name: 'Mexico'}),
                   (coahuila: State {name: 'Coahuila'}),
                   (saltillo: City {name: 'Saltillo'}),
                   (jalisco: State {name: 'Jalisco'}),
                   (guadalajara: City {name: 'Guadalajara'}),
                   (saltillo)-[:CITY_IN_STATE]->(coahuila)-[:STATE_IN_COUNTRY]->(mexico),
                   (guadalajara)-[:CITY_IN_STATE]->(jalisco)-[:STATE_IN_COUNTRY]->(mexico),
                   (jalisco)-[:HAS_STATE_PROP]->(coastal),
                   (mexico)-[:OFFICIAL_LANGUAGE]->(spanish)"""
        )

        query = """
        MATCH (c:City)-[:CITY_IN_STATE]->(state)-[:STATE_IN_COUNTRY]->(country)-[:OFFICIAL_LANGUAGE]->({name: 'English'})
        WITH collect(distinct country) as counties
        UNWIND counties as country
        MATCH (country)-[:OFFICIAL_LANGUAGE]->(lang)
        WITH country, collect(distinct lang.name) as langs
        MATCH (country)<-[:STATE_IN_COUNTRY]->(state)
        CALL {
            WITH state
            MATCH (state)<-[:CITY_IN_STATE]-(city:City)
            CALL {
                WITH city
                RETURN {type: labels(city)[0], name: city.name} as cityDetails
            }
            WITH state, collect(cityDetails) as citiesDetails
            RETURN {type: labels(state)[0], name:state.name, cities:citiesDetails} as stateDetails
        }
        WITH country, langs, collect(stateDetails) as statesDetails
        ORDER BY statesDetails ASC
        RETURN {name: country.name, langs: langs, states: statesDetails}
        """

        res = self.graph.query(query)

        # assert results
        expected_res = {'name': 'Canada',
                        'langs': ['French', 'English'],
                        'states':
                            [OrderedDict({'type': 'State',
                                          'name': 'Ontario',
                                          'cities':
                                            [OrderedDict({'type': 'City', 'name': 'Toronto'})]}),
                             OrderedDict({'type': 'State',
                                          'name': 'British Columbia',
                                          'cities':
                                            [OrderedDict({'type': 'City', 'name': 'Victoria'})]})]}

        self.env.assertEquals(res.result_set[0][0], expected_res)

    def test16_rewrite_same_clauses(self):
        """Tests that the AST-rewriting of same consecutive clauses works
        properly in a subquery"""

        # Test SET compression
        query = """
            CALL {
                MATCH (x:X)
                SET x.v = 3 
                SET x.t ='a'
            }"""
        plan = self.graph.explain(query)
        # make sure that Update is called only once
        _assert_subquery_contains_single(plan, "Update", self.env)

        # Test REMOVE compression
        query = """
            CALL {
                MATCH (x:X)
                REMOVE x.v
                REMOVE x.t
            }"""
        plan = self.graph.explain(query)
        # make sure that Update is called only once
        _assert_subquery_contains_single(plan, "Update", self.env)

        # Test DELETE compression
        query = """
            CALL {
                MATCH (x:X)
                MATCH (m:M)
                DELETE x
                DELETE m
            }
            """
        plan = self.graph.explain(query)
        # make sure that DELETE is called only once
        _assert_subquery_contains_single(plan, "Delete", self.env)

    def test17_leading_with(self):
        """Tests that we can use leading WITH queries with non-simple
        projections if they do not import any outer-scope data"""

        query_to_expected_result = [
            ("""
            CALL {
                WITH 1 AS b 
                RETURN b
            }
            RETURN b
            """
            , [[1]]),
            ("""
            CALL {
                WITH {} AS b 
                RETURN b
            }
            RETURN b
            """
            , [[{}]]),
            ("""
            CALL {
                WITH 'a' AS b 
                RETURN b
            } 
            RETURN b"""
            , [['a']]),
            ("""
            CALL { 
                WITH ['foo', 'bar'] AS l1 
                RETURN l1
            } 
            RETURN l1
            """
            , [[['foo', 'bar']]]),
            ("""
            CALL {
                WITH toUpper('a') AS A, 5 - 2 AS num
                RETURN A, num
            }
            RETURN *
            """, [['A', 3]])
        ]
        for query, expected_result in query_to_expected_result:
            self.get_res_and_assertEquals(query, expected_result)

    def test18_returning_aggregations(self):
        """Tests that we deal properly with returning aggregations instead of
        regular projections"""

        self.graph.delete()

        # create a node with label N
        self.graph.query("CREATE (:N)")

        query = """
        MATCH (n)
        CALL {
            WITH n
            OPTIONAL MATCH (m:M)
            SET m.v = 1
            RETURN collect(n) AS cn
        }
        WITH n, cn
        UNWIND cn as ns
        RETURN ns
        """

        res = self.graph.query(query)

        # assert results
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(res.result_set[0][0], Node(labels='N'))

        query = """
        UNWIND ['a', 'b', 'c', 'a', 'b', 'b'] AS x
        CALL {
            WITH x
            RETURN toUpper(x) AS key, count(1) AS count
        }
        RETURN key, count ORDER BY key
        """

        res = self.graph.query(query)

        # assert results
        self.env.assertEquals(len(res.result_set), 6)
        self.env.assertEquals(res.result_set[0][0], 'A')
        self.env.assertEquals(res.result_set[0][1], 1)
        self.env.assertEquals(res.result_set[1][0], 'A')
        self.env.assertEquals(res.result_set[1][1], 1)

        self.env.assertEquals(res.result_set[2][0], 'B')
        self.env.assertEquals(res.result_set[2][1], 1)
        self.env.assertEquals(res.result_set[3][0], 'B')
        self.env.assertEquals(res.result_set[3][1], 1)
        self.env.assertEquals(res.result_set[4][0], 'B')
        self.env.assertEquals(res.result_set[4][1], 1)

        self.env.assertEquals(res.result_set[5][0], 'C')
        self.env.assertEquals(res.result_set[5][1], 1)


        query = """UNWIND [ [1,2], [1,3], [1,2] ] AS x
            CALL {
                WITH x
                RETURN x[0] AS k, max(x[1]) AS m
            }
            RETURN x, m"""

        res = self.graph.query(query).result_set

        # assert results
        # expecting 3 records
        self.env.assertEquals(len(res), 3)

        self.env.assertEquals(res[0][0], [1, 2])
        self.env.assertEquals(res[0][1], 2)
        self.env.assertEquals(res[1][0], [1, 3] )
        self.env.assertEquals(res[1][1], 3)
        self.env.assertEquals(res[2][0], [1, 2])
        self.env.assertEquals(res[2][1], 2)

    def test19_optional_match(self):
        """Tests that we deal properly with `OPTIONAL MATCH` clauses in a
        subquery"""

        res = self.graph.query (
            """
            CALL { 
                UNWIND ['S','T'] AS l1 
                OPTIONAL MATCH (n:XX) 
                WITH n 
                RETURN n
            } 
            RETURN n
            """
        )

        # validate the results
        self.env.assertEquals(len(res.result_set), 2)
        self.env.assertEquals(res.result_set[0][0], None)
        self.env.assertEquals(res.result_set[1][0], None)

    def test20_aggregation_in_subquery(self):
        """Tests that we deal properly with aggregations in a subquery"""

        # Create 3 nodes
        self.graph.query("UNWIND range(1, 3) AS i CREATE (n:A {v:i})")
        
        query_to_expected_result = [
            ("""
            CALL {
                MATCH (n:A) 
                    WHERE n.v % 2 = 0 
                WITH max(n.v) AS cn
                RETURN cn
            } 
            RETURN cn
            """ 
            , [[2]]),
            ("""
            CALL {
                OPTIONAL MATCH (n:A)
                    WHERE n.v % 2 = 0
                WITH collect(n.v) AS cn
                OPTIONAL MATCH (m:A)
                    WHERE m.v % 2 = 1
                WITH sum(m.v) AS cm, cn
                RETURN cn, cm
            }
            RETURN cn, cm"""
            , [[[2], 4]])
        ]
        for query, expected_result in query_to_expected_result:
            self.get_res_and_assertEquals(query, expected_result)

    def test21_union(self):
        """Tests that UNION works properly within a subquery"""

        self.graph.delete()

        # a simple subquery, returning 2 rows
        res = self.graph.query(
            """
            CALL {
                RETURN 1 AS num
                UNION
                RETURN 2 AS num
            }
            RETURN num
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 2)
        self.env.assertEquals(res.result_set[0][0], 1)
        self.env.assertEquals(res.result_set[1][0], 2)

        # a simple subquery, using input from the outer query
        res = self.graph.query(
            """
            UNWIND range(1, 2) AS i
            WITH i AS i, i AS x
            CALL {
                WITH i
                RETURN i AS num
                UNION
                WITH x
                RETURN x + 1 AS num
            }
            RETURN i, num ORDER BY i, num ASC
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 4)
        self.env.assertEquals(res.result_set[0][0], 1)
        self.env.assertEquals(res.result_set[0][1], 1)
        self.env.assertEquals(res.result_set[1][0], 1)
        self.env.assertEquals(res.result_set[1][1], 2)
        self.env.assertEquals(res.result_set[2][0], 2)
        self.env.assertEquals(res.result_set[2][1], 2)
        self.env.assertEquals(res.result_set[3][0], 2)
        self.env.assertEquals(res.result_set[3][1], 3)

        # create nodes in both branches of the UNION
        res = self.graph.query(
            """
            CALL {
                CREATE (n:N {v: 1})
                RETURN n AS node
                UNION
                CREATE (m:M {v: 2})
                RETURN m AS node
            }
            RETURN node
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 2)
        self.env.assertEquals(res.result_set[0][0], Node(labels='N',
            properties={'v': 1}))
        self.env.assertEquals(res.result_set[1][0], Node(labels='M',
            properties={'v': 2}))

        # match nodes in one branch, and create nodes in the other
        res = self.graph.query(
            """
            CALL {
                MATCH (n:N)
                RETURN n AS node
                UNION
                CREATE (m:M {v: 2})
                RETURN m AS node
            }
            RETURN node
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 2)
        self.env.assertEquals(res.result_set[0][0], Node(labels='N',
            properties={'v': 1}))
        self.env.assertEquals(res.result_set[1][0], Node(labels='M',
            properties={'v': 2}))

        # use bound data
        res = self.graph.query(
            """
            UNWIND range(1, 2) AS i
            CALL {
                WITH i
                CREATE (n:TEMP {v: i})
                RETURN n.v AS num
                UNION ALL
                RETURN 1 AS num
            }
            RETURN i, num ORDER BY i, num ASC
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 4)
        self.env.assertEquals(res.result_set[0][0], 1)
        self.env.assertEquals(res.result_set[0][1], 1)
        self.env.assertEquals(res.result_set[1][0], 1)
        self.env.assertEquals(res.result_set[1][1], 1)
        self.env.assertEquals(res.result_set[2][0], 2)
        self.env.assertEquals(res.result_set[2][1], 1)
        self.env.assertEquals(res.result_set[3][0], 2)
        self.env.assertEquals(res.result_set[3][1], 2)

        # match nodes in both branches
        # the graph contains: (:N {v: 1}), (:M {v: 2}), (:M {v: 2})
        res = self.graph.query(
            """
            CALL {
                MATCH (n:N)
                RETURN n AS node
                UNION
                MATCH (m:M)
                RETURN m AS node
            }
            RETURN node ORDER BY node.v ASC
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 3)
        self.env.assertEquals(res.result_set[0][0], Node(labels='N',
            properties={'v': 1}))
        self.env.assertEquals(res.result_set[1][0], Node(labels='M',
            properties={'v': 2}))
        self.env.assertEquals(res.result_set[2][0], Node(labels='M',
            properties={'v': 2}))

        # simple embedded call with UNION
        res = self.graph.query(
            """
            CALL {
                CALL {
                    RETURN 1 AS num
                    UNION
                    RETURN 2 AS num
                }
                RETURN num
            }
            RETURN num
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 2)
        self.env.assertEquals(res.result_set[0][0], 1)
        self.env.assertEquals(res.result_set[1][0], 2)

        # simple embedded call with UNION
        res = self.graph.query(
            """
            CALL {
                CALL {
                    RETURN 1 AS num
                    UNION
                    RETURN 2 AS num
                }
                RETURN num
            }
            RETURN num, 5
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 2)
        self.env.assertEquals(res.result_set[0][0], 1)
        self.env.assertEquals(res.result_set[0][1], 5)
        self.env.assertEquals(res.result_set[1][0], 2)
        self.env.assertEquals(res.result_set[1][1], 5)

        # TODO:
        # this is a subquery that will require a change for the Join operation,
        # as it requires the changes of one input record to be visible to the
        # next input record

        # self.graph.delete()

        # # create two nodes with label N and property v=1, v=2
        # self.graph.query("CREATE (:N {v: 1}), (:N {v: 2})")

        # res = self.graph.query(
        #     """
        #     MATCH (n:N)
        #     CALL {
        #         WITH n
        #         MERGE (:N {v: n.v + 1})
        #         RETURN 1 AS ret
        #         UNION
        #         WITH n
        #         CREATE (:N {v: n.v + 2})
        #         RETURN 1 AS ret
        #     }
        #     RETURN count(1)
        #     """
        # )

        # # assert results
        # self.env.assertEquals(len(res.result_set), 1)
        # self.env.assertEquals(res.result_set[0][0], 2)
        # self.env.assertEquals(res.nodes_created, 2)

        # union and aggregation function
        res = self.graph.query (
            """
            CALL {
                RETURN 1 AS v
                UNION
                MATCH(m:SUM)
                RETURN tointeger(sum(m.v)) AS v
            }
            RETURN v ORDER BY v ASC
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 2)
        self.env.assertEquals(res.result_set[0][0], 0)
        self.env.assertEquals(res.result_set[1][0], 1)

        # one branch is eager and the other is not
        res = self.graph.query (
            """
            CALL {
                WITH 0 AS i
                RETURN i AS v
                UNION
                CREATE (n:EAGER {v:1})
                RETURN n.v AS v
            }
            RETURN v ORDER BY v ASC
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 2)
        self.env.assertEquals(res.result_set[0][0], 0)
        self.env.assertEquals(res.result_set[1][0], 1)
        self.env.assertEquals(res.nodes_created, 1)
        self.env.assertEquals(res.properties_set, 1)

        # both branches are eager
        res = self.graph.query (
            """
            CALL {
                MERGE (m:EAGER {v:0})
                RETURN m.v AS v
                UNION
                MERGE (n:EAGER {v:1})
                SET n.v = 2
                RETURN n.v AS v
            }
            RETURN v ORDER BY v ASC
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 2)
        self.env.assertEquals(res.result_set[0][0], 0)
        self.env.assertEquals(res.result_set[1][0], 2)
        self.env.assertEquals(res.nodes_created, 1)
        self.env.assertEquals(res.properties_set, 2)

        # both branches are not eager
        self.graph.query("CREATE (:X {v: 1})")

        res = self.graph.query (
            """
            CALL {
                MATCH (n:X)
                RETURN n.v AS v
                UNION
                MATCH (n:X {v: 1})
                RETURN n.v AS v
            }
            RETURN v
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(res.result_set[0][0], 1)
        self.env.assertEquals(res.nodes_created, 0)
        self.env.assertEquals(res.properties_set, 0)

        # UNION ALL with more than 2 branches
        res = self.graph.query (
            """
            CALL {
                MERGE (n:EAGER {v:7})
                WITH n
                RETURN n.v AS v
                UNION ALL
                MATCH (n:X {v: 1})
                RETURN n.v AS v
                UNION ALL
                UNWIND range(1, 3) AS i
                RETURN i AS v
                UNION ALL
                CALL {
                    MERGE (n:X {v: 1})
                    RETURN n.v AS j
                }
                RETURN j AS v
            }
            RETURN v
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 6)
        self.env.assertEquals(res.result_set[0][0], 7)
        self.env.assertEquals(res.result_set[1][0], 1)
        self.env.assertEquals(res.result_set[2][0], 1)
        self.env.assertEquals(res.result_set[3][0], 2)
        self.env.assertEquals(res.result_set[4][0], 3)
        self.env.assertEquals(res.result_set[5][0], 1)
        self.env.assertEquals(res.nodes_created, 1)
        self.env.assertEquals(res.properties_set, 1)

        # single eager & returning branch out of 2, using input data inside and
        # outside the subquery
        res = self.graph.query(
            """
            UNWIND [1, 2] AS data
            CALL {
                WITH data
                RETURN data AS v
                UNION
                WITH data
                UNWIND data AS i
                RETURN collect(data) AS v
            }
            RETURN data, v ORDER BY data, v ASC
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 4)
        self.env.assertEquals(res.result_set[0][0], 1)
        self.env.assertEquals(res.result_set[0][1], [1])
        self.env.assertEquals(res.result_set[1][0], 1)
        self.env.assertEquals(res.result_set[1][1], 1)
        self.env.assertEquals(res.result_set[2][0], 2)
        self.env.assertEquals(res.result_set[2][1], [2])
        self.env.assertEquals(res.result_set[3][0], 2)
        self.env.assertEquals(res.result_set[3][1], 2)

        # both branches are eager and returning
        res = self.graph.query(
            """
            UNWIND [1, 2] AS data
            CALL {
                WITH data
                RETURN collect(data) AS v
                UNION ALL
                WITH data
                RETURN collect(data) AS v
            }
            RETURN data, v ORDER BY data ASC, v ASC
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 4)
        for i, subres in enumerate([[1, [1]], [1, [1]], [2, [2]], [2, [2]]]):
            self.env.assertEquals(res.result_set[i], subres)

        # same query with Distinct (UNION instead of UNION ALL)
        res = self.graph.query(
            """
            UNWIND [1, 2] AS data
            CALL {
                WITH data
                RETURN collect(data) AS v
                UNION
                WITH data
                RETURN collect(data) AS v
            }
            RETURN data, v ORDER BY data ASC, v ASC
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 2)
        for i in [1, 2]:
            self.env.assertEquals(res.result_set[i-1][0], i)
            self.env.assertEquals(res.result_set[i-1][1], [i])

        # mix union and union all in call sub query
        res = self.graph.query(
            """
            CALL { 
                RETURN 0 AS n1
                UNION ALL
                RETURN 0 AS n1
            }
            RETURN 0 AS n2
            UNION
            RETURN 0 AS n2
            """
        )
        self.env.assertEquals(res.result_set, [[0]])

    def test22_indexes(self):
        """Tests that operations on indexes are properly executed (and reset)
        in subqueries"""

        self.graph.delete()

        # polulate the graph
        query = """UNWIND range(10,20) AS i
            CREATE (n:N {v:tostring(i)})-[:R]->(m:M {v:tostring(i+1)})"""
        self.graph.query(query)
        create_node_range_index(self.graph, "N", "v", sync=True)

        # use the index in a scan as the lhs of a CallSubquery op, which
        # contains an eager operation
        query = """
        MATCH (n:N {v:'10'})
        CALL {
            WITH n
            SET n.v = '11'
        }
        RETURN n.v
        """

        # assert that we indeed utilize the index to find the node
        plan = self.graph.explain(query)
        index_scan = locate_operation(plan.structured_plan,
            "Node By Index Scan")
        self.env.assertNotEqual(index_scan, None)

        res = self.graph.query(query)

        # assert results
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(res.result_set[0][0], '11')

    def test25_named_paths(self):
        """Tests that named paths are handled correctly, when defined/referred
        inside or outside of a subquery"""

        self.graph.delete()

        # create a node with label N
        self.graph.query("CREATE (:N {v: 1})")
        node = Node(labels='N', properties={'v': 1})

        # refer to a named path defined outside of a subquery, from within the
        # subquery
        query = """
        MATCH p = (n:N)
        CALL {
            WITH p
            RETURN nodes(p)[0] AS s
        }
        RETURN s
        """

        res = self.graph.query(query)

        # assert results
        self.env.assertEquals(len(self.graph.query(query).result_set), 1)
        self.env.assertEquals(res.result_set[0][0], node)

        # refer a named path defined inside of a subquery, from within the
        # subquery
        query = """
        CALL {
            MATCH p = (n:N)
            RETURN nodes(p)[0] AS s
        }
        RETURN s
        """

        res = self.graph.query(query)

        # assert results
        self.env.assertEquals(len(self.graph.query(query).result_set), 1)
        self.env.assertEquals(res.result_set[0][0], node)

        # return a path from a subquery and refer to it outside of the subquery
        query = """
        CALL {
            MATCH p = (n:N)
            RETURN p
        }
        RETURN nodes(p)[0]
        """

        res = self.graph.query(query)

        # assert results
        self.env.assertEquals(len(self.graph.query(query).result_set), 1)
        self.env.assertEquals(res.result_set[0][0], node)

        # refer to a named path defined in a subquery, from a nested subquery
        query = """
        CALL {
            MATCH p = (n:N)
            CALL {
                WITH p
                RETURN nodes(p)[0] AS s
            }
            RETURN s
        }
        RETURN s
        """

        res = self.graph.query(query)

        # assert results
        self.env.assertEquals(len(self.graph.query(query).result_set), 1)
        self.env.assertEquals(res.result_set[0][0], node)

        # import named paths inside a subquery containing a UNION clause
        query = """
        MATCH p = (n:N)
        CALL {
            RETURN {v: 2} AS node
            UNION
            WITH p
            RETURN nodes(p)[0] AS node
        }
        RETURN node ORDER BY node.v ASC
        """

        res = self.graph.query(query)

        # assert results
        self.env.assertEquals(len(self.graph.query(query).result_set), 2)
        self.env.assertEquals(res.result_set[0][0], node)
        self.env.assertEquals(res.result_set[1][0], OrderedDict([('v', 2)]))

    def test26_eager_returning(self):
        """Tests the eager and returning case of Call {}"""

        self.graph.delete()

        query = """
        OPTIONAL MATCH (m)
        CALL {
            CREATE (n:N {name: 'Raz'})
            RETURN n
        }
        RETURN n
        """

        res = self.graph.query(query)

        # assert results
        self.env.assertEquals(res.nodes_created, 1)
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(res.result_set[0][0],
            Node(labels='N', properties={'name': 'Raz'}))

        # use memory from outer scope after the subquery
        query = """
        MATCH (n)
        CALL {
            CREATE (m:M {name: 'Moshe'})
            RETURN m
        }
        RETURN n, m
        """

        res = self.graph.query(query)

        # assert results
        self.env.assertEquals(res.nodes_created, 1)
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(len(res.result_set[0]), 2)
        self.env.assertEquals(res.result_set[0][0], Node(labels='N',
            properties={'name': 'Raz'}))
        self.env.assertEquals(res.result_set[0][1], Node(labels='M',
            properties={'name': 'Moshe'}))

        # nested eager & returning subquery
        query = """
        MATCH (n:N)
        CALL {
            CREATE (m:M {name: 'Moshe'})
            WITH m
            CALL {
                CREATE (o:O {name: 'Omer'})
                RETURN o
            }
            RETURN m, o
        }
        RETURN n, m, o
        """

        res = self.graph.query(query)

        # assert results
        self.env.assertEquals(res.nodes_created, 2)
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(len(res.result_set[0]), 3)
        self.env.assertEquals(res.result_set[0][0], Node(labels='N',
            properties={'name': 'Raz'}))
        self.env.assertEquals(res.result_set[0][1], Node(labels='M',
            properties={'name': 'Moshe'}))
        self.env.assertEquals(res.result_set[0][2], Node(labels='O',
            properties={'name': 'Omer'}))

        # highly nested eager & returning subquery
        query = """
        MATCH (n:N)
        CALL {
            CREATE (m:M {name: 'Moshe'})
            WITH m
            CALL {
                CREATE (o:O {name: 'Omer'})
                WITH o
                CALL {
                    CREATE (p:P {name: 'Pini'})
                    RETURN p
                }
                RETURN o, p
            }
            RETURN m, o, p
        }
        RETURN n, m, o, p
        """

        res = self.graph.query(query)

        # assert results
        self.env.assertEquals(res.nodes_created, 3)
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(len(res.result_set[0]), 4)
        self.env.assertEquals(res.result_set[0][0], Node(labels='N',
            properties={'name': 'Raz'}))
        self.env.assertEquals(res.result_set[0][1], Node(labels='M',
            properties={'name': 'Moshe'}))
        self.env.assertEquals(res.result_set[0][2], Node(labels='O',
            properties={'name': 'Omer'}))
        self.env.assertEquals(res.result_set[0][3], Node(labels='P',
            properties={'name': 'Pini'}))

        # multiple eager & returning subqueries sequentially
        query = """
        MATCH (n:N)
        CALL {
            CREATE (m:M {name: 'Moshe'})
            RETURN m
        }
        CALL {
            CREATE (o:O {name: 'Omer'})
            RETURN o
        }
        RETURN n, m, o
        """

        res = self.graph.query(query)

        # assert results
        self.env.assertEquals(res.nodes_created, 2)
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(len(res.result_set[0]), 3)
        self.env.assertEquals(res.result_set[0][0], Node(labels='N',
            properties={'name': 'Raz'}))
        self.env.assertEquals(res.result_set[0][1], Node(labels='M',
            properties={'name': 'Moshe'}))
        self.env.assertEquals(res.result_set[0][2], Node(labels='O',
            properties={'name': 'Omer'}))

        # nested eager & returning subquery in a non-(eager & returning)
        # subquery
        query = """
        MATCH (n:N)
        CALL {
            CALL {
                CREATE (m:M {name: 'Moshe'})
                RETURN m
            }
            RETURN m
        }
        RETURN n, m
        """

        res = self.graph.query(query)

        # assert results
        self.env.assertEquals(res.nodes_created, 1)
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(len(res.result_set[0]), 2)
        self.env.assertEquals(res.result_set[0][0], Node(labels='N',
            properties={'name': 'Raz'}))
        self.env.assertEquals(res.result_set[0][1], Node(labels='M',
            properties={'name': 'Moshe'}))

        # outer {} is not eager, inner is
        res = self.graph.query(
            """
            CALL {
                WITH 1 AS x
                CALL {
                    UNWIND range(1, 5) AS x
                    RETURN collect(x) as y
                }
                RETURN x, y
            }
            RETURN x, y
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(len(res.result_set[0]), 2)
        self.env.assertEquals(res.result_set[0][0], 1)
        self.env.assertEquals(res.result_set[0][1], list(range(1, 6)))

        # outer {} is eager, but not returning, inner is
        res = self.graph.query(
            """
            MATCH (n:N)
            CALL {
                WITH n
                WITH n, 1 AS x
                CALL {
                    CREATE (m:M {name: 'Moshe'})
                    RETURN m
                }
                SET n.v = 1
            }
            RETURN n
            """
        )

        # assert results
        self.env.assertEquals(res.nodes_created, 1)
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(len(res.result_set[0]), 1)
        self.env.assertEquals(res.result_set[0][0], Node(labels='N',
            properties={'name': 'Raz', 'v': 1}))

        # same concept as the above test, using input data
        # create 3 nodes (:TEMP {v: 1\2\3})
        res = self.graph.query("UNWIND range(1, 3) AS x CREATE (n:TEMP {v: x})")
        self.env.assertEquals(res.nodes_created, 3)

        res = self.graph.query(
            """
            UNWIND range(1, 3) AS x
            CALL {
                WITH x
                MATCH (n:TEMP {v: x})
                CALL {
                    WITH n
                    UNWIND range(0, n.v) AS num
                    RETURN collect(num) AS y
                }
                RETURN y
            }
            RETURN x, y ORDER BY x
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 3)
        self.env.assertEquals(res.result_set[0][0], 1)
        self.env.assertEquals(res.result_set[0][1], list(range(0, 2)))
        self.env.assertEquals(res.result_set[1][0], 2)
        self.env.assertEquals(res.result_set[1][1], list(range(0, 3)))
        self.env.assertEquals(res.result_set[2][0], 3)
        self.env.assertEquals(res.result_set[2][1], list(range(0, 4)))

    def test27_read_no_with_after_writing_subquery(self):
        """Tests that a read clause following a writing subquery is handled
        correctly with\without a separating `WITH` clause"""

        self.graph.delete()

        # using a separating `WITH`
        res = self.graph.query(
            """
            CALL {
                CREATE (n:N {name: 'Roi'})
                RETURN n
            }
            WITH n
            MATCH (n2:N)
            RETURN n2
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(res.result_set[0][0], Node(labels='N',
            properties={'name': 'Roi'}))

        # delete the node
        res = self.graph.query("MATCH (n) DELETE n")
        self.env.assertEquals(res.nodes_deleted, 1)

        # same query, without a separating `WITH`
        res = self.graph.query(
            """
            CALL {
                CREATE (n:N {name: 'Roi'})
                RETURN n
            }
            MATCH (n2:N)
            RETURN n2
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(res.result_set[0][0], Node(labels='N',
            properties={'name': 'Roi'}))

        # delete the node
        res = self.graph.query("MATCH (n) DELETE n")
        self.env.assertEquals(res.nodes_deleted, 1)

        # same query, without a separating `WITH`, and without a `RETURN` clause
        res = self.graph.query(
            """
            CALL {
                CREATE (n:N {name: 'Roi'})
            }
            MATCH (n2:N)
            RETURN n2
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(res.result_set[0][0], Node(labels='N',
            properties={'name': 'Roi'}))

    def test28_reset(self):
        """Tests that the resetting of a call {} operation is handled
        correctly"""

        self.graph.delete()

        # create 2 nodes (:N {v: 2\4})
        res = self.graph.query("UNWIND [2, 4] AS x CREATE (n:N {v: x})")

        # in the following query, the inner call {} is reset (lhs of outer {})
        res = self.graph.query(
            """
            UNWIND [1, 2 ,3, 4] AS x
            CALL {
                WITH x
                CALL {
                    WITH x
                    MATCH (n {v: x})
                    RETURN n
                }
            RETURN n
            }
            RETURN x, n
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 2)
        self.env.assertEquals(res.result_set[0][0], 2)
        self.env.assertEquals(res.result_set[0][1], Node(labels='N',
            properties={'v': 2}))
        self.env.assertEquals(res.result_set[1][0], 4)
        self.env.assertEquals(res.result_set[1][1], Node(labels='N',
            properties={'v': 4}))

    def test29_rewrite_star_projections(self):
        """Tests that star projections within call {}are rewritten correctly"""

        self.graph.delete()

        # import data with *
        res = self.graph.query(
            """
            WITH 1 AS a, 2 AS b
            CALL {
                WITH *
                RETURN a + b AS c
            }
            RETURN a, b, c
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(len(res.result_set[0]), 3)
        for i in range(3):
            self.env.assertEquals(res.result_set[0][i], i + 1)

        # return with *
        res = self.graph.query(
            """
            CALL {
                WITH 1 AS a, 2 AS b
                RETURN *
            }
            RETURN a, b
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(len(res.result_set[0]), 2)
        for i in range(2):
            self.env.assertEquals(res.result_set[0][i], i + 1)

        # both import with * and return with *
        res = self.graph.query(
            """
            WITH 1 AS a, 2 AS b
            CALL {
                WITH *
                WITH a + b AS c
                RETURN *
            }
            RETURN a, b, c
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(len(res.result_set[0]), 3)
        for i in range(3):
            self.env.assertEquals(res.result_set[0][i], i + 1)

        # using union
        res = self.graph.query(
            """
            WITH 1 AS a
            CALL {
                WITH *
                RETURN a AS num
                UNION
                WITH *
                RETURN a + 1 AS num
            }
            RETURN a, num order by a, num
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 2)
        self.env.assertEquals(len(res.result_set[0]), 2)
        self.env.assertEquals(len(res.result_set[1]), 2)
        self.env.assertEquals(res.result_set[0][0], 1)
        self.env.assertEquals(res.result_set[0][1], 1)
        self.env.assertEquals(res.result_set[1][0], 1)
        self.env.assertEquals(res.result_set[1][1], 2)

        # embedded call {}
        query = """
            WITH 1 AS a
            CALL {
                WITH *
                CALL {
                    WITH *
                    RETURN a AS num
                    UNION
                    WITH *
                    RETURN a + 1 AS num
                }
                RETURN *
            }
            RETURN a, num order by a, num
            """

        self.expect_error(query, "Variable `a` already declared in outer scope")

        # intermediate with *

        # create a node with label N
        res = self.graph.query("CREATE (:N)")

        res = self.graph.query(
            """
            WITH 1 AS a, 2 AS b
            CALL {
                WITH *
                MATCH (n)
                WITH *
                RETURN n
            }
            RETURN a, b, n
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(len(res.result_set[0]), 3)
        self.env.assertEquals(res.result_set[0][0], 1)
        self.env.assertEquals(res.result_set[0][1], 2)
        self.env.assertEquals(res.result_set[0][2], Node(labels='N'))

        # create node and project it with star
        res = self.graph.query(
            """
            WITH 1 AS a, 2 AS b
            CALL {
                WITH *
                CREATE (n:C)
                WITH *
                RETURN n
            }
            RETURN a, b, n
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(len(res.result_set[0]), 3)
        self.env.assertEquals(res.result_set[0][0], 1)
        self.env.assertEquals(res.result_set[0][1], 2)
        self.env.assertEquals(res.result_set[0][2], Node(labels='C'))

        # merge node and project it with star
        res = self.graph.query(
            """
            WITH 1 AS a, 2 AS b
            CALL {
                WITH *
                MERGE (n:C)
                WITH *
                RETURN n
            }
            RETURN a, b, n
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(len(res.result_set[0]), 3)
        self.env.assertEquals(res.result_set[0][0], 1)
        self.env.assertEquals(res.result_set[0][1], 2)
        self.env.assertEquals(res.result_set[0][2], Node(labels='C'))

        # create node and return it with star
        res = self.graph.query(
            """
            WITH 1 AS a, 2 AS b
            CALL {
                WITH *
                CREATE (n:C)
                WITH n
                RETURN *
            }
            RETURN a, b, n
            """
        )

        # assert results
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(len(res.result_set[0]), 3)
        self.env.assertEquals(res.result_set[0][0], 1)
        self.env.assertEquals(res.result_set[0][1], 2)
        self.env.assertEquals(res.result_set[0][2], Node(labels='C'))

    def test30_surrounding_matches(self):
        """Tests that in case the call {} is surrounded by matches, the
        following match does not affect the input records to the call {} op"""

        self.graph.delete()

        # create two nodes with label N, and one with label I with increasing v
        # property values for the N nodes
        res = self.graph.query("CREATE (:N {v: 1}), (:N {v: 2}), (:I)")

        # query with a surrounding match clauses, such that the right-most match
        # is prioritized (less nodes with its label in the graph)
        query = """
            MATCH (n:N)
            CALL {
                RETURN 1 AS x
            }
            MATCH (i:I)
            RETURN n, i, x ORDER BY n.v ASC
            """

        # assert that (i:I) was not scanned and inserted to the call {} op
        plan = self.graph.explain(query)
        # make sure that CREATE is called only once
        label_scan = locate_operation(plan.structured_plan, "Node By Label Scan")
        self.env.assertIsNotNone(label_scan)
        cartesian = locate_operation(plan.structured_plan, "CartesianProduct")
        self.env.assertIsNone(cartesian)

        res = self.graph.query(query)
        # assert results
        n1 = Node(labels='N', properties={'v': 1})
        n2 = Node(labels='N', properties={'v': 2})
        n3 = Node(labels='I')
        self.env.assertEquals(len(res.result_set), 2)
        self.env.assertEquals(len(res.result_set[0]), 3)
        self.env.assertEquals(res.result_set[0][0], n1)
        self.env.assertEquals(res.result_set[0][1], n3)
        self.env.assertEquals(res.result_set[0][2], 1)
        self.env.assertEquals(res.result_set[1][0], n2)
        self.env.assertEquals(res.result_set[1][1], n3)
        self.env.assertEquals(res.result_set[1][2], 1)

    def test31_following_scans(self):
        """Tests that in case the call {} is followed by scans, the
        following scans are planned and executed properly"""

        self.graph.delete()

        # create the node (:N {v: 1})
        res = self.graph.query("CREATE (:N {v: 1})")

        # query with a match clause following a call {} clause with a scan
        res = self.graph.query(
            """
            CALL {
                MATCH (m:M)
                SET m.v = 2
            }
            MATCH (n:N)
            RETURN n
            """
        )

        # assert results
        n = Node(labels='N', properties={'v': 1})
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(len(res.result_set[0]), 1)
        self.env.assertEquals(res.result_set[0][0], n)

        # query with a match clause following a call {} clause with a scan,
        # using the same alias (reduce-scans optimization fixed)
        query = """
            CALL {
                MATCH (n:M)
                SET n.v = 2
            }
            MATCH (n:N)
            RETURN n
        """

        # assert that the execution-plan holds a label scan for :M, and not a
        # conditional-traverse. This is true if we have to label-scans in the
        # plan
        plan = self.graph.explain(query)
        n_label_scans = count_operation(plan.structured_plan,
            "Node By Label Scan")
        self.env.assertEquals(n_label_scans, 2)

        res = self.graph.query(query)
        # assert results
        self.env.assertEquals(len(res.result_set), 1)
        self.env.assertEquals(len(res.result_set[0]), 1)
        self.env.assertEquals(res.result_set[0][0], n)

        # query with a match clause, followed by a call {} clause with a scan
        # with the same alias, followed by a match clause with the same alias.
        # we expect the aliases before and after the call {} to be tied
        # (the `UNWIND` is placed to separate the scan and the call {}, since
        # this was a bug in the past)
        query = """
        MATCH (n:N)
        CALL {
            MATCH (n:M)
            SET n.v = 5
        }
        UNWIND [1] AS one
        MATCH (n:O)
        RETURN n
        """

        plan = self.graph.explain(query)
        self.env.assertEquals(count_operation(plan.structured_plan, "Node By Label Scan"), 2)
        self.env.assertEquals(count_operation(plan.structured_plan, "Conditional Traverse"), 1)

        # assert that the `O` label is scanned via cond-traverse
        scan = locate_operation(plan.structured_plan, "Conditional Traverse")
        self.env.assertEquals(str(scan), "Conditional Traverse | (n:O)->(n:O)")

        # return a bound variable from the call {} clause, with a scan on it
        # after
        query = """
            CALL {
                MATCH (n:M)
                SET n.v = 2
                RETURN n
            }
            MATCH (n:N)
            RETURN n
        """

        # assert that the execution-plan holds a cond-traverse for :N
        plan = self.graph.explain(query)
        scan = locate_operation(plan.structured_plan, "Conditional Traverse")
        self.env.assertEquals(str(scan), "Conditional Traverse | (n:N)->(n:N)")

    def test32_rewrite_call_subquery(self):
        self.graph.delete()

        # create the node (:N {v: 1})
        self.graph.query("CREATE (:N {v: 1})")

        res = self.graph.query("""
               MATCH (n)
               CALL {
                       FOREACH ( n2 IN [] | CREATE () )
                       RETURN 0 AS n3
                   UNION
                       RETURN 0 AS n3
               }
               RETURN 0""")
        self.env.assertEquals(res.result_set, [[0]])

    def test33_merge_after_call_subquery(self):
        self.graph.delete()

        # create the node (:N {v: 1})
        self.graph.query("CREATE (:N {v: 1})")
        res = self.graph.query("""
            CALL {
                RETURN 1 AS x
            }
            MERGE (n:N {v: 1})
            RETURN n.v
            """)
        self.env.assertEquals(res.result_set, [[1]])

    def test34_create_within_call(self):
        # The query will:
        # 1. Create an entity within a CALL sub query
        # 2. a Match outside the sub query will pull as many records as it can
        #    untill the sub query is depleted

        q = """CALL {
                        CREATE ()
                        RETURN 0 AS X
                    }
               MATCH ()--()
               RETURN 0"""

        res = self.graph.query(q)
        self.env.assertEquals(res.nodes_created, 1)
        self.env.assertEquals(len(res.result_set), 0)
