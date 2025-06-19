from common import *

node_names = ["A", "B", "C", "D"]

# A can reach 3 nodes, B can reach 2 nodes, C can reach 1 node
max_results = 6

GRAPH_ID = "variable_length_traversals"

class testVariableLengthTraversals(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)
        self.populate_graph()

    def populate_graph(self):
        nodes = []
        # Create nodes
        for idx, n in enumerate(node_names):
            node = Node(alias=f"n{idx}", labels="node", properties={"name": n})
            nodes.append(node)

        # Create edges
        edges = []
        for i in range(len(nodes) - 1):
            edges.append(Edge(nodes[i], "knows", nodes[i+1], properties={"connects": node_names[i] + node_names[i+1]}))

        nodes_str = [str(node) for node in nodes]
        edges_str = [str(edge) for edge in edges]
        self.graph.query(f"CREATE {','.join(nodes_str + edges_str)}")

    # Sanity check against single-hop traversal
    def test01_conditional_traverse(self):
        query = """MATCH (a)-[e]->(b)
                   RETURN a.name, e.connects, b.name
                   ORDER BY a.name, b.name"""
        actual_result = self.graph.query(query)
        expected_result = [['A', 'AB', 'B'],
                           ['B', 'BC', 'C'],
                           ['C', 'CD', 'D']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Traversal with no labels
    def test02_unlabeled_traverse(self):
        query = """MATCH (a)-[*]->(b)
                   RETURN a.name, b.name
                   ORDER BY a.name, b.name"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(len(actual_result.result_set), max_results)

        query = """MATCH (a)<-[*]-(b)
                   RETURN a, b
                   ORDER BY a.name, b.name"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(len(actual_result.result_set), max_results)

    # Traversal with labeled source
    def test03_source_labeled(self):
        query = """MATCH (a:node)-[*]->(b)
                   RETURN a.name, b.name
                   ORDER BY a.name, b.name"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(len(actual_result.result_set), max_results)

        query = """MATCH (a:node)<-[*]-(b)
                   RETURN a.name, b.name
                   ORDER BY a.name, b.name"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(len(actual_result.result_set), max_results)

    # Traversal with labeled dest
    def test04_dest_labeled(self):
        query = """MATCH (a)-[*]->(b:node)
                   RETURN a.name, b.name
                   ORDER BY a.name, b.name"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(len(actual_result.result_set), max_results)

        query = """MATCH (a)<-[*]-(b:node)
                   RETURN a.name, b.name
                   ORDER BY a.name, b.name"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(len(actual_result.result_set), max_results)

    # Attempt to traverse non-existent relationship type.
    def test05_invalid_traversal(self):
        query = """MATCH (a)-[:no_edge*]->(b) RETURN a.name"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(len(actual_result.result_set), 0)

    # Test bidirectional traversal
    def test06_bidirectional_traversal(self):
        query = """MATCH (a)-[*]-(b)
                   RETURN a.name, b.name
                   ORDER BY a.name, b.name"""
        actual_result = self.graph.query(query)
        # The undirected traversal should represent every combination twice.
        self.env.assertEquals(len(actual_result.result_set), max_results * 2)

    def test07_non_existing_edge_traversal_with_zero_length(self):
        # Verify that zero length traversals always return source, even for non existing edges.
        query = """MATCH (a)-[:not_knows*0..1]->(b)
                   RETURN a"""
        actual_result = self.graph.query(query)
        self.env.assertEquals(len(actual_result.result_set), 4)

    # Test traversal with a possibly-null source.
    def test08_optional_source(self):
        query = """OPTIONAL MATCH (a:fake)
                   OPTIONAL MATCH (a)-[*]->(b)
                   RETURN a.name, b.name
                   ORDER BY a.name, b.name"""
        actual_result = self.graph.query(query)
        expected_result = [[None, None]]
        self.env.assertEquals(actual_result.result_set, expected_result)

        query = """OPTIONAL MATCH (a:node {name: 'A'})
                   OPTIONAL MATCH (a)-[*]->(b {name: 'B'})
                   RETURN a.name, b.name
                   ORDER BY a.name, b.name"""
        actual_result = self.graph.query(query)
        expected_result = [['A', 'B']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Test traversals with filters on variable-length edges
    def test09_filtered_edges(self):
        # Test an inline equality predicate
        query = """MATCH (a)-[* {connects: 'BC'}]->(b)
                   RETURN a.name, b.name
                   ORDER BY a.name, b.name"""
        # The filter op should have been optimized out
        plan = str(self.graph.explain(query))
        self.env.assertNotIn("Filter", plan)
        actual_result = self.graph.query(query)
        expected_result = [['B', 'C']]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test a WHERE clause predicate
        query = """MATCH (a)-[e*]->(b)
                   WHERE e.connects IN ['BC', 'CD']
                   RETURN a.name, b.name
                   ORDER BY a.name, b.name"""
        # The filter op should have been optimized out
        plan = str(self.graph.explain(query))
        self.env.assertNotIn("Filter", plan)
        actual_result = self.graph.query(query)
        expected_result = [['B', 'C'],
                           ['B', 'D'],
                           ['C', 'D']]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test a WHERE clause predicate with an OR condition
        query = """MATCH (a)-[e*]->(b)
                   WHERE e.connects = 'BC' OR e.connects = 'CD'
                   RETURN a.name, b.name
                   ORDER BY a.name, b.name"""
        # The filter op should have been optimized out
        plan = str(self.graph.explain(query))
        self.env.assertNotIn("Filter", plan)
        actual_result = self.graph.query(query)
        # Expecting the same result
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test the concatenation of multiple predicates
        query = """MATCH (a)-[e*]->(b)
                   WHERE e.connects IN ['AB', 'BC', 'CD'] AND e.connects <> 'CD'
                   RETURN a.name, b.name
                   ORDER BY a.name, b.name"""
        # The filter op should have been optimized out
        plan = str(self.graph.explain(query))
        self.env.assertNotIn("Filter", plan)
        actual_result = self.graph.query(query)
        expected_result = [['A', 'B'],
                           ['A', 'C'],
                           ['B', 'C']]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Test the concatenation of AND and OR conditions
        query = """MATCH (a)-[e*]->(b)
                   WHERE e.connects IN ['AB', 'BC', 'CD'] AND (e.connects = 'AB' OR e.connects = 'BC')  AND e.connects <> 'CD'
                   RETURN a.name, b.name
                   ORDER BY a.name, b.name"""
        # The filter op should have been optimized out
        plan = str(self.graph.explain(query))
        self.env.assertNotIn("Filter", plan)
        actual_result = self.graph.query(query)
        expected_result = [['A', 'B'],
                           ['A', 'C'],
                           ['B', 'C']]
        self.env.assertEquals(actual_result.result_set, expected_result)

        # Validate that WHERE clause predicates are applied to edges lower than the minHops value
        query = """MATCH (a)-[e*2..]->(b)
                   WHERE e.connects <> 'AB'
                   RETURN a.name, b.name
                   ORDER BY a.name, b.name"""
        actual_result = self.graph.query(query)
        expected_result = [['B', 'D']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Test traversals with filters on variable-length edges in WITH...OPTIONAL MATCH constructs
    def test10_filtered_edges_after_segment_change(self):
        # Test a query that produces the subtree:
        #   Project
        #       Filter
        #           All Node Scan | (a)
        #   Optional
        #       Conditional Variable Length Traverse | (a)-[anon_0*1..INF]->(b)
        #           Argument
        #
        # The scan op and the variable-length traversal and its filter are
        # built in different ExecutionPlan segments. The segments must be
        # updated before cloning the Optional subtree,
        # or else the variable-length edge reference will be lost.
        query = """MATCH (a {name: 'A'})
                   WITH a
                   OPTIONAL MATCH (a)-[* {connects: 'AB'}]->(b)
                   RETURN a.name, b.name
                   ORDER BY a.name, b.name"""
        actual_result = self.graph.query(query)
        expected_result = [['A', 'B']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    # Test range-length edges
    def test11_range_length_edges(self):
        # clear previous data
        self.graph.delete()

        # populate graph
        # create a graph with 4 nodes
        # a->b
        # b->c
        # c->a
        # d->d
        query = """CREATE (a {v:'a'}), (b {v:'b'}), (c {v:'c'}), (d {v:'d'}),
                          (a)-[:R]->(b), (b)-[:R]->(c), (c)-[:R]->(a), (d)-[:R]->(d)"""

        actual_result = self.graph.query(query)

        # validation queries
        query_to_expected_result = {
            "MATCH p = (a {v:'a'})-[*2]-(c {v:'c'}) RETURN length(p)" : [[2]],
            "MATCH p = (a {v:'a'})-[*2..]-(c {v:'c'}) RETURN length(p)" : [[2]],
            "MATCH p = (a {v:'a'})-[*2..2]-(c {v:'c'}) RETURN length(p)" : [[2]],
            "MATCH p = (a {v:'a'})-[*]-(c {v:'c'}) WITH length(p) AS len RETURN len ORDER BY len" : [[1],[2]],
            "MATCH p = (a {v:'a'})-[*..]-(c {v:'c'}) WITH length(p) as len RETURN len ORDER BY len" : [[1],[2]],
            "MATCH p = (d {v:'d'})-[*0]-() RETURN length(p)" : [[0]],
        }

        # validate query results
        for query, expected_result in query_to_expected_result.items():
            actual_result = self.graph.query(query)
            self.env.assertEquals(actual_result.result_set, expected_result)

    def test12_close_cycle(self):
        # create a graph with a cycle in it
        # a->d
        # a->b->c->a
        #
        # variable length traversal should not get stuck in a cycle
        # in addition, the traversal mustn't continue traversing once a cycle
        # is detected, for the test graph that means that the path: a->b->c->a->d/b
        # can't be matched

        # clear previous data
        self.graph.delete()

        # create graph
        query = """CREATE (a:A {v:'a'}), (b:B {v:'b'}), (c:C {v:'c'}), (d:D {v:'d'}),
                          (a)-[:R]->(b)-[:R]->(c)-[:R]->(a),
                          (a)-[:R]->(d)"""

        result = self.graph.query(query)
        self.env.assertEquals(result.nodes_created, 4)
        self.env.assertEquals(result.relationships_created, 4)

        # perform variable length traverse from 'a'
        query = """MATCH (a:A)-[*2..]->(z)
                   RETURN z.v
                   ORDER BY z.v"""

        result = self.graph.query(query).result_set
        self.env.assertEquals(len(result), 2)
        self.env.assertEquals(result[0][0], 'a')
        self.env.assertEquals(result[1][0], 'c')

    def test13_fanout(self):
        # create a tree structure graph with a fanout of 3
        # root->a1
        # root->a2
        # root->a3
        # a1->b1
        # a1->b2
        # a1->b3
        # ...
        # a3->d3

        self.graph.delete()

        # create a tree structure with a fanout of 3
        q = """CREATE (root {l:0, id:0})
               WITH root
               UNWIND range(0, 2) AS i
               CREATE (root)-[:R]->(a{l:1, id:i})
               WITH collect(a) as nodes
               UNWIND nodes AS n
               UNWIND range(0, 2) AS i
               CREATE (n)-[:R]->(a{l:2, id:n.id*3+i})"""

        res = self.graph.query(q)
        self.env.assertEquals(res.nodes_created, 13)

        # get all reachable nodes from root
        q = """MATCH (root {l:0})-[*0..]->(n)
               RETURN n.l, n.id
               ORDER BY n.l, n.id"""
        res = self.graph.query(q).result_set
        self.env.assertEquals(len(res), 13)

        # root
        self.env.assertEquals(res[0][0], 0)
        self.env.assertEquals(res[0][1], 0)

        # children of root
        for i in range(3):
            l = res[i+1][0]
            identity = res[i+1][1]
            self.env.assertEquals(l, 1)
            self.env.assertEquals(identity, i)

        # grandchildren of root
        for i in range(9):
            l = res[i+4][0]
            identity = res[i+4][1]
            self.env.assertEquals(l, 2)
            self.env.assertEquals(identity, i)

    def test14_no_hops(self):
        self.graph.delete()

        # create graph
        # (a)->(b)->(c)
        q = "CREATE (:A {v:1})-[:R {v:2}]->(:B {v:3})-[:R {v:4}]->(:C {v:5})"
        res = self.graph.query(q)
        self.env.assertEquals(res.nodes_created, 3)
        self.env.assertEquals(res.relationships_created, 2)

        # perform 0 hop traversal
        q = "MATCH (a)-[*0]->(b) RETURN a.v, b.v ORDER BY a.v, b.v"
        res = self.graph.query(q)
        self.env.assertEquals(res.result_set, [[1, 1], [3, 3], [5, 5]])

        # specify node label
        q = "MATCH (a:A)-[*0]->(b) RETURN a.v, b.v"
        res = self.graph.query(q)
        self.env.assertEquals(res.result_set, [[1, 1]])

        # reposition label
        q = "MATCH (a)-[*0]->(b:A) RETURN a.v, b.v"
        res = self.graph.query(q)
        self.env.assertEquals(res.result_set, [[1, 1]])

        # conflicting labels
        q = "MATCH (a:A)-[*0]->(b:B) RETURN a.v, b.v"
        res = self.graph.query(q)
        self.env.assertEquals(res.result_set, [])

        # return zero length edge
        # 'e' is a path object of length 0
        q = "MATCH (a)-[e:R*0]->(b) RETURN e"
        res = self.graph.query(q)
        for row in res.result_set:
            path = row[0]
            self.env.assertEquals(path.edge_count(), 0)

        # filter none existing edge
        # all edges (none) setisfy the filter
        q = "MATCH (a)-[e:R*0]->(b) WHERE e.v = 1 RETURN a.v, b.v ORDER BY a.v, b.v"
        res = self.graph.query(q)
        self.env.assertEquals(res.result_set, [[1, 1], [3, 3], [5, 5]])

        # zero length traversal with follow up
        q = "MATCH (a)-[e0]->(b)-[e1*0]->(c) RETURN a.v, b.v, c.v"
        res  = self.graph.query(q)
        self.env.assertEquals(res.result_set, [[1, 3, 3], [3, 5, 5]])

        q = "MATCH (a:A)-[e0]->(b)-[e1*0]->(c) RETURN a.v, b.v, c.v"
        res  = self.graph.query(q)
        self.env.assertEquals(res.result_set, [[1, 3, 3]])

        q = "MATCH (a)-[e0]->(b:B)-[e1*0]->(c) RETURN a.v, b.v, c.v"
        res  = self.graph.query(q)
        self.env.assertEquals(res.result_set, [[1, 3, 3]])

        q = "MATCH (a:A)-[e0]->(b:B)-[e1*0]->(c) RETURN a.v, b.v, c.v"
        res  = self.graph.query(q)
        self.env.assertEquals(res.result_set, [[1, 3, 3]])

        q = "MATCH (a)-[e0]->(b)-[e1*0]->(c) RETURN a.v, b.v, c.v"
        res  = self.graph.query(q)
        self.env.assertEquals(res.result_set, [[1, 3, 3], [3, 5, 5]])

        # same queries as before, swap zero length edge
        q = "MATCH (a)-[e0*0]->(b)-[e1]->(c) RETURN a.v, b.v, c.v"
        res  = self.graph.query(q)
        self.env.assertEquals(res.result_set, [[1, 1, 3], [3, 3, 5]])

        q = "MATCH (a:A)-[e0*0]->(b)-[e1]->(c) RETURN a.v, b.v, c.v"
        res  = self.graph.query(q)
        self.env.assertEquals(res.result_set, [[1, 1, 3]])

        q = "MATCH (a)-[e0*0]->(b:A)-[e1]->(c) RETURN a.v, b.v, c.v"
        res  = self.graph.query(q)
        self.env.assertEquals(res.result_set, [[1, 1, 3]])

        q = "MATCH (a:A)-[e0*0]->(b:A)-[e1]->(c) RETURN a.v, b.v, c.v"
        res  = self.graph.query(q)
        self.env.assertEquals(res.result_set, [[1, 1, 3]])

        q = "MATCH (a)-[e0*0]->(b)-[e1]->(c) RETURN a.v, b.v, c.v"
        res  = self.graph.query(q)
        self.env.assertEquals(res.result_set, [[1, 1, 3], [3, 3, 5]])

        # multi zero length edges
        q = "MATCH (a)-[e0*0]->(b)-[e1*0]->(c) RETURN a.v, b.v, c.v"
        res  = self.graph.query(q)
        self.env.assertEquals(res.result_set, [[1, 1, 1], [3, 3, 3], [5, 5, 5]])

    def test15_var_len_with_filter_path_filter(self):
        self.graph.delete()

        # create graph
        q = """CREATE (a0:A {v:1}),
                      (b0:B {v:3}),
                      (c0:C {v:5}),
                      (a1:A {v:6}),
                      (b1:B {v:8}),
                      (c1:C {v:10}),

                      (a0)-[a0b0:R {v:2}]->(b0),
                      (b0)-[b0c0:R {v:4}]->(c0),
                      (a1)-[a1b1:R {v:9}]->(b1),
                      (b1)-[b1c1:R {v:7}]->(c1)

               RETURN a0, b0, c0, a1, b1, c1, a0b0, b0c0, a1b1, b1c1"""

        res = self.graph.query(q).result_set

        a0   = res[0][0]
        b0   = res[0][1]
        c0   = res[0][2]
        a1   = res[0][3]
        b1   = res[0][4]
        c1   = res[0][5]
        a0b0 = res[0][6]
        b0c0 = res[0][7]
        a1b1 = res[0][8]
        b1c1 = res[0][9]

        q = """MATCH p=(a:A)-[e*]->(b)
               WHERE length(intermediate_path(e)) = 1 OR
                     relationships(intermediate_path(e))[-2].v <
                     relationships(intermediate_path(e))[-1].v
               RETURN p"""

        plan = self.graph.explain(q)

        expected_ops = ["Results",
                        "Project",
                        "Conditional Variable Length Traverse",
                        "Node By Label Scan"]
        expected_ops.reverse()

        op = plan.structured_plan
        while len(op.children) > 0:
            self.env.assertEquals(op.name, expected_ops.pop())
            op = op.children[0]

        # last op
        self.env.assertEquals(op.name, expected_ops.pop())

        res = self.graph.query(q)
        self.env.assertEquals(len(res.result_set), 3)

        self.env.assertEquals(res.result_set[0][0], Path([a0, b0], [a0b0]))
        self.env.assertEquals(res.result_set[1][0], Path([a0, b0, c0], [a0b0, b0c0]))
        self.env.assertEquals(res.result_set[2][0], Path([a1, b1], [a1b1]))

    def test16_multi_var_len_filters(self):
        self.graph.delete()

        # create graph
        q = """
        CREATE (a:A {v:1}),
               (b:B {v:2}),
               (c:C {v:3}),
               (d:D {v:4}),
               (e:E {v:3}),
               (f:F {v:2}),
               (g:G {v:2})

        CREATE (a)-[ab:R]->(b),
               (b)-[bc:R]->(c),
               (c)-[cd:R]->(d),
               (d)-[de:R]->(e),
               (e)-[ef:R]->(f),
               (f)-[fg:R]->(g)

        RETURN a, b, c, d, e, f, g, ab, bc, cd, de, ef, fg
        """

        res = self.graph.query(q).result_set
        a = res[0][0]
        b = res[0][1]
        c = res[0][2]
        d = res[0][3]
        e = res[0][4]
        f = res[0][5]
        g = res[0][6]

        ab = res[0][7]
        bc = res[0][8]
        cd = res[0][9]
        de = res[0][10]
        ef = res[0][11]
        fg = res[0][12]

        q = """MATCH p0 = (a:A)-[e*]->(d:D)
               WHERE length(intermediate_path(e)) = 1 OR
                     nodes(intermediate_path(e))[-2].v < nodes(intermediate_path(e))[-1].v

               MATCH p1 = (d)-[f*]->(z)
               WHERE length(intermediate_path(f)) = 1 OR
                     nodes(intermediate_path(f))[-1].v < nodes(intermediate_path(f))[-2].v

               RETURN p0, p1"""

        plan = self.graph.explain(q)
        expected_ops = ["Results",
                        "Project",
                        "Conditional Variable Length Traverse",
                        "Conditional Traverse",
                        "Conditional Variable Length Traverse",
                        "Node By Label Scan"]
        expected_ops.reverse()

        op = plan.structured_plan
        while len(op.children) > 0:
            self.env.assertEquals(op.name, expected_ops.pop())
            op = op.children[0]

        # last op
        self.env.assertEquals(op.name, expected_ops.pop())

        self.env.assertEquals(len(expected_ops), 0)

        res = self.graph.query(q).result_set
        self.env.assertEquals(len(res), 2)

        expected_p0 = Path([a, b, c, d], [ab, bc, cd])
        expected_p1 = Path([d, e], [de])

        self.env.assertEquals(res[0][0], expected_p0)
        self.env.assertEquals(res[0][1], expected_p1)

        expected_p1 = Path([d, e, f], [de, ef])
        self.env.assertEquals(res[1][0], expected_p0)
        self.env.assertEquals(res[1][1], expected_p1)

        #-----------------------------------------------------------------------
        # check variable length incoming edges
        #-----------------------------------------------------------------------

        # force backward traversal via incoming edges
        q = """MATCH (d:D)
               WITH d
               MATCH p = (d)<-[e*]-(a)
               WHERE nodes(intermediate_path(e))[-1].v > 1

               RETURN p
               ORDER BY length(p)"""

        res = self.graph.query(q).result_set
        self.env.assertEquals(len(res), 2)

        expected_p = Path([d, c], [cd])

        self.env.assertEquals(res[0][0], expected_p)

        expected_p = Path([d, c, b], [cd, bc])
        self.env.assertEquals(res[1][0], expected_p)

