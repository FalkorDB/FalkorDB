from common import *

GRAPH_ID = "max_flow"

class testMaxFlow(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        self.graph.delete()

    def max_flow(self, sourceNodes, targetNodes, capacityProperty=None,
                 nodeLabels=[], relationshipTypes=None):
        config = {
            'srcs': sourceNodes,
            'snks': targetNodes,
            'cap': capacityProperty,
            'rels': relationshipTypes,
            'lbls': nodeLabels
        }

        return self.graph.query("""
            MATCH (s) WHERE id(s) IN $srcs
            MATCH (t) WHERE id(t) IN $snks
            WITH collect(DISTINCT s) AS sourceNodes, collect(DISTINCT t) AS targetNodes
            CALL algo.maxFlow({
                sourceNodes:       sourceNodes,
                targetNodes:       targetNodes,
                capacityProperty:  $cap,
                nodeLabels:        $lbls,
                relationshipTypes: $rels
            })
        """, config)

    # ------------------------------------------------------------------ #
    #  invalid invocations                                               #
    # ------------------------------------------------------------------ #

#    def test_invalid_invocation(self):
#        """Procedure must reject every malformed configuration."""
#        # need at least one node in the graph so that id-based queries compile
#        self.graph.query("CREATE (:N), (:N)")
#
#        invalid_queries = [
#            # missing argument entirely
#            """CALL algo.maxFlow()""",
#
#            # wrong argument type (string instead of map)
#            """CALL algo.maxFlow('invalid')""",
#
#            # unknown key
#            """CALL algo.maxFlow({unknownKey: 1})""",
#
#            # sourceNodes not an array
#            """CALL algo.maxFlow({sourceNodes: 1, targetNodes: [],
#                                  relationshipTypes: ['R']})""",
#
#            # targetNodes not an array
#            """CALL algo.maxFlow({sourceNodes: [], targetNodes: 'bad',
#                                  relationshipTypes: ['R']})""",
#
#            # sourceNodes array contains non-nodes
#            """CALL algo.maxFlow({sourceNodes: [1, 2], targetNodes: [],
#                                  relationshipTypes: ['R']})""",
#
#            # targetNodes array contains non-nodes
#            """CALL algo.maxFlow({sourceNodes: [], targetNodes: ['x'],
#                                  relationshipTypes: ['R']})""",
#
#            # missing sourceNodes
#            """MATCH (t:N) CALL algo.maxFlow({
#                targetNodes: [t], relationshipTypes: ['R']
#               }) YIELD flow RETURN flow""",
#
#            # missing targetNodes
#            """MATCH (s:N) CALL algo.maxFlow({
#                sourceNodes: [s], relationshipTypes: ['R']
#               }) YIELD flow RETURN flow""",
#
#            # missing relationshipTypes
#            """MATCH (s:N),(t:N) WHERE id(s)<>id(t)
#               CALL algo.maxFlow({sourceNodes:[s], targetNodes:[t]})
#               YIELD flow RETURN flow""",
#
#            # relationshipTypes must contain exactly one type
#            """MATCH (s:N),(t:N) WHERE id(s)<>id(t)
#               CALL algo.maxFlow({sourceNodes:[s], targetNodes:[t],
#                   relationshipTypes:['R','S']})
#               YIELD flow RETURN flow""",
#
#            # nodeLabels must be an array of strings
#            """MATCH (s:N),(t:N) WHERE id(s)<>id(t)
#               CALL algo.maxFlow({sourceNodes:[s], targetNodes:[t],
#                   nodeLabels: 'Person', relationshipTypes:['R']})
#               YIELD flow RETURN flow""",
#
#            # nodeLabels array must contain strings
#            """MATCH (s:N),(t:N) WHERE id(s)<>id(t)
#               CALL algo.maxFlow({sourceNodes:[s], targetNodes:[t],
#                   nodeLabels:[1,2], relationshipTypes:['R']})
#               YIELD flow RETURN flow""",
#
#            # capacityProperty must be a string
#            """MATCH (s:N),(t:N) WHERE id(s)<>id(t)
#               CALL algo.maxFlow({sourceNodes:[s], targetNodes:[t],
#                   capacityProperty:42, relationshipTypes:['R']})
#               YIELD flow RETURN flow""",
#
#            # non-existent yield field
#            """MATCH (s:N),(t:N) WHERE id(s)<>id(t)
#               CALL algo.maxFlow({sourceNodes:[s], targetNodes:[t],
#                   relationshipTypes:['R']})
#               YIELD flow, badField RETURN flow""",
#
#            # extra / misspelled key alongside valid ones
#            """MATCH (s:N),(t:N) WHERE id(s)<>id(t)
#               CALL algo.maxFlow({sourceNodes:[s], targetNodes:[t],
#                   relationshipTypes:['R'], oops:'yes'})
#               YIELD flow RETURN flow""",
#        ]
#
#        for q in invalid_queries:
#            try:
#                self.graph.query(q)
#                self.env.assertFalse(True)
#            except:
#                pass

    # ------------------------------------------------------------------ #
    #  empty / trivial graphs                                            #
    # ------------------------------------------------------------------ #

#    def test_max_flow_empty_graph(self):
#        """Flow on an empty graph should throw an exceptiont."""
#        self.graph.query("CREATE (a)-[e:PIPE]->(b) DELETE a,e,b")
#
#        try:
#            self.graph.query("""
#                WITH [] AS sourceNodes, [] AS targetNodes
#                CALL algo.maxFlow({
#                    sourceNodes: sourceNodes,
#                    targetNodes: targetNodes,
#                    relationshipTypes: ['PIPE']
#                })
#                YIELD flow
#            """)
#            self.env.assertFalse(True)
#        except:
#            pass

    def test_max_flow_no_path_between_source_and_sink(self):
        """Two disconnected nodes yield zero flow."""
        self.graph.query("""
            CREATE (:Node {name:'A'})-[e:PIPE {cap:1}]->(:Node {name:'B'})
            DELETE e
        """)
        result = self.graph.query("""
            MATCH (s:Node {name:'A'}), (t:Node {name:'B'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [t],
                relationshipTypes: ['PIPE'],
                capacityProperty: 'cap'
            })
            YIELD flow
        """)
        # either empty result set or flow == 0
        if len(result.result_set) > 0:
            self.env.assertEqual(result.result_set[0][0], [])

    # ------------------------------------------------------------------ #
    #  single source, single sink – correctness                          #
    # ------------------------------------------------------------------ #

    def test_max_flow_simple_path(self):
        """A -5-> B -5-> C  →  max flow = 5."""
        self.graph.query("""
            CREATE (a:Node {name:'A'})-[:PIPE {cap:5}]->
                   (b:Node {name:'B'})-[:PIPE {cap:5}]->
                   (c:Node {name:'C'})
        """)
        result = self.graph.query("""
            MATCH (s:Node {name:'A'}), (t:Node {name:'C'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [t],
                capacityProperty:  'cap',
                relationshipTypes: ['PIPE']
            })
            YIELD flow
        """)
        self.env.assertEqual(len(result.result_set), 1)
        self.env.assertEqual(result.result_set[0][0], [5,5])

    def test_max_flow_bottleneck(self):
        """
        A -10-> B -3-> C -10-> D
        The bottleneck edge B→C limits flow to 3.
        """
        self.graph.query("""
            CREATE (a:Node {name:'A'})-[:PIPE {cap:10}]->
                   (b:Node {name:'B'})-[:PIPE {cap:3}]->
                   (c:Node {name:'C'})-[:PIPE {cap:10}]->
                   (d:Node {name:'D'})
        """)
        result = self.graph.query("""
            MATCH (s:Node {name:'A'}), (t:Node {name:'D'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [t],
                capacityProperty:  'cap',
                relationshipTypes: ['PIPE']
            })
            YIELD flow
        """)
        self.env.assertEqual(result.result_set[0][0], [3, 3, 3])

    def test_max_flow_parallel_paths(self):
        """
        Two parallel paths A->B->D and A->C->D, each with capacity 4.
        Max flow = 8.

          A -4-> B -4-> D
          |             ^
          +--4-> C -4--+
        """
        self.graph.query("""
            CREATE (a:Node {name:'A'}),
                   (b:Node {name:'B'}),
                   (c:Node {name:'C'}),
                   (d:Node {name:'D'}),
                   (a)-[:PIPE {cap:4}]->(b),
                   (a)-[:PIPE {cap:4}]->(c),
                   (b)-[:PIPE {cap:4}]->(d),
                   (c)-[:PIPE {cap:4}]->(d)
        """)
        result = self.graph.query("""
            MATCH (s:Node {name:'A'}), (t:Node {name:'D'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [t],
                capacityProperty:  'cap',
                relationshipTypes: ['PIPE']
            })
            YIELD flow
        """)
        self.env.assertEqual(result.result_set[0][0], [4, 4, 4, 4])
#
#    def test_max_flow_default_capacity(self):
#        """Without capacityProperty every edge defaults to capacity 1."""
#        self.graph.query("""
#            CREATE (a:Node {name:'A'}),
#                   (b:Node {name:'B'}),
#                   (c:Node {name:'C'}),
#                   (d:Node {name:'D'}),
#                   (a)-[:PIPE {cap:'a'}]->(b),
#                   (a)-[:PIPE]->(c),
#                   (b)-[:PIPE]->(d),
#                   (c)-[:PIPE {cap: []}]->(d)
#        """)
#        result = self.graph.query("""
#            MATCH (s:Node {name:'A'}), (t:Node {name:'D'})
#            CALL algo.maxFlow({
#                sourceNodes:       [s],
#                targetNodes:       [t],
#                relationshipTypes: ['PIPE']
#            })
#            YIELD flow
#        """)
#        # two unit-capacity parallel paths -> flow = 2
#        self.env.assertEqual(result.result_set[0][0], 2)
#
#    # ------------------------------------------------------------------ #
#    #  yield – subsets of output columns                                 #
#    # ------------------------------------------------------------------ #
#
#    def test_yield_flow_only(self):
#        """Requesting only YIELD flow must succeed and return a numeric value."""
#        self.graph.query("""
#            CREATE (a:Node {name:'A'})-[:PIPE {cap:7}]->(b:Node {name:'B'})
#        """)
#        result = self.graph.query("""
#            MATCH (s:Node {name:'A'}), (t:Node {name:'B'})
#            CALL algo.maxFlow({
#                sourceNodes:       [s],
#                targetNodes:       [t],
#                capacityProperty:  'cap',
#                relationshipTypes: ['PIPE']
#            })
#            YIELD flow
#        """)
#        self.env.assertEqual(len(result.result_set), 1)
#        self.env.assertEqual(result.result_set[0][0], 7)
#
#    def test_yield_nodes_only(self):
#        """Requesting only YIELD nodes must return a non-empty array."""
#        nodes = self.graph.query("""
#            CREATE (a:Node {name:'A'})-[:PIPE {cap:7}]->(b:Node {name:'B'})
#            RETURN a, b
#        """).result_set[0]
#
#        result = self.graph.query("""
#            MATCH (s:Node {name:'A'}), (t:Node {name:'B'})
#            CALL algo.maxFlow({
#                sourceNodes:       [s],
#                targetNodes:       [t],
#                capacityProperty:  'cap',
#                relationshipTypes: ['PIPE']
#            })
#            YIELD nodes
#        """)
#        self.env.assertEqual(len(result.result_set), 1)
#        yield_nodes = result.result_set[0][0]
#        self.env.assertEqual(nodes, yield_nodes)
#
#    def test_yield_edges_only(self):
#        """Requesting only YIELD edges must return a non-empty array."""
#        edges = self.graph.query("""
#            CREATE (a:Node {name:'A'})-[e:PIPE {cap:7}]->(b:Node {name:'B'})
#            RETURN e
#        """).result_set[0]
#        result = self.graph.query("""
#            MATCH (s:Node {name:'A'}), (t:Node {name:'B'})
#            CALL algo.maxFlow({
#                sourceNodes:       [s],
#                targetNodes:       [t],
#                capacityProperty:  'cap',
#                relationshipTypes: ['PIPE']
#            })
#            YIELD edges
#        """)
#        self.env.assertEqual(len(result.result_set), 1)
#        yield_edges = result.result_set[0][0]
#        self.env.assertEqual(edges, yield_edges)
#
#    def test_yield_nodes_edges_flow(self):
#        """
#        All three yield columns must be consistent with each other:
#        - nodes includes exactly the endpoints that appear in edges
#        - len(edges) == len(flows)
#        - each flow value is > 0
#        """
#        self.graph.query("""
#            CREATE (a:Node {name:'A'}),
#                   (b:Node {name:'B'}),
#                   (c:Node {name:'C'}),
#                   (a)-[:PIPE {cap:5}]->(b),
#                   (b)-[:PIPE {cap:5}]->(c)
#        """)
#        result = self.graph.query("""
#            MATCH (s:Node {name:'A'}), (t:Node {name:'C'})
#            CALL algo.maxFlow({
#                sourceNodes:       [s],
#                targetNodes:       [t],
#                capacityProperty:  'cap',
#                relationshipTypes: ['PIPE']
#            })
#            YIELD nodes, edges, flow
#        """)
#        self.env.assertEqual(len(result.result_set), 1)
#        nodes, edges, flow = result.result_set[0]
#
#        self.env.assertGreater(len(nodes), 0)
#        self.env.assertGreater(len(edges), 0)
#        self.env.assertEqual(flow, 5)
#
#        node_ids = [n.id for n in nodes]
#        for idx, e in enumerate(edges):
#            self.env.assertTrue(e.src_node.id in node_ids)
#            self.env.assertTrue(e.properties['cap'] >= flow[idx])
#
#    # ------------------------------------------------------------------ #
#    #  multi-source, single sink                                           #
#    # ------------------------------------------------------------------ #
#
#    def test_max_flow_multi_source_single_sink(self):
#        """
#        Two independent sources feeding a common sink.
#
#          A -5-> C
#          B -3-> C
#
#        Super-source S' connects to both A and B with ∞ capacity.
#        Max flow = 5 + 3 = 8.
#        """
#        self.graph.query("""
#            CREATE (a:Node {name:'A'}),
#                   (b:Node {name:'B'}),
#                   (c:Node {name:'C'}),
#                   (a)-[:PIPE {cap:5}]->(c),
#                   (b)-[:PIPE {cap:3}]->(c)
#        """)
#        result = self.graph.query("""
#            MATCH (a:Node {name:'A'}),
#                  (b:Node {name:'B'}),
#                  (t:Node {name:'C'})
#            CALL algo.maxFlow({
#                sourceNodes:       [a, b],
#                targetNodes:       [t],
#                capacityProperty:  'cap',
#                relationshipTypes: ['PIPE']
#            })
#            YIELD nodes, flow
#        """)
#        self.env.assertEqual(result.result_set[0][1], 8)
#        self.env.assertEqual(len(result.result_set[0][0]), 3)
#
#    def test_max_flow_multi_source_bottleneck(self):
#        """
#        Sources A and B both feed into C which is a bottleneck (cap 4) to D.
#
#          A -6-> C -4-> D
#          B -6-> C
#
#        Total in to C = 12, but outgoing cap = 4 → max flow = 4.
#        """
#        self.graph.query("""
#            CREATE (a:Node {name:'A'}),
#                   (b:Node {name:'B'}),
#                   (c:Node {name:'C'}),
#                   (d:Node {name:'D'}),
#                   (a)-[:PIPE {cap:6}]->(c),
#                   (b)-[:PIPE {cap:6}]->(c),
#                   (c)-[:PIPE {cap:4}]->(d)
#        """)
#        result = self.graph.query("""
#            MATCH (a:Node {name:'A'}),
#                  (b:Node {name:'B'}),
#                  (t:Node {name:'D'})
#            CALL algo.maxFlow({
#                sourceNodes:       [a, b],
#                targetNodes:       [t],
#                capacityProperty:  'cap',
#                relationshipTypes: ['PIPE']
#            })
#            YIELD flow
#        """)
#        self.env.assertEqual(result.result_set[0][0], 4)
#
#    # ------------------------------------------------------------------ #
#    #  single source, multi-sink                                         #
#    # ------------------------------------------------------------------ #
#
#    def test_max_flow_single_source_multi_sink(self):
#        """
#        One source, two independent sinks.
#
#          A -5-> B
#          A -3-> C
#
#        Super-sink T' receives from B and C with ∞ capacity.
#        Max flow = 5 + 3 = 8.
#        """
#        self.graph.query("""
#            CREATE (a:Node {name:'A'}),
#                   (b:Node {name:'B'}),
#                   (c:Node {name:'C'}),
#                   (a)-[:PIPE {cap:5}]->(b),
#                   (a)-[:PIPE {cap:3}]->(c)
#        """)
#        result = self.graph.query("""
#            MATCH (s:Node  {name:'A'}),
#                  (b:Node  {name:'B'}),
#                  (c:Node  {name:'C'})
#            CALL algo.maxFlow({
#                sourceNodes:       [s],
#                targetNodes:       [b, c],
#                capacityProperty:  'cap',
#                relationshipTypes: ['PIPE']
#            })
#            YIELD nodes, flow
#        """)
#        self.env.assertEqual(len(result.result_set[0][0]), 3)
#        self.env.assertEqual(result.result_set[0][1], 8)
#
#    def test_max_flow_single_source_multi_sink_bottleneck(self):
#        """
#        Source bottleneck constrains multi-sink total.
#
#          A -2-> B -6-> D
#          A -4-> C -6-> E
#
#        Source cap out = 6, flow = 8.
#        """
#        self.graph.query("""
#            CREATE (a:Node {name:'A'}),
#                   (b:Node {name:'B'}),
#                   (c:Node {name:'C'}),
#                   (d:Node {name:'D'}),
#                   (e:Node {name:'E'}),
#                   (a)-[:PIPE {cap:2}]->(b),
#                   (a)-[:PIPE {cap:4}]->(c),
#                   (b)-[:PIPE {cap:6}]->(d),
#                   (c)-[:PIPE {cap:6}]->(e)
#        """)
#        result = self.graph.query("""
#            MATCH (s:Node {name:'A'}),
#                  (d:Node {name:'D'}),
#                  (e:Node {name:'E'})
#            CALL algo.maxFlow({
#                sourceNodes:       [s],
#                targetNodes:       [d, e],
#                capacityProperty:  'cap',
#                relationshipTypes: ['PIPE']
#            })
#            YIELD flow
#        """)
#        self.env.assertEqual(result.result_set[0][0], 6)
#
#    # ------------------------------------------------------------------ #
#    #  multi-source, multi-sink                                          #
#    # ------------------------------------------------------------------ #
#
#    def test_max_flow_multi_source_multi_sink(self):
#        """
#        Two sources and two sinks in a diamond.
#
#          A -5-> C -5-> D
#          B -5-> C -5-> E
#
#        Sources: A, B  /  Sinks: D, E
#        Max flow = 10.
#        """
#        self.graph.query("""
#            CREATE (a:Node {name:'A'}),
#                   (b:Node {name:'B'}),
#                   (c:Node {name:'C'}),
#                   (d:Node {name:'D'}),
#                   (e:Node {name:'E'}),
#                   (a)-[:PIPE {cap:5}]->(c),
#                   (b)-[:PIPE {cap:5}]->(c),
#                   (c)-[:PIPE {cap:5}]->(d),
#                   (c)-[:PIPE {cap:5}]->(e)
#        """)
#        result = self.graph.query("""
#            MATCH (a:Node {name:'A'}),
#                  (b:Node {name:'B'}),
#                  (d:Node {name:'D'}),
#                  (e:Node {name:'E'})
#            CALL algo.maxFlow({
#                sourceNodes:       [a, b],
#                targetNodes:       [d, e],
#                capacityProperty:  'cap',
#                relationshipTypes: ['PIPE']
#            })
#            YIELD flow
#        """)
#        self.env.assertEqual(result.result_set[0][0], 10)
#
#    def test_max_flow_multi_source_multi_sink_asymmetric(self):
#        """
#        Asymmetric capacities ensure the super-source / super-sink trick
#        does not over-count or under-count.
#
#          A -3-> C -2-> E
#          A -3-> D -4-> F
#          B -6-> C
#          B -6-> D
#
#        Sources: A, B  /  Sinks: E, F
#        Bottlenecks: C->E (2), A->C (3), A->D (3), B->D (6)
#        Max flow = 2 + 4 = 6
#        """
#        self.graph.query("""
#            CREATE (a:Node {name:'A'}),
#                   (b:Node {name:'B'}),
#                   (c:Node {name:'C'}),
#                   (d:Node {name:'D'}),
#                   (e:Node {name:'E'}),
#                   (f:Node {name:'F'}),
#                   (a)-[:PIPE {cap:3}]->(c),
#                   (a)-[:PIPE {cap:3}]->(d),
#                   (b)-[:PIPE {cap:6}]->(c),
#                   (b)-[:PIPE {cap:6}]->(d),
#                   (c)-[:PIPE {cap:2}]->(e),
#                   (d)-[:PIPE {cap:4}]->(f)
#        """)
#        result = self.graph.query("""
#            MATCH (a:Node {name:'A'}),
#                  (b:Node {name:'B'}),
#                  (e:Node {name:'E'}),
#                  (f:Node {name:'F'})
#            CALL algo.maxFlow({
#                sourceNodes:       [a, b],
#                targetNodes:       [e, f],
#                capacityProperty:  'cap',
#                relationshipTypes: ['PIPE']
#            })
#            YIELD flow
#        """)
#        self.env.assertEqual(result.result_set[0][0], 6)
#
#    # ------------------------------------------------------------------ #
#    #  multi-source / multi-sink output sanity                           #
#    # ------------------------------------------------------------------ #
#
#    def test_multi_source_sink_no_super_nodes_in_output(self):
#        """
#        The synthetic super-source and super-sink must NOT appear in the
#        returned nodes or edges arrays – they are implementation details.
#        """
#        self.graph.query("""
#            CREATE (a:Node {name:'A'}),
#                   (b:Node {name:'B'}),
#                   (c:Node {name:'C'}),
#                   (d:Node {name:'D'}),
#                   (a)-[:PIPE {cap:5}]->(c),
#                   (b)-[:PIPE {cap:5}]->(c),
#                   (c)-[:PIPE {cap:5}]->(d)
#        """)
#        # fetch the max node id before the call so we can detect phantoms
#        max_id_result = self.graph.query("MATCH (n) RETURN max(id(n))")
#        max_real_id   = max_id_result.result_set[0][0]
#
#        result = self.graph.query("""
#            MATCH (a:Node {name:'A'}),
#                  (b:Node {name:'B'}),
#                  (t:Node {name:'D'})
#            CALL algo.maxFlow({
#                sourceNodes:       [a, b],
#                targetNodes:       [t],
#                capacityProperty:  'cap',
#                relationshipTypes: ['PIPE']
#            })
#            YIELD nodes, edges
#        """)
#
#        self.env.assertEqual(len(result.result_set), 1)
#        nodes, edges = result.result_set[0]
#
#        for node in nodes:
#            self.env.assertLessEqual(node.id, max_real_id)
#
#        for edge in edges:
#            self.env.assertLessEqual(edge.src_node, max_real_id)
#            self.env.assertLessEqual(edge.dest_node, max_real_id)
#
#        node_count = self.graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
#        self.env.assertEqual(node_count, 3)
#
#    # ------------------------------------------------------------------ #
#    #  nodeLabels filter                                                 #
#    # ------------------------------------------------------------------ #
#
#    def test_max_flow_node_labels_filter(self):
#        """
#        Graph has two node types; filtering to only 'Pipe' nodes should
#        exclude 'Valve' nodes and therefore reduce the max flow.
#
#          A:Pipe -5-> V:Valve -5-> B:Pipe
#          A:Pipe -3-----------> B:Pipe    (direct edge, same label)
#
#        With nodeLabels=['Pipe']: only the direct A->B path (cap 3) is visible
#        -> flow = 3.
#        Without filter both paths exist -> flow = 8 (if we remove the valve
#        from the path the shortest route carries 3; including valve path
#        gives 5+3=8 but through nodeLabels only Pipe nodes count so valve
#        path disappears -> flow = 3).
#        """
#        self.graph.query("""
#            CREATE (a:Pipe   {name:'A'}),
#                   (v:Valve  {name:'V'}),
#                   (b:Pipe   {name:'B'}),
#                   (a)-[:FLOW {cap:5}]->(v),
#                   (v)-[:FLOW {cap:5}]->(b),
#                   (a)-[:FLOW {cap:3}]->(b)
#        """)
#
#        # without label filter
#        result_all = self.graph.query("""
#            MATCH (s:Pipe {name:'A'}), (t:Pipe {name:'B'})
#            CALL algo.maxFlow({
#                sourceNodes:       [s],
#                targetNodes:       [t],
#                capacityProperty:  'cap',
#                relationshipTypes: ['FLOW']
#            })
#            YIELD flow
#        """)
#
#        # with label filter (Valve node excluded)
#        result_filtered = self.graph.query("""
#            MATCH (s:Pipe {name:'A'}), (t:Pipe {name:'B'})
#            CALL algo.maxFlow({
#                sourceNodes:       [s],
#                targetNodes:       [t],
#                capacityProperty:  'cap',
#                nodeLabels:        ['Pipe'],
#                relationshipTypes: ['FLOW']
#            })
#            YIELD flow
#        """)
#
#        flow_all      = result_all.result_set[0][0]
#        flow_filtered = result_filtered.result_set[0][0]
#
#        # Filtering out the valve node must reduce (or equal) flow
#        self.env.assertGreaterEqual(flow_all, flow_filtered)
#        self.env.assertEqual(flow_filtered, 3)
#
#    # ------------------------------------------------------------------ #
#    #  relationshipTypes filter                                          #
#    # ------------------------------------------------------------------ #
#
#    def test_max_flow_relationship_type_filter(self):
#        """
#        Same node topology, different relationship types.
#        Using only 'PIPE' edges ignores 'CABLE' edges.
#
#          A -5[PIPE]-> B -5[PIPE]-> D
#          A -5[CABLE]-> C -5[CABLE]-> D
#
#        YIELD flow with PIPE only -> 5.
#        """
#        self.graph.query("""
#            CREATE (a:Node {name:'A'}),
#                   (b:Node {name:'B'}),
#                   (c:Node {name:'C'}),
#                   (d:Node {name:'D'}),
#                   (a)-[:PIPE  {cap:5}]->(b),
#                   (b)-[:PIPE  {cap:5}]->(d),
#                   (a)-[:CABLE {cap:5}]->(c),
#                   (c)-[:CABLE {cap:5}]->(d)
#        """)
#        result = self.graph.query("""
#            MATCH (s:Node {name:'A'}), (t:Node {name:'D'})
#            CALL algo.maxFlow({
#                sourceNodes:       [s],
#                targetNodes:       [t],
#                capacityProperty:  'cap',
#                relationshipTypes: ['PIPE']
#            })
#            YIELD flow RETURN flow
#        """)
#        self.env.assertEqual(result.result_set[0][0], 5)
#
#    # ------------------------------------------------------------------ #
#    #  single-element arrays equal single source / sink behaviour        #
#    # ------------------------------------------------------------------ #
#
#    def test_single_element_array_equals_scalar(self):
#        """
#        [src] / [sink] (length-1 arrays) must produce the same flow as the
#        plain single-source / single-sink call.
#        """
#        self.graph.query("""
#            CREATE (a:Node {name:'A'})-[:PIPE {cap:9}]->(b:Node {name:'B'})
#        """)
#
#        result = self.graph.query("""
#            MATCH (s:Node {name:'A'}), (t:Node {name:'B'})
#            CALL algo.maxFlow({
#                sourceNodes:       [s],
#                targetNodes:       [t],
#                capacityProperty:  'cap',
#                relationshipTypes: ['PIPE']
#            })
#            YIELD flow RETURN flow
#        """)
#        self.env.assertEqual(result.result_set[0][0], 9)
#
#    # ------------------------------------------------------------------ #
#    #  source == sink (degenerate case)                                  #
#    # ------------------------------------------------------------------ #
#
#    def test_source_equals_sink(self):
#        """When source and sink are the same node, flow must be 0."""
#        self.graph.query("""
#            CREATE (a:Node {name:'A'})-[:PIPE {cap:5}]->(b:Node {name:'B'})
#        """)
#        result = self.graph.query("""
#            MATCH (s:Node {name:'A'})
#            CALL algo.maxFlow({
#                sourceNodes:       [s],
#                targetNodes:       [s],
#                capacityProperty:  'cap',
#                relationshipTypes: ['PIPE']
#            })
#            YIELD flow RETURN flow
#        """)
#        if len(result.result_set) > 0:
#            self.env.assertEqual(result.result_set[0][0], 0)

