from common import *

GRAPH_ID = "max_flow"

class testMaxFlow(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        self.graph.delete()
        # re-create the graph object to get a fresh client-side schema cache
        self.graph = self.db.select_graph(GRAPH_ID)

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

    def test_01_invalid_invocation(self):
        """Procedure must reject every malformed configuration."""
        # need at least one node in the graph so that id-based queries compile
        self.graph.query("CREATE (:N), (:N)")

        invalid_queries = [
            # missing argument entirely
            """CALL algo.maxFlow()""",

            # wrong argument type (string instead of map)
            """CALL algo.maxFlow('invalid')""",

            # unknown key
            """CALL algo.maxFlow({unknownKey: 1})""",

            # sourceNodes not an array
            """CALL algo.maxFlow({sourceNodes: 1, targetNodes: [],
                                  relationshipTypes: ['R']})""",

            # targetNodes not an array
            """CALL algo.maxFlow({sourceNodes: [], targetNodes: 'bad',
                                  relationshipTypes: ['R']})""",

            # sourceNodes array contains non-nodes
            """CALL algo.maxFlow({sourceNodes: [1, 2], targetNodes: [],
                                  relationshipTypes: ['R']})""",

            # targetNodes array contains non-nodes
            """CALL algo.maxFlow({sourceNodes: [], targetNodes: ['x'],
                                  relationshipTypes: ['R']})""",

            # missing sourceNodes
            """MATCH (t:N) CALL algo.maxFlow({
                targetNodes: [t], relationshipTypes: ['R']
               }) YIELD maxFlow""",

            # missing targetNodes
            """MATCH (s:N) CALL algo.maxFlow({
                sourceNodes: [s], relationshipTypes: ['R']
               }) YIELD maxFlow""",

            # missing relationshipTypes
            """MATCH (s:N),(t:N) WHERE id(s)<>id(t)
               CALL algo.maxFlow({sourceNodes:[s], targetNodes:[t]})
               YIELD maxFlow""",

            # relationshipTypes must contain exactly one type
            """MATCH (s:N),(t:N) WHERE id(s)<>id(t)
               CALL algo.maxFlow({sourceNodes:[s], targetNodes:[t],
                   relationshipTypes:['R','S']})
               YIELD maxFlow""",

            # nodeLabels must be an array of strings
            """MATCH (s:N),(t:N) WHERE id(s)<>id(t)
               CALL algo.maxFlow({sourceNodes:[s], targetNodes:[t],
                   nodeLabels: 'Person', relationshipTypes:['R']})
               YIELD maxFlow""",

            # nodeLabels array must contain strings
            """MATCH (s:N),(t:N) WHERE id(s)<>id(t)
               CALL algo.maxFlow({sourceNodes:[s], targetNodes:[t],
                   nodeLabels:[1,2], relationshipTypes:['R']})
               YIELD maxFlow""",

            # capacityProperty must be a string
            """MATCH (s:N),(t:N) WHERE id(s)<>id(t)
               CALL algo.maxFlow({sourceNodes:[s], targetNodes:[t],
                   capacityProperty:42, relationshipTypes:['R']})
               YIELD maxFlow""",

            # non-existent yield field
            """MATCH (s:N),(t:N) WHERE id(s)<>id(t)
               CALL algo.maxFlow({sourceNodes:[s], targetNodes:[t],
                   relationshipTypes:['R']})
               YIELD maxFlow, badField""",

            # extra / misspelled key alongside valid ones
            """MATCH (s:N),(t:N) WHERE id(s)<>id(t)
               CALL algo.maxFlow({sourceNodes:[s], targetNodes:[t],
                   relationshipTypes:['R'], oops:'yes'})
               YIELD maxFlow""",

            # defaultCapacity must be numeric
            """MATCH (s:N),(t:N) WHERE id(s)<>id(t)
               CALL algo.maxFlow({sourceNodes:[s], targetNodes:[t],
                   relationshipTypes:['R'], capacityProperty:'cap',
                   defaultCapacity:'bad'})
               YIELD maxFlow""",

            # defaultCapacity must be non-negative
            """MATCH (s:N),(t:N) WHERE id(s)<>id(t)
               CALL algo.maxFlow({sourceNodes:[s], targetNodes:[t],
                   relationshipTypes:['R'], capacityProperty:'cap',
                   defaultCapacity:-1})
               YIELD maxFlow""",
        ]

        for q in invalid_queries:
            try:
                self.graph.query(q)
                self.env.assertFalse(True)
            except Exception:
                # Each query exercises a different config-validation path; a
                # server-side error is the only acceptable outcome here.
                pass

    def test_01b_tensor_relationship_rejected(self):
        """Relationship types that contain multi-edges (tensors) must be
        rejected because maxFlow requires a simple adjacency matrix."""
        self.graph.query("""
            CREATE (a:Node {name:'A'})-[:R {cap:1}]->(b:Node {name:'B'}),
                   (a)-[:R {cap:2}]->(b)
        """)
        try:
            self.graph.query("""
                MATCH (s:Node {name:'A'}), (t:Node {name:'B'})
                CALL algo.maxFlow({
                    sourceNodes:       [s],
                    targetNodes:       [t],
                    capacityProperty:  'cap',
                    relationshipTypes: ['R']
                })
                YIELD maxFlow
            """)
            self.env.assertTrue(False, "Expected error for tensor relationship")
        except Exception as e:
            # Multi-edge (tensor) relationship → procedure rejects it.
            if "multi-edges" not in str(e):
                raise

    def test_01c_capacity_range_too_wide(self):
        """Reject graphs where the max capacity is >= min capacity * 2^32.
        LAGraph internally computes (a - b) for capacity values; when max and
        min differ by >= 2^32 the subtraction loses the smaller value entirely
        in double precision (i.e. (max - min) == max), causing the solver to
        hang.  The 2^32 threshold is a conservative cushion below the true
        double-precision limit to catch dangerous inputs early."""
        # min=1, max=2^32 satisfies max >= min * UINT32_MAX (2^32-1)
        cap_max = 2 ** 32
        self.graph.query(f"""
            CREATE (a:Node {{name:'A'}})-[:PIPE {{cap: 1}}]->
                   (b:Node {{name:'B'}})-[:PIPE {{cap: {cap_max}}}]->
                   (c:Node {{name:'C'}})
        """)
        try:
            self.graph.query("""
                MATCH (s:Node {name:'A'}), (t:Node {name:'C'})
                CALL algo.maxFlow({
                    sourceNodes:       [s],
                    targetNodes:       [t],
                    capacityProperty:  'cap',
                    relationshipTypes: ['PIPE']
                })
                YIELD maxFlow
            """)
            self.env.assertTrue(False, "Expected error for wide capacity range")
        except Exception as e:
            # Max capacity ≥ min * 2^32 would cause the solver to hang.
            if "capacity range too wide" not in str(e):
                raise

    # ------------------------------------------------------------------ #
    #  empty / trivial graphs                                            #
    # ------------------------------------------------------------------ #

    def test_02_max_flow_empty_graph(self):
        """Flow on an empty graph should throw an exceptiont."""
        self.graph.query("CREATE (a)-[e:PIPE]->(b) DELETE a,e,b")

        try:
            self.graph.query("""
                WITH [] AS sourceNodes, [] AS targetNodes
                CALL algo.maxFlow({
                    sourceNodes: sourceNodes,
                    targetNodes: targetNodes,
                    relationshipTypes: ['PIPE']
                })
                YIELD maxFlow
            """)
            self.env.assertFalse(True)
        except Exception as e:
            # Empty source/sink lists → procedure must raise an error.
            if "expects at least one source" not in str(e):
                raise

    def test_03_max_flow_no_path_between_source_and_sink(self):
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
            YIELD edgeFlows, maxFlow
        """)
        # either empty result set or flow == 0
        if len(result.result_set) > 0:
            self.env.assertEqual(result.result_set[0][0], [])
            self.env.assertEqual(result.result_set[0][1], 0)

    # ------------------------------------------------------------------ #
    #  single source, single sink – correctness                          #
    # ------------------------------------------------------------------ #

    def test_04_max_flow_simple_path(self):
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
            YIELD edgeFlows, maxFlow
        """)
        self.env.assertEqual(len(result.result_set), 1)
        self.env.assertEqual(result.result_set[0][1], 5)
        self.env.assertEqual(result.result_set[0][0], [5,5])

    def test_05_max_flow_bottleneck(self):
        """
        A -10-> B -3-> C -10-> D
        The bottleneck edge B->C limits flow to 3.
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
            YIELD edgeFlows, maxFlow
        """)
        self.env.assertEqual(result.result_set[0][1], 3)
        self.env.assertEqual(result.result_set[0][0], [3, 3, 3])

    def test_06_max_flow_parallel_paths(self):
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
            YIELD edgeFlows, maxFlow
        """)
        self.env.assertEqual(result.result_set[0][1], 8)
        self.env.assertEqual(result.result_set[0][0], [4, 4, 4, 4])

    def test_07_max_flow_default_capacity(self):
        """When defaultCapacity is given, edges with missing or non-numeric
        capacity attributes use that fallback value."""
        self.graph.query("""
            CREATE (a:Node {name:'A'}),
                   (b:Node {name:'B'}),
                   (c:Node {name:'C'}),
                   (d:Node {name:'D'}),
                   (a)-[:PIPE {cap:'a'}]->(b),
                   (a)-[:PIPE]->(c),
                   (b)-[:PIPE]->(d),
                   (c)-[:PIPE {cap: []}]->(d)
        """)
        result = self.graph.query("""
            MATCH (s:Node {name:'A'}), (t:Node {name:'D'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [t],
                capacityProperty:  'cap',
                defaultCapacity:   1,
                relationshipTypes: ['PIPE']
            })
            YIELD edgeFlows, maxFlow
        """)
        # two unit-capacity parallel paths -> flow = 2
        self.env.assertEqual(result.result_set[0][0], [1,1,1,1])
        self.env.assertEqual(result.result_set[0][1], 2)

    # ------------------------------------------------------------------ #
    #  yield – subsets of output columns                                 #
    # ------------------------------------------------------------------ #

    def test_08_yield_flow_only(self):
        """Requesting only YIELD flow must succeed and return a numeric value."""
        self.graph.query("""
            CREATE (a:Node {name:'A'})-[:PIPE {cap:7}]->(b:Node {name:'B'})
        """)
        result = self.graph.query("""
            MATCH (s:Node {name:'A'}), (t:Node {name:'B'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [t],
                capacityProperty:  'cap',
                relationshipTypes: ['PIPE']
            })
            YIELD maxFlow
        """)
        self.env.assertEqual(len(result.result_set), 1)
        self.env.assertEqual(result.result_set[0][0], 7)

    def test_09_yield_nodes_only(self):
        """Requesting only YIELD nodes must return a non-empty array."""
        nodes = self.graph.query("""
            CREATE (a:Node {name:'A'})-[:PIPE {cap:7}]->(b:Node {name:'B'})
            RETURN a, b
        """).result_set[0]

        result = self.graph.query("""
            MATCH (s:Node {name:'A'}), (t:Node {name:'B'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [t],
                capacityProperty:  'cap',
                relationshipTypes: ['PIPE']
            })
            YIELD nodes
        """)
        self.env.assertEqual(len(result.result_set), 1)
        yield_nodes = result.result_set[0][0]
        self.env.assertEqual(nodes, yield_nodes)

    def test_10_yield_edges_only(self):
        """Requesting only YIELD edges must return a non-empty array."""
        edges = self.graph.query("""
            CREATE (a:Node {name:'A'})-[e:PIPE {cap:7}]->(b:Node {name:'B'})
            RETURN e
        """).result_set[0]
        result = self.graph.query("""
            MATCH (s:Node {name:'A'}), (t:Node {name:'B'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [t],
                capacityProperty:  'cap',
                relationshipTypes: ['PIPE']
            })
            YIELD edges
        """)
        self.env.assertEqual(len(result.result_set), 1)
        yield_edges = result.result_set[0][0]
        self.env.assertEqual(edges, yield_edges)

    def test_11_yield_nodes_edges_flow(self):
        """
        All three yield columns must be consistent with each other:
        - nodes includes exactly the endpoints that appear in edges
        - len(edges) == len(edgeFlows)
        - each flow value is > 0
        """
        self.graph.query("""
            CREATE (a:Node {name:'A'}),
                   (b:Node {name:'B'}),
                   (c:Node {name:'C'}),
                   (a)-[:PIPE {cap:5}]->(b),
                   (b)-[:PIPE {cap:5}]->(c)
        """)

        # warm the client schema so edge parsing succeeds
        self.graph.query("MATCH ()-[e:PIPE]->() RETURN e LIMIT 1")

        result = self.graph.query("""
            MATCH (s:Node {name:'A'}), (t:Node {name:'C'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [t],
                capacityProperty:  'cap',
                relationshipTypes: ['PIPE']
            })
            YIELD nodes, edges, edgeFlows, maxFlow
        """)
        self.env.assertEqual(len(result.result_set), 1)
        nodes, edges, flows, maxflow = result.result_set[0]

        self.env.assertEqual(len(nodes), 3)
        self.env.assertEqual(len(edges), 2)
        self.env.assertEqual(flows, [5,5])
        self.env.assertEqual(maxflow, 5)

        node_ids = [n.id for n in nodes]
        for idx, e in enumerate(edges):
            self.env.assertIn(e.src_node, node_ids)
            self.env.assertTrue(e.properties['cap'] == flows[idx])

    # ------------------------------------------------------------------ #
    #  multi-source, single sink                                         #
    # ------------------------------------------------------------------ #

    def test_12_max_flow_multi_source_single_sink(self):
        """
        Two independent sources feeding a common sink.

          A -5-> C
          B -3-> C

        Super-source S' connects to both A and B with ∞ capacity.
        Max flow = 5 + 3 = 8.
        """
        self.graph.query("""
            CREATE (a:Node {name:'A'}),
                   (b:Node {name:'B'}),
                   (c:Node {name:'C'}),
                   (a)-[:PIPE {cap:5}]->(c),
                   (b)-[:PIPE {cap:3}]->(c)
        """)
        result = self.graph.query("""
            MATCH (a:Node {name:'A'}),
                  (b:Node {name:'B'}),
                  (t:Node {name:'C'})
            CALL algo.maxFlow({
                sourceNodes:       [a, b],
                targetNodes:       [t],
                capacityProperty:  'cap',
                relationshipTypes: ['PIPE']
            })
            YIELD nodes, maxFlow
        """)
        self.env.assertEqual(result.result_set[0][1], 8)
        self.env.assertEqual(len(result.result_set[0][0]), 3)

    def test_13_max_flow_multi_source_bottleneck(self):
        """
        Sources A and B both feed into C which is a bottleneck (cap 4) to D.

          A -6-> C -4-> D
          B -6-> C

        Total in to C = 12, but outgoing cap = 4 -> max flow = 4.
        """
        self.graph.query("""
            CREATE (a:Node {name:'A'}),
                   (b:Node {name:'B'}),
                   (c:Node {name:'C'}),
                   (d:Node {name:'D'}),
                   (a)-[:PIPE {cap:6}]->(c),
                   (b)-[:PIPE {cap:6}]->(c),
                   (c)-[:PIPE {cap:4}]->(d)
        """)
        result = self.graph.query("""
            MATCH (a:Node {name:'A'}),
                  (b:Node {name:'B'}),
                  (t:Node {name:'D'})
            CALL algo.maxFlow({
                sourceNodes:       [a, b],
                targetNodes:       [t],
                capacityProperty:  'cap',
                relationshipTypes: ['PIPE']
            })
            YIELD maxFlow
        """)
        self.env.assertEqual(result.result_set[0][0], 4)

    # ------------------------------------------------------------------ #
    #  single source, multi-sink                                         #
    # ------------------------------------------------------------------ #

    def test_14_max_flow_single_source_multi_sink(self):
        """
        One source, two independent sinks.

          A -5-> B
          A -3-> C

        Super-sink T' receives from B and C with ∞ capacity.
        Max flow = 5 + 3 = 8.
        """
        self.graph.query("""
            CREATE (a:Node {name:'A'}),
                   (b:Node {name:'B'}),
                   (c:Node {name:'C'}),
                   (a)-[:PIPE {cap:5}]->(b),
                   (a)-[:PIPE {cap:3}]->(c)
        """)
        result = self.graph.query("""
            MATCH (s:Node  {name:'A'}),
                  (b:Node  {name:'B'}),
                  (c:Node  {name:'C'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [b, c],
                capacityProperty:  'cap',
                relationshipTypes: ['PIPE']
            })
            YIELD nodes, maxFlow
        """)
        self.env.assertEqual(len(result.result_set[0][0]), 3)
        self.env.assertEqual(result.result_set[0][1], 8)

    def test_15_max_flow_single_source_multi_sink_bottleneck(self):
        """
        Source bottleneck constrains multi-sink total.

          A -2-> B -6-> D
          A -4-> C -6-> E

        Source cap out = 6, flow = 6.
        """
        self.graph.query("""
            CREATE (a:Node {name:'A'}),
                   (b:Node {name:'B'}),
                   (c:Node {name:'C'}),
                   (d:Node {name:'D'}),
                   (e:Node {name:'E'}),
                   (a)-[:PIPE {cap:2}]->(b),
                   (a)-[:PIPE {cap:4}]->(c),
                   (b)-[:PIPE {cap:6}]->(d),
                   (c)-[:PIPE {cap:6}]->(e)
        """)
        result = self.graph.query("""
            MATCH (s:Node {name:'A'}),
                  (d:Node {name:'D'}),
                  (e:Node {name:'E'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [d, e],
                capacityProperty:  'cap',
                relationshipTypes: ['PIPE']
            })
            YIELD maxFlow
        """)
        self.env.assertEqual(result.result_set[0][0], 6)

    # ------------------------------------------------------------------ #
    #  multi-source, multi-sink                                          #
    # ------------------------------------------------------------------ #

    def test_16_max_flow_multi_source_multi_sink(self):
        """
        Two sources and two sinks in a diamond.

          A -5-> C -5-> D
          B -5-> C -5-> E

        Sources: A, B  /  Sinks: D, E
        Max flow = 10.
        """
        self.graph.query("""
            CREATE (a:Node {name:'A'}),
                   (b:Node {name:'B'}),
                   (c:Node {name:'C'}),
                   (d:Node {name:'D'}),
                   (e:Node {name:'E'}),
                   (a)-[:PIPE {cap:5}]->(c),
                   (b)-[:PIPE {cap:5}]->(c),
                   (c)-[:PIPE {cap:5}]->(d),
                   (c)-[:PIPE {cap:5}]->(e)
        """)
        result = self.graph.query("""
            MATCH (a:Node {name:'A'}),
                  (b:Node {name:'B'}),
                  (d:Node {name:'D'}),
                  (e:Node {name:'E'})
            CALL algo.maxFlow({
                sourceNodes:       [a, b],
                targetNodes:       [d, e],
                capacityProperty:  'cap',
                relationshipTypes: ['PIPE']
            })
            YIELD maxFlow
        """)
        self.env.assertEqual(result.result_set[0][0], 10)

    def test_17_max_flow_multi_source_multi_sink_asymmetric(self):
        """
        Asymmetric capacities ensure the super-source / super-sink trick
        does not over-count or under-count.

          A -3-> C -2-> E
          A -3-> D -4-> F
          B -6-> C
          B -6-> D

        Sources: A, B  /  Sinks: E, F
        Bottlenecks: C->E (2), A->C (3), A->D (3), B->D (6)
        Max flow = 2 + 4 = 6
        """
        self.graph.query("""
            CREATE (a:Node {name:'A'}),
                   (b:Node {name:'B'}),
                   (c:Node {name:'C'}),
                   (d:Node {name:'D'}),
                   (e:Node {name:'E'}),
                   (f:Node {name:'F'}),
                   (a)-[:PIPE {cap:3}]->(c),
                   (a)-[:PIPE {cap:3}]->(d),
                   (b)-[:PIPE {cap:6}]->(c),
                   (b)-[:PIPE {cap:6}]->(d),
                   (c)-[:PIPE {cap:2}]->(e),
                   (d)-[:PIPE {cap:4}]->(f)
        """)
        result = self.graph.query("""
            MATCH (a:Node {name:'A'}),
                  (b:Node {name:'B'}),
                  (e:Node {name:'E'}),
                  (f:Node {name:'F'})
            CALL algo.maxFlow({
                sourceNodes:       [a, b],
                targetNodes:       [e, f],
                capacityProperty:  'cap',
                relationshipTypes: ['PIPE']
            })
            YIELD maxFlow
        """)
        self.env.assertEqual(result.result_set[0][0], 6)

    # ------------------------------------------------------------------ #
    #  multi-source / multi-sink output sanity                           #
    # ------------------------------------------------------------------ #

    def test_18_multi_source_sink_no_super_nodes_in_output(self):
        """
        The synthetic super-source and super-sink must NOT appear in the
        returned nodes or edges arrays – they are implementation details.
        """
        self.graph.query("""
            CREATE (a:Node {name:'A'}),
                   (b:Node {name:'B'}),
                   (c:Node {name:'C'}),
                   (d:Node {name:'D'}),
                   (a)-[:PIPE {cap:5}]->(c),
                   (b)-[:PIPE {cap:5}]->(c),
                   (c)-[:PIPE {cap:5}]->(d)
        """)
        # fetch the max node id before the call so we can detect phantoms
        max_id_result = self.graph.query("MATCH (n) RETURN max(id(n))")
        max_real_id   = max_id_result.result_set[0][0]

        result = self.graph.query("""
            MATCH (a:Node {name:'A'}),
                  (b:Node {name:'B'}),
                  (t:Node {name:'D'})
            CALL algo.maxFlow({
                sourceNodes:       [a, b],
                targetNodes:       [t],
                capacityProperty:  'cap',
                relationshipTypes: ['PIPE']
            })
            YIELD nodes, edges
        """)

        self.env.assertEqual(len(result.result_set), 1)
        nodes, edges = result.result_set[0]

        for node in nodes:
            self.env.assertLessEqual(node.id, max_real_id)

        for edge in edges:
            self.env.assertLessEqual(edge.src_node, max_real_id)
            self.env.assertLessEqual(edge.dest_node, max_real_id)

        node_count = self.graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
        self.env.assertEqual(node_count, 4)

    # ------------------------------------------------------------------ #
    #  nodeLabels filter                                                 #
    # ------------------------------------------------------------------ #

    def test_19_max_flow_node_labels_filter(self):
        """
        Graph has two node types; filtering to only 'Pipe' nodes should
        exclude 'Valve' nodes and therefore reduce the max flow.

          A:Pipe -5-> V:Valve -5-> B:Pipe
          A:Pipe -3-----------> B:Pipe    (direct edge, same label)

        With nodeLabels=['Pipe']: only the direct A->B path (cap 3) is visible
        -> flow = 3.
        Without filter both paths exist -> flow = 8 (if we remove the valve
        from the path the shortest route carries 3; including valve path
        gives 5+3=8 but through nodeLabels only Pipe nodes count so valve
        path disappears -> flow = 3).
        """
        self.graph.query("""
            CREATE (a:Pipe   {name:'A'}),
                   (v:Valve  {name:'V'}),
                   (b:Pipe   {name:'B'}),
                   (a)-[:FLOW {cap:5}]->(v),
                   (v)-[:FLOW {cap:5}]->(b),
                   (a)-[:FLOW {cap:3}]->(b)
        """)

        # without label filter
        result_all = self.graph.query("""
            MATCH (s:Pipe {name:'A'}), (t:Pipe {name:'B'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [t],
                capacityProperty:  'cap',
                relationshipTypes: ['FLOW']
            })
            YIELD maxFlow
        """)

        # with label filter (Valve node excluded)
        result_filtered = self.graph.query("""
            MATCH (s:Pipe {name:'A'}), (t:Pipe {name:'B'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [t],
                capacityProperty:  'cap',
                nodeLabels:        ['Pipe'],
                relationshipTypes: ['FLOW']
            })
            YIELD maxFlow
        """)

        flow_all      = result_all.result_set[0][0]
        flow_filtered = result_filtered.result_set[0][0]

        # Filtering out the valve node must reduce (or equal) flow
        self.env.assertGreaterEqual(flow_all, flow_filtered)
        self.env.assertEqual(flow_filtered, 3)

    # ------------------------------------------------------------------ #
    #  relationshipTypes filter                                          #
    # ------------------------------------------------------------------ #

    def test_20_max_flow_relationship_type_filter(self):
        """
        Same node topology, different relationship types.
        Using only 'PIPE' edges ignores 'CABLE' edges.

          A -5[PIPE]-> B -5[PIPE]-> D
          A -5[CABLE]-> C -5[CABLE]-> D

        YIELD maxFlow with PIPE only -> 5.
        """
        self.graph.query("""
            CREATE (a:Node {name:'A'}),
                   (b:Node {name:'B'}),
                   (c:Node {name:'C'}),
                   (d:Node {name:'D'}),
                   (a)-[:PIPE  {cap:5}]->(b),
                   (b)-[:PIPE  {cap:5}]->(d),
                   (a)-[:CABLE {cap:5}]->(c),
                   (c)-[:CABLE {cap:5}]->(d)
        """)
        result = self.graph.query("""
            MATCH (s:Node {name:'A'}), (t:Node {name:'D'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [t],
                capacityProperty:  'cap',
                relationshipTypes: ['PIPE']
            })
            YIELD maxFlow
        """)
        self.env.assertEqual(result.result_set[0][0], 5)

    # ------------------------------------------------------------------ #
    #  single-element arrays equal single source / sink behaviour        #
    # ------------------------------------------------------------------ #

    def test_21_single_element_array_equals_scalar(self):
        """
        [src] / [sink] (length-1 arrays) must produce the same flow as the
        plain single-source / single-sink call.
        """
        self.graph.query("""
            CREATE (a:Node {name:'A'})-[:PIPE {cap:9}]->(b:Node {name:'B'})
        """)

        result = self.graph.query("""
            MATCH (s:Node {name:'A'}), (t:Node {name:'B'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [t],
                capacityProperty:  'cap',
                relationshipTypes: ['PIPE']
            })
            YIELD maxFlow
        """)
        self.env.assertEqual(result.result_set[0][0], 9)

    # ------------------------------------------------------------------ #
    #  source == sink (degenerate case)                                  #
    # ------------------------------------------------------------------ #

    def test_22_source_equals_sink(self):
        """When source and sink are the same node, the procedure must error
        because source and sink sets are not disjoint."""
        self.graph.query("""
            CREATE (a:Node {name:'A'})-[:PIPE {cap:5}]->(b:Node {name:'B'})
        """)
        try:
            self.graph.query("""
                MATCH (s:Node {name:'A'})
                CALL algo.maxFlow({
                    sourceNodes:       [s],
                    targetNodes:       [s],
                    capacityProperty:  'cap',
                    relationshipTypes: ['PIPE']
                })
                YIELD maxFlow
            """)
            self.env.assertTrue(False, "Expected error for overlapping "
                                       "source and sink sets")
        except Exception as e:
            # source == sink means source/sink sets are not disjoint.
            if "disjoint" not in str(e):
                raise

    def test_23_overlapping_multi_source_sink_errors(self):
        """When a node appears in both sourceNodes and targetNodes the
        procedure must raise an error (sets not disjoint)."""
        self.graph.query("""
            CREATE (a:Node {name:'A'}),
                   (b:Node {name:'B'}),
                   (c:Node {name:'C'}),
                   (a)-[:PIPE {cap:5}]->(b),
                   (b)-[:PIPE {cap:5}]->(c)
        """)
        try:
            self.graph.query("""
                MATCH (a:Node {name:'A'}),
                      (b:Node {name:'B'}),
                      (c:Node {name:'C'})
                CALL algo.maxFlow({
                    sourceNodes:       [a, b],
                    targetNodes:       [c, a],
                    capacityProperty:  'cap',
                    relationshipTypes: ['PIPE']
                })
                YIELD maxFlow
            """)
            self.env.assertTrue(False, "Expected error for overlapping "
                                       "source and sink sets")
        except Exception as e:
            # A node in both sourceNodes and targetNodes → sets not disjoint.
            if "disjoint" not in str(e):
                raise

    def test_23b_source_also_in_targets(self):
        """When a source node also appears in the target set the procedure
        must raise an error (sets not disjoint)."""
        self.graph.query("""
            CREATE (s:Node {name:'S'}),
                   (t:Node {name:'T'}),
                   (s)-[:PIPE {cap:5}]->(t)
        """)
        try:
            self.graph.query("""
                MATCH (s:Node {name:'S'}),
                      (t:Node {name:'T'})
                CALL algo.maxFlow({
                    sourceNodes:       [s, t],
                    targetNodes:       [t],
                    capacityProperty:  'cap',
                    relationshipTypes: ['PIPE']
                })
                YIELD maxFlow
            """)
            self.env.assertTrue(False, "Expected error for overlapping "
                                       "source and sink sets")
        except Exception as e:
            # Source set contains the sink node → sets not disjoint.
            if "disjoint" not in str(e):
                raise

    def test_23c_target_also_in_sources(self):
        """When a target node also appears in the source set the procedure
        must raise an error (sets not disjoint)."""
        self.graph.query("""
            CREATE (s:Node {name:'S'}),
                   (t:Node {name:'T'}),
                   (s)-[:PIPE {cap:5}]->(t)
        """)
        try:
            self.graph.query("""
                MATCH (s:Node {name:'S'}),
                      (t:Node {name:'T'})
                CALL algo.maxFlow({
                    sourceNodes:       [s],
                    targetNodes:       [s, t],
                    capacityProperty:  'cap',
                    relationshipTypes: ['PIPE']
                })
                YIELD maxFlow
            """)
            self.env.assertTrue(False, "Expected error for overlapping "
                                       "source and sink sets")
        except Exception as e:
            # Target set contains the source node → sets not disjoint.
            if "disjoint" not in str(e):
                raise

    def test_23d_multi_source_multi_sink_shared_node(self):
        """Multi-source and multi-sink sets that share exactly one node
        must be rejected."""
        self.graph.query("""
            CREATE (s0:Node {name:'S0'}),
                   (s1:Node {name:'S1'}),
                   (s2:Node {name:'S2'}),
                   (x:Node  {name:'X'}),
                   (t0:Node {name:'T0'}),
                   (t1:Node {name:'T1'}),
                   (t2:Node {name:'T2'}),
                   (s0)-[:PIPE {cap:1}]->(t0),
                   (s1)-[:PIPE {cap:1}]->(t1),
                   (s2)-[:PIPE {cap:1}]->(t2),
                   (x)-[:PIPE  {cap:1}]->(t0)
        """)
        try:
            self.graph.query("""
                MATCH (s0:Node {name:'S0'}),
                      (s1:Node {name:'S1'}),
                      (s2:Node {name:'S2'}),
                      (x:Node  {name:'X'}),
                      (t0:Node {name:'T0'}),
                      (t1:Node {name:'T1'}),
                      (t2:Node {name:'T2'})
                CALL algo.maxFlow({
                    sourceNodes:       [s0, s1, s2, x],
                    targetNodes:       [t0, t1, t2, x],
                    capacityProperty:  'cap',
                    relationshipTypes: ['PIPE']
                })
                YIELD maxFlow
            """)
            self.env.assertTrue(False, "Expected error for overlapping "
                                       "source and sink sets")
        except Exception as e:
            # Node x in both source and target sets → sets not disjoint.
            if "disjoint" not in str(e):
                raise

    # ------------------------------------------------------------------ #
    #  numeric capacity types: integer, double, mixed, zero              #
    # ------------------------------------------------------------------ #

    def test_24_max_flow_integer_capacities(self):
        """Integer capacities of various magnitudes."""
        self.graph.query("""
            CREATE (a:Node {name:'A'}),
                   (b:Node {name:'B'}),
                   (c:Node {name:'C'}),
                   (d:Node {name:'D'}),
                   (a)-[:PIPE {cap:100}]->(b),
                   (a)-[:PIPE {cap:200}]->(c),
                   (b)-[:PIPE {cap:150}]->(d),
                   (c)-[:PIPE {cap:50}]->(d)
        """)
        result = self.graph.query("""
            MATCH (s:Node {name:'A'}), (t:Node {name:'D'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [t],
                capacityProperty:  'cap',
                relationshipTypes: ['PIPE']
            })
            YIELD maxFlow
        """)
        # path A->B->D carries min(100,150)=100; path A->C->D carries min(200,50)=50
        # total = 150
        self.env.assertEqual(result.result_set[0][0], 150)

    def test_25_max_flow_double_capacities(self):
        """Fractional (double) capacities."""
        self.graph.query("""
            CREATE (a:Node {name:'A'}),
                   (b:Node {name:'B'}),
                   (c:Node {name:'C'}),
                   (a)-[:PIPE {cap:2.5}]->(b),
                   (b)-[:PIPE {cap:1.75}]->(c)
        """)
        result = self.graph.query("""
            MATCH (s:Node {name:'A'}), (t:Node {name:'C'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [t],
                capacityProperty:  'cap',
                relationshipTypes: ['PIPE']
            })
            YIELD maxFlow
        """)
        # bottleneck is B->C at 1.75
        self.env.assertAlmostEqual(result.result_set[0][0], 1.75, 5)

    def test_26_max_flow_mixed_int_double_capacities(self):
        """Mix of integer and double capacities on different edges."""
        self.graph.query("""
            CREATE (a:Node {name:'A'}),
                   (b:Node {name:'B'}),
                   (c:Node {name:'C'}),
                   (d:Node {name:'D'}),
                   (a)-[:PIPE {cap:10}]->(b),
                   (a)-[:PIPE {cap:3.5}]->(c),
                   (b)-[:PIPE {cap:7.25}]->(d),
                   (c)-[:PIPE {cap:6}]->(d)
        """)
        result = self.graph.query("""
            MATCH (s:Node {name:'A'}), (t:Node {name:'D'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [t],
                capacityProperty:  'cap',
                relationshipTypes: ['PIPE']
            })
            YIELD maxFlow
        """)
        # path A->B->D carries min(10,7.25)=7.25
        # path A->C->D carries min(3.5,6)=3.5
        # total = 10.75
        self.env.assertAlmostEqual(result.result_set[0][0], 10.75, 5)

    def test_27_max_flow_zero_capacity_edge(self):
        """An edge with capacity 0 should not carry any flow."""
        self.graph.query("""
            CREATE (a:Node {name:'A'}),
                   (b:Node {name:'B'}),
                   (c:Node {name:'C'}),
                   (a)-[:PIPE {cap:0}]->(b),
                   (b)-[:PIPE {cap:10}]->(c)
        """)
        result = self.graph.query("""
            MATCH (s:Node {name:'A'}), (t:Node {name:'C'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [t],
                capacityProperty:  'cap',
                relationshipTypes: ['PIPE']
            })
            YIELD maxFlow
        """)
        self.env.assertEqual(result.result_set[0][0], 0)

    def test_28_max_flow_large_integer_capacities(self):
        """Large integer capacities (millions)."""
        self.graph.query("""
            CREATE (a:Node {name:'A'})-[:PIPE {cap:1000000}]->(b:Node {name:'B'})
        """)
        result = self.graph.query("""
            MATCH (s:Node {name:'A'}), (t:Node {name:'B'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [t],
                capacityProperty:  'cap',
                relationshipTypes: ['PIPE']
            })
            YIELD maxFlow
        """)
        self.env.assertEqual(result.result_set[0][0], 1000000)

    def test_29_max_flow_small_double_capacities(self):
        """Very small fractional capacities."""
        self.graph.query("""
            CREATE (a:Node {name:'A'})-[:PIPE {cap:0.001}]->(b:Node {name:'B'})
        """)
        result = self.graph.query("""
            MATCH (s:Node {name:'A'}), (t:Node {name:'B'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [t],
                capacityProperty:  'cap',
                relationshipTypes: ['PIPE']
            })
            YIELD maxFlow
        """)
        self.env.assertAlmostEqual(result.result_set[0][0], 0.001, 6)

    # ------------------------------------------------------------------ #
    #  missing / non-numeric capacity without defaultCapacity → error    #
    # ------------------------------------------------------------------ #

    def test_30_max_flow_missing_attr_no_default_errors(self):
        """When an edge lacks the capacity attribute and no defaultCapacity
        is configured, the procedure must raise an error."""
        self.graph.query("""
            CREATE (a:Node {name:'A'})-[:PIPE]->(b:Node {name:'B'})
        """)
        # ensure the attribute 'cap' exists in the schema
        self.graph.query("""
            CREATE (:Tmp {cap:1})
        """)
        try:
            self.graph.query("""
                MATCH (s:Node {name:'A'}), (t:Node {name:'B'})
                CALL algo.maxFlow({
                    sourceNodes:       [s],
                    targetNodes:       [t],
                    capacityProperty:  'cap',
                    relationshipTypes: ['PIPE']
                })
                YIELD maxFlow
            """)
            self.env.assertTrue(False, "Expected error for missing attribute "
                                       "without defaultCapacity")
        except Exception as e:
            # Edge missing the capacity attribute with no defaultCapacity → error.
            if "invalid or missing attribute" not in str(e):
                raise

    def test_31_max_flow_string_attr_no_default_errors(self):
        """When the capacity attribute is a string and no defaultCapacity
        is configured, the procedure must raise an error."""
        self.graph.query("""
            CREATE (a:Node {name:'A'})-[:PIPE {cap:'not_a_number'}]->(b:Node {name:'B'})
        """)
        try:
            self.graph.query("""
                MATCH (s:Node {name:'A'}), (t:Node {name:'B'})
                CALL algo.maxFlow({
                    sourceNodes:       [s],
                    targetNodes:       [t],
                    capacityProperty:  'cap',
                    relationshipTypes: ['PIPE']
                })
                YIELD maxFlow
            """)
            self.env.assertTrue(False, "Expected error for string capacity "
                                       "without defaultCapacity")
        except Exception as e:
            # String capacity attribute with no defaultCapacity → error.
            if "invalid or missing attribute" not in str(e):
                raise

    def test_32_max_flow_array_attr_no_default_errors(self):
        """When the capacity attribute is an array and no defaultCapacity
        is configured, the procedure must raise an error."""
        self.graph.query("""
            CREATE (a:Node {name:'A'})-[:PIPE {cap:[1,2,3]}]->(b:Node {name:'B'})
        """)
        try:
            self.graph.query("""
                MATCH (s:Node {name:'A'}), (t:Node {name:'B'})
                CALL algo.maxFlow({
                    sourceNodes:       [s],
                    targetNodes:       [t],
                    capacityProperty:  'cap',
                    relationshipTypes: ['PIPE']
                })
                YIELD maxFlow
            """)
            self.env.assertTrue(False, "Expected error for array capacity "
                                       "without defaultCapacity")
        except Exception as e:
            # Array capacity attribute with no defaultCapacity → error.
            if "invalid or missing attribute" not in str(e):
                raise

    def test_33_max_flow_mixed_valid_invalid_attr_no_default_errors(self):
        """Even if some edges have valid numeric capacity, one edge with a
        non-numeric attribute and no defaultCapacity must cause an error."""
        self.graph.query("""
            CREATE (a:Node {name:'A'}),
                   (b:Node {name:'B'}),
                   (c:Node {name:'C'}),
                   (a)-[:PIPE {cap:5}]->(b),
                   (b)-[:PIPE {cap:'bad'}]->(c)
        """)
        try:
            self.graph.query("""
                MATCH (s:Node {name:'A'}), (t:Node {name:'C'})
                CALL algo.maxFlow({
                    sourceNodes:       [s],
                    targetNodes:       [t],
                    capacityProperty:  'cap',
                    relationshipTypes: ['PIPE']
                })
                YIELD maxFlow
            """)
            self.env.assertTrue(False, "Expected error for mixed valid/invalid "
                                       "capacities without defaultCapacity")
        except Exception as e:
            # Even one non-numeric edge with no defaultCapacity → error.
            if "invalid or missing attribute" not in str(e):
                raise

    # ------------------------------------------------------------------ #
    #  defaultCapacity fallback                                          #
    # ------------------------------------------------------------------ #

    def test_34_max_flow_default_capacity_with_missing_attr(self):
        """When defaultCapacity is provided, edges missing the capacity
        attribute use the fallback value."""
        self.graph.query("""
            CREATE (a:Node {name:'A'}),
                   (b:Node {name:'B'}),
                   (c:Node {name:'C'}),
                   (a)-[:PIPE {cap:10}]->(b),
                   (b)-[:PIPE]->(c)
        """)
        # ensure 'cap' attribute exists in the schema
        self.graph.query("CREATE (:Tmp {cap:1})")
        result = self.graph.query("""
            MATCH (s:Node {name:'A'}), (t:Node {name:'C'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [t],
                capacityProperty:  'cap',
                defaultCapacity:   3,
                relationshipTypes: ['PIPE']
            })
            YIELD maxFlow
        """)
        # A->B cap=10, B->C cap=3 (default) → bottleneck = 3
        self.env.assertEqual(result.result_set[0][0], 3)

    def test_35_max_flow_default_capacity_zero(self):
        """defaultCapacity: 0 turns missing-attribute edges into zero-capacity
        edges, effectively blocking flow through them."""
        self.graph.query("""
            CREATE (a:Node {name:'A'}),
                   (b:Node {name:'B'}),
                   (c:Node {name:'C'}),
                   (a)-[:PIPE {cap:10}]->(b),
                   (b)-[:PIPE]->(c)
        """)
        self.graph.query("CREATE (:Tmp {cap:1})")
        result = self.graph.query("""
            MATCH (s:Node {name:'A'}), (t:Node {name:'C'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [t],
                capacityProperty:  'cap',
                defaultCapacity:   0,
                relationshipTypes: ['PIPE']
            })
            YIELD maxFlow
        """)
        self.env.assertEqual(result.result_set[0][0], 0)

    def test_36_max_flow_default_capacity_double(self):
        """defaultCapacity accepts a fractional (double) fallback value."""
        self.graph.query("""
            CREATE (a:Node {name:'A'})-[:PIPE {cap:'text'}]->(b:Node {name:'B'})
        """)
        result = self.graph.query("""
            MATCH (s:Node {name:'A'}), (t:Node {name:'B'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [t],
                capacityProperty:  'cap',
                defaultCapacity:   4.5,
                relationshipTypes: ['PIPE']
            })
            YIELD maxFlow
        """)
        self.env.assertAlmostEqual(result.result_set[0][0], 4.5, 5)

    def test_37_no_edges_of_relationship_type(self):
        """When the relationship type exists but no edges survive the label
        filter, maxFlow should return one row with maxFlow=0 and empty arrays."""
        # Create a PIPE edge (registers the rel-type in the schema) between
        # nodes labelled 'X', then query between nodes labelled 'Y' where no
        # PIPE edges exist – Build_Matrix produces an empty matrix.
        self.graph.query("""
            CREATE (:X {name:'P'})-[:PIPE {cap:5}]->(:X {name:'Q'}),
                   (:Y {name:'A'}),
                   (:Y {name:'B'})
        """)
        result = self.graph.query("""
            MATCH (s:Y {name:'A'}), (t:Y {name:'B'})
            CALL algo.maxFlow({
                sourceNodes:       [s],
                targetNodes:       [t],
                capacityProperty:  'cap',
                nodeLabels:        ['Y'],
                relationshipTypes: ['PIPE']
            })
            YIELD nodes, edges, edgeFlows, maxFlow
        """)
        self.env.assertEqual(len(result.result_set), 1)
        row = result.result_set[0]
        self.env.assertEqual(row[0], [])       # nodes
        self.env.assertEqual(row[1], [])       # edges
        self.env.assertEqual(row[2], [])       # edgeFlows
        self.env.assertEqual(row[3], 0.0)      # maxFlow
