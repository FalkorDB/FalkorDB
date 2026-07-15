from common import *

GRAPH_ID = "contraction_hierarchies"

class testContractionHierarchies(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        try:
            self.graph.delete()
        except:
            pass

    def test_invalid_invocation(self):
        invalid_queries = [
            # missing required keys
            """CALL algo.contractionHierarchies({})""",
            """CALL algo.contractionHierarchies({relTypes: ['ROAD']})""",
            """CALL algo.contractionHierarchies({relTypes: ['ROAD'],
                weightProp: 'cost'})""",
            """CALL algo.contractionHierarchies({relTypes: ['ROAD'],
                weightProp: 'cost', shortcutRelType: 'SHORTCUT'})""",

            # wrong types
            """CALL algo.contractionHierarchies({relTypes: 'ROAD',
                weightProp: 'cost', shortcutRelType: 'SHORTCUT',
                rankProperty: 'rank'})""",                          # non-array relTypes
            """CALL algo.contractionHierarchies({relTypes: [1, 2],
                weightProp: 'cost', shortcutRelType: 'SHORTCUT',
                rankProperty: 'rank'})""",                          # integers in relTypes
            """CALL algo.contractionHierarchies({relTypes: ['ROAD'],
                weightProp: 4, shortcutRelType: 'SHORTCUT',
                rankProperty: 'rank'})""",                          # non-string weightProp
            """CALL algo.contractionHierarchies({relTypes: ['ROAD'],
                weightProp: 'cost', shortcutRelType: 4,
                rankProperty: 'rank'})""",                          # non-string shortcutRelType
            """CALL algo.contractionHierarchies({relTypes: ['ROAD'],
                weightProp: 'cost', shortcutRelType: 'SHORTCUT',
                rankProperty: 4})""",                                # non-string rankProperty

            # empty relTypes
            """CALL algo.contractionHierarchies({relTypes: [],
                weightProp: 'cost', shortcutRelType: 'SHORTCUT',
                rankProperty: 'rank'})""",

            # non-existent relationship type / attribute
            """CALL algo.contractionHierarchies({relTypes: ['FAKE'],
                weightProp: 'cost', shortcutRelType: 'SHORTCUT',
                rankProperty: 'rank'})""",
            """CALL algo.contractionHierarchies({relTypes: ['ROAD'],
                weightProp: 'fake', shortcutRelType: 'SHORTCUT',
                rankProperty: 'rank'})""",

            # unexpected extra key
            """CALL algo.contractionHierarchies({relTypes: ['ROAD'],
                weightProp: 'cost', shortcutRelType: 'SHORTCUT',
                rankProperty: 'rank', extra: 'value'})""",

            # invalid configuration type / missing argument entirely
            """CALL algo.contractionHierarchies('invalid')""",
            """CALL algo.contractionHierarchies()""",
        ]

        # a single ROAD edge so 'weightProp'/'relTypes' resolve to real ids
        # for the queries above that expect that part of validation to pass
        self.graph.query("""
            CREATE (a {id: 1})-[:ROAD {cost: 1}]->(b {id: 2})
        """)

        for q in invalid_queries:
            try:
                self.graph.query(q)
                self.env.assertFalse(True)
            except redis.exceptions.ResponseError:
                pass

    def test_contraction_hierarchies_on_cycle(self):
        """A 4-node directed cycle A->B->C->D->A (unit weights) has a
        deterministic contraction outcome regardless of tie-breaking order:
        contracting any node in a simple cycle always requires exactly one
        shortcut (there's no alternate route around the cycle cheap enough
        to serve as a witness), and this holds recursively as the cycle
        shrinks 4 -> 3 -> 2 nodes; the final 2-node "cycle" resolves
        trivially (its two neighbors are the same node), requiring no
        further shortcuts. So the expected shortcut count -- exactly 2 --
        doesn't depend on which node the heap happens to pop first."""

        self.graph.query("""
            CREATE
            (a {id: 1}),
            (b {id: 2}),
            (c {id: 3}),
            (d {id: 4}),
            (a)-[:ROAD {cost: 1}]->(b),
            (b)-[:ROAD {cost: 1}]->(c),
            (c)-[:ROAD {cost: 1}]->(d),
            (d)-[:ROAD {cost: 1}]->(a)
        """)

        result = self.graph.query("""
            CALL algo.contractionHierarchies({relTypes: ['ROAD'],
                weightProp: 'cost', shortcutRelType: 'SHORTCUT',
                rankProperty: 'rank'}) YIELD shortcutsCreated
            RETURN shortcutsCreated
        """)
        self.env.assertEqual(result.result_set, [[2]])

        # the reported count matches the actual number of SHORTCUT edges.
        # the first shortcut always combines two original unit-weight edges
        # (weight 2); the second's exact weight depends on which of the two
        # remaining nodes the heap happens to contract next (the "3-cycle"
        # left behind is no longer symmetric -- one of its edges is now the
        # first shortcut, weight 2 -- so it isn't order-invariant the way
        # the original 4-cycle was), but every combined weight is still at
        # least 2 (never cheaper than a single original edge).
        shortcuts = self.graph.query("""
            MATCH ()-[r:SHORTCUT]->() RETURN r.cost ORDER BY r.cost
        """).result_set
        self.env.assertEqual(len(shortcuts), 2)
        self.env.assertEqual(shortcuts[0], [2])
        self.env.assertTrue(shortcuts[1][0] >= 2)

        # every node was contracted (a 4-node graph fully drains the heap)
        # and got a distinct rank in 1..4
        ranks = self.graph.query("""
            MATCH (n) RETURN n.rank ORDER BY n.rank
        """).result_set
        self.env.assertEqual(ranks, [[1], [2], [3], [4]])

    def test_contraction_hierarchies_missing_weight_on_edge(self):
        """A specific edge missing 'weightProp' defaults to weight 1,
        matching Dijkstra_ShortestPath/AStar_ShortestPath's existing
        per-edge fallback convention -- the procedure shouldn't error just
        because one edge lacks the configured attribute."""

        self.graph.query("""
            CREATE
            (a {id: 1}),
            (b {id: 2}),
            (c {id: 3}),
            (d {id: 4}),
            (a)-[:ROAD {cost: 1}]->(b),
            (b)-[:ROAD]->(c),
            (c)-[:ROAD {cost: 1}]->(d),
            (d)-[:ROAD {cost: 1}]->(a)
        """)

        result = self.graph.query("""
            CALL algo.contractionHierarchies({relTypes: ['ROAD'],
                weightProp: 'cost', shortcutRelType: 'SHORTCUT',
                rankProperty: 'rank'}) YIELD shortcutsCreated
            RETURN shortcutsCreated
        """)
        self.env.assertEqual(result.result_set, [[2]])
