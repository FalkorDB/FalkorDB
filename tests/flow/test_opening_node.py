from common import *
from itertools import permutations, chain, combinations

# this test file checks the planner / optimizer decision regarding
# which node in a search pattern is used as the entry point to a traversal
# e.g.
# (a:A)-[:R]->(b:B)
# we would like the entry point to be the one with the least number of nodes
# associated with it. and so if there are 4 A nodes and 19 B nodes we would like
# the search to begin with A and if the situation was reversed i.e. 19 A nodes
# and 4 B nodes we would like B to become the starting point

GRAPH_ID = "opening_node"

def all_combinations_and_permutations(lst):
    # Generate all subsets except the empty set
    subsets = chain.from_iterable(combinations(lst, r) for r in range(1, len(lst) + 1))

    # Generate all permutations for each subset
    result = set(chain.from_iterable(permutations(subset) for subset in subsets))

    # Sort and convert to list of lists
    return [list(x) for x in sorted(result, key=len)]

class testOpeningNode():
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_ID)

    # in a pattern where both ends have only a single label
    # make sure the label with the least number of nodes associated with it
    # is used as the starting point
    def test01_single_label(self):
        # create the graph
        # number of A nodes: 2
        # number of B nodes: 4

        q = "CREATE (:A), (:A), (:B), (:B), (:B), (:B)"
        self.graph.query(q)

        # expecting A to open the traversal 
        queries = [
            "MATCH (a:A)-[:R]->(b:B) RETURN a, b", # left to right
            "MATCH (b:B)<-[:R]-(a:A) RETURN a, b"  # right to left
        ]

        for q in queries:
            plan = str(self.graph.explain(q))
            self.env.assertIn("Node By Label Scan | (a:A)", plan)
            self.env.assertIn("Conditional Traverse | (a)->(b:B)", plan)

        # increase number of A nodes
        # number of A nodes: 8
        # number of B nodes: 4

        q = "UNWIND range(0, 5) AS x CREATE (:A)"
        self.graph.query(q)

        # expecting B to open the traversal 
        for q in queries:
            plan = str(self.graph.explain(q))
            self.env.assertIn("Node By Label Scan | (b:B)", plan)
            self.env.assertIn("Conditional Traverse | (b)->(a:A)", plan)

    # when there are multiple labels to pick from
    # make sure the label with the least number of nodes associated with it is
    # used as the traversal starting point
    def test02_multi_label(self):
        # start fresh
        self.graph.delete()

        # create a graph with:
        # 1 A node
        # 2 B nodes
        # 3 C nodes
        # 4 X nodes
        # 5 Y nodes
        # 6 Z nodes

        q = """CREATE
            (:A),
            (:B), (:B),
            (:C), (:C), (:C),
            (:X), (:X), (:X), (:X),
            (:Y), (:Y), (:Y), (:Y), (:Y),
            (:Z), (:Z), (:Z), (:Z), (:Z), (:Z)
            """
        self.graph.query(q)

        # form a connection from (:A:B:C) to (:X:Y:Z)
        # used to make sure that indeed the following traversals return
        # expected values
        self.graph.query("CREATE (n:A:B:C)-[:R]->(m:X:Y:Z), (n)<-[:R]-(m)")

        srcs  = ['A', 'B', 'C']
        dests = ['X', 'Y', 'Z']
        lbls  = ['A', 'B', 'C', 'X', 'Y', 'Z']

        # for every possible permutation of the src and dest lables
        # e.g. (:A) / (:X:Y), (:C:A) / (:Z:Y)...
        # make sure the opening node is the one with the least number of nodes
        # associated with it
        src_perms  = all_combinations_and_permutations(srcs)
        dest_perms = all_combinations_and_permutations(dests)

        # query all possible permutations, making sure min_lbl
        # is used as the starting point
        for sp in src_perms:
            for dp in dest_perms:
                # determine min label
                min_lbl = min(sp + dp)

                queries = [
                    f"MATCH ({':' + ':'.join(sp)})-[:R]->({':' + ':'.join(dp)}) RETURN 1", # sp -> dp
                    f"MATCH ({':' + ':'.join(dp)})<-[:R]-({':' + ':'.join(sp)}) RETURN 1", # reverse pattern dp <- sp
                    f"MATCH ({':' + ':'.join(dp)})-[:R]->({':' + ':'.join(sp)}) RETURN 1", # dp -> sp
                    f"MATCH ({':' + ':'.join(sp)})<-[:R]-({':' + ':'.join(dp)}) RETURN 1"  # reverse pattern sp <- dp
                ]

                for q in queries:
                    plan = self.graph.explain(q)
                    label_scan = plan.collect_operations('Node By Label Scan')[0]
                    self.env.assertIn(min_lbl, str(label_scan))

                    res = self.graph.query(q).result_set
                    self.env.assertEqual(len(res), 1)
                    self.env.assertEqual(res[0][0], 1)

