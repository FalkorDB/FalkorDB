from RLTest import Env

from neo4j import GraphDatabase
# from neo4j.debug import watch

# watch("neo4j", out=open("neo4j.log", "w"))

r = None
bolt_con = None
driver = None

def redis():
    return Env.RTestInstance.currEnv


def _brand_new_redis():
    global r
    global driver
    if r is not None:
        r.flush()

    r = redis()

    if driver is None:
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("falkordb", ""))

    return driver


def empty_graph():
    global bolt_con

    bolt_con = _brand_new_redis()
    with bolt_con.session() as session:
        # Create a graph with a single node.
        session.run("CREATE ()").consume()
        # Delete node to have an empty graph.
        session.run("MATCH (n) DETACH DELETE n").consume()


def any_graph():
    return empty_graph()


def binary_tree_graph1():
    global bolt_con

    bolt_con = _brand_new_redis()
    with bolt_con.session() as session:
        session.run("CREATE(a: A {name: 'a'}),        \
                        (b1: X {name: 'b1'}),         \
                        (b2: X {name: 'b2'}),         \
                        (b3: X {name: 'b3'}),         \
                        (b4: X {name: 'b4'}),         \
                        (c11: X {name: 'c11'}),       \
                        (c12: X {name: 'c12'}),       \
                        (c21: X {name: 'c21'}),       \
                        (c22: X {name: 'c22'}),       \
                        (c31: X {name: 'c31'}),       \
                        (c32: X {name: 'c32'}),       \
                        (c41: X {name: 'c41'}),       \
                        (c42: X {name: 'c42'})        \
                        CREATE(a)-[:KNOWS] -> (b1),   \
                        (a)-[:KNOWS] -> (b2),         \
                        (a)-[:FOLLOWS] -> (b3),       \
                        (a)-[:FOLLOWS] -> (b4)        \
                        CREATE(b1)-[:FRIEND] -> (c11),\
                        (b1)-[:FRIEND] -> (c12),      \
                        (b2)-[:FRIEND] -> (c21),      \
                        (b2)-[:FRIEND] -> (c22),      \
                        (b3)-[:FRIEND] -> (c31),      \
                        (b3)-[:FRIEND] -> (c32),      \
                        (b4)-[:FRIEND] -> (c41),      \
                        (b4)-[:FRIEND] -> (c42)       \
                        CREATE(b1)-[:FRIEND] -> (b2), \
                        (b2)-[:FRIEND] -> (b3),       \
                        (b3)-[:FRIEND] -> (b4),       \
                        (b4)-[:FRIEND] -> (b1)        \
                        ")


def binary_tree_graph2():
    global bolt_con

    bolt_con = _brand_new_redis()
    with bolt_con.session() as session:
        session.run("CREATE(a: A {name: 'a'}),        \
                        (b1: X {name: 'b1'}),         \
                        (b2: X {name: 'b2'}),         \
                        (b3: X {name: 'b3'}),         \
                        (b4: X {name: 'b4'}),         \
                        (c11: X {name: 'c11'}),       \
                        (c12: Y {name: 'c12'}),       \
                        (c21: X {name: 'c21'}),       \
                        (c22: Y {name: 'c22'}),       \
                        (c31: X {name: 'c31'}),       \
                        (c32: Y {name: 'c32'}),       \
                        (c41: X {name: 'c41'}),       \
                        (c42: Y {name: 'c42'})        \
                        CREATE(a)-[:KNOWS] -> (b1),   \
                        (a)-[:KNOWS] -> (b2),         \
                        (a)-[:FOLLOWS] -> (b3),       \
                        (a)-[:FOLLOWS] -> (b4)        \
                        CREATE(b1)-[:FRIEND] -> (c11),\
                        (b1)-[:FRIEND] -> (c12),      \
                        (b2)-[:FRIEND] -> (c21),      \
                        (b2)-[:FRIEND] -> (c22),      \
                        (b3)-[:FRIEND] -> (c31),      \
                        (b3)-[:FRIEND] -> (c32),      \
                        (b4)-[:FRIEND] -> (c41),      \
                        (b4)-[:FRIEND] -> (c42)       \
                        CREATE(b1)-[:FRIEND] -> (b2), \
                        (b2)-[:FRIEND] -> (b3),       \
                        (b3)-[:FRIEND] -> (b4),       \
                        (b4)-[:FRIEND] -> (b1)        \
                        ")

class BoltResult:
    def __init__(self, result_set, summary):
        self.result_set = result_set
        self.summary = summary

def query(q):
    with bolt_con.session() as session:
        res = session.run(q)
        result_set = list(res)
        summary = res.consume()
        return BoltResult(result_set, summary)
