import os
import sys
from RLTest import Env

from falkordb import FalkorDB, Graph, Node, Edge

r = None
graph_name = "G"
graph = None


def redis():
    return Env.RTestInstance.currEnv


def _brand_new_redis():
    global r
    if r is not None:
        r.flush()

    r = redis()
    db = FalkorDB("localhost", r.port)
    db = FalkorDB("localhost", 6379)
    return db

def empty_graph():
    global graph
    db = _brand_new_redis()
    graph = db.select_graph("G")

    # Create a graph with a single node.
    graph.query("CREATE ()")

    # Delete node to have an empty graph.
    graph.query("MATCH (n) DELETE n")

def any_graph():
    empty_graph()


def binary_tree_graph1():
    global graph

    db = _brand_new_redis()
    graph = db.select_graph("G1")
    graph.query("""CREATE(a: A {name: 'a'}),
                       (b1: X {name: 'b1'}),
                       (b2: X {name: 'b2'}),
                       (b3: X {name: 'b3'}),
                       (b4: X {name: 'b4'}),
                       (c11: X {name: 'c11'}),
                       (c12: X {name: 'c12'}),
                       (c21: X {name: 'c21'}),
                       (c22: X {name: 'c22'}),
                       (c31: X {name: 'c31'}),
                       (c32: X {name: 'c32'}),
                       (c41: X {name: 'c41'}),
                       (c42: X {name: 'c42'})
                       CREATE (a)-[:KNOWS] -> (b1),
                       (a)-[:KNOWS] -> (b2),
                       (a)-[:FOLLOWS] -> (b3),
                       (a)-[:FOLLOWS] -> (b4)
                       CREATE (b1)-[:FRIEND] -> (c11),
                       (b1)-[:FRIEND] -> (c12),
                       (b2)-[:FRIEND] -> (c21),
                       (b2)-[:FRIEND] -> (c22),
                       (b3)-[:FRIEND] -> (c31),
                       (b3)-[:FRIEND] -> (c32),
                       (b4)-[:FRIEND] -> (c41),
                       (b4)-[:FRIEND] -> (c42)
                       CREATE (b1)-[:FRIEND] -> (b2),
                       (b2)-[:FRIEND] -> (b3),
                       (b3)-[:FRIEND] -> (b4),
                       (b4)-[:FRIEND] -> (b1)""")


def binary_tree_graph2():
    global graph

    db = _brand_new_redis()
    graph = db.select_graph("G2")
    graph.query("""CREATE(a: A {name: 'a'}),
                      (b1: X {name: 'b1'}),
                      (b2: X {name: 'b2'}),
                      (b3: X {name: 'b3'}),
                      (b4: X {name: 'b4'}),
                      (c11: X {name: 'c11'}),
                      (c12: Y {name: 'c12'}),
                      (c21: X {name: 'c21'}),
                      (c22: Y {name: 'c22'}),
                      (c31: X {name: 'c31'}),
                      (c32: Y {name: 'c32'}),
                      (c41: X {name: 'c41'}),
                      (c42: Y {name: 'c42'})
                      CREATE (a)-[:KNOWS] -> (b1),
                      (a)-[:KNOWS] -> (b2),
                      (a)-[:FOLLOWS] -> (b3),
                      (a)-[:FOLLOWS] -> (b4)
                      CREATE (b1)-[:FRIEND] -> (c11),
                      (b1)-[:FRIEND] -> (c12),
                      (b2)-[:FRIEND] -> (c21),
                      (b2)-[:FRIEND] -> (c22),
                      (b3)-[:FRIEND] -> (c31),
                      (b3)-[:FRIEND] -> (c32),
                      (b4)-[:FRIEND] -> (c41),
                      (b4)-[:FRIEND] -> (c42)
                      CREATE (b1)-[:FRIEND] -> (b2),
                      (b2)-[:FRIEND] -> (b3),
                      (b3)-[:FRIEND] -> (b4),
                      (b4)-[:FRIEND] -> (b1)""")


def query(q):
    return graph.query(q)

