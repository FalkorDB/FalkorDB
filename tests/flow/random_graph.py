from index_utils import *
from random import randint, random, choice
from xmlrpc.client import MAXINT, MININT
import string


def random_string():
    size = randint(0, 100)
    return ''.join(choice(string.ascii_uppercase + string.ascii_lowercase + string.whitespace + string.digits) for _ in range(size))


def random_int():
    return randint(MININT, MAXINT)


def random_bool():
    return randint(0, 1) == 0


def random_double():
    return random()


ALL_VALUES_EXCEPT_ARRAY = [random_string, random_int,
                           random_bool, random_double]


def random_array():
    l = randint(0, 10)
    arr = []
    for i in range(0, l):
        arr.append(random_graph_value(ALL_VALUES_EXCEPT_ARRAY))
    return arr

def random_point():
    raise "not implemented"

def random_vector():
    raise "not implemented"

ALL_VALUES = [random_string, random_int,
              random_bool, random_double, random_array]


def random_graph_value(values=ALL_VALUES):
    type = randint(0, len(values) - 1)
    return ALL_VALUES[type]()


def create_node_map(node, i):
    item = {"v": i}
    for j in range(0, node["properties"]):
        item[f"v{i}"] = random_graph_value()
    return item


def create_edge_map(edge, nodes):
    source = edge["source"]
    source_count = nodes[source]["count"]
    target = edge["target"]
    target_count = nodes[target]["count"]
    item = {"from": randint(1, source_count),
            "to": randint(1, target_count)}
    return item


def create_nodes_range(node, map):
    node_count = node["count"]
    data = []
    for i in range(0, node_count):
        item = create_node_map(node, i)
        data.append(item)

    return data, f"UNWIND $nodes AS {map}"


def create_node_pattern(node, map):
    labels = ":".join(node["labels"])
    return f"CREATE (n:{labels}) SET n = {map}"


def create_edge_range(edge, nodes, map):
    edges_count = edge["count"]
    data = []
    for i in range(0, edges_count):
        item = create_edge_map(edge, nodes)
        data.append(item)
    return data, f"UNWIND $edges AS {map}"


def create_edge_pattern(edge, nodes, map):
    type = edge["type"]
    source = edge["source"]
    source_label = nodes[source]["labels"][0]
    target = edge["target"]
    target_label = nodes[target]["labels"][0]
    return f"MATCH (n:{source_label} {{v: {map}.from}}), (m:{target_label} {{v: {map}.to}}) CREATE (n)-[:{type}]->(m)"


def create_random_graph(g, nodes, edges):
    """
    Create random graph
    Example:
    nodes, edges = create_random_schema()
    res = create_random_graph(graph, nodes, edges)
    """
    result = {}
    result["indexes"] = []
    result["nodes"] = []
    result["edges"] = []

    for node in nodes:
        # print(node["labels"][0])
        # print(node["count"])

        index_label = node["labels"][0]
        res = create_node_range_index(g, index_label, 'v', sync=True)
        result["indexes"].append(res)

        data, range_pattern = create_nodes_range(node, "map")
        node_pattern = create_node_pattern(node, "map")
        res = g.query(f"{range_pattern} {node_pattern}", {"nodes": data})
        result["nodes"].append(res)

    for edge in edges:
        # print(edge["type"])
        # print(edge["count"])

        data, range_pattern = create_edge_range(edge, nodes, "map")
        edge_pattern = create_edge_pattern(edge, nodes, "map")
        res = g.query(
            f"{range_pattern} {edge_pattern}", {"edges": data})
        result["edges"].append(res)

    return result


def create_node(nodes, edges):
    labels = len(nodes)
    node = nodes[randint(0, labels - 1)]
    query = create_node_pattern(node, "$map")
    params = {"map": create_node_map(node, randint(100, 1000))}
    return params, query


def create_edge(nodes, edges):
    rel_types = len(edges)
    edge = edges[randint(0, rel_types - 1)]
    query = create_edge_pattern(edge, nodes, "$map")
    params = {"map": create_edge_map(edge, nodes)}
    # TODO: create (n)-[new]->(new), (new)-[new]->(new)
    return params, query


def delete_node(nodes, edges):
    labels = len(nodes)
    node = nodes[randint(0, labels - 1)]
    label = node["labels"][0]
    label_count = node["count"]
    query = f"MATCH (n:{label} {{v: ToInteger(rand()*{label_count})}}) DELETE n"
    params = {}
    return params, query


def delete_edge(nodes, edges):
    rel_types = len(edges)
    edge = edges[randint(0, rel_types - 1)]
    type = edge["type"]
    source = edge["source"]
    source_label = nodes[source]["labels"][0]
    source_count = nodes[source]["count"]
    target = edge["target"]
    target_label = nodes[target]["labels"][0]
    target_count = nodes[target]["count"]
    query = f"MATCH (n:{source_label} {{v: ToInteger(rand()*{source_count})}})-[r:{type}]->(m:{target_label} {{v: ToInteger(rand()*{target_count})}}) DELETE r"
    params = {}
    return params, query


def update_node(nodes, edges):
    labels = len(nodes)
    node = nodes[randint(0, labels - 1)]
    label = node["labels"][0]
    label_count = node["count"]
    query = f"MATCH (n:{label} {{v: ToInteger(rand()*{label_count})}}) SET n.v = rand()"
    params = {}
    return params, query


def update_edge(nodes, edges):
    rel_types = len(edges)
    edge = edges[randint(0, rel_types - 1)]
    type = edge["type"]
    source = edge["source"]
    source_label = nodes[source]["labels"][0]
    source_count = nodes[source]["count"]
    target = edge["target"]
    target_label = nodes[target]["labels"][0]
    target_count = nodes[target]["count"]
    query = f"MATCH (n:{source_label} {{v: ToInteger(rand()*{source_count})}})-[r:{type}]->(m:{target_label} {{v: ToInteger(rand()*{target_count})}}) SET r.v = rand()"
    params = {}
    return params, query


def run_random_graph_ops(g, nodes, edges, ops):
    """
    Run random graph write operations
    Example:
    nodes, edges = create_random_schema()
    res = create_random_graph(graph, nodes, edges)
    res = run_random_graph_ops(graph, nodes, edges, ALL_OPS)
    """
    result = []
    total = {}
    for i in range(0, randint(10, 1000)):
        op = randint(0, len(ops) - 1)
        params, query = ops[op](nodes, edges)
        # print(query)
        res = g.query(query, params)
        result.append(res)

    return total, result


def create_random_schema():
    """
    Create random graph schema
    Example:
    res = create_random_schema(graph, {"N": 100, "M": 200}, {
                           "R": {"source": "N", "target": "M", "count": 500}})
    """
    nodes = []
    edges = []

    labels_count = randint(1, 10)
    for i in range(0, labels_count):
        count = randint(10, 100)
        properties = randint(0, 10)
        labels = [f"N{i}"]
        for j in range(0, randint(0, 5)):
            l = randint(1, labels_count)
            labels.append(f"N{l}")

        nodes.append(
            {"count": count, "properties": properties, "labels": labels})

    rel_types = randint(1, labels_count * 2)
    for i in range(0, rel_types):
        source_label = randint(0, labels_count - 1)
        source_count = nodes[source_label]["count"]
        target_label = randint(0, labels_count - 1)
        target_count = nodes[target_label]["count"]
        edge_count = randint(10, source_count * target_count)
        edges.append({"type": f"R{i}", "source": source_label,
                     "target": target_label, "count": edge_count})

    return nodes, edges


ALL_OPS = [create_node, create_edge, delete_node,
           delete_edge, update_node, update_edge]

