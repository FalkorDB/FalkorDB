import time

# wait for index to be operational
def _wait_on_index(graph, label):
    q = f"""CALL db.indexes() YIELD label, status
    WHERE label = '{label}' AND status <> 'OPERATIONAL'
    RETURN count(1)"""

    while True:
        result = graph.ro_query(q)
        if result.result_set[0][0] == 0:
            break

def _create_index(graph, q, label=None, sync=False):
    res = graph.query(q)

    if sync:
        _wait_on_index(graph, label)

    return res

def list_indicies(graph, label=None):
    q = "CALL db.indexes()"
    q += " YIELD label, properties, types, language, stopwords, entitytype, info, status"

    if label is not None:
        q += f" WHERE label = '{label}'"

    q += " RETURN label, properties, types, language, stopwords, entitytype, info, status"

    return graph.ro_query(q)

def _create_typed_index(graph, idx_type, entity_type, label, *properties, options=None, sync=False):
    if entity_type == "NODE":
        pattern = f"(e:{label})"
    elif entity_type == "EDGE":
        pattern = f"()-[e:{label}]->()"
    else:
        raise ValueError("Invalid entity type")

    if idx_type == "RANGE":
        idx_type = ""

    q = f"CREATE {idx_type} INDEX FOR {pattern} ON ("
    q += ",".join(map("e.{0}".format, properties))
    q += ")"

    if options is not None:
        # convert options to a Cypher map
        options_map = "{"
        for key, value in options.items():
            if type(value) == str:
                options_map += key + ":'" + value + "',"
            else:
                options_map += key + ':' + str(value) + ','
        options_map = options_map[:-1] + "}"
        q += f" OPTIONS {options_map}"

    return _create_index(graph, q, label, sync)

def create_node_range_index(graph, label, *properties, sync=False):
    return _create_typed_index(graph, "RANGE", "NODE", label, *properties, sync=sync)

def create_node_fulltext_index(graph, label, *properties, sync=False):
    return _create_typed_index(graph, "FULLTEXT", "NODE", label, *properties, sync=sync)

def create_node_vector_index(graph, label, *properties, dim=0, similarity_function="euclidean", m=16, efConstruction=200, efRuntime=10, sync=False):
    options = {'dimension': dim, 'similarityFunction': similarity_function, 'M': m, 'efConstruction': efConstruction, 'efRuntime': efRuntime}
    return _create_typed_index(graph, "VECTOR", "NODE", label, *properties, options=options, sync=sync)

def create_edge_range_index(graph, relation, *properties, sync=False):
    return _create_typed_index(graph, "RANGE", "EDGE", relation, *properties, sync=sync)

def create_edge_fulltext_index(graph, relation, *properties, sync=False):
    return _create_typed_index(graph, "FULLTEXT", "EDGE", relation, *properties, sync=sync)

def create_edge_vector_index(graph, relation, *properties, dim, similarity_function="euclidean", m=16, efConstruction=200, efRuntime=10, sync=False):
    options = {'dimension': dim, 'similarityFunction': similarity_function, 'M': m, 'efConstruction': efConstruction, 'efRuntime': efRuntime}
    return _create_typed_index(graph, "VECTOR", "EDGE", relation, *properties, options=options, sync=sync)

def _drop_index(graph, idx_type, entity_type, label, attribute=None):
    # set pattern
    if entity_type == "NODE":
        pattern = f"(e:{label})"
    elif entity_type == "EDGE":
        pattern = f"()-[e:{label}]->()"
    else:
        raise ValueError("Invalid entity type")

    # build drop index command
    if idx_type == "RANGE":
        q = f"DROP INDEX FOR {pattern} ON (e.{attribute})"
    elif idx_type == "VECTOR":
        q = f"DROP VECTOR INDEX FOR {pattern} ON (e.{attribute})"
    elif idx_type == "FULLTEXT":
        q = f"DROP FULLTEXT INDEX FOR {pattern} ON (e.{attribute})"
    else:
        raise ValueError("Invalid index type")

    return graph.query(q)

def drop_node_range_index(graph, label, attribute):
    return _drop_index(graph, "RANGE", "NODE", label, attribute)

def drop_node_fulltext_index(graph, label, attribute):
    return _drop_index(graph, "FULLTEXT", "NODE", label, attribute)

def drop_node_vector_index(graph, label, attribute):
    return _drop_index(graph, "VECTOR", "NODE", label, attribute)

def drop_edge_range_index(graph, label, attribute):
    return _drop_index(graph, "RANGE", "EDGE", label, attribute)

def drop_edge_fulltext_index(graph, label, attribute):
    return _drop_index(graph, "FULLTEXT", "EDGE", label, attribute)

def drop_edge_vector_index(graph, label, attribute):
    return _drop_index(graph, "VECTOR", "EDGE", label, attribute)

# validate index is being populated
def index_under_construction(graph, label):
    params = {'lbl': label}
    q = "CALL db.indexes() YIELD label, status WHERE label = $lbl RETURN status"
    res = graph.ro_query(q, params)
    return "UNDER CONSTRUCTION" in res.result_set[0][0]

# wait for all graph indices to by operational
def wait_for_indices_to_sync(graph):
    q = "CALL db.indexes() YIELD status WHERE status <> 'OPERATIONAL' RETURN count(1)"
    while True:
        result = graph.ro_query(q)
        if result.result_set[0][0] == 0:
            break
        time.sleep(0.5) # sleep 500ms

def query_node_vector_index(graph, label, attribute, k, q):
    params = {'lbl': label, 'attr': attribute, 'k': k, 'q': q}
    return graph.query("CALL db.idx.vector.queryNodes($lbl, $attr, $k, vecf32($q))", params=params)

def query_edge_vector_index(graph, relation, attribute, k, q):
    params = {'lbl': relation, 'attr': attribute, 'k': k, 'q': q}
    return graph.query("CALL db.idx.vector.queryRelationships($lbl, $attr, $k, vecf32($q))", params=params)

