import time

def _wait_on_index(graph, label, t):
    q = f"""CALL db.indexes() YIELD label, status
    WHERE label = '{label}' AND status <> 'OPERATIONAL'
    RETURN count(1)"""

    while True:
        result = graph.query(q, read_only=True)
        if result.result_set[0][0] == 0:
            break

def _create_index(graph, q, label=None, t=None, sync=False):
    res = graph.query(q)

    if sync:
        _wait_on_index(graph, label, t)

    return res

def list_indicies(graph, label=None):
    q = "CALL db.indexes()"
    q += " YIELD label, properties, language, stopwords, entitytype, info, status"
    
    if label is not None:
        q += f" WHERE label = '{label}'"

    q += " RETURN label, properties, language, stopwords, entitytype, info, status"

    return graph.query(q, read_only=True)

def create_node_exact_match_index(graph, label, *properties, sync=False):
    q = f"CREATE INDEX for (n:{label}) on (" + ','.join(map('n.{0}'.format, properties)) + ")"
    return _create_index(graph, q, label, "exact-match", sync)

def create_edge_exact_match_index(graph, relation, *properties, sync=False):
    q = f"CREATE INDEX for ()-[r:{relation}]->() on (" + ','.join(map('r.{0}'.format, properties)) +")"
    return _create_index(graph, q, relation, "exact-match", sync)

def create_fulltext_index(graph, label, *properties, sync=False):
    q = f"CALL db.idx.fulltext.createNodeIndex('{label}', "
    q += ','.join(map("'{0}'".format, properties))
    q += ")"
    return _create_index(graph, q, label, "full-text", sync)

def create_vector_index(graph, entity_type, label, attribute, dim, similarity_function="euclidean", sync=False):
    q = f"""CALL db.idx.vector.createIndex({{
                type:'{entity_type}',
                label:'{label}',
                attribute:'{attribute}',
                dim:{dim},
                similarityFunction:'{similarity_function}'
            }})"""
    return _create_index(graph, q, label, "exact-match", sync)

def create_node_vector_index(graph, label, attribute, dim, similarity_function="euclidean", sync=False):
    return create_vector_index(graph, "NODE", label, attribute, dim, similarity_function, sync)

def create_edge_vector_index(graph, relation, attribute, dim, similarity_function="euclidean", sync=False):
    return create_vector_index(graph, "RELATIONSHIP", relation, attribute, dim, similarity_function, sync)

def drop_exact_match_index(graph, label, attribute):
    q = f"DROP INDEX ON :{label}({attribute})"
    return graph.query(q)

def drop_fulltext_index(graph, label):
    q = f"CALL db.idx.fulltext.drop('{label}')"
    return graph.query(q)

# validate index is being populated
def index_under_construction(graph, label):
    params = {'lbl': label}
    q = "CALL db.indexes() YIELD label, status WHERE label = $lbl RETURN status"
    res = graph.query(q, params, read_only=True)
    return "UNDER CONSTRUCTION" in res.result_set[0][0]

# wait for all graph indices to by operational
def wait_for_indices_to_sync(graph):
    q = "CALL db.indexes() YIELD status WHERE status <> 'OPERATIONAL' RETURN count(1)"
    while True:
        result = graph.query(q, read_only=True)
        if result.result_set[0][0] == 0:
            break
        time.sleep(0.5) # sleep 500ms

def query_vector_index(graph, entity_type, label, attribute, k, q):
    params = {'type': entity_type, 'label': label, 'attribute': attribute, 'k': k, 'query': q}

    return graph.query("""CALL db.idx.vector.query({
            type: $type,
            label: $label,
            attribute: $attribute,
            query: vector32f($query),
            k:$k})""", params=params)

def query_node_vector_index(graph, label, attribute, k, q):
    return query_vector_index(graph, "NODE", label, attribute, k, q)

def query_edge_vector_index(graph, relation, attribute, k, q):
    return query_vector_index(graph, "RELATIONSHIP", relation, attribute, k, q)

