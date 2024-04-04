# returns True if graphs have the same:
# set of labels
# set of relations
# set of properties
# node
# edges
# indices
# constrains

def graph_eq(A, B):

    queries = [
            # labels
            "CALL db.labels() YIELD label RETURN label ORDER BY label",
            
            # relationships
            """CALL db.relationshiptypes() YIELD relationshipType
               RETURN relationshipType ORDER BY relationshipType""",
            
            # properties
            """CALL db.propertyKeys() YIELD propertyKey
               RETURN propertyKey ORDER BY propertyKey""",

            # nodes
            "MATCH (n) RETURN n ORDER BY(n)",

            # validate relationships
            "MATCH ()-[e]->() RETURN e ORDER BY(e)",

            # indices
            """CALL db.indexes()
               YIELD label, properties, types, language, stopwords, entitytype
               RETURN label, properties, types, language, stopwords, entitytype
               ORDER BY label, properties, types, language, stopwords, entitytype""",

            # constraints
            """CALL db.constraints()
               YIELD type, label, properties, entitytype, status
               RETURN type, label, properties, entitytype, status
               ORDER BY type, label, properties, entitytype, status"""
            ]

    for q in queries:
        A_labels = A.ro_query(q).result_set
        B_labels = B.ro_query(q).result_set
        if A_labels != B_labels:
            return False

    return True

