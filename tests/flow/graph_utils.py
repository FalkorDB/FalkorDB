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
            ('labels', "CALL db.labels() YIELD label RETURN label ORDER BY label"),
            
            # relationships-types
            ('relationships-types', """CALL db.relationshiptypes() YIELD relationshipType
               RETURN relationshipType ORDER BY relationshipType"""),
            
            # properties
            ('properties', """CALL db.propertyKeys() YIELD propertyKey
               RETURN propertyKey ORDER BY propertyKey"""),

            # node count stats
            ('node count stats', "MATCH (n) return count(n)"),

            # node count
            ('node count', "MATCH (n) WHERE ID(n) = ID(n) return count(n)"),

            # nodes
            ('nodes', "MATCH (n) RETURN n ORDER BY(n)"),

            # relation count stats
            ('relation count stats', "MATCH ()-[e]->() return count(e)"),

            # relation count
            ('relation count', "MATCH ()-[e]->() WHERE ID(e) = ID(e) return count(e)"),

            # validate relations
            ('relations', "MATCH ()-[e]->() RETURN e ORDER BY(e)"),

            # indices
            ('indices', """CALL db.indexes()
               YIELD label, properties, types, language, stopwords, entitytype
               RETURN label, properties, types, language, stopwords, entitytype
               ORDER BY label, properties, types, language, stopwords, entitytype"""),

            # constraints
            ('constraints', """CALL db.constraints()
               YIELD type, label, properties, entitytype, status
               RETURN type, label, properties, entitytype, status
               ORDER BY type, label, properties, entitytype, status""")
            ]

    for category, q in queries:
        A_res = A.ro_query(q).result_set
        B_res = B.ro_query(q).result_set
        if A_res != B_res:
            print(f"diff in {category}")
            for i in range(0, len(A_res)):
                if A_res[i] != B_res[i]:
                    print(f"A_res[{i}]: {A_res[i]}, B_res[{i}]: {B_res[i]}")

            return False

    return True

