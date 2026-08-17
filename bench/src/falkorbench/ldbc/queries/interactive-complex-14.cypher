// Q14. Trusted connection paths
/*
:param [{ person1Id, person2Id }] => { RETURN
  8796093022357 AS person1Id,
  8796093022390 AS person2Id
}
*/
// FalkorDB rewrite: allShortestPaths() requires both endpoints to be already
// bound ("Source and destination must already be resolved to call
// allShortestPaths"), so the two :Person lookups move into a preceding MATCH.
MATCH (person1:Person { id: $person1Id }), (person2:Person { id: $person2Id })
WITH person1, person2
MATCH path = allShortestPaths((person1)-[:KNOWS*0..]-(person2))
WITH collect(path) as paths
UNWIND paths as path
WITH path, relationships(path) as rels_in_path
WITH
    [n in nodes(path) | n.id ] as personIdsInPath,
    [r in rels_in_path |
        reduce(w=0.0, v in [
            (a:Person)<-[:HAS_CREATOR]-(:Comment)-[:REPLY_OF]->(:Post)-[:HAS_CREATOR]->(b:Person)
            WHERE
                (a.id = startNode(r).id and b.id=endNode(r).id) OR (a.id=endNode(r).id and b.id=startNode(r).id)
            | 1.0] | w+v)
    ] as weight1,
    [r in rels_in_path |
        reduce(w=0.0,v in [
        (a:Person)<-[:HAS_CREATOR]-(:Comment)-[:REPLY_OF]->(:Comment)-[:HAS_CREATOR]->(b:Person)
        WHERE
                (a.id = startNode(r).id and b.id=endNode(r).id) OR (a.id=endNode(r).id and b.id=startNode(r).id)
        | 0.5] | w+v)
    ] as weight2
WITH
    personIdsInPath,
    reduce(w=0.0,v in weight1| w+v) as w1,
    reduce(w=0.0,v in weight2| w+v) as w2
RETURN
    personIdsInPath,
    (w1+w2) as pathWeight
ORDER BY pathWeight desc
