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
// FalkorDB rewrite (2): a pattern comprehension's WHERE cannot reference the
// variable of an enclosing list comprehension — `startNode(r)` there resolves
// to Null ("Type mismatch: expected Edge but was Null"; the C engine reports
// "Unable to resolve filtered alias 'r'", so this is a dialect gap, not a
// regression). The relationships are therefore UNWINDed and their endpoints
// hoisted into plain variables before the comprehensions run, then summed back
// per path. `reduce(w=0.0, v IN [... | 1.0] | w+v)` is a count times its
// weight, so size(...) * weight is equivalent.
MATCH path = allShortestPaths((person1)-[:KNOWS*0..]-(person2))
WITH path, [n IN nodes(path) | n.id] AS personIdsInPath
UNWIND relationships(path) AS r
WITH path, personIdsInPath, startNode(r).id AS aId, endNode(r).id AS bId
WITH path, personIdsInPath,
    size([(a:Person)<-[:HAS_CREATOR]-(:Comment)-[:REPLY_OF]->(:Post)-[:HAS_CREATOR]->(b:Person)
          WHERE (a.id = aId AND b.id = bId) OR (a.id = bId AND b.id = aId) | 1.0]) * 1.0 AS w1,
    size([(a:Person)<-[:HAS_CREATOR]-(:Comment)-[:REPLY_OF]->(:Comment)-[:HAS_CREATOR]->(b:Person)
          WHERE (a.id = aId AND b.id = bId) OR (a.id = bId AND b.id = aId) | 0.5]) * 0.5 AS w2
WITH path, personIdsInPath, sum(w1 + w2) AS pathWeight
RETURN
    personIdsInPath,
    pathWeight
ORDER BY pathWeight desc
