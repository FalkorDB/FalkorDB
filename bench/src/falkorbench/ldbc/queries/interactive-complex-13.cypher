// Q13. Single shortest path
/*
:param [{ person1Id, person2Id }] => { RETURN
  8796093022390 AS person1Id,
  8796093022357 AS person2Id
}
*/
// FalkorDB rewrite: shortestPath() is rejected inside MATCH ("FalkorDB
// currently only supports shortestPaths in WITH or RETURN clauses"), so the
// path binding moves into its own WITH. The CASE below already handles the
// null (disconnected) case, so the -1 result is unchanged.
MATCH
    (person1:Person {id: $person1Id}),
    (person2:Person {id: $person2Id})
WITH shortestPath((person1)-[:KNOWS*]-(person2)) AS path
RETURN
    CASE path IS NULL
        WHEN true THEN -1
        ELSE length(path)
    END AS shortestPathLength
