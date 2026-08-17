// Q9. Recent messages by friends or friends of friends
/*
:param [{ personId, maxDate }] => { RETURN
  4398046511268 AS personId,
  1289908800000 AS maxDate
}
*/
// FalkorDB rewrite: this one works around a *bug*, not a dialect gap. A WHERE
// predicate on a node matched from an UNWIND-bound anchor silently returns
// zero rows — upstream's `collect(distinct friend)` + `UNWIND` + `WHERE
// message.creationDate < $maxDate` yields nothing at all. The C engine returns
// the correct result. `WITH DISTINCT friend` expresses the same de-duplication
// without the UNWIND.
// Tracked as FalkorDB/FalkorDB#2557 — revert this when that is fixed.
MATCH (root:Person {id: $personId })-[:KNOWS*1..2]-(friend:Person)
WHERE NOT friend = root
WITH DISTINCT friend
    MATCH (friend)<-[:HAS_CREATOR]-(message:Message)
    WHERE message.creationDate < $maxDate
RETURN
    friend.id AS personId,
    friend.firstName AS personFirstName,
    friend.lastName AS personLastName,
    message.id AS commentOrPostId,
    coalesce(message.content,message.imageFile) AS commentOrPostContent,
    message.creationDate AS commentOrPostCreationDate
ORDER BY
    commentOrPostCreationDate DESC,
    message.id ASC
LIMIT 20
