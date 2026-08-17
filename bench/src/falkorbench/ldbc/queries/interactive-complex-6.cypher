// Q6. Tag co-occurrence
/*
:param [{ personId, tagName }] => { RETURN
  4398046511333 AS personId,
  "Carl_Gustaf_Emil_Mannerheim" AS tagName
}
*/
// FalkorDB rewrite: this one works around a *bug*, not a dialect gap. With `f`
// bound by UNWIND of a collected list, an inline property filter on a pattern
// that is followed by further patterns makes the match fail with "Type
// mismatch: expected Map, Node, Edge, ... but was List". The C engine runs it
// correctly. Moving the id filter from the inline map into WHERE avoids it
// without changing meaning.
// Tracked as FalkorDB/FalkorDB#2556 — revert this when that is fixed.
MATCH (knownTag:Tag { name: $tagName })
WITH knownTag.id as knownTagId

MATCH (person:Person { id: $personId })-[:KNOWS*1..2]-(friend)
WHERE NOT person=friend
WITH
    knownTagId,
    collect(distinct friend) as friends
UNWIND friends as f
    MATCH (f)<-[:HAS_CREATOR]-(post:Post),
          (post)-[:HAS_TAG]->(t:Tag),
          (post)-[:HAS_TAG]->(tag:Tag)
    WHERE t.id = knownTagId AND NOT t = tag
    WITH
        tag.name as tagName,
        count(post) as postCount
RETURN
    tagName,
    postCount
ORDER BY
    postCount DESC,
    tagName ASC
LIMIT 10
