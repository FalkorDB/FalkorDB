// Q7. Recent likers
/*
:param personId: 4398046511268
*/
// FalkorDB rewrite: a traversal pattern is not a boolean-valued expression in a
// projection — `not((liker)-[:KNOWS]-(person))` is rejected with "Type
// mismatch: expected Boolean or Null but was List" (on the C engine too, so
// this is a dialect gap rather than a regression), and exists() explicitly
// refuses traversal patterns. `size(pattern) = 0` is the equivalent test.
MATCH (person:Person {id: $personId})<-[:HAS_CREATOR]-(message:Message)<-[like:LIKES]-(liker:Person)
    WITH liker, message, like.creationDate AS likeTime, person
    ORDER BY likeTime DESC, toInteger(message.id) ASC
    WITH liker, head(collect({msg: message, likeTime: likeTime})) AS latestLike, person
RETURN
    liker.id AS personId,
    liker.firstName AS personFirstName,
    liker.lastName AS personLastName,
    latestLike.likeTime AS likeCreationDate,
    latestLike.msg.id AS commentOrPostId,
    coalesce(latestLike.msg.content, latestLike.msg.imageFile) AS commentOrPostContent,
    toInteger(floor(toFloat(latestLike.likeTime - latestLike.msg.creationDate)/1000.0)/60.0) AS minutesLatency,
    size((liker)-[:KNOWS]-(person)) = 0 AS isNew
ORDER BY
    likeCreationDate DESC,
    toInteger(personId) ASC
LIMIT 20
