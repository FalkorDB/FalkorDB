// Q10. Friend recommendation
/*
:param [{ personId, month }] => { RETURN
  4398046511333 AS personId,
  5 AS month
}
*/
MATCH (person:Person {id: $personId})-[:KNOWS*2..2]-(friend),
       (friend)-[:IS_LOCATED_IN]->(city:City)
WHERE NOT friend=person AND
      NOT (friend)-[:KNOWS]-(person)
// FalkorDB rewrite: FalkorDB has no `datetime()` / temporal type, so
// `datetime({epochMillis: friend.birthday}).month` and `.day` are unavailable.
// The loader stores the two components the query actually needs as derived
// integer properties (birthdayMonth, birthdayDay) alongside the raw epoch
// birthday; see falkorbench.ldbc.schema. The predicate is otherwise identical.
WITH person, city, friend
WHERE  (friend.birthdayMonth=$month AND friend.birthdayDay>=21) OR
        (friend.birthdayMonth=($month%12)+1 AND friend.birthdayDay<22)
WITH DISTINCT friend, city, person
OPTIONAL MATCH (friend)<-[:HAS_CREATOR]-(post:Post)
WITH friend, city, collect(post) AS posts, person
WITH friend,
     city,
     size(posts) AS postCount,
     size([p IN posts WHERE (p)-[:HAS_TAG]->()<-[:HAS_INTEREST]-(person)]) AS commonPostCount
RETURN friend.id AS personId,
       friend.firstName AS personFirstName,
       friend.lastName AS personLastName,
       commonPostCount - (postCount - commonPostCount) AS commonInterestScore,
       friend.gender AS personGender,
       city.name AS personCityName
ORDER BY commonInterestScore DESC, personId ASC
LIMIT 10
