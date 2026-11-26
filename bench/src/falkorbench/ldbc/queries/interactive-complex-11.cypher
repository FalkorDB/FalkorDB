// Q11. Job referral
/*
:param [{ personId, countryName, workFromYear }] => { RETURN
  10995116277918 AS personId,
  "Hungary" AS countryName,
  2011 AS workFromYear
}
*/
MATCH (person:Person {id: $personId })-[:KNOWS*1..2]-(friend:Person)
WHERE not(person=friend)
WITH DISTINCT friend
MATCH (friend)-[workAt:WORK_AT]->(company:Company)-[:IS_LOCATED_IN]->(:Country {name: $countryName })
WHERE workAt.workFrom < $workFromYear
RETURN
        friend.id AS personId,
        friend.firstName AS personFirstName,
        friend.lastName AS personLastName,
        company.name AS organizationName,
        workAt.workFrom AS organizationWorkFromYear
ORDER BY
        organizationWorkFromYear ASC,
        toInteger(personId) ASC,
        organizationName DESC
LIMIT 10
