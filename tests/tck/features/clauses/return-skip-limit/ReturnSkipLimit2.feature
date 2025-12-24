#
# Copyright (c) 2015-2022 "Neo Technology,"
# Network Engine for Objects in Lund AB [http://neotechnology.com]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Attribution Notice under the terms of the Apache License 2.0
#
# This work was created by the collective efforts of the openCypher community.
# Without limiting the terms of Section 6, any Derivative Work that is not
# approved by the public consensus process of the openCypher Implementers Group
# should not be described as “Cypher” (and Cypher® is a registered trademark of
# Neo4j Inc.) or as "openCypher". Extensions by implementers or prototypes or
# proposals for change that have been documented or implemented should only be
# described as "implementation extensions to Cypher" or as "proposed changes to
# Cypher that are not yet approved by the openCypher community".
#

#encoding: utf-8

Feature: ReturnSkipLimit2 - Limit

  Scenario: [1] Limit to two hits
    Given an empty graph
    When executing query:
      """
      UNWIND [1, 1, 1, 1, 1] AS i
      RETURN i
      LIMIT 2
      """
    Then the result should be, in any order:
      | i |
      | 1 |
      | 1 |
    And no side effects

  Scenario: [2] Limit to two hits with explicit order
    Given an empty graph
    And having executed:
      """
      CREATE ({name: 'A'}),
        ({name: 'B'}),
        ({name: 'C'}),
        ({name: 'D'}),
        ({name: 'E'})
      """
    When executing query:
      """
      MATCH (n)
      RETURN n
      ORDER BY n.name ASC
      LIMIT 2
      """
    Then the result should be, in order:
      | n             |
      | ({name: 'A'}) |
      | ({name: 'B'}) |
    And no side effects

  Scenario: [3] LIMIT 0 should return an empty result
    Given an empty graph
    And having executed:
      """
      CREATE (), (), ()
      """
    When executing query:
      """
      MATCH (n)
      RETURN n
        LIMIT 0
      """
    Then the result should be, in any order:
      | n |
    And no side effects

  Scenario: [4] Handle ORDER BY with LIMIT 1
    Given an empty graph
    And having executed:
      """
      CREATE (s:Person {name: 'Steven'}),
        (c:Person {name: 'Craig'})
      """
    When executing query:
      """
      MATCH (p:Person)
      RETURN p.name AS name
      ORDER BY p.name
      LIMIT 1
      """
    Then the result should be, in order:
      | name    |
      | 'Craig' |
    And no side effects

  Scenario: [5] ORDER BY with LIMIT 0 should not generate errors
    Given any graph
    When executing query:
      """
      MATCH (p:Person)
      RETURN p.name AS name
      ORDER BY p.name
      LIMIT 0
      """
    Then the result should be, in order:
      | name |
    And no side effects

  
  Scenario: [6] LIMIT with an expression that does not depend on variables
    Given any graph
    And having executed:
      """
      UNWIND range(1, 3) AS i
      CREATE ({nr: i})
      """
    When executing query:
      """
      MATCH (n)
      WITH n LIMIT toInteger(ceil(1.7))
      RETURN count(*) AS count
      """
    Then the result should be, in any order:
      | count |
      | 2     |
    And no side effects

  Scenario: [7] Limit to more rows than actual results 1
    Given an empty graph
    And having executed:
      """
      CREATE ({num: 1}), ({num: 3}), ({num: 2})
      """
    When executing query:
      """
      MATCH (foo)
      RETURN foo.num AS x
        ORDER BY x DESC
        LIMIT 4
      """
    Then the result should be, in order:
      | x |
      | 3 |
      | 2 |
      | 1 |
    And no side effects

  Scenario: [8] Limit to more rows than actual results 2
    Given an empty graph
    And having executed:
      """
      CREATE (a:A), (n1 {num: 1}), (n2 {num: 2}),
             (m1), (m2)
      CREATE (a)-[:T]->(n1),
             (n1)-[:T]->(m1),
             (a)-[:T]->(n2),
             (n2)-[:T]->(m2)
      """
    When executing query:
      """
      MATCH (a:A)-->(n)-->(m)
      RETURN n.num, count(*)
        ORDER BY n.num
        LIMIT 1000
      """
    Then the result should be, in order:
      | n.num | count(*) |
      | 1     | 1        |
      | 2     | 1        |
    And no side effects

  Scenario: [9] Fail when using non-constants in LIMIT
    Given any graph
    When executing query:
      """
      MATCH (n) RETURN n LIMIT n.count
      """
    Then a SyntaxError should be raised at compile time: NonConstantExpression

  
  Scenario: [10] Negative parameter for LIMIT should fail
    Given any graph
    And having executed:
      """
      CREATE (s:Person {name: 'Steven'}),
             (c:Person {name: 'Craig'})
      """
    And parameters are:
      | _limit | -1 |
    When executing query:
      """
      MATCH (p:Person)
      RETURN p.name AS name
      LIMIT $_limit
      """
    Then a SyntaxError should be raised at runtime: NegativeIntegerArgument

  
  Scenario: [11] Negative parameter for LIMIT with ORDER BY should fail
    Given any graph
    And having executed:
      """
      CREATE (s:Person {name: 'Steven'}),
             (c:Person {name: 'Craig'})
      """
    And parameters are:
      | _limit | -1 |
    When executing query:
      """
      MATCH (p:Person)
      RETURN p.name AS name
      ORDER BY name LIMIT $_limit
      """
    Then a SyntaxError should be raised at runtime: NegativeIntegerArgument

  Scenario: [12] Fail when using negative value in LIMIT 1
    Given any graph
    When executing query:
      """
      MATCH (n)
      RETURN n
        LIMIT -1
      """
    Then a SyntaxError should be raised at compile time: NegativeIntegerArgument

  Scenario: [13] Fail when using negative value in LIMIT 2
    Given any graph
    And having executed:
      """
      CREATE (s:Person {name: 'Steven'}),
             (c:Person {name: 'Craig'})
      """
    When executing query:
      """
      MATCH (p:Person)
      RETURN p.name AS name
      LIMIT -1
      """
    Then a SyntaxError should be raised at compile time: NegativeIntegerArgument

  
  Scenario: [14] Floating point parameter for LIMIT should fail
    Given any graph
    And having executed:
      """
      CREATE (s:Person {name: 'Steven'}),
             (c:Person {name: 'Craig'})
      """
    And parameters are:
      | _limit | 1.5 |
    When executing query:
      """
      MATCH (p:Person)
      RETURN p.name AS name
      LIMIT $_limit
      """
    Then a SyntaxError should be raised at runtime: InvalidArgumentType

  
  Scenario: [15] Floating point parameter for LIMIT with ORDER BY should fail
    Given any graph
    And having executed:
      """
      CREATE (s:Person {name: 'Steven'}),
             (c:Person {name: 'Craig'})
      """
    And parameters are:
      | _limit | 1.5 |
    When executing query:
      """
      MATCH (p:Person)
      RETURN p.name AS name
      ORDER BY name LIMIT $_limit
      """
    Then a SyntaxError should be raised at runtime: InvalidArgumentType

  
  Scenario: [16] Fail when using floating point in LIMIT 1
    Given any graph
    When executing query:
      """
      MATCH (n)
      RETURN n
        LIMIT 1.7
      """
    Then a SyntaxError should be raised at compile time: InvalidArgumentType

  
  Scenario: [17] Fail when using floating point in LIMIT 2
    Given any graph
    And having executed:
      """
      CREATE (s:Person {name: 'Steven'}),
             (c:Person {name: 'Craig'})
      """
    When executing query:
      """
      MATCH (p:Person)
      RETURN p.name AS name
      LIMIT 1.5
      """
    Then a SyntaxError should be raised at compile time: InvalidArgumentType
