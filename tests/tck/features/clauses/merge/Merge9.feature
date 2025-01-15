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

Feature: Merge9 - Merge clause interoperation with other clauses

  Scenario: [1] UNWIND with one MERGE
    Given an empty graph
    When executing query:
      """
      UNWIND [1, 2, 3, 4] AS int
      MERGE (n {id: int})
      RETURN count(*)
      """
    Then the result should be, in any order:
      | count(*) |
      | 4        |
    And the side effects should be:
      | +nodes      | 4 |
      | +properties | 4 |

  Scenario: [2] UNWIND with multiple MERGE
    Given an empty graph
    When executing query:
      """
      UNWIND ['Keanu Reeves', 'Hugo Weaving', 'Carrie-Anne Moss', 'Laurence Fishburne'] AS actor
      MERGE (m:Movie {name: 'The Matrix'})
      MERGE (p:Person {name: actor})
      MERGE (p)-[:ACTED_IN]->(m)
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes         | 5 |
      | +relationships | 4 |
      | +labels        | 2 |
      | +properties    | 5 |

  Scenario: [3] Mixing MERGE with CREATE
    Given an empty graph
    When executing query:
      """
      CREATE (a:A), (b:B)
      MERGE (a)-[:KNOWS]->(b)
      CREATE (b)-[:KNOWS]->(c:C)
      RETURN count(*)
      """
    Then the result should be, in any order:
      | count(*) |
      | 1        |
    And the side effects should be:
      | +nodes         | 3 |
      | +relationships | 2 |
      | +labels        | 3 |

  Scenario: [4] MERGE after WITH with predicate and WITH with aggregation
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 42})
      """
    When executing query:
      """
      UNWIND [42] AS props
      WITH props WHERE props > 32
      WITH DISTINCT props AS p
      MERGE (a:A {num: p})
      RETURN a.num AS prop
      """
    Then the result should be, in any order:
      | prop |
      | 42   |
    And no side effects

