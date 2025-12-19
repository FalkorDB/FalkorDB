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

Feature: Merge7 - Merge relationships - on match

  Scenario: [1] Using ON MATCH on created node
    Given an empty graph
    And having executed:
      """
      CREATE (:A), (:B)
      """
    When executing query:
      """
      MATCH (a:A), (b:B)
      MERGE (a)-[:KNOWS]->(b)
        ON MATCH SET b.created = 1
      """
    Then the result should be empty
    And the side effects should be:
      | +relationships | 1 |

  Scenario: [2] Using ON MATCH on created relationship
    Given an empty graph
    And having executed:
      """
      CREATE (:A), (:B)
      """
    When executing query:
      """
      MATCH (a:A), (b:B)
      MERGE (a)-[r:KNOWS]->(b)
        ON MATCH SET r.created = 1
      """
    Then the result should be empty
    And the side effects should be:
      | +relationships | 1 |

  Scenario: [3] Using ON MATCH on a relationship
    Given an empty graph
    And having executed:
      """
      CREATE (a:A), (b:B)
      CREATE (a)-[:TYPE]->(b)
      """
    When executing query:
      """
      MATCH (a:A), (b:B)
      MERGE (a)-[r:TYPE]->(b)
        ON MATCH SET r.name = 'Lola'
      RETURN count(r)
      """
    Then the result should be, in any order:
      | count(r) |
      | 1        |
    And the side effects should be:
      | +properties | 1 |

  Scenario: [4] Copying properties from node with ON MATCH
    Given an empty graph
    And having executed:
      """
      CREATE (:A {name: 'A'}), (:B {name: 'B'})
      """
    And having executed:
      """
      MATCH (a:A), (b:B)
      CREATE (a)-[:TYPE {name: 'bar'}]->(b)
      """
    When executing query:
      """
      MATCH (a {name: 'A'}), (b {name: 'B'})
      MERGE (a)-[r:TYPE]->(b)
        ON MATCH SET r = a
      """
    Then the result should be empty
    And the side effects should be:
      | +properties | 1 |
      | -properties | 1 |
    When executing control query:
      """
      MATCH ()-[r:TYPE]->()
      RETURN [key IN keys(r) | key + '->' + r[key]] AS keyValue
      """
    Then the result should be, in any order:
      | keyValue    |
      | ['name->A'] |

  @skip
  Scenario: [5] Copying properties from literal map with ON MATCH
    Given an empty graph
    And having executed:
      """
      CREATE (:A {name: 'A'}), (:B {name: 'B'})
      """
    And having executed:
      """
      MATCH (a:A), (b:B)
      CREATE (a)-[:TYPE {name: 'bar'}]->(b)
      """
    When executing query:
      """
      MATCH (a {name: 'A'}), (b {name: 'B'})
      MERGE (a)-[r:TYPE]->(b)
        ON MATCH SET r += {name: 'baz', name2: 'baz'}
      """
    Then the result should be empty
    And the side effects should be:
      | +properties | 2 |
      | -properties | 1 |
    When executing control query:
      """
      MATCH ()-[r:TYPE]->()
      RETURN [key IN keys(r) | key + '->' + r[key]] AS keyValue
      """
    Then the result should be (ignoring element order for lists):
      | keyValue                    |
      | ['name->baz', 'name2->baz'] |
