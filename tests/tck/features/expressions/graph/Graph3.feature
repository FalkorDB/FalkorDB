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

Feature: Graph3 - Node labels

    # similar to Create1.[1] => rename to make clear labels() is tested
  Scenario: [1] Creating node without label
    Given an empty graph
    When executing query:
      """
      CREATE (node)
      RETURN labels(node)
      """
    Then the result should be, in any order:
      | labels(node) |
      | []           |
    And the side effects should be:
      | +nodes | 1 |

  # similar to Create1.[3]/[4] => rename to make clear labels() is tested
  Scenario: [2] Creating node with two labels
    Given an empty graph
    When executing query:
      """
      CREATE (node:Foo:Bar {name: 'Mattias'})
      RETURN labels(node)
      """
    Then the result should be, in any order:
      | labels(node)   |
      | ['Foo', 'Bar'] |
    And the side effects should be:
      | +nodes      | 1 |
      | +labels     | 2 |
      | +properties | 1 |

  # consider adding to Create1
  Scenario: [3] Ignore space when creating node with labels
    Given an empty graph
    When executing query:
      """
      CREATE (node :Foo:Bar)
      RETURN labels(node)
      """
    Then the result should be, in any order:
      | labels(node)   |
      | ['Foo', 'Bar'] |
    And the side effects should be:
      | +nodes  | 1 |
      | +labels | 2 |

  # consider adding to Create1
  Scenario: [4] Create node with label in pattern
    Given an empty graph
    When executing query:
      """
      CREATE (n:Person)-[:OWNS]->(:Dog)
      RETURN labels(n)
      """
    Then the result should be, in any order:
      | labels(n)  |
      | ['Person'] |
    And the side effects should be:
      | +nodes         | 2 |
      | +relationships | 1 |
      | +labels        | 2 |

  Scenario: [5] Using `labels()` in return clauses
    Given an empty graph
    And having executed:
      """
      CREATE ()
      """
    When executing query:
      """
      MATCH (n)
      RETURN labels(n)
      """
    Then the result should be, in any order:
      | labels(n) |
      | []        |
    And no side effects

  
  Scenario: [6] `labels()` should accept type Any
    Given an empty graph
    And having executed:
      """
      CREATE (:Foo), (:Foo:Bar)
      """
    When executing query:
      """
      MATCH (a)
      WITH [a, 1] AS list
      RETURN labels(list[0]) AS l
      """
    Then the result should be (ignoring element order for lists):
      | l              |
      | ['Foo']        |
      | ['Foo', 'Bar'] |
    And no side effects

  Scenario: [7] `labels()` on null node
    Given an empty graph
    When executing query:
      """
      OPTIONAL MATCH (n:DoesNotExist)
      RETURN labels(n), labels(null)
      """
    Then the result should be, in any order:
      | labels(n) | labels(null) |
      | null      | null         |
    And no side effects

  Scenario: [8] `labels()` failing on a path
    Given an empty graph
    And having executed:
      """
      CREATE (:Foo), (:Foo:Bar)
      """
    When executing query:
      """
      MATCH p = (a)
      RETURN labels(p) AS l
      """
    Then a SyntaxError should be raised at compile time: InvalidArgumentType

  Scenario: [9] `labels()` failing on invalid arguments
    Given an empty graph
    And having executed:
      """
      CREATE (:Foo), (:Foo:Bar)
      """
    When executing query:
      """
      MATCH (a)
      WITH [a, 1] AS list
      RETURN labels(list[1]) AS l
      """
    Then a TypeError should be raised at runtime: InvalidArgumentValue
