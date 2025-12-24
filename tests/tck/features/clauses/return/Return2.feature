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

Feature: Return2 - Return single expression (correctly projecting an expression)

  Scenario: [1] Arithmetic expressions should propagate null values
    Given any graph
    When executing query:
      """
      RETURN 1 + (2 - (3 * (4 / (5 ^ (6 % null))))) AS a
      """
    Then the result should be, in any order:
      | a    |
      | null |
    And no side effects

  Scenario: [2] Returning a node property value
    Given an empty graph
    And having executed:
      """
      CREATE ({num: 1})
      """
    When executing query:
      """
      MATCH (a)
      RETURN a.num
      """
    Then the result should be, in any order:
      | a.num |
      | 1     |
    And no side effects

  Scenario: [3] Missing node property should become null
    Given an empty graph
    And having executed:
      """
      CREATE ({num: 1})
      """
    When executing query:
      """
      MATCH (a)
      RETURN a.name
      """
    Then the result should be, in any order:
      | a.name |
      | null   |
    And no side effects

  Scenario: [4] Returning a relationship property value
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:T {num: 1}]->()
      """
    When executing query:
      """
      MATCH ()-[r]->()
      RETURN r.num
      """
    Then the result should be, in any order:
      | r.num |
      | 1     |
    And no side effects

  Scenario: [5] Missing relationship property should become null
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:T {name: 1}]->()
      """
    When executing query:
      """
      MATCH ()-[r]->()
      RETURN r.name2
      """
    Then the result should be, in any order:
      | r.name2 |
      | null    |
    And no side effects

  Scenario: [6] Adding a property and a literal in projection
    Given an empty graph
    And having executed:
      """
      CREATE ({num: 1})
      """
    When executing query:
      """
      MATCH (a)
      RETURN a.num + 1 AS foo
      """
    Then the result should be, in any order:
      | foo |
      | 2   |
    And no side effects

  Scenario: [7] Adding list properties in projection
    Given an empty graph
    And having executed:
      """
      CREATE ({list1: [1, 2, 3], list2: [4, 5]})
      """
    When executing query:
      """
      MATCH (a)
      RETURN a.list2 + a.list1 AS foo
      """
    Then the result should be, in any order:
      | foo             |
      | [4, 5, 1, 2, 3] |
    And no side effects

  Scenario: [8] Returning label predicate expression
    Given an empty graph
    And having executed:
      """
      CREATE (), (:Foo)
      """
    When executing query:
      """
      MATCH (n)
      RETURN (n:Foo)
      """
    Then the result should be, in any order:
      | (n:Foo) |
      | true    |
      | false   |
    And no side effects

  Scenario: [9] Returning a projected map
    Given an empty graph
    And having executed:
      """
      CREATE ({numbers: [1, 2, 3]})
      """
    When executing query:
      """
      RETURN {a: 1, b: 'foo'}
      """
    Then the result should be, in any order:
      | {a: 1, b: 'foo'} |
      | {a: 1, b: 'foo'} |
    And no side effects

  Scenario: [10] Return count aggregation over an empty graph
    Given an empty graph
    When executing query:
      """
      MATCH (a)
      RETURN count(a) > 0
      """
    Then the result should be, in any order:
      | count(a) > 0 |
      | false        |
    And no side effects

  Scenario: [11] RETURN does not lose precision on large integers
    Given an empty graph
    And having executed:
      """
      CREATE (:TheLabel {id: 4611686018427387905})
      """
    When executing query:
      """
      MATCH (p:TheLabel)
      RETURN p.id
      """
    Then the result should be, in any order:
      | p.id                |
      | 4611686018427387905 |
    And no side effects

  Scenario: [12] Projecting a list of nodes and relationships
    Given an empty graph
    And having executed:
      """
      CREATE (a:A), (b:B)
      CREATE (a)-[:T]->(b)
      """
    When executing query:
      """
      MATCH (n)-[r]->(m)
      RETURN [n, r, m] AS r
      """
    Then the result should be, in any order:
      | r                  |
      | [(:A), [:T], (:B)] |
    And no side effects

  Scenario: [13] Projecting a map of nodes and relationships
    Given an empty graph
    And having executed:
      """
      CREATE (a:A), (b:B)
      CREATE (a)-[:T]->(b)
      """
    When executing query:
      """
      MATCH (n)-[r]->(m)
      RETURN {node1: n, rel: r, node2: m} AS m
      """
    Then the result should be, in any order:
      | m                                     |
      | {node1: (:A), rel: [:T], node2: (:B)} |
    And no side effects

  Scenario: [14] Do not fail when returning type of deleted relationships
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:T]->()
      """
    When executing query:
      """
      MATCH ()-[r]->()
      DELETE r
      RETURN type(r)
      """
    Then the result should be, in any order:
      | type(r) |
      | 'T'     |
    And the side effects should be:
      | -relationships | 1 |

  
  Scenario: [15] Fail when returning properties of deleted nodes
    Given an empty graph
    And having executed:
      """
      CREATE ({num: 0})
      """
    When executing query:
      """
      MATCH (n)
      DELETE n
      RETURN n.num
      """
    Then a EntityNotFound should be raised at runtime: DeletedEntityAccess

  
  Scenario: [16] Fail when returning labels of deleted nodes
    Given an empty graph
    And having executed:
      """
      CREATE (:A)
      """
    When executing query:
      """
      MATCH (n)
      DELETE n
      RETURN labels(n)
      """
    Then a EntityNotFound should be raised at runtime: DeletedEntityAccess

  
  Scenario: [17] Fail when returning properties of deleted relationships
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:T {num: 0}]->()
      """
    When executing query:
      """
      MATCH ()-[r]->()
      DELETE r
      RETURN r.num
      """
    Then a EntityNotFound should be raised at runtime: DeletedEntityAccess

  Scenario: [18] Fail on projecting a non-existent function
    Given any graph
    When executing query:
      """
      MATCH (a)
      RETURN foo(a)
      """
    Then a SyntaxError should be raised at compile time: UnknownFunction
