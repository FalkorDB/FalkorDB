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

Feature: List12 - List Comprehension

  Scenario: [1] Collect and extract using a list comprehension
    Given an empty graph
    And having executed:
      """
      CREATE (:Label1 {name: 'original'})
      """
    When executing query:
      """
      MATCH (a:Label1)
      WITH collect(a) AS nodes
      WITH nodes, [x IN nodes | x.name] AS oldNames
      UNWIND nodes AS n
      SET n.name = 'newName'
      RETURN n.name, oldNames
      """
    Then the result should be, in any order:
      | n.name    | oldNames     |
      | 'newName' | ['original'] |
    And the side effects should be:
      | +properties | 1 |
      | -properties | 1 |

  Scenario: [2] Collect and filter using a list comprehension
    Given an empty graph
    And having executed:
      """
      CREATE (:Label1 {name: 'original'})
      """
    When executing query:
      """
      MATCH (a:Label1)
      WITH collect(a) AS nodes
      WITH nodes, [x IN nodes WHERE x.name = 'original'] AS noopFiltered
      UNWIND nodes AS n
      SET n.name = 'newName'
      RETURN n.name, size(noopFiltered)
      """
    Then the result should be, in any order:
      | n.name    | size(noopFiltered) |
      | 'newName' | 1                  |
    And the side effects should be:
      | +properties | 1 |
      | -properties | 1 |

  Scenario: [3] Size of list comprehension
    Given an empty graph
    When executing query:
      """
      MATCH (n)
      OPTIONAL MATCH (n)-[r]->(m)
      RETURN size([x IN collect(r) WHERE x <> null]) AS cn
      """
    Then the result should be, in any order:
      | cn |
      | 0  |
    And no side effects

  Scenario: [4] Returning a list comprehension
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)
      CREATE (a)-[:T]->(:B),
             (a)-[:T]->(:C)
      """
    When executing query:
      """
      MATCH p = (n)-->()
      RETURN [x IN collect(p) | head(nodes(x))] AS p
      """
    Then the result should be, in any order:
      | p            |
      | [(:A), (:A)] |
    And no side effects

  Scenario: [5] Using a list comprehension in a WITH
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)
      CREATE (a)-[:T]->(:B),
             (a)-[:T]->(:C)
      """
    When executing query:
      """
      MATCH p = (n:A)-->()
      WITH [x IN collect(p) | head(nodes(x))] AS p, count(n) AS c
      RETURN p, c
      """
    Then the result should be, in any order:
      | p            | c |
      | [(:A), (:A)] | 2 |
    And no side effects

  Scenario: [6] Using a list comprehension in a WHERE
    Given an empty graph
    And having executed:
      """
      CREATE (a:A {name: 'c'})
      CREATE (a)-[:T]->(:B),
             (a)-[:T]->(:C)
      """
    When executing query:
      """
      MATCH (n)-->(b)
      WHERE n.name IN [x IN labels(b) | toLower(x)]
      RETURN b
      """
    Then the result should be, in any order:
      | b    |
      | (:C) |
    And no side effects

  
  Scenario: [7] Fail when using aggregation in list comprehension
    Given any graph
    When executing query:
      """
      MATCH (n)
      RETURN [x IN [1, 2, 3, 4, 5] | count(*)]
      """
    Then a SyntaxError should be raised at compile time: InvalidAggregation
