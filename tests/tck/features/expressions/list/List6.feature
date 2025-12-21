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

Feature: List6 - List size

  Scenario: [1] Return list size
    Given any graph
    When executing query:
      """
      RETURN size([1, 2, 3]) AS n
      """
    Then the result should be, in any order:
      | n |
      | 3 |
    And no side effects

  Scenario: [2] Setting and returning the size of a list property
    Given an empty graph
    And having executed:
      """
      CREATE (:TheLabel)
      """
    When executing query:
      """
      MATCH (n:TheLabel)
      SET n.numbers = [1, 2, 3]
      RETURN size(n.numbers)
      """
    Then the result should be, in any order:
      | size(n.numbers) |
      | 3               |
    And the side effects should be:
      | +properties | 1 |

  Scenario: [3] Concatenating and returning the size of literal lists
    Given any graph
    When executing query:
      """
      RETURN size([[], []] + [[]]) AS l
      """
    Then the result should be, in any order:
      | l |
      | 3 |
    And no side effects

  Scenario: [4] `size()` on null list
    Given any graph
    When executing query:
      """
      WITH null AS l
      RETURN size(l), size(null)
      """
    Then the result should be, in any order:
      | size(l) | size(null) |
      | null    | null       |
    And no side effects

  
  Scenario: [5] Fail for `size()` on paths
    Given any graph
    When executing query:
      """
      MATCH p = (a)-[*]->(b)
      RETURN size(p)
      """
    Then a SyntaxError should be raised at compile time: InvalidArgumentType

  
  Scenario Outline: [6] Fail for `size()` on pattern predicates
    Given any graph
    When executing query:
      """
      MATCH (a), (b), (c)
      RETURN size(<pattern>)
      """
    Then a SyntaxError should be raised at compile time: UnexpectedSyntax

    Examples:
      | pattern                                    |
      | ()--()                                     |
      | ()--(a)                                    |
      | (a)-->()                                   |
      | (a)<--(a {})                               |
      | (a)-[:REL]->(b)                            |
      | (a)-[:REL]->(b)                            |
      | (a)-[:REL]->(:C)<-[:REL]-(a {num: 5})      |
      | ()-[:REL*0..2]->()<-[:REL]-(:A {num: 5})   |

  Scenario: [7] Using size of pattern comprehension to test existence
    Given an empty graph
    And having executed:
      """
      CREATE (a:X {num: 42}), (:X {num: 43})
      CREATE (a)-[:T]->()
      """
    When executing query:
      """
      MATCH (n:X)
      RETURN n, size([(n)--() | 1]) > 0 AS b
      """
    Then the result should be, in any order:
      | n              | b     |
      | (:X {num: 42}) | true  |
      | (:X {num: 43}) | false |
    And no side effects

  Scenario: [8] Get node degree via size of pattern comprehension
    Given an empty graph
    And having executed:
      """
      CREATE (x:X),
        (x)-[:T]->(),
        (x)-[:T]->(),
        (x)-[:T]->()
      """
    When executing query:
      """
      MATCH (a:X)
      RETURN size([(a)-->() | 1]) AS length
      """
    Then the result should be, in any order:
      | length |
      | 3      |
    And no side effects

  Scenario: [9] Get node degree via size of pattern comprehension that specifies a relationship type
    Given an empty graph
    And having executed:
      """
      CREATE (x:X),
        (x)-[:T]->(),
        (x)-[:T]->(),
        (x)-[:T]->(),
        (x)-[:OTHER]->()
      """
    When executing query:
      """
      MATCH (a:X)
      RETURN size([(a)-[:T]->() | 1]) AS length
      """
    Then the result should be, in any order:
      | length |
      | 3      |
    And no side effects

  Scenario: [10] Get node degree via size of pattern comprehension that specifies multiple relationship types
    Given an empty graph
    And having executed:
      """
      CREATE (x:X),
        (x)-[:T]->(),
        (x)-[:T]->(),
        (x)-[:T]->(),
        (x)-[:OTHER]->()
      """
    When executing query:
      """
      MATCH (a:X)
      RETURN size([(a)-[:T|OTHER]->() | 1]) AS length
      """
    Then the result should be, in any order:
      | length |
      | 4      |
    And no side effects
