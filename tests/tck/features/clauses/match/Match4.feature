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

Feature: Match4 - Match variable length patterns scenarios

  @skip
  Scenario: [1] Handling fixed-length variable length pattern
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:T]->()
      """
    When executing query:
      """
      MATCH (a)-[r*1..1]->(b)
      RETURN r
      """
    Then the result should be, in any order:
      | r      |
      | [[:T]] |
    And no side effects

  Scenario: [2] Simple variable length pattern
    Given an empty graph
    And having executed:
      """
      CREATE (a {name: 'A'}), (b {name: 'B'}),
             (c {name: 'C'}), (d {name: 'D'})
      CREATE (a)-[:CONTAINS]->(b),
             (b)-[:CONTAINS]->(c),
             (c)-[:CONTAINS]->(d)
      """
    When executing query:
      """
      MATCH (a {name: 'A'})-[*]->(x)
      RETURN x
      """
    Then the result should be, in any order:
      | x             |
      | ({name: 'B'}) |
      | ({name: 'C'}) |
      | ({name: 'D'}) |
    And no side effects

  Scenario: [3] Zero-length variable length pattern in the middle of the pattern
    Given an empty graph
    And having executed:
      """
      CREATE (a {name: 'A'}), (b {name: 'B'}),
             (c {name: 'C'}), ({name: 'D'}),
             ({name: 'E'})
      CREATE (a)-[:CONTAINS]->(b),
             (b)-[:FRIEND]->(c)
      """
    When executing query:
      """
      MATCH (a {name: 'A'})-[:CONTAINS*0..1]->(b)-[:FRIEND*0..1]->(c)
      RETURN a, b, c
      """
    Then the result should be, in any order:
      | a             | b             | c             |
      | ({name: 'A'}) | ({name: 'A'}) | ({name: 'A'}) |
      | ({name: 'A'}) | ({name: 'B'}) | ({name: 'B'}) |
      | ({name: 'A'}) | ({name: 'B'}) | ({name: 'C'}) |
    And no side effects

  Scenario: [4] Matching longer variable length paths
    Given an empty graph
    And having executed:
      """
      CREATE (a {var: 'start'}), (b {var: 'end'})
      WITH *
      UNWIND range(1, 20) AS i
      CREATE (n {var: i})
      WITH a, b, [a] + collect(n) + [b] AS nodeList
      UNWIND range(0, size(nodeList) - 2, 1) AS i
      WITH nodeList[i] AS n1, nodeList[i+1] AS n2
      CREATE (n1)-[:T]->(n2)
      """
    When executing query:
      """
      MATCH (n {var: 'start'})-[:T*]->(m {var: 'end'})
      RETURN m
      """
    Then the result should be, in any order:
      | m              |
      | ({var: 'end'}) |
    And no side effects

  Scenario: [5] Matching variable length pattern with property predicate
    Given an empty graph
    And having executed:
      """
      CREATE (a:Artist:A), (b:Artist:B), (c:Artist:C)
      CREATE (a)-[:WORKED_WITH {year: 1987}]->(b),
             (b)-[:WORKED_WITH {year: 1988}]->(c)
      """
    When executing query:
      """
      MATCH (a:Artist)-[:WORKED_WITH* {year: 1988}]->(b:Artist)
      RETURN *
      """
    Then the result should be, in any order:
      | a           | b           |
      | (:Artist:B) | (:Artist:C) |
    And no side effects

  @skip
  Scenario: [6] Matching variable length patterns from a bound node
    Given an empty graph
    And having executed:
      """
      CREATE (a:A), (b), (c)
      CREATE (a)-[:X]->(b),
             (b)-[:Y]->(c)
      """
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[r*2]->()
      RETURN r
      """
    Then the result should be (ignoring element order for lists):
      | r            |
      | [[:X], [:Y]] |
    And no side effects

  @skip
  Scenario: [7] Matching variable length patterns including a bound relationship
    Given an empty graph
    And having executed:
      """
      CREATE (n0:Node),
             (n1:Node),
             (n2:Node),
             (n3:Node),
             (n0)-[:EDGE]->(n1),
             (n1)-[:EDGE]->(n2),
             (n2)-[:EDGE]->(n3)
      """
    When executing query:
      """
      MATCH ()-[r:EDGE]-()
      MATCH p = (n)-[*0..1]-()-[r]-()-[*0..1]-(m)
      RETURN count(p) AS c
      """
    Then the result should be, in any order:
      | c  |
      | 32 |
    And no side effects

  Scenario: [8] Matching relationships into a list and matching variable length using the list
    Given an empty graph
    And having executed:
      """
      CREATE (a:A), (b:B), (c:C)
      CREATE (a)-[:Y]->(b),
             (b)-[:Y]->(c)
      """
    When executing query:
      """
      MATCH ()-[r1]->()-[r2]->()
      WITH [r1, r2] AS rs
        LIMIT 1
      MATCH (first)-[rs*]->(second)
      RETURN first, second
      """
    Then the result should be, in any order:
      | first | second |
      | (:A)  | (:C)   |
    And no side effects

  @skipGrammarCheck
  Scenario: [9] Fail when asterisk operator is missing
    Given an empty graph
    And having executed:
      """
      CREATE (n0:A {name: 'n0'}),
             (n00:B {name: 'n00'}),
             (n01:B {name: 'n01'}),
             (n000:C {name: 'n000'}),
             (n001:C {name: 'n001'}),
             (n010:C {name: 'n010'}),
             (n011:C {name: 'n011'}),
             (n0000:D {name: 'n0000'}),
             (n0001:D {name: 'n0001'}),
             (n0010:D {name: 'n0010'}),
             (n0011:D {name: 'n0011'}),
             (n0100:D {name: 'n0100'}),
             (n0101:D {name: 'n0101'}),
             (n0110:D {name: 'n0110'}),
             (n0111:D {name: 'n0111'})
      CREATE (n0)-[:LIKES]->(n00),
             (n0)-[:LIKES]->(n01),
             (n00)-[:LIKES]->(n000),
             (n00)-[:LIKES]->(n001),
             (n01)-[:LIKES]->(n010),
             (n01)-[:LIKES]->(n011),
             (n000)-[:LIKES]->(n0000),
             (n000)-[:LIKES]->(n0001),
             (n001)-[:LIKES]->(n0010),
             (n001)-[:LIKES]->(n0011),
             (n010)-[:LIKES]->(n0100),
             (n010)-[:LIKES]->(n0101),
             (n011)-[:LIKES]->(n0110),
             (n011)-[:LIKES]->(n0111)
      """
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES..]->(c)
      RETURN c.name
      """
    Then a SyntaxError should be raised at compile time: InvalidRelationshipPattern

  @skipGrammarCheck
  Scenario: [10] Fail on negative bound
    Given an empty graph
    And having executed:
      """
      CREATE (n0:A {name: 'n0'}),
             (n00:B {name: 'n00'}),
             (n01:B {name: 'n01'}),
             (n000:C {name: 'n000'}),
             (n001:C {name: 'n001'}),
             (n010:C {name: 'n010'}),
             (n011:C {name: 'n011'}),
             (n0000:D {name: 'n0000'}),
             (n0001:D {name: 'n0001'}),
             (n0010:D {name: 'n0010'}),
             (n0011:D {name: 'n0011'}),
             (n0100:D {name: 'n0100'}),
             (n0101:D {name: 'n0101'}),
             (n0110:D {name: 'n0110'}),
             (n0111:D {name: 'n0111'})
      CREATE (n0)-[:LIKES]->(n00),
             (n0)-[:LIKES]->(n01),
             (n00)-[:LIKES]->(n000),
             (n00)-[:LIKES]->(n001),
             (n01)-[:LIKES]->(n010),
             (n01)-[:LIKES]->(n011),
             (n000)-[:LIKES]->(n0000),
             (n000)-[:LIKES]->(n0001),
             (n001)-[:LIKES]->(n0010),
             (n001)-[:LIKES]->(n0011),
             (n010)-[:LIKES]->(n0100),
             (n010)-[:LIKES]->(n0101),
             (n011)-[:LIKES]->(n0110),
             (n011)-[:LIKES]->(n0111)
      """
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES*-2]->(c)
      RETURN c.name
      """
    Then a SyntaxError should be raised at compile time: InvalidRelationshipPattern
