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

Feature: Pattern2 - Pattern Comprehension

  Scenario: [1] Return a pattern comprehension
    Given an empty graph
    And having executed:
      """
      CREATE (a:A), (b:B)
      CREATE (a)-[:T]->(b),
             (b)-[:T]->(:C)
      """
    When executing query:
      """
      MATCH (n)
      RETURN [p = (n)-->() | p] AS list
      """
    Then the result should be, in any order:
      | list                |
      | [<(:A)-[:T]->(:B)>] |
      | [<(:B)-[:T]->(:C)>] |
      | []                  |
    And no side effects

  Scenario: [2] Return a pattern comprehension with label predicate
    Given an empty graph
    And having executed:
      """
      CREATE (a:A), (b:B), (c:C), (d:D)
      CREATE (a)-[:T]->(b),
             (a)-[:T]->(c),
             (a)-[:T]->(d)
      """
    When executing query:
      """
      MATCH (n:A)
      RETURN [p = (n)-->(:B) | p] AS list
      """
    Then the result should be, in any order:
      | list                |
      | [<(:A)-[:T]->(:B)>] |
    And no side effects

  Scenario: [3] Return a pattern comprehension with bound nodes
    Given an empty graph
    And having executed:
      """
      CREATE (a:A), (b:B)
      CREATE (a)-[:T]->(b)
      """
    When executing query:
      """
      MATCH (a:A), (b:B)
      RETURN [p = (a)-->(b) | p] AS list
      """
    Then the result should be, in any order:
      | list                |
      | [<(:A)-[:T]->(:B)>] |
    And no side effects

  
  Scenario: [4] Introduce a new node variable in pattern comprehension
    Given an empty graph
    And having executed:
      """
      CREATE (a), (b {name: 'val'}), (c)
      CREATE (a)-[:T]->(b),
             (b)-[:T]->(c)
      """
    When executing query:
      """
      MATCH (n)
      RETURN [(n)-[:T]->(b) | b.name] AS list
      """
    Then the result should be, in any order:
      | list    |
      | ['val'] |
      | [null]  |
      | []      |
    And no side effects

  
  Scenario: [5] Introduce a new relationship variable in pattern comprehension
    Given an empty graph
    And having executed:
      """
      CREATE (a), (b), (c)
      CREATE (a)-[:T {name: 'val'}]->(b),
             (b)-[:T]->(c)
      """
    When executing query:
      """
      MATCH (n)
      RETURN [(n)-[r:T]->() | r.name] AS list
      """
    Then the result should be, in any order:
      | list    |
      | ['val'] |
      | [null]  |
      | []      |
    And no side effects

  Scenario: [6] Aggregate on a pattern comprehension
    Given an empty graph
    And having executed:
      """
      CREATE (a:A), (:A), (:A)
      CREATE (a)-[:HAS]->()
      """
    When executing query:
      """
      MATCH (n:A)
      RETURN count([p = (n)-[:HAS]->() | p]) AS c
      """
    Then the result should be, in any order:
      | c |
      | 3 |
    And no side effects

  
  Scenario: [7] Use a pattern comprehension inside a list comprehension
    Given an empty graph
    And having executed:
      """
      CREATE (n1:X {n: 1}), (m1:Y), (i1:Y), (i2:Y)
      CREATE (n1)-[:T]->(m1),
             (m1)-[:T]->(i1),
             (m1)-[:T]->(i2)
      CREATE (n2:X {n: 2}), (m2), (i3:L), (i4:Y)
      CREATE (n2)-[:T]->(m2),
             (m2)-[:T]->(i3),
             (m2)-[:T]->(i4)
      """
    When executing query:
      """
      MATCH p = (n:X)-->()
      RETURN n, [x IN nodes(p) | size([(x)-->(:Y) | 1])] AS list
      """
    Then the result should be, in any order:
      | n           | list   |
      | (:X {n: 1}) | [1, 2] |
      | (:X {n: 2}) | [0, 1] |
    And no side effects

  Scenario: [8] Use a pattern comprehension in WITH
    Given an empty graph
    And having executed:
      """
      CREATE (a:A), (b:B)
      CREATE (a)-[:T]->(b),
             (b)-[:T]->(:C)
      """
    When executing query:
      """
      MATCH (n)-->(b)
      WITH [p = (n)-->() | p] AS ps, count(b) AS c
      RETURN ps, c
      """
    Then the result should be, in any order:
      | ps                  | c |
      | [<(:A)-[:T]->(:B)>] | 1 |
      | [<(:B)-[:T]->(:C)>] | 1 |
    And no side effects

  Scenario: [9] Use a variable-length pattern comprehension in WITH
    Given an empty graph
    And having executed:
      """
      CREATE (:A)-[:T]->(:B)
      """
    When executing query:
      """
      MATCH (a:A), (b:B)
      WITH [p = (a)-[*]->(b) | p] AS paths, count(a) AS c
      RETURN paths, c
      """
    Then the result should be, in any order:
      | paths               | c |
      | [<(:A)-[:T]->(:B)>] | 1 |
    And no side effects

  Scenario: [10] Use a pattern comprehension in RETURN
    Given an empty graph
    And having executed:
      """
      CREATE (a:A), (:A), (:A)
      CREATE (a)-[:HAS]->()
      """
    When executing query:
      """
      MATCH (n:A)
      RETURN [p = (n)-[:HAS]->() | p] AS ps
      """
    Then the result should be, in any order:
      | ps                  |
      | [<(:A)-[:HAS]->()>] |
      | []                  |
      | []                  |
    And no side effects

  Scenario: [11] Use a pattern comprehension and ORDER BY
    Given an empty graph
    And having executed:
      """
      CREATE (a {time: 10}), (b {time: 20})
      CREATE (a)-[:T]->(b)
      """
    When executing query:
      """
      MATCH (liker)
      RETURN [p = (liker)--() | p] AS isNew
        ORDER BY liker.time
      """
    Then the result should be, in order:
      | isNew                               |
      | [<({time: 10})-[:T]->({time: 20})>] |
      | [<({time: 20})<-[:T]-({time: 10})>] |
    And no side effects
