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

Feature: Match3 - Match fixed length patterns

  Scenario: [1] Get neighbours
    Given an empty graph
    And having executed:
      """
      CREATE (a:A {num: 1})-[:KNOWS]->(b:B {num: 2})
      """
    When executing query:
      """
      MATCH (n1)-[rel:KNOWS]->(n2)
      RETURN n1, n2
      """
    Then the result should be, in any order:
      | n1            | n2            |
      | (:A {num: 1}) | (:B {num: 2}) |
    And no side effects

  Scenario: [2] Directed match of a simple relationship
    Given an empty graph
    And having executed:
      """
      CREATE (:A)-[:LOOP]->(:B)
      """
    When executing query:
      """
      MATCH (a)-[r]->(b)
      RETURN a, r, b
      """
    Then the result should be, in any order:
      | a    | r       | b    |
      | (:A) | [:LOOP] | (:B) |
    And no side effects

  Scenario: [3] Undirected match on simple relationship graph
    Given an empty graph
    And having executed:
      """
      CREATE (:A)-[:LOOP]->(:B)
      """
    When executing query:
      """
      MATCH (a)-[r]-(b)
      RETURN a, r, b
      """
    Then the result should be, in any order:
      | a    | r       | b    |
      | (:A) | [:LOOP] | (:B) |
      | (:B) | [:LOOP] | (:A) |
    And no side effects

  Scenario: [4] Get two related nodes
    Given an empty graph
    And having executed:
      """
      CREATE (a:A {num: 1}),
        (a)-[:KNOWS]->(b:B {num: 2}),
        (a)-[:KNOWS]->(c:C {num: 3})
      """
    When executing query:
      """
      MATCH ()-[rel:KNOWS]->(x)
      RETURN x
      """
    Then the result should be, in any order:
      | x             |
      | (:B {num: 2}) |
      | (:C {num: 3}) |
    And no side effects

  Scenario: [5] Return two subgraphs with bound undirected relationship
    Given an empty graph
    And having executed:
      """
      CREATE (a:A {num: 1})-[:REL {name: 'r'}]->(b:B {num: 2})
      """
    When executing query:
      """
      MATCH (a)-[r {name: 'r'}]-(b)
      RETURN a, b
      """
    Then the result should be, in any order:
      | a             | b             |
      | (:B {num: 2}) | (:A {num: 1}) |
      | (:A {num: 1}) | (:B {num: 2}) |
    And no side effects

  Scenario: [6] Matching a relationship pattern using a label predicate
    Given an empty graph
    And having executed:
      """
      CREATE (a), (b1:Foo), (b2)
      CREATE (a)-[:T]->(b1),
             (a)-[:T]->(b2)
      """
    When executing query:
      """
      MATCH (a)-->(b:Foo)
      RETURN b
      """
    Then the result should be, in any order:
      | b      |
      | (:Foo) |
    And no side effects

  
  Scenario: [7] Matching nodes with many labels
    Given an empty graph
    And having executed:
      """
      CREATE (a:A:B:C:D:E:F:G:H:I:J:K:L:M),
             (b:U:V:W:X:Y:Z)
      CREATE (a)-[:T]->(b)
      """
    When executing query:
      """
      MATCH (n:A:B:C:D:E:F:G:H:I:J:K:L:M)-[:T]->(m:Z:Y:X:W:V:U)
      RETURN n, m
      """
    Then the result should be, in any order:
      | n                            | m              |
      | (:A:B:C:D:E:F:G:H:I:J:K:L:M) | (:Z:Y:X:W:V:U) |
    And no side effects

  Scenario: [8] Matching using relationship predicate with multiples of the same type
    Given an empty graph
    And having executed:
      """
      CREATE (a:A), (b:B)
      CREATE (a)-[:T]->(b)
      """
    When executing query:
      """
      MATCH (a)-[:T|:T]->(b)
      RETURN b
      """
    Then the result should be, in any order:
      | b    |
      | (:B) |
    And no side effects

  Scenario: [9] Get related to related to
    Given an empty graph
    And having executed:
      """
      CREATE (a:A {num: 1})-[:KNOWS]->(b:B {num: 2})-[:FRIEND]->(c:C {num: 3})
      """
    When executing query:
      """
      MATCH (n)-->(a)-->(b)
      RETURN b
      """
    Then the result should be, in any order:
      | b             |
      | (:C {num: 3}) |
    And no side effects

  Scenario: [10] Matching using self-referencing pattern returns no result
    Given an empty graph
    And having executed:
      """
      CREATE (a), (b), (c)
      CREATE (a)-[:T]->(b),
             (b)-[:T]->(c)
      """
    When executing query:
      """
      MATCH (a)-->(b), (b)-->(b)
      RETURN b
      """
    Then the result should be, in any order:
      | b |
    And no side effects

  Scenario: [11] Undirected match in self-relationship graph
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:LOOP]->(a)
      """
    When executing query:
      """
      MATCH (a)-[r]-(b)
      RETURN a, r, b
      """
    Then the result should be, in any order:
      | a    | r       | b    |
      | (:A) | [:LOOP] | (:A) |
    And no side effects

  Scenario: [12] Undirected match of self-relationship in self-relationship graph
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:LOOP]->(a)
      """
    When executing query:
      """
      MATCH (n)-[r]-(n)
      RETURN n, r
      """
    Then the result should be, in any order:
      | n    | r       |
      | (:A) | [:LOOP] |
    And no side effects

  Scenario: [13] Directed match on self-relationship graph
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:LOOP]->(a)
      """
    When executing query:
      """
      MATCH (a)-[r]->(b)
      RETURN a, r, b
      """
    Then the result should be, in any order:
      | a    | r       | b    |
      | (:A) | [:LOOP] | (:A) |
    And no side effects

  Scenario: [14] Directed match of self-relationship on self-relationship graph
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:LOOP]->(a)
      """
    When executing query:
      """
      MATCH (n)-[r]->(n)
      RETURN n, r
      """
    Then the result should be, in any order:
      | n    | r       |
      | (:A) | [:LOOP] |
    And no side effects

  
  Scenario: [15] Mixing directed and undirected pattern parts with self-relationship, simple
    Given an empty graph
    And having executed:
      """
      CREATE (:A)-[:T1]->(l:Looper),
             (l)-[:LOOP]->(l),
             (l)-[:T2]->(:B)
      """
    When executing query:
      """
      MATCH (x:A)-[r1]->(y)-[r2]-(z)
      RETURN x, r1, y, r2, z
      """
    Then the result should be, in any order:
      | x    | r1    | y         | r2      | z         |
      | (:A) | [:T1] | (:Looper) | [:LOOP] | (:Looper) |
      | (:A) | [:T1] | (:Looper) | [:T2]   | (:B)      |
    And no side effects

  
  Scenario: [16] Mixing directed and undirected pattern parts with self-relationship, undirected
    Given an empty graph
    And having executed:
      """
      CREATE (:A)-[:T1]->(l:Looper),
             (l)-[:LOOP]->(l),
             (l)-[:T2]->(:B)
      """
    When executing query:
      """
      MATCH (x)-[r1]-(y)-[r2]-(z)
      RETURN x, r1, y, r2, z
      """
    Then the result should be, in any order:
      | x         | r1      | y         | r2      | z         |
      | (:A)      | [:T1]   | (:Looper) | [:LOOP] | (:Looper) |
      | (:A)      | [:T1]   | (:Looper) | [:T2]   | (:B)      |
      | (:Looper) | [:LOOP] | (:Looper) | [:T1]   | (:A)      |
      | (:Looper) | [:LOOP] | (:Looper) | [:T2]   | (:B)      |
      | (:B)      | [:T2]   | (:Looper) | [:LOOP] | (:Looper) |
      | (:B)      | [:T2]   | (:Looper) | [:T1]   | (:A)      |
    And no side effects

  Scenario: [17] Handling cyclic patterns
    Given an empty graph
    And having executed:
      """
      CREATE (a {name: 'a'}), (b {name: 'b'}), (c {name: 'c'})
      CREATE (a)-[:A]->(b),
             (b)-[:B]->(a),
             (b)-[:B]->(c)
      """
    When executing query:
      """
      MATCH (a)-[:A]->()-[:B]->(a)
      RETURN a.name
      """
    Then the result should be, in any order:
      | a.name |
      | 'a'    |
    And no side effects

  Scenario: [18] Handling cyclic patterns when separated into two parts
    Given an empty graph
    And having executed:
      """
      CREATE (a {name: 'a'}), (b {name: 'b'}), (c {name: 'c'})
      CREATE (a)-[:A]->(b),
             (b)-[:B]->(a),
             (b)-[:B]->(c)
      """
    When executing query:
      """
      MATCH (a)-[:A]->(b), (b)-[:B]->(a)
      RETURN a.name
      """
    Then the result should be, in any order:
      | a.name |
      | 'a'    |
    And no side effects

  Scenario: [19] Two bound nodes pointing to the same node
    Given an empty graph
    And having executed:
      """
      CREATE (a {name: 'A'}), (b {name: 'B'}),
             (x1 {name: 'x1'}), (x2 {name: 'x2'})
      CREATE (a)-[:KNOWS]->(x1),
             (a)-[:KNOWS]->(x2),
             (b)-[:KNOWS]->(x1),
             (b)-[:KNOWS]->(x2)
      """
    When executing query:
      """
      MATCH (a {name: 'A'}), (b {name: 'B'})
      MATCH (a)-->(x)<-->(b)
      RETURN x
      """
    Then the result should be, in any order:
      | x              |
      | ({name: 'x1'}) |
      | ({name: 'x2'}) |
    And no side effects

  Scenario: [20] Three bound nodes pointing to the same node
    Given an empty graph
    And having executed:
      """
      CREATE (a {name: 'A'}), (b {name: 'B'}), (c {name: 'C'}),
             (x1 {name: 'x1'}), (x2 {name: 'x2'})
      CREATE (a)-[:KNOWS]->(x1),
             (a)-[:KNOWS]->(x2),
             (b)-[:KNOWS]->(x1),
             (b)-[:KNOWS]->(x2),
             (c)-[:KNOWS]->(x1),
             (c)-[:KNOWS]->(x2)
      """
    When executing query:
      """
      MATCH (a {name: 'A'}), (b {name: 'B'}), (c {name: 'C'})
      MATCH (a)-->(x), (b)-->(x), (c)-->(x)
      RETURN x
      """
    Then the result should be, in any order:
      | x              |
      | ({name: 'x1'}) |
      | ({name: 'x2'}) |
    And no side effects

  Scenario: [21] Three bound nodes pointing to the same node with extra connections
    Given an empty graph
    And having executed:
      """
      CREATE (a {name: 'a'}), (b {name: 'b'}), (c {name: 'c'}),
             (d {name: 'd'}), (e {name: 'e'}), (f {name: 'f'}),
             (g {name: 'g'}), (h {name: 'h'}), (i {name: 'i'}),
             (j {name: 'j'}), (k {name: 'k'})
      CREATE (a)-[:KNOWS]->(d),
             (a)-[:KNOWS]->(e),
             (a)-[:KNOWS]->(f),
             (a)-[:KNOWS]->(g),
             (a)-[:KNOWS]->(i),
             (b)-[:KNOWS]->(d),
             (b)-[:KNOWS]->(e),
             (b)-[:KNOWS]->(f),
             (b)-[:KNOWS]->(h),
             (b)-[:KNOWS]->(k),
             (c)-[:KNOWS]->(d),
             (c)-[:KNOWS]->(e),
             (c)-[:KNOWS]->(h),
             (c)-[:KNOWS]->(g),
             (c)-[:KNOWS]->(j)
      """
    When executing query:
      """
      MATCH (a {name: 'a'}), (b {name: 'b'}), (c {name: 'c'})
      MATCH (a)-->(x), (b)-->(x), (c)-->(x)
      RETURN x
      """
    Then the result should be, in any order:
      | x             |
      | ({name: 'd'}) |
      | ({name: 'e'}) |
    And no side effects

  Scenario: [22] Returning bound nodes that are not part of the pattern
    Given an empty graph
    And having executed:
      """
      CREATE (a {name: 'A'}), (b {name: 'B'}),
             (c {name: 'C'})
      CREATE (a)-[:KNOWS]->(b)
      """
    When executing query:
      """
      MATCH (a {name: 'A'}), (c {name: 'C'})
      MATCH (a)-->(b)
      RETURN a, b, c
      """
    Then the result should be, in any order:
      | a             | b             | c             |
      | ({name: 'A'}) | ({name: 'B'}) | ({name: 'C'}) |
    And no side effects

  Scenario: [23] Matching disconnected patterns
    Given an empty graph
    And having executed:
      """
      CREATE (a:A), (b:B), (c:C)
      CREATE (a)-[:T]->(b),
             (a)-[:T]->(c)
      """
    When executing query:
      """
      MATCH (a)-->(b)
      MATCH (c)-->(d)
      RETURN a, b, c, d
      """
    Then the result should be, in any order:
      | a    | b    | c    | d    |
      | (:A) | (:B) | (:A) | (:B) |
      | (:A) | (:B) | (:A) | (:C) |
      | (:A) | (:C) | (:A) | (:B) |
      | (:A) | (:C) | (:A) | (:C) |
    And no side effects

  Scenario: [24] Matching twice with duplicate relationship types on same relationship
    Given an empty graph
    And having executed:
      """
      CREATE (:A)-[:T]->(:B)
      """
    When executing query:
      """
      MATCH (a1)-[r:T]->()
      WITH r, a1
      MATCH (a1)-[r:T]->(b2)
      RETURN a1, r, b2
      """
    Then the result should be, in any order:
      | a1   | r    | b2   |
      | (:A) | [:T] | (:B) |
    And no side effects

  Scenario: [25] Matching twice with an additional node label
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:T]->()
      """
    When executing query:
      """
      MATCH (a1)-[r]->()
      WITH r, a1
      MATCH (a1:X)-[r]->(b2)
      RETURN a1, r, b2
      """
    Then the result should be, in any order:
      | a1 | r | b2 |
    And no side effects

  Scenario: [26] Matching twice with a duplicate predicate
    Given an empty graph
    And having executed:
      """
      CREATE (:X:Y)-[:T]->()
      """
    When executing query:
      """
      MATCH (a1:X:Y)-[r]->()
      WITH r, a1
      MATCH (a1:Y)-[r]->(b2)
      RETURN a1, r, b2
      """
    Then the result should be, in any order:
      | a1     | r    | b2 |
      | (:X:Y) | [:T] | () |
    And no side effects

  Scenario: [27] Matching from null nodes should return no results owing to finding no matches
    Given an empty graph
    When executing query:
      """
      OPTIONAL MATCH (a)
      WITH a
      MATCH (a)-->(b)
      RETURN b
      """
    Then the result should be, in any order:
      | b |
    And no side effects

  Scenario: [28] Matching from null nodes should return no results owing to matches being filtered out
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:T]->()
      """
    When executing query:
      """
      OPTIONAL MATCH (a:TheLabel)
      WITH a
      MATCH (a)-->(b)
      RETURN b
      """
    Then the result should be, in any order:
      | b |
    And no side effects

  Scenario: [29] Fail when re-using a relationship in the same pattern
    Given any graph
    When executing query:
      """
      MATCH (a)-[r]->()-[r]->(a)
      RETURN r
      """
    Then a SyntaxError should be raised at compile time: RelationshipUniquenessViolation

  
  Scenario: [30] Fail when using a list or nodes as a node
    Given any graph
    When executing query:
      """
      MATCH (n)
      WITH [n] AS users
      MATCH (users)-->(messages)
      RETURN messages
      """
    Then a SyntaxError should be raised at compile time: VariableTypeConflict
