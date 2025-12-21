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

#consider splitting into separate category optional-match
Feature: Match7 - Optional match

  @skip
  Scenario: [1] Simple OPTIONAL MATCH on empty graph
    Given an empty graph
    When executing query:
      """
      OPTIONAL MATCH (n)
      RETURN n
      """
    Then the result should be, in any order:
      | n    |
      | null |
    And no side effects

  @skip
  Scenario: [2] OPTIONAL MATCH with previously bound nodes
    Given an empty graph
    And having executed:
      """
      CREATE ()
      """
    When executing query:
      """
      MATCH (n)
      OPTIONAL MATCH (n)-[:NOT_EXIST]->(x)
      RETURN n, x
      """
    Then the result should be, in any order:
      | n  | x    |
      | () | null |
    And no side effects

  @skip
  Scenario: [3] OPTIONAL MATCH and bound nodes
    Given an empty graph
    And having executed:
      """
      CREATE (s:Single), (a:A {num: 42}),
             (b:B {num: 46}), (c:C)
      CREATE (s)-[:REL]->(a),
             (s)-[:REL]->(b),
             (a)-[:REL]->(c),
             (b)-[:LOOP]->(b)
      """
    When executing query:
      """
      MATCH (a:A), (b:C)
      OPTIONAL MATCH (x)-->(b)
      RETURN x
      """
    Then the result should be, in any order:
      | x              |
      | (:A {num: 42}) |
    And no side effects

  @skip
  Scenario: [4] Optionally matching relationship with bound nodes in reverse direction
    Given an empty graph
    And having executed:
      """
      CREATE (:A)-[:T]->(:B)
      """
    When executing query:
      """
      MATCH (a1)-[r]->()
      WITH r, a1
        LIMIT 1
      OPTIONAL MATCH (a1)<-[r]-(b2)
      RETURN a1, r, b2
      """
    Then the result should be, in any order:
      | a1   | r    | b2   |
      | (:A) | [:T] | null |
    And no side effects

  @skip
  Scenario: [5] Optionally matching relationship with a relationship that is already bound
    Given an empty graph
    And having executed:
      """
      CREATE (:A)-[:T]->(:B)
      """
    When executing query:
      """
      MATCH ()-[r]->()
      WITH r
        LIMIT 1
      OPTIONAL MATCH (a2)-[r]->(b2)
      RETURN a2, r, b2
      """
    Then the result should be, in any order:
      | a2   | r    | b2   |
      | (:A) | [:T] | (:B) |
    And no side effects

  @skip
  Scenario: [6] Optionally matching relationship with a relationship and node that are both already bound
    Given an empty graph
    And having executed:
      """
      CREATE (:A)-[:T]->(:B)
      """
    When executing query:
      """
      MATCH (a1)-[r]->()
      WITH r, a1
        LIMIT 1
      OPTIONAL MATCH (a1)-[r]->(b2)
      RETURN a1, r, b2
      """
    Then the result should be, in any order:
      | a1   | r    | b2   |
      | (:A) | [:T] | (:B) |
    And no side effects

  @skip
  Scenario: [7] MATCH with OPTIONAL MATCH in longer pattern
    Given an empty graph
    And having executed:
      """
      CREATE (a {name: 'A'}), (b {name: 'B'}), (c {name: 'C'})
      CREATE (a)-[:KNOWS]->(b),
             (b)-[:KNOWS]->(c)
      """
    When executing query:
      """
      MATCH (a {name: 'A'})
      OPTIONAL MATCH (a)-[:KNOWS]->()-[:KNOWS]->(foo)
      RETURN foo
      """
    Then the result should be, in any order:
      | foo           |
      | ({name: 'C'}) |
    And no side effects

  @skip
  Scenario: [8] Longer pattern with bound nodes without matches
    Given an empty graph
    And having executed:
      """
      CREATE (s:Single), (a:A {num: 42}),
             (b:B {num: 46}), (c:C)
      CREATE (s)-[:REL]->(a),
             (s)-[:REL]->(b),
             (a)-[:REL]->(c),
             (b)-[:LOOP]->(b)
      """
    When executing query:
      """
      MATCH (a:A), (c:C)
      OPTIONAL MATCH (a)-->(b)-->(c)
      RETURN b
      """
    Then the result should be, in any order:
      | b    |
      | null |
    And no side effects

  @skip
  Scenario: [9] Longer pattern with bound nodes
    Given an empty graph
    And having executed:
      """
      CREATE (s:Single), (a:A {num: 42}),
             (b:B {num: 46}), (c:C)
      CREATE (s)-[:REL]->(a),
             (s)-[:REL]->(b),
             (a)-[:REL]->(c),
             (b)-[:LOOP]->(b)
      """
    When executing query:
      """
      MATCH (a:Single), (c:C)
      OPTIONAL MATCH (a)-->(b)-->(c)
      RETURN b
      """
    Then the result should be, in any order:
      | b              |
      | (:A {num: 42}) |
    And no side effects

  @skip
  Scenario: [10] Optionally matching from null nodes should return null
    Given an empty graph
    When executing query:
      """
      OPTIONAL MATCH (a)
      WITH a
      OPTIONAL MATCH (a)-->(b)
      RETURN b
      """
    Then the result should be, in any order:
      | b    |
      | null |
    And no side effects

  @skip
  Scenario: [11] Return two subgraphs with bound undirected relationship and optional relationship
    Given an empty graph
    And having executed:
      """
      CREATE (a:A {num: 1})-[:REL {name: 'r1'}]->(b:B {num: 2})-[:REL {name: 'r2'}]->(c:C {num: 3})
      """
    When executing query:
      """
      MATCH (a)-[r {name: 'r1'}]-(b)
      OPTIONAL MATCH (b)-[r2]-(c)
      WHERE r <> r2
      RETURN a, b, c
      """
    Then the result should be, in any order:
      | a             | b             | c             |
      | (:A {num: 1}) | (:B {num: 2}) | (:C {num: 3}) |
      | (:B {num: 2}) | (:A {num: 1}) | null          |
    And no side effects

  @skip
  Scenario: [12] Variable length optional relationships
    Given an empty graph
    And having executed:
      """
      CREATE (s:Single), (a:A {num: 42}),
             (b:B {num: 46}), (c:C)
      CREATE (s)-[:REL]->(a),
             (s)-[:REL]->(b),
             (a)-[:REL]->(c),
             (b)-[:LOOP]->(b)
      """
    When executing query:
      """
      MATCH (a:Single)
      OPTIONAL MATCH (a)-[*]->(b)
      RETURN b
      """
    Then the result should be, in any order:
      | b              |
      | (:A {num: 42}) |
      | (:B {num: 46}) |
      | (:B {num: 46}) |
      | (:C)           |
    And no side effects

  @skip
  Scenario: [13] Variable length optional relationships with bound nodes
    Given an empty graph
    And having executed:
      """
      CREATE (s:Single), (a:A {num: 42}),
             (b:B {num: 46}), (c:C)
      CREATE (s)-[:REL]->(a),
             (s)-[:REL]->(b),
             (a)-[:REL]->(c),
             (b)-[:LOOP]->(b)
      """
    When executing query:
      """
      MATCH (a:Single), (x:C)
      OPTIONAL MATCH (a)-[*]->(x)
      RETURN x
      """
    Then the result should be, in any order:
      | x    |
      | (:C) |
    And no side effects

  @skip
  Scenario: [14] Variable length optional relationships with length predicates
    Given an empty graph
    And having executed:
      """
      CREATE (s:Single), (a:A {num: 42}),
             (b:B {num: 46}), (c:C)
      CREATE (s)-[:REL]->(a),
             (s)-[:REL]->(b),
             (a)-[:REL]->(c),
             (b)-[:LOOP]->(b)
      """
    When executing query:
      """
      MATCH (a:Single)
      OPTIONAL MATCH (a)-[*3..]-(b)
      RETURN b
      """
    Then the result should be, in any order:
      | b    |
      | null |
    And no side effects

  @skip
  Scenario: [15] Variable length patterns and nulls
    Given an empty graph
    And having executed:
      """
      CREATE (a:A), (b:B)
      """
    When executing query:
      """
      MATCH (a:A)
      OPTIONAL MATCH (a)-[:FOO]->(b:B)
      OPTIONAL MATCH (b)<-[:BAR*]-(c:B)
      RETURN a, b, c
      """
    Then the result should be, in any order:
      | a    | b    | c    |
      | (:A) | null | null |
    And no side effects

  @skip
  Scenario: [16] Optionally matching named paths - null result
    Given an empty graph
    And having executed:
      """
      CREATE (s:Single), (a:A {num: 42}),
             (b:B {num: 46}), (c:C)
      CREATE (s)-[:REL]->(a),
             (s)-[:REL]->(b),
             (a)-[:REL]->(c),
             (b)-[:LOOP]->(b)
      """
    When executing query:
      """
      MATCH (a:A)
      OPTIONAL MATCH p = (a)-[:X]->(b)
      RETURN p
      """
    Then the result should be, in any order:
      | p    |
      | null |
    And no side effects

  @skip
  Scenario: [17] Optionally matching named paths - existing result
    Given an empty graph
    And having executed:
      """
      CREATE (a {name: 'A'}), (b {name: 'B'}), (c {name: 'C'})
      CREATE (a)-[:X]->(b)
      """
    When executing query:
      """
      MATCH (a {name: 'A'}), (x)
      WHERE x.name IN ['B', 'C']
      OPTIONAL MATCH p = (a)-->(x)
      RETURN x, p
      """
    Then the result should be, in any order:
      | x             | p                                   |
      | ({name: 'B'}) | <({name: 'A'})-[:X]->({name: 'B'})> |
      | ({name: 'C'}) | null                                |
    And no side effects

  @skip
  Scenario: [18] Named paths inside optional matches with node predicates
    Given an empty graph
    And having executed:
      """
      CREATE (s:Single), (a:A {num: 42}),
             (b:B {num: 46}), (c:C)
      CREATE (s)-[:REL]->(a),
             (s)-[:REL]->(b),
             (a)-[:REL]->(c),
             (b)-[:LOOP]->(b)
      """
    When executing query:
      """
      MATCH (a:A), (b:B)
      OPTIONAL MATCH p = (a)-[:X]->(b)
      RETURN p
      """
    Then the result should be, in any order:
      | p    |
      | null |
    And no side effects

  @skip
  Scenario: [19] Optionally matching named paths with single and variable length patterns
    Given an empty graph
    And having executed:
      """
      CREATE (a {name: 'A'}), (b {name: 'B'})
      CREATE (a)-[:X]->(b)
      """
    When executing query:
      """
      MATCH (a {name: 'A'})
      OPTIONAL MATCH p = (a)-->(b)-[*]->(c)
      RETURN p
      """
    Then the result should be, in any order:
      | p    |
      | null |
    And no side effects

  @skip
  Scenario: [20] Variable length optional relationships with bound nodes, no matches
    Given an empty graph
    And having executed:
      """
      CREATE (s:Single), (a:A {num: 42}),
             (b:B {num: 46}), (c:C)
      CREATE (s)-[:REL]->(a),
             (s)-[:REL]->(b),
             (a)-[:REL]->(c),
             (b)-[:LOOP]->(b)
      """
    When executing query:
      """
      MATCH (a:A), (b:B)
      OPTIONAL MATCH p = (a)-[*]->(b)
      RETURN p
      """
    Then the result should be, in any order:
      | p    |
      | null |
    And no side effects

  @skip
  Scenario: [21] Handling optional matches between nulls
    Given an empty graph
    And having executed:
      """
      CREATE (s:Single), (a:A {num: 42}),
             (b:B {num: 46}), (c:C)
      CREATE (s)-[:REL]->(a),
             (s)-[:REL]->(b),
             (a)-[:REL]->(c),
             (b)-[:LOOP]->(b)
      """
    When executing query:
      """
      OPTIONAL MATCH (a:NotThere)
      OPTIONAL MATCH (b:NotThere)
      WITH a, b
      OPTIONAL MATCH (b)-[r:NOR_THIS]->(a)
      RETURN a, b, r
      """
    Then the result should be, in any order:
      | a    | b    | r    |
      | null | null | null |
    And no side effects

  @skip
  Scenario: [22] MATCH after OPTIONAL MATCH
    Given an empty graph
    And having executed:
      """
      CREATE (s:Single), (a:A {num: 42}),
             (b:B {num: 46}), (c:C)
      CREATE (s)-[:REL]->(a),
             (s)-[:REL]->(b),
             (a)-[:REL]->(c),
             (b)-[:LOOP]->(b)
      """
    When executing query:
      """
      MATCH (a:Single)
      OPTIONAL MATCH (a)-->(b:NonExistent)
      OPTIONAL MATCH (a)-->(c:NonExistent)
      WITH coalesce(b, c) AS x
      MATCH (x)-->(d)
      RETURN d
      """
    Then the result should be, in any order:
      | d |
    And no side effects

  @skip
  Scenario: [23] OPTIONAL MATCH with labels on the optional end node
    Given an empty graph
    And having executed:
      """
      CREATE (:X), (x:X), (y1:Y), (y2:Y:Z)
      CREATE (x)-[:REL]->(y1),
             (x)-[:REL]->(y2)
      """
    When executing query:
      """
      MATCH (a:X)
      OPTIONAL MATCH (a)-->(b:Y)
      RETURN b
      """
    Then the result should be, in any order:
      | b      |
      | null   |
      | (:Y)   |
      | (:Y:Z) |
    And no side effects

  @skip
  Scenario: [24] Optionally matching self-loops
    Given an empty graph
    And having executed:
      """
      CREATE (s:Single), (a:A {num: 42}),
             (b:B {num: 46}), (c:C)
      CREATE (s)-[:REL]->(a),
             (s)-[:REL]->(b),
             (a)-[:REL]->(c),
             (b)-[:LOOP]->(b)
      """
    When executing query:
      """
      MATCH (a:B)
      OPTIONAL MATCH (a)-[r]-(a)
      RETURN r
      """
    Then the result should be, in any order:
      | r       |
      | [:LOOP] |
    And no side effects

  @skip
  Scenario: [25] Optionally matching self-loops without matches
    Given an empty graph
    And having executed:
      """
      CREATE (s:Single), (a:A {num: 42}),
             (b:B {num: 46}), (c:C)
      CREATE (s)-[:REL]->(a),
             (s)-[:REL]->(b),
             (a)-[:REL]->(c),
             (b)-[:LOOP]->(b)
      """
    When executing query:
      """
      MATCH (a)
      WHERE NOT (a:B)
      OPTIONAL MATCH (a)-[r]->(a)
      RETURN r
      """
    Then the result should be, in any order:
      | r    |
      | null |
      | null |
      | null |
    And no side effects

  @skip
  Scenario: [26] Handling correlated optional matches; first does not match implies second does not match
    Given an empty graph
    And having executed:
      """
      CREATE (s:Single), (a:A {num: 42}),
             (b:B {num: 46}), (c:C)
      CREATE (s)-[:REL]->(a),
             (s)-[:REL]->(b),
             (a)-[:REL]->(c),
             (b)-[:LOOP]->(b)
      """
    When executing query:
      """
      MATCH (a:A), (b:B)
      OPTIONAL MATCH (a)-->(x)
      OPTIONAL MATCH (x)-[r]->(b)
      RETURN x, r
      """
    Then the result should be, in any order:
      | x    | r    |
      | (:C) | null |
    And no side effects

  @skip
  Scenario: [27] Handling optional matches between optionally matched entities
    Given an empty graph
    And having executed:
      """
      CREATE (s:Single), (a:A {num: 42}),
             (b:B {num: 46}), (c:C)
      CREATE (s)-[:REL]->(a),
             (s)-[:REL]->(b),
             (a)-[:REL]->(c),
             (b)-[:LOOP]->(b)
      """
    When executing query:
      """
      OPTIONAL MATCH (a:NotThere)
      WITH a
      MATCH (b:B)
      WITH a, b
      OPTIONAL MATCH (b)-[r:NOR_THIS]->(a)
      RETURN a, b, r
      """
    Then the result should be, in any order:
      | a    | b              | r    |
      | null | (:B {num: 46}) | null |
    And no side effects

  @skip
  Scenario: [28] Handling optional matches with inline label predicate
    Given an empty graph
    And having executed:
      """
      CREATE (s:Single), (a:A {num: 42}),
             (b:B {num: 46}), (c:C)
      CREATE (s)-[:REL]->(a),
             (s)-[:REL]->(b),
             (a)-[:REL]->(c),
             (b)-[:LOOP]->(b)
      """
    When executing query:
      """
      MATCH (n:Single)
      OPTIONAL MATCH (n)-[r]-(m:NonExistent)
      RETURN r
      """
    Then the result should be, in any order:
      | r    |
      | null |
    And no side effects

  @skip
  Scenario: [29] Satisfies the open world assumption, relationships between same nodes
    Given an empty graph
    And having executed:
      """
      CREATE (a:Player), (b:Team)
      CREATE (a)-[:PLAYS_FOR]->(b),
             (a)-[:SUPPORTS]->(b)
      """
    When executing query:
      """
      MATCH (p:Player)-[:PLAYS_FOR]->(team:Team)
      OPTIONAL MATCH (p)-[s:SUPPORTS]->(team)
      RETURN count(*) AS matches, s IS NULL AS optMatch
      """
    Then the result should be, in any order:
      | matches | optMatch |
      | 1       | false    |
    And no side effects

  @skip
  Scenario: [30] Satisfies the open world assumption, single relationship
    Given an empty graph
    And having executed:
      """
      CREATE (a:Player), (b:Team)
      CREATE (a)-[:PLAYS_FOR]->(b)
      """
    When executing query:
      """
      MATCH (p:Player)-[:PLAYS_FOR]->(team:Team)
      OPTIONAL MATCH (p)-[s:SUPPORTS]->(team)
      RETURN count(*) AS matches, s IS NULL AS optMatch
      """
    Then the result should be, in any order:
      | matches | optMatch |
      | 1       | true     |
    And no side effects

  @skip
  Scenario: [31] Satisfies the open world assumption, relationships between different nodes
    Given an empty graph
    And having executed:
      """
      CREATE (a:Player), (b:Team), (c:Team)
      CREATE (a)-[:PLAYS_FOR]->(b),
             (a)-[:SUPPORTS]->(c)
      """
    When executing query:
      """
      MATCH (p:Player)-[:PLAYS_FOR]->(team:Team)
      OPTIONAL MATCH (p)-[s:SUPPORTS]->(team)
      RETURN count(*) AS matches, s IS NULL AS optMatch
      """
    Then the result should be, in any order:
      | matches | optMatch |
      | 1       | true     |
    And no side effects
