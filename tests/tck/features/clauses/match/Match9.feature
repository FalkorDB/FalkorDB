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

Feature: Match9 - Match deprecated scenarios

  
  Scenario: [1] Variable length relationship variables are lists of relationships
    Given an empty graph
    And having executed:
      """
      CREATE (a), (b), (c)
      CREATE (a)-[:T]->(b)
      """
    When executing query:
      """
      MATCH ()-[r*0..1]-()
      RETURN last(r) AS l
      """
    Then the result should be, in any order:
      | l    |
      | [:T] |
      | [:T] |
      | null |
      | null |
      | null |
    And no side effects

  
  Scenario: [2] Return relationships by collecting them as a list - directed, one way
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:REL {num: 1}]->(b:B)-[:REL {num: 2}]->(e:End)
      """
    When executing query:
      """
      MATCH (a)-[r:REL*2..2]->(b:End)
      RETURN r
      """
    Then the result should be, in any order:
      | r                                  |
      | [[:REL {num: 1}], [:REL {num: 2}]] |
    And no side effects

  
  Scenario: [3] Return relationships by collecting them as a list - undirected, starting from two extremes
    Given an empty graph
    And having executed:
      """
      CREATE (a:End)-[:REL {num: 1}]->(b:B)-[:REL {num: 2}]->(c:End)
      """
    When executing query:
      """
      MATCH (a)-[r:REL*2..2]-(b:End)
      RETURN r
      """
    Then the result should be, in any order:
      | r                                |
      | [[:REL {num:1}], [:REL {num:2}]] |
      | [[:REL {num:2}], [:REL {num:1}]] |
    And no side effects

  
  Scenario: [4] Return relationships by collecting them as a list - undirected, starting from one extreme
    Given an empty graph
    And having executed:
      """
      CREATE (s:Start)-[:REL {num: 1}]->(b:B)-[:REL {num: 2}]->(c:C)
      """
    When executing query:
      """
      MATCH (a:Start)-[r:REL*2..2]-(b)
      RETURN r
      """
    Then the result should be, in any order:
      | r                                  |
      | [[:REL {num: 1}], [:REL {num: 2}]] |
    And no side effects

  
  Scenario: [5] Variable length pattern with label predicate on both sides
    Given an empty graph
    And having executed:
      """
      CREATE (a:Blue), (b:Red), (c:Green), (d:Yellow)
      CREATE (a)-[:T]->(b),
             (b)-[:T]->(c),
             (b)-[:T]->(d)
      """
    When executing query:
      """
      MATCH (a:Blue)-[r*]->(b:Green)
      RETURN count(r)
      """
    Then the result should be, in any order:
      | count(r) |
      | 1        |
    And no side effects

  Scenario: [6] Matching relationships into a list and matching variable length using the list, with bound nodes
    Given an empty graph
    And having executed:
      """
      CREATE (a:A), (b:B), (c:C)
      CREATE (a)-[:Y]->(b),
             (b)-[:Y]->(c)
      """
    When executing query:
      """
      MATCH (a)-[r1]->()-[r2]->(b)
      WITH [r1, r2] AS rs, a AS first, b AS second
        LIMIT 1
      MATCH (first)-[*]->(second)
      RETURN first, second
      """
    Then the result should be, in any order:
      | first | second |
      | (:A)  | (:C)   |
    And no side effects

  Scenario: [7] Matching relationships into a list and matching variable length using the list, with bound nodes, wrong direction
    Given an empty graph
    And having executed:
      """
      CREATE (a:A), (b:B), (c:C)
      CREATE (a)-[:Y]->(b),
             (b)-[:Y]->(c)
      """
    When executing query:
      """
      MATCH (a)-[r1]->()-[r2]->(b)
      WITH [r1, r2] AS rs, a AS second, b AS first
        LIMIT 1
      MATCH (first)-[*]->(second)
      RETURN first, second
      """
    Then the result should be, in any order:
      | first | second |
    And no side effects

  Scenario: [8] Variable length relationship in OPTIONAL MATCH
    Given an empty graph
    And having executed:
      """
      CREATE (:A), (:B)
      """
    When executing query:
      """
      MATCH (a:A), (b:B)
      OPTIONAL MATCH (a)-[r*]-(b)
      WHERE r IS NULL
        AND a <> b
      RETURN b
      """
    Then the result should be, in any order:
      | b    |
      | (:B) |
    And no side effects

  
  Scenario: [9] Optionally matching named paths with variable length patterns
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
      OPTIONAL MATCH p = (a)-[r*]->(x)
      RETURN r, x, p
      """
    Then the result should be, in any order:
      | r      | x             | p                                   |
      | [[:X]] | ({name: 'B'}) | <({name: 'A'})-[:X]->({name: 'B'})> |
      | null   | ({name: 'C'}) | null                                |
    And no side effects
