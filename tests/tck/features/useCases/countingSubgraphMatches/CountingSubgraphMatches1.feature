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

Feature: CountingSubgraphMatches1 - Matching subgraph patterns and count the number of matches

  Scenario: [1] Undirected match in self-relationship graph, count
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:LOOP]->(a)
      """
    When executing query:
      """
      MATCH ()--()
      RETURN count(*)
      """
    Then the result should be, in any order:
      | count(*) |
      | 1        |
    And no side effects

  Scenario: [2] Undirected match of self-relationship in self-relationship graph, count
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:LOOP]->(a)
      """
    When executing query:
      """
      MATCH (n)--(n)
      RETURN count(*)
      """
    Then the result should be, in any order:
      | count(*) |
      | 1        |
    And no side effects

  Scenario: [3] Undirected match on simple relationship graph, count
    Given an empty graph
    And having executed:
      """
      CREATE (:A)-[:LOOP]->(:B)
      """
    When executing query:
      """
      MATCH ()--()
      RETURN count(*)
      """
    Then the result should be, in any order:
      | count(*) |
      | 2        |
    And no side effects

  Scenario: [4] Directed match on self-relationship graph, count
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:LOOP]->(a)
      """
    When executing query:
      """
      MATCH ()-->()
      RETURN count(*)
      """
    Then the result should be, in any order:
      | count(*) |
      | 1        |
    And no side effects

  Scenario: [5] Directed match of self-relationship on self-relationship graph, count
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:LOOP]->(a)
      """
    When executing query:
      """
      MATCH (n)-->(n)
      RETURN count(*)
      """
    Then the result should be, in any order:
      | count(*) |
      | 1        |
    And no side effects

  Scenario: [6] Counting undirected self-relationships in self-relationship graph
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:LOOP]->(a)
      """
    When executing query:
      """
      MATCH (n)-[r]-(n)
      RETURN count(r)
      """
    Then the result should be, in any order:
      | count(r) |
      | 1        |
    And no side effects

  Scenario: [7] Counting distinct undirected self-relationships in self-relationship graph
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:LOOP]->(a)
      """
    When executing query:
      """
      MATCH (n)-[r]-(n)
      RETURN count(DISTINCT r)
      """
    Then the result should be, in any order:
      | count(DISTINCT r) |
      | 1                 |
    And no side effects

  Scenario: [8] Directed match of a simple relationship, count
    Given an empty graph
    And having executed:
      """
      CREATE (:A)-[:LOOP]->(:B)
      """
    When executing query:
      """
      MATCH ()-->()
      RETURN count(*)
      """
    Then the result should be, in any order:
      | count(*) |
      | 1        |
    And no side effects

  Scenario: [9] Counting directed self-relationships
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:LOOP]->(a),
             ()-[:T]->()
      """
    When executing query:
      """
      MATCH (n)-[r]->(n)
      RETURN count(r)
      """
    Then the result should be, in any order:
      | count(r) |
      | 1        |
    And no side effects

  @skip
  Scenario: [10] Mixing directed and undirected pattern parts with self-relationship, count
    Given an empty graph
    And having executed:
      """
      CREATE (:A)-[:T1]->(l:Looper),
             (l)-[:LOOP]->(l),
             (l)-[:T2]->(:B)
      """
    When executing query:
      """
      MATCH (:A)-->()--()
      RETURN count(*)
      """
    Then the result should be, in any order:
      | count(*) |
      | 2        |
    And no side effects

  @skip
  Scenario: [11] Mixing directed and undirected pattern parts with self-relationship, undirected count
    Given an empty graph
    And having executed:
      """
      CREATE (:A)-[:T1]->(l:Looper),
             (l)-[:LOOP]->(l),
             (l)-[:T2]->(:B)
      """
    When executing query:
      """
      MATCH ()-[]-()-[]-()
      RETURN count(*)
      """
    Then the result should be, in any order:
      | count(*) |
      | 6        |
    And no side effects
