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

Feature: With1 - Forward single variable
  # correctly forward of values according to their type, no other effects

  Scenario: [1] Forwarind a node variable 1
    Given an empty graph
    And having executed:
      """
      CREATE (:A)-[:REL]->(:B)
      """
    When executing query:
      """
      MATCH (a:A)
      WITH a
      MATCH (a)-->(b)
      RETURN *
      """
    Then the result should be, in any order:
      | a    | b    |
      | (:A) | (:B) |
    And no side effects

  Scenario: [2] Forwarind a node variable 2
    Given an empty graph
    And having executed:
      """
      CREATE (:A)-[:REL]->(:B)
      CREATE (:X)
      """
    When executing query:
      """
      MATCH (a:A)
      WITH a
      MATCH (x:X), (a)-->(b)
      RETURN *
      """
    Then the result should be, in any order:
      | a    | b    | x    |
      | (:A) | (:B) | (:X) |
    And no side effects

  
  Scenario: [3] Forwarding a relationship variable
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:T1]->(:X),
             ()-[:T2]->(:X),
             ()-[:T3]->()
      """
    When executing query:
      """
      MATCH ()-[r1]->(:X)
      WITH r1 AS r2
      MATCH ()-[r2]->()
      RETURN r2 AS rel
      """
    Then the result should be, in any order:
      | rel   |
      | [:T1] |
      | [:T2] |
    And no side effects

  Scenario: [4] Forwarding a path variable
    Given an empty graph
    And having executed:
      """
      CREATE ()
      """
    When executing query:
      """
      MATCH p = (a)
      WITH p
      RETURN p
      """
    Then the result should be, in any order:
      | p    |
      | <()> |
    And no side effects

  Scenario: [5] Forwarding null
    Given an empty graph
    When executing query:
      """
      OPTIONAL MATCH (a:Start)
      WITH a
      MATCH (a)-->(b)
      RETURN *
      """
    Then the result should be, in any order:
      | a | b |
    And no side effects

  Scenario: [6] Forwarind a node variable possibly null
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
      OPTIONAL MATCH (a:A)
      WITH a AS a
      MATCH (b:B)
      RETURN a, b
      """
    Then the result should be, in any order:
      | a              | b              |
      | (:A {num: 42}) | (:B {num: 46}) |
    And no side effects
