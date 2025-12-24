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

Feature: Graph5 - Node and edge label expressions

  Scenario: [1] Single-labels expression on nodes
    Given an empty graph
    And having executed:
      """
      CREATE (:A:B:C), (:A:B), (:A:C), (:B:C),
             (:A), (:B), (:C), ()
      """
    When executing query:
      """
      MATCH (a)
      RETURN a, a:B AS result
      """
    Then the result should be, in any order:
      | a        | result |
      | (:A:B:C) | true   |
      | (:A:B)   | true   |
      | (:A:C)   | false  |
      | (:B:C)   | true   |
      | (:A)     | false  |
      | (:B)     | true   |
      | (:C)     | false  |
      | ()       | false  |
    And no side effects

  # This scenario does not work in Cypher. Although that is a little bit odd.
  @ignore StyleCheck
  
  Scenario: [2] Single-labels expression on relationships
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:T1]->(),
             ()-[:T2]->(),
             ()-[:t2]->(),
             (:T2)-[:T3]->(),
             ()-[:T4]->(:T2)
      """
    When executing query:
      """
      MATCH ()-[r]->()
      RETURN r, r:T2 AS result
      """
    Then the result should be, in any order:
      | r     | result |
      | [:T1] | false  |
      | [:T2] | true   |
      | [:t2] | false  |
      | [:T3] | false  |
      | [:T4] | false  |
    And no side effects

  Scenario: [3] Conjunctive labels expression on nodes
    Given an empty graph
    And having executed:
      """
      CREATE (:A:B:C), (:A:B), (:A:C), (:B:C),
             (:A), (:B), (:C), ()
      """
    When executing query:
      """
      MATCH (a)
      RETURN a, a:A:B AS result
      """
    Then the result should be, in any order:
      | a        | result |
      | (:A:B:C) | true   |
      | (:A:B)   | true   |
      | (:A:C)   | false  |
      | (:B:C)   | false  |
      | (:A)     | false  |
      | (:B)     | false  |
      | (:C)     | false  |
      | ()       | false  |
    And no side effects

  Scenario Outline: [4] Conjunctive labels expression on nodes with varying order and repeating labels
    Given an empty graph
    And having executed:
      """
      CREATE (:A:B), (:A:C), (:B:C),
             (:A), (:B), (:C), ()
      """
    When executing query:
      """
      MATCH (a)
      WHERE a<labelexp>
      RETURN a
      """
    Then the result should be, in any order:
      | a        |
      | <result> |
    And no side effects

    Examples:
      | labelexp | result |
      | :A:C     | (:A:C) |
      | :C:A     | (:A:C) |
      | :A:C:A   | (:A:C) |
      | :C:C:A   | (:A:C) |
      | :C:A:A:C | (:A:C) |

  Scenario: [5] Label expression on null
    Given an empty graph
    And having executed:
      """
      CREATE (s:Single)
      """
    When executing query:
      """
      MATCH (n:Single)
      OPTIONAL MATCH (n)-[r:TYPE]-(m)
      RETURN m:TYPE
      """
    Then the result should be, in any order:
      | m:TYPE |
      | null   |
    And no side effects
