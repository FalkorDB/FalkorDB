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

Feature: WithWhere1 - Filter single variable

  Scenario: [1] Filter node with property predicate on a single variable with multiple bindings
    Given an empty graph
    And having executed:
      """
      CREATE ({name: 'A'}),
             ({name: 'B'}),
             ({name: 'C'})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
      WHERE a.name = 'B'
      RETURN a
      """
    Then the result should be, in any order:
      | a             |
      | ({name: 'B'}) |
    And no side effects

  Scenario: [2] Filter node with property predicate on a single variable with multiple distinct bindings
    Given an empty graph
    And having executed:
      """
      CREATE ({name2: 'A'}),
             ({name2: 'A'}),
             ({name2: 'B'})
      """
    When executing query:
      """
      MATCH (a)
      WITH DISTINCT a.name2 AS name
      WHERE a.name2 = 'B'
      RETURN *
      """
    Then the result should be, in any order:
      | name |
      | 'B'  |
    And no side effects

  @skip
  Scenario: [3] Filter for an unbound relationship variable
    Given an empty graph
    And having executed:
      """
      CREATE (a:A), (b:B {id: 1}), (:B {id: 2})
      CREATE (a)-[:T]->(b)
      """
    When executing query:
      """
      MATCH (a:A), (other:B)
      OPTIONAL MATCH (a)-[r]->(other)
      WITH other WHERE r IS NULL
      RETURN other
      """
    Then the result should be, in any order:
      | other        |
      | (:B {id: 2}) |
    And no side effects

  Scenario: [4] Filter for an unbound node variable
    Given an empty graph
    And having executed:
      """
      CREATE (a:A), (b:B {id: 1}), (:B {id: 2})
      CREATE (a)-[:T]->(b)
      """
    When executing query:
      """
      MATCH (other:B)
      OPTIONAL MATCH (a)-[r]->(other)
      WITH other WHERE a IS NULL
      RETURN other
      """
    Then the result should be, in any order:
      | other        |
      | (:B {id: 2}) |
    And no side effects
