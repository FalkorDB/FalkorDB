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

Feature: WithWhere7 - Variable visibility under aliasing

  Scenario: [1] WHERE sees a variable bound before but not after WITH
    Given an empty graph
    And having executed:
      """
      CREATE ({name2: 'A'}),
             ({name2: 'B'}),
             ({name2: 'C'})
      """
    When executing query:
      """
      MATCH (a)
      WITH a.name2 AS name
      WHERE a.name2 = 'B'
      RETURN *
      """
    Then the result should be, in any order:
      | name |
      | 'B'  |
    And no side effects

  Scenario: [2] WHERE sees a variable bound after but not before WITH
    Given an empty graph
    And having executed:
      """
      CREATE ({name2: 'A'}),
             ({name2: 'B'}),
             ({name2: 'C'})
      """
    When executing query:
      """
      MATCH (a)
      WITH a.name2 AS name
      WHERE name = 'B'
      RETURN *
      """
    Then the result should be, in any order:
      | name |
      | 'B'  |
    And no side effects

  @skip
  Scenario: [3] WHERE sees both, variable bound before but not after WITH and variable bound after but not before WITH
    Given an empty graph
    And having executed:
      """
      CREATE ({name2: 'A'}),
             ({name2: 'B'}),
             ({name2: 'C'})
      """
    When executing query:
      """
      MATCH (a)
      WITH a.name2 AS name
      WHERE name = 'B' OR a.name2 = 'C'
      RETURN *
      """
    Then the result should be, in any order:
      | name |
      | 'B'  |
      | 'C'  |
    And no side effects
