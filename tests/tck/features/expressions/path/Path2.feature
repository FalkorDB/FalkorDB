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

Feature: Path2 - Relationships of a path

  
  Scenario: [1] Return relationships by fetching them from the path
    Given an empty graph
    And having executed:
      """
      CREATE (s:Start)-[:REL {num: 1}]->(b:B)-[:REL {num: 2}]->(c:C)
      """
    When executing query:
      """
      MATCH p = (a:Start)-[:REL*2..2]->(b)
      RETURN relationships(p)
      """
    Then the result should be, in any order:
      | relationships(p)                   |
      | [[:REL {num: 1}], [:REL {num: 2}]] |
    And no side effects


  
  Scenario: [2] Return relationships by fetching them from the path - starting from the end
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:REL {num: 1}]->(b:B)-[:REL {num: 2}]->(e:End)
      """
    When executing query:
      """
      MATCH p = (a)-[:REL*2..2]->(b:End)
      RETURN relationships(p)
      """
    Then the result should be, in any order:
      | relationships(p)                   |
      | [[:REL {num: 1}], [:REL {num: 2}]] |
    And no side effects

  Scenario: [3] `relationships()` on null path
    Given any graph
    When executing query:
      """
      WITH null AS a
      OPTIONAL MATCH p = (a)-[r]->()
      RETURN relationships(p), relationships(null)
      """
    Then the result should be, in any order:
      | relationships(p) | relationships(null) |
      | null             | null                |
    And no side effects
