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


#
# Copyright (c) 2015-2021 "Neo Technology,"
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

Feature: WithSkipLimit1 - Skip

  
  Scenario: [1] Handle dependencies across WITH with SKIP
    Given an empty graph
    And having executed:
      """
      CREATE (a {name: 'A', num: 0, id: 0}),
             ({name: 'B', num: a.id, id: 1}),
             ({name: 'C', num: 0, id: 2})
      """
    When executing query:
      """
      MATCH (a)
      WITH a.name AS property, a.num AS idToUse
        ORDER BY property
        SKIP 1
      MATCH (b)
      WHERE b.id = idToUse
      RETURN DISTINCT b
      """
    Then the result should be, in any order:
      | b                            |
      | ({name: 'A', num: 0, id: 0}) |
    And no side effects

  Scenario: [2] Ordering and skipping on aggregate
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:T1 {num: 3}]->(x:X),
             ()-[:T2 {num: 2}]->(x),
             ()-[:T3 {num: 1}]->(:Y)
      """
    When executing query:
      """
      MATCH ()-[r1]->(x)
      WITH x, sum(r1.num) AS c
        ORDER BY c SKIP 1
      RETURN x, c
      """
    Then the result should be, in any order:
      | x    | c   |
      | (:X) | 5.0 |
    And no side effects
