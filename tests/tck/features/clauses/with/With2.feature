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

Feature: With2 - Forward single expression
  # correctly projecting an expression, no other effects

  
  Scenario: [1] Forwarding a property to express a join
    Given an empty graph
    And having executed:
      """
      CREATE (a:End {num: 42, id: 0}),
             (:End {num: 3}),
             (:Begin {num: a.id})
      """
    When executing query:
      """
      MATCH (a:Begin)
      WITH a.num AS property
      MATCH (b)
      WHERE b.id = property
      RETURN b
      """
    Then the result should be, in any order:
      | b                       |
      | (:End {num: 42, id: 0}) |
    And no side effects

  Scenario: [2] Forwarding a nested map literal
    Given an empty graph
    When executing query:
      """
      WITH {name: {name2: 'baz'}} AS nestedMap
      RETURN nestedMap.name.name2
      """
    Then the result should be, in any order:
      | nestedMap.name.name2 |
      | 'baz'                |
    And no side effects
