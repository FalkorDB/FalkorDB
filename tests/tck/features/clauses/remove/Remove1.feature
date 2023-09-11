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

Feature: Remove1 - Remove a Property

  Scenario: [1] Remove a single node property
    Given an empty graph
    And having executed:
      """
      CREATE (:L {num: 42})
      """
    When executing query:
      """
      MATCH (n)
      REMOVE n.num
      RETURN n.num IS NOT NULL AS still_there
      """
    Then the result should be, in any order:
      | still_there |
      | false       |
    And the side effects should be:
      | -properties | 1 |

  Scenario: [2] Remove multiple node properties
    Given an empty graph
    And having executed:
      """
      CREATE (:L {num: 42, name: 'a', name2: 'B'})
      """
    When executing query:
      """
      MATCH (n)
      REMOVE n.num, n.name
      RETURN size(keys(n)) AS props
      """
    Then the result should be, in any order:
      | props |
      | 1     |
    And the side effects should be:
      | -properties | 2 |

  Scenario: [3] Remove a single relationship property
    Given an empty graph
    And having executed:
      """
      CREATE (a), (b), (a)-[:X {num: 42}]->(b)
      """
    When executing query:
      """
      MATCH ()-[r]->()
      REMOVE r.num
      RETURN r.num IS NOT NULL AS still_there
      """
    Then the result should be, in any order:
      | still_there |
      | false       |
    And the side effects should be:
      | -properties | 1 |

  Scenario: [4] Remove multiple relationship properties
    Given an empty graph
    And having executed:
      """
      CREATE (a), (b), (a)-[:X {num: 42, a: 'a', b: 'B'}]->(b)
      """
    When executing query:
      """
      MATCH ()-[r]->()
      REMOVE r.num, r.a
      RETURN size(keys(r)) AS props
      """
    Then the result should be, in any order:
      | props |
      | 1     |
    And the side effects should be:
      | -properties | 2 |

  Scenario: [5] Ignore null when removing property from a node
    Given an empty graph
    When executing query:
      """
      OPTIONAL MATCH (a:DoesNotExist)
      REMOVE a.num
      RETURN a
      """
    Then the result should be, in any order:
      | a    |
      | null |
    And no side effects

  Scenario: [6] Ignore null when removing property from a relationship
    Given an empty graph
    And having executed:
      """
      CREATE ({num: 42})
      """
    When executing query:
      """
      MATCH (n)
      OPTIONAL MATCH (n)-[r]->()
      REMOVE r.num
      RETURN n
      """
    Then the result should be, in any order:
      | n           |
      | ({num: 42}) |
    And no side effects

  Scenario: [7] Remove a missing node property
    Given an empty graph
    And having executed:
      """
      CREATE (), (), ()
      """
    When executing query:
      """
      MATCH (n)
      REMOVE n.num
      RETURN sum(size(keys(n))) AS totalNumberOfProps
      """
    Then the result should be, in any order:
      | totalNumberOfProps |
      | 0.0                |
    And no side effects
