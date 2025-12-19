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

Feature: Delete6 - Persistence of delete clause side effects
  Scenario: [1] Limiting to zero results after deleting nodes affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE (:N {num: 42})
      """
    When executing query:
      """
      MATCH (n:N)
      DELETE n
      RETURN 42 AS num
      LIMIT 0
      """
    Then the result should be, in any order:
      | num |
    And the side effects should be:
      | -nodes      | 1 |
      | -labels     | 1 |
      | -properties | 1 |
  Scenario: [2] Skipping all results after deleting nodes affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE (:N {num: 42})
      """
    When executing query:
      """
      MATCH (n:N)
      DELETE n
      RETURN 42 AS num
      SKIP 1
      """
    Then the result should be, in any order:
      | num |
    And the side effects should be:
      | -nodes      | 1 |
      | -labels     | 1 |
      | -properties | 1 |
  Scenario: [3] Skipping and limiting to a few results after deleting nodes affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE (:N {num: 1})
      CREATE (:N {num: 2})
      CREATE (:N {num: 3})
      CREATE (:N {num: 4})
      CREATE (:N {num: 5})
      """
    When executing query:
      """
      MATCH (n:N)
      DELETE n
      RETURN 42 AS num
      SKIP 2 LIMIT 2
      """
    Then the result should be, in any order:
      | num |
      | 42  |
      | 42  |
    And the side effects should be:
      | -nodes      | 5 |
      | -labels     | 1 |
      | -properties | 5 |
  Scenario: [4] Skipping zero results and limiting to all results after deleting nodes does not affect the result set nor the side effects
    Given an empty graph
    And having executed:
      """
      CREATE (:N {num: 1})
      CREATE (:N {num: 2})
      CREATE (:N {num: 3})
      CREATE (:N {num: 4})
      CREATE (:N {num: 5})
      """
    When executing query:
      """
      MATCH (n:N)
      DELETE n
      RETURN 42 AS num
      SKIP 0 LIMIT 5
      """
    Then the result should be, in any order:
      | num |
      | 42  |
      | 42  |
      | 42  |
      | 42  |
      | 42  |
    And the side effects should be:
      | -nodes      | 5 |
      | -labels     | 1 |
      | -properties | 5 |
  Scenario: [5] Filtering after deleting nodes affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE (:N {num: 1})
      CREATE (:N {num: 2})
      CREATE (:N {num: 3})
      CREATE (:N {num: 4})
      CREATE (:N {num: 5})
      """
    When executing query:
      """
      MATCH (n:N)
      WITH n, n.num AS num
      DELETE n
      WITH num
      WHERE num % 2 = 0
      RETURN num
      """
    Then the result should be, in any order:
      | num |
      | 2   |
      | 4   |
    And the side effects should be:
      | -nodes      | 5 |
      | -labels     | 1 |
      | -properties | 5 |
  Scenario: [6] Aggregating in `RETURN` after deleting nodes affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE (:N {num: 1})
      CREATE (:N {num: 2})
      CREATE (:N {num: 3})
      CREATE (:N {num: 4})
      CREATE (:N {num: 5})
      """
    When executing query:
      """
      MATCH (n:N)
      WITH n, n.num AS num
      DELETE n
      RETURN sum(num) AS sum
      """
    Then the result should be, in any order:
      | sum |
      | 15  |
    And the side effects should be:
      | -nodes      | 5 |
      | -labels     | 1 |
      | -properties | 5 |
  Scenario: [7] Aggregating in `WITH` after deleting nodes affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE (:N {num: 1})
      CREATE (:N {num: 2})
      CREATE (:N {num: 3})
      CREATE (:N {num: 4})
      CREATE (:N {num: 5})
      """
    When executing query:
      """
      MATCH (n:N)
      WITH n, n.num AS num
      DELETE n
      WITH sum(num) AS sum
      RETURN sum
      """
    Then the result should be, in any order:
      | sum |
      | 15  |
    And the side effects should be:
      | -nodes      | 5 |
      | -labels     | 1 |
      | -properties | 5 |
  Scenario: [8] Limiting to zero results after deleting relationships affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE ()-[r:R {num: 42}]->()
      """
    When executing query:
      """
      MATCH ()-[r:R]->()
      DELETE r
      RETURN 42 AS num
      LIMIT 0
      """
    Then the result should be, in any order:
      | num |
    And the side effects should be:
      | -relationships | 1 |
      | -properties    | 1 |
  Scenario: [9] Skipping all results after deleting relationships affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE ()-[r:R {num: 42}]->()
      """
    When executing query:
      """
      MATCH ()-[r:R]->()
      DELETE r
      RETURN 42 AS num
      SKIP 1
      """
    Then the result should be, in any order:
      | num |
    And the side effects should be:
      | -relationships | 1 |
      | -properties    | 1 |
  Scenario: [10] Skipping and limiting to a few results after deleting relationships affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:R {num: 1}]->()
      CREATE ()-[:R {num: 2}]->()
      CREATE ()-[:R {num: 3}]->()
      CREATE ()-[:R {num: 4}]->()
      CREATE ()-[:R {num: 5}]->()
      """
    When executing query:
      """
      MATCH ()-[r:R]->()
      DELETE r
      RETURN 42 AS num
      SKIP 2 LIMIT 2
      """
    Then the result should be, in any order:
      | num |
      | 42  |
      | 42  |
    And the side effects should be:
      | -relationships | 5 |
      | -properties    | 5 |
  Scenario: [11] Skipping zero result and limiting to all results after deleting relationships does not affect the result set nor the side effects
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:R {num: 1}]->()
      CREATE ()-[:R {num: 2}]->()
      CREATE ()-[:R {num: 3}]->()
      CREATE ()-[:R {num: 4}]->()
      CREATE ()-[:R {num: 5}]->()
      """
    When executing query:
      """
      MATCH ()-[r:R]->()
      DELETE r
      RETURN 42 AS num
      SKIP 0 LIMIT 5
      """
    Then the result should be, in any order:
      | num |
      | 42  |
      | 42  |
      | 42  |
      | 42  |
      | 42  |
    And the side effects should be:
      | -relationships | 5 |
      | -properties    | 5 |
  Scenario: [12] Filtering after deleting relationships affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:R {num: 1}]->()
      CREATE ()-[:R {num: 2}]->()
      CREATE ()-[:R {num: 3}]->()
      CREATE ()-[:R {num: 4}]->()
      CREATE ()-[:R {num: 5}]->()
      """
    When executing query:
      """
      MATCH ()-[r:R]->()
      WITH r, r.num AS num
      DELETE r
      WITH num
      WHERE num % 2 = 0
      RETURN num
      """
    Then the result should be, in any order:
      | num |
      | 2   |
      | 4   |
    And the side effects should be:
      | -relationships | 5 |
      | -properties    | 5 |
  Scenario: [13] Aggregating in `RETURN` after deleting relationships affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:R {num: 1}]->()
      CREATE ()-[:R {num: 2}]->()
      CREATE ()-[:R {num: 3}]->()
      CREATE ()-[:R {num: 4}]->()
      CREATE ()-[:R {num: 5}]->()
      """
    When executing query:
      """
      MATCH ()-[r:R]->()
      WITH r, r.num AS num
      DELETE r
      RETURN sum(num) AS sum
      """
    Then the result should be, in any order:
      | sum |
      | 15  |
    And the side effects should be:
      | -relationships | 5 |
      | -properties    | 5 |
  Scenario: [14] Aggregating in `WITH` after deleting relationships affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:R {num: 1}]->()
      CREATE ()-[:R {num: 2}]->()
      CREATE ()-[:R {num: 3}]->()
      CREATE ()-[:R {num: 4}]->()
      CREATE ()-[:R {num: 5}]->()
      """
    When executing query:
      """
      MATCH ()-[r:R]->()
      WITH r, r.num AS num
      DELETE r
      WITH sum(num) AS sum
      RETURN sum
      """
    Then the result should be, in any order:
      | sum |
      | 15  |
    And the side effects should be:
      | -relationships | 5 |
      | -properties    | 5 |
