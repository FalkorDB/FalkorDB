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

Feature: ReturnOrderBy1 - Order by a single variable (correct order of values according to their type)

  @skip
  Scenario: [1] ORDER BY should order booleans in the expected order
    Given an empty graph
    When executing query:
      """
      UNWIND [true, false] AS bools
      RETURN bools
      ORDER BY bools
      """
    Then the result should be, in order:
      | bools |
      | false |
      | true  |
    And no side effects

  @skip
  Scenario: [2] ORDER BY DESC should order booleans in the expected order
    Given an empty graph
    When executing query:
      """
      UNWIND [true, false] AS bools
      RETURN bools
      ORDER BY bools DESC
      """
    Then the result should be, in order:
      | bools |
      | true  |
      | false |
    And no side effects

  @skip
  Scenario: [3] ORDER BY should order strings in the expected order
    Given an empty graph
    When executing query:
      """
      UNWIND ['.*', '', ' ', 'one'] AS strings
      RETURN strings
      ORDER BY strings
      """
    Then the result should be, in order:
      | strings |
      | ''      |
      | ' '     |
      | '.*'    |
      | 'one'   |
    And no side effects

  @skip
  Scenario: [4] ORDER BY DESC should order strings in the expected order
    Given an empty graph
    When executing query:
      """
      UNWIND ['.*', '', ' ', 'one'] AS strings
      RETURN strings
      ORDER BY strings DESC
      """
    Then the result should be, in order:
      | strings |
      | 'one'   |
      | '.*'    |
      | ' '     |
      | ''      |
    And no side effects

  @skip
  Scenario: [5] ORDER BY should order ints in the expected order
    Given an empty graph
    When executing query:
      """
      UNWIND [1, 3, 2] AS ints
      RETURN ints
      ORDER BY ints
      """
    Then the result should be, in order:
      | ints |
      | 1    |
      | 2    |
      | 3    |
    And no side effects

  @skip
  Scenario: [6] ORDER BY DESC should order ints in the expected order
    Given an empty graph
    When executing query:
      """
      UNWIND [1, 3, 2] AS ints
      RETURN ints
      ORDER BY ints DESC
      """
    Then the result should be, in order:
      | ints |
      | 3    |
      | 2    |
      | 1    |
    And no side effects

  @skip
  Scenario: [7] ORDER BY should order floats in the expected order
    Given an empty graph
    When executing query:
      """
      UNWIND [1.5, 1.3, 999.99] AS floats
      RETURN floats
      ORDER BY floats
      """
    Then the result should be, in order:
      | floats |
      | 1.3    |
      | 1.5    |
      | 999.99 |
    And no side effects

  @skip
  Scenario: [8] ORDER BY DESC should order floats in the expected order
    Given an empty graph
    When executing query:
      """
      UNWIND [1.5, 1.3, 999.99] AS floats
      RETURN floats
      ORDER BY floats DESC
      """
    Then the result should be, in order:
      | floats |
      | 999.99 |
      | 1.5    |
      | 1.3    |
    And no side effects

  @skip
  Scenario: [9] ORDER BY should order lists in the expected order
    Given an empty graph
    When executing query:
      """
      UNWIND [[], ['a'], ['a', 1], [1], [1, 'a'], [1, null], [null, 1], [null, 2]] AS lists
      RETURN lists
      ORDER BY lists
      """
    Then the result should be, in order:
      | lists     |
      | []        |
      | ['a']     |
      | ['a', 1]  |
      | [1]       |
      | [1, 'a']  |
      | [1, null] |
      | [null, 1] |
      | [null, 2] |
    And no side effects

  @skip
  Scenario: [10] ORDER BY DESC should order lists in the expected order
    Given an empty graph
    When executing query:
      """
      UNWIND [[], ['a'], ['a', 1], [1], [1, 'a'], [1, null], [null, 1], [null, 2]] AS lists
      RETURN lists
      ORDER BY lists DESC
      """
    Then the result should be, in order:
      | lists     |
      | [null, 2] |
      | [null, 1] |
      | [1, null] |
      | [1, 'a']  |
      | [1]       |
      | ['a', 1]  |
      | ['a']     |
      | []        |
    And no side effects

  @skip
  Scenario: [11] ORDER BY should order distinct types in the expected order
    Given an empty graph
    And having executed:
      """
      CREATE (:N)-[:REL]->()
      """
    When executing query:
      """
      MATCH p = (n:N)-[r:REL]->()
      UNWIND [n, r, p, 1.5, ['list'], 'text', null, false, 0.0 / 0.0, {a: 'map'}] AS types
      RETURN types
      ORDER BY types
      """
    Then the result should be, in order:
      | types             |
      | {a: 'map'}        |
      | (:N)              |
      | [:REL]            |
      | ['list']          |
      | <(:N)-[:REL]->()> |
      | 'text'            |
      | false             |
      | 1.5               |
      | NaN               |
      | null              |
    And no side effects

  @skip
  Scenario: [12] ORDER BY DESC should order distinct types in the expected order
    Given an empty graph
    And having executed:
      """
      CREATE (:N)-[:REL]->()
      """
    When executing query:
      """
      MATCH p = (n:N)-[r:REL]->()
      UNWIND [n, r, p, 1.5, ['list'], 'text', null, false, 0.0 / 0.0, {a: 'map'}] AS types
      RETURN types
      ORDER BY types DESC
      """
    Then the result should be, in order:
      | types             |
      | null              |
      | NaN               |
      | 1.5               |
      | false             |
      | 'text'            |
      | <(:N)-[:REL]->()> |
      | ['list']          |
      | [:REL]            |
      | (:N)              |
      | {a: 'map'}        |
    And no side effects
