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

Feature: Quantifier12 - All quantifier invariants

  
  Scenario: [1] All quantifier is always false if the predicate is statically false and the list is not empty
    Given any graph
    When executing query:
      """
      WITH [1, null, true, 4.5, 'abc', false, '', [234, false], {a: null, b: true, c: 15.2}, {}, [], [null], [[{b: [null]}]]] AS inputList
      UNWIND inputList AS x
      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list
      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list
      UNWIND inputList AS x
      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list
      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list
      UNWIND inputList AS x
      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list
      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list
      WITH list WHERE size(list) > 0
      WITH all(x IN list WHERE false) AS result, count(*) AS cnt
      RETURN result
      """
    Then the result should be, in any order:
      | result |
      | false  |
    And no side effects

  
  Scenario: [2] All quantifier is always true if the predicate is statically true and the list is not empty
    Given any graph
    When executing query:
      """
      WITH [1, null, true, 4.5, 'abc', false, '', [234, false], {a: null, b: true, c: 15.2}, {}, [], [null], [[{b: [null]}]]] AS inputList
      UNWIND inputList AS x
      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list
      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list
      UNWIND inputList AS x
      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list
      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list
      UNWIND inputList AS x
      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list
      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list
      WITH list WHERE size(list) > 0
      WITH all(x IN list WHERE true) AS result, count(*) AS cnt
      RETURN result
      """
    Then the result should be, in any order:
      | result |
      | true   |
    And no side effects

  Scenario Outline: [3] All quantifier is always equal the none quantifier on the boolean negative of the predicate
    Given any graph
    When executing query:
      """
      WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS inputList
      UNWIND inputList AS x
      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list
      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list
      UNWIND inputList AS x
      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list
      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list
      UNWIND inputList AS x
      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list
      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list
      WITH all(x IN list WHERE <predicate>) = none(x IN list WHERE NOT (<predicate>)) AS result, count(*) AS cnt
      RETURN result
      """
    Then the result should be, in any order:
      | result |
      | true   |
    And no side effects

    Examples:
      | predicate |
      | x = 2     |
      | x % 2 = 0 |
      | x % 3 = 0 |
      | x < 7     |
      | x >= 3    |

  Scenario Outline: [4] All quantifier is always equal the boolean negative of the any quantifier on the boolean negative of the predicate
    Given any graph
    When executing query:
      """
      WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS inputList
      UNWIND inputList AS x
      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list
      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list
      UNWIND inputList AS x
      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list
      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list
      UNWIND inputList AS x
      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list
      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list
      WITH all(x IN list WHERE <predicate>) = (NOT any(x IN list WHERE NOT (<predicate>))) AS result, count(*) AS cnt
      RETURN result
      """
    Then the result should be, in any order:
      | result |
      | true   |
    And no side effects

    Examples:
      | predicate |
      | x = 2     |
      | x % 2 = 0 |
      | x % 3 = 0 |
      | x < 7     |
      | x >= 3    |

  Scenario Outline: [5] All quantifier is always equal whether the size of the list filtered with same the predicate is equal the size of the unfiltered list
    Given any graph
    When executing query:
      """
      UNWIND [{list: [2], fixed: true},
              {list: [6], fixed: true},
              {list: [7], fixed: true},
              {list: [1, 2, 3, 4, 5, 6, 7, 8, 9], fixed: false}] AS input
      WITH CASE WHEN input.fixed THEN input.list ELSE null END AS fixedList,
           CASE WHEN NOT input.fixed THEN input.list ELSE [1] END AS inputList
      UNWIND inputList AS x
      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list
      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list
      UNWIND inputList AS x
      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list
      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list
      UNWIND inputList AS x
      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list
      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list
      WITH coalesce(fixedList, list) AS list
      WITH all(x IN list WHERE <predicate>) = (size([x IN list WHERE <predicate> | x]) = size(list)) AS result, count(*) AS cnt
      RETURN result
      """
    Then the result should be, in any order:
      | result |
      | true   |
    And no side effects

    Examples:
      | predicate |
      | x = 2     |
      | x % 2 = 0 |
      | x % 3 = 0 |
      | x < 7     |
      | x >= 3    |
