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

Feature: Precedence3 - On list values

  Scenario: [1] List element access takes precedence over list appending
    Given an empty graph
    When executing query:
      """
      RETURN [[1], [2, 3], [4, 5]] + [5, [6, 7], [8, 9], 10][3] AS a,
             [[1], [2, 3], [4, 5]] + ([5, [6, 7], [8, 9], 10][3]) AS b,
             ([[1], [2, 3], [4, 5]] + [5, [6, 7], [8, 9], 10])[3] AS c
      """
    Then the result should be, in any order:
      | a                         | b                         | c |
      | [[1], [2, 3], [4, 5], 10] | [[1], [2, 3], [4, 5], 10] | 5 |
    And no side effects

  Scenario: [2] List element access takes precedence over list concatenation
    Given an empty graph
    When executing query:
      """
      RETURN [[1], [2, 3], [4, 5]] + [5, [6, 7], [8, 9], 10][2] AS a,
             [[1], [2, 3], [4, 5]] + ([5, [6, 7], [8, 9], 10][2]) AS b,
             ([[1], [2, 3], [4, 5]] + [5, [6, 7], [8, 9], 10])[2] AS c
      """
    Then the result should be, in any order:
      | a                           | b                           | c      |
      | [[1], [2, 3], [4, 5], 8, 9] | [[1], [2, 3], [4, 5], 8, 9] | [4, 5] |
    And no side effects

  Scenario: [3] List slicing takes precedence over list concatenation
    Given an empty graph
    When executing query:
      """
      RETURN [[1], [2, 3], [4, 5]] + [5, [6, 7], [8, 9], 10][1..3] AS a,
             [[1], [2, 3], [4, 5]] + ([5, [6, 7], [8, 9], 10][1..3]) AS b,
             ([[1], [2, 3], [4, 5]] + [5, [6, 7], [8, 9], 10])[1..3] AS c
      """
    Then the result should be, in any order:
      | a                                     | b                                     | c                |
      | [[1], [2, 3], [4, 5], [6, 7], [8, 9]] | [[1], [2, 3], [4, 5], [6, 7], [8, 9]] | [[2, 3], [4, 5]] |
    And no side effects

  Scenario: [4] List appending takes precedence over list element containment
    Given an empty graph
    When executing query:
      """
      RETURN [1]+2 IN [3]+4 AS a,
             ([1]+2) IN ([3]+4) AS b,
             [1]+(2 IN [3])+4 AS c
      """
    Then the result should be, in any order:
      | a     | b     | c             |
      | false | false | [1, false, 4] |
    And no side effects

  Scenario: [5] List concatenation takes precedence over list element containment
    Given an empty graph
    When executing query:
      """
      RETURN [1]+[2] IN [3]+[4] AS a,
             ([1]+[2]) IN ([3]+[4]) AS b,
             (([1]+[2]) IN [3])+[4] AS c,
             [1]+([2] IN [3])+[4] AS d
      """
    Then the result should be, in any order:
      | a     | b     | c          | d             |
      | false | false | [false, 4] | [1, false, 4] |
    And no side effects

  Scenario Outline: [6] List element containment takes precedence over comparison operator
    Given an empty graph
    When executing query:
      """
      RETURN [1, 2] <comp> [3, 4] IN [[3, 4], false] AS a,
             [1, 2] <comp> ([3, 4] IN [[3, 4], false]) AS b,
             ([1, 2] <comp> [3, 4]) IN [[3, 4], false] AS c
      """
    Then the result should be, in any order:
      | a       | b       | c       |
      | <right> | <right> | <wrong> |
    And no side effects

    Examples:
      | comp | right | wrong |
      | =    | false | true  |
      | <>   | true  | false |
      | <    | null  | false |
      | >    | null  | true  |
      | <=   | null  | false |
      | >=   | null  | true  |
