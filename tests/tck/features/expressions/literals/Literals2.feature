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

Feature: Literals2 - Decimal integer

  Scenario: [1] Return a short positive integer
    Given any graph
    When executing query:
      """
      RETURN 1 AS literal
      """
    Then the result should be, in any order:
      | literal |
      | 1       |
    And no side effects

  Scenario: [2] Return a long positive integer
    Given any graph
    When executing query:
      """
      RETURN 372036854 AS literal
      """
    Then the result should be, in any order:
      | literal    |
      | 372036854  |
    And no side effects

  Scenario: [3] Return the largest integer
    Given any graph
    When executing query:
      """
      RETURN 9223372036854775807 AS literal
      """
    Then the result should be, in any order:
      | literal              |
      | 9223372036854775807  |
    And no side effects

  Scenario: [4] Return a positive zero
    Given any graph
    When executing query:
      """
      RETURN 0 AS literal
      """
    Then the result should be, in any order:
      | literal |
      | 0       |
    And no side effects

  Scenario: [5] Return a negative zero
    Given any graph
    When executing query:
      """
      RETURN -0 AS literal
      """
    Then the result should be, in any order:
      | literal |
      | 0       |
    And no side effects

  Scenario: [6] Return a short negative integer
    Given any graph
    When executing query:
      """
      RETURN -1 AS literal
      """
    Then the result should be, in any order:
      | literal |
      | -1      |
    And no side effects

  Scenario: [7] Return a long negative integer
    Given any graph
    When executing query:
      """
      RETURN -372036854 AS literal
      """
    Then the result should be, in any order:
      | literal    |
      | -372036854 |
    And no side effects

  Scenario: [8] Return the smallest integer
    Given any graph
    When executing query:
      """
      RETURN -9223372036854775808 AS literal
      """
    Then the result should be, in any order:
      | literal              |
      | -9223372036854775808 |
    And no side effects

  Scenario: [9] Fail on a too large integer
    Given any graph
    When executing query:
      """
      RETURN 9223372036854775808 AS literal
      """
    Then a SyntaxError should be raised at compile time: IntegerOverflow

  Scenario: [10] Fail on a too small integer
    Given any graph
    When executing query:
      """
      RETURN -9223372036854775809 AS literal
      """
    Then a SyntaxError should be raised at compile time: IntegerOverflow

  
  Scenario: [11] Fail on an integer containing a alphabetic character
    Given any graph
    When executing query:
      """
      RETURN 9223372h54775808 AS literal
      """
    Then a SyntaxError should be raised at compile time: InvalidNumberLiteral

  
  Scenario: [12] Fail on an integer containing a invalid symbol character
    Given any graph
    When executing query:
      """
      RETURN 9223372#54775808 AS literal
      """
    Then a SyntaxError should be raised at compile time: UnexpectedSyntax
