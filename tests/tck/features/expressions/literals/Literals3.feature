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

Feature: Literals3 - Hexadecimal integer

  Scenario: [1] Return a short positive hexadecimal integer
    Given any graph
    When executing query:
      """
      RETURN 0x1 AS literal
      """
    Then the result should be, in any order:
      | literal |
      | 1       |
    And no side effects

  Scenario: [2] Return a long positive hexadecimal integer
    Given any graph
    When executing query:
      """
      RETURN 0x162CD4F6 AS literal
      """
    Then the result should be, in any order:
      | literal    |
      | 372036854  |
    And no side effects

  Scenario: [3] Return the largest hexadecimal integer
    Given any graph
    When executing query:
      """
      RETURN 0x7FFFFFFFFFFFFFFF AS literal
      """
    Then the result should be, in any order:
      | literal              |
      | 9223372036854775807  |
    And no side effects

  Scenario: [4] Return a positive hexadecimal zero
    Given any graph
    When executing query:
      """
      RETURN 0x0 AS literal
      """
    Then the result should be, in any order:
      | literal |
      | 0       |
    And no side effects

  Scenario: [5] Return a negative hexadecimal zero
    Given any graph
    When executing query:
      """
      RETURN -0x0 AS literal
      """
    Then the result should be, in any order:
      | literal |
      | 0       |
    And no side effects

  Scenario: [6] Return a short negative hexadecimal integer
    Given any graph
    When executing query:
      """
      RETURN -0x1 AS literal
      """
    Then the result should be, in any order:
      | literal |
      | -1      |
    And no side effects

  Scenario: [7] Return a long negative hexadecimal integer
    Given any graph
    When executing query:
      """
      RETURN -0x162CD4F6 AS literal
      """
    Then the result should be, in any order:
      | literal    |
      | -372036854 |
    And no side effects

  Scenario: [8] Return the smallest hexadecimal integer
    Given any graph
    When executing query:
      """
      RETURN -0x8000000000000000 AS literal
      """
    Then the result should be, in any order:
      | literal              |
      | -9223372036854775808 |
    And no side effects

  Scenario: [9] Return a lower case hexadecimal integer
    Given any graph
    When executing query:
      """
      RETURN 0x1a2b3c4d5e6f7 AS literal
      """
    Then the result should be, in any order:
      | literal         |
      | 460367961908983 |
    And no side effects

  Scenario: [10] Return a upper case hexadecimal integer
    Given any graph
    When executing query:
      """
      RETURN 0x1A2B3C4D5E6F7 AS literal
      """
    Then the result should be, in any order:
      | literal         |
      | 460367961908983 |
    And no side effects

  Scenario: [11] Return a mixed case hexadecimal integer
    Given any graph
    When executing query:
      """
      RETURN 0x1A2b3c4D5E6f7 AS literal
      """
    Then the result should be, in any order:
      | literal         |
      | 460367961908983 |
    And no side effects

  
  Scenario: [12] Fail on an incomplete hexadecimal integer
    Given any graph
    When executing query:
      """
      RETURN 0x AS literal
      """
    Then a SyntaxError should be raised at compile time: InvalidNumberLiteral

  
  Scenario: [13] Fail on an hexadecimal literal containing a lower case invalid alphanumeric character
    Given any graph
    When executing query:
      """
      RETURN 0x1A2b3j4D5E6f7 AS literal
      """
    Then a SyntaxError should be raised at compile time: InvalidNumberLiteral

  
  Scenario: [14] Fail on an hexadecimal literal containing a upper case invalid alphanumeric character
    Given any graph
    When executing query:
      """
      RETURN 0x1A2b3c4Z5E6f7 AS literal
      """
    Then a SyntaxError should be raised at compile time: InvalidNumberLiteral

### Error detail not necessarily recognizable
#  @NegativeTest
#  Scenario: [15] Fail oning an hexadecimal literal containing a invalid symbol character
#    Given any graph
#    When executing query:
#      """
#      RETURN 0x1A2b3c4#5E6f7 AS literal
#      """
#    Then a SyntaxError should be raised at compile time: InvalidNumberLiteral

  Scenario: [16] Fail on a too large hexadecimal integer
    Given any graph
    When executing query:
      """
      RETURN 0x8000000000000000 AS literal
      """
    Then a SyntaxError should be raised at compile time: IntegerOverflow

  Scenario: [17] Fail on a too small hexadecimal integer
    Given any graph
    When executing query:
      """
      RETURN -0x8000000000000001 AS literal
      """
    Then a SyntaxError should be raised at compile time: IntegerOverflow
