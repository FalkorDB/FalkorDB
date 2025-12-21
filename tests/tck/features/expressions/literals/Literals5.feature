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

Feature: Literals5 - Float

  Scenario: [1] Return a short positive float
    Given any graph
    When executing query:
      """
      RETURN 1.0 AS literal
      """
    Then the result should be, in any order:
      | literal |
      | 1.0     |
    And no side effects

  Scenario: [2] Return a short positive float without integer digits
    Given any graph
    When executing query:
      """
      RETURN .1 AS literal
      """
    Then the result should be, in any order:
      | literal |
      | 0.1     |
    And no side effects

  
  Scenario: [3] Return a long positive float
    Given any graph
    When executing query:
      """
      RETURN 3985764.3405892687 AS literal
      """
    Then the result should be, in any order:
      | literal            |
      | 3985764.3405892686 |
    And no side effects

  Scenario: [4] Return a long positive float without integer digits
    Given any graph
    When executing query:
      """
      RETURN .3405892687 AS literal
      """
    Then the result should be, in any order:
      | literal      |
      | 0.3405892687 |
    And no side effects

  
  Scenario: [5] Return a very long positive float
    Given any graph
    When executing query:
      """
      RETURN 126354186523812635418263552340512384016094862983471987543918591348961093487896783409268730945879405123840160948812635418265234051238401609486298347198754391859134896109348789678340926873094587962983471812635265234051238401609486298348126354182652340512384016094862983471987543918591348961093487896783409218.0 AS literal
      """
    Then the result should be, in any order:
      | literal                   |
      | 1.2635418652381264e305 |
    And no side effects

  Scenario: [6] Return a very long positive float without integer digits
    Given any graph
    When executing query:
      """
      RETURN .00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001 AS literal
      """
    Then the result should be, in any order:
      | literal |
      | 1e-305  |
    And no side effects

  Scenario: [7] Return a positive zero float
    Given any graph
    When executing query:
      """
      RETURN 0.0 AS literal
      """
    Then the result should be, in any order:
      | literal |
      | 0.0     |
    And no side effects

  Scenario: [8] Return a positive zero float without integer digits
    Given any graph
    When executing query:
      """
      RETURN .0 AS literal
      """
    Then the result should be, in any order:
      | literal |
      | 0.0     |
    And no side effects

  Scenario: [9] Return a negative zero float
    Given any graph
    When executing query:
      """
      RETURN -0.0 AS literal
      """
    Then the result should be, in any order:
      | literal |
      | 0.0     |
    And no side effects

  Scenario: [10] Return a negative zero float without integer digits
    Given any graph
    When executing query:
      """
      RETURN -.0 AS literal
      """
    Then the result should be, in any order:
      | literal |
      | 0.0     |
    And no side effects

  
  Scenario: [11] Return a very long negative float
    Given any graph
    When executing query:
      """
      RETURN -126354186523812635418263552340512384016094862983471987543918591348961093487896783409268730945879405123840160948812635418265234051238401609486298347198754391859134896109348789678340926873094587962983471812635265234051238401609486298348126354182652340512384016094862983471987543918591348961093487896783409218.0 AS literal
      """
    Then the result should be, in any order:
      | literal                    |
      | -1.2635418652381264e305 |
    And no side effects

  Scenario: [12] Return a very long negative float without integer digits
    Given any graph
    When executing query:
      """
      RETURN -.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001 AS literal
      """
    Then the result should be, in any order:
      | literal |
      | -1e-305 |
    And no side effects

  Scenario: [13] Return a positive float with positive lower case exponent
    Given any graph
    When executing query:
      """
      RETURN 1e9 AS literal
      """
    Then the result should be, in any order:
      | literal      |
      | 1000000000.0 |
    And no side effects

  Scenario: [14] Return a positive float with positive upper case exponent
    Given any graph
    When executing query:
      """
      RETURN 1E9 AS literal
      """
    Then the result should be, in any order:
      | literal      |
      | 1000000000.0 |
    And no side effects

  Scenario: [15] Return a positive float with positive lower case exponent without integer digits
    Given any graph
    When executing query:
      """
      RETURN .1e9 AS literal
      """
    Then the result should be, in any order:
      | literal     |
      | 100000000.0 |
    And no side effects

  Scenario: [16] Return a positive float with negative lower case exponent
    Given any graph
    When executing query:
      """
      RETURN 1e-5 AS literal
      """
    Then the result should be, in any order:
      | literal |
      | 0.00001 |
    And no side effects

  
  Scenario: [17] Return a positive float with negative lower case exponent without integer digits
    Given any graph
    When executing query:
      """
      RETURN .1e-5 AS literal
      """
    Then the result should be, in any order:
      | literal  |
      | 0.000001 |
    And no side effects

  
  Scenario: [18] Return a positive float with negative upper case exponent without integer digits
    Given any graph
    When executing query:
      """
      RETURN .1E-5 AS literal
      """
    Then the result should be, in any order:
      | literal  |
      | 0.000001 |
    And no side effects

  Scenario: [19] Return a negative float in with positive lower case exponent
    Given any graph
    When executing query:
      """
      RETURN -1e9 AS literal
      """
    Then the result should be, in any order:
      | literal       |
      | -1000000000.0 |
    And no side effects

  Scenario: [20] Return a negative float in with positive upper case exponent
    Given any graph
    When executing query:
      """
      RETURN -1E9 AS literal
      """
    Then the result should be, in any order:
      | literal       |
      | -1000000000.0 |
    And no side effects

  Scenario: [21] Return a negative float with positive lower case exponent without integer digits
    Given any graph
    When executing query:
      """
      RETURN -.1e9 AS literal
      """
    Then the result should be, in any order:
      | literal      |
      | -100000000.0 |
    And no side effects

  Scenario: [22] Return a negative float with negative lower case exponent
    Given any graph
    When executing query:
      """
      RETURN -1e-5 AS literal
      """
    Then the result should be, in any order:
      | literal  |
      | -0.00001 |
    And no side effects

  
  Scenario: [23] Return a negative float with negative lower case exponent without integer digits
    Given any graph
    When executing query:
      """
      RETURN -.1e-5 AS literal
      """
    Then the result should be, in any order:
      | literal   |
      | -0.000001 |
    And no side effects

  
  Scenario: [24] Return a negative float with negative upper case exponent without integer digits
    Given any graph
    When executing query:
      """
      RETURN -.1E-5 AS literal
      """
    Then the result should be, in any order:
      | literal   |
      | -0.000001 |
    And no side effects

  Scenario: [25] Return a positive float with one integer digit and maximum positive exponent
    Given any graph
    When executing query:
      """
      RETURN 1e308 AS literal
      """
    Then the result should be, in any order:
      | literal |
      | 1e308   |
    And no side effects

  Scenario: [26] Return a positive float with nine integer digit and maximum positive exponent
    Given any graph
    When executing query:
      """
      RETURN 123456789e300 AS literal
      """
    Then the result should be, in any order:
      | literal        |
      | 1.23456789e308 |
    And no side effects

  Scenario: [27] Fail when float value is too large
    Given any graph
    When executing query:
      """
      RETURN 1.34E999
      """
    Then a SyntaxError should be raised at compile time: FloatingPointOverflow
