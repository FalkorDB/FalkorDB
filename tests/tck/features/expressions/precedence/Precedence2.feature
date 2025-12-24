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

Feature: Precedence2 - On numeric values

  Scenario Outline: [1] Numeric multiplicative operations takes precedence over numeric additive operations
    Given an empty graph
    When executing query:
      """
      RETURN 4 <mult1> 2 <add> 3 <mult2> 2 AS a,
             4 <mult1> 2 <add> (3 <mult2> 2) AS b,
             4 <mult1> (2 <add> 3) <mult2> 2 AS c
      """
    Then the result should be, in any order:
      | a       | b       | c       |
      | <right> | <right> | <wrong> |
    And no side effects

    Examples:
      | mult1 | add | mult2 | right | wrong |
      | *     | +   | *     | 14    | 40    |
      | *     | +   | /     | 9     | 10    |
      | *     | +   | %     | 9     | 0     |
      | *     | -   | *     | 2     | -8    |
      | *     | -   | /     | 7     | -2    |
      | *     | -   | %     | 7     | 0     |
      | /     | +   | *     | 8     | 0     |
      | /     | +   | /     | 3     | 0     |
      | /     | +   | %     | 3     | 0     |
      | /     | -   | *     | -4    | -8    |
      | /     | -   | /     | 1     | -2    |
      | /     | -   | %     | 1     | 0     |
      | %     | +   | *     | 6     | 8     |
      | %     | +   | /     | 1     | 2     |
      | %     | +   | %     | 1     | 0     |
      | %     | -   | *     | -6    | 0     |
      | %     | -   | /     | -1    | 0     |
      | %     | -   | %     | -1    | 0     |

  
  Scenario Outline: [2] Exponentiation takes precedence over numeric multiplicative operations
    Given an empty graph
    When executing query:
      """
      RETURN 4 ^ 3 <mult> 2 ^ 3 AS a,
             (4 ^ 3) <mult> (2 ^ 3) AS b,
             4 ^ (3 <mult> 2) ^ 3 AS c
      """
    Then the result should be, in any order:
      | a       | b       | c       |
      | <right> | <right> | <wrong> |
    And no side effects

    Examples:
      | mult | right | wrong         |
      | *    | 512.0 | 68719476736.0 |
      | /    | 8.0   | 64.0          |
      | %    | 0.0   | 64.0          |

  
  Scenario Outline: [3] Exponentiation takes precedence over numeric additive operations
    Given an empty graph
    When executing query:
      """
      RETURN 4 ^ 3 <add> 2 ^ 3 AS a,
             (4 ^ 3) <add> (2 ^ 3) AS b,
             4 ^ (3 <add> 2) ^ 3 AS c
      """
    Then the result should be, in any order:
      | a       | b       | c       |
      | <right> | <right> | <wrong> |
    And no side effects

    Examples:
      | add | right | wrong        |
      | +   | 72.0  | 1073741824.0 |
      | -   | 56.0  | 64.0         |

  Scenario: [4] Numeric unary negative takes precedence over exponentiation
    Given an empty graph
    When executing query:
      """
      RETURN -3 ^ 2 AS a,
             (-3) ^ 2 AS b,
             -(3 ^ 2) AS c
      """
    Then the result should be, in any order:
      | a   | b   | c    |
      | 9.0 | 9.0 | -9.0 |
    And no side effects

  # Numeric additive inverse and numeric multiplicative operation are associative

  Scenario Outline: [5] Numeric unary negative takes precedence over numeric additive operations
    Given an empty graph
    When executing query:
      """
      RETURN -3 <add> 2 AS a,
             (-3) <add> 2 AS b,
             -(3 <add> 2) AS c
      """
    Then the result should be, in any order:
      | a       | b       | c       |
      | <right> | <right> | <wrong> |
    And no side effects

    Examples:
      | add | right | wrong |
      | +   | -1    | -5    |
      | -   | -5    | -1    |
