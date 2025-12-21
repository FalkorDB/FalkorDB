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

Feature: Call5 - Results projection

  
  Scenario: [1] Explicit procedure result projection
    Given an empty graph
    And there exists a procedure test.my.proc(in :: INTEGER?) :: (out :: STRING?):
      | in   | out   |
      | null | 'nix' |
    When executing query:
      """
      CALL test.my.proc(null) YIELD out
      RETURN out
      """
    Then the result should be, in order:
      | out   |
      | 'nix' |
    And no side effects

  
  Scenario: [2] Explicit procedure result projection with RETURN *
    Given an empty graph
    And there exists a procedure test.my.proc(in :: INTEGER?) :: (out :: STRING?):
      | in   | out   |
      | null | 'nix' |
    When executing query:
      """
      CALL test.my.proc(null) YIELD out
      RETURN *
      """
    Then the result should be, in order:
      | out   |
      | 'nix' |
    And no side effects

  
  Scenario Outline: [3] The order of yield items is irrelevant
    Given an empty graph
    And there exists a procedure test.my.proc(in :: INTEGER?) :: (a :: INTEGER?, b :: INTEGER?) :
      | in   | a | b |
      | null | 1 | 2 |
    When executing query:
      """
      CALL test.my.proc(null) YIELD <yield>
      RETURN a, b
      """
    Then the result should be, in order:
      | a | b |
      | 1 | 2 |
    And no side effects

    Examples:
      | yield |
      | a, b  |
      | b, a  |

  
  Scenario Outline: [4] Rename outputs to unbound variable names
    Given an empty graph
    And there exists a procedure test.my.proc(in :: INTEGER?) :: (a :: INTEGER?, b :: INTEGER?) :
      | in   | a | b |
      | null | 1 | 2 |
    When executing query:
      """
      CALL test.my.proc(null) YIELD <rename>
      RETURN <a>, <b>
      """
    Then the result should be, in order:
      | <a> | <b> |
      | 1   | 2   |
    And no side effects

    Examples:
      | rename         | a | b |
      | a AS c, b AS d | c | d |
      | a AS b, b AS d | b | d |
      | a AS c, b AS a | c | a |
      | a AS b, b AS a | b | a |
      | a AS c, b AS b | c | b |
      | a AS c, b      | c | b |
      | a AS a, b AS d | a | d |
      | a, b AS d      | a | d |
      | a AS a, b AS b | a | b |
      | a AS a, b      | a | b |
      | a, b AS b      | a | b |


  
  Scenario: [5] Fail on renaming to an already bound variable name
    Given an empty graph
    And there exists a procedure test.my.proc(in :: INTEGER?) :: (a :: INTEGER?, b :: INTEGER?) :
      | in   | a | b |
      | null | 1 | 2 |
    When executing query:
      """
      CALL test.my.proc(null) YIELD a, b AS a
      RETURN a
      """
    Then a SyntaxError should be raised at compile time: VariableAlreadyBound

  
  Scenario: [6] Fail on renaming all outputs to the same variable name
    Given an empty graph
    And there exists a procedure test.my.proc(in :: INTEGER?) :: (a :: INTEGER?, b :: INTEGER?) :
      | in   | a | b |
      | null | 1 | 2 |
    When executing query:
      """
      CALL test.my.proc(null) YIELD a AS c, b AS c
      RETURN c
      """
    Then a SyntaxError should be raised at compile time: VariableAlreadyBound

  
  
  Scenario: [7] Fail on in-query call to procedure with YIELD *
    Given an empty graph
    And there exists a procedure test.my.proc(name :: STRING?, id :: INTEGER?) :: (city :: STRING?, country_code :: INTEGER?):
      | name     | id | city      | country_code |
      | 'Andres' | 1  | 'Malmö'   | 46           |
      | 'Tobias' | 1  | 'Malmö'   | 46           |
      | 'Mats'   | 1  | 'Malmö'   | 46           |
      | 'Stefan' | 1  | 'Berlin'  | 49           |
      | 'Stefan' | 2  | 'München' | 49           |
      | 'Petra'  | 1  | 'London'  | 44           |
    When executing query:
      """
      CALL test.my.proc('Stefan', 1) YIELD *
      RETURN city, country_code
      """
    Then a SyntaxError should be raised at compile time: UnexpectedSyntax

  
  Scenario: [8] Allow standalone call to procedure with YIELD *
    Given an empty graph
    And there exists a procedure test.my.proc(name :: STRING?, id :: INTEGER?) :: (city :: STRING?, country_code :: INTEGER?):
      | name     | id | city      | country_code |
      | 'Andres' | 1  | 'Malmö'   | 46           |
      | 'Tobias' | 1  | 'Malmö'   | 46           |
      | 'Mats'   | 1  | 'Malmö'   | 46           |
      | 'Stefan' | 1  | 'Berlin'  | 49           |
      | 'Stefan' | 2  | 'München' | 49           |
      | 'Petra'  | 1  | 'London'  | 44           |
    When executing query:
      """
      CALL test.my.proc('Stefan', 1) YIELD *
      """
    Then the result should be, in order:
      | city     | country_code |
      | 'Berlin' | 49           |
    And no side effects
