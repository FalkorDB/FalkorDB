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

Feature: Call2 - Procedure arguments

  
  Scenario: [1] In-query call to procedure with explicit arguments
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
      CALL test.my.proc('Stefan', 1) YIELD city, country_code
      RETURN city, country_code
      """
    Then the result should be, in order:
      | city     | country_code |
      | 'Berlin' | 49           |
    And no side effects

  
  Scenario: [2] Standalone call to procedure with explicit arguments
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
      CALL test.my.proc('Stefan', 1)
      """
    Then the result should be, in order:
      | city     | country_code |
      | 'Berlin' | 49           |
    And no side effects

  
  Scenario: [3] Standalone call to procedure with implicit arguments
    Given an empty graph
    And there exists a procedure test.my.proc(name :: STRING?, id :: INTEGER?) :: (city :: STRING?, country_code :: INTEGER?):
      | name     | id | city      | country_code |
      | 'Andres' | 1  | 'Malmö'   | 46           |
      | 'Tobias' | 1  | 'Malmö'   | 46           |
      | 'Mats'   | 1  | 'Malmö'   | 46           |
      | 'Stefan' | 1  | 'Berlin'  | 49           |
      | 'Stefan' | 2  | 'München' | 49           |
      | 'Petra'  | 1  | 'London'  | 44           |
    And parameters are:
      | name | 'Stefan' |
      | id   | 1        |
    When executing query:
      """
      CALL test.my.proc
      """
    Then the result should be, in order:
      | city     | country_code |
      | 'Berlin' | 49           |
    And no side effects

  
  
  Scenario: [4] In-query call to procedure that takes arguments fails when trying to pass them implicitly
    Given an empty graph
    And there exists a procedure test.my.proc(in :: INTEGER?) :: (out :: INTEGER?):
      | in | out |
    When executing query:
      """
      CALL test.my.proc YIELD out
      RETURN out
      """
    Then a SyntaxError should be raised at compile time: InvalidArgumentPassingMode

  
  Scenario: [5] Standalone call to procedure should fail if input type is wrong
    Given an empty graph
    And there exists a procedure test.my.proc(in :: INTEGER?) :: (out :: INTEGER?):
      | in | out |
    When executing query:
      """
      CALL test.my.proc(true)
      """
    Then a SyntaxError should be raised at compile time: InvalidArgumentType

  
  Scenario: [6] In-query call to procedure should fail if input type is wrong
    Given an empty graph
    And there exists a procedure test.my.proc(in :: INTEGER?) :: (out :: INTEGER?):
      | in | out |
    When executing query:
      """
      CALL test.my.proc(true) YIELD out
      RETURN out
      """
    Then a SyntaxError should be raised at compile time: InvalidArgumentType
