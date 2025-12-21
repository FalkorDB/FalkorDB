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

Feature: Call1 - Basic procedure calling

  
  Scenario: [1] Standalone call to procedure that takes no arguments and yields no results
    Given an empty graph
    And there exists a procedure test.doNothing() :: ():
      |
    When executing query:
      """
      CALL test.doNothing()
      """
    Then the result should be empty
    And no side effects

  
  Scenario: [2] Standalone call to procedure that takes no arguments and yields no results, called with implicit arguments
    Given an empty graph
    And there exists a procedure test.doNothing() :: ():
      |
    When executing query:
      """
      CALL test.doNothing
      """
    Then the result should be empty
    And no side effects

  
  Scenario: [3] In-query call to procedure that takes no arguments and yields no results
    Given an empty graph
    And there exists a procedure test.doNothing() :: ():
      |
    When executing query:
      """
      MATCH (n)
      CALL test.doNothing()
      RETURN n
      """
    Then the result should be, in any order:
      | n |
    And no side effects

  
  Scenario: [4] In-query call to procedure that takes no arguments and yields no results and consumes no rows
    Given an empty graph
    And there exists a procedure test.doNothing() :: ():
      |
    And having executed:
      """
      CREATE (:A {name: 'a'})
      CREATE (:B {name: 'b'})
      CREATE (:C {name: 'c'})
      """
    When executing query:
      """
      MATCH (n)
      CALL test.doNothing()
      RETURN n.name AS `name`
      """
    Then the result should be, in any order:
      | name |
      | 'a'  |
      | 'b'  |
      | 'c'  |
    And no side effects

  
  Scenario: [5] Standalone call to STRING procedure that takes no arguments
    Given an empty graph
    And there exists a procedure test.labels() :: (label :: STRING?):
      | label |
      | 'A'   |
      | 'B'   |
      | 'C'   |
    When executing query:
      """
      CALL test.labels()
      """
    Then the result should be, in order:
      | label |
      | 'A'   |
      | 'B'   |
      | 'C'   |
    And no side effects

  
  Scenario: [6] In-query call to STRING procedure that takes no arguments
    Given an empty graph
    And there exists a procedure test.labels() :: (label :: STRING?):
      | label |
      | 'A'   |
      | 'B'   |
      | 'C'   |
    When executing query:
      """
      CALL test.labels() YIELD label
      RETURN label
      """
    Then the result should be, in order:
      | label |
      | 'A'   |
      | 'B'   |
      | 'C'   |
    And no side effects

  
  Scenario: [7] Standalone call to procedure should fail if explicit argument is missing
    Given an empty graph
    And there exists a procedure test.my.proc(name :: STRING?, in :: INTEGER?) :: (out :: INTEGER?):
      | name | in | out |
    When executing query:
      """
      CALL test.my.proc('Dobby')
      """
    Then a SyntaxError should be raised at compile time: InvalidNumberOfArguments

  
  Scenario: [8] In-query call to procedure should fail if explicit argument is missing
    Given an empty graph
    And there exists a procedure test.my.proc(name :: STRING?, in :: INTEGER?) :: (out :: INTEGER?):
      | name | in | out |
    When executing query:
      """
      CALL test.my.proc('Dobby') YIELD out
      RETURN out
      """
    Then a SyntaxError should be raised at compile time: InvalidNumberOfArguments

  
  Scenario: [9] Standalone call to procedure should fail if too many explicit argument are given
    Given an empty graph
    And there exists a procedure test.my.proc(in :: INTEGER?) :: (out :: INTEGER?):
      | in | out |
    When executing query:
      """
      CALL test.my.proc(1, 2, 3, 4)
      """
    Then a SyntaxError should be raised at compile time: InvalidNumberOfArguments

  
  Scenario: [10] In-query call to procedure should fail if too many explicit argument are given
    Given an empty graph
    And there exists a procedure test.my.proc(in :: INTEGER?) :: (out :: INTEGER?):
      | in | out |
    When executing query:
      """
      CALL test.my.proc(1, 2, 3, 4) YIELD out
      RETURN out
      """
    Then a SyntaxError should be raised at compile time: InvalidNumberOfArguments

  
  Scenario: [11] Standalone call to procedure should fail if implicit argument is missing
    Given an empty graph
    And there exists a procedure test.my.proc(name :: STRING?, in :: INTEGER?) :: (out :: INTEGER?):
      | name | in | out |
    And parameters are:
      | name | 'Stefan' |
    When executing query:
      """
      CALL test.my.proc
      """
    Then a ParameterMissing should be raised at compile time: MissingParameter

  
  Scenario: [12] In-query call to procedure that has outputs fails if no outputs are yielded
    Given an empty graph
    And there exists a procedure test.my.proc(in :: INTEGER?) :: (out :: INTEGER?):
      | in | out |
    When executing query:
      """
      CALL test.my.proc(1)
      RETURN out
      """
    Then a SyntaxError should be raised at compile time: UndefinedVariable

  
  Scenario: [13] Standalone call to unknown procedure should fail
    Given an empty graph
    When executing query:
      """
      CALL test.my.proc
      """
    Then a ProcedureError should be raised at compile time: ProcedureNotFound

  
  Scenario: [14] In-query call to unknown procedure should fail
    Given an empty graph
    When executing query:
      """
      CALL test.my.proc() YIELD out
      RETURN out
      """
    Then a ProcedureError should be raised at compile time: ProcedureNotFound

  
  Scenario: [15] In-query procedure call should fail if shadowing an already bound variable
    Given an empty graph
    And there exists a procedure test.labels() :: (label :: STRING?):
      | label |
      | 'A'   |
      | 'B'   |
      | 'C'   |
    When executing query:
      """
      WITH 'Hi' AS label
      CALL test.labels() YIELD label
      RETURN *
      """
    Then a SyntaxError should be raised at compile time: VariableAlreadyBound

  
  Scenario: [16] In-query procedure call should fail if one of the argument expressions uses an aggregation function
    Given an empty graph
    And there exists a procedure test.labels(in :: INTEGER?) :: (label :: STRING?):
      | in | label |
    When executing query:
      """
      MATCH (n)
      CALL test.labels(count(n)) YIELD label
      RETURN label
      """
    Then a SyntaxError should be raised at compile time: InvalidAggregation
