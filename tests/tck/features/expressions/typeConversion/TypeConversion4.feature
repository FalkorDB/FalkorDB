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

Feature: TypeConversion4 - To String

  Scenario: [1] `toString()` handling integer literal
    Given any graph
    When executing query:
      """
      RETURN toString(42) AS bool
      """
    Then the result should be, in any order:
      | bool   |
      | '42' |
    And no side effects

  
  Scenario: [2] `toString()` handling boolean literal
    Given any graph
    When executing query:
      """
      RETURN toString(true) AS bool
      """
    Then the result should be, in any order:
      | bool   |
      | 'true' |
    And no side effects

  
  Scenario: [3] `toString()` handling inlined boolean
    Given any graph
    When executing query:
      """
      RETURN toString(1 < 0) AS bool
      """
    Then the result should be, in any order:
      | bool    |
      | 'false' |
    And no side effects

  
  Scenario: [4] `toString()` handling boolean properties
    Given an empty graph
    And having executed:
      """
      CREATE (:Movie {watched: true})
      """
    When executing query:
      """
      MATCH (m:Movie)
      RETURN toString(m.watched)
      """
    Then the result should be, in any order:
      | toString(m.watched) |
      | 'true'              |
    And no side effects

  
  Scenario: [5] `toString()` should work on Any type
    Given any graph
    When executing query:
      """
      RETURN [x IN [1, 2.3, true, 'apa'] | toString(x) ] AS list
      """
    Then the result should be, in any order:
      | list                        |
      | ['1', '2.3', 'true', 'apa'] |
    And no side effects

  Scenario: [6] `toString()` on a list of integers
    Given any graph
    When executing query:
      """
      WITH [1, 2, 3] AS numbers
      RETURN [n IN numbers | toString(n)] AS string_numbers
      """
    Then the result should be, in any order:
      | string_numbers  |
      | ['1', '2', '3'] |
    And no side effects

  Scenario: [7] `toString()` on node property
    Given an empty graph
    And having executed:
      """
      CREATE (:Movie {rating: 4})
      """
    When executing query:
      """
      MATCH (m:Movie { rating: 4 })
      WITH *
      MATCH (n)
      RETURN toString(n.rating)
      """
    Then the result should be, in any order:
      | toString(n.rating) |
      | '4'                |
    And no side effects

  Scenario: [8] `toString()` should accept potentially correct types 1
    Given any graph
    When executing query:
      """
      UNWIND ['male', 'female', null] AS gen
      RETURN coalesce(toString(gen), 'x') AS result
      """
    Then the result should be, in any order:
      | result   |
      | 'male'   |
      | 'female' |
      | 'x'      |
    And no side effects

  Scenario: [9] `toString()` should accept potentially correct types 2
    Given any graph
    When executing query:
      """
      UNWIND ['male', 'female', null] AS gen
      RETURN toString(coalesce(gen, 'x')) AS result
      """
    Then the result should be, in any order:
      | result   |
      | 'male'   |
      | 'female' |
      | 'x'      |
    And no side effects

  Scenario Outline: [10] Fail `toString()` on invalid types #Example: <exampleName>
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:T]->()
      """
    When executing query:
      """
      MATCH p = (n)-[r:T]->()
      RETURN [x IN [1, '', <invalid>] | toString(x) ] AS list
      """
    Then a TypeError should be raised at runtime: InvalidArgumentValue

    Examples:
      | invalid | exampleName  |
      | []      | list         |
      | {}      | map          |
      | n       | node         |
      | r       | relationship |
      | p       | path         |
