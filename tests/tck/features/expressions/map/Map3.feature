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

Feature: Map3 - Keys function

  
  Scenario: [1] Using `keys()` on a literal map
    Given any graph
    When executing query:
      """
      RETURN keys({name: 'Alice', age: 38, address: {city: 'London', residential: true}}) AS k
      """
    Then the result should be (ignoring element order for lists):
      | k                          |
      | ['name', 'age', 'address'] |
    And no side effects

  
  Scenario: [2] Using `keys()` on a parameter map
    Given any graph
    And parameters are:
      | param | {name: 'Alice', age: 38, address: {city: 'London', residential: true}} |
    When executing query:
      """
      RETURN keys($param) AS k
      """
    Then the result should be (ignoring element order for lists):
      | k                          |
      | ['address', 'name', 'age'] |
    And no side effects

  Scenario: [3] Using `keys()` on null map
    Given any graph
    When executing query:
      """
      WITH null AS m
      RETURN keys(m), keys(null)
      """
    Then the result should be, in any order:
      | keys(m) | keys(null) |
      | null    | null       |
    And no side effects

  
  Scenario Outline: [4] Using `keys()` on map with null values
    Given any graph
    When executing query:
      """
      RETURN keys(<map>) AS keys
      """
    Then the result should be (ignoring element order for lists):
      | keys     |
      | <result> |
    And no side effects

    Examples:
      | map                   | result          |
      | {}                    | []              |
      | {k: 1}                | ['k']           |
      | {k: null}             | ['k']           |
      | {k: null, l: 1}       | ['k', 'l']      |
      | {k: 1, l: null}       | ['k', 'l']      |
      | {k: null, l: null}    | ['k', 'l']      |
      | {k: 1, l: null, m: 1} | ['k', 'l', 'm'] |

  
  Scenario: [5] Using `keys()` and `IN` to check field existence
    Given any graph
    When executing query:
      """
      WITH {exists: 42, notMissing: null} AS map
      RETURN 'exists' IN keys(map) AS a,
             'notMissing' IN keys(map) AS b,
             'missing' IN keys(map) AS c
      """
    Then the result should be, in any order:
      | a    | b    | c     |
      | true | true | false |
    And no side effects
