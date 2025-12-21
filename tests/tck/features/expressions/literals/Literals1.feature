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

Feature: Literals1 - Boolean and Null

  Scenario: [1] Return a boolean true lower case
    Given any graph
    When executing query:
      """
      RETURN true AS literal
      """
    Then the result should be, in any order:
      | literal |
      | true    |
    And no side effects

  @skipStyleCheck
  Scenario: [2] Return a boolean true upper case
    Given any graph
    When executing query:
      """
      RETURN TRUE AS literal
      """
    Then the result should be, in any order:
      | literal |
      | true    |
    And no side effects

  Scenario: [3] Return a boolean false lower case
    Given any graph
    When executing query:
      """
      RETURN false AS literal
      """
    Then the result should be, in any order:
      | literal |
      | false   |
    And no side effects

  @skipStyleCheck
  Scenario: [4] Return a boolean false upper case
    Given any graph
    When executing query:
      """
      RETURN FALSE AS literal
      """
    Then the result should be, in any order:
      | literal |
      | false    |
    And no side effects

  Scenario: [5] Return null lower case
    Given any graph
    When executing query:
      """
      RETURN null AS literal
      """
    Then the result should be, in any order:
      | literal |
      | null    |
    And no side effects

  @skipStyleCheck
  Scenario: [6] Return null upper case
    Given any graph
    When executing query:
      """
      RETURN NULL AS literal
      """
    Then the result should be, in any order:
      | literal |
      | null    |
    And no side effects