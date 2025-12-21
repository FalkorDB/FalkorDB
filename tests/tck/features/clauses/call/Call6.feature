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

Feature: Call6 - Call clause interoperation with other clauses

  
  Scenario: [1] Calling the same STRING procedure twice using the same outputs in each call
    Given an empty graph
    And there exists a procedure test.labels() :: (label :: STRING?):
      | label |
      | 'A'   |
      | 'B'   |
      | 'C'   |
    When executing query:
      """
      CALL test.labels() YIELD label
      WITH count(*) AS c
      CALL test.labels() YIELD label
      RETURN *
      """
    Then the result should be, in order:
      | c | label |
      | 3 | 'A'   |
      | 3 | 'B'   |
      | 3 | 'C'   |
    And no side effects

  
  Scenario: [2] Project procedure results between query scopes with WITH clause
    Given an empty graph
    And there exists a procedure test.my.proc(in :: INTEGER?) :: (out :: STRING?):
      | in   | out   |
      | null | 'nix' |
    When executing query:
      """
      CALL test.my.proc(null) YIELD out
      WITH out RETURN out
      """
    Then the result should be, in order:
      | out   |
      | 'nix' |
    And no side effects

  
  Scenario: [3] Project procedure results between query scopes with WITH clause and rename the projection
    Given an empty graph
    And there exists a procedure test.my.proc(in :: INTEGER?) :: (out :: STRING?):
      | in   | out   |
      | null | 'nix' |
    When executing query:
      """
      CALL test.my.proc(null) YIELD out
      WITH out AS a RETURN a
      """
    Then the result should be, in order:
      | a     |
      | 'nix' |
    And no side effects
