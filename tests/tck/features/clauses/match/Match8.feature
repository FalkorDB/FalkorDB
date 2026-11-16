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

Feature: Match8 - Match clause interoperation with other clauses

  Scenario: [1] Pattern independent of bound variables results in cross product
    Given an empty graph
    And having executed:
      """
      CREATE (:A), (:B)
      """
    When executing query:
      """
      MATCH (a)
      WITH a
      MATCH (b)
      RETURN a, b
      """
    Then the result should be, in any order:
      | a    | b    |
      | (:A) | (:A) |
      | (:A) | (:B) |
      | (:B) | (:A) |
      | (:B) | (:B) |
    And no side effects

  @skip
  Scenario: [2] Counting rows after MATCH, MERGE, OPTIONAL MATCH
    Given an empty graph
    And having executed:
      """
      CREATE (a:A), (b:B)
      CREATE (a)-[:T1]->(b),
             (b)-[:T2]->(a)
      """
    When executing query:
      """
      MATCH (a)
      MERGE (b)
      WITH *
      OPTIONAL MATCH (a)--(b)
      RETURN count(*)
      """
    Then the result should be, in any order:
      | count(*) |
      | 6        |
    And no side effects

  @skip
  Scenario: [3] Matching and disregarding output, then matching again
    Given an empty graph
    And having executed:
      """
      CREATE (andres {name: 'Andres'}),
             (michael {name: 'Michael'}),
             (peter {name: 'Peter'}),
             (bread {type: 'Bread'}),
             (veggies {type: 'Veggies'}),
             (meat {type: 'Meat'})
      CREATE (andres)-[:ATE {times: 10}]->(bread),
             (andres)-[:ATE {times: 8}]->(veggies),
             (michael)-[:ATE {times: 4}]->(veggies),
             (michael)-[:ATE {times: 6}]->(bread),
             (michael)-[:ATE {times: 9}]->(meat),
             (peter)-[:ATE {times: 7}]->(veggies),
             (peter)-[:ATE {times: 7}]->(bread),
             (peter)-[:ATE {times: 4}]->(meat)
      """
    When executing query:
      """
      MATCH ()-->()
      WITH 1 AS x
      MATCH ()-[r1]->()<--()
      RETURN sum(r1.times)
      """
    Then the result should be, in any order:
      | sum(r1.times) |
      | 776           |
    And no side effects
