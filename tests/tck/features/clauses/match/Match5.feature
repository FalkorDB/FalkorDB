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

Feature: Match5 - Match variable length patterns over given graphs scenarios

  # TODO: Replace this with a named graph (or two)
  Background:
    Given an empty graph
    And having executed:
      """
      CREATE (n0:A {name: 'n0'}),
             (n00:B {name: 'n00'}),
             (n01:B {name: 'n01'}),
             (n000:C {name: 'n000'}),
             (n001:C {name: 'n001'}),
             (n010:C {name: 'n010'}),
             (n011:C {name: 'n011'}),
             (n0000:D {name: 'n0000'}),
             (n0001:D {name: 'n0001'}),
             (n0010:D {name: 'n0010'}),
             (n0011:D {name: 'n0011'}),
             (n0100:D {name: 'n0100'}),
             (n0101:D {name: 'n0101'}),
             (n0110:D {name: 'n0110'}),
             (n0111:D {name: 'n0111'})
      CREATE (n0)-[:LIKES]->(n00),
             (n0)-[:LIKES]->(n01),
             (n00)-[:LIKES]->(n000),
             (n00)-[:LIKES]->(n001),
             (n01)-[:LIKES]->(n010),
             (n01)-[:LIKES]->(n011),
             (n000)-[:LIKES]->(n0000),
             (n000)-[:LIKES]->(n0001),
             (n001)-[:LIKES]->(n0010),
             (n001)-[:LIKES]->(n0011),
             (n010)-[:LIKES]->(n0100),
             (n010)-[:LIKES]->(n0101),
             (n011)-[:LIKES]->(n0110),
             (n011)-[:LIKES]->(n0111)
      """

  Scenario: [1] Handling unbounded variable length match
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES*]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name  |
      | 'n00'   |
      | 'n01'   |
      | 'n000'  |
      | 'n001'  |
      | 'n010'  |
      | 'n011'  |
      | 'n0000' |
      | 'n0001' |
      | 'n0010' |
      | 'n0011' |
      | 'n0100' |
      | 'n0101' |
      | 'n0110' |
      | 'n0111' |
    And no side effects

  Scenario: [2] Handling explicitly unbounded variable length match
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES*..]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name  |
      | 'n00'   |
      | 'n01'   |
      | 'n000'  |
      | 'n001'  |
      | 'n010'  |
      | 'n011'  |
      | 'n0000' |
      | 'n0001' |
      | 'n0010' |
      | 'n0011' |
      | 'n0100' |
      | 'n0101' |
      | 'n0110' |
      | 'n0111' |
    And no side effects

  Scenario: [3] Handling single bounded variable length match 1
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES*0]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'n0'   |
    And no side effects

  Scenario: [4] Handling single bounded variable length match 2
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES*1]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'n00'  |
      | 'n01'  |
    And no side effects

  Scenario: [5] Handling single bounded variable length match 3
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES*2]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'n000' |
      | 'n001' |
      | 'n010' |
      | 'n011' |
    And no side effects

  Scenario: [6] Handling upper and lower bounded variable length match 1
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES*0..2]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'n0'   |
      | 'n00'  |
      | 'n01'  |
      | 'n000' |
      | 'n001' |
      | 'n010' |
      | 'n011' |
    And no side effects

  Scenario: [7] Handling upper and lower bounded variable length match 2
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES*1..2]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'n00'  |
      | 'n01'  |
      | 'n000' |
      | 'n001' |
      | 'n010' |
      | 'n011' |
    And no side effects

  Scenario: [8] Handling symmetrically bounded variable length match, bounds are zero
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES*0..0]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'n0'   |
    And no side effects

  Scenario: [9] Handling symmetrically bounded variable length match, bounds are one
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES*1..1]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'n00'  |
      | 'n01'  |
    And no side effects

  Scenario: [10] Handling symmetrically bounded variable length match, bounds are two
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES*2..2]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'n000' |
      | 'n001' |
      | 'n010' |
      | 'n011' |
    And no side effects

  @skip
  Scenario: [11] Handling upper and lower bounded variable length match, empty interval 1
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES*2..1]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
    And no side effects

  @skip
  Scenario: [12] Handling upper and lower bounded variable length match, empty interval 2
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES*1..0]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
    And no side effects

  @skip
  Scenario: [13] Handling upper bounded variable length match, empty interval
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES*..0]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
    And no side effects

  Scenario: [14] Handling upper bounded variable length match 1
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES*..1]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'n00'  |
      | 'n01'  |
    And no side effects

  Scenario: [15] Handling upper bounded variable length match 2
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES*..2]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'n00'  |
      | 'n01'  |
      | 'n000' |
      | 'n001' |
      | 'n010' |
      | 'n011' |
    And no side effects

  Scenario: [16] Handling lower bounded variable length match 1
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES*0..]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name  |
      | 'n0'    |
      | 'n00'   |
      | 'n01'   |
      | 'n000'  |
      | 'n001'  |
      | 'n010'  |
      | 'n011'  |
      | 'n0000' |
      | 'n0001' |
      | 'n0010' |
      | 'n0011' |
      | 'n0100' |
      | 'n0101' |
      | 'n0110' |
      | 'n0111' |
    And no side effects

  Scenario: [17] Handling lower bounded variable length match 2
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES*1..]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name  |
      | 'n00'   |
      | 'n01'   |
      | 'n000'  |
      | 'n001'  |
      | 'n010'  |
      | 'n011'  |
      | 'n0000' |
      | 'n0001' |
      | 'n0010' |
      | 'n0011' |
      | 'n0100' |
      | 'n0101' |
      | 'n0110' |
      | 'n0111' |
    And no side effects

  Scenario: [18] Handling lower bounded variable length match 3
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES*2..]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name  |
      | 'n000'  |
      | 'n001'  |
      | 'n010'  |
      | 'n011'  |
      | 'n0000' |
      | 'n0001' |
      | 'n0010' |
      | 'n0011' |
      | 'n0100' |
      | 'n0101' |
      | 'n0110' |
      | 'n0111' |
    And no side effects

  Scenario: [19] Handling a variable length relationship and a standard relationship in chain, zero length 1
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES*0]->()-[:LIKES]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'n00'  |
      | 'n01'  |
    And no side effects

  Scenario: [20] Handling a variable length relationship and a standard relationship in chain, zero length 2
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES]->()-[:LIKES*0]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'n00'  |
      | 'n01'  |
    And no side effects

  Scenario: [21] Handling a variable length relationship and a standard relationship in chain, single length 1
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES*1]->()-[:LIKES]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'n000' |
      | 'n001' |
      | 'n010' |
      | 'n011' |
    And no side effects

  Scenario: [22] Handling a variable length relationship and a standard relationship in chain, single length 2
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES]->()-[:LIKES*1]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'n000' |
      | 'n001' |
      | 'n010' |
      | 'n011' |
    And no side effects

  Scenario: [23] Handling a variable length relationship and a standard relationship in chain, longer 1
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES*2]->()-[:LIKES]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name  |
      | 'n0000' |
      | 'n0001' |
      | 'n0010' |
      | 'n0011' |
      | 'n0100' |
      | 'n0101' |
      | 'n0110' |
      | 'n0111' |
    And no side effects

  Scenario: [24] Handling a variable length relationship and a standard relationship in chain, longer 2
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES]->()-[:LIKES*2]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name  |
      | 'n0000' |
      | 'n0001' |
      | 'n0010' |
      | 'n0011' |
      | 'n0100' |
      | 'n0101' |
      | 'n0110' |
      | 'n0111' |
    And no side effects

  Scenario: [25] Handling a variable length relationship and a standard relationship in chain, longer 3
    And having executed:
      """
      MATCH (d:D)
      CREATE (e1:E {name: d.name + '0'}),
             (e2:E {name: d.name + '1'})
      CREATE (d)-[:LIKES]->(e1),
             (d)-[:LIKES]->(e2)
      """
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES]->()-[:LIKES*3]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name   |
      | 'n00000' |
      | 'n00001' |
      | 'n00010' |
      | 'n00011' |
      | 'n00100' |
      | 'n00101' |
      | 'n00110' |
      | 'n00111' |
      | 'n01000' |
      | 'n01001' |
      | 'n01010' |
      | 'n01011' |
      | 'n01100' |
      | 'n01101' |
      | 'n01110' |
      | 'n01111' |
    And no side effects

  Scenario: [26] Handling mixed relationship patterns and directions 1
    And having executed:
      """
      MATCH (a:A)-[r]->(b)
      DELETE r
      CREATE (b)-[:LIKES]->(a)
      """
    And having executed:
      """
      MATCH (d:D)
      CREATE (e1:E {name: d.name + '0'}),
             (e2:E {name: d.name + '1'})
      CREATE (d)-[:LIKES]->(e1),
             (d)-[:LIKES]->(e2)
      """
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)<-[:LIKES]-()-[:LIKES*3]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name   |
      | 'n00000' |
      | 'n00001' |
      | 'n00010' |
      | 'n00011' |
      | 'n00100' |
      | 'n00101' |
      | 'n00110' |
      | 'n00111' |
      | 'n01000' |
      | 'n01001' |
      | 'n01010' |
      | 'n01011' |
      | 'n01100' |
      | 'n01101' |
      | 'n01110' |
      | 'n01111' |
    And no side effects

  @skip
  Scenario: [27] Handling mixed relationship patterns and directions 2
    # This gets hard to follow for a human mind. The answer is named graphs, but it's not crucial to fix.
    And having executed:
      """
      MATCH (a)-[r]->(b)
      WHERE NOT a:A
      DELETE r
      CREATE (b)-[:LIKES]->(a)
      """
    And having executed:
      """
      MATCH (d:D)
      CREATE (e1:E {name: d.name + '0'}),
             (e2:E {name: d.name + '1'})
      CREATE (d)-[:LIKES]->(e1),
             (d)-[:LIKES]->(e2)
      """
    When executing query:
      """
      MATCH (a:A)
      MATCH (a)-[:LIKES]->()<-[:LIKES*3]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name   |
      | 'n00000' |
      | 'n00001' |
      | 'n00010' |
      | 'n00011' |
      | 'n00100' |
      | 'n00101' |
      | 'n00110' |
      | 'n00111' |
      | 'n01000' |
      | 'n01001' |
      | 'n01010' |
      | 'n01011' |
      | 'n01100' |
      | 'n01101' |
      | 'n01110' |
      | 'n01111' |
    And no side effects

  Scenario: [28] Handling mixed relationship patterns 1
    And having executed:
      """
      MATCH (d:D)
      CREATE (e1:E {name: d.name + '0'}),
             (e2:E {name: d.name + '1'})
      CREATE (d)-[:LIKES]->(e1),
             (d)-[:LIKES]->(e2)
      """
    When executing query:
      """
      MATCH (a:A)
      MATCH (p)-[:LIKES*1]->()-[:LIKES]->()-[r:LIKES*2]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name   |
      | 'n00000' |
      | 'n00001' |
      | 'n00010' |
      | 'n00011' |
      | 'n00100' |
      | 'n00101' |
      | 'n00110' |
      | 'n00111' |
      | 'n01000' |
      | 'n01001' |
      | 'n01010' |
      | 'n01011' |
      | 'n01100' |
      | 'n01101' |
      | 'n01110' |
      | 'n01111' |
    And no side effects

  Scenario: [29] Handling mixed relationship patterns 2
    And having executed:
      """
      MATCH (d:D)
      CREATE (e1:E {name: d.name + '0'}),
             (e2:E {name: d.name + '1'})
      CREATE (d)-[:LIKES]->(e1),
             (d)-[:LIKES]->(e2)
      """
    When executing query:
      """
      MATCH (a:A)
      MATCH (p)-[:LIKES]->()-[:LIKES*2]->()-[r:LIKES]->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name   |
      | 'n00000' |
      | 'n00001' |
      | 'n00010' |
      | 'n00011' |
      | 'n00100' |
      | 'n00101' |
      | 'n00110' |
      | 'n00111' |
      | 'n01000' |
      | 'n01001' |
      | 'n01010' |
      | 'n01011' |
      | 'n01100' |
      | 'n01101' |
      | 'n01110' |
      | 'n01111' |
    And no side effects
