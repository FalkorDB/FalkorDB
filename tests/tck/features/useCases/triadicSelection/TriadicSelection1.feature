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

Feature: TriadicSelection1 - Query three related nodes on binary-tree graphs

  Scenario: [1] Handling triadic friend of a friend
    Given the binary-tree-1 graph
    When executing query:
      """
      MATCH (a:A)-[:KNOWS]->(b)-->(c)
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'b2'   |
      | 'b3'   |
      | 'c11'  |
      | 'c12'  |
      | 'c21'  |
      | 'c22'  |
    And no side effects

  
  Scenario: [2] Handling triadic friend of a friend that is not a friend
    Given the binary-tree-1 graph
    When executing query:
      """
      MATCH (a:A)-[:KNOWS]->(b)-->(c)
      OPTIONAL MATCH (a)-[r:KNOWS]->(c)
      WITH c WHERE r IS NULL
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'b3'   |
      | 'c11'  |
      | 'c12'  |
      | 'c21'  |
      | 'c22'  |
    And no side effects

  
  Scenario: [3] Handling triadic friend of a friend that is not a friend with different relationship type
    Given the binary-tree-1 graph
    When executing query:
      """
      MATCH (a:A)-[:KNOWS]->(b)-->(c)
      OPTIONAL MATCH (a)-[r:FOLLOWS]->(c)
      WITH c WHERE r IS NULL
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'b2'   |
      | 'c11'  |
      | 'c12'  |
      | 'c21'  |
      | 'c22'  |
    And no side effects

  
  Scenario: [4] Handling triadic friend of a friend that is not a friend with superset of relationship type
    Given the binary-tree-1 graph
    When executing query:
      """
      MATCH (a:A)-[:KNOWS]->(b)-->(c)
      OPTIONAL MATCH (a)-[r]->(c)
      WITH c WHERE r IS NULL
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'c11'  |
      | 'c12'  |
      | 'c21'  |
      | 'c22'  |
    And no side effects

  
  Scenario: [5] Handling triadic friend of a friend that is not a friend with implicit subset of relationship type
    Given the binary-tree-1 graph
    When executing query:
      """
      MATCH (a:A)-->(b)-->(c)
      OPTIONAL MATCH (a)-[r:KNOWS]->(c)
      WITH c WHERE r IS NULL
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'b3'   |
      | 'b4'   |
      | 'c11'  |
      | 'c12'  |
      | 'c21'  |
      | 'c22'  |
      | 'c31'  |
      | 'c32'  |
      | 'c41'  |
      | 'c42'  |
    And no side effects

  
  Scenario: [6] Handling triadic friend of a friend that is not a friend with explicit subset of relationship type
    Given the binary-tree-1 graph
    When executing query:
      """
      MATCH (a:A)-[:KNOWS|FOLLOWS]->(b)-->(c)
      OPTIONAL MATCH (a)-[r:KNOWS]->(c)
      WITH c WHERE r IS NULL
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'b3'   |
      | 'b4'   |
      | 'c11'  |
      | 'c12'  |
      | 'c21'  |
      | 'c22'  |
      | 'c31'  |
      | 'c32'  |
      | 'c41'  |
      | 'c42'  |
    And no side effects

  
  Scenario: [7] Handling triadic friend of a friend that is not a friend with same labels
    Given the binary-tree-2 graph
    When executing query:
      """
      MATCH (a:A)-[:KNOWS]->(b:X)-->(c:X)
      OPTIONAL MATCH (a)-[r:KNOWS]->(c)
      WITH c WHERE r IS NULL
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'b3'   |
      | 'c11'  |
      | 'c21'  |
    And no side effects

  
  Scenario: [8] Handling triadic friend of a friend that is not a friend with different labels
    Given the binary-tree-2 graph
    When executing query:
      """
      MATCH (a:A)-[:KNOWS]->(b:X)-->(c:Y)
      OPTIONAL MATCH (a)-[r:KNOWS]->(c)
      WITH c WHERE r IS NULL
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'c12'  |
      | 'c22'  |
    And no side effects

  
  Scenario: [9] Handling triadic friend of a friend that is not a friend with implicit subset of labels
    Given the binary-tree-2 graph
    When executing query:
      """
      MATCH (a:A)-[:KNOWS]->(b)-->(c:X)
      OPTIONAL MATCH (a)-[r:KNOWS]->(c)
      WITH c WHERE r IS NULL
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'b3'   |
      | 'c11'  |
      | 'c21'  |
    And no side effects

  
  Scenario: [10] Handling triadic friend of a friend that is not a friend with implicit superset of labels
    Given the binary-tree-2 graph
    When executing query:
      """
      MATCH (a:A)-[:KNOWS]->(b:X)-->(c)
      OPTIONAL MATCH (a)-[r:KNOWS]->(c)
      WITH c WHERE r IS NULL
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'b3'   |
      | 'c11'  |
      | 'c12'  |
      | 'c21'  |
      | 'c22'  |
    And no side effects

  
  Scenario: [11] Handling triadic friend of a friend that is a friend
    Given the binary-tree-2 graph
    When executing query:
      """
      MATCH (a:A)-[:KNOWS]->(b)-->(c)
      OPTIONAL MATCH (a)-[r:KNOWS]->(c)
      WITH c WHERE r IS NOT NULL
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'b2'   |
    And no side effects

  
  Scenario: [12] Handling triadic friend of a friend that is a friend with different relationship type
    Given the binary-tree-1 graph
    When executing query:
      """
      MATCH (a:A)-[:KNOWS]->(b)-->(c)
      OPTIONAL MATCH (a)-[r:FOLLOWS]->(c)
      WITH c WHERE r IS NOT NULL
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'b3'   |
    And no side effects

  
  Scenario: [13] Handling triadic friend of a friend that is a friend with superset of relationship type
    Given the binary-tree-1 graph
    When executing query:
      """
      MATCH (a:A)-[:KNOWS]->(b)-->(c)
      OPTIONAL MATCH (a)-[r]->(c)
      WITH c WHERE r IS NOT NULL
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'b2'   |
      | 'b3'   |
    And no side effects

  
  Scenario: [14] Handling triadic friend of a friend that is a friend with implicit subset of relationship type
    Given the binary-tree-1 graph
    When executing query:
      """
      MATCH (a:A)-->(b)-->(c)
      OPTIONAL MATCH (a)-[r:KNOWS]->(c)
      WITH c WHERE r IS NOT NULL
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'b1'   |
      | 'b2'   |
    And no side effects

  
  Scenario: [15] Handling triadic friend of a friend that is a friend with explicit subset of relationship type
    Given the binary-tree-1 graph
    When executing query:
      """
      MATCH (a:A)-[:KNOWS|FOLLOWS]->(b)-->(c)
      OPTIONAL MATCH (a)-[r:KNOWS]->(c)
      WITH c WHERE r IS NOT NULL
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'b1'   |
      | 'b2'   |
    And no side effects

  
  Scenario: [16] Handling triadic friend of a friend that is a friend with same labels
    Given the binary-tree-2 graph
    When executing query:
      """
      MATCH (a:A)-[:KNOWS]->(b:X)-->(c:X)
      OPTIONAL MATCH (a)-[r:KNOWS]->(c)
      WITH c WHERE r IS NOT NULL
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'b2'   |
    And no side effects

  
  Scenario: [17] Handling triadic friend of a friend that is a friend with different labels
    Given the binary-tree-2 graph
    When executing query:
      """
      MATCH (a:A)-[:KNOWS]->(b:X)-->(c:Y)
      OPTIONAL MATCH (a)-[r:KNOWS]->(c)
      WITH c WHERE r IS NOT NULL
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
    And no side effects

  
  Scenario: [18] Handling triadic friend of a friend that is a friend with implicit subset of labels
    Given the binary-tree-2 graph
    When executing query:
      """
      MATCH (a:A)-[:KNOWS]->(b)-->(c:X)
      OPTIONAL MATCH (a)-[r:KNOWS]->(c)
      WITH c WHERE r IS NOT NULL
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'b2'   |
    And no side effects

  
  Scenario: [19] Handling triadic friend of a friend that is a friend with implicit superset of labels
    Given the binary-tree-2 graph
    When executing query:
      """
      MATCH (a:A)-[:KNOWS]->(b:X)-->(c)
      OPTIONAL MATCH (a)-[r:KNOWS]->(c)
      WITH c WHERE r IS NOT NULL
      RETURN c.name
      """
    Then the result should be, in any order:
      | c.name |
      | 'b2'   |
    And no side effects
