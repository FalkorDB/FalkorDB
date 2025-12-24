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

Feature: ExistentialSubquery3 - Nested existential subquery

  
  Scenario: [1] Nested simple existential subquery
    Given an empty graph
    And having executed:
      """
      CREATE (a:A {prop: 1})-[:R]->(b:B {prop: 1}), 
             (a)-[:R]->(:C {prop: 2}), 
             (a)-[:R]->(:D {prop: 3})
      """
    When executing query:
      """
      MATCH (n) WHERE exists {
        MATCH (m) WHERE exists {
          (n)-[]->(m) WHERE n.prop = m.prop
        }
        RETURN true
      }
      RETURN n
      """
    Then the result should be, in any order:
      | n             |
      | (:A {prop:1}) |
    And no side effects

  
  Scenario: [2] Nested full existential subquery
    Given an empty graph
    And having executed:
      """
      CREATE (a:A {prop: 1})-[:R]->(b:B {prop: 1}), 
             (a)-[:R]->(:C {prop: 2}), 
             (a)-[:R]->(:D {prop: 3})
      """
    When executing query:
      """
      MATCH (n) WHERE exists {
        MATCH (m) WHERE exists {
          MATCH (l)<-[:R]-(n)-[:R]->(m) RETURN true
        }
        RETURN true
      }
      RETURN n
      """
    Then the result should be, in any order:
      | n             |
      | (:A {prop:1}) |
    And no side effects
  
  
  Scenario: [3] Nested full existential subquery with pattern predicate
    Given an empty graph
    And having executed:
      """
      CREATE (a:A {prop: 1})-[:R]->(b:B {prop: 1}), 
             (a)-[:R]->(:C {prop: 2}), 
             (a)-[:R]->(:D {prop: 3})
      """
    When executing query:
      """
      MATCH (n) WHERE exists {
        MATCH (m) WHERE exists {
          MATCH (l) WHERE (l)<-[:R]-(n)-[:R]->(m) RETURN true
        }
        RETURN true
      }
      RETURN n
      """
    Then the result should be, in any order:
      | n             |
      | (:A {prop:1}) |
    And no side effects
