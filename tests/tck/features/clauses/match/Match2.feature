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

Feature: Match2 - Match relationships

  Scenario: [1] Match non-existent relationships returns empty
    Given an empty graph
    When executing query:
      """
      MATCH ()-[r]->()
      RETURN r
      """
    Then the result should be, in any order:
      | r |
    And no side effects

  Scenario: [2] Matching a relationship pattern using a label predicate on both sides
    Given an empty graph
    And having executed:
      """
      CREATE (:A)-[:T1]->(:B),
             (:B)-[:T2]->(:A),
             (:B)-[:T3]->(:B),
             (:A)-[:T4]->(:A)
      """
    When executing query:
      """
      MATCH (:A)-[r]->(:B)
      RETURN r
      """
    Then the result should be, in any order:
      | r     |
      | [:T1] |
    And no side effects

  Scenario: [3] Matching a self-loop with an undirected relationship pattern
    Given an empty graph
    And having executed:
      """
      CREATE (a)
      CREATE (a)-[:T]->(a)
      """
    When executing query:
      """
      MATCH ()-[r]-()
      RETURN type(r) AS r
      """
    Then the result should be, in any order:
      | r   |
      | 'T' |
    And no side effects

  Scenario: [4] Matching a self-loop with a directed relationship pattern
    Given an empty graph
    And having executed:
      """
      CREATE (a)
      CREATE (a)-[:T]->(a)
      """
    When executing query:
      """
      MATCH ()-[r]->()
      RETURN type(r) AS r
      """
    Then the result should be, in any order:
      | r   |
      | 'T' |
    And no side effects

  Scenario: [5] Match relationship with inline property value
    Given an empty graph
    And having executed:
      """
      CREATE (:A)<-[:KNOWS {name: 'monkey'}]-()-[:KNOWS {name: 'woot'}]->(:B)
      """
    When executing query:
      """
      MATCH (node)-[r:KNOWS {name: 'monkey'}]->(a)
      RETURN a
      """
    Then the result should be, in any order:
      | a    |
      | (:A) |
    And no side effects

  Scenario: [6] Match relationships with multiple types
    Given an empty graph
    And having executed:
      """
      CREATE (a {name: 'A'}),
        (b {name: 'B'}),
        (c {name: 'C'}),
        (a)-[:KNOWS]->(b),
        (a)-[:HATES]->(c),
        (a)-[:WONDERS]->(c)
      """
    When executing query:
      """
      MATCH (n)-[r:KNOWS|HATES]->(x)
      RETURN r
      """
    Then the result should be, in any order:
      | r        |
      | [:KNOWS] |
      | [:HATES] |
    And no side effects

  Scenario: [7] Matching twice with conflicting relationship types on same relationship
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:T]->()
      """
    When executing query:
      """
      MATCH (a1)-[r:T]->()
      WITH r, a1
      MATCH (a1)-[r:Y]->(b2)
      RETURN a1, r, b2
      """
    Then the result should be, in any order:
      | a1 | r | b2 |
    And no side effects

  Scenario: [8] Fail when using parameter as relationship predicate in MATCH
    Given any graph
    When executing query:
      """
      MATCH ()-[r:FOO $param]->()
      RETURN r
      """
    Then a SyntaxError should be raised at compile time: InvalidParameterUse

  Scenario Outline: [9] Fail when a node has the same variable in a preceding MATCH
    Given any graph
    When executing query:
      """
      MATCH <pattern>
      MATCH ()-[r]-()
      RETURN r
      """
    Then a SyntaxError should be raised at compile time: VariableTypeConflict

    Examples:
      | pattern                                    |
      | (r)                                        |
      | (r)-[]-()                                  |
      | (r)-[]->()                                 |
      | (r)<-[]-()                                 |
      | (r)-[]-(r)                                 |
      | ()-[]->(r)                                 |
      | ()<-[]-(r)                                 |
      | ()-[]-(r)                                  |
      | (r)-[]->(r)                                |
      | (r)<-[]-(r)                                |
      | (r)-[]-()-[]-()                            |
      | ()-[]-(r)-[]-()                            |
      | (r)-[]-()-[*]-()                           |
      | ()-[]-(r)-[*]-()                           |
      | (r), ()-[]-()                              |
      | (r)-[]-(), ()-[]-()                        |
      | ()-[]-(r), ()-[]-()                        |
      | ()-[]-(), (r)-[]-()                        |
      | ()-[]-(), ()-[]-(r)                        |
      | (r)-[]-(t), (s)-[]-(t)                     |
      | (s)-[]-(r), (s)-[]-(t)                     |
      | (s)-[]-(t), (r)-[]-(t)                     |
      | (s)-[]-(t), (s)-[]-(r)                     |
      | (s), (a)-[q]-(b), (r), (s)-[]-(t)-[]-(b)   |
      | (s), (a)-[q]-(b), (r), (s)-[]->(t)<-[]-(b) |
      | (s), (a)-[q]-(b), (t), (s)-[]->(r)<-[]-(b) |

  
  Scenario Outline: [10] Fail when a path has the same variable in a preceding MATCH
    Given any graph
    When executing query:
      """
      MATCH <pattern>
      MATCH ()-[r]-()
      RETURN r
      """
    Then a SyntaxError should be raised at compile time: VariableTypeConflict

    Examples:
      | pattern                                     |
      | r = ()-[]-()                                |
      | r = ()-[]->()                               |
      | r = ()<-[]-()                               |
      | r = ()-[*]-()                               |
      | r = ()-[*]->()                              |
      | r = ()<-[*]-()                              |
      | r = ()-[p*]-()                              |
      | r = ()-[p*]->()                             |
      | r = ()<-[p*]-()                             |
      | (), r = ()-[]-()                            |
      | ()-[]-(), r = ()-[]-()                      |
      | ()-[]->(), r = ()<-[]-()                    |
      | ()<-[]-(), r = ()-[]->()                    |
      | ()-[*]->(), r = ()<-[]-()                   |
      | ()<-[p*]-(), r = ()-[*]->()                 |
      | (x), (a)-[q]-(b), (r), (s)-[]->(t)<-[]-(b)  |
      | (x), (a)-[q]-(b), r = (s)-[p]->(t)<-[]-(b)  |
      | (x), (a)-[q*]-(b), r = (s)-[p]->(t)<-[]-(b) |
      | (x), (a)-[q]-(b), r = (s)-[p*]->(t)<-[]-(b) |

  Scenario Outline: [11] Fail when a node has the same variable in the same pattern
    Given any graph
    When executing query:
      """
      MATCH <pattern>
      RETURN r
      """
    Then a SyntaxError should be raised at compile time: VariableTypeConflict

    Examples:
      | pattern                                     |
      | (r)-[r]-()                                  |
      | (r)-[r]->()                                 |
      | (r)<-[r]-()                                 |
      | (r)-[r]-(r)                                 |
      | (r)-[r]->(r)                                |
      | (r)<-[r]-(r)                                |
      | (r)-[]-()-[r]-()                            |
      | ()-[]-(r)-[r]-()                            |
      | (r)-[]-()-[r*]-()                           |
      | ()-[]-(r)-[r*]-()                           |
      | (r), ()-[r]-()                              |
      | (r)-[]-(), ()-[r]-()                        |
      | ()-[]-(r), ()-[r]-()                        |
      | (r)-[]-(t), (s)-[r]-(t)                     |
      | (s)-[]-(r), (s)-[r]-(t)                     |
      | (r), (a)-[q]-(b), (s), (s)-[r]-(t)-[]-(b)   |
      | (r), (a)-[q]-(b), (s), (s)-[r]->(t)<-[]-(b) |

  
  Scenario Outline: [12] Fail when a path has the same variable in the same pattern
    Given any graph
    When executing query:
      """
      MATCH <pattern>
      RETURN r
      """
    Then a SyntaxError should be raised at compile time: VariableTypeConflict

    Examples:
      | pattern                                                |
      | r = ()-[]-(), ()-[r]-()                                |
      | r = ()-[]-(), ()-[r*]-()                               |
      | r = (a)-[p]-(s)-[]-(b), (s)-[]-(t), (t), (t)-[r]-(b)   |
      | r = (a)-[p]-(s)-[]-(b), (s)-[]-(t), (t), (t)-[r*]-(b)  |
      | r = (a)-[p]-(s)-[*]-(b), (s)-[]-(t), (t), (t)-[r*]-(b) |
      | (a)-[p]-(s)-[]-(b), r = (s)-[]-(t), (t), (t)-[r*]-(b)  |
      | (a)-[p]-(s)-[]-(b), r = (s)-[*]-(t), (t), (t)-[r]-(b)  |
      | (a)-[p]-(s)-[]-(b), r = (s)-[*]-(t), (t), (t)-[r*]-(b) |

  
  Scenario Outline: [13] Fail when matching a relationship variable bound to a value
    Given any graph
    When executing query:
      """
      WITH <invalid> AS r
      MATCH ()-[r]-()
      RETURN r
      """
    Then a SyntaxError should be raised at compile time: VariableTypeConflict

    Examples:
      | invalid |
      | true    |
      | 123     |
      | 123.4   |
      | 'foo'   |
      | []      |
      | [10]    |
      | {x: 1}  |
      | {x: []} |
