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

Feature: Match1 - Match nodes

  Scenario: [1] Match non-existent nodes returns empty
    Given an empty graph
    When executing query:
      """
      MATCH (n)
      RETURN n
      """
    Then the result should be, in any order:
      | n |
    And no side effects

  Scenario: [2] Matching all nodes
    Given an empty graph
    And having executed:
      """
      CREATE (:A), (:B {name: 'b'}), ({name: 'c'})
      """
    When executing query:
      """
      MATCH (n)
      RETURN n
      """
    Then the result should be, in any order:
      | n                |
      | (:A)             |
      | (:B {name: 'b'}) |
      | ({name: 'c'})    |
    And no side effects

  Scenario: [3] Matching nodes using multiple labels
    Given an empty graph
    And having executed:
      """
      CREATE (:A:B:C), (:A:B), (:A:C), (:B:C),
             (:A), (:B), (:C),
             ({name: ':A:B:C'}), ({abc: 'abc'}), ()
      """
    When executing query:
      """
      MATCH (a:A:B)
      RETURN a
      """
    Then the result should be, in any order:
      | a        |
      | (:A:B)   |
      | (:A:B:C) |
    And no side effects

  Scenario: [4] Simple node inline property predicate
    Given an empty graph
    And having executed:
      """
      CREATE ({name: 'bar'}), ({name: 'monkey'}), ({firstname: 'bar'})
      """
    When executing query:
      """
      MATCH (n {name: 'bar'})
      RETURN n
      """
    Then the result should be, in any order:
      | n               |
      | ({name: 'bar'}) |
    And no side effects

  Scenario: [5] Use multiple MATCH clauses to do a Cartesian product
    Given an empty graph
    And having executed:
      """
      CREATE ({num: 1}),
        ({num: 2}),
        ({num: 3})
      """
    When executing query:
      """
      MATCH (n), (m)
      RETURN n.num AS n, m.num AS m
      """
    Then the result should be, in any order:
      | n | m |
      | 1 | 1 |
      | 1 | 2 |
      | 1 | 3 |
      | 2 | 1 |
      | 2 | 2 |
      | 2 | 3 |
      | 3 | 3 |
      | 3 | 1 |
      | 3 | 2 |
    And no side effects

  Scenario: [6] Fail when using parameter as node predicate in MATCH
    Given any graph
    When executing query:
      """
      MATCH (n $param)
      RETURN n
      """
    Then a SyntaxError should be raised at compile time: InvalidParameterUse

  Scenario Outline: [7] Fail when a relationship has the same variable in a preceding MATCH
    Given any graph
    When executing query:
      """
      MATCH <pattern>
      MATCH (r)
      RETURN r
      """
    Then a SyntaxError should be raised at compile time: VariableTypeConflict

    Examples:
      | pattern                                     |
      | ()-[r]-()                                   |
      | ()-[r]->()                                  |
      | ()<-[r]-()                                  |
      | (), ()-[r]-()                               |
      | ()-[r]-(), ()                               |
      | ()-[]-(), ()-[r]-()                         |
      | ()-[]-()-[r]-()                             |
      | ()-[]-()-[]-(), ()-[r]-()                   |
      | ()-[]-()-[]-(), ()-[r]-(), ()               |
      | ()-[]-()-[]-(), (), ()-[r]-()               |
      | (x), (a)-[q]-(b), (s), (s)-[r]->(t)<-[]-(b) |

  
  Scenario Outline: [8] Fail when a path has the same variable in a preceding MATCH
    Given any graph
    When executing query:
      """
      MATCH <pattern>
      MATCH (r)
      RETURN r
      """
    Then a SyntaxError should be raised at compile time: VariableTypeConflict

    Examples:
      | pattern                                    |
      | r = ()-[]-()                               |
      | r = ()-[]->()                              |
      | r = ()<-[]-()                              |
      | r = ()-[*]-()                              |
      | r = ()-[*]->()                             |
      | (), r = ()-[]-()                           |
      | (), r = ()-[]->()                          |
      | (), r = ()<-[]-()                          |
      | (), r = ()-[*]-()                          |
      | (), r = ()-[*]->()                         |
      | ()-[]-(), r = ()-[]-(), ()                 |
      | r = ()-[]-(), ()-[]-(), ()                 |
      | ()-[]-()<-[]-(), r = ()-[]-()              |
      | (x), r = (a)-[q]-(b), (s)-[p]-(t)-[]-(b)   |
      | (x), (a)-[q]-(b), r = (s)-[p]-(t)-[]-(b)   |
      | (x), (a)-[q]-(b), r = (s)-[p]->(t)<-[]-(b) |

  Scenario Outline: [9] Fail when a relationship has the same variable in the same pattern
    Given any graph
    When executing query:
      """
      MATCH <pattern>
      RETURN r
      """
    Then a SyntaxError should be raised at compile time: VariableTypeConflict

    Examples:
      | pattern                                         |
      | ()-[r]-(r)                                      |
      | ()-[r]->(r)                                     |
      | ()<-[r]-(r)                                     |
      | ()-[r]-()-[]-(r)                                |
      | ()-[r*]-()-[]-(r)                               |
      | ()-[r]-(), (r)                                  |
      | ()-[r]->(), (r)                                 |
      | ()<-[r]-(), (r)                                 |
      | ()-[r]-(), (r)-[]-()                            |
      | ()-[r]-(), ()-[]-(r)                            |
      | (s)-[r]-(t), (r)-[]-(t)                         |
      | (s)-[r]-(t), (s)-[]-(r)                         |
      | (), ()-[r]-(), (r)                              |
      | ()-[r]-(), (), (r)                              |
      | ()-[r]-(), (r), ()                              |
      | ()-[]-(), ()-[r]-(), (r)                        |
      | ()-[]-()-[r]-(), ()-[]-(r)                      |
      | ()-[]-()-[]-(), ()-[r]-(), (r)                  |
      | ()-[]-()-[r]-(), (r), ()-[]-()                  |
      | ()-[]-()-[r]-(), (), (r)-[]-()                  |
      | ()-[]-()-[r*]-(), (r), ()-[]-()                 |
      | ()-[*]-()-[r]-(), (), (r)-[]-()                 |
      | ()-[*]-()-[r]-(), (), (r)-[*]-()                |
      | ()-[*]-()-[r]-(), (), ()-[*]-(r)                |
      | (x), (a)-[r]-(b), (s), (s)-[]->(r)<-[]-(b)      |

  
  Scenario Outline: [10] Fail when a path has the same variable in the same pattern
    Given any graph
    When executing query:
      """
      MATCH <pattern>
      RETURN r
      """
    Then a SyntaxError should be raised at compile time: VariableTypeConflict

    Examples:
      | pattern                                         |
      | r = ()-[]-(), (r)                               |
      | r = ()-[]->(), (r)                              |
      | r = ()<-[]-(), (r)                              |
      | r = ()-[*]-(), (r)                              |
      | r = ()-[*]->(), (r)                             |
      | (), r = ()-[]-(), (r)                           |
      | (), r = ()-[]->(), (r)                          |
      | (), r = ()<-[]-(), (r)                          |
      | (), r = ()-[*]-(), (r)                          |
      | (), r = ()-[*]->(), (r)                         |
      | ()-[]-(), r = ()-[]-(), (), (r)                 |
      | r = ()-[]-(), ()-[]-(), (), (r)                 |
      | ()-[]-()<-[]-(), r = ()-[]-(), (r)              |
      | (x), r = (a)-[q]-(b), (s)-[p]-(t)-[]-(b), (r)   |
      | (x), (a)-[q]-(b), r = (s)-[p]-(t)-[]-(b), (r)   |
      | (x), (a)-[q]-(b), r = (s)-[p]->(t)<-[]-(b), (r) |
      | (x), r = (s)-[p]-(t)-[]-(b), (r), (a)-[q]-(b)   |
      | (x), r = (s)-[p]->(t)<-[]-(b), (r), (a)-[q]-(b) |
      | (x), r = (s)-[p]-(t)-[]-(b), (a)-[q]-(r)        |
      | (x), r = (s)-[p]->(t)<-[]-(b), (r)-[q]-(b)      |

  
  Scenario Outline: [11] Fail when matching a node variable bound to a value
    Given any graph
    When executing query:
      """
      WITH <invalid> AS n
      MATCH (n)
      RETURN n
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
