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

Feature: Set1 - Set a Property

  Scenario: [1] Set a property
    Given any graph
    And having executed:
      """
      CREATE (:A {name: 'Andres'})
      """
    When executing query:
      """
      MATCH (n:A)
      WHERE n.name = 'Andres'
      SET n.name = 'Michael'
      RETURN n
      """
    Then the result should be, in any order:
      | n                      |
      | (:A {name: 'Michael'}) |
    And the side effects should be:
      | +properties | 1 |
      | -properties | 1 |

  Scenario: [2] Set a property to an expression
    Given an empty graph
    And having executed:
      """
      CREATE (:A {name: 'Andres'})
      """
    When executing query:
      """
      MATCH (n:A)
      WHERE n.name = 'Andres'
      SET n.name = n.name + ' was here'
      RETURN n
      """
    Then the result should be, in any order:
      | n                              |
      | (:A {name: 'Andres was here'}) |
    And the side effects should be:
      | +properties | 1 |
      | -properties | 1 |

  Scenario: [3] Set a property by selecting the node using a simple expression
    Given an empty graph
    And having executed:
      """
      CREATE (:A)
      """
    When executing query:
      """
      MATCH (n:A)
      SET (n).name = 'neo4j'
      RETURN n
      """
    Then the result should be, in any order:
      | n                    |
      | (:A {name: 'neo4j'}) |
    And the side effects should be:
      | +properties | 1 |

  Scenario: [4] Set a property by selecting the relationship using a simple expression
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:REL]->()
      """
    When executing query:
      """
      MATCH ()-[r:REL]->()
      SET (r).name = 'neo4j'
      RETURN r
      """
    Then the result should be, in any order:
      | r                      |
      | [:REL {name: 'neo4j'}] |
    And the side effects should be:
      | +properties | 1 |

  Scenario: [5] Adding a list property
    Given an empty graph
    And having executed:
      """
      CREATE (:A)
      """
    When executing query:
      """
      MATCH (n:A)
      SET n.numbers = [1, 2, 3]
      RETURN [i IN n.numbers | i / 2.0] AS x
      """
    Then the result should be, in any order:
      | x               |
      | [0.5, 1.0, 1.5] |
    And the side effects should be:
      | +properties | 1 |

  @skip
  Scenario: [6] Concatenate elements onto a list property
    Given any graph
    When executing query:
      """
      CREATE (a {numbers: [1, 2, 3]})
      SET a.numbers = a.numbers + [4, 5]
      RETURN a.numbers
      """
    Then the result should be, in any order:
      | a.numbers       |
      | [1, 2, 3, 4, 5] |
    And the side effects should be:
      | +nodes      | 1 |
      | +properties | 1 |

  @skip
  Scenario: [7] Concatenate elements in reverse onto a list property
    Given any graph
    When executing query:
      """
      CREATE (a {numbers: [3, 4, 5]})
      SET a.numbers = [1, 2] + a.numbers
      RETURN a.numbers
      """
    Then the result should be, in any order:
      | a.numbers       |
      | [1, 2, 3, 4, 5] |
    And the side effects should be:
      | +nodes      | 1 |
      | +properties | 1 |

  Scenario: [8] Ignore null when setting property
    Given an empty graph
    When executing query:
      """
      OPTIONAL MATCH (a:DoesNotExist)
      SET a.num = 42
      RETURN a
      """
    Then the result should be, in any order:
      | a    |
      | null |
    And no side effects

  Scenario: [9] Failing when using undefined variable in SET
    Given any graph
    When executing query:
      """
      MATCH (a)
      SET a.name = missing
      RETURN a
      """
    Then a SyntaxError should be raised at compile time: UndefinedVariable

  Scenario: [10] Failing when setting a list of maps as a property
    Given any graph
    When executing query:
      """
      CREATE (a)
      SET a.maplist = [{num: 1}]
      """
    Then a TypeError should be raised at runtime: InvalidPropertyType

  Scenario: [11] Set multiple node properties
    Given an empty graph
    And having executed:
      """
      CREATE (:X)
      """
    When executing query:
      """
      MATCH (n:X)
      SET n.name = 'A', n.name2 = 'B', n.num = 5
      RETURN n
      """
    Then the result should be, in any order:
      | n                                    |
      | (:X {name: 'A', name2: 'B', num: 5}) |
    And the side effects should be:
      | +properties | 3 |
