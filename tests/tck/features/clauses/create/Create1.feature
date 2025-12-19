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

Feature: Create1 - Creating nodes

  Scenario: [1] Create a single node
    Given any graph
    When executing query:
      """
      CREATE ()
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes | 1 |

  Scenario: [2] Create two nodes
    Given any graph
    When executing query:
      """
      CREATE (), ()
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes | 2 |

  Scenario: [3] Create a single node with a label
    Given an empty graph
    When executing query:
      """
      CREATE (:Label)
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes  | 1 |
      | +labels | 1 |

  Scenario: [4] Create two nodes with same label
    Given an empty graph
    When executing query:
      """
      CREATE (:Label), (:Label)
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes  | 2 |
      | +labels | 1 |

  Scenario: [5] Create a single node with multiple labels
    Given an empty graph
    When executing query:
      """
      CREATE (:A:B:C:D)
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes  | 1 |
      | +labels | 4 |

  Scenario: [6] Create three nodes with multiple labels
    Given an empty graph
    When executing query:
      """
      CREATE (:B:A:D), (:B:C), (:D:E:B)
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes  | 3 |
      | +labels | 5 |

  Scenario: [7] Create a single node with a property
    Given any graph
    When executing query:
      """
      CREATE ({created: true})
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes      | 1 |
      | +properties | 1 |

  Scenario: [8] Create a single node with a property and return it
    Given any graph
    When executing query:
      """
      CREATE (n {name: 'foo'})
      RETURN n.name AS p
      """
    Then the result should be, in any order:
      | p     |
      | 'foo' |
    And the side effects should be:
      | +nodes      | 1 |
      | +properties | 1 |

  Scenario: [9] Create a single node with two properties
    Given any graph
    When executing query:
      """
      CREATE (n {id: 12, name: 'foo'})
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes      | 1 |
      | +properties | 2 |

  Scenario: [10] Create a single node with two properties and return them
    Given any graph
    When executing query:
      """
      CREATE (n {id: 12, name: 'foo'})
      RETURN n.id AS id, n.name AS p
      """
    Then the result should be, in any order:
      | id | p     |
      | 12 | 'foo' |
    And the side effects should be:
      | +nodes      | 1 |
      | +properties | 2 |

  Scenario: [11] Create a single node with null properties should not return those properties
    Given any graph
    When executing query:
      """
      CREATE (n {id: 12, name: null})
      RETURN n.id AS id, n.name AS p
      """
    Then the result should be, in any order:
      | id | p    |
      | 12 | null |
    And the side effects should be:
      | +nodes      | 1 |
      | +properties | 1 |

  Scenario: [12] CREATE does not lose precision on large integers
    Given an empty graph
    When executing query:
      """
      CREATE (p:TheLabel {id: 4611686018427387905})
      RETURN p.id
      """
    Then the result should be, in any order:
      | p.id                |
      | 4611686018427387905 |
    And the side effects should be:
      | +nodes      | 1 |
      | +properties | 1 |
      | +labels     | 1 |

  Scenario: [13] Fail when creating a node that is already bound
    Given any graph
    When executing query:
      """
      MATCH (a)
      CREATE (a)
      """
    Then a SyntaxError should be raised at compile time: VariableAlreadyBound

  Scenario: [14] Fail when creating a node with properties that is already bound
    Given any graph
    When executing query:
      """
      MATCH (a)
      CREATE (a {name: 'foo'})
      RETURN a
      """
    Then a SyntaxError should be raised at compile time: VariableAlreadyBound

  Scenario: [15] Fail when adding a new label predicate on a node that is already bound 1
    Given an empty graph
    When executing query:
      """
      CREATE (n:Foo)-[:T1]->(),
             (n:Bar)-[:T2]->()
      """
    Then a SyntaxError should be raised at compile time: VariableAlreadyBound

  # Consider improve naming of this and the next three scenarios, they seem to test invariant nature of node patterns
  Scenario: [16] Fail when adding new label predicate on a node that is already bound 2
    Given an empty graph
    When executing query:
      """
      CREATE ()<-[:T2]-(n:Foo),
             (n:Bar)<-[:T1]-()
      """
    Then a SyntaxError should be raised at compile time: VariableAlreadyBound

  Scenario: [17] Fail when adding new label predicate on a node that is already bound 3
    Given an empty graph
    When executing query:
      """
      CREATE (n:Foo)
      CREATE (n:Bar)-[:OWNS]->(:Dog)
      """
    Then a SyntaxError should be raised at compile time: VariableAlreadyBound

  Scenario: [18] Fail when adding new label predicate on a node that is already bound 4
    Given an empty graph
    When executing query:
      """
      CREATE (n {})
      CREATE (n:Bar)-[:OWNS]->(:Dog)
      """
    Then a SyntaxError should be raised at compile time: VariableAlreadyBound

  Scenario: [19] Fail when adding new label predicate on a node that is already bound 5
    Given an empty graph
    When executing query:
      """
      CREATE (n:Foo)
      CREATE (n {})-[:OWNS]->(:Dog)
      """
    Then a SyntaxError should be raised at compile time: VariableAlreadyBound

  Scenario: [20] Fail when creating a node using undefined variable in pattern
    Given any graph
    When executing query:
      """
      CREATE (b {name: missing})
      RETURN b
      """
    Then a SyntaxError should be raised at compile time: UndefinedVariable
