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

Feature: Create2 - Creating relationships

  Scenario: [1] Create two nodes and a single relationship in a single pattern
    Given any graph
    When executing query:
      """
      CREATE ()-[:R]->()
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes         | 2 |
      | +relationships | 1 |

  Scenario: [2] Create two nodes and a single relationship in separate patterns
    Given any graph
    When executing query:
      """
      CREATE (a), (b),
             (a)-[:R]->(b)
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes         | 2 |
      | +relationships | 1 |

  Scenario: [3] Create two nodes and a single relationship in separate clauses
    Given any graph
    When executing query:
      """
      CREATE (a)
      CREATE (b)
      CREATE (a)-[:R]->(b)
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes         | 2 |
      | +relationships | 1 |

  Scenario: [4] Create two nodes and a single relationship in the reverse direction
    Given an empty graph
    When executing query:
      """
      CREATE (:A)<-[:R]-(:B)
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes         | 2 |
      | +relationships | 1 |
      | +labels        | 2 |
    When executing control query:
      """
      MATCH (a:A)<-[:R]-(b:B)
      RETURN a, b
      """
    Then the result should be, in any order:
      | a    | b    |
      | (:A) | (:B) |

  Scenario: [5] Create a single relationship between two existing nodes
    Given an empty graph
    And having executed:
      """
      CREATE (:X)
      CREATE (:Y)
      """
    When executing query:
      """
      MATCH (x:X), (y:Y)
      CREATE (x)-[:R]->(y)
      """
    Then the result should be empty
    And the side effects should be:
      | +relationships | 1 |

  Scenario: [6] Create a single relationship between two existing nodes in the reverse direction
    Given an empty graph
    And having executed:
      """
      CREATE (:X)
      CREATE (:Y)
      """
    When executing query:
      """
      MATCH (x:X), (y:Y)
      CREATE (x)<-[:R]-(y)
      """
    Then the result should be empty
    And the side effects should be:
      | +relationships | 1 |
    When executing control query:
      """
      MATCH (x:X)<-[:R]-(y:Y)
      RETURN x, y
      """
    Then the result should be, in any order:
      | x    |  y   |
      | (:X) | (:Y) |

  Scenario: [7] Create a single node and a single self loop in a single pattern
    Given any graph
    When executing query:
      """
      CREATE (root)-[:LINK]->(root)
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes         | 1 |
      | +relationships | 1 |

  Scenario: [8] Create a single node and a single self loop in separate patterns
    Given any graph
    When executing query:
      """
      CREATE (root),
             (root)-[:LINK]->(root)
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes         | 1 |
      | +relationships | 1 |

  Scenario: [9] Create a single node and a single self loop in separate clauses
    Given any graph
    When executing query:
      """
      CREATE (root)
      CREATE (root)-[:LINK]->(root)
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes         | 1 |
      | +relationships | 1 |

  Scenario: [10] Create a single self loop on an existing node
    Given an empty graph
    And having executed:
      """
      CREATE (:Root)
      """
    When executing query:
      """
      MATCH (root:Root)
      CREATE (root)-[:LINK]->(root)
      """
    Then the result should be empty
    And the side effects should be:
      | +relationships | 1 |

  Scenario: [11] Create a single relationship and an end node on an existing starting node
    Given an empty graph
    And having executed:
      """
      CREATE (:Begin)
      """
    When executing query:
      """
      MATCH (x:Begin)
      CREATE (x)-[:TYPE]->(:End)
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes         | 1 |
      | +relationships | 1 |
      | +labels        | 1 |
    When executing control query:
      """
      MATCH (x:Begin)-[:TYPE]->(y:End)
      RETURN x, y
      """
    Then the result should be, in any order:
      | x        | y      |
      | (:Begin) | (:End) |

  Scenario: [12] Create a single relationship and a starting node on an existing end node
    Given an empty graph
    And having executed:
      """
      CREATE (:End)
      """
    When executing query:
      """
      MATCH (x:End)
      CREATE (:Begin)-[:TYPE]->(x)
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes         | 1 |
      | +relationships | 1 |
      | +labels        | 1 |
    When executing control query:
      """
      MATCH (x:Begin)-[:TYPE]->(y:End)
      RETURN x, y
      """
    Then the result should be, in any order:
      | x        | y      |
      | (:Begin) | (:End) |

  Scenario: [13] Create a single relationship with a property
    Given any graph
    When executing query:
      """
      CREATE ()-[:R {num: 42}]->()
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes         | 2 |
      | +relationships | 1 |
      | +properties    | 1 |

  Scenario: [14] Create a single relationship with a property and return it
    Given any graph
    When executing query:
      """
      CREATE ()-[r:R {num: 42}]->()
      RETURN r.num AS num
      """
    Then the result should be, in any order:
      | num |
      | 42  |
    And the side effects should be:
      | +nodes         | 2 |
      | +relationships | 1 |
      | +properties    | 1 |

  Scenario: [15] Create a single relationship with two properties
    Given any graph
    When executing query:
      """
      CREATE ()-[:R {id: 12, name: 'foo'}]->()
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes         | 2 |
      | +relationships | 1 |
      | +properties    | 2 |

  Scenario: [16] Create a single relationship with two properties and return them
    Given any graph
    When executing query:
      """
      CREATE ()-[r:R {id: 12, name: 'foo'}]->()
      RETURN r.id AS id, r.name AS name
      """
    Then the result should be, in any order:
      | id | name  |
      | 12 | 'foo' |
    And the side effects should be:
      | +nodes         | 2 |
      | +relationships | 1 |
      | +properties    | 2 |

  Scenario: [17] Create a single relationship with null properties should not return those properties
    Given any graph
    When executing query:
      """
      CREATE ()-[r:X {id: 12, name: null}]->()
      RETURN r.id, r.name AS name
      """
    Then the result should be, in any order:
      | r.id | name |
      | 12   | null |
    And the side effects should be:
      | +nodes         | 2 |
      | +relationships | 1 |
      | +properties    | 1 |

  Scenario: [18] Fail when creating a relationship without a type
    Given any graph
    When executing query:
      """
      CREATE ()-->()
      """
    Then a SyntaxError should be raised at compile time: NoSingleRelationshipType

  Scenario: [19] Fail when creating a relationship without a direction
    Given any graph
    When executing query:
      """
      CREATE (a)-[:FOO]-(b)
      """
    Then a SyntaxError should be raised at compile time: RequiresDirectedRelationship

  Scenario: [20] Fail when creating a relationship with two directions
    Given any graph
    When executing query:
      """
      CREATE (a)<-[:FOO]->(b)
      """
    Then a SyntaxError should be raised at compile time: RequiresDirectedRelationship

  Scenario: [21] Fail when creating a relationship with more than one type
    Given any graph
    When executing query:
      """
      CREATE ()-[:A|:B]->()
      """
    Then a SyntaxError should be raised at compile time: NoSingleRelationshipType

  
  Scenario: [22] Fail when creating a variable-length relationship
    Given any graph
    When executing query:
      """
      CREATE ()-[:FOO*2]->()
      """
    Then a SyntaxError should be raised at compile time: CreatingVarLength

  Scenario: [23] Fail when creating a relationship that is already bound
    Given any graph
    When executing query:
      """
      MATCH ()-[r]->()
      CREATE ()-[r]->()
      """
    Then a SyntaxError should be raised at compile time: VariableAlreadyBound

  Scenario: [24] Fail when creating a relationship using undefined variable in pattern
    Given any graph
    When executing query:
      """
      MATCH (a)
      CREATE (a)-[:KNOWS]->(b {name: missing})
      RETURN b
      """
    Then a SyntaxError should be raised at compile time: UndefinedVariable
