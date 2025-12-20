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

Feature: Delete5 - Delete clause interoperation with built-in data types

  Scenario: [1] Delete node from a list
    Given an empty graph
    And having executed:
      """
      CREATE (u:User)
      CREATE (u)-[:FRIEND]->()
      CREATE (u)-[:FRIEND]->()
      CREATE (u)-[:FRIEND]->()
      CREATE (u)-[:FRIEND]->()
      """
    And parameters are:
      | friendIndex | 1 |
    When executing query:
      """
      MATCH (:User)-[:FRIEND]->(n)
      WITH collect(n) AS friends
      DETACH DELETE friends[$friendIndex]
      """
    Then the result should be empty
    And the side effects should be:
      | -nodes         | 1 |
      | -relationships | 1 |

  Scenario: [2] Delete relationship from a list
    Given an empty graph
    And having executed:
      """
      CREATE (u:User)
      CREATE (u)-[:FRIEND]->()
      CREATE (u)-[:FRIEND]->()
      CREATE (u)-[:FRIEND]->()
      CREATE (u)-[:FRIEND]->()
      """
    And parameters are:
      | friendIndex | 1 |
    When executing query:
      """
      MATCH (:User)-[r:FRIEND]->()
      WITH collect(r) AS friendships
      DETACH DELETE friendships[$friendIndex]
      """
    Then the result should be empty
    And the side effects should be:
      | -relationships | 1 |

  @skip
  Scenario: [3] Delete nodes from a map
    Given an empty graph
    And having executed:
      """
      CREATE (:User), (:User)
      """
    When executing query:
      """
      MATCH (u:User)
      WITH {key: u} AS nodes
      DELETE nodes.key
      """
    Then the result should be empty
    And the side effects should be:
      | -nodes  | 2 |
      | -labels | 1 |

  @skip
  Scenario: [4] Delete relationships from a map
    Given an empty graph
    And having executed:
      """
      CREATE (a:User), (b:User)
      CREATE (a)-[:R]->(b)
      CREATE (b)-[:R]->(a)
      """
    When executing query:
      """
      MATCH (:User)-[r]->(:User)
      WITH {key: r} AS rels
      DELETE rels.key
      """
    Then the result should be empty
    And the side effects should be:
      | -relationships | 2 |

  Scenario: [5] Detach delete nodes from nested map/list
    Given an empty graph
    And having executed:
      """
      CREATE (a:User), (b:User)
      CREATE (a)-[:R]->(b)
      CREATE (b)-[:R]->(a)
      """
    When executing query:
      """
      MATCH (u:User)
      WITH {key: collect(u)} AS nodeMap
      DETACH DELETE nodeMap.key[0]
      """
    Then the result should be empty
    And the side effects should be:
      | -nodes         | 1 |
      | -relationships | 2 |

  Scenario: [6] Delete relationships from nested map/list
    Given an empty graph
    And having executed:
      """
      CREATE (a:User), (b:User)
      CREATE (a)-[:R]->(b)
      CREATE (b)-[:R]->(a)
      """
    When executing query:
      """
      MATCH (:User)-[r]->(:User)
      WITH {key: {key: collect(r)}} AS rels
      DELETE rels.key.key[0]
      """
    Then the result should be empty
    And the side effects should be:
      | -relationships | 1 |

  @skip
  Scenario: [7] Delete paths from nested map/list
    Given an empty graph
    And having executed:
      """
      CREATE (a:User), (b:User)
      CREATE (a)-[:R]->(b)
      CREATE (b)-[:R]->(a)
      """
    When executing query:
      """
      MATCH p = (:User)-[r]->(:User)
      WITH {key: collect(p)} AS pathColls
      DELETE pathColls.key[0], pathColls.key[1]
      """
    Then the result should be empty
    And the side effects should be:
      | -nodes         | 2 |
      | -relationships | 2 |
      | -labels        | 1 |

  Scenario: [8] Failing when using undefined variable in DELETE
    Given any graph
    When executing query:
      """
      MATCH (a)
      DELETE x
      """
    Then a SyntaxError should be raised at compile time: UndefinedVariable

  @skip
  Scenario: [9] Failing when deleting an integer expression
    Given any graph
    When executing query:
      """
      MATCH ()
      DELETE 1 + 1
      """
    Then a SyntaxError should be raised at compile time: InvalidArgumentType
