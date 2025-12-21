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

Feature: Remove3 - Persistence of remove clause side effects

  @skip
  Scenario: [1] Limiting to zero results after removing a property from nodes affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE (:N {num: 42})
      """
    When executing query:
      """
      MATCH (n:N)
      REMOVE n.num
      RETURN n
      LIMIT 0
      """
    Then the result should be, in any order:
      | n |
    And the side effects should be:
      | -properties | 1 |

  @skip
  Scenario: [2] Skipping all results after removing a property from nodes affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE (:N {num: 42})
      """
    When executing query:
      """
      MATCH (n:N)
      REMOVE n.num
      RETURN n
      SKIP 1
      """
    Then the result should be, in any order:
      | n |
    And the side effects should be:
      | -properties | 1 |

  @skip
  Scenario: [3] Skipping and limiting to a few results after removing a property from nodes affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE (:N {name: 'a', num: 42})
      CREATE (:N {name: 'a', num: 42})
      CREATE (:N {name: 'a', num: 42})
      CREATE (:N {name: 'a', num: 42})
      CREATE (:N {name: 'a', num: 42})
      """
    When executing query:
      """
      MATCH (n:N)
      REMOVE n.name
      RETURN n.num AS num
      SKIP 2 LIMIT 2
      """
    Then the result should be, in any order:
      | num |
      | 42  |
      | 42  |
    And the side effects should be:
      | -properties | 5 |

  @skip
  Scenario: [4] Skipping zero results and limiting to all results after removing a property from nodes does not affect the result set nor the side effects
    Given an empty graph
    And having executed:
      """
      CREATE (:N {name: 'a', num: 42})
      CREATE (:N {name: 'a', num: 42})
      CREATE (:N {name: 'a', num: 42})
      CREATE (:N {name: 'a', num: 42})
      CREATE (:N {name: 'a', num: 42})
      """
    When executing query:
      """
      MATCH (n:N)
      REMOVE n.name
      RETURN n.num AS num
      SKIP 0 LIMIT 5
      """
    Then the result should be, in any order:
      | num |
      | 42  |
      | 42  |
      | 42  |
      | 42  |
      | 42  |
    And the side effects should be:
      | -properties | 5 |

  @skip
  Scenario: [5] Filtering after removing a property from nodes affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE (:N {name: 'a', num: 1})
      CREATE (:N {name: 'a', num: 2})
      CREATE (:N {name: 'a', num: 3})
      CREATE (:N {name: 'a', num: 4})
      CREATE (:N {name: 'a', num: 5})
      """
    When executing query:
      """
      MATCH (n:N)
      REMOVE n.name
      WITH n
      WHERE n.num % 2 = 0
      RETURN n.num AS num
      """
    Then the result should be, in any order:
      | num |
      | 2   |
      | 4   |
    And the side effects should be:
      | -properties | 5 |

  @skip
  Scenario: [6] Aggregating in `RETURN` after removing a property from nodes affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE (:N {name: 'a', num: 1})
      CREATE (:N {name: 'a', num: 2})
      CREATE (:N {name: 'a', num: 3})
      CREATE (:N {name: 'a', num: 4})
      CREATE (:N {name: 'a', num: 5})
      """
    When executing query:
      """
      MATCH (n:N)
      REMOVE n.name
      RETURN sum(n.num) AS sum
      """
    Then the result should be, in any order:
      | sum   |
      | 15.0  |
    And the side effects should be:
      | -properties | 5 |

  @skip
  Scenario: [7] Aggregating in `WITH` after removing a property from nodes affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE (:N {name: 'a', num: 1})
      CREATE (:N {name: 'a', num: 2})
      CREATE (:N {name: 'a', num: 3})
      CREATE (:N {name: 'a', num: 4})
      CREATE (:N {name: 'a', num: 5})
      """
    When executing query:
      """
      MATCH (n:N)
      REMOVE n.name
      WITH sum(n.num) AS sum
      RETURN sum
      """
    Then the result should be, in any order:
      | sum   |
      | 15.0  |
    And the side effects should be:
      | -properties | 5 |

  @skip
  Scenario: [8] Limiting to zero results after removing a label from nodes affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE (:N {num: 42})
      """
    When executing query:
      """
      MATCH (n:N)
      REMOVE n:N
      RETURN n
      LIMIT 0
      """
    Then the result should be, in any order:
      | n |
    And the side effects should be:
      | -labels | 1 |

  @skip
  Scenario: [9] Skipping all results after removing a label from nodes affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE (:N {num: 42})
      """
    When executing query:
      """
      MATCH (n:N)
      REMOVE n:N
      RETURN n
      SKIP 1
      """
    Then the result should be, in any order:
      | n |
    And the side effects should be:
      | -labels | 1 |

  @skip
  Scenario: [10] Skipping and limiting to a few results after removing a label from nodes affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE (:N {num: 42})
      CREATE (:N {num: 42})
      CREATE (:N {num: 42})
      CREATE (:N {num: 42})
      CREATE (:N {num: 42})
      """
    When executing query:
      """
      MATCH (n:N)
      REMOVE n:N
      RETURN n.num AS num
      SKIP 2 LIMIT 2
      """
    Then the result should be, in any order:
      | num |
      | 42  |
      | 42  |
    And the side effects should be:
      | -labels | 1 |

  @skip
  Scenario: [11] Skipping zero result and limiting to all results after removing a label from nodes does not affect the result set nor the side effects
    Given an empty graph
    And having executed:
      """
      CREATE (:N {num: 42})
      CREATE (:N {num: 42})
      CREATE (:N {num: 42})
      CREATE (:N {num: 42})
      CREATE (:N {num: 42})
      """
    When executing query:
      """
      MATCH (n:N)
      REMOVE n:N
      RETURN n.num AS num
      SKIP 0 LIMIT 5
      """
    Then the result should be, in any order:
      | num |
      | 42  |
      | 42  |
      | 42  |
      | 42  |
      | 42  |
    And the side effects should be:
      | -labels | 1 |

  @skip
  Scenario: [12] Filtering after removing a label from nodes affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE (:N {num: 1})
      CREATE (:N {num: 2})
      CREATE (:N {num: 3})
      CREATE (:N {num: 4})
      CREATE (:N {num: 5})
      """
    When executing query:
      """
      MATCH (n:N)
      REMOVE n:N
      WITH n
      WHERE n.num % 2 = 0
      RETURN n.num AS num
      """
    Then the result should be, in any order:
      | num |
      | 2   |
      | 4   |
    And the side effects should be:
      | -labels | 1 |

  @skip
  Scenario: [13] Aggregating in `RETURN` after removing a label from nodes affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE (:N {num: 1})
      CREATE (:N {num: 2})
      CREATE (:N {num: 3})
      CREATE (:N {num: 4})
      CREATE (:N {num: 5})
      """
    When executing query:
      """
      MATCH (n:N)
      REMOVE n:N
      RETURN sum(n.num) AS sum
      """
    Then the result should be, in any order:
      | sum   |
      | 15.0  |
    And the side effects should be:
      | -labels | 1 |

  @skip
  Scenario: [14] Aggregating in `WITH` after removing a label from nodes affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE (:N {num: 1})
      CREATE (:N {num: 2})
      CREATE (:N {num: 3})
      CREATE (:N {num: 4})
      CREATE (:N {num: 5})
      """
    When executing query:
      """
      MATCH (n:N)
      REMOVE n:N
      WITH sum(n.num) AS sum
      RETURN sum
      """
    Then the result should be, in any order:
      | sum   |
      | 15.0  |
    And the side effects should be:
      | -labels | 1 |

  @skip
  Scenario: [15] Limiting to zero results after removing a property from relationships affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE ()-[r:R {num: 42}]->()
      """
    When executing query:
      """
      MATCH ()-[r:R]->()
      REMOVE r.num
      RETURN r
      LIMIT 0
      """
    Then the result should be, in any order:
      | r |
    And the side effects should be:
      | -properties | 1 |

  @skip
  Scenario: [16] Skipping all results after removing a property from relationships affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE ()-[r:R {num: 42}]->()
      """
    When executing query:
      """
      MATCH ()-[r:R]->()
      REMOVE r.num
      RETURN r
      SKIP 1
      """
    Then the result should be, in any order:
      | r |
    And the side effects should be:
      | -properties | 1 |

  @skip
  Scenario: [17] Skipping and limiting to a few results after removing a property from relationships affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:R {name: 'a', num: 42}]->()
      CREATE ()-[:R {name: 'a', num: 42}]->()
      CREATE ()-[:R {name: 'a', num: 42}]->()
      CREATE ()-[:R {name: 'a', num: 42}]->()
      CREATE ()-[:R {name: 'a', num: 42}]->()
      """
    When executing query:
      """
      MATCH ()-[r:R]->()
      REMOVE r.name
      RETURN r.num AS num
      SKIP 2 LIMIT 2
      """
    Then the result should be, in any order:
      | num |
      | 42  |
      | 42  |
    And the side effects should be:
      | -properties | 5 |

  @skip
  Scenario: [18] Skipping zero result and limiting to all results after removing a property from relationships does not affect the result set nor the side effects
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:R {name: 'a', num: 42}]->()
      CREATE ()-[:R {name: 'a', num: 42}]->()
      CREATE ()-[:R {name: 'a', num: 42}]->()
      CREATE ()-[:R {name: 'a', num: 42}]->()
      CREATE ()-[:R {name: 'a', num: 42}]->()
      """
    When executing query:
      """
      MATCH ()-[r:R]->()
      REMOVE r.name
      RETURN r.num AS num
      SKIP 0 LIMIT 5
      """
    Then the result should be, in any order:
      | num |
      | 42  |
      | 42  |
      | 42  |
      | 42  |
      | 42  |
    And the side effects should be:
      | -properties | 5 |

  @skip
  Scenario: [19] Filtering after removing a property from relationships affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:R {name: 'a', num: 1}]->()
      CREATE ()-[:R {name: 'a', num: 2}]->()
      CREATE ()-[:R {name: 'a', num: 3}]->()
      CREATE ()-[:R {name: 'a', num: 4}]->()
      CREATE ()-[:R {name: 'a', num: 5}]->()
      """
    When executing query:
      """
      MATCH ()-[r:R]->()
      REMOVE r.name
      WITH r
      WHERE r.num % 2 = 0
      RETURN r.num AS num
      """
    Then the result should be, in any order:
      | num |
      | 2   |
      | 4   |
    And the side effects should be:
      | -properties | 5 |

  @skip
  Scenario: [20] Aggregating in `RETURN` after removing a property from relationships affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:R {name: 'a', num: 1}]->()
      CREATE ()-[:R {name: 'a', num: 2}]->()
      CREATE ()-[:R {name: 'a', num: 3}]->()
      CREATE ()-[:R {name: 'a', num: 4}]->()
      CREATE ()-[:R {name: 'a', num: 5}]->()
      """
    When executing query:
      """
      MATCH ()-[r:R]->()
      REMOVE r.name
      RETURN sum(r.num) AS sum
      """
    Then the result should be, in any order:
      | sum   |
      | 15.0  |
    And the side effects should be:
      | -properties | 5 |

  @skip
  Scenario: [21] Aggregating in `WITH` after removing a property from relationships affects the result set but not the side effects
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:R {name: 'a', num: 1}]->()
      CREATE ()-[:R {name: 'a', num: 2}]->()
      CREATE ()-[:R {name: 'a', num: 3}]->()
      CREATE ()-[:R {name: 'a', num: 4}]->()
      CREATE ()-[:R {name: 'a', num: 5}]->()
      """
    When executing query:
      """
      MATCH ()-[r:R]->()
      REMOVE r.name
      WITH sum(r.num) AS sum
      RETURN sum
      """
    Then the result should be, in any order:
      | sum   |
      | 15.0  |
    And the side effects should be:
      | -properties | 5 |
