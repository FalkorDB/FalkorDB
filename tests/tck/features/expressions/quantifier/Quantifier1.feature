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

Feature: Quantifier1 - None quantifier

  Scenario: [1] None quantifier is always true on empty list
    Given any graph
    When executing query:
      """
      RETURN none(x IN [] WHERE true) AS a, none(x IN [] WHERE false) AS b, none(x IN [] WHERE x) AS c
      """
    Then the result should be, in any order:
      | a    | b    | c    |
      | true | true | true |
    And no side effects

  Scenario Outline: [2] None quantifier on list literal containing booleans
    Given any graph
    When executing query:
      """
      RETURN none(x IN <list> WHERE <condition>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | list                   | condition | result |
      | []                     | x         | true   |
      | [true]                 | x         | false  |
      | [false]                | x         | true   |
      | [true, false]          | x         | false  |
      | [false, true]          | x         | false  |
      | [true, false, true]    | x         | false  |
      | [false, true, false]   | x         | false  |
      | [true, true, true]     | x         | false  |
      | [false, false, false]  | x         | true   |

  Scenario Outline: [3] None quantifier on list literal containing integers
    Given any graph
    When executing query:
      """
      RETURN none(x IN <list> WHERE <condition>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | list                   | condition | result |
      | []                     | x = 2     | true   |
      | [1]                    | x = 2     | true   |
      | [1, 3]                 | x = 2     | true   |
      | [1, 3, 20, 5000]       | x = 2     | true   |
      | [20, 3, 5000, -2]      | x = 2     | true   |
      | [2]                    | x = 2     | false  |
      | [1, 2]                 | x = 2     | false  |
      | [1, 2, 3]              | x = 2     | false  |
      | [2, 2]                 | x = 2     | false  |
      | [2, 3]                 | x = 2     | false  |
      | [3, 2, 3]              | x = 2     | false  |
      | [2, 3, 2]              | x = 2     | false  |
      | [2, -10, 3, 9, 0]      | x < 10    | false  |
      | [2, -10, 3, 2, 10]     | x < 10    | false  |
      | [2, -10, 3, 21, 10]    | x < 10    | false  |
      | [200, -10, 36, 21, 10] | x < 10    | false  |
      | [200, 15, 36, 21, 10]  | x < 10    | true   |

  Scenario Outline: [4] None quantifier on list literal containing floats
    Given any graph
    When executing query:
      """
      RETURN none(x IN <list> WHERE <condition>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | list                       | condition | result |
      | []                         | x = 2.1   | true   |
      | [1.1]                      | x = 2.1   | true   |
      | [1.1, 3.5]                 | x = 2.1   | true   |
      | [1.1, 3.5, 20.0, 50.42435] | x = 2.1   | true   |
      | [20.0, 3.4, 50.2, -2.1]    | x = 2.1   | true   |
      | [2.1]                      | x = 2.1   | false  |
      | [1.43, 2.1]                | x = 2.1   | false  |
      | [1.43, 2.1, 3.5]           | x = 2.1   | false  |
      | [2.1, 2.1]                 | x = 2.1   | false  |
      | [2.1, 3.5]                 | x = 2.1   | false  |
      | [3.5, 2.1, 3.5]            | x = 2.1   | false  |
      | [2.1, 3.5, 2.1]            | x = 2.1   | false  |

  Scenario Outline: [5] None quantifier on list literal containing strings
    Given any graph
    When executing query:
      """
      RETURN none(x IN <list> WHERE <condition>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | list                  | condition   | result |
      | []                    | size(x) = 3 | true   |
      | ['abc']               | size(x) = 3 | false  |
      | ['ef']                | size(x) = 3 | true   |
      | ['abc', 'ef']         | size(x) = 3 | false  |
      | ['ef', 'abc']         | size(x) = 3 | false  |
      | ['abc', 'ef', 'abc']  | size(x) = 3 | false  |
      | ['ef', 'abc', 'ef']   | size(x) = 3 | false  |
      | ['abc', 'abc', 'abc'] | size(x) = 3 | false  |
      | ['ef', 'ef', 'ef']    | size(x) = 3 | true   |

  Scenario Outline: [6] None quantifier on list literal containing lists
    Given any graph
    When executing query:
      """
      RETURN none(x IN <list> WHERE <condition>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | list                              | condition   | result |
      | []                                | size(x) = 3 | true   |
      | [[1, 2, 3]]                       | size(x) = 3 | false  |
      | [['a']]                           | size(x) = 3 | true   |
      | [[1, 2, 3], ['a']]                | size(x) = 3 | false  |
      | [['a'], [1, 2, 3]]                | size(x) = 3 | false  |
      | [[1, 2, 3], ['a'], [1, 2, 3]]     | size(x) = 3 | false  |
      | [['a'], [1, 2, 3], ['a']]         | size(x) = 3 | false  |
      | [[1, 2, 3], [1, 2, 3], [1, 2, 3]] | size(x) = 3 | false  |
      | [['a'], ['a'], ['a']]             | size(x) = 3 | true   |

  Scenario Outline: [7] None quantifier on list literal containing maps
    Given any graph
    When executing query:
      """
      RETURN none(x IN <list> WHERE <condition>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | list                                       | condition | result |
      | []                                         | x.a = 2   | true   |
      | [{a: 2, b: 5}]                             | x.a = 2   | false  |
      | [{a: 4}]                                   | x.a = 2   | true   |
      | [{a: 2, b: 5}, {a: 4}]                     | x.a = 2   | false  |
      | [{a: 4}, {a: 2, b: 5}]                     | x.a = 2   | false  |
      | [{a: 2, b: 5}, {a: 4}, {a: 2, b: 5}]       | x.a = 2   | false  |
      | [{a: 4}, {a: 2, b: 5}, {a: 4}]             | x.a = 2   | false  |
      | [{a: 2, b: 5}, {a: 2, b: 5}, {a: 2, b: 5}] | x.a = 2   | false  |
      | [{a: 4}, {a: 4}, {a: 4}]                   | x.a = 2   | true   |

  Scenario: [8] None quantifier on list containing nodes
    Given an empty graph
    And having executed:
      """
      CREATE (s1:SRelationships), (s2:SNodes)
      CREATE (a:A {name: 'a'}), (b:B {name: 'b'})
      CREATE (aa:A {name: 'a'}), (ab:B {name: 'b'}),
             (ba:A {name: 'a'}), (bb:B {name: 'b'})
      CREATE (aaa:A {name: 'a'}), (aab:B {name: 'b'}),
             (aba:A {name: 'a'}), (abb:B {name: 'b'}),
             (baa:A {name: 'a'}), (bab:B {name: 'b'}),
             (bba:A {name: 'a'}), (bbb:B {name: 'b'})
      CREATE (s1)-[:I]->(s2),
             (s2)-[:RA {name: 'a'}]->(a), (s2)-[:RB {name: 'b'}]->(b)
      CREATE (a)-[:RA {name: 'a'}]->(aa), (a)-[:RB {name: 'b'}]->(ab),
             (b)-[:RA {name: 'a'}]->(ba), (b)-[:RB {name: 'b'}]->(bb)
      CREATE (aa)-[:RA {name: 'a'}]->(aaa), (aa)-[:RB {name: 'b'}]->(aab),
             (ab)-[:RA {name: 'a'}]->(aba), (ab)-[:RB {name: 'b'}]->(abb),
             (ba)-[:RA {name: 'a'}]->(baa), (ba)-[:RB {name: 'b'}]->(bab),
             (bb)-[:RA {name: 'a'}]->(bba), (bb)-[:RB {name: 'b'}]->(bbb)
      """
    When executing query:
      """
      MATCH p = (:SNodes)-[*0..3]->(x)
      WITH tail(nodes(p)) AS nodes
      RETURN nodes, none(x IN nodes WHERE x.name = 'a') AS result
      """
    Then the result should be, in any order:
      | nodes                                                  | result |
      | []                                                     | true   |
      | [(:A {name: 'a'})]                                     | false  |
      | [(:A {name: 'a'}), (:A {name: 'a'})]                   | false  |
      | [(:A {name: 'a'}), (:A {name: 'a'}), (:A {name: 'a'})] | false  |
      | [(:A {name: 'a'}), (:A {name: 'a'}), (:B {name: 'b'})] | false  |
      | [(:A {name: 'a'}), (:B {name: 'b'})]                   | false  |
      | [(:A {name: 'a'}), (:B {name: 'b'}), (:A {name: 'a'})] | false  |
      | [(:A {name: 'a'}), (:B {name: 'b'}), (:B {name: 'b'})] | false  |
      | [(:B {name: 'b'})]                                     | true   |
      | [(:B {name: 'b'}), (:A {name: 'a'})]                   | false  |
      | [(:B {name: 'b'}), (:A {name: 'a'}), (:A {name: 'a'})] | false  |
      | [(:B {name: 'b'}), (:A {name: 'a'}), (:B {name: 'b'})] | false  |
      | [(:B {name: 'b'}), (:B {name: 'b'})]                   | true   |
      | [(:B {name: 'b'}), (:B {name: 'b'}), (:A {name: 'a'})] | false  |
      | [(:B {name: 'b'}), (:B {name: 'b'}), (:B {name: 'b'})] | true   |
    And no side effects

  Scenario: [9] None quantifier on list containing relationships
    Given an empty graph
    And having executed:
      """
      CREATE (s1:SRelationships), (s2:SNodes)
      CREATE (a:A {name: 'a'}), (b:B {name: 'b'})
      CREATE (aa:A {name: 'a'}), (ab:B {name: 'b'}),
             (ba:A {name: 'a'}), (bb:B {name: 'b'})
      CREATE (aaa:A {name: 'a'}), (aab:B {name: 'b'}),
             (aba:A {name: 'a'}), (abb:B {name: 'b'}),
             (baa:A {name: 'a'}), (bab:B {name: 'b'}),
             (bba:A {name: 'a'}), (bbb:B {name: 'b'})
      CREATE (s1)-[:I]->(s2),
             (s2)-[:RA {name: 'a'}]->(a), (s2)-[:RB {name: 'b'}]->(b)
      CREATE (a)-[:RA {name: 'a'}]->(aa), (a)-[:RB {name: 'b'}]->(ab),
             (b)-[:RA {name: 'a'}]->(ba), (b)-[:RB {name: 'b'}]->(bb)
      CREATE (aa)-[:RA {name: 'a'}]->(aaa), (aa)-[:RB {name: 'b'}]->(aab),
             (ab)-[:RA {name: 'a'}]->(aba), (ab)-[:RB {name: 'b'}]->(abb),
             (ba)-[:RA {name: 'a'}]->(baa), (ba)-[:RB {name: 'b'}]->(bab),
             (bb)-[:RA {name: 'a'}]->(bba), (bb)-[:RB {name: 'b'}]->(bbb)
      """
    When executing query:
      """
      MATCH p = (:SRelationships)-[*0..4]->(x)
      WITH tail(relationships(p)) AS relationships, COUNT(*) AS c
      RETURN relationships, none(x IN relationships WHERE x.name = 'a') AS result
      """
    Then the result should be, in any order:
      | relationships                                             | result |
      | []                                                        | true   |
      | [[:RA {name: 'a'}]]                                       | false  |
      | [[:RA {name: 'a'}], [:RA {name: 'a'}]]                    | false  |
      | [[:RA {name: 'a'}], [:RA {name: 'a'}], [:RA {name: 'a'}]] | false  |
      | [[:RA {name: 'a'}], [:RA {name: 'a'}], [:RB {name: 'b'}]] | false  |
      | [[:RA {name: 'a'}], [:RB {name: 'b'}]]                    | false  |
      | [[:RA {name: 'a'}], [:RB {name: 'b'}], [:RA {name: 'a'}]] | false  |
      | [[:RA {name: 'a'}], [:RB {name: 'b'}], [:RB {name: 'b'}]] | false  |
      | [[:RB {name: 'b'}]]                                       | true   |
      | [[:RB {name: 'b'}], [:RA {name: 'a'}]]                    | false  |
      | [[:RB {name: 'b'}], [:RA {name: 'a'}], [:RA {name: 'a'}]] | false  |
      | [[:RB {name: 'b'}], [:RA {name: 'a'}], [:RB {name: 'b'}]] | false  |
      | [[:RB {name: 'b'}], [:RB {name: 'b'}]]                    | true   |
      | [[:RB {name: 'b'}], [:RB {name: 'b'}], [:RA {name: 'a'}]] | false  |
      | [[:RB {name: 'b'}], [:RB {name: 'b'}], [:RB {name: 'b'}]] | true   |
    And no side effects

  Scenario Outline: [10] None quantifier on lists containing nulls
    Given any graph
    When executing query:
      """
      RETURN none(x IN <list> WHERE <condition>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | list                    | condition | result |
      | [null]                  | x = 2     | null   |
      | [null, null]            | x = 2     | null   |
      | [0, null]               | x = 2     | null   |
      | [2, null]               | x = 2     | false  |
      | [null, 2]               | x = 2     | false  |
      | [34, 0, null, 5, 900]   | x < 10    | false  |
      | [34, 10, null, 15, 900] | x < 10    | null   |
      | [4, 0, null, -15, 9]    | x < 10    | false  |

  Scenario Outline: [11] None quantifier with IS NULL predicate
    Given any graph
    When executing query:
      """
      RETURN none(x IN <list> WHERE x IS NULL) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | list                     | result |
      | []                       | true   |
      | [0]                      | true   |
      | [34, 0, 8, 900]          | true   |
      | [null]                   | false  |
      | [null, null]             | false  |
      | [0, null]                | false  |
      | [2, null]                | false  |
      | [null, 2]                | false  |
      | [34, 0, null, 8, 900]    | false  |
      | [34, 0, null, 8, null]   | false  |
      | [null, 123, null, null]  | false  |
      | [null, null, null, null] | false  |

  Scenario Outline: [12] None quantifier with IS NOT NULL predicate
    Given any graph
    When executing query:
      """
      RETURN none(x IN <list> WHERE x IS NOT NULL) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | list                     | result |
      | []                       | true   |
      | [0]                      | false  |
      | [34, 0, 8, 900]          | false  |
      | [null]                   | true   |
      | [null, null]             | true   |
      | [0, null]                | false  |
      | [2, null]                | false  |
      | [null, 2]                | false  |
      | [34, 0, null, 8, 900]    | false  |
      | [34, 0, null, 8, null]   | false  |
      | [null, 123, null, null]  | false  |
      | [null, null, null, null] | true   |

  Scenario: [13] None quantifier is true if the predicate is statically false and the list is not empty
    Given any graph
    When executing query:
      """
      RETURN none(x IN [1, null, true, 4.5, 'abc', false] WHERE false) AS result
      """
    Then the result should be, in any order:
      | result |
      | true   |
    And no side effects

  Scenario: [14] None quantifier is false if the predicate is statically true and the list is not empty
    Given any graph
    When executing query:
      """
      RETURN none(x IN [1, null, true, 4.5, 'abc', false] WHERE true) AS result
      """
    Then the result should be, in any order:
      | result |
      | false  |
    And no side effects

  Scenario Outline: [15] Fail none quantifier on type mismatch between list elements and predicate
    Given any graph
    When executing query:
      """
      RETURN none(x IN <list> WHERE <condition>) AS result
      """
    Then a SyntaxError should be raised at compile time: InvalidArgumentType

    Examples:
      | list                              | condition |
      | ['Clara']                         | x % 2 = 0 |
      | [false, true]                     | x % 2 = 0 |
      | ['Clara', 'Bob', 'Dave', 'Alice'] | x % 2 = 0 |
      # add examples with heterogeneously-typed lists
