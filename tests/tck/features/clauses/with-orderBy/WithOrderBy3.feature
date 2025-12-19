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

Feature: WithOrderBy3 - Order by multiple expressions
# LIMIT is used in the following scenarios to surface the effects or WITH ... ORDER BY ...
# which are otherwise lost after the WITH clause according to Cypher semantics

  Scenario Outline: [1] Sort by two expressions, both in ascending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 9, bool: true}),
             (:B {num: 5, bool: false}),
             (:C {num: -30, bool: false}),
             (:D {num: -41, bool: true}),
             (:E {num: 7054, bool: false})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 4
      RETURN a
      """
    Then the result should be, in any order:
      | a                             |
      | (:C {num: -30, bool: false})  |
      | (:B {num: 5, bool: false})    |
      | (:E {num: 7054, bool: false}) |
      | (:D {num: -41, bool: true})   |
    And no side effects

    Examples:
      | sort                              |
      | a.bool, a.num                     |
      | a.bool, a.num ASC                 |
      | a.bool, a.num ASCENDING           |
      | a.bool ASC, a.num                 |
      | a.bool ASC, a.num ASC             |
      | a.bool ASC, a.num ASCENDING       |
      | a.bool ASCENDING, a.num           |
      | a.bool ASCENDING, a.num ASC       |
      | a.bool ASCENDING, a.num ASCENDING |

  Scenario Outline: [2] Sort by two expressions, first in ascending order, second in descending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 9, bool: true}),
             (:B {num: 5, bool: false}),
             (:C {num: -30, bool: false}),
             (:D {num: -41, bool: true}),
             (:E {num: 7054, bool: false})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 4
      RETURN a
      """
    Then the result should be, in any order:
      | a                             |
      | (:C {num: -30, bool: false})  |
      | (:B {num: 5, bool: false})    |
      | (:E {num: 7054, bool: false}) |
      | (:A {num: 9, bool: true})     |
    And no side effects

    Examples:
      | sort                               |
      | a.bool, a.num DESC                 |
      | a.bool, a.num DESCENDING           |
      | a.bool ASC, a.num DESC             |
      | a.bool ASC, a.num DESCENDING       |
      | a.bool ASCENDING, a.num DESC       |
      | a.bool ASCENDING, a.num DESCENDING |

  Scenario Outline: [3] Sort by two expressions, first in descending order, second in ascending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 9, bool: true}),
             (:B {num: 5, bool: false}),
             (:C {num: -30, bool: false}),
             (:D {num: -41, bool: true}),
             (:E {num: 7054, bool: false})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 4
      RETURN a
      """
    Then the result should be, in any order:
      | a                            |
      | (:D {num: -41, bool: true})  |
      | (:A {num: 9, bool: true})    |
      | (:C {num: -30, bool: false}) |
      | (:B {num: 5, bool: false})   |
    And no side effects

    Examples:
      | sort                               |
      | a.bool DESC, a.num                 |
      | a.bool DESC, a.num ASC             |
      | a.bool DESC, a.num ASCENDING       |
      | a.bool DESCENDING, a.num           |
      | a.bool DESCENDING, a.num ASC       |
      | a.bool DESCENDING, a.num ASCENDING |

  Scenario Outline: [4] Sort by two expressions, both in descending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 9, bool: true}),
             (:B {num: 5, bool: false}),
             (:C {num: -30, bool: false}),
             (:D {num: -41, bool: true}),
             (:E {num: 7054, bool: false})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 4
      RETURN a
      """
    Then the result should be, in any order:
      | a                             |
      | (:A {num: 9, bool: true})     |
      | (:D {num: -41, bool: true})   |
      | (:E {num: 7054, bool: false}) |
      | (:B {num: 5, bool: false})    |
    And no side effects

    Examples:
      | sort                                |
      | a.bool DESC, a.num DESC             |
      | a.bool DESC, a.num DESCENDING       |
      | a.bool DESCENDING, a.num DESC       |
      | a.bool DESCENDING, a.num DESCENDING |

  Scenario Outline: [5] An expression without explicit sort direction is sorted in ascending order
    Given an empty graph
    And having executed:
      """
      CREATE ({num: 3, text: 'a'}),
             ({num: 3, text: 'b'}),
             ({num: 1, text: 'a'}),
             ({num: 1, text: 'b'}),
             ({num: 2, text: 'a'}),
             ({num: 2, text: 'b'}),
             ({num: 4, text: 'a'}),
             ({num: 4, text: 'b'})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY a.num % 2 <dir1>, a.num, a.text <dir2>
        LIMIT 1
      RETURN a
      """
    Then the result should be, in any order:
      | a                              |
      | ({num: <num>, text: '<text>'}) |
    And no side effects

    Examples:
      | dir1 | dir2 | num | text |
      | ASC  | ASC  | 2   | a    |
      | ASC  | DESC | 2   | b    |
      | DESC | DESC | 1   | b    |
      | DESC | ASC  | 1   | a    |

  Scenario Outline: [6] An constant expression does not influence the order determined by other expression before and after the constant expression
    Given an empty graph
    And having executed:
      """
      CREATE ({num: 3, text: 'a'}),
             ({num: 3, text: 'b'}),
             ({num: 1, text: 'a'}),
             ({num: 1, text: 'b'}),
             ({num: 2, text: 'a'}),
             ({num: 2, text: 'b'}),
             ({num: 4, text: 'a'}),
             ({num: 4, text: 'b'})
      """
    When executing query:
      """
      MATCH (a)
      WITH a
        ORDER BY <sort>
        LIMIT 1
      RETURN a
      """
    Then the result should be, in any order:
      | a                              |
      | ({num: <num>, text: '<text>'}) |
    And no side effects

    Examples:
      | sort                                                | num | text |
      | 4 + ((a.num * 2) % 2) ASC, a.num ASC, a.text ASC    | 1   | a    |
      | 4 + ((a.num * 2) % 2) DESC, a.num ASC, a.text ASC   | 1   | a    |
      | a.num ASC, 4 + ((a.num * 2) % 2) ASC, a.text ASC    | 1   | a    |
      | a.num ASC, 4 + ((a.num * 2) % 2) DESC, a.text ASC   | 1   | a    |
      | a.num ASC, a.text ASC, 4 + ((a.num * 2) % 2) ASC    | 1   | a    |
      | a.num ASC, a.text ASC, 4 + ((a.num * 2) % 2) DESC   | 1   | a    |
      | 4 + ((a.num * 2) % 2) ASC, a.num ASC, a.text DESC   | 1   | b    |
      | 4 + ((a.num * 2) % 2) DESC, a.num ASC, a.text DESC  | 1   | b    |
      | a.num ASC, 4 + ((a.num * 2) % 2) ASC, a.text DESC   | 1   | b    |
      | a.num ASC, 4 + ((a.num * 2) % 2) DESC, a.text DESC  | 1   | b    |
      | a.num ASC, a.text DESC, 4 + ((a.num * 2) % 2) ASC   | 1   | b    |
      | a.num ASC, a.text DESC, 4 + ((a.num * 2) % 2) DESC  | 1   | b    |
      | 4 + ((a.num * 2) % 2) ASC, a.num DESC, a.text DESC  | 4   | b    |
      | 4 + ((a.num * 2) % 2) DESC, a.num DESC, a.text DESC | 4   | b    |
      | a.num DESC, 4 + ((a.num * 2) % 2) ASC, a.text DESC  | 4   | b    |
      | a.num DESC, 4 + ((a.num * 2) % 2) DESC, a.text DESC | 4   | b    |
      | a.num DESC, a.text DESC, 4 + ((a.num * 2) % 2) ASC  | 4   | b    |
      | a.num DESC, a.text DESC, 4 + ((a.num * 2) % 2) DESC | 4   | b    |
      | 4 + ((a.num * 2) % 2) ASC, a.num DESC, a.text ASC   | 4   | a    |
      | 4 + ((a.num * 2) % 2) DESC, a.num DESC, a.text ASC  | 4   | a    |
      | a.num DESC, 4 + ((a.num * 2) % 2) ASC, a.text ASC   | 4   | a    |
      | a.num DESC, 4 + ((a.num * 2) % 2) DESC, a.text ASC  | 4   | a    |
      | a.num DESC, a.text ASC, 4 + ((a.num * 2) % 2) ASC   | 4   | a    |
      | a.num DESC, a.text ASC, 4 + ((a.num * 2) % 2) DESC  | 4   | a    |

  @skip
  Scenario Outline: [7] The order direction cannot be overwritten
    Given any graph
    When executing query:
      """
      UNWIND [1, 2, 3] AS a
      WITH a
        ORDER BY <sort>
        LIMIT 1
      RETURN a
      """
    Then the result should be, in any order:
      | a       |
      | <value> |
    And no side effects

    Examples:
      | sort                  | value |
      | a ASC, a DESC         | 1     |
      | a + 2 ASC, a + 2 DESC | 1     |
      | a * a ASC, a * a DESC | 1     |
      | a ASC, -1 * a ASC     | 1     |
      | -1 * a DESC, a ASC    | 1     |
      | a DESC, a ASC         | 3     |
      | a + 2 DESC, a + 2 ASC | 3     |
      | a * a DESC, a * a ASC | 3     |
      | a DESC, -1 * a DESC   | 3     |
      | -1 * a ASC, a DESC    | 3     |

  @skip
  Scenario Outline: [8] Fail on sorting by any number of undefined variables in any position #Example: <exampleName>
    Given any graph
    When executing query:
      """
      WITH 1 AS a, 'b' AS b, 3 AS c, true AS d
      WITH a, b
      WITH a
        ORDER BY <sort>
      RETURN a
      """
    Then a SyntaxError should be raised at compile time: UndefinedVariable

    Examples:
      | sort                | exampleName   |
      | a, c                | out of scope  |
      | a, c ASC            | out of scope  |
      | a, c DESC           | out of scope  |
      | a, c, d             | out of scope  |
      | a, c ASC, d         | out of scope  |
      | a, c DESC, d        | out of scope  |
      | c, a, d             | out of scope  |
      | c ASC, a, d         | out of scope  |
      | c DESC, a, d        | out of scope  |
      | c, d, a             | out of scope  |
      | b, c, d, a          | out of scope  |
      | c, b, c, d, a       | out of scope  |
      | c, d, b, b, d, c, a | out of scope  |
      | a, e                | never defined |
      | a, e ASC            | never defined |
      | a, e DESC           | never defined |
      | a, e, f             | never defined |
      | a, e ASC, f         | never defined |
      | a, e DESC, f        | never defined |
      | e, a, f             | never defined |
      | e ASC, a, f         | never defined |
      | e DESC, a, f        | never defined |
      | e, f, a             | never defined |
      | b, e, f, a          | never defined |
      | e, b, e, f, a       | never defined |
      | e, f, b, b, f, e, a | never defined |
      | a, c, e             | mixed         |
      | a, c, e, b          | mixed         |
      | b, c, a, f, a       | mixed         |
      | d, f, b, b, f, c, a | mixed         |
