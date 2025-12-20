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

Feature: WithOrderBy4 - Order by in combination with projection and aliasing
# LIMIT is used in the following scenarios to surface the effects or WITH ... ORDER BY ...
# which are otherwise lost after the WITH clause according to Cypher semantics

  @skip
  Scenario: [1] Sort by a projected expression
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 1, num2: 4}), //num + num2 = 5
             (:A {num: 5, num2: 2}), //num + num2 = 7
             (:A {num: 9, num2: 0}), //num + num2 = 9
             (:A {num: 3, num2: 3}), //num + num2 = 6
             (:A {num: 7, num2: 1})  //num + num2 = 8
      """
    When executing query:
      """
      MATCH (a:A)
      WITH a, a.num + a.num2 AS sum
        ORDER BY a.num + a.num2
        LIMIT 3
      RETURN a, sum
      """
    Then the result should be, in any order:
      | a                      | sum |
      | (:A {num: 1, num2: 4}) | 5   |
      | (:A {num: 3, num2: 3}) | 6   |
      | (:A {num: 5, num2: 2}) | 7   |
    And no side effects

  @skip
  Scenario: [2] Sort by an alias of a projected expression
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 1, num2: 4}), //num + num2 = 5
             (:A {num: 5, num2: 2}), //num + num2 = 7
             (:A {num: 9, num2: 0}), //num + num2 = 9
             (:A {num: 3, num2: 3}), //num + num2 = 6
             (:A {num: 7, num2: 1})  //num + num2 = 8
      """
    When executing query:
      """
      MATCH (a:A)
      WITH a, a.num + a.num2 AS sum
        ORDER BY sum
        LIMIT 3
      RETURN a, sum
      """
    Then the result should be, in any order:
      | a                      | sum |
      | (:A {num: 1, num2: 4}) | 5   |
      | (:A {num: 3, num2: 3}) | 6   |
      | (:A {num: 5, num2: 2}) | 7   |
    And no side effects

  @skip
  Scenario: [3] Sort by two projected expressions with order priority being different than projection order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 1, num2: 4}), //num2 % 3 = 1, num + num2 = 5
             (:A {num: 5, num2: 2}), //num2 % 3 = 2, num + num2 = 7
             (:A {num: 9, num2: 0}), //num2 % 3 = 0, num + num2 = 9
             (:A {num: 3, num2: 3}), //num2 % 3 = 0, num + num2 = 6
             (:A {num: 7, num2: 1})  //num2 % 3 = 1, num + num2 = 8
      """
    When executing query:
      """
      MATCH (a:A)
      WITH a, a.num + a.num2 AS sum, a.num2 % 3 AS mod
        ORDER BY a.num2 % 3, a.num + a.num2
        LIMIT 3
      RETURN a, sum, mod
      """
    Then the result should be, in any order:
      | a                      | sum | mod |
      | (:A {num: 3, num2: 3}) | 6   | 0   |
      | (:A {num: 9, num2: 0}) | 9   | 0   |
      | (:A {num: 1, num2: 4}) | 5   | 1   |
    And no side effects

  @skip
  Scenario: [4] Sort by one projected expression and one alias of a projected expression with order priority being different than projection order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 1, num2: 4}), //num2 % 3 = 1, num + num2 = 5
             (:A {num: 5, num2: 2}), //num2 % 3 = 2, num + num2 = 7
             (:A {num: 9, num2: 0}), //num2 % 3 = 0, num + num2 = 9
             (:A {num: 3, num2: 3}), //num2 % 3 = 0, num + num2 = 6
             (:A {num: 7, num2: 1})  //num2 % 3 = 1, num + num2 = 8
      """
    When executing query:
      """
      MATCH (a:A)
      WITH a, a.num + a.num2 AS sum, a.num2 % 3 AS mod
        ORDER BY a.num2 % 3, sum
        LIMIT 3
      RETURN a, sum, mod
      """
    Then the result should be, in any order:
      | a                      | sum | mod |
      | (:A {num: 3, num2: 3}) | 6   | 0   |
      | (:A {num: 9, num2: 0}) | 9   | 0   |
      | (:A {num: 1, num2: 4}) | 5   | 1   |
    And no side effects

  @skip
  Scenario: [5] Sort by one alias of a projected expression and one projected expression with order priority being different than projection order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 1, num2: 4}), //num2 % 3 = 1, num + num2 = 5
             (:A {num: 5, num2: 2}), //num2 % 3 = 2, num + num2 = 7
             (:A {num: 9, num2: 0}), //num2 % 3 = 0, num + num2 = 9
             (:A {num: 3, num2: 3}), //num2 % 3 = 0, num + num2 = 6
             (:A {num: 7, num2: 1})  //num2 % 3 = 1, num + num2 = 8
      """
    When executing query:
      """
      MATCH (a:A)
      WITH a, a.num + a.num2 AS sum, a.num2 % 3 AS mod
        ORDER BY mod, a.num + a.num2
        LIMIT 3
      RETURN a, sum, mod
      """
    Then the result should be, in any order:
      | a                      | sum | mod |
      | (:A {num: 3, num2: 3}) | 6   | 0   |
      | (:A {num: 9, num2: 0}) | 9   | 0   |
      | (:A {num: 1, num2: 4}) | 5   | 1   |
    And no side effects

  @skip
  Scenario: [6] Sort by aliases of two projected expressions with order priority being different than projection order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 1, num2: 4}), //num2 % 3 = 1, num + num2 = 5
             (:A {num: 5, num2: 2}), //num2 % 3 = 2, num + num2 = 7
             (:A {num: 9, num2: 0}), //num2 % 3 = 0, num + num2 = 9
             (:A {num: 3, num2: 3}), //num2 % 3 = 0, num + num2 = 6
             (:A {num: 7, num2: 1})  //num2 % 3 = 1, num + num2 = 8
      """
    When executing query:
      """
      MATCH (a:A)
      WITH a, a.num + a.num2 AS sum, a.num2 % 3 AS mod
        ORDER BY mod, sum
        LIMIT 3
      RETURN a, sum, mod
      """
    Then the result should be, in any order:
      | a                      | sum | mod |
      | (:A {num: 3, num2: 3}) | 6   | 0   |
      | (:A {num: 9, num2: 0}) | 9   | 0   |
      | (:A {num: 1, num2: 4}) | 5   | 1   |
    And no side effects

  @skip
  Scenario: [7] Sort by an alias of a projected expression where the alias shadows an existing variable
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 1, num2: 4}), //num2 % 3 = 1, num + num2 = 5
             (:A {num: 5, num2: 2}), //num2 % 3 = 2, num + num2 = 7
             (:A {num: 9, num2: 0}), //num2 % 3 = 0, num + num2 = 9
             (:A {num: 3, num2: 3}), //num2 % 3 = 0, num + num2 = 6
             (:A {num: 7, num2: 1})  //num2 % 3 = 1, num + num2 = 8
      """
    When executing query:
      """
      MATCH (a:A)
      WITH a, a.num2 % 3 AS x
      WITH a, a.num + a.num2 AS x
        ORDER BY x
        LIMIT 3
      RETURN a, x
      """
    Then the result should be, in any order:
      | a                      | x |
      | (:A {num: 1, num2: 4}) | 5 |
      | (:A {num: 3, num2: 3}) | 6 |
      | (:A {num: 5, num2: 2}) | 7 |
    And no side effects

  @skip
  Scenario: [8] Sort by non-projected existing variable
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 1, num2: 4}), //num2 % 3 = 1, num + num2 = 5
             (:A {num: 5, num2: 2}), //num2 % 3 = 2, num + num2 = 7
             (:A {num: 9, num2: 0}), //num2 % 3 = 0, num + num2 = 9
             (:A {num: 3, num2: 3}), //num2 % 3 = 0, num + num2 = 6
             (:A {num: 7, num2: 1})  //num2 % 3 = 1, num + num2 = 8
      """
    When executing query:
      """
      MATCH (a:A)
      WITH a, a.num + a.num2 AS sum
      WITH a, a.num2 % 3 AS mod
        ORDER BY sum
        LIMIT 3
      RETURN a, mod
      """
    Then the result should be, in any order:
      | a                      | mod |
      | (:A {num: 1, num2: 4}) | 1   |
      | (:A {num: 3, num2: 3}) | 0   |
      | (:A {num: 5, num2: 2}) | 2   |
    And no side effects

  @skip
  Scenario: [9] Sort by an alias of a projected expression containing the variable shadowed by the alias
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 1, num2: 4}), //num2 % 3 = 1, num + num2 = 5
             (:A {num: 5, num2: 2}), //num2 % 3 = 2, num + num2 = 7
             (:A {num: 9, num2: 0}), //num2 % 3 = 0, num + num2 = 9
             (:A {num: 3, num2: 3}), //num2 % 3 = 0, num + num2 = 6
             (:A {num: 7, num2: 1})  //num2 % 3 = 1, num + num2 = 8
      """
    When executing query:
      """
      MATCH (a:A)
      WITH a.num2 AS x
      WITH x % 3 AS x
        ORDER BY x
        LIMIT 3
      RETURN x
      """
    Then the result should be, in any order:
      | x |
      | 0 |
      | 0 |
      | 1 |
    And no side effects

  @skip
  Scenario: [10] Sort by a non-projected expression containing an alias of a projected expression containing the variable shadowed by the alias
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 1, num2: 4}), //num2 % 3 = 1, num + num2 = 5
             (:A {num: 5, num2: 2}), //num2 % 3 = 2, num + num2 = 7
             (:A {num: 9, num2: 0}), //num2 % 3 = 0, num + num2 = 9
             (:A {num: 3, num2: 3}), //num2 % 3 = 0, num + num2 = 6
             (:A {num: 7, num2: 1})  //num2 % 3 = 1, num + num2 = 8
      """
    When executing query:
      """
      MATCH (a:A)
      WITH a.num2 AS x
      WITH x % 3 AS x
        ORDER BY x * -1
        LIMIT 3
      RETURN x
      """
    Then the result should be, in any order:
      | x |
      | 2 |
      | 1 |
      | 1 |
    And no side effects

  @skip
  Scenario: [11] Sort by an aggregate projection
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 1, num2: 4}), //num2 % 3 = 1, num + num2 = 5
             (:A {num: 5, num2: 2}), //num2 % 3 = 2, num + num2 = 7
             (:A {num: 9, num2: 0}), //num2 % 3 = 0, num + num2 = 9
             (:A {num: 3, num2: 3}), //num2 % 3 = 0, num + num2 = 6
             (:A {num: 7, num2: 1})  //num2 % 3 = 1, num + num2 = 8
      """
    When executing query:
      """
      MATCH (a:A)
      WITH a.num2 % 3 AS mod, sum(a.num + a.num2) AS sum
        ORDER BY sum(a.num + a.num2)
        LIMIT 2
      RETURN mod, sum
      """
    Then the result should be, in any order:
      | mod | sum   |
      | 2   | 7.0   |
      | 1   | 13.0  |
    And no side effects

  @skip
  Scenario: [12] Sort by an aliased aggregate projection
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 1, num2: 4}), //num2 % 3 = 1, num + num2 = 5
             (:A {num: 5, num2: 2}), //num2 % 3 = 2, num + num2 = 7
             (:A {num: 9, num2: 0}), //num2 % 3 = 0, num + num2 = 9
             (:A {num: 3, num2: 3}), //num2 % 3 = 0, num + num2 = 6
             (:A {num: 7, num2: 1})  //num2 % 3 = 1, num + num2 = 8
      """
    When executing query:
      """
      MATCH (a:A)
      WITH a.num2 % 3 AS mod, sum(a.num + a.num2) AS sum
        ORDER BY sum
        LIMIT 2
      RETURN mod, sum
      """
    Then the result should be, in any order:
      | mod | sum   |
      | 2   | 7.0   |
      | 1   | 13.0  |
    And no side effects

  @skip
  Scenario: [13] Fail on sorting by a non-projected aggregation on a variable
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 1, num2: 4}), //num2 % 3 = 1, num + num2 = 5
             (:A {num: 5, num2: 2}), //num2 % 3 = 2, num + num2 = 7
             (:A {num: 9, num2: 0}), //num2 % 3 = 0, num + num2 = 9
             (:A {num: 3, num2: 3}), //num2 % 3 = 0, num + num2 = 6
             (:A {num: 7, num2: 1})  //num2 % 3 = 1, num + num2 = 8
      """
    When executing query:
      """
      MATCH (a:A)
      WITH a, a.num + a.num2 AS sum
      WITH a.num2 % 3 AS mod, min(sum) AS min
        ORDER BY sum(sum)
        LIMIT 2
      RETURN mod, min
      """
    Then a SyntaxError should be raised at compile time: AmbiguousAggregationExpression

  @skip
  Scenario: [14] Fail on sorting by a non-projected aggregation on an expression
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 1, num2: 4}), //num2 % 3 = 1, num + num2 = 5
             (:A {num: 5, num2: 2}), //num2 % 3 = 2, num + num2 = 7
             (:A {num: 9, num2: 0}), //num2 % 3 = 0, num + num2 = 9
             (:A {num: 3, num2: 3}), //num2 % 3 = 0, num + num2 = 6
             (:A {num: 7, num2: 1})  //num2 % 3 = 1, num + num2 = 8
      """
    When executing query:
      """
      MATCH (a:A)
      WITH a.num2 % 3 AS mod, min(a.num + a.num2) AS min
        ORDER BY sum(a.num + a.num2)
        LIMIT 2
      RETURN mod, min
      """
    Then a SyntaxError should be raised at compile time: AmbiguousAggregationExpression

  @skip
  Scenario: [15] Sort by an aliased aggregate projection does allow subsequent matching
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:T1 {id: 0}]->(:X),
             ()-[:T2 {id: 1}]->(:X),
             ()-[:T2 {id: 2}]->()
      """
    When executing query:
      """
      MATCH (a)-[r]->(b:X)
      WITH a, r, b, count(*) AS c
        ORDER BY c
      MATCH (a)-[r]->(b)
      RETURN r AS rel
        ORDER BY rel.id
      """
    Then the result should be, in order:
      | rel           |
      | [:T1 {id: 0}] |
      | [:T2 {id: 1}] |
    And no side effects

  @skip
  Scenario: [16] Handle constants and parameters inside an order by item which contains an aggregation expression
    Given an empty graph
    And parameters are:
      | age | 38 |
    When executing query:
      """
      MATCH (person)
      WITH avg(person.age) AS avgAge
      ORDER BY $age + avg(person.age) - 1000
      RETURN avgAge
      """
    Then the result should be, in any order:
      | avgAge |
      | null   |
    And no side effects

  @skip
  Scenario: [17] Handle projected variables inside an order by item which contains an aggregation expression
    Given an empty graph
    When executing query:
      """
      MATCH (me: Person)--(you: Person)
      WITH me.age AS age, count(you.age) AS cnt
      ORDER BY age, age + count(you.age)
      RETURN age
      """
    Then the result should be, in any order:
      | age |
    And no side effects

  @skip
  Scenario: [18]  Handle projected property accesses inside an order by item which contains an aggregation expression
    Given an empty graph
    When executing query:
      """
      MATCH (me: Person)--(you: Person)
      WITH me.age AS age, count(you.age) AS cnt
      ORDER BY me.age + count(you.age)
      RETURN age
      """
    Then the result should be, in any order:
      | age |
    And no side effects

  @skip
  Scenario: [19] Fail if not projected variables are used inside an order by item which contains an aggregation expression
    Given an empty graph
    When executing query:
      """
      MATCH (me: Person)--(you: Person)
      WITH count(you.age) AS agg
      ORDER BY me.age + count(you.age)
      RETURN *
      """
    Then a SyntaxError should be raised at compile time: AmbiguousAggregationExpression

  @skip
  Scenario: [20] Fail if more complex expressions, even if projected, are used inside an order by item which contains an aggregation expression
    Given an empty graph
    When executing query:
      """
      MATCH (me: Person)--(you: Person)
      WITH me.age + you.age, count(*) AS cnt
      ORDER BY me.age + you.age + count(*)
      RETURN *
      """
    Then a SyntaxError should be raised at compile time: AmbiguousAggregationExpression
