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

Feature: Return6 - Implicit grouping with aggregates

  @skip
  Scenario: [1] Return count aggregation over nodes
    Given an empty graph
    And having executed:
      """
      CREATE ({num: 42})
      """
    When executing query:
      """
      MATCH (n)
      RETURN n.num AS n, count(n) AS count
      """
    Then the result should be, in any order:
      | n  | count |
      | 42 | 1     |
    And no side effects

  @skip
  Scenario: [2] Projecting an arithmetic expression with aggregation
    Given an empty graph
    And having executed:
      """
      CREATE ({id: 42})
      """
    When executing query:
      """
      MATCH (a)
      RETURN a, count(a) + 3
      """
    Then the result should be, in any order:
      | a          | count(a) + 3 |
      | ({id: 42}) | 4            |
    And no side effects

  @skip
  Scenario: [3] Aggregating by a list property has a correct definition of equality
    Given an empty graph
    And having executed:
      """
      CREATE ({a: [1, 2, 3]}), ({a: [1, 2, 3]})
      """
    When executing query:
      """
      MATCH (a)
      WITH a.num AS a, count(*) AS count
      RETURN count
      """
    Then the result should be, in any order:
      | count |
      | 2     |
    And no side effects

  @skip
  Scenario: [4] Support multiple divisions in aggregate function
    Given an empty graph
    And having executed:
      """
      UNWIND range(0, 7250) AS i
      CREATE ()
      """
    When executing query:
      """
      MATCH (n)
      RETURN count(n) / 60 / 60 AS count
      """
    Then the result should be, in any order:
      | count |
      | 2     |
    And no side effects

  @skip
  Scenario: [5] Aggregates inside normal functions
    Given an empty graph
    And having executed:
      """
      UNWIND range(0, 10) AS i
      CREATE ()
      """
    When executing query:
      """
      MATCH (a)
      RETURN size(collect(a))
      """
    Then the result should be, in any order:
      | size(collect(a)) |
      | 11               |
    And no side effects

  @skip
  Scenario: [6] Handle aggregates inside non-aggregate expressions
    Given an empty graph
    When executing query:
      """
      MATCH (a {name: 'Andres'})<-[:FATHER]-(child)
      RETURN a.name, {foo: a.name='Andres', kids: collect(child.name)}
      """
    Then the result should be, in any order:
      | a.name | {foo: a.name='Andres', kids: collect(child.name)} |
    And no side effects

  @skip
  Scenario: [7] Aggregate on property
    Given an empty graph
    And having executed:
      """
      CREATE ({num: 33})
      CREATE ({num: 33})
      CREATE ({num: 42})
      """
    When executing query:
      """
      MATCH (n)
      RETURN n.num, count(*)
      """
    Then the result should be, in any order:
      | n.num | count(*) |
      | 42    | 1        |
      | 33    | 2        |
    And no side effects

  @skip
  Scenario: [8] Handle aggregation on functions
    Given an empty graph
    And having executed:
      """
      CREATE (a:L), (b1), (b2)
      CREATE (a)-[:A]->(b1), (a)-[:A]->(b2)
      """
    When executing query:
      """
      MATCH p=(a:L)-[*]->(b)
      RETURN b, avg(length(p))
      """
    Then the result should be, in any order:
      | b  | avg(length(p)) |
      | () | 1.0            |
      | () | 1.0            |
    And no side effects

  @skip
  Scenario: [9] Aggregates with arithmetics
    Given an empty graph
    And having executed:
      """
      CREATE ()
      """
    When executing query:
      """
      MATCH ()
      RETURN count(*) * 10 AS c
      """
    Then the result should be, in any order:
      | c  |
      | 10 |
    And no side effects

  @skip
  Scenario: [10] Multiple aggregates on same variable
    Given an empty graph
    And having executed:
      """
      CREATE ()
      """
    When executing query:
      """
      MATCH (n)
      RETURN count(n), collect(n)
      """
    Then the result should be, in any order:
      | count(n) | collect(n) |
      | 1        | [()]       |
    And no side effects

  @skip
  Scenario: [11] Counting matches
    Given an empty graph
    And having executed:
      """
      UNWIND range(1, 100) AS i
      CREATE ()
      """
    When executing query:
      """
      MATCH ()
      RETURN count(*)
      """
    Then the result should be, in any order:
      | count(*) |
      | 100      |
    And no side effects

  @skip
  Scenario: [12] Counting matches per group
    Given an empty graph
    And having executed:
      """
      CREATE (a:L), (b1), (b2)
      CREATE (a)-[:A]->(b1), (a)-[:A]->(b2)
      """
    When executing query:
      """
      MATCH (a:L)-[rel]->(b)
      RETURN a, count(*)
      """
    Then the result should be, in any order:
      | a    | count(*) |
      | (:L) | 2        |
    And no side effects

  @skip
  Scenario: [13] Returning the minimum length of paths
    Given an empty graph
    And having executed:
      """
      CREATE (a:T {name: 'a'}), (b:T {name: 'b'}), (c:T {name: 'c'})
      CREATE (a)-[:R]->(b)
      CREATE (a)-[:R]->(c)
      CREATE (c)-[:R]->(b)
      """
    When executing query:
      """
      MATCH p = (a:T {name: 'a'})-[:R*]->(other:T)
      WHERE other <> a
      WITH a, other, min(length(p)) AS len
      RETURN a.name AS name, collect(other.name) AS others, len
      """
    Then the result should be (ignoring element order for lists):
      | name | others     | len |
      | 'a'  | ['c', 'b'] | 1   |
    And no side effects

  @skip
  Scenario: [14] Aggregates in aggregates
    Given any graph
    When executing query:
      """
      RETURN count(count(*))
      """
    Then a SyntaxError should be raised at compile time: NestedAggregation

  @skip
  Scenario: [15] Using `rand()` in aggregations
    Given any graph
    When executing query:
      """
      RETURN count(rand())
      """
    Then a SyntaxError should be raised at compile time: NonConstantExpression

  @skip
  Scenario: [16] Aggregation on complex expressions
    Given an empty graph
    And having executed:
      """
      CREATE (andres {name: 'Andres'}),
             (michael {name: 'Michael'}),
             (peter {name: 'Peter'}),
             (bread {type: 'Bread'}),
             (veggies {type: 'Veggies'}),
             (meat {type: 'Meat'})
      CREATE (andres)-[:ATE {times: 10}]->(bread),
             (andres)-[:ATE {times: 8}]->(veggies),
             (michael)-[:ATE {times: 4}]->(veggies),
             (michael)-[:ATE {times: 6}]->(bread),
             (michael)-[:ATE {times: 9}]->(meat),
             (peter)-[:ATE {times: 7}]->(veggies),
             (peter)-[:ATE {times: 7}]->(bread),
             (peter)-[:ATE {times: 4}]->(meat)
      """
    When executing query:
      """
      MATCH (me)-[r1:ATE]->()<-[r2:ATE]-(you)
      WHERE me.name = 'Michael'
      WITH me, count(DISTINCT r1) AS H1, count(DISTINCT r2) AS H2, you
      MATCH (me)-[r1:ATE]->()<-[r2:ATE]-(you)
      RETURN me, you, sum((1 - abs(r1.times / H1 - r2.times / H2)) * (r1.times + r2.times) / (H1 + H2)) AS sum
      """
    Then the result should be, in any order:
      | me                  | you                | sum |
      | ({name: 'Michael'}) | ({name: 'Andres'}) | -7  |
      | ({name: 'Michael'}) | ({name: 'Peter'})  | 0   |
    And no side effects

  @skip
  Scenario: [17] Handle constants and parameters inside an expression which contains an aggregation expression
    Given an empty graph
    And parameters are:
      | age | 38 |
    When executing query:
      """
      MATCH (person)
      RETURN $age + avg(person.age) - 1000
      """
    Then the result should be, in any order:
      | $age + avg(person.age) - 1000 |
      | null                          |
    And no side effects

  @skip
  Scenario: [18] Handle returned variables inside an expression which contains an aggregation expression
    Given an empty graph
    When executing query:
      """
      MATCH (me: Person)--(you: Person)
      WITH me.age AS age, you
      RETURN age, age + count(you.age)
      """
    Then the result should be, in any order:
      | age | age + count(you.age) |
    And no side effects

  @skip
  Scenario: [19] Handle returned property accesses inside an expression which contains an aggregation expression
    Given an empty graph
    When executing query:
      """
      MATCH (me: Person)--(you: Person)
      RETURN me.age, me.age + count(you.age)
      """
    Then the result should be, in any order:
      | me.age | me.age + count(you.age) |
    And no side effects

  @skip
  Scenario: [20] Fail if not returned variables are used inside an expression which contains an aggregation expression
    Given an empty graph
    When executing query:
      """
      MATCH (me: Person)--(you: Person)
      RETURN me.age + count(you.age)
      """
    Then a SyntaxError should be raised at compile time: AmbiguousAggregationExpression

  @skip
  Scenario: [21] Fail if more complex expressions, even if returned, are used inside expression which contains an aggregation expression
    Given an empty graph
    When executing query:
      """
      MATCH (me: Person)--(you: Person)
      RETURN me.age + you.age, me.age + you.age + count(*)
      """
    Then a SyntaxError should be raised at compile time: AmbiguousAggregationExpression
