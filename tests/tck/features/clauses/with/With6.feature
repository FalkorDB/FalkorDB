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

Feature: With6 - Implicit grouping with aggregates

  Scenario: [1] Implicit grouping with single expression as grouping key and single aggregation
    Given an empty graph
    And having executed:
      """
      CREATE ({name: 'A'}),
             ({name: 'A'}),
             ({name: 'B'})
      """
    When executing query:
      """
      MATCH (a)
      WITH a.name AS name, count(*) AS relCount
      RETURN name, relCount
      """
    Then the result should be, in any order:
      | name | relCount |
      | 'A'  | 2        |
      | 'B'  | 1        |
    And no side effects

  
  Scenario: [2] Implicit grouping with single relationship variable as grouping key and single aggregation
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:T1]->(:X),
             ()-[:T2]->(:X),
             ()-[:T3]->()
      """
    When executing query:
      """
      MATCH ()-[r1]->(:X)
      WITH r1 AS r2, count(*) AS c
      MATCH ()-[r2]->()
      RETURN r2 AS rel
      """
    Then the result should be, in any order:
      | rel   |
      | [:T1] |
      | [:T2] |
    And no side effects

  Scenario: [3] Implicit grouping with multiple node and relationship variables as grouping key and single aggregation
    Given an empty graph
    And having executed:
      """
      CREATE ()-[:T1]->(:X),
             ()-[:T2]->(:X),
             ()-[:T3]->()
      """
    When executing query:
      """
      MATCH (a)-[r1]->(b:X)
      WITH a, r1 AS r2, b, count(*) AS c
      MATCH (a)-[r2]->(b)
      RETURN r2 AS rel
      """
    Then the result should be, in any order:
      | rel   |
      | [:T1] |
      | [:T2] |
    And no side effects

  Scenario: [4] Implicit grouping with single path variable as grouping key and single aggregation
    Given an empty graph
    And having executed:
      """
      CREATE (n1 {num: 1}), (n2 {num: 2}),
             (n3 {num: 3}), (n4 {num: 4})
      CREATE (n1)-[:T]->(n2),
             (n3)-[:T]->(n4)
      """
    When executing query:
      """
      MATCH p = ()-[*]->()
      WITH count(*) AS count, p AS p
      RETURN nodes(p) AS nodes
      """
    Then the result should be, in any order:
      | nodes                    |
      | [({num: 1}), ({num: 2})] |
      | [({num: 3}), ({num: 4})] |
    And no side effects

  Scenario: [5] Handle constants and parameters inside an expression which contains an aggregation expression
    Given an empty graph
    And parameters are:
      | age | 38 |
    When executing query:
      """
      MATCH (person)
      WITH $age + avg(person.age) - 1000 AS agg
      RETURN *
      """
    Then the result should be, in any order:
      | agg  |
      | null |
    And no side effects

  Scenario: [6] Handle projected variables inside an expression which contains an aggregation expression
    Given an empty graph
    When executing query:
      """
      MATCH (me: Person)--(you: Person)
      WITH me.age AS age, you
      WITH age, age + count(you.age) AS agg
      RETURN *
      """
    Then the result should be, in any order:
      | age | agg |
    And no side effects

  Scenario: [7] Handle projected property accesses inside an expression which contains an aggregation expression
    Given an empty graph
    When executing query:
      """
      MATCH (me: Person)--(you: Person)
      WITH me.age AS age, me.age + count(you.age) AS agg
      RETURN *
      """
    Then the result should be, in any order:
      | age | agg |
    And no side effects

  
  Scenario: [8] Fail if not projected variables are used inside an expression which contains an aggregation expression
    Given an empty graph
    When executing query:
      """
      MATCH (me: Person)--(you: Person)
      WITH me.age + count(you.age) AS agg
      RETURN *
      """
    Then a SyntaxError should be raised at compile time: AmbiguousAggregationExpression

  
  Scenario: [9] Fail if more complex expression, even if projected, are used inside expression which contains an aggregation expression
    Given an empty graph
    When executing query:
      """
      MATCH (me: Person)--(you: Person)
      WITH me.age + you.age AS grp, me.age + you.age + count(*) AS agg
      RETURN *
      """
    Then a SyntaxError should be raised at compile time: AmbiguousAggregationExpression
