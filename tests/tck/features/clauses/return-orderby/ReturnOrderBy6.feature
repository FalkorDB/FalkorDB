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

Feature: ReturnOrderBy6 - Aggregation expressions in order by

  Scenario: [1] Handle constants and parameters inside an order by item which contains an aggregation expression
    Given an empty graph
    And parameters are:
      | age | 38 |
    When executing query:
      """
      MATCH (person)
      RETURN avg(person.age) AS avgAge
      ORDER BY $age + avg(person.age) - 1000
      """
    Then the result should be, in any order:
      | avgAge |
      | null   |
    And no side effects

  Scenario: [2] Handle returned aliases inside an order by item which contains an aggregation expression
    Given an empty graph
    When executing query:
      """
      MATCH (me: Person)--(you: Person)
      RETURN me.age AS age, count(you.age) AS cnt
      ORDER BY age, age + count(you.age)
      """
    Then the result should be, in any order:
      | age | cnt |
    And no side effects

  Scenario: [3] Handle returned property accesses inside an order by item which contains an aggregation expression
    Given an empty graph
    When executing query:
      """
      MATCH (me: Person)--(you: Person)
      RETURN me.age AS age, count(you.age) AS cnt
      ORDER BY me.age + count(you.age)
      """
    Then the result should be, in any order:
      | age | cnt |
    And no side effects

  @skip
  Scenario: [4] Fail if not returned variables are used inside an order by item which contains an aggregation expression
    Given an empty graph
    When executing query:
      """
      MATCH (me: Person)--(you: Person)
      RETURN count(you.age) AS agg
      ORDER BY me.age + count(you.age)
      """
    Then a SyntaxError should be raised at compile time: AmbiguousAggregationExpression

  @skip
  Scenario: [5] Fail if more complex expressions, even if returned, are used inside an order by item which contains an aggregation expression
    Given an empty graph
    When executing query:
      """
      MATCH (me: Person)--(you: Person)
      RETURN me.age + you.age, count(*) AS cnt
      ORDER BY me.age + you.age + count(*)
      """
    Then a SyntaxError should be raised at compile time: AmbiguousAggregationExpression
