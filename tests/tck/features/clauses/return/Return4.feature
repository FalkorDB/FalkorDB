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

Feature: Return4 - Column renaming

  Scenario: [1] Honour the column name for RETURN items
    Given an empty graph
    And having executed:
      """
      CREATE ({name: 'Someone'})
      """
    When executing query:
      """
      MATCH (a)
      WITH a.name AS a
      RETURN a
      """
    Then the result should be, in any order:
      | a         |
      | 'Someone' |
    And no side effects

  Scenario: [2] Support column renaming
    Given an empty graph
    And having executed:
      """
      CREATE (:Singleton)
      """
    When executing query:
      """
      MATCH (a)
      RETURN a AS ColumnName
      """
    Then the result should be, in any order:
      | ColumnName   |
      | (:Singleton) |
    And no side effects

  Scenario: [3] Aliasing expressions
    Given an empty graph
    And having executed:
      """
      CREATE ({id: 42})
      """
    When executing query:
      """
      MATCH (a)
      RETURN a.id AS a, a.id
      """
    Then the result should be, in any order:
      | a  | a.id |
      | 42 | 42   |
    And no side effects

  Scenario: [4] Keeping used expression 1
    Given an empty graph
    And having executed:
      """
      CREATE ()
      """
    When executing query:
      """
      MATCH (n)
      RETURN cOuNt( * )
      """
    Then the result should be, in any order:
      | cOuNt( * ) |
      | 1          |
    And no side effects

  Scenario: [5] Keeping used expression 2
    Given an empty graph
    And having executed:
      """
      CREATE ()
      """
    When executing query:
      """
      MATCH p = (n)-->(b)
      RETURN nOdEs( p )
      """
    Then the result should be, in any order:
      | nOdEs( p ) |
    And no side effects

  @skipStyleCheck
  Scenario: [6] Keeping used expression 3
    Given an empty graph
    And having executed:
      """
      CREATE ()
      """
    When executing query:
      """
      MATCH p = (n)-->(b)
      RETURN coUnt( dIstInct p )
      """
    Then the result should be, in any order:
      | coUnt( dIstInct p ) |
      | 0                   |
    And no side effects

  Scenario: [7] Keeping used expression 4
    Given an empty graph
    And having executed:
      """
      CREATE ()
      """
    When executing query:
      """
      MATCH p = (n)-->(b)
      RETURN aVg(    n.aGe     )
      """
    Then the result should be, in any order:
      | aVg(    n.aGe     ) |
      | null                |
    And no side effects

  Scenario: [8] Support column renaming for aggregations
    Given an empty graph
    And having executed:
      """
      UNWIND range(0, 10) AS i
      CREATE ()
      """
    When executing query:
      """
      MATCH ()
      RETURN count(*) AS columnName
      """
    Then the result should be, in any order:
      | columnName |
      | 11         |
    And no side effects

  Scenario: [9] Handle subexpression in aggregation also occurring as standalone expression with nested aggregation in a literal map
    Given an empty graph
    And having executed:
      """
      CREATE (:A), (:B {num: 42})
      """
    When executing query:
      """
      MATCH (a:A), (b:B)
      RETURN coalesce(a.num, b.num) AS foo,
        b.num AS bar,
        {name: count(b)} AS baz
      """
    Then the result should be, in any order:
      | foo | bar | baz       |
      | 42  | 42  | {name: 1} |
    And no side effects

  Scenario: [10] Fail when returning multiple columns with same name
    Given any graph
    When executing query:
      """
      RETURN 1 AS a, 2 AS a
      """
    Then a SyntaxError should be raised at compile time: ColumnNameConflict

  Scenario: [11] Reusing variable names in RETURN
    Given an empty graph
    And having executed:
      """
      CREATE (a:Person), (b:Person), (m:Message {id: 10})
      CREATE (a)-[:LIKE {creationDate: 20160614}]->(m)-[:POSTED_BY]->(b)
      """
    When executing query:
      """
      MATCH (person:Person)<--(message)<-[like]-(:Person)
      WITH like.creationDate AS likeTime, person AS person
        ORDER BY likeTime, message.id
      WITH head(collect({likeTime: likeTime})) AS latestLike, person AS person
      RETURN latestLike.likeTime AS likeTime
        ORDER BY likeTime
      """
    Then the result should be, in order:
      | likeTime |
      | 20160614 |
    And no side effects
