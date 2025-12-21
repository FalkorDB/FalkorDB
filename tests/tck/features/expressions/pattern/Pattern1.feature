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

Feature: Pattern1 - Pattern predicate

  Scenario: [1] Matching on any single outgoing directed connection
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
      """
    When executing query:
      """
      MATCH (n) WHERE (n)-[]->() RETURN n
      """
    Then the result should be, in any order:
      | n    |
      | (:A) |
      | (:B) |
    And no side effects

  Scenario: [2] Matching on a single undirected connection
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
      """
    When executing query:
      """
      MATCH (n) WHERE (n)-[]-() RETURN n
      """
    Then the result should be, in any order:
      | n    |
      | (:A) |
      | (:B) |
      | (:C) |
      | (:D) |
    And no side effects

  Scenario: [3] Matching on any single incoming directed connection
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
      """
    When executing query:
      """
      MATCH (n) WHERE (n)<-[]-() RETURN n
      """
    Then the result should be, in any order:
      | n    |
      | (:A) |
      | (:B) |
      | (:C) |
      | (:D) |
    And no side effects

  Scenario: [4] Matching on a specific type of single outgoing directed connection
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
      """
    When executing query:
      """
      MATCH (n) WHERE (n)-[:REL1]->() RETURN n
      """
    Then the result should be, in any order:
      | n    |
      | (:A) |
    And no side effects

  Scenario: [5] Matching on a specific type of single undirected connection
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
      """
    When executing query:
      """
      MATCH (n) WHERE (n)-[:REL1]-() RETURN n
      """
    Then the result should be, in any order:
      | n    |
      | (:A) |
      | (:B) |
      | (:D) |
    And no side effects

  Scenario: [6] Matching on a specific type of single incoming directed connection
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
      """
    When executing query:
      """
      MATCH (n) WHERE (n)<-[:REL1]-() RETURN n
      """
    Then the result should be, in any order:
      | n    |
      | (:B) |
      | (:D) |
    And no side effects

  Scenario: [7] Matching on a specific type of a variable length outgoing directed connection
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
      """
    When executing query:
      """
      MATCH (n) WHERE (n)-[:REL1*]->() RETURN n
      """
    Then the result should be, in any order:
      | n    |
      | (:A) |
    And no side effects

  Scenario: [8] Matching on a specific type of variable length undirected connection
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
      """
    When executing query:
      """
      MATCH (n) WHERE (n)-[:REL1*]-() RETURN n
      """
    Then the result should be, in any order:
      | n    |
      | (:A) |
      | (:B) |
      | (:D) |
    And no side effects

  Scenario: [9] Matching on a specific type of variable length incoming directed connection
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
      """
    When executing query:
      """
      MATCH (n) WHERE (n)<-[:REL1*]-() RETURN n
      """
    Then the result should be, in any order:
      | n    |
      | (:B) |
      | (:D) |
    And no side effects

  
  Scenario: [10] Matching on a specific type of undirected connection with length 2
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
      """
    When executing query:
      """
      MATCH (n) WHERE (n)-[:REL1*2]-() RETURN n
      """
    Then the result should be, in any order:
      | n    |
      | (:B) |
      | (:D) |
    And no side effects

  
  Scenario Outline: [10] Fail on introducing unbounded variables in pattern
    Given any graph
    When executing query:
	"""
	MATCH (n) WHERE <pattern> RETURN n
	"""
    Then a SyntaxError should be raised at compile time: UndefinedVariable

    Examples:
      | pattern                                 |
      | (a)                                     |
      | (n)-[r]->(a)                            |
      | (a)-[r]->(n)                            |
      | (n)<-[r {}]-(a)                         |
      | (n)-[r {}]-(a)                          |
      | (n)-[r]->()                             |
      | ()-[r]->(n)                             |
      | (n)<-[r]-()                             |
      | (n)-[r]-()                              |
      | ()-[r]->()                              |
      | ()<-[r]-()                              |
      | ()-[r]-()                               |
      | (n)-[r:REL]->(a {num: 5})               |
      | (n)-[r:REL*0..2]->(a {num: 5})          |
      | (n)-[r:REL]->(:C)<-[s:REL]-(a {num: 5}) |

  
  Scenario: [11] Fail on checking self pattern
    Given any graph
    When executing query:
      """
      MATCH (n) WHERE (n) RETURN n
      """
    Then a SyntaxError should be raised at compile time: InvalidArgumentType

  Scenario: [12] Matching two nodes on a single directed connection between them
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
      """
    When executing query:
      """
      MATCH (n), (m) WHERE (n)-[]->(m) RETURN n, m
      """
    Then the result should be, in any order:
      | n    | m    |
      | (:A) | (:B) |
      | (:B) | (:A) |
      | (:A) | (:C) |
      | (:A) | (:D) |
    And no side effects

  Scenario: [13] Fail on matching two nodes on a single undirected connection between them
    Given an empty graph
    And having executed:
	 """
	 CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
	 """
    When executing query:
	 """
	 MATCH (n), (m) WHERE (n)-[:REL1|REL2|REL3|REL4]-(m) RETURN n, m
	 """
    Then the result should be, in any order:
      | n    | m    |
      | (:A) | (:B) |
      | (:B) | (:A) |
      | (:A) | (:C) |
      | (:A) | (:D) |
      | (:C) | (:A) |
      | (:D) | (:A) |
    And no side effects

  Scenario: [14] Matching two nodes on a specific type of single outgoing directed connection
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
      """
    When executing query:
      """
      MATCH (n), (m) WHERE (n)-[:REL1]->(m) RETURN n, m
      """
    Then the result should be, in any order:
      | n    | m    |
      | (:A) | (:B) |
      | (:A) | (:D) |
    And no side effects

  Scenario: [15] Matching two nodes on a specific type of single undirected connection
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
      """
    When executing query:
      """
      MATCH (n), (m) WHERE (n)-[:REL1]-(m) RETURN n, m
      """
    Then the result should be, in any order:
      | n    | m    |
      | (:A) | (:B) |
      | (:B) | (:A) |
      | (:A) | (:D) |
      | (:D) | (:A) |
    And no side effects

  Scenario: [16] Matching two nodes on a specific type of a variable length outgoing directed connection
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
      """
    When executing query:
      """
      MATCH (n), (m) WHERE (n)-[:REL1*]->(m) RETURN n, m
      """
    Then the result should be, in any order:
      | n    | m    |
      | (:A) | (:B) |
      | (:A) | (:D) |
    And no side effects

  Scenario: [17] Matching two nodes on a specific type of variable length undirected connection
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
      """
    When executing query:
      """
      MATCH (n), (m) WHERE (n)-[:REL1*]-(m) RETURN n, m
      """
    Then the result should be, in any order:
      | n    | m    |
      | (:A) | (:B) |
      | (:A) | (:D) |
      | (:B) | (:A) |
      | (:B) | (:D) |
      | (:D) | (:A) |
      | (:D) | (:B) |
    And no side effects

  
  Scenario: [18] Matching two nodes on a specific type of undirected connection with length 2
    Given an empty graph
    And having executed:
      """
      CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
      """
    When executing query:
      """
      MATCH (n), (m) WHERE (n)-[:REL1*2]-(m) RETURN n, m
      """
    Then the result should be, in any order:
      | n    | m    |
      | (:D) | (:B) |
      | (:B) | (:D) |
    And no side effects


  Scenario: [19] Using a negated existential pattern predicate
    Given an empty graph
    And having executed:
	"""
	CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
	"""
    When executing query:
	"""
	MATCH (n) WHERE NOT (n)-[:REL2]-() RETURN n
	"""
    Then the result should be, in any order:
      | n    |
      | (:C) |
      | (:D) |
    And no side effects

  Scenario: [20] Using two existential pattern predicates in a conjunction
    Given an empty graph
    And having executed:
	"""
	CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
	"""
    When executing query:
	"""
	MATCH (n) WHERE (n)-[:REL1]-() AND (n)-[:REL3]-() RETURN n
	"""
    Then the result should be, in any order:
      | n    |
      | (:A) |
    And no side effects

  Scenario: [21] Using two existential pattern predicates in a disjunction
    Given an empty graph
    And having executed:
	"""
	CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
	"""
    When executing query:
	"""
	MATCH (n) WHERE (n)-[:REL1]-() OR (n)-[:REL2]-() RETURN n
	"""
    Then the result should be, in any order:
      | n    |
      | (:A) |
      | (:B) |
      | (:D) |
    And no side effects

  
  Scenario: [22] Fail on using pattern in RETURN projection
    Given any graph
    When executing query:
	"""
	MATCH (n) RETURN (n)-[]->()
	"""
    Then a SyntaxError should be raised at compile time: UnexpectedSyntax

  
  Scenario: [23] Fail on using pattern in WITH projection
    Given any graph
    When executing query:
	"""
	MATCH (n) WITH (n)-[]->() AS x RETURN x
	"""
    Then a SyntaxError should be raised at compile time: UnexpectedSyntax

  
  Scenario: [24] Fail on using pattern in right-hand side of SET
    Given any graph
    When executing query:
	"""
	MATCH (n) SET n.prop = head(nodes(head((n)-[:REL]->()))).foo
	"""
    Then a SyntaxError should be raised at compile time: UnexpectedSyntax
