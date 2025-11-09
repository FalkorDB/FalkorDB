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

Feature: Merge2 - Merge node - on create

  Scenario: [1] Merge node with label add label on create
    Given an empty graph
    When executing query:
      """
      MERGE (a:TheLabel)
        ON CREATE SET a:Foo
      RETURN labels(a)
      """
    Then the result should be, in any order:
      | labels(a)           |
      | ['TheLabel', 'Foo'] |
    And the side effects should be:
      | +nodes  | 1 |
      | +labels | 2 |

  Scenario: [2] ON CREATE on created nodes
    Given an empty graph
    When executing query:
      """
      MERGE (b)
        ON CREATE SET b.created = 1
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes      | 1 |
      | +properties | 1 |

  Scenario: [3] Merge node with label add property on create
    Given an empty graph
    When executing query:
      """
      MERGE (a:TheLabel)
        ON CREATE SET a.num = 42
      RETURN a.num
      """
    Then the result should be, in any order:
      | a.num |
      | 42    |
    And the side effects should be:
      | +nodes      | 1 |
      | +labels     | 1 |
      | +properties | 1 |

  Scenario: [4] Merge node with label add property on update when it exists
    Given an empty graph
    And having executed:
      """
      CREATE (:TheLabel)
      """
    When executing query:
      """
      MERGE (a:TheLabel)
        ON CREATE SET a.num = 42
      RETURN a.num
      """
    Then the result should be, in any order:
      | a.num |
      | null  |
    And no side effects

  Scenario: [5] Merge should be able to use properties of bound node in ON CREATE
    Given an empty graph
    And having executed:
      """
      CREATE (:Person {bornIn: 'New York'}),
        (:Person {bornIn: 'Ohio'})
      """
    When executing query:
      """
      MATCH (person:Person)
      MERGE (city:City)
        ON CREATE SET city.name = person.bornIn
      RETURN person.bornIn
      """
    Then the result should be, in any order:
      | person.bornIn |
      | 'New York'    |
      | 'Ohio'        |
    And the side effects should be:
      | +nodes      | 1 |
      | +labels     | 1 |
      | +properties | 1 |

  Scenario: [6] Fail when using undefined variable in ON CREATE
    Given any graph
    When executing query:
      """
      MERGE (n)
        ON CREATE SET x.num = 1
      """
    Then a SyntaxError should be raised at compile time: UndefinedVariable
