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

Feature: Precedence4 - On null value
  # includes otherwise type-incompatible expressions

  Scenario Outline: [1] Null predicate takes precedence over comparison operator
    Given an empty graph
    When executing query:
      """
      RETURN null IS <null1> <comp> null IS <null2> AS a,
             (null IS <null1>) <comp> (null IS <null2>) AS b,
             (null IS <null1> <comp> null) IS <null2> AS c
      """
    Then the result should be, in any order:
      | a       | b       | c       |
      | <right> | <right> | <wrong> |
    And no side effects

    Examples:
      | null1    | comp | null2    | right | wrong |
      | NOT NULL | =    | NULL     | false | true  |
      | NULL     | <>   | NULL     | false | true  |
      | NULL     | <>   | NOT NULL | true  | false |

  Scenario: [2] Null predicate takes precedence over boolean negation
    Given an empty graph
    When executing query:
      """
      RETURN NOT null IS NULL AS a,
             NOT (null IS NULL) AS b,
             (NOT null) IS NULL AS c
      """
    Then the result should be, in any order:
      | a     | b     | c    |
      | false | false | true |
    And no side effects

  Scenario Outline: [3] Null predicate takes precedence over binary boolean operator
    Given an empty graph
    When executing query:
      """
      RETURN <truth1> <op> <truth2> IS <null> AS a,
             <truth1> <op> (<truth2> IS <null>) AS b,
             (<truth1> <op> <truth2>) IS <null> AS c
      """
    Then the result should be, in any order:
      | a       | b       | c       |
      | <right> | <right> | <wrong> |
    And no side effects

    Examples:
      | truth1 | op  | truth2 | null     | right | wrong |
      | null   | AND | null   | NULL     | null  | true  |
      | null   | AND | true   | NULL     | false | true  |
      | false  | AND | false  | NOT NULL | false | true  |
      | null   | OR  | false  | NULL     | null  | true  |
      | true   | OR  | null   | NULL     | true  | false |
      | true   | XOR | null   | NOT NULL | true  | false |
      | true   | XOR | false  | NULL     | true  | false |

  
  Scenario: [4] String predicate takes precedence over binary boolean operator
    Given an empty graph
    When executing query:
      """
      RETURN ('abc' STARTS WITH null OR true) = (('abc' STARTS WITH null) OR true) AS a,
             ('abc' STARTS WITH null OR true) <> ('abc' STARTS WITH (null OR true)) AS b,
             (true OR null STARTS WITH 'abc') = (true OR (null STARTS WITH 'abc')) AS c,
             (true OR null STARTS WITH 'abc') <> ((true OR null) STARTS WITH 'abc') AS d
      """
    Then the result should be, in any order:
      | a    | b    | c    | d    |
      | true | null | true | null |
    And no side effects
