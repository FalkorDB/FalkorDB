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

Feature: WithOrderBy1 - Order by a single variable
# LIMIT is used in the following scenarios to surface the effects or WITH ... ORDER BY ...
# which are otherwise lost after the WITH clause according to Cypher semantics

  @skip
  Scenario: [1] Sort booleans in ascending order
    Given an empty graph
    When executing query:
      """
      UNWIND [true, false] AS bools
      WITH bools
        ORDER BY bools
        LIMIT 1
      RETURN bools
      """
    Then the result should be, in order:
      | bools |
      | false |
    And no side effects

  @skip
  Scenario: [2] Sort booleans in descending order
    Given an empty graph
    When executing query:
      """
      UNWIND [true, false] AS bools
      WITH bools
        ORDER BY bools DESC
        LIMIT 1
      RETURN bools
      """
    Then the result should be, in order:
      | bools |
      | true  |
    And no side effects

  @skip
  Scenario: [3] Sort integers in ascending order
    Given an empty graph
    When executing query:
      """
      UNWIND [1, 3, 2] AS ints
      WITH ints
        ORDER BY ints
        LIMIT 2
      RETURN ints
      """
    Then the result should be, in order:
      | ints |
      | 1    |
      | 2    |
    And no side effects

  @skip
  Scenario: [4] Sort integers in descending order
    Given an empty graph
    When executing query:
      """
      UNWIND [1, 3, 2] AS ints
      WITH ints
        ORDER BY ints DESC
        LIMIT 2
      RETURN ints
      """
    Then the result should be, in order:
      | ints |
      | 3    |
      | 2    |
    And no side effects

  @skip
  Scenario: [5] Sort floats in ascending order
    Given an empty graph
    When executing query:
      """
      UNWIND [1.5, 1.3, 999.99] AS floats
      WITH floats
        ORDER BY floats
        LIMIT 2
      RETURN floats
      """
    Then the result should be, in order:
      | floats |
      | 1.3    |
      | 1.5    |
    And no side effects

  @skip
  Scenario: [6] Sort floats in descending order
    Given an empty graph
    When executing query:
      """
      UNWIND [1.5, 1.3, 999.99] AS floats
      WITH floats
        ORDER BY floats DESC
        LIMIT 2
      RETURN floats
      """
    Then the result should be, in order:
      | floats |
      | 999.99 |
      | 1.5    |
    And no side effects

  @skip
  Scenario: [7] Sort strings in ascending order
    Given an empty graph
    When executing query:
      """
      UNWIND ['.*', '', ' ', 'one'] AS strings
      WITH strings
        ORDER BY strings
        LIMIT 2
      RETURN strings
      """
    Then the result should be, in order:
      | strings |
      | ''      |
      | ' '     |
    And no side effects

  @skip
  Scenario: [8] Sort strings in descending order
    Given an empty graph
    When executing query:
      """
      UNWIND ['.*', '', ' ', 'one'] AS strings
      WITH strings
        ORDER BY strings DESC
        LIMIT 2
      RETURN strings
      """
    Then the result should be, in order:
      | strings |
      | 'one'   |
      | '.*'    |
    And no side effects

  @skip
  Scenario: [9] Sort lists in ascending order
    Given an empty graph
    When executing query:
      """
      UNWIND [[], ['a'], ['a', 1], [1], [1, 'a'], [1, null], [null, 1], [null, 2]] AS lists
      WITH lists
        ORDER BY lists
        LIMIT 4
      RETURN lists
      """
    Then the result should be, in order:
      | lists     |
      | []        |
      | ['a']     |
      | ['a', 1]  |
      | [1]       |
    And no side effects

  @skip
  Scenario: [10] Sort lists in descending order
    Given an empty graph
    When executing query:
      """
      UNWIND [[], ['a'], ['a', 1], [1], [1, 'a'], [1, null], [null, 1], [null, 2]] AS lists
      WITH lists
        ORDER BY lists DESC
        LIMIT 4
      RETURN lists
      """
    Then the result should be, in order:
      | lists     |
      | [null, 2] |
      | [null, 1] |
      | [1, null] |
      | [1, 'a']  |
    And no side effects

  @skip
  Scenario: [11] Sort dates in ascending order
    Given an empty graph
    When executing query:
      """
      UNWIND [date({year: 1910, month: 5, day: 6}),
              date({year: 1980, month: 12, day: 24}),
              date({year: 1984, month: 10, day: 12}),
              date({year: 1985, month: 5, day: 6}),
              date({year: 1980, month: 10, day: 24}),
              date({year: 1984, month: 10, day: 11})] AS dates
      WITH dates
        ORDER BY dates
        LIMIT 2
      RETURN dates
      """
    Then the result should be, in order:
      | dates        |
      | '1910-05-06' |
      | '1980-10-24' |
    And no side effects

  @skip
  Scenario: [12] Sort dates in descending order
    Given an empty graph
    When executing query:
      """
      UNWIND [date({year: 1910, month: 5, day: 6}),
              date({year: 1980, month: 12, day: 24}),
              date({year: 1984, month: 10, day: 12}),
              date({year: 1985, month: 5, day: 6}),
              date({year: 1980, month: 10, day: 24}),
              date({year: 1984, month: 10, day: 11})] AS dates
      WITH dates
        ORDER BY dates DESC
        LIMIT 2
      RETURN dates
      """
    Then the result should be, in order:
      | dates        |
      | '1985-05-06' |
      | '1984-10-12' |
    And no side effects

  @skip
  Scenario: [13] Sort local times in ascending order
    Given an empty graph
    When executing query:
      """
      UNWIND [localtime({hour: 10, minute: 35}),
              localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}),
              localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124}),
              localtime({hour: 12, minute: 35, second: 13}),
              localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})] AS localtimes
      WITH localtimes
        ORDER BY localtimes
        LIMIT 3
      RETURN localtimes
      """
    Then the result should be, in order:
      | localtimes           |
      | '10:35'              |
      | '12:30:14.645876123' |
      | '12:31:14.645876123' |
    And no side effects

  @skip
  Scenario: [14] Sort local times in descending order
    Given an empty graph
    When executing query:
      """
      UNWIND [localtime({hour: 10, minute: 35}),
              localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}),
              localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124}),
              localtime({hour: 12, minute: 35, second: 13}),
              localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})] AS localtimes
      WITH localtimes
        ORDER BY localtimes DESC
        LIMIT 3
      RETURN localtimes
      """
    Then the result should be, in order:
      | localtimes           |
      | '12:35:13'           |
      | '12:31:14.645876124' |
      | '12:31:14.645876123' |
    And no side effects

  @skip
  Scenario: [15] Sort times in ascending order
    Given an empty graph
    When executing query:
      """
      UNWIND [time({hour: 10, minute: 35, timezone: '-08:00'}),
              time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}),
              time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'}),
              time({hour: 12, minute: 35, second: 15, timezone: '+05:00'}),
              time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})] AS times
      WITH times
        ORDER BY times
        LIMIT 3
      RETURN times
      """
    Then the result should be, in order:
      | times                      |
      | '12:35:15+05:00'           |
      | '12:30:14.645876123+01:01' |
      | '12:31:14.645876123+01:00' |
    And no side effects

  @skip
  Scenario: [16] Sort times in descending order
    Given an empty graph
    When executing query:
      """
      UNWIND [time({hour: 10, minute: 35, timezone: '-08:00'}),
              time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}),
              time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'}),
              time({hour: 12, minute: 35, second: 15, timezone: '+05:00'}),
              time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})] AS times
      WITH times
        ORDER BY times DESC
        LIMIT 3
      RETURN times
      """
    Then the result should be, in order:
      | times                      |
      | '10:35-08:00'              |
      | '12:31:14.645876124+01:00' |
      | '12:31:14.645876123+01:00' |
    And no side effects

  @skip
  Scenario: [17] Sort local date times in ascending order
    Given an empty graph
    When executing query:
      """
      UNWIND [localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12}),
              localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}),
              localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1}),
              localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999}),
              localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})] AS localdatetimes
      WITH localdatetimes
        ORDER BY localdatetimes
        LIMIT 3
      RETURN localdatetimes
      """
    Then the result should be, in order:
      | localdatetimes                  |
      | '0001-01-01T01:01:01.000000001' |
      | '1980-12-11T12:31:14'           |
      | '1984-10-11T12:30:14.000000012' |
    And no side effects

  @skip
  Scenario: [18] Sort local date times in descending order
    Given an empty graph
    When executing query:
      """
      UNWIND [localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12}),
              localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}),
              localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1}),
              localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999}),
              localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})] AS localdatetimes
      WITH localdatetimes
        ORDER BY localdatetimes DESC
        LIMIT 3
      RETURN localdatetimes
      """
    Then the result should be, in order:
      | localdatetimes                  |
      | '9999-09-09T09:59:59.999999999' |
      | '1984-10-11T12:31:14.645876123' |
      | '1984-10-11T12:30:14.000000012' |
    And no side effects

  @skip
  Scenario: [19] Sort date times in ascending order
    Given an empty graph
    When executing query:
      """
      UNWIND [datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'}),
              datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'}),
              datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'}),
              datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'}),
              datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})] AS datetimes
      WITH datetimes
        ORDER BY datetimes
        LIMIT 3
      RETURN datetimes
      """
    Then the result should be, in order:
      | datetimes                             |
      | '0001-01-01T01:01:01.000000001-11:59' |
      | '1980-12-11T12:31:14-11:59'           |
      | '1984-10-11T12:31:14.645876123+00:17' |
    And no side effects

  @skip
  Scenario: [20] Sort date times in descending order
    Given an empty graph
    When executing query:
      """
      UNWIND [datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'}),
              datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'}),
              datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'}),
              datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'}),
              datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})] AS datetimes
      WITH datetimes
        ORDER BY datetimes DESC
        LIMIT 3
      RETURN datetimes
      """
    Then the result should be, in order:
      | datetimes                             |
      | '9999-09-09T09:59:59.999999999+11:59' |
      | '1984-10-11T12:30:14.000000012+00:15' |
      | '1984-10-11T12:31:14.645876123+00:17' |
    And no side effects

  @skip
  Scenario: [21] Sort distinct types in ascending order
    Given an empty graph
    And having executed:
      """
      CREATE (:N)-[:REL]->()
      """
    When executing query:
      """
      MATCH p = (n:N)-[r:REL]->()
      UNWIND [n, r, p, 1.5, ['list'], 'text', null, false, 0.0 / 0.0, {a: 'map'}] AS types
      WITH types
        ORDER BY types
        LIMIT 5
      RETURN types
      """
    Then the result should be, in any order:
      | types             |
      | {a: 'map'}        |
      | (:N)              |
      | [:REL]            |
      | ['list']          |
      | <(:N)-[:REL]->()> |
    And no side effects

  @skip
  Scenario: [22] Sort distinct types in descending order
    Given an empty graph
    And having executed:
      """
      CREATE (:N)-[:REL]->()
      """
    When executing query:
      """
      MATCH p = (n:N)-[r:REL]->()
      UNWIND [n, r, p, 1.5, ['list'], 'text', null, false, 0.0 / 0.0, {a: 'map'}] AS types
      WITH types
        ORDER BY types DESC
        LIMIT 5
      RETURN types
      """
    Then the result should be, in any order:
      | types             |
      | null              |
      | NaN               |
      | 1.5               |
      | false             |
      | 'text'            |
    And no side effects

  Scenario Outline: [23] Sort by a boolean variable projected from a node property in ascending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {bool: true}),
             (:B {bool: false}),
             (:C {bool: false}),
             (:D {bool: true}),
             (:E {bool: false})
      """
    When executing query:
      """
      MATCH (a)
      WITH a, a.bool AS bool
      WITH a, bool
        ORDER BY <sort>
        LIMIT 3
      RETURN a, bool
      """
    Then the result should be, in any order:
      | a                  | bool  |
      | (:B {bool: false}) | false |
      | (:C {bool: false}) | false |
      | (:E {bool: false}) | false |
    And no side effects

    Examples:
      | sort           |
      | bool           |
      | bool ASC       |
      | bool ASCENDING |

  Scenario Outline: [24] Sort by a boolean variable projected from a node property in descending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {bool: true}),
             (:B {bool: false}),
             (:C {bool: false}),
             (:D {bool: true}),
             (:E {bool: false})
      """
    When executing query:
      """
      MATCH (a)
      WITH a, a.bool AS bool
      WITH a, bool
        ORDER BY <sort>
        LIMIT 2
      RETURN a, bool
      """
    Then the result should be, in any order:
      | a                 | bool |
      | (:A {bool: true}) | true |
      | (:D {bool: true}) | true |
    And no side effects

    Examples:
      | sort            |
      | bool DESC       |
      | bool DESCENDING |

  Scenario Outline: [25] Sort by an integer variable projected from a node property in ascending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 9}),
             (:B {num: 5}),
             (:C {num: 30}),
             (:D {num: -11}),
             (:E {num: 7054})
      """
    When executing query:
      """
      MATCH (a)
      WITH a, a.num AS num
      WITH a, num
        ORDER BY <sort>
        LIMIT 3
      RETURN a, num
      """
    Then the result should be, in any order:
      | a               | num |
      | (:D {num: -11}) | -11 |
      | (:B {num: 5})   | 5   |
      | (:A {num: 9})   | 9   |
    And no side effects

    Examples:
      | sort          |
      | num           |
      | num ASC       |
      | num ASCENDING |

  Scenario Outline: [26] Sort by an integer variable projected from a node property in descending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 9}),
             (:B {num: 5}),
             (:C {num: 30}),
             (:D {num: -11}),
             (:E {num: 7054})
      """
    When executing query:
      """
      MATCH (a)
      WITH a, a.num AS num
      WITH a, num
        ORDER BY <sort>
        LIMIT 3
      RETURN a, num
      """
    Then the result should be, in any order:
      | a                | num  |
      | (:E {num: 7054}) | 7054 |
      | (:C {num: 30})   | 30   |
      | (:A {num: 9})    | 9    |
    And no side effects

    Examples:
      | sort           |
      | num DESC       |
      | num DESCENDING |

  Scenario Outline: [27] Sort by a float variable projected from a node property in ascending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 5.025648}),
             (:B {num: 30.94857}),
             (:C {num: 30.94856}),
             (:D {num: -11.2943}),
             (:E {num: 7054.008})
      """
    When executing query:
      """
      MATCH (a)
      WITH a, a.num AS num
      WITH a, num
        ORDER BY <sort>
        LIMIT 3
      RETURN a, num
      """
    Then the result should be, in any order:
      | a                    | num      |
      | (:D {num: -11.2943}) | -11.2943 |
      | (:A {num: 5.025648}) | 5.025648 |
      | (:C {num: 30.94856}) | 30.94856 |
    And no side effects

    Examples:
      | sort          |
      | num           |
      | num ASC       |
      | num ASCENDING |

  Scenario Outline: [28] Sort by a float variable projected from a node property in descending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {num: 5.025648}),
             (:B {num: 30.94857}),
             (:C {num: 30.94856}),
             (:D {num: -11.2943}),
             (:E {num: 7054.008})
      """
    When executing query:
      """
      MATCH (a)
      WITH a, a.num AS num
      WITH a, num
        ORDER BY <sort>
        LIMIT 3
      RETURN a, num
      """
    Then the result should be, in any order:
      | a                    | num      |
      | (:E {num: 7054.008}) | 7054.008 |
      | (:B {num: 30.94857}) | 30.94857 |
      | (:C {num: 30.94856}) | 30.94856 |
    And no side effects

    Examples:
      | sort           |
      | num DESC       |
      | num DESCENDING |

  Scenario Outline: [29] Sort by a string variable projected from a node property in ascending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {name: 'lorem'}),
             (:B {name: 'ipsum'}),
             (:C {name: 'dolor'}),
             (:D {name: 'sit'}),
             (:E {name: 'amet'})
      """
    When executing query:
      """
      MATCH (a)
      WITH a, a.name AS name
      WITH a, name
        ORDER BY <sort>
        LIMIT 3
      RETURN a, name
      """
    Then the result should be, in any order:
      | a                    | name    |
      | (:E {name: 'amet'})  | 'amet'  |
      | (:C {name: 'dolor'}) | 'dolor' |
      | (:B {name: 'ipsum'}) | 'ipsum' |
    And no side effects

    Examples:
      | sort           |
      | name           |
      | name ASC       |
      | name ASCENDING |

  Scenario Outline: [30] Sort by a string variable projected from a node property in descending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {name: 'lorem'}),
             (:B {name: 'ipsum'}),
             (:C {name: 'dolor'}),
             (:D {name: 'sit'}),
             (:E {name: 'amet'})
      """
    When executing query:
      """
      MATCH (a)
      WITH a, a.name AS name
      WITH a, name
        ORDER BY <sort>
        LIMIT 3
      RETURN a, name
      """
    Then the result should be, in any order:
      | a                    | name    |
      | (:D {name: 'sit'})   | 'sit'   |
      | (:A {name: 'lorem'}) | 'lorem' |
      | (:B {name: 'ipsum'}) | 'ipsum' |
    And no side effects

    Examples:
      | sort            |
      | name DESC       |
      | name DESCENDING |

  Scenario Outline: [31] Sort by a list variable projected from a node property in ascending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {list: [2, -2]}),
             (:B {list: [1, 2]}),
             (:C {list: [300, 0]}),
             (:D {list: [1, -20]}),
             (:E {list: [2, -2, 100]})
      """
    When executing query:
      """
      MATCH (a)
      WITH a, a.list AS list
      WITH a, list
        ORDER BY <sort>
        LIMIT 3
      RETURN a, list
      """
    Then the result should be, in any order:
      | a                      | list      |
      | (:B {list: [1, 2]})    | [1, 2]    |
      | (:D {list: [1, -20]})  | [1, -20]  |
      | (:A {list: [2, -2]})   | [2, -2]   |
    And no side effects

    Examples:
      | sort           |
      | list           |
      | list ASC       |
      | list ASCENDING |

  Scenario Outline: [32] Sort by a list variable projected from a node property in descending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {list: [2, -2]}),
             (:B {list: [1, 2]}),
             (:C {list: [300, 0]}),
             (:D {list: [1, -20]}),
             (:E {list: [2, -2, 100]})
      """
    When executing query:
      """
      MATCH (a)
      WITH a, a.list AS list
      WITH a, list
        ORDER BY <sort>
        LIMIT 3
      RETURN a, list
      """
    Then the result should be, in any order:
      | a                         | list         |
      | (:C {list: [300, 0]})     | [300, 0]     |
      | (:E {list: [2, -2, 100]}) | [2, -2, 100] |
      | (:A {list: [2, -2]})      | [2, -2]      |
    And no side effects

    Examples:
      | sort            |
      | list DESC       |
      | list DESCENDING |

  @skip
  Scenario Outline: [33] Sort by a date variable projected from a node property in ascending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {date: date({year: 1910, month: 5, day: 6})}),
             (:B {date: date({year: 1980, month: 12, day: 24})}),
             (:C {date: date({year: 1984, month: 10, day: 12})}),
             (:D {date: date({year: 1985, month: 5, day: 6})}),
             (:E {date: date({year: 1980, month: 10, day: 24})}),
             (:F {date: date({year: 1984, month: 10, day: 11})})
      """
    When executing query:
      """
      MATCH (a)
      WITH a, a.date AS date
      WITH a, date
        ORDER BY <sort>
        LIMIT 2
      RETURN a, date
      """
    Then the result should be, in any order:
      | a                         | date         |
      | (:A {date: '1910-05-06'}) | '1910-05-06' |
      | (:E {date: '1980-10-24'}) | '1980-10-24' |
    And no side effects

    Examples:
      | sort           |
      | date           |
      | date ASC       |
      | date ASCENDING |

  @skip
  Scenario Outline: [34] Sort by a date variable projected from a node property in descending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {date: date({year: 1910, month: 5, day: 6})}),
             (:B {date: date({year: 1980, month: 12, day: 24})}),
             (:C {date: date({year: 1984, month: 10, day: 12})}),
             (:D {date: date({year: 1985, month: 5, day: 6})}),
             (:E {date: date({year: 1980, month: 10, day: 24})}),
             (:F {date: date({year: 1984, month: 10, day: 11})})
      """
    When executing query:
      """
      MATCH (a)
      WITH a, a.date AS date
      WITH a, date
        ORDER BY <sort>
        LIMIT 2
      RETURN a, date
      """
    Then the result should be, in any order:
      | a                         | date         |
      | (:D {date: '1985-05-06'}) | '1985-05-06' |
      | (:C {date: '1984-10-12'}) | '1984-10-12' |
    And no side effects

    Examples:
      | sort            |
      | date DESC       |
      | date DESCENDING |

  @skip
  Scenario Outline: [35] Sort by a local time variable projected from a node property in ascending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {time: localtime({hour: 10, minute: 35})}),
             (:B {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
             (:C {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124})}),
             (:D {time: localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})}),
             (:E {time: localtime({hour: 12, minute: 31, second: 15})})
      """
    When executing query:
      """
      MATCH (a)
      WITH a, a.time AS time
      WITH a, time
        ORDER BY <sort>
        LIMIT 3
      RETURN a, time
      """
    Then the result should be, in any order:
      | a                                 | time                 |
      | (:A {time: '10:35'})              | '10:35'              |
      | (:D {time: '12:30:14.645876123'}) | '12:30:14.645876123' |
      | (:B {time: '12:31:14.645876123'}) | '12:31:14.645876123' |
    And no side effects

    Examples:
      | sort           |
      | time           |
      | time ASC       |
      | time ASCENDING |

  @skip
  Scenario Outline: [36] Sort by a local time variable projected from a node property in descending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {time: localtime({hour: 10, minute: 35})}),
             (:B {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
             (:C {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124})}),
             (:D {time: localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})}),
             (:E {time: localtime({hour: 12, minute: 31, second: 15})})
      """
    When executing query:
      """
      MATCH (a)
      WITH a, a.time AS time
      WITH a, time
        ORDER BY <sort>
        LIMIT 3
      RETURN a, time
      """
    Then the result should be, in any order:
      | a                                 | time                 |
      | (:E {time: '12:31:15'})           | '12:31:15'           |
      | (:C {time: '12:31:14.645876124'}) | '12:31:14.645876124' |
      | (:B {time: '12:31:14.645876123'}) | '12:31:14.645876123' |
    And no side effects

    Examples:
      | sort            |
      | time DESC       |
      | time DESCENDING |

  @skip
  Scenario Outline: [37] Sort by a time variable projected from a node property in ascending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {time: time({hour: 10, minute: 35, timezone: '-08:00'})}),
             (:B {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})}),
             (:C {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'})}),
             (:D {time: time({hour: 12, minute: 35, second: 15, timezone: '+05:00'})}),
             (:E {time: time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})})
      """
    When executing query:
      """
      MATCH (a)
      WITH a, a.time AS time
      WITH a, time
        ORDER BY <sort>
        LIMIT 3
      RETURN a, time
      """
    Then the result should be, in any order:
      | a                                       | time                       |
      | (:D {time: '12:35:15+05:00'})           | '12:35:15+05:00'           |
      | (:E {time: '12:30:14.645876123+01:01'}) | '12:30:14.645876123+01:01' |
      | (:B {time: '12:31:14.645876123+01:00'}) | '12:31:14.645876123+01:00' |
    And no side effects

    Examples:
      | sort           |
      | time           |
      | time ASC       |
      | time ASCENDING |

  @skip
  Scenario Outline: [38] Sort by a time variable projected from a node property in descending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {time: time({hour: 10, minute: 35, timezone: '-08:00'})}),
             (:B {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})}),
             (:C {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'})}),
             (:D {time: time({hour: 12, minute: 35, second: 15, timezone: '+05:00'})}),
             (:E {time: time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})})
      """
    When executing query:
      """
      MATCH (a)
      WITH a, a.time AS time
      WITH a, time
        ORDER BY <sort>
        LIMIT 3
      RETURN a, time
      """
    Then the result should be, in any order:
      | a                                       | time                       |
      | (:A {time: '10:35-08:00'})              | '10:35-08:00'              |
      | (:C {time: '12:31:14.645876124+01:00'}) | '12:31:14.645876124+01:00' |
      | (:B {time: '12:31:14.645876123+01:00'}) | '12:31:14.645876123+01:00' |
    And no side effects

    Examples:
      | sort            |
      | time DESC       |
      | time DESCENDING |

  @skip
  Scenario Outline: [39] Sort by a local date time variable projected from a node property in ascending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12})}),
             (:B {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
             (:C {datetime: localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1})}),
             (:D {datetime: localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999})}),
             (:E {datetime: localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})})
      """
    When executing query:
      """
      MATCH (a)
      WITH a, a.datetime AS datetime
      WITH a, datetime
        ORDER BY <sort>
        LIMIT 3
      RETURN a, datetime
      """
    Then the result should be, in any order:
      | a                                                | datetime                        |
      | (:C {datetime: '0001-01-01T01:01:01.000000001'}) | '0001-01-01T01:01:01.000000001' |
      | (:E {datetime: '1980-12-11T12:31:14'})           | '1980-12-11T12:31:14'           |
      | (:A {datetime: '1984-10-11T12:30:14.000000012'}) | '1984-10-11T12:30:14.000000012' |
    And no side effects

    Examples:
      | sort               |
      | datetime           |
      | datetime ASC       |
      | datetime ASCENDING |

  @skip
  Scenario Outline: [40] Sort by a local date time variable projected from a node property in descending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12})}),
             (:B {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
             (:C {datetime: localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1})}),
             (:D {datetime: localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999})}),
             (:E {datetime: localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})})
      """
    When executing query:
      """
      MATCH (a)
      WITH a, a.datetime AS datetime
      WITH a, datetime
        ORDER BY <sort>
        LIMIT 3
      RETURN a, datetime
      """
    Then the result should be, in any order:
      | a                                                | datetime                        |
      | (:D {datetime: '9999-09-09T09:59:59.999999999'}) | '9999-09-09T09:59:59.999999999' |
      | (:B {datetime: '1984-10-11T12:31:14.645876123'}) | '1984-10-11T12:31:14.645876123' |
      | (:A {datetime: '1984-10-11T12:30:14.000000012'}) | '1984-10-11T12:30:14.000000012' |
    And no side effects

    Examples:
      | sort                |
      | datetime DESC       |
      | datetime DESCENDING |

  @skip
  Scenario Outline: [41] Sort by a date time variable projected from a node property in ascending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'})}),
             (:B {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'})}),
             (:C {datetime: datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'})}),
             (:D {datetime: datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'})}),
             (:E {datetime: datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})})
      """
    When executing query:
      """
      MATCH (a)
      WITH a, a.datetime AS datetime
      WITH a, datetime
        ORDER BY <sort>
        LIMIT 3
      RETURN a, datetime
      """
    Then the result should be, in any order:
      | a                                                      | datetime                              |
      | (:C {datetime: '0001-01-01T01:01:01.000000001-11:59'}) | '0001-01-01T01:01:01.000000001-11:59' |
      | (:E {datetime: '1980-12-11T12:31:14-11:59'})           | '1980-12-11T12:31:14-11:59'           |
      | (:B {datetime: '1984-10-11T12:31:14.645876123+00:17'}) | '1984-10-11T12:31:14.645876123+00:17' |
    And no side effects

    Examples:
      | sort               |
      | datetime           |
      | datetime ASC       |
      | datetime ASCENDING |

  @skip
  Scenario Outline: [42] Sort by a date time variable projected from a node property in descending order
    Given an empty graph
    And having executed:
      """
      CREATE (:A {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'})}),
             (:B {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'})}),
             (:C {datetime: datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'})}),
             (:D {datetime: datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'})}),
             (:E {datetime: datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})})
      """
    When executing query:
      """
      MATCH (a)
      WITH a, a.datetime AS datetime
      WITH a, datetime
        ORDER BY <sort>
        LIMIT 3
      RETURN a, datetime
      """
    Then the result should be, in any order:
      | a                                                      | datetime                              |
      | (:D {datetime: '9999-09-09T09:59:59.999999999+11:59'}) | '9999-09-09T09:59:59.999999999+11:59' |
      | (:A {datetime: '1984-10-11T12:30:14.000000012+00:15'}) | '1984-10-11T12:30:14.000000012+00:15' |
      | (:B {datetime: '1984-10-11T12:31:14.645876123+00:17'}) | '1984-10-11T12:31:14.645876123+00:17' |
    And no side effects

    Examples:
      | sort                |
      | datetime DESC       |
      | datetime DESCENDING |

  Scenario Outline: [43] Sort by a variable that is only partially orderable on a non-distinct binding table
    Given an empty graph
    When executing query:
      """
      UNWIND [0, 2, 1, 2, 0, 1] AS x
      WITH x
        ORDER BY x <dir>
        LIMIT 2
      RETURN x
      """
    Then the result should be, in any order:
      | x   |
      | <x> |
      | <x> |
    And no side effects

    Examples:
      | dir  | x |
      | ASC  | 0 |
      | DESC | 2 |

  Scenario Outline: [44] Sort by a variable that is only partially orderable on a non-distinct binding table, but made distinct
    Given an empty graph
    When executing query:
      """
      UNWIND [0, 2, 1, 2, 0, 1] AS x
      WITH DISTINCT x
        ORDER BY x <dir>
        LIMIT 1
      RETURN x
      """
    Then the result should be, in any order:
      | x   |
      | <x> |
    And no side effects

    Examples:
      | dir  | x |
      | ASC  | 0 |
      | DESC | 2 |

  @skip
  Scenario Outline: [45] Sort order should be consistent with comparisons where comparisons are defined #Example: <exampleName>
    Given an empty graph
    When executing query:
      """
      WITH <values> AS values
      WITH values, size(values) AS numOfValues
      UNWIND values AS value
      WITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues
        ORDER BY value
      WITH numOfValues, collect(x) AS orderedX
      RETURN orderedX = range(0, numOfValues-1) AS equal
      """
    Then the result should be, in any order:
      | equal |
      | true  |
    And no side effects

    Examples:
      | exampleName    | values                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
      | booleans       | [true, false]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
      | integers       | [351, -3974856, 93, -3, 123, 0, 3, -2, 20934587, 1, 20934585, 20934586, -10]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
      | floats         | [351.5, -3974856.01, -3.203957, 123.0002, 123.0001, 123.00013, 123.00011, 0.0100000, 0.0999999, 0.00000001, 3.0, 209345.87, -10.654]                                                                                                                                                                                                                                                                                                                                                                                                                                         |
      | string         | ['Sort', 'order', ' ', 'should', 'be', '', 'consistent', 'with', 'comparisons', ', ', 'where', 'comparisons are', 'defined', '!']                                                                                                                                                                                                                                                                                                                                                                                                                                            |
      | lists          | [[2, 2], [2, -2], [1, 2], [], [1], [300, 0], [1, -20], [2, -2, 100]]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
      | dates          | [date({year: 1910, month: 5, day: 6}), date({year: 1980, month: 12, day: 24}), date({year: 1984, month: 10, day: 12}), date({year: 1985, month: 5, day: 6}), date({year: 1980, month: 10, day: 24}), date({year: 1984, month: 10, day: 11})]                                                                                                                                                                                                                                                                                                                                 |
      | localtimes     | [localtime({hour: 10, minute: 35}), localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124}), localtime({hour: 12, minute: 35, second: 13}), localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123}), localtime({hour: 12, minute: 31, second: 15})]                                                                                                                                                                                                                          |
      | times          | [time({hour: 10, minute: 35, timezone: '-08:00'}), time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'}), time({hour: 12, minute: 35, second: 15, timezone: '+05:00'}), time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'}), time({hour: 12, minute: 35, second: 15, timezone: '+01:00'})]                                                                                                                                |
      | localdatetimes | [localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12}), localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1}), localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999}), localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})]                                                                            |
      | datetimes      | [datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'}), datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'}), datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'}), datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'}), datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})] |

  @skip
  Scenario Outline: [46] Fail on sorting by an undefined variable #Example: <exampleName>
    Given an empty graph
    And having executed:
      """
      CREATE (:A), (:A), (:B), (:B), (:C)
      """
    When executing query:
      """
      MATCH (a:A), (b:B), (c:C)
      WITH a, b
      WITH a
        ORDER BY <sort>
      RETURN a
      """
    Then a SyntaxError should be raised at compile time: UndefinedVariable

    Examples:
      | sort         | exampleName   |
      | c            | out of scope  |
      | c ASC        | out of scope  |
      | c ASCENDING  | out of scope  |
      | c DESC       | out of scope  |
      | c DESCENDING | out of scope  |
      | d            | never defined |
      | d ASC        | never defined |
      | d ASCENDING  | never defined |
      | d DESC       | never defined |
      | d DESCENDING | never defined |