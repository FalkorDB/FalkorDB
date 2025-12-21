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

Feature: Temporal1 - Create Temporal Values from a Map

  
  Scenario Outline: [1] Should construct week date
    Given any graph
    When executing query:
      """
      RETURN date(<map>) AS d
      """
    Then the result should be, in any order:
      | d        |
      | <result> |
    And no side effects

    Examples:
      | map                                               | result       |
      | {year: 1816, week: 1}                               | '1816-01-01' |
      | {year: 1816, week: 52}                              | '1816-12-23' |
      | {year: 1817, week: 1}                               | '1816-12-30' |
      | {year: 1817, week: 10}                              | '1817-03-03' |
      | {year: 1817, week: 30}                              | '1817-07-21' |
      | {year: 1817, week: 52}                              | '1817-12-22' |
      | {year: 1818, week: 1}                               | '1817-12-29' |
      | {year: 1818, week: 52}                              | '1818-12-21' |
      | {year: 1818, week: 53}                              | '1818-12-28' |
      | {year: 1819, week: 1}                               | '1819-01-04' |
      | {year: 1819, week: 52}                              | '1819-12-27' |
      | {dayOfWeek: 2, year: 1817, week: 1}               | '1816-12-31' |
      | {date: date('1816-12-30'), week: 2, dayOfWeek: 3} | '1817-01-08' |
      | {date: date('1816-12-31'), week: 2}               | '1817-01-07' |
      | {date: date('1816-12-31'), year: 1817, week: 2}   | '1817-01-07' |

  
  Scenario Outline: [2] Should construct week localdatetime
    Given any graph
    When executing query:
      """
      RETURN localdatetime(<map>) AS d
      """
    Then the result should be, in any order:
      | d        |
      | <result> |
    And no side effects

    Examples:
      | map                                               | result             |
      | {year: 1816, week: 1}                               | '1816-01-01T00:00' |
      | {year: 1816, week: 52}                              | '1816-12-23T00:00' |
      | {year: 1817, week: 1}                               | '1816-12-30T00:00' |
      | {year: 1817, week: 10}                              | '1817-03-03T00:00' |
      | {year: 1817, week: 30}                              | '1817-07-21T00:00' |
      | {year: 1817, week: 52}                              | '1817-12-22T00:00' |
      | {year: 1818, week: 1}                               | '1817-12-29T00:00' |
      | {year: 1818, week: 52}                              | '1818-12-21T00:00' |
      | {year: 1818, week: 53}                              | '1818-12-28T00:00' |
      | {year: 1819, week: 1}                               | '1819-01-04T00:00' |
      | {year: 1819, week: 52}                              | '1819-12-27T00:00' |
      | {dayOfWeek: 2, year: 1817, week: 1}               | '1816-12-31T00:00' |
      | {date: date('1816-12-30'), week: 2, dayOfWeek: 3} | '1817-01-08T00:00' |
      | {date: date('1816-12-31'), week: 2}               | '1817-01-07T00:00' |
      | {date: date('1816-12-31'), year: 1817, week: 2}   | '1817-01-07T00:00' |

  
  Scenario Outline: [3] Should construct week datetime
    Given any graph
    When executing query:
      """
      RETURN datetime(<map>) AS d
      """
    Then the result should be, in any order:
      | d        |
      | <result> |
    And no side effects

    Examples:
      | map                                               | result              |
      | {year: 1816, week: 1}                               | '1816-01-01T00:00Z' |
      | {year: 1816, week: 52}                              | '1816-12-23T00:00Z' |
      | {year: 1817, week: 1}                               | '1816-12-30T00:00Z' |
      | {year: 1817, week: 10}                              | '1817-03-03T00:00Z' |
      | {year: 1817, week: 30}                              | '1817-07-21T00:00Z' |
      | {year: 1817, week: 52}                              | '1817-12-22T00:00Z' |
      | {year: 1818, week: 1}                               | '1817-12-29T00:00Z' |
      | {year: 1818, week: 52}                              | '1818-12-21T00:00Z' |
      | {year: 1818, week: 53}                              | '1818-12-28T00:00Z' |
      | {year: 1819, week: 1}                               | '1819-01-04T00:00Z' |
      | {year: 1819, week: 52}                              | '1819-12-27T00:00Z' |
      | {dayOfWeek: 2, year: 1817, week: 1}               | '1816-12-31T00:00Z' |
      | {date: date('1816-12-30'), week: 2, dayOfWeek: 3} | '1817-01-08T00:00Z' |
      | {date: date('1816-12-31'), week: 2}               | '1817-01-07T00:00Z' |
      | {date: date('1816-12-31'), year: 1817, week: 2}   | '1817-01-07T00:00Z' |

  
  Scenario Outline: [4] Should construct date
    Given any graph
    When executing query:
      """
      RETURN date(<map>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | map                                        | result       |
      | {year: 1984, month: 10, day: 11}           | '1984-10-11' |
      | {year: 1984, month: 10}                    | '1984-10-01' |
      | {year: 1984, week: 10, dayOfWeek: 3}       | '1984-03-07' |
      | {year: 1984, week: 10}                     | '1984-03-05' |
      | {year: 1984}                               | '1984-01-01' |
      | {year: 1984, ordinalDay: 202}              | '1984-07-20' |
      | {year: 1984, quarter: 3, dayOfQuarter: 45} | '1984-08-14' |
      | {year: 1984, quarter: 3}                   | '1984-07-01' |

  
  Scenario Outline: [5] Should construct local time
    Given any graph
    When executing query:
      """
      RETURN localtime(<map>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | map                                                                                     | result               |
      | {hour: 12, minute: 31, second: 14, nanosecond: 789, millisecond: 123, microsecond: 456} | '12:31:14.123456789' |
      | {hour: 12, minute: 31, second: 14, nanosecond: 645876123}                               | '12:31:14.645876123' |
      | {hour: 12, minute: 31, second: 14, microsecond: 645876}                                 | '12:31:14.645876'    |
      | {hour: 12, minute: 31, second: 14, millisecond: 645}                                    | '12:31:14.645'       |
      | {hour: 12, minute: 31, second: 14}                                                      | '12:31:14'           |
      | {hour: 12, minute: 31}                                                                  | '12:31'              |
      | {hour: 12}                                                                              | '12:00'              |

  
  Scenario Outline: [6] Should construct time
    Given any graph
    When executing query:
      """
      RETURN time(<map>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | map                                                                                     | result                     |
      | {hour: 12, minute: 31, second: 14, nanosecond: 789, millisecond: 123, microsecond: 456} | '12:31:14.123456789Z'      |
      | {hour: 12, minute: 31, second: 14, nanosecond: 645876123}                               | '12:31:14.645876123Z'      |
      | {hour: 12, minute: 31, second: 14, nanosecond: 3}                                       | '12:31:14.000000003Z'      |
      | {hour: 12, minute: 31, second: 14, microsecond: 645876}                                 | '12:31:14.645876Z'         |
      | {hour: 12, minute: 31, second: 14, millisecond: 645}                                    | '12:31:14.645Z'            |
      | {hour: 12, minute: 31, second: 14}                                                      | '12:31:14Z'                |
      | {hour: 12, minute: 31}                                                                  | '12:31Z'                   |
      | {hour: 12}                                                                              | '12:00Z'                   |
      | {hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}           | '12:31:14.645876123+01:00' |
      | {hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}             | '12:31:14.645876+01:00'    |
      | {hour: 12, minute: 31, second: 14, millisecond: 645, timezone: '+01:00'}                | '12:31:14.645+01:00'       |
      | {hour: 12, minute: 31, second: 14, timezone: '+01:00'}                                  | '12:31:14+01:00'           |
      | {hour: 12, minute: 31, timezone: '+01:00'}                                              | '12:31+01:00'              |
      | {hour: 12, timezone: '+01:00'}                                                          | '12:00+01:00'              |

  
  Scenario Outline: [7] Should construct local date time
    Given any graph
    When executing query:
      """
      RETURN localdatetime(<map>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | map                                                                                                                     | result                          |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 789, millisecond: 123, microsecond: 456} | '1984-10-11T12:31:14.123456789' |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}                               | '1984-10-11T12:31:14.645876123' |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 3}                                       | '1984-10-11T12:31:14.000000003' |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, microsecond: 645876}                                 | '1984-10-11T12:31:14.645876'    |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, millisecond: 645}                                    | '1984-10-11T12:31:14.645'       |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14}                                                      | '1984-10-11T12:31:14'           |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31}                                                                  | '1984-10-11T12:31'              |
      | {year: 1984, month: 10, day: 11, hour: 12}                                                                              | '1984-10-11T12:00'              |
      | {year: 1984, month: 10, day: 11}                                                                                        | '1984-10-11T00:00'              |
      | {year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, nanosecond: 645876123}                           | '1984-03-07T12:31:14.645876123' |
      | {year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, microsecond: 645876}                             | '1984-03-07T12:31:14.645876'    |
      | {year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}                                | '1984-03-07T12:31:14.645'       |
      | {year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14}                                                  | '1984-03-07T12:31:14'           |
      | {year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31}                                                              | '1984-03-07T12:31'              |
      | {year: 1984, week: 10, dayOfWeek: 3, hour: 12}                                                                          | '1984-03-07T12:00'              |
      | {year: 1984, week: 10, dayOfWeek: 3}                                                                                    | '1984-03-07T00:00'              |
      | {year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, nanosecond: 645876123}                                  | '1984-07-20T12:31:14.645876123' |
      | {year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, microsecond: 645876}                                    | '1984-07-20T12:31:14.645876'    |
      | {year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, millisecond: 645}                                       | '1984-07-20T12:31:14.645'       |
      | {year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14}                                                         | '1984-07-20T12:31:14'           |
      | {year: 1984, ordinalDay: 202, hour: 12, minute: 31}                                                                     | '1984-07-20T12:31'              |
      | {year: 1984, ordinalDay: 202, hour: 12}                                                                                 | '1984-07-20T12:00'              |
      | {year: 1984, ordinalDay: 202}                                                                                           | '1984-07-20T00:00'              |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, nanosecond: 645876123}                     | '1984-08-14T12:31:14.645876123' |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, microsecond: 645876}                       | '1984-08-14T12:31:14.645876'    |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, millisecond: 645}                          | '1984-08-14T12:31:14.645'       |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14}                                            | '1984-08-14T12:31:14'           |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31}                                                        | '1984-08-14T12:31'              |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12}                                                                    | '1984-08-14T12:00'              |
      | {year: 1984, quarter: 3, dayOfQuarter: 45}                                                                              | '1984-08-14T00:00'              |
      | {year: 1984}                                                                                                            | '1984-01-01T00:00'              |

  
  Scenario Outline: [8] Should construct date time with default time zone
    Given any graph
    When executing query:
      """
      RETURN datetime(<map>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | map                                                                                                                     | result                           |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 789, millisecond: 123, microsecond: 456} | '1984-10-11T12:31:14.123456789Z' |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}                               | '1984-10-11T12:31:14.645876123Z' |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, microsecond: 645876}                                 | '1984-10-11T12:31:14.645876Z'    |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, millisecond: 645}                                    | '1984-10-11T12:31:14.645Z'       |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14}                                                      | '1984-10-11T12:31:14Z'           |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31}                                                                  | '1984-10-11T12:31Z'              |
      | {year: 1984, month: 10, day: 11, hour: 12}                                                                              | '1984-10-11T12:00Z'              |
      | {year: 1984, month: 10, day: 11}                                                                                        | '1984-10-11T00:00Z'              |
      | {year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, nanosecond: 645876123}                           | '1984-03-07T12:31:14.645876123Z' |
      | {year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, microsecond: 645876}                             | '1984-03-07T12:31:14.645876Z'    |
      | {year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}                                | '1984-03-07T12:31:14.645Z'       |
      | {year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14}                                                  | '1984-03-07T12:31:14Z'           |
      | {year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31}                                                              | '1984-03-07T12:31Z'              |
      | {year: 1984, week: 10, dayOfWeek: 3, hour: 12}                                                                          | '1984-03-07T12:00Z'              |
      | {year: 1984, week: 10, dayOfWeek: 3}                                                                                    | '1984-03-07T00:00Z'              |
      | {year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, nanosecond: 645876123}                                  | '1984-07-20T12:31:14.645876123Z' |
      | {year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, microsecond: 645876}                                    | '1984-07-20T12:31:14.645876Z'    |
      | {year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, millisecond: 645}                                       | '1984-07-20T12:31:14.645Z'       |
      | {year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14}                                                         | '1984-07-20T12:31:14Z'           |
      | {year: 1984, ordinalDay: 202, hour: 12, minute: 31}                                                                     | '1984-07-20T12:31Z'              |
      | {year: 1984, ordinalDay: 202, hour: 12}                                                                                 | '1984-07-20T12:00Z'              |
      | {year: 1984, ordinalDay: 202}                                                                                           | '1984-07-20T00:00Z'              |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, nanosecond: 645876123}                     | '1984-08-14T12:31:14.645876123Z' |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, microsecond: 645876}                       | '1984-08-14T12:31:14.645876Z'    |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, millisecond: 645}                          | '1984-08-14T12:31:14.645Z'       |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14}                                            | '1984-08-14T12:31:14Z'           |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31}                                                        | '1984-08-14T12:31Z'              |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12}                                                                    | '1984-08-14T12:00Z'              |
      | {year: 1984, quarter: 3, dayOfQuarter: 45}                                                                              | '1984-08-14T00:00Z'              |
      | {year: 1984}                                                                                                            | '1984-01-01T00:00Z'              |

  
  Scenario Outline: [9] Should construct date time with offset time zone
    Given any graph
    When executing query:
      """
      RETURN datetime(<map>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | map                                                                                                                     | result                                |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}           | '1984-10-11T12:31:14.645876123+01:00' |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}             | '1984-10-11T12:31:14.645876+01:00'    |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, millisecond: 645, timezone: '+01:00'}                | '1984-10-11T12:31:14.645+01:00'       |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: '+01:00'}                                  | '1984-10-11T12:31:14+01:00'           |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, timezone: '+01:00'}                                              | '1984-10-11T12:31+01:00'              |
      | {year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}                                                          | '1984-10-11T12:00+01:00'              |
      | {year: 1984, month: 10, day: 11, timezone: '+01:00'}                                                                    | '1984-10-11T00:00+01:00'              |
      | {year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}       | '1984-03-07T12:31:14.645876123+01:00' |
      | {year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}         | '1984-03-07T12:31:14.645876+01:00'    |
      | {year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645, timezone: '+01:00'}            | '1984-03-07T12:31:14.645+01:00'       |
      | {year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, timezone: '+01:00'}                              | '1984-03-07T12:31:14+01:00'           |
      | {year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, timezone: '+01:00'}                                          | '1984-03-07T12:31+01:00'              |
      | {year: 1984, week: 10, dayOfWeek: 3, hour: 12, timezone: '+01:00'}                                                      | '1984-03-07T12:00+01:00'              |
      | {year: 1984, week: 10, dayOfWeek: 3, timezone: '+01:00'}                                                                | '1984-03-07T00:00+01:00'              |
      | {year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}              | '1984-07-20T12:31:14.645876123+01:00' |
      | {year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}                | '1984-07-20T12:31:14.645876+01:00'    |
      | {year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, millisecond: 645, timezone: '+01:00'}                   | '1984-07-20T12:31:14.645+01:00'       |
      | {year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, timezone: '+01:00'}                                     | '1984-07-20T12:31:14+01:00'           |
      | {year: 1984, ordinalDay: 202, hour: 12, minute: 31, timezone: '+01:00'}                                                 | '1984-07-20T12:31+01:00'              |
      | {year: 1984, ordinalDay: 202, hour: 12, timezone: '+01:00'}                                                             | '1984-07-20T12:00+01:00'              |
      | {year: 1984, ordinalDay: 202, timezone: '+01:00'}                                                                       | '1984-07-20T00:00+01:00'              |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'} | '1984-08-14T12:31:14.645876123+01:00' |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}   | '1984-08-14T12:31:14.645876+01:00'    |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, millisecond: 645, timezone: '+01:00'}      | '1984-08-14T12:31:14.645+01:00'       |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, timezone: '+01:00'}                        | '1984-08-14T12:31:14+01:00'           |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, timezone: '+01:00'}                                    | '1984-08-14T12:31+01:00'              |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, timezone: '+01:00'}                                                | '1984-08-14T12:00+01:00'              |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, timezone: '+01:00'}                                                          | '1984-08-14T00:00+01:00'              |
      | {year: 1984, timezone: '+01:00'}                                                                                        | '1984-01-01T00:00+01:00'              |

  
  Scenario Outline: [10] Should construct date time with named time zone
    Given any graph
    When executing query:
      """
      RETURN datetime(<map>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | map                                                                                                                               | result                                                  |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: 'Europe/Stockholm'}           | '1984-10-11T12:31:14.645876123+01:00[Europe/Stockholm]' |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: 'Europe/Stockholm'}             | '1984-10-11T12:31:14.645876+01:00[Europe/Stockholm]'    |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, millisecond: 645, timezone: 'Europe/Stockholm'}                | '1984-10-11T12:31:14.645+01:00[Europe/Stockholm]'       |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: 'Europe/Stockholm'}                                  | '1984-10-11T12:31:14+01:00[Europe/Stockholm]'           |
      | {year: 1984, month: 10, day: 11, hour: 12, minute: 31, timezone: 'Europe/Stockholm'}                                              | '1984-10-11T12:31+01:00[Europe/Stockholm]'              |
      | {year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}                                                          | '1984-10-11T12:00+01:00[Europe/Stockholm]'              |
      | {year: 1984, month: 10, day: 11, timezone: 'Europe/Stockholm'}                                                                    | '1984-10-11T00:00+01:00[Europe/Stockholm]'              |
      | {year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: 'Europe/Stockholm'}       | '1984-03-07T12:31:14.645876123+01:00[Europe/Stockholm]' |
      | {year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: 'Europe/Stockholm'}         | '1984-03-07T12:31:14.645876+01:00[Europe/Stockholm]'    |
      | {year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645, timezone: 'Europe/Stockholm'}            | '1984-03-07T12:31:14.645+01:00[Europe/Stockholm]'       |
      | {year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, timezone: 'Europe/Stockholm'}                              | '1984-03-07T12:31:14+01:00[Europe/Stockholm]'           |
      | {year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, timezone: 'Europe/Stockholm'}                                          | '1984-03-07T12:31+01:00[Europe/Stockholm]'              |
      | {year: 1984, week: 10, dayOfWeek: 3, hour: 12, timezone: 'Europe/Stockholm'}                                                      | '1984-03-07T12:00+01:00[Europe/Stockholm]'              |
      | {year: 1984, week: 10, dayOfWeek: 3, timezone: 'Europe/Stockholm'}                                                                | '1984-03-07T00:00+01:00[Europe/Stockholm]'              |
      | {year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: 'Europe/Stockholm'}              | '1984-07-20T12:31:14.645876123+02:00[Europe/Stockholm]' |
      | {year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: 'Europe/Stockholm'}                | '1984-07-20T12:31:14.645876+02:00[Europe/Stockholm]'    |
      | {year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, millisecond: 645, timezone: 'Europe/Stockholm'}                   | '1984-07-20T12:31:14.645+02:00[Europe/Stockholm]'       |
      | {year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, timezone: 'Europe/Stockholm'}                                     | '1984-07-20T12:31:14+02:00[Europe/Stockholm]'           |
      | {year: 1984, ordinalDay: 202, hour: 12, minute: 31, timezone: 'Europe/Stockholm'}                                                 | '1984-07-20T12:31+02:00[Europe/Stockholm]'              |
      | {year: 1984, ordinalDay: 202, hour: 12, timezone: 'Europe/Stockholm'}                                                             | '1984-07-20T12:00+02:00[Europe/Stockholm]'              |
      | {year: 1984, ordinalDay: 202, timezone: 'Europe/Stockholm'}                                                                       | '1984-07-20T00:00+02:00[Europe/Stockholm]'              |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: 'Europe/Stockholm'} | '1984-08-14T12:31:14.645876123+02:00[Europe/Stockholm]' |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: 'Europe/Stockholm'}   | '1984-08-14T12:31:14.645876+02:00[Europe/Stockholm]'    |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, millisecond: 645, timezone: 'Europe/Stockholm'}      | '1984-08-14T12:31:14.645+02:00[Europe/Stockholm]'       |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, timezone: 'Europe/Stockholm'}                        | '1984-08-14T12:31:14+02:00[Europe/Stockholm]'           |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, timezone: 'Europe/Stockholm'}                                    | '1984-08-14T12:31+02:00[Europe/Stockholm]'              |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, timezone: 'Europe/Stockholm'}                                                | '1984-08-14T12:00+02:00[Europe/Stockholm]'              |
      | {year: 1984, quarter: 3, dayOfQuarter: 45, timezone: 'Europe/Stockholm'}                                                          | '1984-08-14T00:00+02:00[Europe/Stockholm]'              |
      | {year: 1984, timezone: 'Europe/Stockholm'}                                                                                        | '1984-01-01T00:00+01:00[Europe/Stockholm]'              |

  
  Scenario: [11] Should construct date time from epoch
    Given any graph
    When executing query:
      """
      RETURN datetime.fromepoch(416779, 999999999) AS d1,
             datetime.fromepochmillis(237821673987) AS d2
      """
    Then the result should be, in any order:
      | d1                               | d2                         |
      | '1970-01-05T19:46:19.999999999Z' | '1977-07-15T13:34:33.987Z' |
    And no side effects

  
  Scenario Outline: [12] Should construct duration
    Given any graph
    When executing query:
      """
      RETURN duration(<map>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | map                                                                   | result                 |
      | {days: 14, hours: 16, minutes: 12}                                    | 'P14DT16H12M'          |
      | {months: 5, days: 1.5}                                                | 'P5M1DT12H'            |
      | {months: 0.75}                                                        | 'P22DT19H51M49.5S'     |
      | {weeks: 2.5}                                                          | 'P17DT12H'             |
      | {years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70} | 'P12Y5M14DT16H13M10S'  |
      | {days: 14, seconds: 70, milliseconds: 1}                              | 'P14DT1M10.001S'       |
      | {days: 14, seconds: 70, microseconds: 1}                              | 'P14DT1M10.000001S'    |
      | {days: 14, seconds: 70, nanoseconds: 1}                               | 'P14DT1M10.000000001S' |
      | {minutes: 1.5, seconds: 1}                                            | 'PT1M31S'              |

  
  Scenario Outline: [13] Should construct temporal with time offset with second precision
    Given any graph
    When executing query:
      """
      RETURN <temporal> AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | temporal                                                                                            | result                         |
      | time({hour: 12, minute: 34, second: 56, timezone: '+02:05:00'})                                     | '12:34:56+02:05'               |
      | time({hour: 12, minute: 34, second: 56, timezone: '+02:05:59'})                                     | '12:34:56+02:05:59'            |
      | time({hour: 12, minute: 34, second: 56, timezone: '-02:05:07'})                                     | '12:34:56-02:05:07'            |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 34, second: 56, timezone: '+02:05:59'}) | '1984-10-11T12:34:56+02:05:59' |
