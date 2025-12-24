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

Feature: Temporal5 - Access Components of Temporal Values

  
  Scenario: [1] Should provide accessors for date
    Given an empty graph
    And having executed:
      """
      CREATE (:Val {date: date({year: 1984, month: 10, day: 11})})
      """
    When executing query:
      """
      MATCH (v:Val)
      WITH v.date AS d
      RETURN d.year, d.quarter, d.month, d.week, d.weekYear, d.day, d.ordinalDay, d.weekDay, d.dayOfQuarter
      """
    Then the result should be, in any order:
      | d.year | d.quarter | d.month | d.week | d.weekYear | d.day | d.ordinalDay | d.weekDay | d.dayOfQuarter |
      | 1984   | 4         | 10      | 41     | 1984       | 11    | 285          | 4         | 11             |
    And no side effects

  
  Scenario: [2] Should provide accessors for date in last weekYear
    Given an empty graph
    And having executed:
      """
      CREATE (:Val {date: date({year: 1984, month: 1, day: 1})})
      """
    When executing query:
      """
      MATCH (v:Val)
      WITH v.date AS d
      RETURN d.year, d.weekYear, d.week, d.weekDay
      """
    Then the result should be, in any order:
      | d.year | d.weekYear | d.week | d.weekDay |
      | 1984   | 1983       | 52     | 7         |
    And no side effects

  
  Scenario: [3] Should provide accessors for local time
    Given an empty graph
    And having executed:
      """
      CREATE (:Val {date: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})})
      """
    When executing query:
      """
      MATCH (v:Val)
      WITH v.date AS d
      RETURN d.hour, d.minute, d.second, d.millisecond, d.microsecond, d.nanosecond
      """
    Then the result should be, in any order:
      | d.hour | d.minute | d.second | d.millisecond | d.microsecond | d.nanosecond |
      | 12     | 31       | 14       | 645           | 645876        | 645876123    |
    And no side effects

  
  Scenario: [4] Should provide accessors for time
    Given an empty graph
    And having executed:
      """
      CREATE (:Val {date: time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})})
      """
    When executing query:
      """
      MATCH (v:Val)
      WITH v.date AS d
      RETURN d.hour, d.minute, d.second, d.millisecond, d.microsecond, d.nanosecond, d.timezone, d.offset, d.offsetMinutes, d.offsetSeconds
      """
    Then the result should be, in any order:
      | d.hour | d.minute | d.second | d.millisecond | d.microsecond | d.nanosecond | d.timezone | d.offset | d.offsetMinutes | d.offsetSeconds |
      | 12     | 31       | 14       | 645           | 645876        | 645876123    | '+01:00'   | '+01:00' | 60              | 3600            |
    And no side effects

  
  Scenario: [5] Should provide accessors for local date time
    Given an empty graph
    And having executed:
      """
      CREATE (:Val {date: localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})})
      """
    When executing query:
      """
      MATCH (v:Val)
      WITH v.date AS d
      RETURN d.year, d.quarter, d.month, d.week, d.weekYear, d.day, d.ordinalDay, d.weekDay, d.dayOfQuarter,
             d.hour, d.minute, d.second, d.millisecond, d.microsecond, d.nanosecond
      """
    Then the result should be, in any order:
      | d.year | d.quarter | d.month | d.week | d.weekYear | d.day | d.ordinalDay | d.weekDay | d.dayOfQuarter | d.hour | d.minute | d.second | d.millisecond | d.microsecond | d.nanosecond |
      | 1984   | 4         | 11      | 45     | 1984       | 11    | 316          | 7         | 42             | 12     | 31       | 14       | 645           | 645876        | 645876123    |
    And no side effects

  
  Scenario: [6] Should provide accessors for date time
    Given an empty graph
    And having executed:
      """
      CREATE (:Val {date: datetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: 'Europe/Stockholm'})})
      """
    When executing query:
      """
      MATCH (v:Val)
      WITH v.date AS d
      RETURN d.year, d.quarter, d.month, d.week, d.weekYear, d.day, d.ordinalDay, d.weekDay, d.dayOfQuarter,
             d.hour, d.minute, d.second, d.millisecond, d.microsecond, d.nanosecond,
             d.timezone, d.offset, d.offsetMinutes, d.offsetSeconds, d.epochSeconds, d.epochMillis
      """
    Then the result should be, in any order:
      | d.year | d.quarter | d.month | d.week | d.weekYear | d.day | d.ordinalDay | d.weekDay | d.dayOfQuarter | d.hour | d.minute | d.second | d.millisecond | d.microsecond | d.nanosecond | d.timezone         | d.offset | d.offsetMinutes | d.offsetSeconds | d.epochSeconds | d.epochMillis |
      | 1984   | 4         | 11      | 45     | 1984       | 11    | 316          | 7         | 42             | 12     | 31       | 14       | 645           | 645876        | 645876123    | 'Europe/Stockholm' | '+01:00' | 60              | 3600            | 469020674      | 469020674645  |
    And no side effects

  
  Scenario: [7] Should provide accessors for duration
    Given an empty graph
    And having executed:
      """
      CREATE (:Val {date: duration({years: 1, months: 4, days: 10, hours: 1, minutes: 1, seconds: 1, nanoseconds: 111111111})})
      """
    When executing query:
      """
      MATCH (v:Val)
      WITH v.date AS d
      RETURN d.years, d.quarters, d.months, d.weeks, d.days,
             d.hours, d.minutes, d.seconds, d.milliseconds, d.microseconds, d.nanoseconds,
             d.quartersOfYear, d.monthsOfQuarter, d.monthsOfYear, d.daysOfWeek, d.minutesOfHour, d.secondsOfMinute, d.millisecondsOfSecond, d.microsecondsOfSecond, d.nanosecondsOfSecond
      """
    Then the result should be, in any order:
      | d.years | d.quarters | d.months | d.weeks | d.days | d.hours | d.minutes | d.seconds | d.milliseconds | d.microseconds | d.nanoseconds | d.quartersOfYear | d.monthsOfQuarter | d.monthsOfYear | d.daysOfWeek | d.minutesOfHour | d.secondsOfMinute | d.millisecondsOfSecond | d.microsecondsOfSecond | d.nanosecondsOfSecond |
      | 1       | 5          | 16       | 1       | 10     | 1       | 61        | 3661      | 3661111        | 3661111111     | 3661111111111 | 1                | 1                 | 4              | 3            | 1               | 1                 | 111                    | 111111                 | 111111111             |
    And no side effects
