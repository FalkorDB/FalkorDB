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

Feature: Temporal6 - Render Temporal Values as a String

  
  Scenario: [1] Should serialize date
    Given any graph
    When executing query:
      """
      WITH date({year: 1984, month: 10, day: 11}) AS d
      RETURN toString(d) AS ts, date(toString(d)) = d AS b
      """
    Then the result should be, in any order:
      | ts           | b    |
      | '1984-10-11' | true |
    And no side effects

  
  Scenario: [2] Should serialize local time
    Given any graph
    When executing query:
      """
      WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d
      RETURN toString(d) AS ts, localtime(toString(d)) = d AS b
      """
    Then the result should be, in any order:
      | ts                   | b    |
      | '12:31:14.645876123' | true |
    And no side effects

  
  Scenario: [3] Should serialize time
    Given any graph
    When executing query:
      """
      WITH time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) AS d
      RETURN toString(d) AS ts, time(toString(d)) = d AS b
      """
    Then the result should be, in any order:
      | ts                         | b    |
      | '12:31:14.645876123+01:00' | true |
    And no side effects

  
  Scenario: [4] Should serialize local date time
    Given any graph
    When executing query:
      """
      WITH localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d
      RETURN toString(d) AS ts, localdatetime(toString(d)) = d AS b
      """
    Then the result should be, in any order:
      | ts                              | b    |
      | '1984-10-11T12:31:14.645876123' | true |
    And no side effects

  
  Scenario: [5] Should serialize date time
    Given any graph
    When executing query:
      """
      WITH datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) AS d
      RETURN toString(d) AS ts, datetime(toString(d)) = d AS b
      """
    Then the result should be, in any order:
      | ts                                    | b    |
      | '1984-10-11T12:31:14.645876123+01:00' | true |
    And no side effects

  
  Scenario Outline: [6] Should serialize duration
    Given any graph
    When executing query:
      """
      WITH duration(<map>) AS d
      RETURN toString(d) AS ts, duration(toString(d)) = d AS b
      """
    Then the result should be, in any order:
      | ts         | b         |
      | <toString> | <isEqual> |
    And no side effects

    Examples:
      | map                                                                                   | toString                        | isEqual |
      | {years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 1} | 'P12Y5M14DT16H13M10.000000001S' | true    |
      | {years: 12, months: 5, days: -14, hours: 16}                                          | 'P12Y5M-14DT16H'                | true    |
      | {minutes: 12, seconds: -60}                                                           | 'PT11M'                         | true    |
      | {seconds: 2, milliseconds: -1}                                                        | 'PT1.999S'                      | true    |
      | {seconds: -2, milliseconds: 1}                                                        | 'PT-1.999S'                     | true    |
      | {seconds: -2, milliseconds: -1}                                                       | 'PT-2.001S'                     | true    |
      | {days: 1, milliseconds: 1}                                                            | 'P1DT0.001S'                    | true    |
      | {days: 1, milliseconds: -1}                                                           | 'P1DT-0.001S'                   | true    |
      | {seconds: 60, milliseconds: -1}                                                       | 'PT59.999S'                     | true    |
      | {seconds: -60, milliseconds: 1}                                                       | 'PT-59.999S'                    | true    |
      | {seconds: -60, milliseconds: -1}                                                      | 'PT-1M-0.001S'                  | true    |

  
  Scenario: [7] Should serialize timezones correctly
    Given any graph
    When executing query:
      """
      WITH datetime({year: 2017, month: 8, day: 8, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: 'Europe/Stockholm'}) AS d
      RETURN toString(d) AS ts
      """
    Then the result should be, in any order:
      | ts                                                      |
      | '2017-08-08T12:31:14.645876123+02:00[Europe/Stockholm]' |
    And no side effects
