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

Feature: Temporal8 - Compute Arithmetic Operations on Temporal Values

  
  Scenario Outline: [1] Should add or subtract duration to or from date
    Given an empty graph
    And having executed:
      """
      CREATE (:Duration {dur: duration(<map>)})
      """
    When executing query:
      """
      WITH date({year: 1984, month: 10, day: 11}) AS x
      MATCH (d:Duration)
      RETURN x + d.dur AS sum, x - d.dur AS diff
      """
    Then the result should be, in any order:
      | sum   | diff   |
      | <sum> | <diff> |
    And no side effects

    Examples:
      | map                                                                                               | sum          | diff         |
      | {years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 2}             | '1997-03-25' | '1972-04-27' |
      | {months: 1, days: -14, hours: 16, minutes: -12, seconds: 70}                                      | '1984-10-28' | '1984-09-25' |
      | {years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3} | '1997-10-11' | '1971-10-12' |

  
  Scenario Outline: [2] Should add or subtract duration to or from local time
    Given an empty graph
    And having executed:
      """
      CREATE (:Duration {dur: duration(<map>)})
      """
    When executing query:
      """
      WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 1}) AS x
      MATCH (d:Duration)
      RETURN x + d.dur AS sum, x - d.dur AS diff
      """
    Then the result should be, in any order:
      | sum   | diff   |
      | <sum> | <diff> |
    And no side effects

    Examples:
      | map                                                                                               | sum                  | diff                 |
      | {years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 2}             | '04:44:24.000000003' | '20:18:03.999999999' |
      | {months: 1, days: -14, hours: 16, minutes: -12, seconds: 70}                                      | '04:20:24.000000001' | '20:42:04.000000001' |
      | {years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3} | '22:29:27.500000004' | '02:33:00.499999998' |

  
  Scenario Outline: [3] Should add or subtract duration to or from time
    Given an empty graph
    And having executed:
      """
      CREATE (:Duration {dur: duration(<map>)})
      """
    When executing query:
      """
      WITH time({hour: 12, minute: 31, second: 14, nanosecond: 1, timezone: '+01:00'}) AS x
      MATCH (d:Duration)
      RETURN x + d.dur AS sum, x - d.dur AS diff
      """
    Then the result should be, in any order:
      | sum   | diff   |
      | <sum> | <diff> |
    And no side effects

    Examples:
      | map                                                                                               | sum                        | diff                       |
      | {years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 2}             | '04:44:24.000000003+01:00' | '20:18:03.999999999+01:00' |
      | {months: 1, days: -14, hours: 16, minutes: -12, seconds: 70}                                      | '04:20:24.000000001+01:00' | '20:42:04.000000001+01:00' |
      | {years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3} | '22:29:27.500000004+01:00' | '02:33:00.499999998+01:00' |

  
  Scenario Outline: [4] Should add or subtract duration to or from local date time
    Given an empty graph
    And having executed:
      """
      CREATE (:Duration {dur: duration(<map>)})
      """
    When executing query:
      """
      WITH localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 1}) AS x
      MATCH (d:Duration)
      RETURN x + d.dur AS sum, x - d.dur AS diff
      """
    Then the result should be, in any order:
      | sum   | diff   |
      | <sum> | <diff> |
    And no side effects

    Examples:
      | map                                                                                               | sum                             | diff                            |
      | {years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 2}             | '1997-03-26T04:44:24.000000003' | '1972-04-26T20:18:03.999999999' |
      | {months: 1, days: -14, hours: 16, minutes: -12, seconds: 70}                                      | '1984-10-29T04:20:24.000000001' | '1984-09-24T20:42:04.000000001' |
      | {years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3} | '1997-10-11T22:29:27.500000004' | '1971-10-12T02:33:00.499999998' |

  
  Scenario Outline: [5] Should add or subtract duration to or from date time
    Given an empty graph
    And having executed:
      """
      CREATE (:Duration {dur: duration(<map>)})
      """
    When executing query:
      """
      WITH datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 1, timezone: '+01:00'}) AS x
      MATCH (d:Duration)
      RETURN x + d.dur AS sum, x - d.dur AS diff
      """
    Then the result should be, in any order:
      | sum   | diff   |
      | <sum> | <diff> |
    And no side effects

    Examples:
      | map                                                                                               | sum                                   | diff                                  |
      | {years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 2}             | '1997-03-26T04:44:24.000000003+01:00' | '1972-04-26T20:18:03.999999999+01:00' |
      | {months: 1, days: -14, hours: 16, minutes: -12, seconds: 70}                                      | '1984-10-29T04:20:24.000000001+01:00' | '1984-09-24T20:42:04.000000001+01:00' |
      | {years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3} | '1997-10-11T22:29:27.500000004+01:00' | '1971-10-12T02:33:00.499999998+01:00' |

  
  Scenario Outline: [6] Should add or subtract durations
    Given an empty graph
    And having executed:
      """
      CREATE (:Duration1 {date: duration(<map1>)})
      CREATE (:Duration2 {date: duration(<map2>)})
      """
    When executing query:
      """
      MATCH (dur:Duration1), (dur2: Duration2)
      RETURN dur.date + dur2.date AS sum, dur.date - dur2.date AS diff
      """
    Then the result should be, in any order:
      | sum   | diff   |
      | <sum> | <diff> |
    And no side effects

    Examples:
      | map1                                                                                              | map2                                                                                              | sum                              | diff                                 |
      | {years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 1}             | {years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 1}             | 'P24Y10M28DT32H26M20.000000002S' | 'PT0S'                               |
      | {years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 1}             | {months: 1, days: -14, hours: 16, minutes: -12, seconds: 70}                                      | 'P12Y6MT32H2M20.000000001S'      | 'P12Y4M28DT24M0.000000001S'          |
      | {years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 1}             | {years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3} | 'P25Y4M43DT50H11M23.500000004S'  | 'P-6M-15DT-17H-45M-3.500000002S'     |
      | {months: 1, days: -14, hours: 16, minutes: -12, seconds: 70}                                      | {years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 1}             | 'P12Y6MT32H2M20.000000001S'      | 'P-12Y-4M-28DT-24M-0.000000001S'     |
      | {months: 1, days: -14, hours: 16, minutes: -12, seconds: 70}                                      | {months: 1, days: -14, hours: 16, minutes: -12, seconds: 70}                                      | 'P2M-28DT31H38M20S'              | 'PT0S'                               |
      | {months: 1, days: -14, hours: 16, minutes: -12, seconds: 70}                                      | {years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3} | 'P13Y15DT49H47M23.500000003S'    | 'P-12Y-10M-43DT-18H-9M-3.500000003S' |
      | {years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3} | {years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 1}             | 'P25Y4M43DT50H11M23.500000004S'  | 'P6M15DT17H45M3.500000002S'          |
      | {years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3} | {months: 1, days: -14, hours: 16, minutes: -12, seconds: 70}                                      | 'P13Y15DT49H47M23.500000003S'    | 'P12Y10M43DT18H9M3.500000003S'       |
      | {years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3} | {years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3} | 'P25Y10M58DT67H56M27.000000006S' | 'PT0S'                               |

  
  Scenario Outline: [7] Should multiply or divide durations by numbers
    Given an empty graph
    And having executed:
      """
      CREATE (:Duration {date: duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 1})})
      """
    When executing query:
      """
      MATCH (d:Duration)
      RETURN d.date * <num1> AS prod, d.date / <num2> AS div
      """
    Then the result should be, in any order:
      | prod   | div   |
      | <prod> | <div> |
    And no side effects

    Examples:
      | num1 | num2 | prod                             | div                              |
      | 1    | 1    | 'P12Y5M14DT16H13M10.000000001S'  | 'P12Y5M14DT16H13M10.000000001S'  |
      | 2    | 2    | 'P24Y10M28DT32H26M20.000000002S' | 'P6Y2M22DT13H21M8S'              |
      | 0.5  | 0.5  | 'P6Y2M22DT13H21M8S'              | 'P24Y10M28DT32H26M20.000000002S' |
