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
Feature: Temporal3 - Project Temporal Values from other Temporal Values

  
  Scenario Outline: [1] Should select date
    Given any graph
    When executing query:
      """
      WITH <other> AS other
      RETURN date(<expression>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | other                                                                                                    | expression                    | result       |
      | date({year: 1984, month: 11, day: 11})                                                                   | other                         | '1984-11-11' |
      | date({year: 1984, month: 11, day: 11})                                                                   | {date: other}                 | '1984-11-11' |
      | date({year: 1984, month: 11, day: 11})                                                                   | {date: other, year: 28}       | '0028-11-11' |
      | date({year: 1984, month: 11, day: 11})                                                                   | {date: other, day: 28}        | '1984-11-28' |
      | date({year: 1984, month: 11, day: 11})                                                                   | {date: other, week: 1}        | '1984-01-08' |
      | date({year: 1984, month: 11, day: 11})                                                                   | {date: other, ordinalDay: 28} | '1984-01-28' |
      | date({year: 1984, month: 11, day: 11})                                                                   | {date: other, quarter: 3}     | '1984-08-11' |
      | localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) | other                         | '1984-11-11' |
      | localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) | {date: other}                 | '1984-11-11' |
      | localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) | {date: other, year: 28}       | '0028-11-11' |
      | localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) | {date: other, day: 28}        | '1984-11-28' |
      | localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) | {date: other, week: 1}        | '1984-01-08' |
      | localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) | {date: other, ordinalDay: 28} | '1984-01-28' |
      | localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) | {date: other, quarter: 3}     | '1984-08-11' |
      | datetime({year: 1984, month: 11, day: 11, hour: 12, timezone: '+01:00'})                                 | other                         | '1984-11-11' |
      | datetime({year: 1984, month: 11, day: 11, hour: 12, timezone: '+01:00'})                                 | {date: other}                 | '1984-11-11' |
      | datetime({year: 1984, month: 11, day: 11, hour: 12, timezone: '+01:00'})                                 | {date: other, year: 28}       | '0028-11-11' |
      | datetime({year: 1984, month: 11, day: 11, hour: 12, timezone: '+01:00'})                                 | {date: other, day: 28}        | '1984-11-28' |
      | datetime({year: 1984, month: 11, day: 11, hour: 12, timezone: '+01:00'})                                 | {date: other, week: 1}        | '1984-01-08' |
      | datetime({year: 1984, month: 11, day: 11, hour: 12, timezone: '+01:00'})                                 | {date: other, ordinalDay: 28} | '1984-01-28' |
      | datetime({year: 1984, month: 11, day: 11, hour: 12, timezone: '+01:00'})                                 | {date: other, quarter: 3}     | '1984-08-11' |

  
  Scenario Outline: [2] Should select local time
    Given any graph
    When executing query:
      """
      WITH <other> AS other
      RETURN localtime(<expression>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | other                                                                                                   | expression                | result               |
      | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | other                     | '12:31:14.645876123' |
      | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {time: other}             | '12:31:14.645876123' |
      | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {time: other, second: 42} | '12:31:42.645876123' |
      | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | other                     | '12:31:14.645876'    |
      | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {time: other}             | '12:31:14.645876'    |
      | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {time: other, second: 42} | '12:31:42.645876'    |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | other                     | '12:31:14.645'       |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {time: other}             | '12:31:14.645'       |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {time: other, second: 42} | '12:31:42.645'       |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | other                     | '12:00'              |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | {time: other}             | '12:00'              |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | {time: other, second: 42} | '12:00:42'           |

  
  Scenario Outline: [3] Should select time
    Given any graph
    When executing query:
      """
      WITH <other> AS other
      RETURN time(<expression>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | other                                                                                                   | expression                                    | result                     |
      | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | other                                         | '12:31:14.645876123Z'      |
      | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {time: other}                                 | '12:31:14.645876123Z'      |
      | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {time: other, timezone: '+05:00'}             | '12:31:14.645876123+05:00' |
      | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {time: other, second: 42}                     | '12:31:42.645876123Z'      |
      | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {time: other, second: 42, timezone: '+05:00'} | '12:31:42.645876123+05:00' |
      | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | other                                         | '12:31:14.645876+01:00'    |
      | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {time: other}                                 | '12:31:14.645876+01:00'    |
      | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {time: other, timezone: '+05:00'}             | '16:31:14.645876+05:00'    |
      | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {time: other, second: 42}                     | '12:31:42.645876+01:00'    |
      | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {time: other, second: 42, timezone: '+05:00'} | '16:31:42.645876+05:00'    |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | other                                         | '12:31:14.645Z'            |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {time: other}                                 | '12:31:14.645Z'            |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {time: other, timezone: '+05:00'}             | '12:31:14.645+05:00'       |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {time: other, second: 42}                     | '12:31:42.645Z'            |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {time: other, second: 42, timezone: '+05:00'} | '12:31:42.645+05:00'       |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | other                                         | '12:00+01:00'              |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | {time: other}                                 | '12:00+01:00'              |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | {time: other, timezone: '+05:00'}             | '16:00+05:00'              |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | {time: other, second: 42}                     | '12:00:42+01:00'           |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | {time: other, second: 42, timezone: '+05:00'} | '16:00:42+05:00'           |

  
  Scenario Outline: [4] Should select date into local date time
    Given any graph
    When executing query:
      """
      WITH <other> AS other
      RETURN localdatetime(<expression>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | other                                                                                                   | expression                                               | result                |
      | date({year: 1984, month: 10, day: 11})                                                                  | {date: other, hour: 10, minute: 10, second: 10}          | '1984-10-11T10:10:10' |
      | date({year: 1984, month: 10, day: 11})                                                                  | {date: other, day: 28, hour: 10, minute: 10, second: 10} | '1984-10-28T10:10:10' |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {date: other, hour: 10, minute: 10, second: 10}          | '1984-03-07T10:10:10' |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {date: other, day: 28, hour: 10, minute: 10, second: 10} | '1984-03-28T10:10:10' |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | {date: other, hour: 10, minute: 10, second: 10}          | '1984-10-11T10:10:10' |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | {date: other, day: 28, hour: 10, minute: 10, second: 10} | '1984-10-28T10:10:10' |

  
  Scenario Outline: [5] Should select time into local date time
    Given any graph
    When executing query:
      """
      WITH <other> AS other
      RETURN localdatetime(<expression>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | other                                                                                                   | expression                                                | result                          |
      | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {year: 1984, month: 10, day: 11, time: other}             | '1984-10-11T12:31:14.645876123' |
      | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {year: 1984, month: 10, day: 11, time: other, second: 42} | '1984-10-11T12:31:42.645876123' |
      | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {year: 1984, month: 10, day: 11, time: other}             | '1984-10-11T12:31:14.645876'    |
      | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {year: 1984, month: 10, day: 11, time: other, second: 42} | '1984-10-11T12:31:42.645876'    |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {year: 1984, month: 10, day: 11, time: other}             | '1984-10-11T12:31:14.645'       |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {year: 1984, month: 10, day: 11, time: other, second: 42} | '1984-10-11T12:31:42.645'       |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | {year: 1984, month: 10, day: 11, time: other}             | '1984-10-11T12:00'              |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | {year: 1984, month: 10, day: 11, time: other, second: 42} | '1984-10-11T12:00:42'           |

  
  Scenario Outline: [6] Should select date and time into local date time
    Given any graph
    When executing query:
      """
      WITH <otherDate> AS otherDate, <otherTime> AS otherTime
      RETURN localdatetime(<expression>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | otherDate                                                                                               | otherTime                                                                                               | expression                                              | result                          |
      | date({year: 1984, month: 10, day: 11})                                                                  | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {date: otherDate, time: otherTime}                      | '1984-10-11T12:31:14.645876123' |
      | date({year: 1984, month: 10, day: 11})                                                                  | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {date: otherDate, time: otherTime, day: 28, second: 42} | '1984-10-28T12:31:42.645876123' |
      | date({year: 1984, month: 10, day: 11})                                                                  | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {date: otherDate, time: otherTime}                      | '1984-10-11T12:31:14.645876'    |
      | date({year: 1984, month: 10, day: 11})                                                                  | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {date: otherDate, time: otherTime, day: 28, second: 42} | '1984-10-28T12:31:42.645876'    |
      | date({year: 1984, month: 10, day: 11})                                                                  | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {date: otherDate, time: otherTime}                      | '1984-10-11T12:31:14.645'       |
      | date({year: 1984, month: 10, day: 11})                                                                  | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {date: otherDate, time: otherTime, day: 28, second: 42} | '1984-10-28T12:31:42.645'       |
      | date({year: 1984, month: 10, day: 11})                                                                  | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | {date: otherDate, time: otherTime}                      | '1984-10-11T12:00'              |
      | date({year: 1984, month: 10, day: 11})                                                                  | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | {date: otherDate, time: otherTime, day: 28, second: 42} | '1984-10-28T12:00:42'           |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {date: otherDate, time: otherTime}                      | '1984-03-07T12:31:14.645876123' |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {date: otherDate, time: otherTime, day: 28, second: 42} | '1984-03-28T12:31:42.645876123' |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {date: otherDate, time: otherTime}                      | '1984-03-07T12:31:14.645876'    |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {date: otherDate, time: otherTime, day: 28, second: 42} | '1984-03-28T12:31:42.645876'    |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {date: otherDate, time: otherTime}                      | '1984-03-07T12:31:14.645'       |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {date: otherDate, time: otherTime, day: 28, second: 42} | '1984-03-28T12:31:42.645'       |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | {date: otherDate, time: otherTime}                      | '1984-03-07T12:00'              |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | {date: otherDate, time: otherTime, day: 28, second: 42} | '1984-03-28T12:00:42'           |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {date: otherDate, time: otherTime}                      | '1984-10-11T12:31:14.645876123' |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {date: otherDate, time: otherTime, day: 28, second: 42} | '1984-10-28T12:31:42.645876123' |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {date: otherDate, time: otherTime}                      | '1984-10-11T12:31:14.645876'    |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {date: otherDate, time: otherTime, day: 28, second: 42} | '1984-10-28T12:31:42.645876'    |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {date: otherDate, time: otherTime}                      | '1984-10-11T12:31:14.645'       |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {date: otherDate, time: otherTime, day: 28, second: 42} | '1984-10-28T12:31:42.645'       |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | {date: otherDate, time: otherTime}                      | '1984-10-11T12:00'              |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | {date: otherDate, time: otherTime, day: 28, second: 42} | '1984-10-28T12:00:42'           |

  
  Scenario Outline: [7] Should select datetime into local date time
    Given any graph
    When executing query:
      """
      WITH <other> AS other
      RETURN localdatetime(<expression>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | other                                                                                                   | expression                             | result                    |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | other                                  | '1984-03-07T12:31:14.645' |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {datetime: other}                      | '1984-03-07T12:31:14.645' |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {datetime: other, day: 28, second: 42} | '1984-03-28T12:31:42.645' |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | other                                  | '1984-10-11T12:00'        |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | {datetime: other}                      | '1984-10-11T12:00'        |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | {datetime: other, day: 28, second: 42} | '1984-10-28T12:00:42'     |

  
  Scenario Outline: [8] Should select date into date time
    Given any graph
    When executing query:
      """
      WITH <other> AS other
      RETURN datetime(<expression>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | other                                                                                                   | expression                                                                             | result                                        |
      | date({year: 1984, month: 10, day: 11})                                                                  | {date: other, hour: 10, minute: 10, second: 10}                                        | '1984-10-11T10:10:10Z'                        |
      | date({year: 1984, month: 10, day: 11})                                                                  | {date: other, hour: 10, minute: 10, second: 10, timezone: '+05:00'}                    | '1984-10-11T10:10:10+05:00'                   |
      | date({year: 1984, month: 10, day: 11})                                                                  | {date: other, day: 28, hour: 10, minute: 10, second: 10}                               | '1984-10-28T10:10:10Z'                        |
      | date({year: 1984, month: 10, day: 11})                                                                  | {date: other, day: 28, hour: 10, minute: 10, second: 10, timezone: 'Pacific/Honolulu'} | '1984-10-28T10:10:10-10:00[Pacific/Honolulu]' |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {date: other, hour: 10, minute: 10, second: 10}                                        | '1984-03-07T10:10:10Z'                        |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {date: other, hour: 10, minute: 10, second: 10, timezone: '+05:00'}                    | '1984-03-07T10:10:10+05:00'                   |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {date: other, day: 28, hour: 10, minute: 10, second: 10}                               | '1984-03-28T10:10:10Z'                        |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {date: other, day: 28, hour: 10, minute: 10, second: 10, timezone: 'Pacific/Honolulu'} | '1984-03-28T10:10:10-10:00[Pacific/Honolulu]' |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | {date: other, hour: 10, minute: 10, second: 10}                                        | '1984-10-11T10:10:10Z'                        |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | {date: other, hour: 10, minute: 10, second: 10, timezone: '+05:00'}                    | '1984-10-11T10:10:10+05:00'                   |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | {date: other, day: 28, hour: 10, minute: 10, second: 10}                               | '1984-10-28T10:10:10Z'                        |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | {date: other, day: 28, hour: 10, minute: 10, second: 10, timezone: 'Pacific/Honolulu'} | '1984-10-28T10:10:10-10:00[Pacific/Honolulu]' |

  
  Scenario Outline: [9] Should select time into date time
    Given any graph
    When executing query:
      """
      WITH <other> AS other
      RETURN datetime(<expression>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | other                                                                                                   | expression                                                                              | result                                                  |
      | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {year: 1984, month: 10, day: 11, time: other}                                           | '1984-10-11T12:31:14.645876123Z'                        |
      | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {year: 1984, month: 10, day: 11, time: other, timezone: '+05:00'}                       | '1984-10-11T12:31:14.645876123+05:00'                   |
      | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {year: 1984, month: 10, day: 11, time: other, second: 42}                               | '1984-10-11T12:31:42.645876123Z'                        |
      | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {year: 1984, month: 10, day: 11, time: other, second: 42, timezone: 'Pacific/Honolulu'} | '1984-10-11T12:31:42.645876123-10:00[Pacific/Honolulu]' |
      | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {year: 1984, month: 10, day: 11, time: other}                                           | '1984-10-11T12:31:14.645876+01:00'                      |
      | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {year: 1984, month: 10, day: 11, time: other, timezone: '+05:00'}                       | '1984-10-11T16:31:14.645876+05:00'                      |
      | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {year: 1984, month: 10, day: 11, time: other, second: 42}                               | '1984-10-11T12:31:42.645876+01:00'                      |
      | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {year: 1984, month: 10, day: 11, time: other, second: 42, timezone: 'Pacific/Honolulu'} | '1984-10-11T01:31:42.645876-10:00[Pacific/Honolulu]'    |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {year: 1984, month: 10, day: 11, time: other}                                           | '1984-10-11T12:31:14.645Z'                              |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {year: 1984, month: 10, day: 11, time: other, timezone: '+05:00'}                       | '1984-10-11T12:31:14.645+05:00'                         |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {year: 1984, month: 10, day: 11, time: other, second: 42}                               | '1984-10-11T12:31:42.645Z'                              |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {year: 1984, month: 10, day: 11, time: other, second: 42, timezone: 'Pacific/Honolulu'} | '1984-10-11T12:31:42.645-10:00[Pacific/Honolulu]'       |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | {year: 1984, month: 10, day: 11, time: other}                                           | '1984-10-11T12:00+01:00[Europe/Stockholm]'              |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | {year: 1984, month: 10, day: 11, time: other, timezone: '+05:00'}                       | '1984-10-11T16:00+05:00'                                |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | {year: 1984, month: 10, day: 11, time: other, second: 42}                               | '1984-10-11T12:00:42+01:00[Europe/Stockholm]'           |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | {year: 1984, month: 10, day: 11, time: other, second: 42, timezone: 'Pacific/Honolulu'} | '1984-10-11T01:00:42-10:00[Pacific/Honolulu]'           |

  
  Scenario Outline: [10] Should select date and time into date time
    Given any graph
    When executing query:
      """
      WITH <otherDate> AS otherDate, <otherTime> AS otherTime
      RETURN datetime(<expression>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | otherDate                                                                                               | otherTime                                                                                               | expression                                                                            | result                                                  |
      | date({year: 1984, month: 10, day: 11})                                                                  | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {date: otherDate, time: otherTime}                                                    | '1984-10-11T12:31:14.645876123Z'                        |
      | date({year: 1984, month: 10, day: 11})                                                                  | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {date: otherDate, time: otherTime, timezone: '+05:00'}                                | '1984-10-11T12:31:14.645876123+05:00'                   |
      | date({year: 1984, month: 10, day: 11})                                                                  | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {date: otherDate, time: otherTime, day: 28, second: 42}                               | '1984-10-28T12:31:42.645876123Z'                        |
      | date({year: 1984, month: 10, day: 11})                                                                  | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {date: otherDate, time: otherTime, day: 28, second: 42, timezone: 'Pacific/Honolulu'} | '1984-10-28T12:31:42.645876123-10:00[Pacific/Honolulu]' |
      | date({year: 1984, month: 10, day: 11})                                                                  | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {date: otherDate, time: otherTime}                                                    | '1984-10-11T12:31:14.645876+01:00'                      |
      | date({year: 1984, month: 10, day: 11})                                                                  | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {date: otherDate, time: otherTime, timezone: '+05:00'}                                | '1984-10-11T16:31:14.645876+05:00'                      |
      | date({year: 1984, month: 10, day: 11})                                                                  | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {date: otherDate, time: otherTime, day: 28, second: 42}                               | '1984-10-28T12:31:42.645876+01:00'                      |
      | date({year: 1984, month: 10, day: 11})                                                                  | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {date: otherDate, time: otherTime, day: 28, second: 42, timezone: 'Pacific/Honolulu'} | '1984-10-28T01:31:42.645876-10:00[Pacific/Honolulu]'    |
      | date({year: 1984, month: 10, day: 11})                                                                  | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {date: otherDate, time: otherTime}                                                    | '1984-10-11T12:31:14.645Z'                              |
      | date({year: 1984, month: 10, day: 11})                                                                  | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {date: otherDate, time: otherTime, timezone: '+05:00'}                                | '1984-10-11T12:31:14.645+05:00'                         |
      | date({year: 1984, month: 10, day: 11})                                                                  | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {date: otherDate, time: otherTime, day: 28, second: 42}                               | '1984-10-28T12:31:42.645Z'                              |
      | date({year: 1984, month: 10, day: 11})                                                                  | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {date: otherDate, time: otherTime, day: 28, second: 42, timezone: 'Pacific/Honolulu'} | '1984-10-28T12:31:42.645-10:00[Pacific/Honolulu]'       |
      | date({year: 1984, month: 10, day: 11})                                                                  | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | {date: otherDate, time: otherTime}                                                    | '1984-10-11T12:00+01:00[Europe/Stockholm]'              |
      | date({year: 1984, month: 10, day: 11})                                                                  | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | {date: otherDate, time: otherTime, timezone: '+05:00'}                                | '1984-10-11T16:00+05:00'                                |
      | date({year: 1984, month: 10, day: 11})                                                                  | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | {date: otherDate, time: otherTime, day: 28, second: 42}                               | '1984-10-28T12:00:42+01:00[Europe/Stockholm]'           |
      | date({year: 1984, month: 10, day: 11})                                                                  | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | {date: otherDate, time: otherTime, day: 28, second: 42, timezone: 'Pacific/Honolulu'} | '1984-10-28T01:00:42-10:00[Pacific/Honolulu]'           |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {date: otherDate, time: otherTime}                                                    | '1984-03-07T12:31:14.645876123Z'                        |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {date: otherDate, time: otherTime, timezone: '+05:00'}                                | '1984-03-07T12:31:14.645876123+05:00'                   |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {date: otherDate, time: otherTime, day: 28, second: 42}                               | '1984-03-28T12:31:42.645876123Z'                        |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {date: otherDate, time: otherTime, day: 28, second: 42, timezone: 'Pacific/Honolulu'} | '1984-03-28T12:31:42.645876123-10:00[Pacific/Honolulu]' |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {date: otherDate, time: otherTime}                                                    | '1984-03-07T12:31:14.645876+01:00'                      |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {date: otherDate, time: otherTime, timezone: '+05:00'}                                | '1984-03-07T16:31:14.645876+05:00'                      |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {date: otherDate, time: otherTime, day: 28, second: 42}                               | '1984-03-28T12:31:42.645876+01:00'                      |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {date: otherDate, time: otherTime, day: 28, second: 42, timezone: 'Pacific/Honolulu'} | '1984-03-28T01:31:42.645876-10:00[Pacific/Honolulu]'    |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {date: otherDate, time: otherTime}                                                    | '1984-03-07T12:31:14.645Z'                              |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {date: otherDate, time: otherTime, timezone: '+05:00'}                                | '1984-03-07T12:31:14.645+05:00'                         |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {date: otherDate, time: otherTime, day: 28, second: 42}                               | '1984-03-28T12:31:42.645Z'                              |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {date: otherDate, time: otherTime, day: 28, second: 42, timezone: 'Pacific/Honolulu'} | '1984-03-28T12:31:42.645-10:00[Pacific/Honolulu]'       |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | {date: otherDate, time: otherTime}                                                    | '1984-03-07T12:00+01:00[Europe/Stockholm]'              |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | {date: otherDate, time: otherTime, timezone: '+05:00'}                                | '1984-03-07T16:00+05:00'                                |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | {date: otherDate, time: otherTime, day: 28, second: 42}                               | '1984-03-28T12:00:42+02:00[Europe/Stockholm]'           |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | {date: otherDate, time: otherTime, day: 28, second: 42, timezone: 'Pacific/Honolulu'} | '1984-03-28T00:00:42-10:00[Pacific/Honolulu]'           |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {date: otherDate, time: otherTime}                                                    | '1984-10-11T12:31:14.645876123Z'                        |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {date: otherDate, time: otherTime, timezone: '+05:00'}                                | '1984-10-11T12:31:14.645876123+05:00'                   |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {date: otherDate, time: otherTime, day: 28, second: 42}                               | '1984-10-28T12:31:42.645876123Z'                        |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                    | {date: otherDate, time: otherTime, day: 28, second: 42, timezone: 'Pacific/Honolulu'} | '1984-10-28T12:31:42.645876123-10:00[Pacific/Honolulu]' |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {date: otherDate, time: otherTime}                                                    | '1984-10-11T12:31:14.645876+01:00'                      |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {date: otherDate, time: otherTime, timezone: '+05:00'}                                | '1984-10-11T16:31:14.645876+05:00'                      |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {date: otherDate, time: otherTime, day: 28, second: 42}                               | '1984-10-28T12:31:42.645876+01:00'                      |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'})                       | {date: otherDate, time: otherTime, day: 28, second: 42, timezone: 'Pacific/Honolulu'} | '1984-10-28T01:31:42.645876-10:00[Pacific/Honolulu]'    |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {date: otherDate, time: otherTime}                                                    | '1984-10-11T12:31:14.645Z'                              |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {date: otherDate, time: otherTime, timezone: '+05:00'}                                | '1984-10-11T12:31:14.645+05:00'                         |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {date: otherDate, time: otherTime, day: 28, second: 42}                               | '1984-10-28T12:31:42.645Z'                              |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {date: otherDate, time: otherTime, day: 28, second: 42, timezone: 'Pacific/Honolulu'} | '1984-10-28T12:31:42.645-10:00[Pacific/Honolulu]'       |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | {date: otherDate, time: otherTime}                                                    | '1984-10-11T12:00+01:00[Europe/Stockholm]'              |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | {date: otherDate, time: otherTime, timezone: '+05:00'}                                | '1984-10-11T16:00+05:00'                                |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | {date: otherDate, time: otherTime, day: 28, second: 42}                               | '1984-10-28T12:00:42+01:00[Europe/Stockholm]'           |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})                                | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | {date: otherDate, time: otherTime, day: 28, second: 42, timezone: 'Pacific/Honolulu'} | '1984-10-28T01:00:42-10:00[Pacific/Honolulu]'           |

  
  Scenario Outline: [11] Should datetime into date time
    Given any graph
    When executing query:
      """
      WITH <other> AS other
      RETURN datetime(<expression>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | other                                                                                                   | expression                                                           | result                                            |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | other                                                                | '1984-03-07T12:31:14.645Z'                        |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {datetime: other}                                                    | '1984-03-07T12:31:14.645Z'                        |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {datetime: other, timezone: '+05:00'}                                | '1984-03-07T12:31:14.645+05:00'                   |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {datetime: other, day: 28, second: 42}                               | '1984-03-28T12:31:42.645Z'                        |
      | localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) | {datetime: other, day: 28, second: 42, timezone: 'Pacific/Honolulu'} | '1984-03-28T12:31:42.645-10:00[Pacific/Honolulu]' |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | other                                                                | '1984-10-11T12:00+01:00[Europe/Stockholm]'        |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | {datetime: other}                                                    | '1984-10-11T12:00+01:00[Europe/Stockholm]'        |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | {datetime: other, timezone: '+05:00'}                                | '1984-10-11T16:00+05:00'                          |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | {datetime: other, day: 28, second: 42}                               | '1984-10-28T12:00:42+01:00[Europe/Stockholm]'     |
      | datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})                      | {datetime: other, day: 28, second: 42, timezone: 'Pacific/Honolulu'} | '1984-10-28T01:00:42-10:00[Pacific/Honolulu]'     |
