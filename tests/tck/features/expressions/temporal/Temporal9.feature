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

Feature: Temporal9 - Truncate Temporal Values

  
  Scenario Outline: [1] Should truncate date
    Given any graph
    When executing query:
      """
      RETURN date.truncate(<unit>, <other>, <map>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | unit          | other                                                                                                                   | map                            | result                                     |
      | 'millennium'  | date({year: 2017, month: 10, day: 11})                                                                                  | {day: 2}                       | '2000-01-02'                               |
      | 'millennium'  | date({year: 2017, month: 10, day: 11})                                                                                  | {}                             | '2000-01-01'                               |
      | 'millennium'  | datetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {day: 2}                       | '2000-01-02'                               |
      | 'millennium'  | datetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '2000-01-01'                               |
      | 'millennium'  | localdatetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {day: 2}                       | '2000-01-02'                               |
      | 'millennium'  | localdatetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '2000-01-01'                               |
      | 'century'     | date({year: 1984, month: 10, day: 11})                                                                                  | {day: 2}                       | '1900-01-02'                               |
      | 'century'     | date({year: 1984, month: 10, day: 11})                                                                                  | {}                             | '1900-01-01'                               |
      | 'century'     | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {day: 2}                       | '1900-01-02'                               |
      | 'century'     | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1900-01-01'                               |
      | 'century'     | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {day: 2}                       | '1900-01-02'                               |
      | 'century'     | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1900-01-01'                               |
      | 'decade'      | date({year: 1984, month: 10, day: 11})                                                                                  | {day: 2}                       | '1980-01-02'                               |
      | 'decade'      | date({year: 1984, month: 10, day: 11})                                                                                  | {}                             | '1980-01-01'                               |
      | 'decade'      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {day: 2}                       | '1980-01-02'                               |
      | 'decade'      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1980-01-01'                               |
      | 'decade'      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {day: 2}                       | '1980-01-02'                               |
      | 'decade'      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1980-01-01'                               |
      | 'year'        | date({year: 1984, month: 10, day: 11})                                                                                  | {day: 2}                       | '1984-01-02'                               |
      | 'year'        | date({year: 1984, month: 10, day: 11})                                                                                  | {}                             | '1984-01-01'                               |
      | 'year'        | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {day: 2}                       | '1984-01-02'                               |
      | 'year'        | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1984-01-01'                               |
      | 'year'        | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {day: 2}                       | '1984-01-02'                               |
      | 'year'        | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-01-01'                               |
      | 'weekYear'    | date({year: 1984, month: 2, day: 1})                                                                                    | {day: 5}                       | '1984-01-05'                               |
      | 'weekYear'    | date({year: 1984, month: 2, day: 1})                                                                                    | {}                             | '1984-01-02'                               |
      | 'weekYear'    | datetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})   | {day: 5}                       | '1983-01-05'                               |
      | 'weekYear'    | datetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})   | {}                             | '1983-01-03'                               |
      | 'weekYear'    | localdatetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                  | {day: 5}                       | '1983-01-05'                               |
      | 'weekYear'    | localdatetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                  | {}                             | '1983-01-03'                               |
      | 'quarter'     | date({year: 1984, month: 11, day: 11})                                                                                  | {day: 2}                       | '1984-10-02'                               |
      | 'quarter'     | date({year: 1984, month: 11, day: 11})                                                                                  | {}                             | '1984-10-01'                               |
      | 'quarter'     | datetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {day: 2}                       | '1984-10-02'                               |
      | 'quarter'     | datetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1984-10-01'                               |
      | 'quarter'     | localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {day: 2}                       | '1984-10-02'                               |
      | 'quarter'     | localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-10-01'                               |
      | 'month'       | date({year: 1984, month: 10, day: 11})                                                                                  | {day: 2}                       | '1984-10-02'                               |
      | 'month'       | date({year: 1984, month: 10, day: 11})                                                                                  | {}                             | '1984-10-01'                               |
      | 'month'       | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {day: 2}                       | '1984-10-02'                               |
      | 'month'       | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1984-10-01'                               |
      | 'month'       | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {day: 2}                       | '1984-10-02'                               |
      | 'month'       | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-10-01'                               |
      | 'week'        | date({year: 1984, month: 10, day: 11})                                                                                  | {dayOfWeek: 2}                 | '1984-10-09'                               |
      | 'week'        | date({year: 1984, month: 10, day: 11})                                                                                  | {}                             | '1984-10-08'                               |
      | 'week'        | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {dayOfWeek: 2}                 | '1984-10-09'                               |
      | 'week'        | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1984-10-08'                               |
      | 'week'        | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {dayOfWeek: 2}                 | '1984-10-09'                               |
      | 'week'        | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-10-08'                               |
      | 'day'         | date({year: 1984, month: 10, day: 11})                                                                                  | {}                             | '1984-10-11'                               |
      | 'day'         | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1984-10-11'                               |
      | 'day'         | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-10-11'                               |

  
  Scenario Outline: [2] Should truncate datetime
    Given any graph
    When executing query:
      """
      RETURN datetime.truncate(<unit>, <other>, <map>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | unit          | other                                                                                                                   | map                            | result                                     |
      | 'millennium'  | date({year: 2017, month: 10, day: 11})                                                                                  | {day: 2}                       | '2000-01-02T00:00Z'                        |
      | 'millennium'  | date({year: 2017, month: 10, day: 11})                                                                                  | {timezone: 'Europe/Stockholm'} | '2000-01-01T00:00+01:00[Europe/Stockholm]' |
      | 'millennium'  | date({year: 2017, month: 10, day: 11})                                                                                  | {}                             | '2000-01-01T00:00Z'                        |
      | 'millennium'  | datetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {day: 2}                       | '2000-01-02T00:00+01:00'                   |
      | 'millennium'  | datetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '2000-01-01T00:00+01:00'                   |
      | 'millennium'  | datetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}) | {timezone: 'Europe/Stockholm'} | '2000-01-01T00:00+01:00[Europe/Stockholm]' |
      | 'millennium'  | localdatetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {day: 2}                       | '2000-01-02T00:00Z'                        |
      | 'millennium'  | localdatetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {timezone: 'Europe/Stockholm'} | '2000-01-01T00:00+01:00[Europe/Stockholm]' |
      | 'millennium'  | localdatetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '2000-01-01T00:00Z'                        |
      | 'century'     | date({year: 1984, month: 10, day: 11})                                                                                  | {day: 2}                       | '1900-01-02T00:00Z'                        |
      | 'century'     | date({year: 1984, month: 10, day: 11})                                                                                  | {}                             | '1900-01-01T00:00Z'                        |
      | 'century'     | date({year: 2017, month: 10, day: 11})                                                                                  | {timezone: 'Europe/Stockholm'} | '2000-01-01T00:00+01:00[Europe/Stockholm]' |
      | 'century'     | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {day: 2}                       | '1900-01-02T00:00+01:00'                   |
      | 'century'     | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1900-01-01T00:00+01:00'                   |
      | 'century'     | datetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}) | {timezone: 'Europe/Stockholm'} | '2000-01-01T00:00+01:00[Europe/Stockholm]' |
      | 'century'     | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {day: 2}                       | '1900-01-02T00:00Z'                        |
      | 'century'     | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1900-01-01T00:00Z'                        |
      | 'century'     | localdatetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {timezone: 'Europe/Stockholm'} | '2000-01-01T00:00+01:00[Europe/Stockholm]' |
      | 'decade'      | date({year: 1984, month: 10, day: 11})                                                                                  | {day: 2}                       | '1980-01-02T00:00Z'                        |
      | 'decade'      | date({year: 1984, month: 10, day: 11})                                                                                  | {timezone: 'Europe/Stockholm'} | '1980-01-01T00:00+01:00[Europe/Stockholm]' |
      | 'decade'      | date({year: 1984, month: 10, day: 11})                                                                                  | {}                             | '1980-01-01T00:00Z'                        |
      | 'decade'      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {day: 2}                       | '1980-01-02T00:00+01:00'                   |
      | 'decade'      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1980-01-01T00:00+01:00'                   |
      | 'decade'      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}) | {timezone: 'Europe/Stockholm'} | '1980-01-01T00:00+01:00[Europe/Stockholm]' |
      | 'decade'      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {day: 2}                       | '1980-01-02T00:00Z'                        |
      | 'decade'      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {timezone: 'Europe/Stockholm'} | '1980-01-01T00:00+01:00[Europe/Stockholm]' |
      | 'decade'      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1980-01-01T00:00Z'                        |
      | 'year'        | date({year: 1984, month: 10, day: 11})                                                                                  | {day: 2}                       | '1984-01-02T00:00Z'                        |
      | 'year'        | date({year: 1984, month: 10, day: 11})                                                                                  | {timezone: 'Europe/Stockholm'} | '1984-01-01T00:00+01:00[Europe/Stockholm]' |
      | 'year'        | date({year: 1984, month: 10, day: 11})                                                                                  | {}                             | '1984-01-01T00:00Z'                        |
      | 'year'        | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {day: 2}                       | '1984-01-02T00:00+01:00'                   |
      | 'year'        | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1984-01-01T00:00+01:00'                   |
      | 'year'        | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}) | {timezone: 'Europe/Stockholm'} | '1984-01-01T00:00+01:00[Europe/Stockholm]' |
      | 'year'        | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {day: 2}                       | '1984-01-02T00:00Z'                        |
      | 'year'        | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {timezone: 'Europe/Stockholm'} | '1984-01-01T00:00+01:00[Europe/Stockholm]' |
      | 'year'        | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-01-01T00:00Z'                        |
      | 'weekYear'    | date({year: 1984, month: 2, day: 1})                                                                                    | {day: 5}                       | '1984-01-05T00:00Z'                        |
      | 'weekYear'    | date({year: 1984, month: 2, day: 1})                                                                                    | {timezone: 'Europe/Stockholm'} | '1984-01-02T00:00+01:00[Europe/Stockholm]' |
      | 'weekYear'    | date({year: 1984, month: 2, day: 1})                                                                                    | {}                             | '1984-01-02T00:00Z'                        |
      | 'weekYear'    | datetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})   | {day: 5}                       | '1983-01-05T00:00+01:00'                   |
      | 'weekYear'    | datetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})   | {}                             | '1983-01-03T00:00+01:00'                   |
      | 'weekYear'    | datetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'})   | {timezone: 'Europe/Stockholm'} | '1983-01-03T00:00+01:00[Europe/Stockholm]' |
      | 'weekYear'    | localdatetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                  | {day: 5}                       | '1983-01-05T00:00Z'                        |
      | 'weekYear'    | localdatetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                  | {timezone: 'Europe/Stockholm'} | '1983-01-03T00:00+01:00[Europe/Stockholm]' |
      | 'weekYear'    | localdatetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                  | {}                             | '1983-01-03T00:00Z'                        |
      | 'quarter'     | date({year: 1984, month: 11, day: 11})                                                                                  | {day: 2}                       | '1984-10-02T00:00Z'                        |
      | 'quarter'     | date({year: 1984, month: 11, day: 11})                                                                                  | {timezone: 'Europe/Stockholm'} | '1984-10-01T00:00+01:00[Europe/Stockholm]' |
      | 'quarter'     | date({year: 1984, month: 11, day: 11})                                                                                  | {}                             | '1984-10-01T00:00Z'                        |
      | 'quarter'     | datetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {day: 2}                       | '1984-10-02T00:00+01:00'                   |
      | 'quarter'     | datetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1984-10-01T00:00+01:00'                   |
      | 'quarter'     | datetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}) | {timezone: 'Europe/Stockholm'} | '1984-10-01T00:00+01:00[Europe/Stockholm]' |
      | 'quarter'     | localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {day: 2}                       | '1984-10-02T00:00Z'                        |
      | 'quarter'     | localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {timezone: 'Europe/Stockholm'} | '1984-10-01T00:00+01:00[Europe/Stockholm]' |
      | 'quarter'     | localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-10-01T00:00Z'                        |
      | 'month'       | date({year: 1984, month: 10, day: 11})                                                                                  | {day: 2}                       | '1984-10-02T00:00Z'                        |
      | 'month'       | date({year: 1984, month: 10, day: 11})                                                                                  | {timezone: 'Europe/Stockholm'} | '1984-10-01T00:00+01:00[Europe/Stockholm]' |
      | 'month'       | date({year: 1984, month: 10, day: 11})                                                                                  | {}                             | '1984-10-01T00:00Z'                        |
      | 'month'       | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {day: 2}                       | '1984-10-02T00:00+01:00'                   |
      | 'month'       | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1984-10-01T00:00+01:00'                   |
      | 'month'       | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}) | {timezone: 'Europe/Stockholm'} | '1984-10-01T00:00+01:00[Europe/Stockholm]' |
      | 'month'       | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {day: 2}                       | '1984-10-02T00:00Z'                        |
      | 'month'       | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {timezone: 'Europe/Stockholm'} | '1984-10-01T00:00+01:00[Europe/Stockholm]' |
      | 'month'       | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-10-01T00:00Z'                        |
      | 'week'        | date({year: 1984, month: 10, day: 11})                                                                                  | {dayOfWeek: 2}                 | '1984-10-09T00:00Z'                        |
      | 'week'        | date({year: 1984, month: 10, day: 11})                                                                                  | {timezone: 'Europe/Stockholm'} | '1984-10-08T00:00+01:00[Europe/Stockholm]' |
      | 'week'        | date({year: 1984, month: 10, day: 11})                                                                                  | {}                             | '1984-10-08T00:00Z'                        |
      | 'week'        | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {dayOfWeek: 2}                 | '1984-10-09T00:00+01:00'                   |
      | 'week'        | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1984-10-08T00:00+01:00'                   |
      | 'week'        | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}) | {timezone: 'Europe/Stockholm'} | '1984-10-08T00:00+01:00[Europe/Stockholm]' |
      | 'week'        | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {dayOfWeek: 2}                 | '1984-10-09T00:00Z'                        |
      | 'week'        | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {timezone: 'Europe/Stockholm'} | '1984-10-08T00:00+01:00[Europe/Stockholm]' |
      | 'week'        | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-10-08T00:00Z'                        |
      | 'day'         | date({year: 1984, month: 10, day: 11})                                                                                  | {nanosecond: 2}                | '1984-10-11T00:00:00.000000002Z'           |
      | 'day'         | date({year: 1984, month: 10, day: 11})                                                                                  | {timezone: 'Europe/Stockholm'} | '1984-10-11T00:00+01:00[Europe/Stockholm]' |
      | 'day'         | date({year: 1984, month: 10, day: 11})                                                                                  | {}                             | '1984-10-11T00:00Z'                        |
      | 'day'         | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {nanosecond: 2}                | '1984-10-11T00:00:00.000000002+01:00'      |
      | 'day'         | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1984-10-11T00:00+01:00'                   |
      | 'day'         | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}) | {timezone: 'Europe/Stockholm'} | '1984-10-11T00:00+01:00[Europe/Stockholm]' |
      | 'day'         | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {nanosecond: 2}                | '1984-10-11T00:00:00.000000002Z'           |
      | 'day'         | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {timezone: 'Europe/Stockholm'} | '1984-10-11T00:00+01:00[Europe/Stockholm]' |
      | 'day'         | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-10-11T00:00Z'                        |
      | 'hour'        | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}) | {nanosecond: 2}                | '1984-10-11T12:00:00.000000002-01:00'      |
      | 'hour'        | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}) | {timezone: 'Europe/Stockholm'} | '1984-10-11T12:00+01:00[Europe/Stockholm]' |
      | 'hour'        | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}) | {}                             | '1984-10-11T12:00-01:00'                   |
      | 'hour'        | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {nanosecond: 2}                | '1984-10-11T12:00:00.000000002Z'           |
      | 'hour'        | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {timezone: 'Europe/Stockholm'} | '1984-10-11T12:00+01:00[Europe/Stockholm]' |
      | 'hour'        | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-10-11T12:00Z'                        |
      | 'minute'      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}) | {nanosecond: 2}                | '1984-10-11T12:31:00.000000002-01:00'      |
      | 'minute'      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}) | {timezone: 'Europe/Stockholm'} | '1984-10-11T12:31+01:00[Europe/Stockholm]' |
      | 'minute'      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}) | {}                             | '1984-10-11T12:31-01:00'                   |
      | 'minute'      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {nanosecond: 2}                | '1984-10-11T12:31:00.000000002Z'           |
      | 'minute'      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {timezone: 'Europe/Stockholm'} | '1984-10-11T12:31+01:00[Europe/Stockholm]' |
      | 'minute'      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-10-11T12:31Z'                        |
      | 'second'      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {nanosecond: 2}                | '1984-10-11T12:31:14.000000002+01:00'      |
      | 'second'      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1984-10-11T12:31:14+01:00'                |
      | 'second'      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {nanosecond: 2}                | '1984-10-11T12:31:14.000000002Z'           |
      | 'second'      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-10-11T12:31:14Z'                     |
      | 'millisecond' | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {nanosecond: 2}                | '1984-10-11T12:31:14.645000002+01:00'      |
      | 'millisecond' | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1984-10-11T12:31:14.645+01:00'            |
      | 'millisecond' | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {nanosecond: 2}                | '1984-10-11T12:31:14.645000002Z'           |
      | 'millisecond' | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-10-11T12:31:14.645Z'                 |
      | 'microsecond' | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {nanosecond: 2}                | '1984-10-11T12:31:14.645876002+01:00'      |
      | 'microsecond' | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1984-10-11T12:31:14.645876+01:00'         |
      | 'microsecond' | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {nanosecond: 2}                | '1984-10-11T12:31:14.645876002Z'           |
      | 'microsecond' | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-10-11T12:31:14.645876Z'              |

  
  Scenario Outline: [3] Should truncate localdatetime
    Given any graph
    When executing query:
      """
      RETURN localdatetime.truncate(<unit>, <other>, <map>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | unit          | other                                                                                                                   | map                            | result                                     |
      | 'millennium'  | date({year: 2017, month: 10, day: 11})                                                                                  | {day: 2}                       | '2000-01-02T00:00'                         |
      | 'millennium'  | date({year: 2017, month: 10, day: 11})                                                                                  | {}                             | '2000-01-01T00:00'                         |
      | 'millennium'  | datetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {day: 2}                       | '2000-01-02T00:00'                         |
      | 'millennium'  | datetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '2000-01-01T00:00'                         |
      | 'millennium'  | localdatetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {day: 2}                       | '2000-01-02T00:00'                         |
      | 'millennium'  | localdatetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '2000-01-01T00:00'                         |
      | 'century'     | date({year: 1984, month: 10, day: 11})                                                                                  | {day: 2}                       | '1900-01-02T00:00'                         |
      | 'century'     | date({year: 1984, month: 10, day: 11})                                                                                  | {}                             | '1900-01-01T00:00'                         |
      | 'century'     | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {day: 2}                       | '1900-01-02T00:00'                         |
      | 'century'     | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1900-01-01T00:00'                         |
      | 'century'     | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {day: 2}                       | '1900-01-02T00:00'                         |
      | 'century'     | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1900-01-01T00:00'                         |
      | 'decade'      | date({year: 1984, month: 10, day: 11})                                                                                  | {day: 2}                       | '1980-01-02T00:00'                         |
      | 'decade'      | date({year: 1984, month: 10, day: 11})                                                                                  | {}                             | '1980-01-01T00:00'                         |
      | 'decade'      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {day: 2}                       | '1980-01-02T00:00'                         |
      | 'decade'      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1980-01-01T00:00'                         |
      | 'decade'      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {day: 2}                       | '1980-01-02T00:00'                         |
      | 'decade'      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1980-01-01T00:00'                         |
      | 'year'        | date({year: 1984, month: 10, day: 11})                                                                                  | {day: 2}                       | '1984-01-02T00:00'                         |
      | 'year'        | date({year: 1984, month: 10, day: 11})                                                                                  | {}                             | '1984-01-01T00:00'                         |
      | 'year'        | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {day: 2}                       | '1984-01-02T00:00'                         |
      | 'year'        | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1984-01-01T00:00'                         |
      | 'year'        | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {day: 2}                       | '1984-01-02T00:00'                         |
      | 'year'        | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-01-01T00:00'                         |
      | 'weekYear'    | date({year: 1984, month: 2, day: 1})                                                                                    | {day: 5}                       | '1984-01-05T00:00'                         |
      | 'weekYear'    | date({year: 1984, month: 2, day: 1})                                                                                    | {}                             | '1984-01-02T00:00'                         |
      | 'weekYear'    | datetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})   | {day: 5}                       | '1983-01-05T00:00'                         |
      | 'weekYear'    | datetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})   | {}                             | '1983-01-03T00:00'                         |
      | 'weekYear'    | localdatetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                  | {day: 5}                       | '1983-01-05T00:00'                         |
      | 'weekYear'    | localdatetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                  | {}                             | '1983-01-03T00:00'                         |
      | 'quarter'     | date({year: 1984, month: 11, day: 11})                                                                                  | {day: 2}                       | '1984-10-02T00:00'                         |
      | 'quarter'     | date({year: 1984, month: 11, day: 11})                                                                                  | {}                             | '1984-10-01T00:00'                         |
      | 'quarter'     | datetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {day: 2}                       | '1984-10-02T00:00'                         |
      | 'quarter'     | datetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1984-10-01T00:00'                         |
      | 'quarter'     | localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {day: 2}                       | '1984-10-02T00:00'                         |
      | 'quarter'     | localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-10-01T00:00'                         |
      | 'month'       | date({year: 1984, month: 10, day: 11})                                                                                  | {day: 2}                       | '1984-10-02T00:00'                         |
      | 'month'       | date({year: 1984, month: 10, day: 11})                                                                                  | {}                             | '1984-10-01T00:00'                         |
      | 'month'       | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {day: 2}                       | '1984-10-02T00:00'                         |
      | 'month'       | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1984-10-01T00:00'                         |
      | 'month'       | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {day: 2}                       | '1984-10-02T00:00'                         |
      | 'month'       | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-10-01T00:00'                         |
      | 'week'        | date({year: 1984, month: 10, day: 11})                                                                                  | {dayOfWeek: 2}                 | '1984-10-09T00:00'                         |
      | 'week'        | date({year: 1984, month: 10, day: 11})                                                                                  | {}                             | '1984-10-08T00:00'                         |
      | 'week'        | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {dayOfWeek: 2}                 | '1984-10-09T00:00'                         |
      | 'week'        | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1984-10-08T00:00'                         |
      | 'week'        | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {dayOfWeek: 2}                 | '1984-10-09T00:00'                         |
      | 'week'        | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-10-08T00:00'                         |
      | 'day'         | date({year: 1984, month: 10, day: 11})                                                                                  | {nanosecond: 2}                | '1984-10-11T00:00:00.000000002'            |
      | 'day'         | date({year: 1984, month: 10, day: 11})                                                                                  | {}                             | '1984-10-11T00:00'                         |
      | 'day'         | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {nanosecond: 2}                | '1984-10-11T00:00:00.000000002'            |
      | 'day'         | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1984-10-11T00:00'                         |
      | 'day'         | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {nanosecond: 2}                | '1984-10-11T00:00:00.000000002'            |
      | 'day'         | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-10-11T00:00'                         |
      | 'hour'        | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {nanosecond: 2}                | '1984-10-11T12:00:00.000000002'            |
      | 'hour'        | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1984-10-11T12:00'                         |
      | 'hour'        | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {nanosecond: 2}                | '1984-10-11T12:00:00.000000002'            |
      | 'hour'        | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-10-11T12:00'                         |
      | 'minute'      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {nanosecond: 2}                | '1984-10-11T12:31:00.000000002'            |
      | 'minute'      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1984-10-11T12:31'                         |
      | 'minute'      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {nanosecond: 2}                | '1984-10-11T12:31:00.000000002'            |
      | 'minute'      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-10-11T12:31'                         |
      | 'second'      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {nanosecond: 2}                | '1984-10-11T12:31:14.000000002'            |
      | 'second'      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1984-10-11T12:31:14'                      |
      | 'second'      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {nanosecond: 2}                | '1984-10-11T12:31:14.000000002'            |
      | 'second'      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-10-11T12:31:14'                      |
      | 'millisecond' | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {nanosecond: 2}                | '1984-10-11T12:31:14.645000002'            |
      | 'millisecond' | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1984-10-11T12:31:14.645'                  |
      | 'millisecond' | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {nanosecond: 2}                | '1984-10-11T12:31:14.645000002'            |
      | 'millisecond' | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-10-11T12:31:14.645'                  |
      | 'microsecond' | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {nanosecond: 2}                | '1984-10-11T12:31:14.645876002'            |
      | 'microsecond' | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '1984-10-11T12:31:14.645876'               |
      | 'microsecond' | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {nanosecond: 2}                | '1984-10-11T12:31:14.645876002'            |
      | 'microsecond' | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '1984-10-11T12:31:14.645876'               |

  
  Scenario Outline: [4] Should truncate localtime
    Given any graph
    When executing query:
      """
      RETURN localtime.truncate(<unit>, <other>, <map>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | unit          | other                                                                                                                   | map                            | result                                     |
      | 'day'         | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {nanosecond: 2}                | '00:00:00.000000002'                       |
      | 'day'         | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '00:00'                                    |
      | 'day'         | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {nanosecond: 2}                | '00:00:00.000000002'                       |
      | 'day'         | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '00:00'                                    |
      | 'hour'        | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {nanosecond: 2}                | '12:00:00.000000002'                       |
      | 'hour'        | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '12:00'                                    |
      | 'hour'        | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {nanosecond: 2}                | '12:00:00.000000002'                       |
      | 'hour'        | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '12:00'                                    |
      | 'hour'        | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                                    | {nanosecond: 2}                | '12:00:00.000000002'                       |
      | 'hour'        | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                                    | {}                             | '12:00'                                    |
      | 'hour'        | time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})                                     | {nanosecond: 2}                | '12:00:00.000000002'                       |
      | 'hour'        | time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})                                     | {}                             | '12:00'                                    |
      | 'minute'      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {nanosecond: 2}                | '12:31:00.000000002'                       |
      | 'minute'      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '12:31'                                    |
      | 'minute'      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {nanosecond: 2}                | '12:31:00.000000002'                       |
      | 'minute'      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '12:31'                                    |
      | 'minute'      | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                                    | {nanosecond: 2}                | '12:31:00.000000002'                       |
      | 'minute'      | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                                    | {}                             | '12:31'                                    |
      | 'minute'      | time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})                                     | {nanosecond: 2}                | '12:31:00.000000002'                       |
      | 'minute'      | time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})                                     | {}                             | '12:31'                                    |
      | 'second'      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {nanosecond: 2}                | '12:31:14.000000002'                       |
      | 'second'      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '12:31:14'                                 |
      | 'second'      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {nanosecond: 2}                | '12:31:14.000000002'                       |
      | 'second'      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '12:31:14'                                 |
      | 'second'      | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                                    | {nanosecond: 2}                | '12:31:14.000000002'                       |
      | 'second'      | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                                    | {}                             | '12:31:14'                                 |
      | 'second'      | time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})                                     | {nanosecond: 2}                | '12:31:14.000000002'                       |
      | 'second'      | time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})                                     | {}                             | '12:31:14'                                 |
      | 'millisecond' | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {nanosecond: 2}                | '12:31:14.645000002'                       |
      | 'millisecond' | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '12:31:14.645'                             |
      | 'millisecond' | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {nanosecond: 2}                | '12:31:14.645000002'                       |
      | 'millisecond' | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '12:31:14.645'                             |
      | 'millisecond' | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                                    | {nanosecond: 2}                | '12:31:14.645000002'                       |
      | 'millisecond' | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                                    | {}                             | '12:31:14.645'                             |
      | 'millisecond' | time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})                                     | {nanosecond: 2}                | '12:31:14.645000002'                       |
      | 'millisecond' | time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})                                     | {}                             | '12:31:14.645'                             |
      | 'microsecond' | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {nanosecond: 2}                | '12:31:14.645876002'                       |
      | 'microsecond' | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '12:31:14.645876'                          |
      | 'microsecond' | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {nanosecond: 2}                | '12:31:14.645876002'                       |
      | 'microsecond' | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '12:31:14.645876'                          |
      | 'microsecond' | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                                    | {nanosecond: 2}                | '12:31:14.645876002'                       |
      | 'microsecond' | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                                    | {}                             | '12:31:14.645876'                          |
      | 'microsecond' | time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})                                     | {nanosecond: 2}                | '12:31:14.645876002'                       |
      | 'microsecond' | time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})                                     | {}                             | '12:31:14.645876'                          |

  
  Scenario Outline: [5] Should truncate time
    Given any graph
    When executing query:
      """
      RETURN time.truncate(<unit>, <other>, <map>) AS result
      """
    Then the result should be, in any order:
      | result   |
      | <result> |
    And no side effects

    Examples:
      | unit          | other                                                                                                                   | map                            | result                                     |
      | 'day'         | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {nanosecond: 2}                | '00:00:00.000000002+01:00'                 |
      | 'day'         | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '00:00+01:00'                              |
      | 'day'         | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {nanosecond: 2}                | '00:00:00.000000002Z'                      |
      | 'day'         | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '00:00Z'                                   |
      | 'hour'        | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}) | {nanosecond: 2}                | '12:00:00.000000002-01:00'                 |
      | 'hour'        | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}) | {timezone: '+01:00'}           | '12:00+01:00'                              |
      | 'hour'        | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}) | {}                             | '12:00-01:00'                              |
      | 'hour'        | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {nanosecond: 2}                | '12:00:00.000000002Z'                      |
      | 'hour'        | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {timezone: '+01:00'}           | '12:00+01:00'                              |
      | 'hour'        | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '12:00Z'                                   |
      | 'hour'        | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                                    | {nanosecond: 2}                | '12:00:00.000000002Z'                      |
      | 'hour'        | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                                    | {timezone: '+01:00'}           | '12:00+01:00'                              |
      | 'hour'        | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                                    | {}                             | '12:00Z'                                   |
      | 'hour'        | time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'})                                     | {nanosecond: 2}                | '12:00:00.000000002-01:00'                 |
      | 'hour'        | time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'})                                     | {timezone: '+01:00'}           | '12:00+01:00'                              |
      | 'hour'        | time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'})                                     | {}                             | '12:00-01:00'                              |
      | 'minute'      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}) | {nanosecond: 2}                | '12:31:00.000000002-01:00'                 |
      | 'minute'      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}) | {}                             | '12:31-01:00'                              |
      | 'minute'      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {nanosecond: 2}                | '12:31:00.000000002Z'                      |
      | 'minute'      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '12:31Z'                                   |
      | 'minute'      | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                                    | {nanosecond: 2}                | '12:31:00.000000002Z'                      |
      | 'minute'      | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                                    | {}                             | '12:31Z'                                   |
      | 'minute'      | time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'})                                     | {nanosecond: 2}                | '12:31:00.000000002-01:00'                 |
      | 'minute'      | time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'})                                     | {}                             | '12:31-01:00'                              |
      | 'second'      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {nanosecond: 2}                | '12:31:14.000000002+01:00'                 |
      | 'second'      | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '12:31:14+01:00'                           |
      | 'second'      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {nanosecond: 2}                | '12:31:14.000000002Z'                      |
      | 'second'      | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '12:31:14Z'                                |
      | 'second'      | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                                    | {nanosecond: 2}                | '12:31:14.000000002Z'                      |
      | 'second'      | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                                    | {}                             | '12:31:14Z'                                |
      | 'second'      | time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})                                     | {nanosecond: 2}                | '12:31:14.000000002+01:00'                 |
      | 'second'      | time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})                                     | {}                             | '12:31:14+01:00'                           |
      | 'millisecond' | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {nanosecond: 2}                | '12:31:14.645000002+01:00'                 |
      | 'millisecond' | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '12:31:14.645+01:00'                       |
      | 'millisecond' | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {nanosecond: 2}                | '12:31:14.645000002Z'                      |
      | 'millisecond' | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '12:31:14.645Z'                            |
      | 'millisecond' | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                                    | {nanosecond: 2}                | '12:31:14.645000002Z'                      |
      | 'millisecond' | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                                    | {}                             | '12:31:14.645Z'                            |
      | 'millisecond' | time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})                                     | {nanosecond: 2}                | '12:31:14.645000002+01:00'                 |
      | 'millisecond' | time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})                                     | {}                             | '12:31:14.645+01:00'                       |
      | 'microsecond' | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {nanosecond: 2}                | '12:31:14.645876002+01:00'                 |
      | 'microsecond' | datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) | {}                             | '12:31:14.645876+01:00'                    |
      | 'microsecond' | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {nanosecond: 2}                | '12:31:14.645876002Z'                      |
      | 'microsecond' | localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})                | {}                             | '12:31:14.645876Z'                         |
      | 'microsecond' | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                                    | {nanosecond: 2}                | '12:31:14.645876002Z'                      |
      | 'microsecond' | localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})                                                    | {}                             | '12:31:14.645876Z'                         |
      | 'microsecond' | time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})                                     | {nanosecond: 2}                | '12:31:14.645876002+01:00'                 |
      | 'microsecond' | time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})                                     | {}                             | '12:31:14.645876+01:00'                    |
